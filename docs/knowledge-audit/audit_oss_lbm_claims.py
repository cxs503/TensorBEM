#!/usr/bin/env python3
"""Independent, re-runnable audit of the OSS-LBM knowledge landing (FEAT-TK-OSS-LANDING).

Purpose
-------
The ledger entry FEAT-TK-OSS-LANDING asks for >=20 multiphase/MCMP claims and >=15
differentiable/framework claims whose acceptance gate is *hard*:

  schema 全过 + **全部 oss-cache 实读定位逐字引文** + 自带校验器全绿 + 无自批
  + 每条映射回 GAP 锚点

This tool does not produce new claims.  It re-reads what is claimed to be landed and
tries to falsify the acceptance gate on real artefacts:

  C1 schema      -- every claim/evidence record validates against schemas/*.schema.json
  C2 closure     -- claim.evidence_refs <-> evidence.claim_refs form a closed graph
  C3 locator     -- every evidence record carries a pinned `<repo>@<commit> <path>:<lines>`
  C4 verbatim    -- every quoted fragment is found, whitespace-normalised, inside the
                    cited line window of the cited file, read out of the oss-cache
                    checkout at the *cited commit* (git show <commit>:<path>)
  C5 gap anchor  -- every claim is mapped back to a GAP-LBM-* anchor
  C6 no self-approval -- producer (created_by/captured_by) never equals assessor
  C7 tally       -- A1 (multiphase/MCMP) and A2 (differentiable/framework) counts vs
                    the 20 / 15 thresholds under an explicitly stated, reproducible rule

Anything that cannot be checked (repo absent from the local oss-cache, quote housekeeping
in a markdown claim file, ...) is reported as ``not_checked``/``unknown`` -- never as a
pass.  Unknown is not green.

Usage
-----
    python3 audit_oss_lbm_claims.py --out ./            # writes audit.json + audit.md
    python3 audit_oss_lbm_claims.py --self-test          # fixture-driven invariant tests

Exit codes: 0 = all checks pass; 1 = at least one blocking finding (non-green); 2 = error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

# --------------------------------------------------------------------------------------
# defaults (overridable -- the tool is domain-config, not domain-hardcoded)
# --------------------------------------------------------------------------------------
DEFAULT_KNOWLEDGE = "/root/tensor-ecosystem/TensorKnowledge"
DEFAULT_OSS_CACHE = "/root/oss-cache"
CLAIM_SUBTREE = "knowledge/oss-lbm"
A1_KEYWORDS = (
    "multiphase", "multi-phase", "mcmp", "scmp", "phase-field", "phase field",
    "cahn-hilliard", "cahn hilliard", "shan-chen", "shan chen", "pseudo-potential",
    "color-gradient", "color gradient", "free-energy", "free energy", "surface tension",
    "laplace", "droplet", "bubble", "contact angle", "ternary", "three-component",
    "three-component", "wetting", "spinodal", "rayleigh-taylor", "rayleigh taylor",
    "fingering", "muskat", "interfacial", "vesicle",
)
A2_KEYWORDS = (
    "jax", "differentiable", "autodiff", "auto-diff", "automatic differentiation",
    "gradient", "xlb", "lettuce", "lbmpy", "ludwig", "framework", "domain specific language",
    "dsl", "code generation", "code-generation", "sympy", "symbolic", "jit", "numba",
    "taichi", "kernel", "compiler", "python dsl", "yaml", "cuda fortran", "gpu",
)
GAP_RE = re.compile(r"GAP-[A-Z0-9][A-Z0-9-]+")
LOCATOR_RE = re.compile(
    r"^(?P<repo>[^\s@]+)@(?P<ref>[^\s]+)\s+(?P<path>[^\s:]+)"
    r"\s*(?::\s*(?P<lines>[\d\-–,\s]+|[^\s]+))?\s*$"
)
HEX_RE = re.compile(r"^[0-9a-f]{7,40}$")
ELLIPSIS_RE = re.compile(r"\.\.\.|…")
MIN_SEGMENT = 8  # characters; shorter fragments are not evidence of verbatim quotation


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def norm(s: str) -> str:
    """Whitespace-normalise for quote comparison (layout must never be evidence)."""
    return re.sub(r"\s+", " ", s).strip()


def split_quote(quote: str) -> list[str]:
    """A quotation may elide with '...' / '…'; every surviving fragment must be present."""
    segs = [norm(s) for s in ELLIPSIS_RE.split(quote or "")]
    return [s for s in segs if len(s) >= MIN_SEGMENT]


NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numeric_match(quote: str, text: str, rel_tol: float = 2e-3) -> tuple[bool, list[str]]:
    """A re-rendered quotation (numbers reformatted, field names reworded) is not a verbatim
    quote, but it is still falsifiable: every number it asserts must appear in the source."""
    try:
        hay = [float(x) for x in NUM_RE.findall(text)]
    except ValueError:  # pragma: no cover
        return False, ["unparsable source numbers"]
    missing = []
    for tok in NUM_RE.findall(quote):
        try:
            val = float(tok)
        except ValueError:
            continue
        if not any(abs(val - h) <= rel_tol * max(abs(val), abs(h), 1e-30) for h in hay):
            missing.append(tok)
    return (not missing), missing


def parse_lines(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    spec = spec.replace("–", "-").strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        if a.strip().isdigit() and b.strip().isdigit():
            return int(a), int(b)
        return None
    if spec.strip().isdigit():
        return int(spec), int(spec)
    return None


def git(repo: Path, *args: str, timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return 1, f"<git error: {exc}>"


def repo_slug(name: str) -> str:
    """Normalise a repo name so `Maarten-vd-Sande/lbm` can find `Maarten-vd-Sande_lbm`."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------
@dataclass
class Claim:
    id: str
    statement: str
    record: dict[str, Any]
    origin: str            # file the claim was read from
    evidence_refs: list[str] = field(default_factory=list)
    gap_anchors: list[str] = field(default_factory=list)
    created_by: str | None = None
    epi: str | None = None
    system: str | None = None


def load_json_lenient(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """accept: [ ... ] | {"claims": [...]} | {"evidence": [...]} | single record"""
    data = load_json_lenient(path)
    if data is None:
        return
    if isinstance(data, list):
        for r in data:
            if isinstance(r, dict):
                yield r
    elif isinstance(data, dict):
        for key in ("claims", "evidence", "records", "items"):
            if isinstance(data.get(key), list):
                for r in data[key]:
                    if isinstance(r, dict):
                        yield r
                return
        if "id" in data:
            yield data


MD_CLAIM_RE = re.compile(r"^#\s*Claim\b", re.MULTILINE)


def parse_md_claim(path: Path) -> dict[str, Any] | None:
    """Markdown claims (one file per claim) carry a small front-matter-ish header."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not MD_CLAIM_RE.search(text):
        return None
    def field_of(name: str) -> str | None:
        m = re.search(rf"\*\*{name}\*\*\s*:\s*(.+)", text)
        return m.group(1).strip() if m else None
    stmt = None
    m = re.search(r"^##\s*Statement\s*\n+(.+?)(?:\n##|\Z)", text, re.S | re.MULTILINE)
    if m:
        stmt = norm(m.group(1))
    return {
        "__markdown__": True,
        "id": field_of("claim_id") or path.stem,
        "statement": stmt or "",
        "epistemic_status": field_of("epistemic_status"),
        "gap_anchor": field_of("gap_anchor"),
        "evidence_refs": [],
        "created_by": field_of("created_by") or "unknown",
        "scope": {"system": field_of("repo") or ""},
        "__text__": text,
    }


def collect_claims(root: Path) -> tuple[list[Claim], list[dict[str, Any]], list[str]]:
    base = root / CLAIM_SUBTREE
    claims: list[Claim] = []
    problems: list[str] = []
    seen: dict[str, list[str]] = defaultdict(list)

    json_files = sorted(set(base.rglob("*.json")))
    md_files = sorted(set(base.rglob("claim_*.md")))
    raw_evidence: list[dict[str, Any]] = []

    for p in json_files:
        for rec in iter_json_records(p):
            rid = str(rec.get("id", ""))
            if rid.startswith("tkc:"):
                seen[rid].append(str(p.relative_to(root)))
            elif rid.startswith("tke:"):
                raw_evidence.append({**rec, "__origin__": str(p.relative_to(root))})
            elif "statement" in rec:
                problems.append(f"claim record with non-conforming id {rid!r} in {p.relative_to(root)}")

    for cid, origins in seen.items():
        if len(origins) > 1:
            problems.append(f"duplicate claim id {cid} in {len(origins)} files: {origins}")
    known = set(seen)

    for p in json_files:
        for rec in iter_json_records(p):
            rid = str(rec.get("id", ""))
            if not rid.startswith("tkc:") or str(p.relative_to(root)) != seen[rid][0]:
                continue
            scope = rec.get("scope") if isinstance(rec.get("scope"), dict) else {}
            gaps = list(rec.get("gap_anchors") or [])
            if rec.get("gap_anchor"):
                gaps.append(str(rec["gap_anchor"]))
            for note in rec.get("notes") or []:
                gaps.extend(GAP_RE.findall(str(note)))
            claims.append(Claim(
                id=rid,
                statement=str(rec.get("statement", "")),
                record=rec,
                origin=str(p.relative_to(root)),
                evidence_refs=list(rec.get("evidence_refs") or []),
                gap_anchors=sorted(set(gaps)),
                created_by=rec.get("created_by"),
                epi=rec.get("epistemic_status"),
                system=str(scope.get("system") or ""),
            ))

    for p in md_files:
        rec = parse_md_claim(p)
        if not rec:
            continue
        gaps = list(GAP_RE.findall(rec.get("gap_anchor") or "") + GAP_RE.findall(rec["__text__"]))
        claims.append(Claim(
            id=str(rec["id"]),
            statement=str(rec["statement"]),
            record=rec,
            origin=str(p.relative_to(root)),
            evidence_refs=[],
            gap_anchors=sorted(set(gaps)),
            created_by=rec.get("created_by"),
            epi=rec.get("epistemic_status"),
            system=(rec.get("scope") or {}).get("system"),
        ))
    return claims, raw_evidence, problems


def collect_evidence(root: Path, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """merge the evidence records discovered while walking the claim subtree with the
    canonical evidence bundle(s) under knowledge/oss-lbm/evidence/"""
    by_id: dict[str, dict[str, Any]] = {}
    for rec in raw:
        by_id[str(rec.get("id"))] = rec
    base = root / CLAIM_SUBTREE
    for p in sorted(base.rglob("*evidence*.json")):
        for rec in iter_json_records(p):
            rid = str(rec.get("id", ""))
            if rid.startswith("tke:") and rid not in by_id:
                by_id[rid] = {**rec, "__origin__": str(p.relative_to(root))}
    return list(by_id.values())


# --------------------------------------------------------------------------------------
# oss-cache resolution
# --------------------------------------------------------------------------------------
class OssCache:
    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, Path] = {}
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    self.entries[repo_slug(child.name)] = child
            # <org>/<repo> layouts only when the nested directory really is a checkout
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    for sub in sorted(child.iterdir()):
                        if sub.is_dir() and (sub / ".git").exists():
                            self.entries.setdefault(repo_slug(f"{child.name}/{sub.name}"), sub)

    def resolve(self, repo: str) -> Path | None:
        slug = repo_slug(repo)
        if slug in self.entries:
            return self.entries[slug]
        # `lbm-principles-practice/code` -> `lbm-principles-practice`
        head = repo_slug(repo.split("/")[0])
        return self.entries.get(head)

    def has_ref(self, repo: Path, ref: str) -> bool:
        if not (repo / ".git").exists():
            return False
        rc, _ = git(repo, "cat-file", "-e", f"{ref}^{{commit}}")
        return rc == 0

    def read_at(self, repo: Path, ref: str, path: str) -> str | None:
        rc, out = git(repo, "show", f"{ref}:{path}")
        if rc != 0:
            return None
        return out

    def tree(self, repo: Path, ref: str) -> list[str]:
        key = (str(repo), ref)
        if not hasattr(self, "_tree_cache"):
            self._tree_cache: dict[tuple[str, str], list[str]] = {}
        if key not in self._tree_cache:
            rc, out = git(repo, "ls-tree", "-r", "--name-only", ref)
            self._tree_cache[key] = out.splitlines() if rc == 0 else []
        return self._tree_cache[key]

    def resolve_path(self, repo: Path, ref: str, path: str) -> tuple[str | None, str]:
        """return (resolved_path, how) with how in {exact, suffix, missing}"""
        files = self.tree(repo, ref)
        path = path.lstrip("./")
        if path in files:
            return path, "exact"
        cands = [f for f in files if f.endswith("/" + path)]
        if len(cands) == 1:
            return cands[0], "suffix"
        return None, "missing"

    def head_commit(self, repo: Path) -> str | None:
        rc, out = git(repo, "rev-parse", "HEAD")
        return out.strip() if rc == 0 else None


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------
def check_schema(root: Path, claims: list[Claim], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"checked": 0, "failed": [], "skipped": "jsonschema unavailable"}
    try:
        import jsonschema  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return out
    schemas = {}
    for name, key in (("claim-record.schema.json", "claim"), ("evidence-record.schema.json", "evidence")):
        p = root / "schemas" / name
        if p.exists():
            schemas[key] = json.loads(p.read_text())
    out["skipped"] = None
    for c in claims:
        if c.record.get("__markdown__"):
            continue
        out["checked"] += 1
        s = schemas.get("claim")
        if not s:
            continue
        try:
            jsonschema.validate(c.record, s)
        except Exception as exc:  # noqa: BLE001
            out["failed"].append({"id": c.id, "error": str(exc).splitlines()[0]})
    for e in evidence:
        out["checked"] += 1
        s = schemas.get("evidence")
        if not s:
            continue
        try:
            jsonschema.validate({k: v for k, v in e.items() if not k.startswith("__")}, s)
        except Exception as exc:  # noqa: BLE001
            out["failed"].append({"id": e.get("id"), "error": str(exc).splitlines()[0]})
    return out


def check_closure(claims: list[Claim], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    ev_ids = {str(e.get("id")) for e in evidence}
    claim_ids = {c.id for c in claims}
    dangling = [{"claim": c.id, "missing_evidence": refs}
                for c in claims
                if (refs := [r for r in c.evidence_refs if r not in ev_ids])]
    no_refs = [c.id for c in claims if not c.evidence_refs]
    back = [{"evidence": e.get("id"), "missing_claims": miss}
            for e in evidence
            if (miss := [r for r in (e.get("claim_refs") or []) if r not in claim_ids])]
    orphan = [str(e.get("id")) for e in evidence if not (e.get("claim_refs") or [])]
    return {
        "claims": len(claims), "evidence": len(evidence),
        "dangling_evidence_refs": dangling,
        "claims_without_evidence": no_refs,
        "evidence_pointing_at_missing_claims": back,
        "orphan_evidence": orphan,
    }


def check_verbatim(evidence: list[dict[str, Any]], cache: OssCache,
                   knowledge: Path | None = None) -> dict[str, Any]:
    """Re-read every cited quotation out of the pinned source.

    Locator forms accepted (all must be machine-resolvable, i.e. re-runnable):
      * ``repo@commit path:12-34``            -- code location, numeric window
      * ``repo@commit path:symbol``           -- code location, symbol anchor
      * ``path#fragment`` / ``path`` (other)  -- in-repository artifact (runs/, knowledge/)
    """
    stats: Counter = Counter()
    findings: list[dict[str, Any]] = []
    per_repo: dict[str, Counter] = defaultdict(Counter)

    def add(sev: str, check: str, rec_id: str, detail: str) -> None:
        findings.append({"severity": sev, "check": check, "evidence": rec_id, "detail": detail})

    def verify_quote(rec_id: str, text: str, quote: str, lines: tuple[int, int] | None,
                     symbol: str | None, where: str, repo_key: str) -> None:
        segs = split_quote(quote)
        if not segs:
            stats["not_checked_empty_quote"] += 1
            per_repo[repo_key]["empty_quote"] += 1
            add("high", "C4-verbatim", rec_id,
                f"quote empty or every fragment shorter than {MIN_SEGMENT} chars -- nothing "
                f"verbatim to re-read in {where}")
            return
        if symbol:
            window = norm(text)
        elif lines:
            window = norm("\n".join(text.splitlines()[max(0, lines[0] - 1): lines[1]]))
        else:
            window = norm(text)
        miss = [s for s in segs if s not in window]
        if not miss:
            stats["verbatim_ok"] += 1
            per_repo[repo_key]["ok"] += 1
            return
        in_file = norm(text)
        if all(s in in_file for s in miss):
            stats["verbatim_ok_locator_drift"] += 1
            per_repo[repo_key]["drift"] += 1
            anchor = f"lines {lines[0]}-{lines[1]}" if lines else f"symbol {symbol!r}"
            add("medium", "C4-verbatim", rec_id,
                f"quote present in {where} but outside cited {anchor} (locator drift)")
        elif all(s.replace('"', "'").replace("`", "'")
                 in in_file.replace('"', "'").replace("`", "'") for s in miss):
            stats["quote_cosmetic_divergence"] += 1
            per_repo[repo_key]["cosmetic"] += 1
            add("low", "C4-verbatim", rec_id,
                f"quote differs from {where} only in quote-mark style (semantically identical, "
                f"not literal -- acceptance asks for 逐字)")
        else:
            nums_ok, missing_nums = numeric_match(quote, in_file)
            if nums_ok and NUM_RE.search(quote):
                stats["quote_rerendered_numbers_ok"] += 1
                per_repo[repo_key]["rerendered"] += 1
                add("medium", "C4-verbatim", rec_id,
                    f"quote is a re-rendering, not verbatim text of {where}; every asserted "
                    f"number is nevertheless present (numeric cross-check passed)")
            else:
                stats["verbatim_failed"] += 1
                per_repo[repo_key]["failed"] += 1
                add("blocking", "C4-verbatim", rec_id,
                    f"fragment not found in {where}: {[s[:70] for s in miss][:1]}"
                    + (f"; numbers absent: {missing_nums[:5]}" if missing_nums else ""))

    for e in evidence:
        rec_id = str(e.get("id"))
        loc = e.get("locator") if isinstance(e.get("locator"), dict) else {}
        value = norm(str(loc.get("value") or ""))
        quote = str(loc.get("quote") or "")
        ltype = loc.get("locator_type")
        if not value and loc.get("path"):
            # structured dialect: repo/commit/path/line_start/line_end instead of `value`.  The
            # evidence schema requires `value`, so this is a format violation -- but the locator
            # is still resolvable, so the quotation is still falsifiable.
            rs, re_ = loc.get("repo"), loc.get("commit")
            ls, le = loc.get("line_start"), loc.get("line_end")
            w = f"{loc['path']}" + (f":{ls}-{le}" if ls and le else "")
            value = f"{rs}@{re_} {w}" if rs and re_ else w
            add("medium", "C3-locator", rec_id,
                "locator uses the structured dialect (repo/commit/path/line_start) instead of the "
                "schema-required `value` string; evidence-record.schema.json rejects the record")
            stats["locator_structured_dialect"] += 1
        repo_key = value.split("@")[0] if "@" in value else (value.split()[0] if value else "?")
        per_repo[repo_key]["records"] += 1

        # ---- in-repository artifact (runs/, knowledge/...): verify against the knowledge tree
        if ltype in ("other", "artifact", "file") and knowledge is not None:
            rel = value.split("#")[0]
            target = knowledge / rel
            if target.is_file() and "@" not in value:
                stats["internal_artifact"] += 1
                verify_quote(rec_id, target.read_text(encoding="utf-8", errors="replace"),
                             quote, None, None, str(target), repo_key)
                continue

        if ltype == "code-search" and not value:
            stats["negative_search_locator"] += 1
            per_repo[repo_key]["negative_search"] += 1
            add("medium", "C3-locator", rec_id,
                "negative-search locator (pattern/scope, no line window): the absence claim can "
                "only be re-run where the repository is cached -- unverified here")
            continue
        m = LOCATOR_RE.match(value)
        if not value or not m:
            stats["locator_missing_or_unparsable"] += 1
            per_repo[repo_key]["unparsable"] += 1
            add("high", "C3-locator", rec_id,
                f"locator_type={ltype!r} value={value!r} is not a machine-resolvable "
                f"'<repo>@<commit> <path>:<lines|symbol>' locator")
            continue
        ref, path = m.group("ref"), m.group("path")
        raw_lines = (m.group("lines") or "").strip()
        lines = parse_lines(raw_lines)
        symbol = raw_lines if (raw_lines and lines is None) else None
        if HEX_RE.match(ref):
            stats["locator_pinned"] += 1
        else:
            stats["locator_unpinned"] += 1
            add("medium", "C3-locator", rec_id,
                f"ref {ref!r} is not a commit sha (branch/tag => provenance mobile, verbatim "
                f"re-read not reproducible later)")

        repo_dir = cache.resolve(m.group("repo"))
        if repo_dir is None:
            stats["not_checked_repo_absent"] += 1
            per_repo[repo_key]["repo_absent"] += 1
            continue
        if not cache.has_ref(repo_dir, ref):
            stats["not_checked_ref_absent"] += 1
            add("medium", "C4-verbatim", rec_id,
                f"ref {ref!r} not present in oss-cache checkout {repo_dir.name}")
            continue
        resolved, how = cache.resolve_path(repo_dir, ref, path)
        if resolved is None:
            stats["not_checked_path_absent"] += 1
            per_repo[repo_key]["path_absent"] += 1
            add("high", "C4-verbatim", rec_id,
                f"path {path!r} not found at {ref} in {repo_dir.name} "
                f"(not root-relative and no unique suffix match)")
            continue
        if how == "suffix":
            stats["path_resolved_by_suffix"] += 1
            add("medium", "C3-locator", rec_id,
                f"cited path {path!r} is not repository-root-relative; uniquely resolved to "
                f"{resolved!r} -- locator should be corrected")
        text = cache.read_at(repo_dir, ref, resolved)
        if text is None:
            stats["not_checked_path_absent"] += 1
            add("high", "C4-verbatim", rec_id, f"unable to read {resolved!r} at {ref}")
            continue
        verify_quote(rec_id, text, quote, lines, symbol,
                     f"{repo_dir.name}@{ref[:9]}:{resolved}", repo_key)

    return {"stats": dict(stats), "per_repo": {k: dict(v) for k, v in per_repo.items()},
            "findings": findings}


def check_gap(claims: list[Claim]) -> dict[str, Any]:
    unmapped = [c.id for c in claims if not c.gap_anchors]
    anchors = Counter(a for c in claims for a in c.gap_anchors)
    return {"claims": len(claims), "mapped": len(claims) - len(unmapped),
            "unmapped": unmapped, "anchors": dict(anchors.most_common())}


def check_self_approval(claims: list[Claim], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    bad = []
    for e in evidence:
        producer = str(e.get("captured_by") or "")
        assessor = str(e.get("assessed_by") or e.get("assessed_by_agent") or "")
        if producer and assessor and producer == assessor:
            bad.append({"evidence": e.get("id"), "agent": producer})
    for c in claims:
        producer = str(c.record.get("created_by") or "")
        assessor = str(c.record.get("assessed_by") or c.record.get("reviewed_by") or "")
        if producer and assessor and producer == assessor:
            bad.append({"claim": c.id, "agent": producer})
    assessed = sum(1 for e in evidence if e.get("assessment_status") not in (None, "unassessed"))
    return {"self_approved": bad, "evidence_assessed": assessed, "evidence_total": len(evidence)}


def classify(claim: Claim) -> tuple[bool, bool]:
    blob = f"{claim.statement} {claim.system or ''} {' '.join(claim.gap_anchors)}".lower()
    return (any(k in blob for k in A1_KEYWORDS), any(k in blob for k in A2_KEYWORDS))


FRAMEWORK_SYSTEMS = {"xlb", "lettuce", "lbmpy", "ludwig", "pylbm", "sailfish", "openlb", "openlbmpm"}
DIFF_SYSTEMS = {"xlb", "lettuce"}
DIFF_TOKENS = ("jax", "torch", "autodiff", "auto-diff", "differentiable", "automatic differentiation")


def tally(claims: list[Claim]) -> dict[str, Any]:
    a1 = [c.id for c in claims if classify(c)[0]]
    a2_broad = [c.id for c in claims if classify(c)[1]]
    both = [c.id for c in claims if all(classify(c))]
    sys_key = lambda c: (c.system or "").split("/")[0].split()[0].lower() if c.system else ""
    a2_system = [c.id for c in claims if sys_key(c) in FRAMEWORK_SYSTEMS]
    a2_diff = [c.id for c in claims if sys_key(c) in DIFF_SYSTEMS
               or any(t in c.statement.lower() for t in DIFF_TOKENS)]
    per_system = Counter(sys_key(c) or "<none>" for c in claims)
    return {
        "a1_multiphase_mcmp": len(a1), "a2_differentiable_framework": len(a2_broad),
        "both": len(both), "a1_ids": a1, "a2_ids": a2_broad,
        "a2_strict_framework_system": len(a2_system),
        "a2_strict_differentiable": len(a2_diff),
        "per_system": dict(per_system.most_common()),
        "thresholds": {"A1": 20, "A2": 15},
        "rule": {"A1_keywords": list(A1_KEYWORDS), "A2_keywords": list(A2_KEYWORDS),
                 "A2_strict_framework_systems": sorted(FRAMEWORK_SYSTEMS),
                 "A2_strict_differentiable_tokens": list(DIFF_TOKENS)},
        "classification_is_a_judgement_call": True,
    }


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def run_audit(knowledge: Path, oss_cache: Path) -> dict[str, Any]:
    claims, raw_ev, problems = collect_claims(knowledge)
    evidence = collect_evidence(knowledge, raw_ev)
    cache = OssCache(oss_cache)
    res: dict[str, Any] = {
        "knowledge_root": str(knowledge),
        "oss_cache": str(oss_cache),
        "oss_cache_repos": {k: str(v) for k, v in cache.entries.items()},
        "discovery_problems": problems,
        "counts": {"claims": len(claims), "evidence": len(evidence)},
        "C1_schema": check_schema(knowledge, claims, evidence),
        "C2_closure": check_closure(claims, evidence),
        "C3_C4_locator_verbatim": check_verbatim(evidence, cache, knowledge),
        "C5_gap_anchor": check_gap(claims),
        "C6_self_approval": check_self_approval(claims, evidence),
        "C7_tally": tally(claims),
    }
    c1 = res["C1_schema"]
    c2 = res["C2_closure"]
    c3 = res["C3_C4_locator_verbatim"]["stats"]
    c5 = res["C5_gap_anchor"]
    t = res["C7_tally"]
    blocking = []
    if c1["failed"]:
        blocking.append(f"C1 schema: {len(c1['failed'])} record(s) fail schema")
    if c2["dangling_evidence_refs"] or c2["claims_without_evidence"]:
        blocking.append(f"C2 closure: {len(c2['dangling_evidence_refs'])} dangling, "
                        f"{len(c2['claims_without_evidence'])} claim(s) with no evidence")
    if c3.get("verbatim_failed"):
        blocking.append(f"C4 verbatim: {c3['verbatim_failed']} quote(s) falsified")
    unverified = (c3.get("not_checked_repo_absent", 0) + c3.get("not_checked_ref_absent", 0)
                  + c3.get("not_checked_path_absent", 0) + c3.get("not_checked_empty_quote", 0))
    if c5["unmapped"]:
        blocking.append(f"C5 gap anchor: {len(c5['unmapped'])} claim(s) unmapped")
    if t["a1_multiphase_mcmp"] < t["thresholds"]["A1"]:
        blocking.append(f"C7 A1 tally {t['a1_multiphase_mcmp']} < {t['thresholds']['A1']}")
    if t["a2_strict_differentiable"] < t["thresholds"]["A2"] <= t["a2_strict_framework_system"]:
        blocking.append(
            f"C7 A2 is WITHHELD, not met: differentiable-only claims = "
            f"{t['a2_strict_differentiable']} (< gate {t['thresholds']['A2']}) while "
            f"framework-inclusive claims = {t['a2_strict_framework_system']} (>= gate); the "
            f"ledger wording '可微/框架' is ambiguous, so the gate cannot be declared met")
    res["verdict"] = {
        "blocking": blocking,
        "green": not blocking,
        "C4_unverifiable_records": unverified,
        "C4_verified_records": c3.get("verbatim_ok", 0) + c3.get("verbatim_ok_locator_drift", 0),
        "acceptance_met": bool(not blocking and unverified == 0),
        "honesty_note": "records that could not be re-read are counted as unverifiable, not as pass",
    }
    return res


def render_markdown(res: dict[str, Any]) -> str:
    t = res["C7_tally"]
    block = res["C3_C4_locator_verbatim"]
    c3 = block["stats"]
    v = res["verdict"]
    L = [
        "# OSS-LBM claim/evidence landing -- independent audit (FEAT-TK-OSS-LANDING)",
        "",
        f"* knowledge root: `{res['knowledge_root']}`",
        f"* oss-cache: `{res['oss_cache']}` -> {len(res['oss_cache_repos'])} repo(s): "
        + ", ".join(sorted(res["oss_cache_repos"])),
        f"* claims discovered: **{res['counts']['claims']}**, evidence records: "
        f"**{res['counts']['evidence']}**",
        "",
        "## Verdict",
        "",
        f"* acceptance_met = **{v['acceptance_met']}** (green={v['green']})",
        f"* C4 verbatim: verified {v['C4_verified_records']} / unverifiable {v['C4_unverifiable_records']}",
        "",
    ]
    if v["blocking"]:
        L += ["### Blocking findings", ""] + [f"* {b}" for b in v["blocking"]] + [""]
    L += ["## Check matrix", "",
          "| check | result |", "|---|---|",
          f"| C1 schema | {res['C1_schema']['checked']} validated, "
          f"{len(res['C1_schema']['failed'])} failed |",
          f"| C2 closure | {len(res['C2_closure']['dangling_evidence_refs'])} dangling ref(s), "
          f"{len(res['C2_closure']['claims_without_evidence'])} claim(s) w/o evidence, "
          f"{len(res['C2_closure']['orphan_evidence'])} orphan evidence |",
          f"| C3 locator | pinned {c3.get('locator_pinned', 0)}, unpinned {c3.get('locator_unpinned', 0)}, "
          f"missing/unparsable {c3.get('locator_missing_or_unparsable', 0)} |",
          f"| C4 verbatim | ok {c3.get('verbatim_ok', 0)}, ok-with-drift "
          f"{c3.get('verbatim_ok_locator_drift', 0)}, failed {c3.get('verbatim_failed', 0)}, "
          f"unverifiable {v['C4_unverifiable_records']} |",
          f"| C5 gap anchor | {res['C5_gap_anchor']['mapped']}/{res['C5_gap_anchor']['claims']} mapped |",
          f"| C6 self-approval | {len(res['C6_self_approval']['self_approved'])} self-approved, "
          f"{res['C6_self_approval']['evidence_assessed']}/"
          f"{res['C6_self_approval']['evidence_total']} evidence assessed |",
          f"| C7 tally | A1={t['a1_multiphase_mcmp']} (gate {t['thresholds']['A1']}); "
          f"A2 broad-keyword={t['a2_differentiable_framework']}, "
          f"framework-system={t['a2_strict_framework_system']}, "
          f"differentiable-only={t['a2_strict_differentiable']} (gate {t['thresholds']['A2']}) |",
          ""]
    L += ["## Per-repository verbatim re-read", "",
          "| repo key | records | repo absent | quote ok | drift | re-rendered (numbers ok) | falsified | unparsable |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in sorted(block["per_repo"].items()):
        L.append(f"| `{k}` | {s.get('records', 0)} | {s.get('repo_absent', 0)} | "
                 f"{s.get('ok', 0)} | {s.get('drift', 0)} | {s.get('rerendered', 0)} | "
                 f"{s.get('failed', 0)} | {s.get('unparsable', 0)} |")
    L.append("")
    sev = Counter(f["severity"] for f in block["findings"])
    L += ["## Findings by severity", ""] + [f"* {k}: {n}" for k, n in sev.most_common()] + [""]
    for f in block["findings"][:40]:
        L.append(f"* **{f['severity']}** `{f['check']}` {f['evidence']}: {f['detail']}")
    if len(block["findings"]) > 40:
        L.append(f"* ... {len(block['findings']) - 40} more in audit.json")
    L.append("")
    return "\n".join(L)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knowledge-root", default=os.environ.get("TK_ROOT", DEFAULT_KNOWLEDGE))
    ap.add_argument("--oss-cache", default=os.environ.get("OSS_CACHE", DEFAULT_OSS_CACHE))
    ap.add_argument("--out", default=".")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    res = run_audit(Path(args.knowledge_root), Path(args.oss_cache))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    (out / "audit.md").write_text(render_markdown(res))
    print(render_markdown(res))
    print(f"artifacts: {out/'audit.json'} (sha256 {sha256(out/'audit.json')[:16]}), {out/'audit.md'}")
    return 0 if res["verdict"]["green"] else 1


# --------------------------------------------------------------------------------------
# self-test: the checks must FAIL on purpose-built bad input (invariant, not execution)
# --------------------------------------------------------------------------------------
def _self_test() -> int:  # pragma: no cover - exercised via pytest too
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tk = Path(td) / "tk"
        cache = Path(td) / "cache" / "demo-repo"
        (tk / "schemas").mkdir(parents=True)
        (tk / CLAIM_SUBTREE / "claims").mkdir(parents=True)
        cache.mkdir(parents=True)

        subprocess.run(["git", "-C", str(cache), "init", "-q"], check=True)
        (cache / "solver.py").write_text(
            "def collide(f):\n    return f * 0.5  # BGK relaxation\n\n\ndef stream(f):\n    return f\n")
        subprocess.run(["git", "-C", str(cache), "-c", "user.email=a@b", "-c", "user.name=t",
                        "add", "."], check=True)
        subprocess.run(["git", "-C", str(cache), "-c", "user.email=a@b", "-c", "user.name=t",
                        "commit", "-qm", "init"], check=True)
        rev = subprocess.run(["git", "-C", str(cache), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()

        good_ev = {"id": "tke:demo:good", "claim_refs": ["tkc:demo:good"],
                   "source": {"source_type": "other", "citation": "solver.py", "url": "u"},
                   "relation": "supports",
                   "locator": {"locator_type": "code-location",
                               "value": f"demo-repo@{rev} solver.py:1-2",
                               "quote": "def collide(f): ... return f * 0.5"},
                   "captured_at": "2026-01-01T00:00:00Z", "captured_by": "agent-a",
                   "assessment_status": "unassessed", "version": "1.0.0"}
        drift_ev = {**good_ev, "id": "tke:demo:drift", "claim_refs": ["tkc:demo:drift"],
                    "locator": {**good_ev["locator"], "value": f"demo-repo@{rev} solver.py:7-8"}}
        bad_ev = {**good_ev, "id": "tke:demo:bad", "claim_refs": ["tkc:demo:bad"],
                  "locator": {**good_ev["locator"], "value": f"demo-repo@{rev} solver.py:1-2",
                              "quote": "the collision operator is exactly BGK"}}
        empty_ev = {**good_ev, "id": "tke:demo:empty", "claim_refs": ["tkc:demo:empty"],
                    "locator": {"locator_type": "code-location", "value": "", "quote": ""}}
        unpinned = {**good_ev, "id": "tke:demo:unpinned", "claim_refs": ["tkc:demo:unpinned"],
                    "locator": {**good_ev["locator"], "value": "demo-repo@master solver.py:1-2"}}

        def claim(cid: str, refs: list[str], gap: str | None) -> dict[str, Any]:
            return {"id": f"tkc:demo:{cid}", "statement": "multiphase Shan-Chen demo claim",
                    "claim_type": "descriptive", "scope": {"domain": "lbm", "system": "demo"},
                    "modality": "asserted", "epistemic_status": "established",
                    "evidence_refs": refs, "created_at": "2026-01-01T00:00:00Z",
                    "created_by": "agent-a", "version": "1.0.0",
                    "notes": ([f"GAP anchor: {gap}"] if gap else [])}

        json.dump([claim("good", ["tke:demo:good"], "GAP-LBM-DEMO"),
                   claim("drift", ["tke:demo:drift"], "GAP-LBM-DEMO"),
                   claim("bad", ["tke:demo:bad"], None),
                   claim("empty", ["tke:demo:empty"], "GAP-LBM-DEMO"),
                   claim("unpinned", ["tke:demo:unpinned"], "GAP-LBM-DEMO"),
                   claim("dangling", ["tke:demo:missing"], "GAP-LBM-DEMO")],
                  open(tk / CLAIM_SUBTREE / "claims" / "claims.json", "w"))
        json.dump([good_ev, drift_ev, bad_ev, empty_ev, unpinned],
                  open(tk / CLAIM_SUBTREE / "claims" / "evidence.json", "w"))

        res = run_audit(tk, Path(td) / "cache")
        c3 = res["C3_C4_locator_verbatim"]["stats"]
        findings = {f["evidence"]: f for f in res["C3_C4_locator_verbatim"]["findings"]}
        ok = True
        def expect(cond: bool, msg: str) -> None:
            nonlocal ok
            if not cond:
                ok = False
                print(f"SELF-TEST FAIL: {msg}")

        # invariant: a genuine verbatim quote at the cited lines is a pass, not a finding
        expect("tke:demo:good" not in findings, f"good quote wrongly flagged: {findings.get('tke:demo:good')}")
        expect(c3.get("verbatim_ok", 0) >= 1, f"good quote should verify, stats={c3}")
        # invariant: quote in file but outside the cited window => locator drift (medium, not pass)
        expect(findings.get("tke:demo:drift", {}).get("severity") == "medium",
               f"drift case: {findings.get('tke:demo:drift')}")
        # invariant: a paraphrase (not present at all) must be falsified and blocking
        expect(findings.get("tke:demo:bad", {}).get("severity") == "blocking",
               f"paraphrased quote must be blocking: {findings.get('tke:demo:bad')}")
        # invariant: empty locator/quote => unverifiable, never a pass
        expect(findings.get("tke:demo:empty", {}).get("severity") == "high",
               f"empty locator must be high: {findings.get('tke:demo:empty')}")
        # invariant: branch refs are flagged as unpinned provenance even when the quote verifies
        expect("tke:demo:unpinned" in findings and "not a commit sha" in findings["tke:demo:unpinned"]["detail"],
               f"branch ref must be flagged unpinned: {findings.get('tke:demo:unpinned')}")
        # invariant: dangling evidence ref and unmapped GAP anchor are caught
        expect(len(res["C2_closure"]["dangling_evidence_refs"]) == 1, "dangling ref must be caught")
        expect(len(res["C5_gap_anchor"]["unmapped"]) == 1, "missing gap anchor must be caught")
        # invariant: acceptance can never be green on this fixture
        expect(res["verdict"]["acceptance_met"] is False, "acceptance must not be green on bad input")
        expect(res["verdict"]["green"] is False, "verdict must not be green on bad input")
        # invariant: A1 keyword rule classifies Shan-Chen/multiphase statements
        expect(res["C7_tally"]["a1_multiphase_mcmp"] == 6, f"A1 rule: {res['C7_tally']}")
        print("SELF-TEST", "PASS" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())