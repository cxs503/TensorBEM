#!/usr/bin/env python3
"""m17083 — TensorBEM upstream-delta / commit-theme inventory.

Deterministic, zero-hand-copy extraction of the last N commits of the working
tree and their first-parent diffs, classification into theme families, and
rule-based risk annotation.  Every number emitted here is read from ``git``
plumbing; nothing is transcribed by hand.

Outputs (all under docs/m17083_upstream_delta/):
  study.json                       machine-readable inventory
  INVENTORY.md                     human-readable theme list + risk table
  fig_m17083_theme_risk.png        figure (areas per commit + risk flags)
  sha256.txt                       provenance hashes

Usage:  python scripts/m17083_upstream_delta_inventory.py [--window 15]
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "m17083_upstream_delta")

# ---------------------------------------------------------------- git helpers


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def patch_key(commit: str) -> str:
    """Stable patch identity of the first-parent diff (git patch-id)."""
    diff = subprocess.run(
        ["git", "diff", f"{commit}^1", commit], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout
    pid = subprocess.run(
        ["git", "patch-id", "--stable"], cwd=ROOT, input=diff,
        capture_output=True, text=True, check=True).stdout.strip()
    return pid.split()[0] if pid else f"empty:{commit}"


def numstat(commit: str):
    """First-parent diff numstat (works for merge commits too)."""
    raw = git("diff", "--numstat", f"{commit}^1", commit)
    rows = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        add, dele, path = parts
        rows.append(
            {
                "path": path,
                "add": 0 if add == "-" else int(add),
                "del": 0 if dele == "-" else int(dele),
                "binary": add == "-",
            }
        )
    return rows


# ------------------------------------------------------------ classification

# Theme families are recognised from *paths* (primary evidence) and the commit
# subject (secondary).  Keyword lists are explicit so the mapping is auditable.
HYDRO_KERNEL = (
    "green", "kelvin", "neumann", "wavemaker", "flexural", "acoustic",
    "wigley", "propeller", "shape_gradient", "drift", "hydrostatic", "rao",
    "kochin", "ellipsoid", "ursell", "maccamy", "cummins", "added_mass",
    "diffraction", "radiation", "cavity", "sloshing", "bem",
)
EVIDENCE_INFRA = (
    "numeric_evidence", "capability_evidence", "capability_pin",
    "audit_baseline", "coverage", "figure_integrity", "doc_claims",
    "numeric_evidence_registry", "benchmark_pins", "registry",
)
SCRATCH_RE = re.compile(
    r"(^|/)_(?:scratch|patch|deliver|tmp|probe)|tmp_probe|^post_report|"
    r"^run_[a-z]|^publish_report|^make_report|scripts/_tpapi_tmp",
    re.IGNORECASE,
)


def area_of(path: str) -> str:
    if path.startswith("src/"):
        return "src"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("docs/"):
        return "docs"
    if path.startswith(("scripts/", "tools/")):
        return "scripts"
    if path.startswith(("benchmarks/", "examples/", "data/")):
        return "benchmarks"
    return "root"


def theme_of(path: str) -> str:
    low = path.lower()
    if low.startswith("src/"):
        if any(k in low for k in EVIDENCE_INFRA):
            return "T2-evidence-infra"
        if any(k in low for k in HYDRO_KERNEL):
            return "T1-hydro-kernel"
        return "T3-solver-core"
    if low.startswith("tests/"):
        if any(k in low for k in EVIDENCE_INFRA):
            return "T4-test-evidence"
        return "T4-test-physics"
    if low.startswith("docs/"):
        return "T5-docs-report"
    if low.startswith(("scripts/", "tools/", "benchmarks/", "examples/", "data/")):
        return "T6-tooling-data"
    return "T7-root-scratch"


def gate_of(subject: str) -> str:
    m = re.search(r"\(m(\d+)\)\s+via\s+(merge-master|push-merge)", subject)
    if m:
        return f"merge:{m.group(2)}"
    if re.match(r"^m\d+:", subject):
        return "mission-commit"
    if subject.startswith("Merge"):
        return "merge:other"
    return "direct"


def mission_of(subject: str) -> str:
    m = re.search(r"\bm(\d{3,6})\b", subject)
    return f"m{m.group(1)}" if m else "-"


# ------------------------------------------------------------------- analysis


def build(window: int) -> dict:
    head = git("rev-parse", "HEAD").strip()
    upstream = git("rev-parse", "origin/main").strip()
    ahead = git("rev-list", "--count", f"{upstream}..HEAD").strip()
    behind = git("rev-list", "--count", f"HEAD..{upstream}").strip()
    head_main = "main" if git("branch", "--show-current").strip() == "" else \
        git("branch", "--show-current").strip()

    log = git("log", f"-{window}", "--format=%H%x1f%h%x1f%ci%x1f%s%x1f%P")
    commits = []
    for line in log.splitlines():
        sha, abbr, date, subject, parents = line.split("\x1f")
        rows = numstat(sha)
        areas = Counter(area_of(r["path"]) for r in rows)
        themes = Counter(theme_of(r["path"]) for r in rows)
        primary_theme = themes.most_common(1)[0][0] if themes else "empty-diff"
        commits.append(
            {
                "sha": sha,
                "abbr": abbr,
                "patch_key": patch_key(sha),
                "date": date,
                "subject": subject,
                "parents": parents.split(),
                "is_merge": len(parents.split()) > 1,
                "gate": gate_of(subject),
                "mission": mission_of(subject),
                "files": len(rows),
                "add": sum(r["add"] for r in rows),
                "del": sum(r["del"] for r in rows),
                "binary": sum(1 for r in rows if r["binary"]),
                "areas": dict(areas),
                "themes": dict(themes),
                "primary_theme": primary_theme,
                "paths": [r["path"] for r in rows],
                "scratch_paths": [r["path"] for r in rows if SCRATCH_RE.search(r["path"])],
            }
        )

    # ---- deduplicate: a mission commit and the merge that carries it share one
    # patch-id, so the raw window double-counts the same change.  Only the first
    # occurrence (chronologically earliest in the window) is counted; later ones
    # are recorded as duplicates.
    seen = {}
    for c in sorted(commits, key=lambda x: x["date"]):
        c["duplicate_of"] = seen.get(c["patch_key"], "-")
        seen.setdefault(c["patch_key"], c["abbr"])
    primary = [c for c in commits if c["duplicate_of"] == "-"]
    by_mission = defaultdict(lambda: {"commits": [], "files": 0, "add": 0, "del": 0,
                                      "themes": Counter(), "areas": Counter()})
    for c in primary:
        b = by_mission[c["mission"]]
        b["commits"].append(c["abbr"])
        b["files"] += c["files"]
        b["add"] += c["add"]
        b["del"] += c["del"]
        b["themes"].update(c["themes"])
        b["areas"].update(c["areas"])

    # ---- churn hotspots & cross-mission overlap over the window
    touch = Counter()
    touched_by = defaultdict(set)
    for c in primary:
        for p in c["paths"]:
            touch[p] += 1
            if c["mission"] != "-":
                touched_by[p].add(c["mission"])
    hotspots = [{"path": p, "count": n} for p, n in touch.most_common(10)]
    overlap = [
        {"path": p, "missions": sorted(ms), "count": len(ms)}
        for p, ms in touched_by.items()
        if len(ms) >= 2
    ]
    overlap.sort(key=lambda d: (-d["count"], d["path"]))

    # ---- risk flags per commit
    for c in commits:
        flags = []
        if c["gate"] == "merge:push-merge":
            flags.append("RG1-provenance-push-merge")
        if not c["is_merge"] and c["gate"] == "merge:other":
            flags.append("RG1b-merge-unlabelled")
        has_src = c["areas"].get("src", 0) > 0
        has_test = c["areas"].get("tests", 0) > 0
        if has_src and not has_test:
            flags.append("RG2-src-without-test")
        if c["scratch_paths"]:
            flags.append("RG3-scratch-artifacts(%d)" % len(c["scratch_paths"]))
        if any("capability_evidence" in p or "registry" in p for p in c["paths"]):
            flags.append("RG4-evidence-registry-churn")
        if c["areas"].get("docs", 0) > 0 and not has_src and not has_test:
            flags.append("RG7-docs-only")
        if has_test and not has_src:
            flags.append("RG8-test-only-pin")
        if any(p.endswith(".png") for p in c["paths"]):
            flags.append("RG5-binary-figure")
        c["risk_flags"] = flags
        c["risk_score"] = (
            (3 if "RG2-src-without-test" in flags else 0)
            + (2 if any(f.startswith("RG3") for f in flags) else 0)
            + (1 if "RG4-evidence-registry-churn" in flags else 0)
            + (1 if "RG7-docs-only" in flags else 0)
        )

    theme_rows = Counter()
    for c in primary:
        for t, n in c["themes"].items():
            theme_rows[t] += n

    # ---- capability-registry coverage of src modules touched in the window
    # (RG4 operationalised: new kernel modules should appear in the capability
    # registry; matching is a documented heuristic, advisory only)
    registry = []
    try:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from tensorbem import capability_evidence as ce  # noqa: WPS433

        registry = [{"key": r.capability.key, "level": r.level} for r in ce.audit()]
    except Exception as exc:  # pragma: no cover - registry must not break the audit
        registry = [{"key": f"<unavailable: {exc}>", "level": "-"}]
    keys = [r["key"].lower().replace("_", "-") for r in registry]

    def match(stem: str) -> tuple[str, list[str]]:
        """('exact'|'weak'|'none', keys); exact = full module stem appears in a key."""
        stem_n = stem.lower().replace("_", "-")
        hits = [k for k in keys if stem_n in k]
        if hits:
            return "exact", hits
        tokens = [t for t in stem_n.split("-") if len(t) > 2]
        weak = [k for k in keys if tokens and all(t in k for t in tokens)]
        return ("weak" if weak else "none"), weak

    touched_src = set()
    for c in primary:
        for path in c["paths"]:
            if path.startswith("src/tensorbem/") and path.endswith(".py"):
                stem = os.path.basename(path)[:-3]
                if stem in ("__init__",):
                    continue
                touched_src.add((c["mission"], stem))
    coverage = []
    for mission, stem in sorted(touched_src):
        kind, hits = match(stem)
        coverage.append({"mission": mission, "module": stem, "match": kind,
                         "matched_keys": hits, "covered": kind == "exact"})
    uncovered = [c for c in coverage if c["match"] == "none"]
    weak = [c for c in coverage if c["match"] == "weak"]

    size = git("count-objects", "-vH")
    return {
        "repo": "TensorBEM",
        "worktree": ROOT,
        "branch": head_main,
        "head": head,
        "upstream_ref": "origin/main",
        "upstream_sha": upstream,
        "ahead": int(ahead),
        "behind": int(behind),
        "window": window,
        "commits": commits,
        "n_window": len(commits),
        "n_distinct_changes": len(primary),
        "n_duplicate_carriers": len(commits) - len(primary),
        "by_mission": {
            k: {"commits": v["commits"], "files": v["files"], "add": v["add"],
                "del": v["del"], "themes": dict(v["themes"]),
                "areas": dict(v["areas"])}
            for k, v in sorted(by_mission.items())
        },
        "theme_histogram": dict(theme_rows.most_common()),
        "hotspots": hotspots,
        "cross_mission_overlap": overlap,
        "registry": {"n_capabilities": len(registry), "keys": sorted(keys)},
        "registry_coverage": coverage,
        "registry_uncovered": uncovered,
        "registry_weak": weak,
        "repo_size": [l for l in size.splitlines() if l],
    }


# ---------------------------------------------------------------- rendering


def render_md(d: dict) -> str:
    L = []
    A = L.append
    A("# TensorBEM — upstream delta & recent-commit theme inventory (m17083)\n")
    A("## 1. Upstream delta verdict\n")
    A(f"- worktree branch: `{d['branch']}`  HEAD: `{d['head'][:12]}`")
    A(f"- upstream: `{d['upstream_ref']}` = `{d['upstream_sha'][:12]}`")
    A(f"- `origin/main..HEAD` = **{d['ahead']}** commits ahead, "
      f"`HEAD..origin/main` = **{d['behind']}** commits behind")
    A(f"- verdict: **{'UP-TO-DATE (delta set empty)' if d['ahead']==0 and d['behind']==0 else 'DIVERGED'}**")
    A("- consequence: there is no upstream delta to review, so this report falls back "
      "to the mandated inventory of the last %d commits (theme evolution + risk).\n" % d["window"])
    A(f"- de-duplication: the window holds {d['n_window']} commits but only "
      f"**{d['n_distinct_changes']} distinct changes** — {d['n_duplicate_carriers']} commit(s) "
      "are the merge that carries a mission commit already in the window (same `git patch-id`). "
      "All theme/churn accounting below uses distinct changes only; the raw table keeps both "
      "rows so the merge provenance stays visible.\n")
    A("## 2. Theme taxonomy and histogram\n")
    A("| theme | changed file-entries (distinct changes) | meaning |")
    A("|---|---:|---|")
    meaning = {
        "T1-hydro-kernel": "source modules tagged as hydrodynamic/wave kernel or formulation",
        "T2-evidence-infra": "capability/registry/audit/coverage evidence plumbing (src)",
        "T3-solver-core": "other src modules (solver/assembly/IO)",
        "T4-test-physics": "physics/verification test files",
        "T4-test-evidence": "tests covering evidence infrastructure",
        "T5-docs-report": "docs/ reports, figures, JSON evidence",
        "T6-tooling-data": "scripts/, tools/, benchmarks/, data/ assets",
        "T7-root-scratch": "repo-root scratch/probe/report helper scripts",
    }
    for t, n in d["theme_histogram"].items():
        A(f"| `{t}` | {n} | {meaning.get(t,'')} |")
    A("\n## 3. Commit-by-commit inventory (first-parent diff)\n")
    A("| # | commit | date | gate | mission | files | +/− | primary theme | dup-of | risk flags |")
    A("|---:|---|---|---|---|---:|---|---|---|---|")
    for i, c in enumerate(d["commits"], 1):
        A(f"| {i} | `{c['abbr']}` | {c['date'][:16]} | {c['gate']} | {c['mission']} | "
          f"{c['files']} | +{c['add']}/−{c['del']} | {c['primary_theme']} | "
          f"{c['duplicate_of']} | {', '.join(c['risk_flags']) or '—'} |")
    A("\n## 4. Change-theme list (evolution of the window)\n")
    # group commits by primary theme preserving chronological order
    order = [c for c in reversed(d["commits"]) if c["duplicate_of"] == "-"]
    by_theme = defaultdict(list)
    for c in order:
        by_theme[c["primary_theme"]].append(c)
    for t, cs in by_theme.items():
        missions = ", ".join(sorted({c["mission"] for c in cs if c["mission"] != "-"})) or "-"
        A(f"### {t} — {len(cs)} commit(s), missions: {missions}")
        for c in cs:
            A(f"- `{c['abbr']}` {c['subject']} — {c['files']} file(s), "
              f"+{c['add']}/−{c['del']}; areas {c['areas']}")
        A("")
    A("## 5. Mission-level aggregation (distinct changes only)\n")
    A("| mission | commits | files | +/− | theme mix | areas |")
    A("|---|---|---:|---|---|---|")
    for m, b in d["by_mission"].items():
        mix = ", ".join(f"{k}×{v}" for k, v in
                        sorted(b["themes"].items(), key=lambda kv: -kv[1])[:3])
        A(f"| {m} | {', '.join('`'+x+'`' for x in b['commits'])} | {b['files']} | "
          f"+{b['add']}/−{b['del']} | {mix} | {b['areas']} |")
    A("")
    A("## 6. Risk annotations\n")
    A("| id | rule | why it is a risk | commits hit |")
    A("|---|---|---|---|")
    rules = {
        "RG1-provenance-push-merge": ("landed through the push-merge gate instead of merge-master",
                                      "different CI/merge path → gate coverage must be assumed, not verified, per commit"),
        "RG1b-merge-unlabelled": ("merge commit without (mNNNNN) via ... provenance tag",
                                  "traceability gap: merge cannot be attributed to a mission from the subject alone"),
        "RG2-src-without-test": ("src/ changed but tests/ untouched in the same commit",
                                 "new/changed numerical behaviour ships without a same-commit regression pin"),
        "RG4-evidence-registry-churn": ("capability/evidence registry files rewritten",
                                        "registry drift: docs registry, capability index and code must stay consistent"),
        "RG7-docs-only": ("docs-only change", "claims updated without runtime proof in this commit"),
        "RG8-test-only-pin": ("test-only change", "low risk; tightening pins against existing implementation"),
        "RG5-binary-figure": ("new PNG committed", "repository size growth / figure-provenance discipline"),
    }
    counts = defaultdict(list)
    for c in d["commits"]:
        for f in c["risk_flags"]:
            base = f.split("(")[0]
            counts[base].append(c["abbr"])
    for rid, (rule, why) in rules.items():
        hit = counts.get(rid, [])
        A(f"| `{rid}` | {rule} | {why} | {', '.join(hit) if hit else '0'} ({len(hit)}) |")
    rg3 = [f for f in counts if f.startswith("RG3")]
    n3 = len(counts.get("RG3-scratch-artifacts", []))
    A(f"| `RG3-scratch-artifacts` | probe/deliver scripts committed into scripts/ or repo root | "
      f"repo hygiene: scratch helpers become unowned surface area | {n3} commit(s) |")
    A("\n### 6.1 Churn hotspots (files touched more than once in the window)\n")
    A("| file | commits touching it |")
    A("|---|---:|")
    for h in d["hotspots"]:
        if h["count"] > 1:
            A(f"| `{h['path']}` | {h['count']} |")
    A("\n### 6.2 Cross-mission overlap (same file, ≥2 missions)\n")
    if d["cross_mission_overlap"]:
        A("| file | missions |")
        A("|---|---|")
        for o in d["cross_mission_overlap"][:15]:
            A(f"| `{o['path']}` | {', '.join(o['missions'])} |")
    else:
        A("_none in window_")
    A("\n### 6.3 Capability-registry coverage of src modules touched in the window\n")
    A(f"Registry holds {d['registry']['n_capabilities']} capabilities. Matching is a "
      "documented heuristic (module stem substring, else all tokens) and is advisory, "
      "not a proof of absence.\n")
    A("| mission | module | match | matched registry key(s) |")
    A("|---|---|---|---|")
    label = {"exact": "exact", "weak": "weak (token overlap only)", "none": "**none**"}
    for c in d["registry_coverage"]:
        A(f"| {c['mission']} | `{c['module']}` | {label[c['match']]} | "
          f"{', '.join(c['matched_keys']) or '—'} |")
    A(f"\n- no registry entry at all: **{len(d['registry_uncovered'])}** — "
      + (", ".join(f"`{c['module']}` ({c['mission']})" for c in d["registry_uncovered"]) or "none"))
    A(f"- token-overlap only (entry may be a different formulation; needs a human/agent "
      f"decision): **{len(d['registry_weak'])}** — "
      + (", ".join(f"`{c['module']}` ({c['mission']})" for c in d["registry_weak"]) or "none"))
    A("")
    A("\n## 7. Repo size accounting\n```\n" + "\n".join(d["repo_size"]) + "\n```\n")
    A("## 8. Limitations (honest)\n")
    A("- Theme families are keyed on path/subject tokens per the explicit keyword lists in the "
      "extraction script; a commit touching two families is filed under its dominant family.")
    A("- Risk flags are structural (diff shape), not semantic: RG2 says *no same-commit test*, "
      "not *untested* — the suite may cover the module elsewhere.")
    A("- Upstream comparison is against the locally fetched `origin/main` ref; a fetch was not "
      "possible in the sandbox, so a newer server-side commit cannot be excluded.")
    A("\n_Generated by `scripts/m17083_upstream_delta_inventory.py`; every number in this file is "
      "emitted from git plumbing._\n")
    return "\n".join(L)


def figure(d: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = [c for c in reversed(d["commits"]) if c["duplicate_of"] == "-"]
    labels = [f"{c['abbr']} {c['mission']}" for c in cs]
    y = range(len(cs))
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [3, 2]}
    )
    colors = {
        "T1-hydro-kernel": "#1f77b4",
        "T2-evidence-infra": "#2ca02c",
        "T3-solver-core": "#9467bd",
        "T4-test-physics": "#ff7f0e",
        "T4-test-evidence": "#bcbd22",
        "T5-docs-report": "#7f7f7f",
        "T6-tooling-data": "#17becf",
        "T7-root-scratch": "#d62728",
        "empty-diff": "#dddddd",
    }
    bottom = [0] * len(cs)
    for t in d["theme_histogram"]:
        vals = [c["themes"].get(t, 0) for c in cs]
        if not any(vals):
            continue
        ax1.barh(list(y), vals, left=bottom, color=colors.get(t, "#333333"), label=t)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, c in enumerate(cs):
        if c["risk_score"]:
            ax1.text(
                bottom[i] + 0.3, i, "risk=%d" % c["risk_score"],
                va="center", fontsize=7, color="#b30000",
            )
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(labels, fontsize=7.5)
    ax1.set_xlabel("changed file entries in first-parent diff")
    ax1.set_title("(a) theme composition per distinct change (oldest at bottom)", fontsize=10)
    ax1.legend(fontsize=6.5, loc="lower right")
    ax1.grid(axis="x", alpha=0.3)

    flag_names = sorted({f.split("(")[0] for c in d["commits"] for f in c["risk_flags"]})
    M = [[1 if any(f.startswith(fl) for f in c["risk_flags"]) else 0 for c in cs] for fl in flag_names]
    ax2.imshow(M, cmap="Reds", aspect="auto", vmin=0, vmax=1.4)
    ax2.set_xticks(range(len(cs)))
    ax2.set_xticklabels([c["abbr"] for c in cs], rotation=90, fontsize=7)
    ax2.set_yticks(range(len(flag_names)))
    ax2.set_yticklabels(flag_names, fontsize=6.5)
    ax2.set_title("(b) risk-flag incidence (distinct changes)", fontsize=10)
    for i, r in enumerate(M):
        for j, v in enumerate(r):
            if v:
                ax2.text(j, i, "×", ha="center", va="center", fontsize=7, color="black")
    fig.suptitle(
        f"TensorBEM m17083 — upstream delta {d['ahead']}/{d['behind']} (ahead/behind); "
        f"{d['n_distinct_changes']} distinct changes in last {d['window']} commits",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = build(args.window)
    md = render_md(d)
    with open(os.path.join(OUT, "INVENTORY.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(OUT, "study.json"), "w") as fh:
        json.dump(d, fh, indent=2, ensure_ascii=False)
    png = os.path.join(OUT, "fig_m17083_theme_risk.png")
    figure(d, png)
    sha = {}
    for name in sorted(os.listdir(OUT)):
        if name == "sha256.txt":
            continue
        pth = os.path.join(OUT, name)
        if os.path.isfile(pth):
            sha[name] = hashlib.sha256(open(pth, "rb").read()).hexdigest()
    with open(os.path.join(OUT, "sha256.txt"), "w") as fh:
        for k, v in sha.items():
            fh.write(f"{v}  {k}\n")
    print(json.dumps({"ahead": d["ahead"], "behind": d["behind"],
                      "commits": len(d["commits"]),
                      "theme_histogram": d["theme_histogram"],
                      "sha256": sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())