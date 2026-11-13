"""Core data types for context craft (task brief packages + layered memory).

Deliberately domain-agnostic: nothing here knows about any solver, ship,
physics quantity or repository layout.  Items are opaque text plus
metadata; classification uses curation-semantic markers only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

# ---------------------------------------------------------------- kinds
# Ordered by decision relevance: a task brief that has budget for only one
# line should carry the acceptance criterion (TOLERANCE) before anything else.
KINDS: tuple[str, ...] = (
    "TOLERANCE",   # acceptance threshold / criterion with a number
    "NEGATIVE",    # known-bad, forbidden, unsupported, honest FAIL
    "TRAP",        # recorded pitfall
    "VERDICT",     # prior verdict / evidence grade on the same subject
    "PIN",         # frozen reference value / commit / hash anchor
    "CONTEXT",     # background prose, no decision content
)

KIND_WEIGHT: dict[str, float] = {
    "TOLERANCE": 1.00,
    "NEGATIVE": 0.95,
    "TRAP": 0.90,
    "VERDICT": 0.75,
    "PIN": 0.70,
    "CONTEXT": 0.40,
}

# Retention floor: these kinds are never silently dropped by a compressor.
FLOOR_KINDS: tuple[str, ...] = ("TOLERANCE", "NEGATIVE", "TRAP")

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate.

    CJK code points are counted at 1.5 chars/token, everything else at
    4 chars/token.  Documented in docs/context_craft_protocol.md §3; the A/B
    harness reports raw characters too so the conclusion does not depend on
    this estimator alone.
    """
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(math.ceil(cjk / 1.5 + other / 4.0))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    """Whitespace/case folding used for duplicate detection only."""
    return re.sub(r"\s+", " ", text).strip().lower()


def canon_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FactItem:
    """One atomic, citable unit of knowledge."""

    item_id: str
    kind: str
    text: str
    source: str          # repo-relative path
    lineno: int
    sha256: str          # sha256 of the exact source line
    heading: str = ""    # breadcrumb of enclosing section(s)

    @staticmethod
    def build(kind: str, text: str, source: str, lineno: int, heading: str = "") -> "FactItem":
        sha = sha256_text(text)
        return FactItem(
            item_id=sha256_text(f"{source}:{lineno}:{sha}")[:16],
            kind=kind,
            text=text.strip(),
            source=source,
            lineno=int(lineno),
            sha256=sha,
            heading=heading,
        )

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tokens"] = self.tokens
        return d

    @staticmethod
    def from_dict(d: dict) -> "FactItem":
        return FactItem(
            item_id=d["item_id"],
            kind=d["kind"],
            text=d["text"],
            source=d["source"],
            lineno=int(d["lineno"]),
            sha256=d.get("sha256") or sha256_text(d["text"]),
            heading=d.get("heading", ""),
        )


@dataclass
class MemoryEntry:
    """A FactItem as held in one of the three memory layers."""

    item: FactItem
    layer: str
    use_count: int = 0
    last_used_mission: int = 0
    stable: bool = False      # L1 head / append-only sections: never forgotten

    def age(self, mission_index: int) -> int:
        return max(0, int(mission_index) - int(self.last_used_mission))

    def retention_score(self, mission_index: int, half_life: float) -> float:
        decay = 0.5 ** (self.age(mission_index) / max(half_life, 1e-9))
        return KIND_WEIGHT.get(self.item.kind, 0.4) * decay + 0.10 * math.log1p(self.use_count)

    def to_dict(self) -> dict:
        d = self.item.to_dict()
        d.update(
            layer=self.layer,
            use_count=self.use_count,
            last_used_mission=self.last_used_mission,
            stable=self.stable,
        )
        return d

    @staticmethod
    def from_dict(d: dict) -> "MemoryEntry":
        return MemoryEntry(
            item=FactItem.from_dict(d),
            layer=d.get("layer", "L3"),
            use_count=int(d.get("use_count", 0)),
            last_used_mission=int(d.get("last_used_mission", 0)),
            stable=bool(d.get("stable", False)),
        )


@dataclass
class BriefPackage:
    query: str
    budget_tokens: int
    items: list[FactItem] = field(default_factory=list)
    dropped_duplicates: int = 0
    skipped_over_budget: int = 0

    @property
    def tokens_used(self) -> int:
        return sum(i.tokens for i in self.items)

    @property
    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for i in self.items:
            out[i.kind] = out.get(i.kind, 0) + 1
        return out

    @property
    def density(self) -> float:
        """Decision-bearing items per 1k tokens (CONTEXT excluded)."""
        if not self.tokens_used:
            return 0.0
        dec = sum(1 for i in self.items if i.kind != "CONTEXT")
        return 1000.0 * dec / self.tokens_used

    def provenance(self) -> list[dict]:
        """Zero-hand-copy sidecar: one entry per carried line, hash-bound."""
        return [
            {
                "item_id": i.item_id,
                "kind": i.kind,
                "source": i.source,
                "lineno": i.lineno,
                "sha256": i.sha256,
                "tokens": i.tokens,
            }
            for i in self.items
        ]

    @property
    def package_sha256(self) -> str:
        return sha256_text(canon_json(self.provenance()))

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "budget_tokens": self.budget_tokens,
            "tokens_used": self.tokens_used,
            "density_per_1k": round(self.density, 3),
            "kinds": self.kinds,
            "dropped_duplicates": self.dropped_duplicates,
            "skipped_over_budget": self.skipped_over_budget,
            "package_sha256": self.package_sha256,
            "provenance": self.provenance(),
        }

    def to_markdown(self) -> str:
        lines = [
            "# Task brief package",
            "",
            f"- query: `{self.query}`",
            f"- budget: {self.budget_tokens} tok · used: {self.tokens_used} tok "
            f"· density: {self.density:.2f} decision-items/1k tok",
            f"- kinds: {self.kinds}",
            f"- package sha256: `{self.package_sha256}`",
            "",
            "| # | kind | source:line | sha256[:8] | text |",
            "|---|---|---|---|---|",
        ]
        for n, i in enumerate(self.items, 1):
            txt = i.text.replace("|", "\\|")
            lines.append(
                f"| {n} | {i.kind} | `{i.source}:{i.lineno}` | `{i.sha256[:8]}` | {txt} |"
            )
        return "\n".join(lines) + "\n"


def total_tokens(items: Iterable[FactItem]) -> int:
    return sum(i.tokens for i in items)


def dedupe(items: Sequence[FactItem], threshold: float = 0.85) -> tuple[list[FactItem], int]:
    """Exact + near-duplicate removal, order preserving (first wins).

    Near-duplicate test = Jaccard over character 3-grams of the folded text.
    """
    kept: list[FactItem] = []
    seen_norm: set[str] = set()
    shingles: list[set[str]] = []
    dropped = 0
    for it in items:
        norm = normalize(it.text)
        if norm in seen_norm:
            dropped += 1
            continue
        sh = {norm[k : k + 3] for k in range(max(1, len(norm) - 2))}
        dup = False
        for other in shingles:
            inter = len(sh & other)
            union = len(sh | other) or 1
            if inter / union >= threshold:
                dup = True
                break
        if dup:
            dropped += 1
            continue
        seen_norm.add(norm)
        shingles.append(sh)
        kept.append(it)
    return kept, dropped