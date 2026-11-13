"""Three-layer memory (L1 bootstrap head / L2 seat-local / L3 knowledge base).

Compression policy in one place so it can be audited and unit-tested:

* **L1 (global bootstrap)** — stable head is protected (``stable=True``); the
  file's contract is append-only, so the compressor may only add, never
  reorder or drop head entries.
* **L2 (seat-local)** — half-life decay on mission index; content untouched
  for ``min_age_for_forget`` missions and with ``use_count == 0`` is forgotten
  (archived first, see below).
* **L3 (knowledge base)** — same rule with a longer half-life; duplicates
  across layers are collapsed into the surviving entry with the largest
  ``use_count``.

Forgetting is never deletion: every removed entry is written to
``<archive_root>/<layer>/<item_id>.md`` and an append-only JSONL record with
the entry's sha256 is appended to ``forget_log.jsonl`` before the layer file is
rewritten.  That makes "what did we forget and when" re-verifiable offline.

Retention floor: ``FLOOR_KINDS`` (TOLERANCE / NEGATIVE / TRAP) are never
forgotten; if they alone exceed the layer budget the layer reports
``overflow_tokens`` instead of silently dropping the acceptance criteria.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Sequence

from .model import FLOOR_KINDS, MemoryEntry, dedupe, sha256_text


@dataclass(frozen=True)
class LayerConfig:
    name: str
    budget_tokens: int
    half_life: float
    min_age_for_forget: int
    stable_head: bool = False
    allow_forget: bool = True


DEFAULT_LAYERS: tuple[LayerConfig, ...] = (
    LayerConfig("L1", budget_tokens=1400, half_life=1e9, min_age_for_forget=10**6,
                stable_head=True, allow_forget=False),
    LayerConfig("L2", budget_tokens=900, half_life=12.0, min_age_for_forget=6),
    LayerConfig("L3", budget_tokens=2200, half_life=30.0, min_age_for_forget=15),
)

# Compression cadence: whichever comes first.
CADENCE_MISSIONS = 5          # compress at most every N missions
CADENCE_USAGE = 0.85          # ... or as soon as a layer is 85% full


@dataclass
class CompressResult:
    mission_index: int
    kept: dict[str, list[MemoryEntry]] = field(default_factory=dict)
    forgotten: list[dict] = field(default_factory=list)
    overflow_tokens: dict[str, int] = field(default_factory=dict)
    dropped_duplicates: int = 0
    log_path: str = ""
    triggered: bool = True

    @property
    def kept_tokens(self) -> dict[str, int]:
        return {k: sum(e.item.tokens for e in v) for k, v in self.kept.items()}

    def to_dict(self) -> dict:
        return {
            "mission_index": self.mission_index,
            "triggered": self.triggered,
            "kept": {k: [e.to_dict() for e in v] for k, v in self.kept.items()},
            "kept_tokens": self.kept_tokens,
            "forgotten": self.forgotten,
            "overflow_tokens": self.overflow_tokens,
            "dropped_duplicates": self.dropped_duplicates,
            "log_path": self.log_path,
        }


def should_compress(entries: Sequence[MemoryEntry], mission_index: int,
                    layers: Sequence[LayerConfig] = DEFAULT_LAYERS,
                    last_compress_mission: int | None = None) -> bool:
    if last_compress_mission is not None and mission_index - last_compress_mission >= CADENCE_MISSIONS:
        return True
    by_layer = {c.name: c for c in layers}
    for name, cfg in by_layer.items():
        used = sum(e.item.tokens for e in entries if e.layer == name)
        if cfg.budget_tokens and used / cfg.budget_tokens >= CADENCE_USAGE:
            return True
    return False


def _archive(entry: MemoryEntry, archive_root: str) -> str:
    d = os.path.join(archive_root, entry.layer)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{entry.item.item_id}.md")
    body = [
        "---",
        f"item_id: {entry.item.item_id}",
        f"kind: {entry.item.kind}",
        f"source: {entry.item.source}:{entry.item.lineno}",
        f"sha256: {entry.item.sha256}",
        f"layer: {entry.layer}",
        f"use_count: {entry.use_count}",
        f"last_used_mission: {entry.last_used_mission}",
        "---",
        "",
        entry.item.text,
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body))
    return path


def compress(
    entries: Sequence[MemoryEntry],
    mission_index: int,
    out_dir: str,
    layers: Sequence[LayerConfig] = DEFAULT_LAYERS,
    force: bool = False,
    last_compress_mission: int | None = None,
) -> CompressResult:
    cfg = {c.name: c for c in layers}
    result = CompressResult(mission_index=mission_index, triggered=True)

    if not force and not should_compress(entries, mission_index, layers, last_compress_mission):
        result.triggered = False
        result.kept = {
            name: [e for e in entries if e.layer == name] for name in cfg
        }
        return result

    archive_root = os.path.join(out_dir, "archive")
    os.makedirs(archive_root, exist_ok=True)
    log_path = os.path.join(out_dir, "forget_log.jsonl")
    result.log_path = log_path

    records: list[dict] = []
    for name, c in cfg.items():
        mine = [e for e in entries if e.layer == name]

        # --- duplicate collapse (keep the most-used / newest survivor)
        mine.sort(key=lambda e: (-e.use_count, -e.last_used_mission, e.item.item_id))
        survivors, _dropped = dedupe([e.item for e in mine])
        keep_ids = {i.item_id for i in survivors}
        for e in mine:
            if e.item.item_id not in keep_ids:
                result.dropped_duplicates += 1
                records.append(_forget_record(e, mission_index, c, "duplicate", 0.0,
                                              _archive(e, archive_root)))
        mine = [e for e in mine if e.item.item_id in keep_ids]

        # --- rank by retention score
        mine.sort(key=lambda e: (-e.retention_score(mission_index, c.half_life), e.item.item_id))
        used = 0
        overflow = 0
        kept: list[MemoryEntry] = []
        floor_seen: dict[str, int] = {}
        for e in mine:
            score = e.retention_score(mission_index, c.half_life)
            is_floor = e.item.kind in FLOOR_KINDS and floor_seen.get(e.item.kind, 0) < 1
            cost = e.item.tokens
            fits = used + cost <= c.budget_tokens
            if e.stable or is_floor:
                kept.append(e)
                used += cost
                if is_floor:
                    floor_seen[e.item.kind] = floor_seen.get(e.item.kind, 0) + 1
                if used > c.budget_tokens:
                    overflow = max(overflow, used - c.budget_tokens)
                continue
            if fits:
                kept.append(e)
                used += cost
                continue
            forgettable = (
                c.allow_forget
                and e.use_count == 0
                and e.age(mission_index) >= c.min_age_for_forget
            )
            if forgettable:
                records.append(_forget_record(e, mission_index, c, "budget_decay", score,
                                              _archive(e, archive_root)))
                result.forgotten.append(records[-1])
            else:
                kept.append(e)
                used += cost
                overflow = max(overflow, used - c.budget_tokens)
        result.kept[name] = kept
        if overflow:
            result.overflow_tokens[name] = overflow

    # --- append-only forget log
    if records:
        with open(log_path, "a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def _forget_record(e: MemoryEntry, mission_index: int, cfg: LayerConfig,
                   reason: str, score: float, archive_path: str) -> dict:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mission_index": mission_index,
        "layer": cfg.name,
        "item_id": e.item.item_id,
        "kind": e.item.kind,
        "sha256": e.item.sha256,
        "reason": reason,
        "retention_score": round(score, 6),
        "use_count": e.use_count,
        "age": e.age(mission_index),
        "archive_path": archive_path,
    }


def touch(entries: Sequence[MemoryEntry], item_id: str, mission_index: int) -> None:
    for e in entries:
        if e.item.item_id == item_id:
            e.use_count += 1
            e.last_used_mission = mission_index


def load_memory(path: str) -> list[MemoryEntry]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [MemoryEntry.from_dict(d) for d in data]


def save_memory(entries: Sequence[MemoryEntry], path: str) -> str:
    payload = [e.to_dict() for e in entries]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def verify_forget_log(log_path: str, archive_root: str) -> dict:
    """Re-read every forget record and re-hash its archive copy."""
    ok = bad = 0
    problems: list[str] = []
    if not os.path.exists(log_path):
        return {"records": 0, "ok": 0, "bad": 0, "problems": ["log missing"]}
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ap = rec.get("archive_path", "")
            if not ap or not os.path.exists(ap):
                bad += 1
                problems.append(f"missing archive: {rec.get('item_id')}")
                continue
            body = open(ap, "r", encoding="utf-8").read()
            text = body.split("---", 2)[-1].strip()
            if sha256_text(text) == rec.get("sha256"):
                ok += 1
            else:
                bad += 1
                problems.append(f"hash mismatch: {rec.get('item_id')}")
    return {"records": ok + bad, "ok": ok, "bad": bad, "problems": problems}