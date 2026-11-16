#!/usr/bin/env python3
"""Row-level tables for T1 / T2 of the Havelock surface-measure study (round 7).

Why this exists
---------------
``benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py``
(*measure correction study*, m21580/round 5) published its T1 (economy mesh, 320
panels, ``ka = 0.1 ... 5.0``) and T2 (coarsest mesh, 80 panels, ``ka = 0.1 ...
3.0``) *verdicts* only: the original driver was interrupted before
``write_artifacts`` ran, so the per-``ka`` rows behind those verdicts were never
committed (the committed ``verdicts.json`` says so in its own ``not_run`` text).
A verdict whose rows are not on disk cannot be re-derived by a third party -- it
can only be trusted.  Round 6 closed this for T3 (case mesh); this script closes
it for T1/T2.

What it does (no new physics, no new parameters)
-----------------------------------------------
* imports the *merged* study module and reuses ``Tier``, ``_band`` and
  ``_verdict`` verbatim -- the gate (5 % per item), the band and the three
  measures (``flat`` / ``patch`` / ``uniform``) are whatever the study declares;
* recomputes the T1 and T2 rows and re-derives the verdicts **from those rows**;
* cross-checks the re-derived verdicts against the committed
  ``measure_correction/verdicts.json`` (control R2: identity of the published
  verdicts with the re-run, item counts, band-max value and its location);
* writes ``measure_correction/numbers_t1t2_rows.json`` with the rows, the
  re-derived verdicts, the R2 control and the environment.

The point is auditability, not a new claim: if a re-derived verdict disagrees
with the committed one, that disagreement is reported and **not** smoothed over
(the committed verdicts are never rewritten by this script).

    python3 scripts/havelock_measure_t1t2_rows.py --outdir <dir>          # full
    python3 scripts/havelock_measure_t1t2_rows.py --smoke                 # 1 ka/tier
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = (
    ROOT
    / "benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py"
)
COMMITTED_VERDICTS = (
    ROOT / "docs/case_bem_submerged_sphere_havelock/measure_correction/verdicts.json"
)
OUT_NAME = "numbers_t1t2_rows.json"

# Tiers declared by the study (its ``build_study`` lines for T1 / T2).
TIERS = (
    ("T1:economy_s2_band", 2, 2.0, None),   # subdivision, submergence, band (None = study default)
    ("T2:coarsest_s1_band", 1, 2.0, None),
)


def _load_study() -> Any:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location("measure_correction_study", STUDY_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot import {STUDY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bands(mcs: Any) -> dict[str, tuple[float, ...]]:
    return {
        "T1:economy_s2_band": tuple(mcs.TIER_BAND),
        "T2:coarsest_s1_band": tuple(mcs.COARSE_BAND),
    }


def _compare(derived: dict[str, Any], committed: dict[str, Any]) -> dict[str, Any]:
    """Control R2: are the committed T1/T2 verdicts reproducible from the rows?"""
    per: dict[str, Any] = {}
    for measure in ("flat", "patch", "uniform"):
        got = derived.get(measure)
        ref = committed.get(measure)
        if got is None or ref is None:
            per[measure] = {"status": "missing_on_one_side", "derived": got is not None,
                            "committed": ref is not None}
            continue
        rel = abs(float(got["band_max_rel_error"]) - float(ref["band_max_rel_error"])) / max(
            abs(float(ref["band_max_rel_error"])), 1e-300
        )
        per[measure] = {
            "n_items_equal": int(got["n_items"]) == int(ref["n_items"]),
            "n_items_over_gate_equal": int(got["n_items_over_gate"])
            == int(ref["n_items_over_gate"]),
            "verdict_equal": got["verdict"] == ref["verdict"],
            "band_max_at_equal": (
                abs(float(got["band_max_at"]["ka"]) - float(ref["band_max_at"]["ka"])) < 1e-12
                and got["band_max_at"]["quantity"] == ref["band_max_at"]["quantity"]
            ),
            "band_max_rel_error": {
                "derived": float(got["band_max_rel_error"]),
                "committed": float(ref["band_max_rel_error"]),
                "rel_diff": rel,
            },
        }
    ok = all(
        v.get("n_items_equal", False)
        and v.get("n_items_over_gate_equal", False)
        and v.get("verdict_equal", False)
        and v.get("band_max_at_equal", False)
        and v["band_max_rel_error"]["rel_diff"] < 1e-6
        for v in per.values()
        if "band_max_rel_error" in v
    )
    return {"control": "R2 committed verdicts vs rows re-derived here", "per_measure": per,
            "holds": bool(ok)}


def build(*, smoke: bool = False, tier_filter: str | None = None,
          kas: Sequence[float] | None = None) -> dict[str, Any]:
    mcs = _load_study()
    bands = _bands(mcs)
    started = time.perf_counter()
    study: dict[str, Any] = {
        "case_code": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "pool_code": "BEM-SUBMERGED-SPHERE-HAVELOCK",
        "round": "round7",
        "purpose": (
            "publish the per-ka rows behind the committed T1/T2 verdicts of the "
            "surface-measure study (the original driver was interrupted before "
            "write_artifacts, so only the verdicts were on disk)"
        ),
        "study_module": str(STUDY_PATH.relative_to(ROOT)),
        "gate": mcs.GATE,
        "tiers": {},
        "verdicts": {},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "not_run": [],
    }
    committed = json.loads(COMMITTED_VERDICTS.read_text())["verdicts"] if COMMITTED_VERDICTS.exists() else {}
    for label, subdivision, submergence, _ in TIERS:
        if tier_filter and not label.startswith(tier_filter):
            continue
        band = bands[label]
        if kas is not None:
            band = tuple(ka for ka in band if ka in set(kas))
        if smoke:
            band = band[:1]
        tier = mcs.Tier(subdivision, submergence)
        rows = mcs._band(tier, band)
        verdict = mcs._verdict(rows, label)
        study["tiers"][label] = {
            "summary": tier.summary,
            "band": list(band),
            "rows": rows,
        }
        study["verdicts"][label] = verdict
        study.setdefault("controls", {})[label] = _compare(verdict, committed.get(label, {}))
        print(f"{label}: {json.dumps(verdict, default=float)}", flush=True)
    # A partial run (tier/ka filter) is only a shard: it must not claim verdicts,
    # because a verdict over a truncated band is not the study's verdict.
    partial = kas is not None or tier_filter is not None
    if partial:
        study["partial"] = True
        study["verdicts"] = {}
        study["controls"] = {}
        study["not_run"] = ["verdicts need the full band -- merge the shards first"]
    study["runtime_seconds"] = time.perf_counter() - started
    return study


def write(study: dict[str, Any], outdir: Path, name: str = OUT_NAME) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / name
    target.write_text(json.dumps(study, indent=2, default=float) + "\n")
    print(f"  {target}")
    return [target]


def merge(paths: Sequence[Path]) -> dict[str, Any]:
    """Merge tier/ka shards: concatenate rows, then derive verdicts and control R2."""
    shards = [json.loads(Path(p).read_text()) for p in paths]
    base = dict(shards[0])
    mcs = _load_study()
    committed = json.loads(COMMITTED_VERDICTS.read_text())["verdicts"] \
        if COMMITTED_VERDICTS.exists() else {}
    tiers: dict[str, Any] = {}
    for doc in shards:
        for label, payload in doc["tiers"].items():
            tiers.setdefault(label, {"summary": payload["summary"], "band": [], "rows": []})
            tiers[label]["band"] = sorted(set(tiers[label]["band"]) | set(payload["band"]))
            tiers[label]["rows"].extend(payload["rows"])
    study: dict[str, Any] = {k: v for k, v in base.items()
                             if k not in ("tiers", "verdicts", "controls", "partial", "not_run")}
    study["tiers"] = {}
    study["verdicts"] = {}
    study["controls"] = {}
    study["shards"] = [{"tiers": sorted(d.get("tiers", {})),
                        "rows": sum(len(v["rows"]) for v in d.get("tiers", {}).values()),
                        "partial": bool(d.get("partial"))} for d in shards]
    study["not_run"] = []
    dup_cells = 0
    dup_max_rel = 0.0
    for label, payload in tiers.items():
        # shards may overlap on a ka; a cell seen twice must agree to roundoff,
        # so overlapping shards are an independent-repeat control, not a duplication.
        uniq: dict[tuple[float, str], dict[str, Any]] = {}
        for row in payload["rows"]:
            key = (float(row["ka"]), str(row["measure"]))
            if key in uniq:
                dup_cells += 1
                for c in ("a33", "b33", "a11", "b11"):
                    prev, cur = uniq[key][c], row[c]
                    dup_max_rel = max(dup_max_rel, abs(prev - cur) / max(abs(prev), 1e-300))
            else:
                uniq[key] = row
        payload["rows"] = list(uniq.values())
        rows = sorted(payload["rows"], key=lambda r: (float(r["ka"]), r["measure"]))
        payload["rows"] = rows
        payload["band"] = sorted({float(r["ka"]) for r in rows})
        study["tiers"][label] = payload
        study["verdicts"][label] = mcs._verdict(rows, label)
        study["controls"][label] = _compare(study["verdicts"][label], committed.get(label, {}))
        print(f"{label}: {json.dumps(study['verdicts'][label], default=float)}", flush=True)
    return study


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "docs/case_bem_submerged_sphere_havelock/measure_correction")
    parser.add_argument("--smoke", action="store_true", help="one ka per tier (plumbing only)")
    parser.add_argument("--tier", default=None, choices=("T1", "T2"),
                        help="run one tier only (shard mode: no verdicts)")
    parser.add_argument("--kas", default=None,
                        help="comma list of ka to run (shard mode: no verdicts)")
    parser.add_argument("--shard-name", default=None,
                        help="file name for a shard artefact (default: numbers_t1t2_rows.json)")
    parser.add_argument("--merge", nargs="+", type=Path, default=None,
                        help="merge shard artefacts and derive verdicts/controls")
    args = parser.parse_args(argv)
    if args.merge:
        study = merge(args.merge)
        write(study, args.outdir)
        return 0
    kas = tuple(float(x) for x in args.kas.split(",")) if args.kas else None
    study = build(smoke=args.smoke, tier_filter=args.tier, kas=kas)
    write(study, args.outdir, args.shard_name or OUT_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())