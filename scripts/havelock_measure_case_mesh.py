#!/usr/bin/env python3
"""Round 6: run the *declared but never executed* tiers of the surface-measure study.

Anchor: ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`` (pool ``BEM-SUBMERGED-SPHERE-HAVELOCK``,
ref Havelock 1955 multipole, tolerance < 5 %).

Why this driver exists
----------------------
``benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py``
declares five tiers (T1..T5) and two controls, and states a decision that needs
T3..T5 plus the controls:

    success of the decision is (i) T1 every item < 5 % with the exact measure,
    (ii) the same tier > 5 % without it, (iii) *no tier regressing relative to
    its flat control*.

The committed artefact (``docs/case_bem_submerged_sphere_havelock/measure_correction/
verdicts.json``) records ``not_run``: T3 (case mesh, 1280 panels) *and* "per-ka rows
of the study: the run was interrupted before write_artifacts".  Only T1/T2 verdicts
exist, and no row-level table was committed.  So the published claim "the exact
measure removes the flat-panel bias" rests on the two coarse meshes only -- the
case mesh that the pool actually gates on has never been measured under the
intervention.  This driver closes that: it reuses the *same* ``Tier`` /
``_band`` / ``_verdict`` code objects (imported from the benchmark module, not
re-implemented) and adds the missing tiers plus the row-level artefacts.

Predeclared experiment (fixed *before* any number below was produced)
--------------------------------------------------------------------
Intervention (unchanged from the original study)
    ``panel_areas`` told to the operator assembly: ``flat`` = polyhedral chord
    areas, ``patch`` = exact spherical-patch areas, ``uniform`` = flat rescaled
    by ``sum(patch)/sum(flat)`` (control C1).  Panels, normals, collocation
    points and quadrature rules unchanged.

Tiers run here
    T3   case mesh (subdivision 3, 1280 panels), f/a = 2.0, band ka = 0.1 ... 5.0
         (7 wavenumbers; the originally declared T3 band 0.1/1.0/3.0 is a sub-set,
         so nothing declared is dropped -- the band is only *widened*, which
         cannot flatter a threshold that is applied per item).
    T4   economy mesh (division 2, 320 panels), f/a = 1.5 and 3.0, ka = 1.0.
    T5   low-frequency limit, economy mesh, ka = 0.01 ... 0.2 (B33' -> 0).

Predeclared predictions (falsifiable; the round is a success only if all three hold)
    P1  On the case mesh the exact measure lowers the band-max error by at least
        a factor 2 relative to the flat measure (the flat band-max is 1.431 % at
        ka = 0.5, B33'; a purely deficit-driven bias would leave ~0 residual, so
        a factor 2 is a *weak* bar).
    P2  The attribution asserted by the round-5 audit ("the ~1 % residual of the
        case mesh is mainly the 0.48 % area deficit mapped into the
        coefficients") implies, coefficient-wise,
        ``err_flat(ka) - err_patch(ka) > 0.5 * deficit`` for every ka, i.e. the
        *reduction* is at least the deficit itself.  If the reduction is instead
        a small fraction of the deficit, that attribution sentence is falsified.
    P3  Control C1 (``uniform``) does not reproduce the ``patch`` gain on the
        damping coefficients: if it did, the mechanism would be a global area
        normalisation and no surface-measure quadrature improvement would be
        needed.

Threshold / verdict discipline
    5 % per item, unchanged; it is re-evaluated per tier and measure from the
    rows (``_verdict``), never relaxed, and the honest FAIL of the economy/flat
    tier is carried forward untouched.

Controls on the run itself
    R1  routing identity: the ``flat`` rows of this run must reproduce the
        committed ``numbers.json`` ``band_sweep_case`` rows to < 1e-12 relative
        (same code path, same mesh, deterministic) -- otherwise the new numbers
        are not comparable with the gated table.
    R2  geometry: n_panels == 1280, area deficit and h_mean reported.

Reproduce (cluster; ~1.2 h)
    python3 scripts/havelock_measure_case_mesh.py --outdir /tmp/out --out numbers.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "benchmarks" / "verified" / "casepool_bem_submerged_sphere_havelock"))

import measure_correction_study as mcs  # noqa: E402  (same code path as the merged study)

COEFFICIENTS = mcs.COEFFICIENTS
MEASURES = ("flat", "patch", "uniform")
CASE_BAND_FULL = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
CASE_BAND_DECLARED = mcs.CASE_BAND          # (0.1, 1.0, 3.0)
COMMITTED_NUMBERS = (
    _REPO / "docs" / "case_bem_submerged_sphere_havelock" / "numbers.json"
)


def _committed_flat_rows() -> dict[float, dict[str, float]]:
    """Committed case-mesh flat rows, keyed by ka (routing-identity control R1)."""
    if not COMMITTED_NUMBERS.exists():
        return {}
    data = json.loads(COMMITTED_NUMBERS.read_text())
    out: dict[float, dict[str, float]] = {}
    for row in data.get("band_sweep_case", []):
        out[round(float(row["ka"]), 6)] = {
            c: float(row[f"{c}_bem"]) for c in COEFFICIENTS
        }
    return out


def _deficit(subdivision: int, submergence: float, radius: float = 1.0) -> float:
    """Polyhedral (chord) area deficit, matching numbers.json area_deficit_diagnostic."""
    mesh = mcs.submerged_sphere_mesh(subdivision, submergence=submergence, radius=radius)
    return float(1.0 - mesh.surface_area / (4.0 * np.pi * radius * radius))


def _routing_identity(rows: Sequence[dict[str, Any]], committed: dict[float, dict[str, float]]) -> dict[str, Any]:
    """R1: flat rows of this run vs the committed gated table."""
    if not committed:
        return {"status": "not_run", "reason": "committed numbers.json unavailable"}
    worst = 0.0
    per_ka = []
    for row in rows:
        if row["measure"] != "flat":
            continue
        key = round(float(row["ka"]), 6)
        if key not in committed:
            continue
        diffs = {
            c: abs(float(row[c]) - committed[key][c]) / abs(committed[key][c])
            for c in COEFFICIENTS
        }
        worst = max(worst, max(diffs.values()))
        per_ka.append({"ka": key, "max_rel_diff": max(diffs.values())})
    return {
        "status": "PASS" if worst < 1e-12 else "FAIL",
        "worst_rel_diff": worst,
        "tolerance": 1e-12,
        "per_ka": per_ka,
        "note": "same Tier/solve path as numbers.json band_sweep_case (subdivision 3, flat measure)",
    }


def _decomposition(rows: Sequence[dict[str, Any]], subdivision: int = 3,
                   submergence: float = 2.0) -> dict[str, Any]:
    """P1/P2: how much of the flat-measure error does the exact measure remove?"""
    by_ka: dict[float, dict[str, dict[str, float]]] = {}
    for row in rows:
        by_ka.setdefault(round(float(row["ka"]), 6), {})[row["measure"]] = {
            c: float(row[f"{c}_rel_error"]) for c in COEFFICIENTS
        }
    deficit = _deficit(subdivision, submergence)
    cells = []
    for ka in sorted(by_ka):
        cell = by_ka[ka]
        for c in COEFFICIENTS:
            e_flat = cell["flat"][c]
            e_patch = cell["patch"][c]
            e_unif = cell["uniform"][c]
            ratio = (e_flat / e_patch) if e_patch > 0.0 else float("inf")
            cells.append(
                {
                    "ka": ka,
                    "quantity": c,
                    "err_flat": e_flat,
                    "err_patch": e_patch,
                    "err_uniform": e_unif,
                    "reduction_abs": e_flat - e_patch,
                    "reduction_frac_of_deficit": (e_flat - e_patch) / deficit,
                    "flat_over_patch": ratio,
                    "patch_beats_uniform": bool(e_patch <= e_unif),
                }
            )
    max_flat = max(c["err_flat"] for c in cells)
    max_patch = max(c["err_patch"] for c in cells)
    damping = [c for c in cells if c["quantity"] in ("b33", "b11")]
    mass = [c for c in cells if c["quantity"] in ("a33", "a11")]
    p1 = {
        "claim": "band-max rel error drops by at least a factor 2 under the patch measure",
        "max_flat": max_flat,
        "max_patch": max_patch,
        "factor": (max_flat / max_patch) if max_patch > 0 else float("inf"),
        "holds": bool(max_patch * 2.0 <= max_flat),
    }
    p2_rows = [
        {
            "ka": c["ka"],
            "quantity": c["quantity"],
            "reduction_frac_of_deficit": c["reduction_frac_of_deficit"],
            "holds": bool(c["reduction_frac_of_deficit"] > 0.5),
        }
        for c in damping
    ]
    p2 = {
        "claim": "coefficient-wise reduction under the patch measure exceeds half the area deficit (damping items)",
        "deficit": deficit,
        "n_items": len(p2_rows),
        "n_holding": sum(1 for r in p2_rows if r["holds"]),
        "min_frac_of_deficit": min(r["reduction_frac_of_deficit"] for r in p2_rows),
        "median_frac_of_deficit": float(
            np.median([r["reduction_frac_of_deficit"] for r in p2_rows])
        ),
        "holds": bool(all(r["holds"] for r in p2_rows)),
        "rows": p2_rows,
    }
    p3 = {
        "claim": "(flat - patch) removal is not reproduced by the uniform control on damping items",
        "n_damping_cells": len(damping),
        "n_patch_beats_uniform": sum(1 for c in damping if c["patch_beats_uniform"]),
        "flat_minus_patch_min": min(c["reduction_abs"] for c in damping),
        "flat_minus_uniform_min": min(
            c["err_flat"] - c["err_uniform"] for c in damping
        ),
        "holds": bool(
            sum(1 for c in damping if c["patch_beats_uniform"]) > len(damping) / 2
        ),
    }
    return {
        "area_deficit_s3": deficit,
        "prediction_P1": p1,
        "prediction_P2": p2,
        "prediction_P3": p3,
        "cells": cells,
        "max_flat": max_flat,
        "max_patch": max_patch,
        "max_uniform": max(c["err_uniform"] for c in cells),
    }


def _shard_band(band: Sequence[float], shard: int, shards: int) -> tuple[float, ...]:
    """Contiguous slice of ``band`` for sharded cluster execution (no ka dropped)."""
    if shards < 1 or not (0 <= shard < shards):
        raise ValueError("need 0 <= shard < shards")
    n = max(1, int(np.ceil(len(band) / shards)))
    return tuple(band[shard * n : (shard + 1) * n])


def build(outdir_rows: Path, *, smoke: bool = False, shard: int = 0,
          shards: int = 1) -> dict[str, Any]:
    band_full = (0.1,) if smoke else _shard_band(CASE_BAND_FULL, shard, shards)
    band_t4 = (1.0,)
    band_t5 = (0.01,) if smoke else tuple(mcs.LOW_KA)
    started = time.perf_counter()
    study: dict[str, Any] = {
        "case_code": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "pool_code": "BEM-SUBMERGED-SPHERE-HAVELOCK",
        "round": "round6",
        "driver": "scripts/havelock_measure_case_mesh.py",
        "study_module": "benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py",
        "declared_case_band": list(CASE_BAND_DECLARED),
        "bands_actually_run": {"T3": list(band_full), "T4": list(band_t4), "T5": list(band_t5)},
        "tolerance": mcs.GATE,
        "shard": {"index": shard, "count": shards},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "tiers": {},
        "verdicts": {},
        "log": [],
    }

    def _record(label: str, band: Sequence[float], tier: mcs.Tier) -> None:
        t0 = time.perf_counter()
        rows = mcs._band(tier, band)
        study["tiers"][label] = {"summary": tier.summary, "rows": rows}
        study["verdicts"][label] = mcs._verdict(rows, label)
        dt = time.perf_counter() - t0
        study["log"].append(f"{label}: {dt:.1f} s, {len(rows)} rows")
        print(label, json.dumps(study["verdicts"][label], default=float), flush=True)

    # --- T3: the case mesh, i.e. the mesh the pool gate actually uses ----------
    sub_t3 = 2 if smoke else 3
    t3 = mcs.Tier(sub_t3, 2.0)
    _record("T3:case_s3_band_flat_patch_uniform", band_full, t3)
    study["decomposition"] = _decomposition(
        study["tiers"]["T3:case_s3_band_flat_patch_uniform"]["rows"], sub_t3, 2.0
    )
    study["routing_identity_R1"] = _routing_identity(
        study["tiers"]["T3:case_s3_band_flat_patch_uniform"]["rows"], _committed_flat_rows()
    )
    study["geometry"] = {
        "subdivision": sub_t3,
        "n_panels": t3.mesh.panel_count,
        "h_mean": t3.mesh.h_mean,
        "polyhedral_area": t3.mesh.surface_area,
        "sphere_area": 4.0 * np.pi,
        "area_deficit": study["decomposition"]["area_deficit_s3"],
        "patch_area_over_sphere_area": float(t3.patch.sum() / (4.0 * np.pi)),
        "uniform_scale_k": float(t3.uniform.sum() / t3.mesh.surface_area),
    }
    # --- T4: submergence off-design (not used to derive the fix) --------------
    if shard == 0:
        _record("T4:economy_s2_fa1.5", band_t4, mcs.Tier(2, 1.5))
        _record("T4:economy_s2_fa3.0", band_t4, mcs.Tier(2, 3.0))
        # --- T5: low-frequency limit -----------------------------------------
        low = mcs.Tier(2, 2.0)
        _record("T5:economy_s2_low_ka", band_t5, low)
        study["low_frequency_limit"] = [
            {
                "ka": ka,
                "b33_patch": low.at(ka)["patch"]["got"]["b33"],
                "b33_ref": low.at(ka)["patch"]["ref"]["b33"],
                "b33_flat": low.at(ka)["flat"]["got"]["b33"],
            }
            for ka in band_t5
        ]
    study["runtime_seconds"] = time.perf_counter() - started
    return study


def write(study: dict[str, Any], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    numbers = outdir / "numbers_round6.json"
    numbers.write_text(json.dumps(study, indent=2, default=float) + "\n")
    written.append(numbers)
    verdicts = outdir / "verdicts_round6.json"
    verdicts.write_text(json.dumps(
        {
            "case_code": study["case_code"],
            "tolerance": study["tolerance"],
            "verdicts": study["verdicts"],
            "predictions": {
                k: {kk: vv for kk, vv in study["decomposition"][k].items() if kk != "rows"}
                for k in ("prediction_P1", "prediction_P2", "prediction_P3")
            },
            "routing_identity_R1": {
                k: v for k, v in study["routing_identity_R1"].items() if k != "per_ka"
            },
            "not_run": study.get("not_run", []),
            "source": "scripts/havelock_measure_case_mesh.py",
        },
        indent=2, default=float,
    ) + "\n")
    written.append(verdicts)
    return written


def merge(paths: Sequence[Path], outdir: Path) -> dict[str, Any]:
    """Merge sharded runs: concatenate T3 rows, re-derive verdicts/analysis."""
    shards = [json.loads(Path(p).read_text()) for p in paths]
    shards.sort(key=lambda d: d["shard"]["index"])
    base = shards[0]
    rows: list[dict[str, Any]] = []
    for doc in shards:
        rows.extend(doc["tiers"]["T3:case_s3_band_flat_patch_uniform"]["rows"])
    rows.sort(key=lambda r: (float(r["ka"]), str(r["measure"])))
    kas = sorted({round(float(r["ka"]), 6) for r in rows})
    study: dict[str, Any] = {
        "case_code": base["case_code"],
        "pool_code": base["pool_code"],
        "round": "round6",
        "driver": base["driver"],
        "study_module": base["study_module"],
        "declared_case_band": base["declared_case_band"],
        "bands_actually_run": {
            "T3": kas,
            "T4": [1.0],
            "T5": (shards[0]["bands_actually_run"]["T5"]),
        },
        "tolerance": base["tolerance"],
        "environment": base["environment"],
        "shards": [d["shard"] for d in shards],
        "tiers": {"T3:case_s3_band_flat_patch_uniform": {
            "summary": base["tiers"]["T3:case_s3_band_flat_patch_uniform"]["summary"],
            "rows": rows,
        }},
        "verdicts": {"T3:case_s3_band_flat_patch_uniform": mcs._verdict(
            rows, "T3:case_s3_band_flat_patch_uniform")},
        "log": [line for d in shards for line in d["log"]],
        "runtime_seconds": sum(float(d["runtime_seconds"]) for d in shards),
    }
    for doc in shards[1:]:
        study["tiers"].update(doc["tiers"])
        study["verdicts"].update(doc["verdicts"])
    if "low_frequency_limit" in shards[0]:
        study["low_frequency_limit"] = shards[0]["low_frequency_limit"]
    study["decomposition"] = _decomposition(rows, 3, 2.0)
    study["routing_identity_R1"] = _routing_identity(rows, _committed_flat_rows())
    study["geometry"] = base["geometry"]
    missing = sorted(set(CASE_BAND_FULL) - set(kas))
    study["not_run"] = (
        [f"ka={ka} (shard missing)" for ka in missing] if missing else []
    )
    for path in write(study, outdir):
        print(f"  {path}")
    return study


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=Path, default=Path("/tmp/havelock_measure_round6"))
    parser.add_argument("--smoke", action="store_true", help="1 ka per tier: logic check only, no verdict claim")
    parser.add_argument("--shard", type=int, default=0, help="0-based shard index for the T3 band")
    parser.add_argument("--shards", type=int, default=1, help="total shards (1 = single job)")
    parser.add_argument("--merge", nargs="+", type=Path, default=None,
                        help="merge these shard artefacts into outdir instead of computing")
    args = parser.parse_args(argv)
    if args.merge:
        merge(args.merge, args.outdir)
        return 0
    study = build(args.outdir, smoke=args.smoke, shard=args.shard, shards=args.shards)
    tag = "" if args.shards == 1 else f"_shard{args.shard}"
    target = args.outdir / f"numbers_round6{tag}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(study, indent=2, default=float) + "\n")
    print(f"  {target}")
    if args.shards == 1:
        for path in write(study, args.outdir):
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())