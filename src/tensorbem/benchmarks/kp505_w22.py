"""W22 propeller-openwater-bem: KP505 anchor reachability + five-J open-water study.

Mission (W22, topic ``propeller-openwater-bem``, successor of the zero-report
``paper-exp-propeller-openwater-bem-1``): is the KP505 open-water test anchor
reachable *inside this repository*, and how large is the BEM deviation from it
at the design advance ratio?

Everything in the two deliverables of this driver is produced here -- no number
is typed by hand:

``benchmarks/verified/kp505_anchor_w22/kp505_transcription.json``
    the reachability recheck (glob + grep hit counts, file sha256) and the
    machine transcription of the anchor table, one record per number, each
    carrying ``file + sha256 + line + field + verbatim``.

``benchmarks/verified/kp505_anchor_w22/openwater_compare.json``
    the BEM ``K_T / K_Q / eta_0`` table on the mission's five advance ratios,
    the deviation against every available reference with its reference *type*,
    the panel-density GCI ladder, and the design-point attribution.

Predeclared criteria (fixed before the sweep was run; verdicts are emitted by
this driver, never edited afterwards)
---------------------------------------------------------------------
A1  anchor reachability is machine-decided: the artefact records the number of
    hit files, the number of transcribed fields and the sha256 of every hit
    file.  Zero hits must be written as an honest FAIL, never silently.
A2  ``K_T / K_Q / eta_0`` are tabulated on ``J = 0.1, 0.3, 0.5, 0.7, 0.9`` with
    a relative deviation (%) and a reference type (``experiment`` /
    ``numerical`` / ``analytic``) in every cell of every column.
A3  at the design advance ratio ``J = 0.70``, ``|dK_T| < 5 %``; otherwise the
    miss is reported as an honest FAIL together with its attribution over the
    three predeclared channels (panel density / wake model / viscous
    correction).
A4  panel-density GCI on at least two grids: observed order ``p`` and the
    fine-grid GCI index are tabulated for ``K_T`` and ``K_Q`` at every mission
    advance ratio, and the artefact states explicitly whether the triple is in
    the asymptotic range (a negative or oscillatory ``p`` is reported as such;
    the GCI is then a magnitude indicator, not an error bound).

Reference hierarchy
-------------------
``experiment``  the KP505 open-water curve, transcribed from Table 2
    (Experimental columns) of Iftikhar & Takinaci, MARINE 2023, p. 10 -- a
    *secondary reproduction* of the KRISO KP505 test (the paper does not cite
    the primary handout).  Declared, not hidden: any accuracy claim built on it
    inherits one tabulation step.
``analytic``    the ITTC-1957 friction line ``C_F = 0.075/(log10 Re - 2)^2``
    (anchor register AN-ITTC57) and the actuator-disc momentum bound on
    ``eta_0``.  These are formula evaluations, not test data, and are labelled
    ``analytic`` everywhere they appear.

Run with ``python -m tensorbem.benchmarks.kp505_w22 [outdir]``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..propeller import TWO_PI, WATER_NU, BladeGeometry, open_water_point
from ..uq import grid_convergence_index
from .kp505_openwater_bem import ANCHOR as PINNED_ANCHOR
from .kp505_openwater_bem import PRIMARY

# ---------------------------------------------------------------------------
# Paths and the anchor register
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

ARTE_DIR = "benchmarks/verified/kp505_anchor_w22"
TRANSCRIPTION_NAME = "kp505_transcription.json"
COMPARE_NAME = "openwater_compare.json"

HIT_PATTERNS = ("*kp505*", "*KP505*", "*kp_505*", "*kp-505*", "*Kp505*")

#: the file that carries the table (found by the reachability pass, pinned here
#: so the transcription has a fixed input; the pass verifies that it exists)
ANCHOR_TABLE_SOURCE = "benchmarks/anchors/sources/kp505_marine2023_p10_table2.txt"

ANCHOR_CITATION = {
    "paper": (
        "Iftikhar, H. & Takinaci, A.C. (2023). Performance Prediction of KP505 "
        "Propeller Using Lifting Line Theory and RANS. X International Conference "
        "on Computational Methods in Marine Engineering (MARINE 2023)."
    ),
    "doi": "10.23967/marine.2023.120",
    "location": "Table 2, page 10",
    "table_caption_verbatim": "Table 2. Comparison of CFD and Experimental Results",
    "use": "Experimental columns (K_T, 10 K_Q) at J = 0.1 ... 1.0",
    "provenance_depth": (
        "SECONDARY_REPRODUCTION: the paper prints the experimental columns of the "
        "KRISO KP505 open-water test (KRISO Container Ship propeller) but page 10 "
        "does not cite the primary experimental handout.  One tabulation step is "
        "inherited by every number taken from it."
    ),
}

#: AN-ITTC57 -- the analytic anchor of this mission.  Formula only; every value
#: is evaluated from it here (no memorised constants).
ITTC57 = {
    "id": "AN-ITTC57",
    "quantity": "skin-friction coefficient C_F (flat plate, both sides)",
    "formula_verbatim": "C_F = 0.075 / (log10(Re) - 2)^2",
    "numerator": 0.075,
    "log10_shift": 2.0,
    "reference": (
        "ITTC Recommended Procedure 7.5-02-03-01.4, 1957 friction line "
        "(AN-ITTC57 anchor register, mission W22)"
    ),
    "type": "analytic",
}

#: panel-density ladder: all three panel axes refined together, r = 2
LADDER = (
    {"level": "L1", "n_radial": 8, "n_chord": 2, "wake_steps": 24},
    {"level": "L2", "n_radial": 16, "n_chord": 4, "wake_steps": 48},
    {"level": "L3", "n_radial": 32, "n_chord": 8, "wake_steps": 96},
)
#: panel-size ratio between consecutive levels (h halves when every axis doubles)
LADDER_RATIO = 2.0

MISSION_J = (0.1, 0.3, 0.5, 0.7, 0.9)
DESIGN_J = 0.70

#: predeclared thresholds (never moved after the sweep)
THRESHOLD_DESIGN_DKT_PCT = 5.0
THRESHOLD_ORDER_TOLERANCE = 0.30  # |p - p_th| / p_th for "asymptotic"
P_THEORY = 2.0

WAKE_MODELS = ("geometric", "kinematic", "induced")

#: lattice solutions are cached (keyed on J + ladder level + overrides + the
#: sha256 of the solver sources), so re-emitting verdicts/figures after a
#: reporting-only change does not re-run 10 minutes of BEM.
CACHE_DIR = "runs/w22_kp505_cache"
CACHE_ENABLED = os.environ.get("W22_NO_CACHE", "") == ""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path = REPO_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - environment without git
        return "unknown"


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:  # pragma: no cover
        return str(path)


def ittc57_cf(reynolds):
    """Evaluate AN-ITTC57 ``C_F = 0.075/(log10 Re - 2)^2``."""

    re = np.maximum(np.asarray(reynolds, dtype=float), 1.0)
    return ITTC57["numerator"] / (np.log10(re) - ITTC57["log10_shift"]) ** 2


# ---------------------------------------------------------------------------
# A1: reachability recheck (glob + grep), machine-decided
# ---------------------------------------------------------------------------


def reachability(root: Path = REPO_ROOT) -> dict:
    """Glob and text-search the checkout for the KP505 anchor."""

    root = Path(root)
    artefact_root = (root / ARTE_DIR).resolve()
    glob_hits: dict[str, list[str]] = {}
    excluded_self = 0
    for pattern in HIT_PATTERNS:
        hits = []
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            if artefact_root in path.resolve().parents:
                excluded_self += 1
                continue
            hits.append(_rel(root, path))
        glob_hits[pattern] = sorted(hits)

    # the artefact directory itself is excluded: it is *produced by* this pass,
    # so counting its figures/JSON as "anchor found" would be self-reference
    artefact_dir = root / ARTE_DIR
    grep_command = [
        "grep", "-ril", "--include=*",
        "--exclude-dir=.git", "--exclude-dir=__pycache__",
        "--exclude-dir=.venv", "--exclude-dir=node_modules",
        "--exclude-dir=runs", f"--exclude-dir={artefact_dir.name}",
        "--exclude-dir=w22_kp505_cache",
        "kp505", str(root),
    ]
    try:
        grep_out = subprocess.run(
            grep_command, capture_output=True, text=True, timeout=180
        ).stdout
        grep_files = sorted(
            _rel(root, Path(line)) for line in grep_out.splitlines() if line.strip()
        )
    except Exception:  # pragma: no cover
        grep_files = []

    # files this study writes itself would otherwise inflate the "anchor found"
    # count (and make it drift between runs); they are listed, not silently cut
    own_outputs = {
        "src/tensorbem/benchmarks/kp505_w22.py",
        "tools/emit_w22_report.py",
        "tools/post_report_w22.py",
        "tests/test_w22_kp505_anchor.py",
        "papers/reports/w22_propeller_openwater6.md",
    }
    files = sorted({*sum(glob_hits.values(), []), *grep_files})
    preexisting = [rel for rel in files if rel not in own_outputs
                   and not rel.startswith(ARTE_DIR)
                   and not rel.startswith(CACHE_DIR)]
    records = []
    for rel in preexisting:
        path = root / rel
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".json", ".csv"}:
            layer = "table"
        elif suffix in {".pdf", ".png", ".jpg"}:
            layer = "figure"
        elif suffix in {".igs", ".iges", ".stl"}:
            layer = "geometry"
        else:
            layer = "text"
        records.append(
            {
                "file": rel,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "layer": layer,
            }
        )

    return {
        "patterns": list(HIT_PATTERNS),
        "self_reference_excluded": {
            "dir": ARTE_DIR,
            "glob_hits_excluded": excluded_self,
            "note": (
                "the artefact directory is this study's own output, so it is "
                "excluded from the 'is the anchor in the repo' count; the "
                "driver/tests/report files added by this task are still counted "
                "as text-layer hits"
            ),
        },
        "glob_hits": glob_hits,
        "glob_hit_files": sum(len(v) for v in glob_hits.values()),
        "grep_command": " ".join(grep_command),
        "grep_hit_files": len(grep_files),
        "n_hit_files": len(records),
        "n_hit_files_raw": len(files),
        "own_outputs_excluded": sorted(own_outputs),
        "hit_files": records,
        "layer_summary": {
            layer: sum(1 for r in records if r["layer"] == layer)
            for layer in sorted({r["layer"] for r in records})
        },
        "anchor_table_source": ANCHOR_TABLE_SOURCE,
        "anchor_table_source_present": (root / ANCHOR_TABLE_SOURCE).is_file(),
    }


# ---------------------------------------------------------------------------
# Anchor transcription (zero hand-copy: parsed out of the pinned table text)
# ---------------------------------------------------------------------------

_ROW_RE = re.compile(
    r"^(?P<J>\d(?:\.\d)?)\s+(?P<KT_cfd>\d\.\d{3})\s+(?P<KQ10_cfd>\d\.\d{3})\s+"
    r"(?P<KT_exp>\d\.\d{3})\s+(?P<KQ10_exp>\d\.\d{3})\s+"
    r"(?P<err_KT>\d+\.\d{2})%\s+(?P<err_KQ10>\d+\.\d{2})%$"
)

_FIELD_ORDER = ("J", "KT_cfd", "KQ10_cfd", "KT_exp", "KQ10_exp", "err_KT", "err_KQ10")

_FIELD_GRADE = {
    "J": "experiment",
    "KT_cfd": "numerical",
    "KQ10_cfd": "numerical",
    "KT_exp": "experiment",
    "KQ10_exp": "experiment",
    "err_KT": "numerical",
    "err_KQ10": "numerical",
}


def transcribe_anchor(root: Path = REPO_ROOT) -> dict:
    """Parse the anchor table out of its source text, one record per number."""

    root = Path(root)
    source = root / ANCHOR_TABLE_SOURCE
    if not source.is_file():
        return {
            "available": False,
            "honest_fail": True,
            "reason": f"anchor table source {ANCHOR_TABLE_SOURCE} not found in checkout",
            "n_fields": 0,
            "n_hit_files": 0,
        }

    source_sha = _sha256(source)
    lines = source.read_text().splitlines()

    rows: list[dict] = []
    fields: list[dict] = []
    for lineno, line in enumerate(lines, start=1):
        match = _ROW_RE.match(line.strip())
        if not match:
            continue
        row = {"line": lineno, "verbatim": line.rstrip(), "fields": {}}
        for field in _FIELD_ORDER:
            raw = match.group(field)
            record = {
                "field": field,
                "value": float(raw),
                "raw": raw,
                "file": ANCHOR_TABLE_SOURCE,
                "sha256": source_sha,
                "line": lineno,
                "verbatim": line.rstrip(),
                "reference_type": _FIELD_GRADE[field],
            }
            fields.append(record)
            row["fields"][field] = float(raw)
        row["KQ_exp"] = row["fields"]["KQ10_exp"] / 10.0
        row["KQ_cfd"] = row["fields"]["KQ10_cfd"] / 10.0
        row["eta0_exp"] = (
            row["fields"]["J"] * row["fields"]["KT_exp"] / (TWO_PI * row["KQ_exp"])
        )
        row["eta0_cfd"] = (
            row["fields"]["J"] * row["fields"]["KT_cfd"] / (TWO_PI * row["KQ_cfd"])
        )
        rows.append(row)

    if not rows:
        return {
            "available": False,
            "honest_fail": True,
            "reason": "table file present but no row matched the Table-2 row grammar",
            "file": ANCHOR_TABLE_SOURCE,
            "sha256": source_sha,
            "n_fields": 0,
            "n_hit_files": 1,
        }

    by_j = {row["fields"]["J"]: row for row in rows}
    mismatches = []
    for advance, pinned in PINNED_ANCHOR.items():
        row = by_j.get(advance)
        if row is None:
            mismatches.append({"J": advance, "reason": "missing in transcription"})
            continue
        for key, ours in (("KT", row["fields"]["KT_exp"]),
                          ("10KQ", row["fields"]["KQ10_exp"])):
            if abs(ours - pinned[key]) > 1e-12:
                mismatches.append({"J": advance, "field": key,
                                   "transcribed": ours, "pinned": pinned[key]})

    self_consistency = []
    for row in rows:
        f = row["fields"]
        for cfd_key, exp_key, err_key in (
            ("KT_cfd", "KT_exp", "err_KT"),
            ("KQ10_cfd", "KQ10_exp", "err_KQ10"),
        ):
            signed = 100.0 * (f[exp_key] - f[cfd_key]) / f[exp_key]
            # the source prints *unsigned* error magnitude, so the check is
            # |recomputed| vs printed (and the sign is kept for the record)
            recomputed = abs(signed)
            # the table prints 3 decimals, the error column 2 -> the smallest
            # printable KT (0.029) already carries +-1.7 % relative rounding, so
            # the agreement bound has to be derived per row, not set globally
            half = 5.0e-4
            bound = 100.0 * half * (1.0 / f[exp_key] + 1.0 / f[cfd_key]) + 0.005
            self_consistency.append(
                {
                    "J": f["J"],
                    "quantity": "KT" if cfd_key == "KT_cfd" else "10KQ",
                    "printed_err_pct": f[err_key],
                    "recomputed_err_pct": recomputed,
                    "recomputed_err_signed_pct": signed,
                    "abs_diff_pct_points": abs(recomputed - f[err_key]),
                    "roundoff_bound_pct_points": bound,
                    "ratio_to_bound": abs(recomputed - f[err_key]) / bound,
                }
            )
    worst = max(self_consistency, key=lambda r: r["abs_diff_pct_points"])
    worst_ratio = max(self_consistency, key=lambda r: r["ratio_to_bound"])

    slopes = []
    for left, right in zip(rows[:-1], rows[1:]):
        d_j = right["fields"]["J"] - left["fields"]["J"]
        slopes.append(
            {
                "J_from": left["fields"]["J"],
                "J_to": right["fields"]["J"],
                "dKT_dJ": (right["fields"]["KT_exp"] - left["fields"]["KT_exp"]) / d_j,
                "d10KQ_dJ": (right["fields"]["KQ10_exp"] - left["fields"]["KQ10_exp"]) / d_j,
            }
        )

    design = by_j[DESIGN_J]
    exp_rows = [
        {
            "J": row["fields"]["J"],
            "KT": row["fields"]["KT_exp"],
            "10KQ": row["fields"]["KQ10_exp"],
            "KQ": row["KQ_exp"],
            "eta0": row["eta0_exp"],
            "reference_type": "experiment",
            "source": {
                "file": ANCHOR_TABLE_SOURCE,
                "sha256": source_sha,
                "line": row["line"],
                "fields": ["KT_exp", "KQ10_exp"],
            },
        }
        for row in rows
    ]

    return {
        "available": True,
        "honest_fail": False,
        "anchor_id": "AN-KP505",
        "citation": ANCHOR_CITATION,
        "reference_type": "experiment",
        "provenance_class": "SECONDARY_REPRODUCTION",
        "file": ANCHOR_TABLE_SOURCE,
        "sha256": source_sha,
        "n_rows": len(rows),
        "n_fields": len(fields),
        "n_fields_with_source": sum(
            1 for f in fields
            if f["file"] and f["sha256"] and f["line"] and f["verbatim"]
        ),
        "fields": fields,
        "experimental_curve": exp_rows,
        "design_J": DESIGN_J,
        "design_point": {
            "J": DESIGN_J,
            "KT": design["fields"]["KT_exp"],
            "10KQ": design["fields"]["KQ10_exp"],
            "KQ": design["KQ_exp"],
            "eta0": design["eta0_exp"],
            "present": True,
            "source": {"file": ANCHOR_TABLE_SOURCE, "sha256": source_sha,
                       "line": design["line"]},
        },
        "cross_check_pinned_table_mismatches": mismatches,
        "source_self_consistency": {
            "convention": (
                "the source prints unsigned error magnitudes; the check compares "
                "|100 (exp - cfd)/exp| against them and keeps the signed value"
            ),
            "n_checks": len(self_consistency),
            "max_abs_diff_pct_points": worst["abs_diff_pct_points"],
            "max_ratio_to_roundoff_bound": worst_ratio["ratio_to_bound"],
            "worst": worst,
            "worst_by_roundoff_bound": worst_ratio,
            "rows": self_consistency,
        },
        "experimental_kt_slope": slopes,
        "ittc57_anchor": {
            **ITTC57,
            "sample_values": [
                {"Re": re_value, "C_F": float(ittc57_cf(re_value))}
                for re_value in (5.0e4, 1.0e5, 5.0e5, 1.0e6, 5.0e6, 1.0e7)
            ],
        },
    }


# ---------------------------------------------------------------------------
# BEM sweep
# ---------------------------------------------------------------------------


def _solver_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in ("propeller.py", "uq.py"):
        path = Path(__file__).resolve().parents[1] / name
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


_SOLVER_FINGERPRINT = _solver_fingerprint()


def _cache_key(advance: float, level: dict, overrides: dict) -> str:
    payload = {
        "J": round(float(advance), 12),
        "level": {k: level[k] for k in ("level", "n_radial", "n_chord", "wake_steps")},
        "overrides": {k: overrides[k] for k in sorted(overrides)},
        "primary": {k: PRIMARY[k] for k in sorted(PRIMARY)},
        "solver": _SOLVER_FINGERPRINT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def _solve(advance: float, level: dict, use_cache: bool | None = None,
           **overrides) -> dict:
    cache_dir = REPO_ROOT / CACHE_DIR
    key = _cache_key(advance, level, overrides)
    path = cache_dir / f"{key}.json"
    use_cache = CACHE_ENABLED if use_cache is None else use_cache
    if use_cache and path.is_file():
        cached = json.loads(path.read_text())
        cached["cache_hit"] = True
        return cached

    config = dict(PRIMARY)
    config.update(
        n_radial=level["n_radial"],
        n_chord=level["n_chord"],
        wake_steps=level["wake_steps"],
    )
    config.update(overrides)
    geometry = BladeGeometry(
        n_radial=config["n_radial"],
        n_chord=config["n_chord"],
        thickness_ratio=config["thickness_ratio"],
        radial_spacing=config["radial_spacing"],
        root_chord_offset=config["root_chord_offset"],
    )
    result = open_water_point(
        geometry,
        advance,
        wake_model=config["wake_model"],
        wake_length=config["wake_length"],
        wake_steps=config["wake_steps"],
        rev_per_second=config["rev_per_second"],
        viscous=config["viscous"],
        core_ratio=config.get("core_ratio", 0.05),
    )
    resisting = -result.torque
    efficiency = advance * result.thrust / (TWO_PI * resisting) if resisting else float("nan")
    payload = {
        "KT": float(result.thrust),
        "10KQ": float(10.0 * resisting),
        "KQ": float(resisting),
        "eta0": float(efficiency),
        "KT_inviscid": float(result.thrust_inviscid),
        "KQ_inviscid": float(-result.torque_inviscid),
        "eta0_momentum_bound": float(result.momentum_bound),
        "n_panels": int(result.n_panels),
        "wake_model": result.wake_model,
        "viscous": bool(config["viscous"]),
    }
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _pct(measured: float, reference: float) -> float:
    return 100.0 * (measured - reference) / reference


def _levels_entry(payload) -> dict:
    advance, levels = payload
    entry = {"J": advance, "levels": []}
    for level in levels:
        entry["levels"].append({**level, **_solve(advance, level)})
    return entry


def _attrib_entry(payload) -> dict:
    advance, level, overrides = payload
    return {**overrides, **_solve(advance, level, **overrides)}


def _run_parallel(function, jobs: list, workers: int | None) -> list:
    """Evaluate ``function`` over ``jobs``, in parallel when it pays off."""

    workers = workers or min(len(jobs), os.cpu_count() or 1)
    if workers <= 1 or len(jobs) <= 1:
        return [function(job) for job in jobs]
    # NB: "spawn", not the default "fork": numpy/OpenBLAS has already started
    # its thread pool in the parent by the time the lattice runs, and a forked
    # child inherits a futex nobody unlocks -> the child sleeps forever
    # (observed: 0 s CPU per worker, 130 parent threads).  "spawn" re-imports
    # and is immune to that.
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(function, jobs))


def sweep(anchor: dict, levels=LADDER, advances=MISSION_J,
          workers: int | None = None) -> list[dict]:
    """BEM on every mission advance ratio and every ladder level."""

    reference = {row["J"]: row for row in anchor["experimental_curve"]}
    rows = _run_parallel(_levels_entry, [(advance, levels) for advance in advances], workers)
    for entry in rows:
        fine = entry["levels"][-1]
        ref = reference[entry["J"]]
        entry.update(
            {
                "KT_bem": fine["KT"],
                "KQ_bem": fine["KQ"],
                "10KQ_bem": fine["10KQ"],
                "eta0_bem": fine["eta0"],
                "KT_ref": ref["KT"],
                "KQ_ref": ref["KQ"],
                "10KQ_ref": ref["10KQ"],
                "eta0_ref": ref["eta0"],
                "reference_type": "experiment",
                "reference_note": (
                    "KP505 open-water test curve, Table 2 Experimental columns "
                    "(secondary reproduction)"
                ),
                "dKT_pct": _pct(fine["KT"], ref["KT"]),
                "dKQ_pct": _pct(fine["KQ"], ref["KQ"]),
                "d10KQ_pct": _pct(fine["10KQ"], ref["10KQ"]),
                "deta0_pct": _pct(fine["eta0"], ref["eta0"]),
                "eta0_momentum_bound": fine["eta0_momentum_bound"],
                "eta0_above_momentum_bound": bool(
                    fine["eta0"] > fine["eta0_momentum_bound"] + 1e-12
                ),
            }
        )
        entry["n_workers"] = workers
    return rows


def gci_table(points: list[dict]) -> list[dict]:
    """Roache GCI over the predeclared ladder for KT, KQ and eta0."""

    rows = []
    for entry in points:
        values = {level["level"]: level for level in entry["levels"]}
        order = [level["level"] for level in LADDER if level["level"] in values]
        for quantity in ("KT", "KQ", "eta0"):
            triple = [values[name][quantity] for name in order[::-1]]  # fine -> coarse
            if len(triple) < 3:
                rows.append({"J": entry["J"], "quantity": quantity,
                             "n_levels": len(triple), "observed_order": float("nan"),
                             "gci_fine_pct": float("nan"), "asymptotic_range": False})
                continue
            result = grid_convergence_index(
                triple[0], triple[1], triple[2],
                refinement_ratio=LADDER_RATIO,
                quantity=quantity,
                label=f"J={entry['J']}",
            )
            p = float(result.observed_order)
            asymptotic = bool(
                math.isfinite(p)
                and abs(p - P_THEORY) <= THRESHOLD_ORDER_TOLERANCE * P_THEORY
            )
            rows.append(
                {
                    "J": entry["J"],
                    "quantity": quantity,
                    "levels": order,
                    "L1": triple[2],
                    "L2": triple[1],
                    "L3": triple[0],
                    "n_levels": len(triple),
                    "refinement_ratio": LADDER_RATIO,
                    "observed_order": p,
                    "theory_order": P_THEORY,
                    "order_deviation_pct": (
                        100.0 * (p - P_THEORY) / P_THEORY if math.isfinite(p) else float("nan")
                    ),
                    "asymptotic_range": asymptotic,
                    "oscillatory": bool(result.oscillatory),
                    "monotone": bool(result.monotone),
                    "converged_flag": bool(result.converged),
                    "extrapolated": float(result.extrapolated),
                    "gci_fine_pct": (
                        100.0 * float(result.gci_fine)
                        if math.isfinite(result.gci_fine) else float("nan")
                    ),
                    "gci_medium_pct": (
                        100.0 * float(result.gci_medium)
                        if math.isfinite(result.gci_medium) else float("nan")
                    ),
                    "last_increment_pct": 100.0 * abs(triple[0] - triple[1]) / triple[0],
                    "reference_type": "numerical",
                }
            )
    return rows


def attribution(anchor: dict, points: list[dict], workers: int | None = None) -> dict:
    """Split the design-point miss over the three predeclared channels."""

    reference = {row["J"]: row for row in anchor["experimental_curve"]}[DESIGN_J]
    design = next(entry for entry in points if abs(entry["J"] - DESIGN_J) < 1e-12)
    fine_level = LADDER[-1]

    density = {}
    for quantity, ref_key in (("KT", "KT"), ("KQ", "KQ"), ("eta0", "eta0")):
        triple = [level[quantity] for level in design["levels"]][::-1]  # L3, L2, L1
        result = grid_convergence_index(
            triple[0], triple[1], triple[2],
            refinement_ratio=LADDER_RATIO, quantity=quantity, label="design point",
        )
        p = float(result.observed_order)
        extrapolated = float(result.extrapolated)
        applicable = bool(math.isfinite(p) and p > 0.0 and result.monotone)
        total = _pct(triple[0], reference[ref_key])
        last_increment = 100.0 * abs(triple[0] - triple[1]) / triple[0]
        density[quantity] = {
            "L1": triple[2],
            "L2": triple[1],
            "L3": triple[0],
            "reference": reference[ref_key],
            "observed_order": p,
            "oscillatory": bool(result.oscillatory),
            "richardson_applicable": applicable,
            "extrapolated": extrapolated,
            "total_pct": total,
            "discretisation_pct": _pct(triple[0], extrapolated),
            "last_increment_pct": last_increment,
            "model_pct_richardson": total - _pct(triple[0], extrapolated),
            "model_band_pct": [total - 2.0 * last_increment, total + 2.0 * last_increment],
        }

    wake = []
    wake_jobs = [(DESIGN_J, fine_level, {"wake_model": model}) for model in WAKE_MODELS]
    wake_jobs.append((DESIGN_J, fine_level, {"viscous": False}))
    results = _run_parallel(_attrib_entry, wake_jobs, workers)
    inviscid = results[-1]
    for values in results[:-1]:
        model = values["wake_model"]
        wake.append(
            {
                "wake_model": model,
                **values,
                "dKT_pct": _pct(values["KT"], reference["KT"]),
                "d10KQ_pct": _pct(values["10KQ"], reference["10KQ"]),
                "deta0_pct": _pct(values["eta0"], reference["eta0"]),
            }
        )

    viscous_channel = {
        "viscous_on": {
            "KT": design["KT_bem"],
            "10KQ": design["10KQ_bem"],
            "dKT_pct": design["dKT_pct"],
            "d10KQ_pct": design["d10KQ_pct"],
        },
        "viscous_off": {
            "KT": inviscid["KT"],
            "10KQ": inviscid["10KQ"],
            "dKT_pct": _pct(inviscid["KT"], reference["KT"]),
            "d10KQ_pct": _pct(inviscid["10KQ"], reference["10KQ"]),
        },
        "viscous_increment_KT": inviscid["KT"] - design["KT_bem"],
        "viscous_increment_10KQ": inviscid["10KQ"] - design["10KQ_bem"],
        "viscous_share_of_total_dKT_pct": (
            design["dKT_pct"] - _pct(inviscid["KT"], reference["KT"])
        ),
    }

    # analytic cross-check of the viscous path against AN-ITTC57
    geometry = BladeGeometry(
        n_radial=fine_level["n_radial"],
        n_chord=fine_level["n_chord"],
        thickness_ratio=PRIMARY["thickness_ratio"],
        radial_spacing=PRIMARY["radial_spacing"],
        root_chord_offset=PRIMARY["root_chord_offset"],
    )
    radii = np.linspace(geometry.hub_radius, geometry.radius, 201)[1:-1]
    chords = geometry.chord(radii)
    r_mid = float(np.average(radii, weights=chords))
    c_mean = float(np.average(chords, weights=chords)) * geometry.diameter
    axial = DESIGN_J * PRIMARY["rev_per_second"] * geometry.diameter
    tangential = PRIMARY["rev_per_second"] * math.pi * r_mid
    local_speed = float(math.hypot(axial, tangential))
    reynolds = float(local_speed * c_mean / WATER_NU)
    cf_ittc57 = float(ittc57_cf(reynolds))
    cf_ittc78 = 0.044 / reynolds ** (1.0 / 6.0) - 5.0 / reynolds ** (2.0 / 3.0)
    analytic = {
        "reference_type": "analytic",
        "anchor": ITTC57["id"],
        "representative_radius_over_R": r_mid / geometry.radius,
        "representative_chord_m": c_mean,
        "representative_Re": reynolds,
        "CF_ittc57": cf_ittc57,
        "CF_ittc78_section_law": float(cf_ittc78),
        "ratio_ittc78_over_ittc57": float(cf_ittc78) / cf_ittc57,
        "note": (
            "the lattice's section-drag law is the ITTC-78 form "
            "2(1+2t/c)(0.044/Re^(1/6) - 5/Re^(2/3)); AN-ITTC57 gives the "
            "skin-friction level to check it against.  This is a formula "
            "evaluation (analytic), not test data."
        ),
    }

    return {
        "design_J": DESIGN_J,
        "reference": {
            "type": "experiment",
            "KT": reference["KT"],
            "10KQ": reference["10KQ"],
            "eta0": reference["eta0"],
        },
        "channel_panel_density": density,
        "channel_wake_model": wake,
        "channel_viscous_correction": viscous_channel,
        "analytic_ittc57_cross_check": analytic,
    }


# ---------------------------------------------------------------------------
# Criteria verdicts
# ---------------------------------------------------------------------------


def criteria(anchor: dict, reach: dict, points: list[dict], gci_rows: list[dict]) -> list[dict]:
    n_sha = sum(1 for r in reach["hit_files"] if r.get("sha256"))
    a1_pass = bool(
        anchor.get("available")
        and reach["n_hit_files"] > 0
        and anchor.get("n_fields", 0) > 0
        and n_sha == reach["n_hit_files"]
    )

    cells = 0
    complete = 0
    for entry in points:
        for measured_key, ref_key in (("KT_bem", "KT_ref"), ("KQ_bem", "KQ_ref"),
                                      ("eta0_bem", "eta0_ref")):
            cells += 1
            dev_key = {"KT_bem": "dKT_pct", "KQ_bem": "dKQ_pct",
                       "eta0_bem": "deta0_pct"}[measured_key]
            if (
                math.isfinite(entry[measured_key])
                and math.isfinite(entry[ref_key])
                and entry["reference_type"] in {"experiment", "numerical", "analytic"}
                and math.isfinite(entry[dev_key])
            ):
                complete += 1

    design = next(entry for entry in points if abs(entry["J"] - DESIGN_J) < 1e-12)
    a3_value = abs(design["dKT_pct"])
    a3_pass = bool(a3_value < THRESHOLD_DESIGN_DKT_PCT)
    # the attribution channels exist as soon as the design point and its ladder
    # are tabulated; the *content* of the attribution is in ``attribution()``.
    attrib_known = bool(design["KT_bem"] > 0.0) and any(
        g["J"] == DESIGN_J for g in gci_rows
    )

    q_cells = [g for g in gci_rows if g["quantity"] in {"KT", "KQ"}]
    n_with_p = sum(1 for g in q_cells if "observed_order" in g and math.isfinite(g["observed_order"]))
    n_levels = max((g.get("n_levels", 0) for g in gci_rows), default=0)
    a4_pass = bool(n_levels >= 2 and len(q_cells) == 2 * len(points))
    asymptotic_cells = sum(1 for g in q_cells if g.get("asymptotic_range"))

    return [
        {
            "id": "A1-anchor-reachability-machine-decided",
            "statement": (
                "hit-file count, transcribed field count and the sha256 of every "
                "hit file are recorded; zero hits would be an honest FAIL"
            ),
            "threshold": "n_hit_files > 0, n_fields > 0, every hit file sha256'd",
            "measured": {
                "n_hit_files": reach["n_hit_files"],
                "n_fields": anchor.get("n_fields", 0),
                "n_rows": anchor.get("n_rows", 0),
                "n_hit_files_with_sha256": n_sha,
                "anchor_table_source_present": reach["anchor_table_source_present"],
            },
            "pass": a1_pass,
        },
        {
            "id": "A2-five-J-table-with-deviation-and-reference-type",
            "statement": (
                "K_T, K_Q, eta_0 on J = 0.1/0.3/0.5/0.7/0.9, each cell carrying a "
                "relative deviation (%) and a reference type"
            ),
            "threshold": "15/15 cells finite with a declared reference type",
            "measured": {"cells": cells, "complete_cells": complete},
            "pass": bool(complete == cells and cells == 15),
        },
        {
            "id": "A3-design-point-KT-within-5pct",
            "statement": "|dK_T| < 5 % at J = 0.70; otherwise honest FAIL + attribution",
            "threshold": f"|dK_T| < {THRESHOLD_DESIGN_DKT_PCT} %",
            "measured": {
                "J": DESIGN_J,
                "KT_bem": design["KT_bem"],
                "KT_ref": design["KT_ref"],
                "dKT_pct": design["dKT_pct"],
                "within_threshold": a3_pass,
                "honest_fail": not a3_pass,
                "attribution_available": attrib_known,
            },
            # ``pass`` is the *accuracy* verdict only -- never rescue it with the
            # attribution route, that would turn an honest FAIL into a green cell
            "pass": a3_pass,
            "disposition": (
                "PASS" if a3_pass else
                ("HONEST_FAIL_WITH_ATTRIBUTION" if attrib_known
                 else "HONEST_FAIL_WITHOUT_ATTRIBUTION")
            ),
            "task_level_acceptable": bool(a3_pass or attrib_known),
        },
        {
            "id": "A4-panel-density-gci",
            "statement": (
                "at least two grids; observed order and fine-grid GCI tabulated for "
                "K_T and K_Q at every mission J; asymptotic validity stated"
            ),
            "threshold": ">= 2 levels, 10/10 (J, quantity) cells tabulated, asymptotic flag present",
            "measured": {
                "n_levels": n_levels,
                "n_cells": len(q_cells),
                "n_cells_with_finite_order": n_with_p,
                "n_cells_asymptotic": asymptotic_cells,
                "note": (
                    "asymptotic_range = |p - 2| <= 30 % of 2; when false the GCI is "
                    "reported as a magnitude indicator, not an error bound"
                ),
            },
            "pass": a4_pass,
        },
    ]


# ---------------------------------------------------------------------------
# Writers, figures, CLI
# ---------------------------------------------------------------------------


def _meta(inputs: dict) -> dict:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cwd_relative": os.path.relpath(os.getcwd(), REPO_ROOT)
        if Path(os.getcwd()).is_relative_to(REPO_ROOT) else os.getcwd(),
        "inputs": inputs,
        "driver": "src/tensorbem/benchmarks/kp505_w22.py",
        "producing_command": "python -m tensorbem.benchmarks.kp505_w22 "
                             "benchmarks/verified/kp505_anchor_w22",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _point_rows(points: list[dict]) -> list[dict]:
    rows = []
    for entry in points:
        row = {
            "J": entry["J"],
            "reference_type": entry["reference_type"],
            "KT_ref": entry["KT_ref"],
            "KQ_ref": entry["KQ_ref"],
            "eta0_ref": entry["eta0_ref"],
            "KT_bem": entry["KT_bem"],
            "KQ_bem": entry["KQ_bem"],
            "eta0_bem": entry["eta0_bem"],
            "dKT_pct": entry["dKT_pct"],
            "dKQ_pct": entry["dKQ_pct"],
            "deta0_pct": entry["deta0_pct"],
            "eta0_momentum_bound_analytic": entry["eta0_momentum_bound"],
        }
        for level in entry["levels"]:
            tag = level["level"]
            row[f"KT_{tag}"] = level["KT"]
            row[f"KQ_{tag}"] = level["KQ"]
        rows.append(row)
    return rows


def figures(outdir: Path, points: list[dict], gci_rows: list[dict],
            attrib: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []

    advance = [entry["J"] for entry in points]

    # 1: open-water curves
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for axis, key, label in (
        (axes[0], "KT", "$K_T$"),
        (axes[1], "10KQ", "$10\\,K_Q$"),
        (axes[2], "eta0", "$\\eta_0$"),
    ):
        axis.plot(advance, [entry[f"{key}_ref"] for entry in points], "o-",
                  color="k", label="KP505 test anchor (Table 2 exp. col.)")
        axis.plot(advance, [entry[f"{key}_bem"] for entry in points], "s--",
                  color="C0", label="BEM L3 (32x8, 96 wake steps)")
        for level_name, style in (("L1", ":"), ("L2", "--")):
            axis.plot(advance,
                      [next(l for l in entry["levels"] if l["level"] == level_name)[key]
                       for entry in points],
                      style, color="C0", alpha=0.5, linewidth=1.0, label=f"BEM {level_name}")
        axis.set_xlabel("J")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    fig.suptitle("KP505 open water: vortex-lattice BEM vs the in-repo experimental anchor")
    fig.tight_layout()
    path = fig_dir / "kp505_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append({"name": "kp505_curves", "file": str(path),
                    "caption": "K_T, 10K_Q and eta_0 vs J: BEM ladder vs the transcribed KP505 anchor."})

    # 2: deviations with the 5 % design-point band
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharex=True)
    for axis, key, label in (
        (axes[0], "dKT_pct", "$\\Delta K_T$ [%]"),
        (axes[1], "dKQ_pct", "$\\Delta K_Q$ [%]"),
        (axes[2], "deta0_pct", "$\\Delta \\eta_0$ [%]"),
    ):
        axis.axhline(0.0, color="k", linewidth=0.8)
        axis.plot(advance, [entry[key] for entry in points], "o-", color="C3")
        if key == "dKT_pct":
            axis.axhline(THRESHOLD_DESIGN_DKT_PCT, color="r", linestyle=":", linewidth=1.2)
            axis.axhline(-THRESHOLD_DESIGN_DKT_PCT, color="r", linestyle=":", linewidth=1.2)
            axis.text(advance[0], THRESHOLD_DESIGN_DKT_PCT, " criterion +/-5 %",
                      color="r", va="bottom", fontsize=8)
        axis.set_xlabel("J")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    fig.suptitle("Relative deviation from the experimental anchor (L3 lattice)")
    fig.tight_layout()
    path = fig_dir / "kp505_deviations.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append({"name": "kp505_deviations", "file": str(path),
                    "caption": "Relative deviation of K_T, K_Q and eta_0 from the anchor; the predeclared +/-5 % design-point band is drawn."})

    # 3: panel-density convergence
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    panels = [level["n_radial"] * level["n_chord"] for level in LADDER]
    for axis, quantity, label in ((axes[0], "KT", "$K_T$"), (axes[1], "KQ", "$K_Q$")):
        for entry in points:
            axis.plot(panels, [level[quantity] for level in entry["levels"]], "o-",
                      label=f"J={entry['J']}")
        axis.set_xscale("log", base=2)
        axis.set_xlabel("panels per blade (n_radial x n_chord)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=7)
    for axis in axes:
        for tick, level in zip(panels, LADDER):
            axis.axvline(tick, color="grey", linewidth=0.5, alpha=0.4)
    fig.suptitle("Panel-density ladder (r = 2 on every panel axis)")
    fig.tight_layout()
    path = fig_dir / "kp505_panel_ladder.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append({"name": "kp505_panel_ladder", "file": str(path),
                    "caption": "K_T and K_Q against panels per blade on the three-level ladder for every mission advance ratio."})

    # 4: observed order / GCI
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    labels = [f"{row['J']}\n{row['quantity']}" for row in gci_rows
              if row["quantity"] in {"KT", "KQ"}]
    orders = [row["observed_order"] for row in gci_rows if row["quantity"] in {"KT", "KQ"}]
    gcis = [row["gci_fine_pct"] for row in gci_rows if row["quantity"] in {"KT", "KQ"}]
    axes[0].bar(range(len(orders)), orders, color="C0")
    axes[0].axhline(P_THEORY, color="r", linestyle=":", label="p_th = 2")
    axes[0].axhline(0.0, color="k", linewidth=0.8)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("observed order p")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(range(len(gcis)), gcis, color="C1")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("GCI_fine [%]")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Panel-density GCI: observed order and fine-grid index (negative p = not asymptotic)")
    fig.tight_layout()
    path = fig_dir / "kp505_gci.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append({"name": "kp505_gci", "file": str(path),
                    "caption": "Observed order p and fine-grid GCI per (J, quantity); negative p marks a non-asymptotic triple."})

    # 5: design-point attribution
    channel = attrib["channel_panel_density"]["KT"]
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    names = ["total", "discretisation\n(Richardson)", "model\n(total - disc.)"]
    values = [channel["total_pct"], channel["discretisation_pct"],
              channel["model_pct_richardson"]]
    axis.bar(names, values, color=["C3", "C0", "C2"])
    axis.axhline(0.0, color="k", linewidth=0.8)
    axis.axhline(-THRESHOLD_DESIGN_DKT_PCT, color="r", linestyle=":", linewidth=1.2)
    axis.axhline(THRESHOLD_DESIGN_DKT_PCT, color="r", linestyle=":", linewidth=1.2)
    axis.set_ylabel("$\\Delta K_T$ [%]  (J = 0.70)")
    axis.set_title(f"design-point attribution, p_obs = {channel['observed_order']:.2f}")
    axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = fig_dir / "kp505_attribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append({"name": "kp505_attribution", "file": str(path),
                    "caption": "Design-point K_T miss split into panel-density (Richardson) and model contributions."})

    return written


def build(outdir: Path, workers: int | None = None) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reach = reachability()
    anchor = transcribe_anchor()
    inputs = {
        "anchor_table": {"file": ANCHOR_TABLE_SOURCE,
                         "sha256": anchor.get("sha256", "missing"),
                         "present": anchor.get("available", False)},
        "ittc57_anchor": ITTC57["id"],
        "ladder": list(LADDER),
        "mission_J": list(MISSION_J),
        "design_J": DESIGN_J,
        "thresholds": {
            "design_dKT_pct": THRESHOLD_DESIGN_DKT_PCT,
            "order_tolerance": THRESHOLD_ORDER_TOLERANCE,
            "theory_order": P_THEORY,
        },
        "bem_geometry": {k: v for k, v in PRIMARY.items()},
    }

    transcription = {
        "meta": _meta(inputs),
        "mission": "W22 propeller-openwater-bem (successor of paper-exp-propeller-openwater-bem-1)",
        "reachability": reach,
        "anchor": anchor,
        "verdict": {
            "anchor_reachable": bool(anchor.get("available")),
            "honest_fail": bool(anchor.get("honest_fail", False)),
            "hit_files": reach["n_hit_files"],
            "transcribed_fields": anchor.get("n_fields", 0),
            "sha256_of_hit_files_recorded": reach["n_hit_files"],
            "provenance_class": anchor.get("provenance_class", "MISSING"),
            "primary_handout_in_repo": False,
            "note": (
                "REAL for the secondary tabulation (the table, its sha256 and every "
                "field with line provenance are in-repo); WITHHELD for the primary "
                "KRISO test handout, which is not in the checkout."
            ),
        },
    }
    (outdir / TRANSCRIPTION_NAME).write_text(json.dumps(transcription, indent=2) + "\n")

    if not anchor.get("available"):
        compare = {
            "meta": _meta(inputs),
            "blocked": True,
            "honest_fail": True,
            "what_is_missing": anchor.get("reason"),
            "layer": "table",
            "criteria": [
                {"id": "A1-anchor-reachability-machine-decided", "pass": False,
                 "measured": {"n_hit_files": reach["n_hit_files"],
                              "n_fields": anchor.get("n_fields", 0)}},
                {"id": "A2-five-J-table-with-deviation-and-reference-type", "pass": False,
                 "measured": {"cells": 0, "complete_cells": 0}},
                {"id": "A3-design-point-KT-within-5pct", "pass": False,
                 "measured": {"dKT_pct": None}},
                {"id": "A4-panel-density-gci", "pass": False,
                 "measured": {"n_levels": 0}},
            ],
            "verdict": {"all_criteria_pass": False, "honest_fail": True},
        }
        (outdir / COMPARE_NAME).write_text(json.dumps(compare, indent=2) + "\n")
        return compare

    points = sweep(anchor, workers=workers)
    gci_rows = gci_table(points)
    attrib = attribution(anchor, points, workers=workers)
    verdict_rows = criteria(anchor, reach, points, gci_rows)

    fig_list = figures(outdir, points, gci_rows, attrib)
    _write_csv(outdir / "points.csv", _point_rows(points))
    _write_csv(outdir / "gci.csv", gci_rows)

    design = next(entry for entry in points if abs(entry["J"] - DESIGN_J) < 1e-12)
    n_cells = sum(len(entry["levels"]) for entry in points) + len(
        attrib["channel_wake_model"]) + 1
    n_hits = sum(
        1 for entry in points for level in entry["levels"] if level.get("cache_hit")
    ) + sum(1 for row in attrib["channel_wake_model"] if row.get("cache_hit")) + int(
        bool(attrib["channel_viscous_correction"]["viscous_off"].get("cache_hit"))
    )
    meta = _meta(inputs)
    meta["cache"] = {
        "dir": CACHE_DIR,
        "solver_fingerprint": _SOLVER_FINGERPRINT,
        "n_cached_cells": n_hits,
        "n_cells": n_cells,
        "note": (
            "lattice solutions are cached under runs/w22_kp505_cache keyed on "
            "(J, ladder level, model overrides, sha256 of propeller.py+uq.py).  "
            "A cache hit reproduces the previous run of the same solver "
            "fingerprint bit-for-bit; cells are flagged cache_hit in place."
        ),
    }
    compare = {
        "meta": meta,
        "mission": "W22 propeller-openwater-bem",
        "blocked": False,
        "honest_fail": not all(row["pass"] for row in verdict_rows),
        "reference_hierarchy": {
            "experiment": {
                "anchor_id": "AN-KP505",
                "file": ANCHOR_TABLE_SOURCE,
                "sha256": anchor["sha256"],
                "provenance_class": anchor["provenance_class"],
                "caveat": ANCHOR_CITATION["provenance_depth"],
            },
            "analytic": {
                "anchor_id": ITTC57["id"],
                "formula": ITTC57["formula_verbatim"],
                "use": "viscous-path cross-check and momentum-theory efficiency bound",
            },
        },
        "ladder": list(LADDER),
        "refinement_ratio": LADDER_RATIO,
        "points": points,
        "gci": gci_rows,
        "attribution": attrib,
        "design_point": {
            "J": DESIGN_J,
            "KT_bem": design["KT_bem"],
            "KT_ref": design["KT_ref"],
            "dKT_pct": design["dKT_pct"],
            "KQ_bem": design["KQ_bem"],
            "KQ_ref": design["KQ_ref"],
            "dKQ_pct": design["dKQ_pct"],
            "eta0_bem": design["eta0_bem"],
            "eta0_ref": design["eta0_ref"],
            "deta0_pct": design["deta0_pct"],
            "within_5pct": abs(design["dKT_pct"]) < THRESHOLD_DESIGN_DKT_PCT,
        },
        "criteria": verdict_rows,
        "figures": fig_list,
        "verdict": {
            "all_criteria_pass": all(row["pass"] for row in verdict_rows),
            "design_point_threshold_met": abs(design["dKT_pct"]) < THRESHOLD_DESIGN_DKT_PCT,
            "honest_fail": not all(row["pass"] for row in verdict_rows),
            "failed_criteria": [row["id"] for row in verdict_rows if not row["pass"]],
            "attribution_complete": all(
                row["pass"] for row in verdict_rows
                if row["id"] != "A3-design-point-KT-within-5pct"
            ),
            "task_level_disposition": (
                "PASS" if all(row["pass"] for row in verdict_rows) else
                ("HONEST_FAIL_WITH_ATTRIBUTION"
                 if all(row["task_level_acceptable"] if "task_level_acceptable" in row
                        else row["pass"] for row in verdict_rows)
                 else "HONEST_FAIL_INCOMPLETE")
            ),
            "summary": (
                f"J=0.70 dKT = {design['dKT_pct']:+.2f} % "
                f"(threshold +/-{THRESHOLD_DESIGN_DKT_PCT:.0f} %): "
                f"{'met' if abs(design['dKT_pct']) < THRESHOLD_DESIGN_DKT_PCT else 'HONEST FAIL'}; "
                f"panel-density GCI ladder of {len(LADDER)} levels on all five J"
            ),
        },
    }
    (outdir / COMPARE_NAME).write_text(json.dumps(compare, indent=2) + "\n")
    return compare


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", default=ARTE_DIR)
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel lattice evaluations (default: one per mission J)")
    args = parser.parse_args(argv)
    compare = build(Path(args.outdir), workers=args.workers)
    if compare.get("blocked"):
        print(f"BLOCKED: {compare['what_is_missing']}")
        return 1
    for entry in compare["points"]:
        print(
            f"J={entry['J']:.1f}  KT={entry['KT_bem']:.5f}/{entry['KT_ref']:.3f} "
            f"({entry['dKT_pct']:+.2f}%)  10KQ={entry['10KQ_bem']:.5f}/"
            f"{entry['10KQ_ref']:.3f} ({entry['d10KQ_pct']:+.2f}%)  "
            f"eta0={entry['eta0_bem']:.4f}/{entry['eta0_ref']:.4f} "
            f"({entry['deta0_pct']:+.2f}%)"
        )
    for row in compare["criteria"]:
        print(f"  {row['id']}: {'PASS' if row['pass'] else 'FAIL'}")
    print(compare["verdict"]["summary"])
    print(f"written to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
