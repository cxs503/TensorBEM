"""Executable pins for the numbers printed in ``docs/kkr_causality_band_study.md``.

Provenance
----------
The m3664 study that produced the document (``outputs/m3664/study.py`` and
``kkr_floor_grid.json``) was never committed, so the document's 119 number
tokens had no executable guard: :mod:`tensorbem.benchmarks.doc_claim_audit`
classified the file ``UNGUARDED`` (paper-level, the largest of the eight).
This module recomputes every result table from the current tree and compares
it against the document text, so a future edit that changes a printed digit
turns the test red.

Tables guarded (document section -> function)
----------------------------------------------
``§3.1``  truncation-floor grid, ``ka_max`` x samples      -> :func:`floor_grid`
``§3.2``  infinite-frequency added-mass limit              -> :func:`high_frequency_limit_row`
``§3.3``  panel audit vs floor (ka<=3, 20 samples)         -> :func:`panel_audit`
``§3.4``  non-causal gain detection threshold              -> :func:`gain_sweep`
``§6``    error-verdict rows restating §3.1/§3.2/§3.3      -> :func:`verdict_rows`

The document prints 4-decimal values, so comparisons use an absolute
tolerance of half an ulp of the printed precision (plus a small relative
term); the raw floats are compared, not re-rounded.

Run: ``python -m tensorbem.benchmarks.kkr_band_doc_pins --json-out DIR``
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.kkr_causality import (
    _GRAVITY,
    _DENSITY,
    _SERIES_TERMS,
    _analytic_pair,
    _sweep,
    truncation_floor,
    ursell_high_frequency_limit,
)
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.causality import kkr_check
from tensorbem.free_surface_solver import solve_radiation_indirect

DOC_NAME = "kkr_causality_band_study.md"

#: (ka_max, samples) ladder of the §3.1 table.
GRID_KA_MAX = (1.0, 3.0, 10.0, 30.0)
GRID_POINTS = (16, 24, 32, 64)

#: §3.3 audit band and panel counts.
PANEL_KA_MAX = 3.0
PANEL_POINTS = 20
PANEL_COUNTS = (32, 64)

#: §3.4 gain ladder (multiplicative on the analytic damping).
GAIN_LADDER = (1.02, 1.05, 1.10, 1.25, 1.50)
GAIN_POINTS = 32

_RADIUS = 1.0
_SUBMERGENCE = 2.0


def floor_grid() -> dict[str, object]:
    """§3.1: causality residual of the exactly causal reference, per band x samples."""
    rows = []
    for ka_max in GRID_KA_MAX:
        for points in GRID_POINTS:
            record = truncation_floor(
                _RADIUS,
                _SUBMERGENCE,
                _DENSITY,
                _GRAVITY,
                points=points,
                bands=(ka_max,),
                band_of_interest=ka_max,
                terms=_SERIES_TERMS,
            )
            row = record["floor_at_band"]
            rows.append(
                {
                    "ka_max": float(ka_max),
                    "points": int(points),
                    "max_damping_relative_error": float(row["max_damping_relative_error"]),
                    "max_added_mass_relative_error": float(
                        row["max_added_mass_relative_error"]
                    ),
                }
            )
    return {"ka_min": 0.05, "rows": rows}


def high_frequency_limit_row() -> dict[str, object]:
    """§3.2: A_inf from the 1/(ka) extrapolation over ka = 8..30."""
    added_mass_inf, residual = ursell_high_frequency_limit(
        _RADIUS, _SUBMERGENCE, _DENSITY, _GRAVITY, terms=_SERIES_TERMS
    )
    displaced = _DENSITY * math.pi * _RADIUS**2
    return {
        "added_mass_inf": float(added_mass_inf),
        "extrapolation_max_residual": float(residual),
        "added_mass_inf_over_displaced": float(added_mass_inf / displaced),
        "displaced_mass_per_span": float(displaced),
        "extrapolation_relative_residual": float(residual / added_mass_inf),
    }


def _panel_pairs(panel_count: int, omega: np.ndarray) -> np.ndarray:
    mesh = submerged_circle_mesh(panel_count, _RADIUS, _SUBMERGENCE)
    pairs = []
    for w in omega:
        result = solve_radiation_indirect(
            mesh, float(w / _RADIUS), "heave", density=_DENSITY, panel_nodes=24
        )
        pairs.append((float(result.added_mass[1]), float(result.damping[1])))
    return np.array(pairs)


def panel_audit() -> dict[str, object]:
    """§3.3: KKR residuals and Ursell errors of the analytic pair and 32/64 panels."""
    ka, omega = _sweep(0.05, PANEL_KA_MAX, PANEL_POINTS, _RADIUS, _GRAVITY)
    added_mass_inf, _ = ursell_high_frequency_limit(
        _RADIUS, _SUBMERGENCE, _DENSITY, _GRAVITY, terms=_SERIES_TERMS
    )
    ref_mass, ref_damp = _analytic_pair(
        omega, _RADIUS, _SUBMERGENCE, _DENSITY, _GRAVITY, terms=_SERIES_TERMS
    )
    solutions = {"analytic_multipole": (ref_mass, ref_damp)}
    for n in PANEL_COUNTS:
        solutions[f"panels_{n}"] = _panel_pairs(n, omega)
    rows = []
    for name, (mass, damp) in solutions.items():
        report = kkr_check(omega, mass, damp, added_mass_inf=added_mass_inf)
        rows.append(
            {
                "solution": name,
                "kkr_damping_residual": float(report.max_damping_relative_error),
                "kkr_added_mass_residual": float(report.max_added_mass_relative_error),
                "added_mass_error_vs_ursell": float(
                    np.max(np.abs(mass - ref_mass)) / np.max(np.abs(ref_mass))
                ),
                "damping_error_vs_ursell": float(
                    np.max(np.abs(damp - ref_damp)) / np.max(np.abs(ref_damp))
                ),
            }
        )
    return {"ka_max": PANEL_KA_MAX, "points": PANEL_POINTS, "rows": rows}


def gain_sweep() -> dict[str, object]:
    """§3.4: multiplicative non-causal gain on the analytic damping, ka<=3, 32 samples."""
    ka, omega = _sweep(0.05, PANEL_KA_MAX, GAIN_POINTS, _RADIUS, _GRAVITY)
    mass, damp = _analytic_pair(
        omega, _RADIUS, _SUBMERGENCE, _DENSITY, _GRAVITY, terms=_SERIES_TERMS
    )
    added_mass_inf, _ = ursell_high_frequency_limit(
        _RADIUS, _SUBMERGENCE, _DENSITY, _GRAVITY, terms=_SERIES_TERMS
    )
    clean = kkr_check(omega, mass, damp, added_mass_inf=added_mass_inf)
    clean_residual = float(clean.max_damping_relative_error)
    rows = []
    for gain in GAIN_LADDER:
        report = kkr_check(omega, mass, gain * damp, added_mass_inf=added_mass_inf)
        residual = float(report.max_damping_relative_error)
        rows.append(
            {
                "gain": float(gain),
                "max_damping_residual": residual,
                "ratio_to_clean": residual / clean_residual,
            }
        )
    return {
        "ka_max": PANEL_KA_MAX,
        "points": GAIN_POINTS,
        "clean_residual": clean_residual,
        "rows": rows,
    }


def verdict_rows(grid: dict, audit: dict, limit: dict) -> dict[str, object]:
    """§6: the verdict rows that restate §3.1/§3.2/§3.3 numbers."""
    by_key = {(r["ka_max"], r["points"]): r for r in grid["rows"]}
    by_sol = {r["solution"]: r for r in audit["rows"]}
    return {
        "p32_added_mass_error": by_sol["panels_32"]["added_mass_error_vs_ursell"],
        "p32_damping_error": by_sol["panels_32"]["damping_error_vs_ursell"],
        "p64_added_mass_error": by_sol["panels_64"]["added_mass_error_vs_ursell"],
        "p64_damping_error": by_sol["panels_64"]["damping_error_vs_ursell"],
        "floor_default_band": by_key[(3.0, 32)]["max_damping_relative_error"],
        "floor_wide_band": by_key[(30.0, 32)]["max_damping_relative_error"],
        "a_inf_fit_relative_residual": limit["extrapolation_relative_residual"],
    }


def _full_study() -> dict[str, object]:
    """All guarded quantities in one JSON-serialisable record."""
    grid = floor_grid()
    limit = high_frequency_limit_row()
    audit = panel_audit()
    gains = gain_sweep()
    return {
        "provenance": {
            "tool": "tensorbem.benchmarks.kkr_band_doc_pins",
            "document": DOC_NAME,
            "series_terms": _SERIES_TERMS,
            "radius": _RADIUS,
            "submergence": _SUBMERGENCE,
            "density": _DENSITY,
            "gravity": _GRAVITY,
        },
        "floor_grid": grid,
        "high_frequency_limit": limit,
        "panel_audit": audit,
        "gain_sweep": gains,
        "verdict": verdict_rows(grid, audit, limit),
    }


study = _full_study


# ---------------------------------------------------------------------------
# document parsing
# ---------------------------------------------------------------------------


def _table_after(text: str, header: str) -> list[list[str]]:
    """Rows of the first markdown table whose header line contains ``header``."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("|") and header in line:
            start = i
            break
    if start is None:
        raise AssertionError(f"table header not found: {header!r}")
    rows = []
    for line in lines[start + 2 :]:
        if not line.lstrip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def _value(cell: str) -> float:
    return float(cell.replace("\\_", "").strip())


def _leading_float(cell: str) -> float:
    """First number of a cell that may carry a suffix, e.g. ``0.0424（4.24%）``."""
    cleaned = cell.replace(chr(92) + "_", "").strip()
    match = re.match(r"[-]?[0-9.]+(?:[eE][-+]?[0-9]+)?", cleaned)
    if match is None:
        raise AssertionError(f"no number in cell: {cell!r}")
    return float(match.group(0))


def check_document(doc_path: Path | None = None, study: dict | None = None) -> list[str]:
    """Parse the document and compare every guarded number. Returns failures."""
    if doc_path is None:
        doc_path = Path(__file__).resolve().parents[2] / "docs" / DOC_NAME
    if study is None:
        study = _full_study()
    text = doc_path.read_text(encoding="utf-8")
    failures: list[str] = []

    def close(label: str, expected: float, actual: float, abs_tol: float) -> None:
        tol = abs_tol + 1e-6 * abs(expected)
        if not math.isfinite(actual) or abs(actual - expected) > tol:
            failures.append(f"{label}: doc={expected!r} recomputed={actual!r}")

    # §3.1 floor grid
    grid = study["floor_grid"]
    doc_rows = _table_after(text, "max 阻尼 KKR 残差")
    if len(doc_rows) != len(grid["rows"]):
        failures.append(f"§3.1: {len(doc_rows)} doc rows vs {len(grid['rows'])} recomputed")
    else:
        for doc_row, row in zip(doc_rows, grid["rows"]):
            ka_max, points = float(doc_row[0]), int(doc_row[1])
            if (ka_max, points) != (row["ka_max"], row["points"]):
                failures.append(
                    f"§3.1: row order drifted: doc=({ka_max},{points}) "
                    f"recomputed=({row['ka_max']},{row['points']})"
                )
                continue
            close(
                f"§3.1 ({ka_max},{points}) damping",
                _value(doc_row[2]),
                row["max_damping_relative_error"],
                5.5e-5,
            )
            close(
                f"§3.1 ({ka_max},{points}) added mass",
                _value(doc_row[3]),
                row["max_added_mass_relative_error"],
                5.5e-5,
            )

    # §3.2 inline values
    limit = study["high_frequency_limit"]
    for pattern, key, tol in (
        (r"A∞ = ([\d.]+) kg/m", "added_mass_inf", 5.5e-3),
        (r"外推最大残差 ([\d.]+) kg/m", "extrapolation_max_residual", 5.5e-4),
        (r"A∞/\(ρπa²\) = ([\d.]+)", "added_mass_inf_over_displaced", 5.5e-5),
        (r"ρπa² = ([\d.]+) kg/m", "displaced_mass_per_span", 5.5e-3),
    ):
        match = re.search(pattern, text)
        if match is None:
            failures.append(f"§3.2: pattern not found: {pattern}")
        else:
            close(f"§3.2 {key}", float(match.group(1)), limit[key], tol)

    # §3.3 panel audit
    audit = study["panel_audit"]
    doc_rows = _table_after(text, "附加质量误差 vs Ursell")
    if len(doc_rows) != len(audit["rows"]):
        failures.append(f"§3.3: {len(doc_rows)} doc rows vs {len(audit['rows'])} recomputed")
    else:
        for doc_row, row in zip(doc_rows, audit["rows"]):
            name = doc_row[0].replace("\\_", "_")
            if name != row["solution"]:
                failures.append(f"§3.3: solution order drifted: {name} vs {row['solution']}")
                continue
            for col, key in (
                (1, "kkr_damping_residual"),
                (2, "kkr_added_mass_residual"),
                (3, "added_mass_error_vs_ursell"),
                (4, "damping_error_vs_ursell"),
            ):
                close(f"§3.3 {name} {key}", _value(doc_row[col]), row[key], 5.5e-5)

    # §3.4 gain sweep
    gains = study["gain_sweep"]
    doc_rows = _table_after(text, "相对干净解倍数")
    if len(doc_rows) != len(gains["rows"]):
        failures.append(f"§3.4: {len(doc_rows)} doc rows vs {len(gains['rows'])} recomputed")
    else:
        for doc_row, row in zip(doc_rows, gains["rows"]):
            if abs(float(doc_row[0]) - row["gain"]) > 5.5e-3:
                failures.append(
                    f"§3.4: gain order drifted: doc={doc_row[0]} recomputed={row['gain']}"
                )
                continue
            close(
                f"§3.4 gain={row['gain']} residual",
                _value(doc_row[1]),
                row["max_damping_residual"],
                5.5e-5,
            )
            close(
                f"§3.4 gain={row['gain']} ratio",
                _value(doc_row[2]),
                row["ratio_to_clean"],
                5.5e-3,
            )

    # §6 verdict rows (table parse; a regex would catch the "真值 0" cell)
    verdict = study["verdict"]
    doc_rows = _table_after(text, "是否 <3%")
    by_label = {row[0]: row for row in doc_rows}
    for label, key, tol in (
        ("32 面元附加质量 vs Ursell", "p32_added_mass_error", 5.5e-5),
        ("32 面元阻尼 vs Ursell", "p32_damping_error", 5.5e-5),
        ("64 面元附加质量 vs Ursell", "p64_added_mass_error", 5.5e-5),
        ("64 面元阻尼 vs Ursell", "p64_damping_error", 5.5e-5),
        ("解析解 KKR 阻尼残差（默认频带）", "floor_default_band", 5.5e-5),
        ("解析解 KKR 阻尼残差（k a≤30）", "floor_wide_band", 5.5e-5),
    ):
        row = by_label.get(label)
        if row is None:
            failures.append(f"§6: row not found: {label}")
            continue
        close(f"§6 {label}", _leading_float(row[2]), verdict[key], tol)
    row = by_label.get("A∞ 外推自拟合残差")
    if row is None:
        failures.append("§6: row not found: A∞ 外推自拟合残差")
    else:
        match = re.match(r"([\d.]+)e-04", row[2])
        if match is None:
            failures.append(f"§6 A∞ fit residual: unparseable cell {row[2]!r}")
        else:
            close(
                "§6 A∞ fit relative residual",
                float(match.group(1)) * 1e-4,
                verdict["a_inf_fit_relative_residual"],
                5.5e-6,
            )
    return failures


def format_markdown(study: dict) -> str:
    """Render the guarded tables in the document's own layout."""
    out: list[str] = []
    out.append("| k a_max | 采样数 | max 阻尼 KKR 残差 | max 附加质量 KKR 残差 |")
    out.append("|---|---|---|---|")
    for row in study["floor_grid"]["rows"]:
        out.append(
            f"| {row['ka_max']:g} | {row['points']} | "
            f"{row['max_damping_relative_error']:.4f} | "
            f"{row['max_added_mass_relative_error']:.4f} |"
        )
    limit = study["high_frequency_limit"]
    out += [
        "",
        "## §3.2",
        f"A∞ = {limit['added_mass_inf']:.2f} kg/m"
        f"（外推最大残差 {limit['extrapolation_max_residual']:.3f} kg/m），"
        f"A∞/(ρπa²) = {limit['added_mass_inf_over_displaced']:.4f}"
        f"（ρπa² = {limit['displaced_mass_per_span']:.2f} kg/m）",
        "",
        "## §3.3",
        "| 解 | KKR 阻尼残差 | KKR 附加质量残差 | 附加质量误差 vs Ursell | 阻尼误差 vs Ursell |",
        "|---|---|---|---|---|",
    ]
    for row in study["panel_audit"]["rows"]:
        name = row["solution"].replace("_", chr(92) + "_")
        out.append(
            f"| {name} | {row['kkr_damping_residual']:.4f} | "
            f"{row['kkr_added_mass_residual']:.4f} | {row['added_mass_error_vs_ursell']:.4f} | "
            f"{row['damping_error_vs_ursell']:.4f} |"
        )
    out += ["", "## §3.4", "| 阻尼增益 | max KKR 阻尼残差 | 相对干净解倍数 |", "|---|---|---|"]
    for row in study["gain_sweep"]["rows"]:
        out.append(
            f"| {row['gain']:.2f} | {row['max_damping_residual']:.4f} | "
            f"{row['ratio_to_clean']:.2f} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)
    result = study()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"wrote {args.json_out}")
    failures = check_document(None, result)
    if failures:
        print("DOC PIN FAILURES:")
        for failure in failures:
            print("  -", failure)
        return 1
    print(f"all {DOC_NAME} pins hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
