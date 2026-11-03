"""Machine-checked pins for the numbers quoted in ``docs/three_dimensional_bem.md``.

Why this module exists
----------------------
``docs/three_dimensional_bem.md`` quotes four result tables (interior Laplace,
exterior constant Dirichlet, exterior sphere added mass, exterior sphere
velocity).  Those tables are hand-written prose LaTeX; the numbers in them come
from code that keeps moving.  The 2-D counterpart of this hole is already
plugged by :mod:`tensorbem.benchmarks.doc_pins`, which was written *after* a
quadrature change moved the documented 128-panel Ursell added-mass error from
``7.72e-3`` to ``8.51e-3`` and nothing in CI noticed
(``docs/ursell_quadrature_provenance.md``).  The 3-D tables were never pinned,
so the same silent drift was still possible for every number in them.

The design point here is that **no number is copied by hand**: the pin registry
stores only *where* each number lives (table, panel count, column) and the
checker reads the declared value back out of the Markdown, decodes the LaTeX,
recomputes the corresponding benchmark metric, and compares.  A changed doc, a
changed tolerance or a changed solver all show up as a machine-readable verdict.

Workflow
--------
* ``python -m tensorbem.benchmarks.doc_pins_3d``           recompute and check
* ``python -m tensorbem.benchmarks.doc_pins_3d --write``   refresh the JSON record
* ``python -m tensorbem.benchmarks.doc_pins_3d --levels 1 2``  fast subset
* ``python -m tensorbem.benchmarks.doc_pins_3d --figure docs/figures/x.png``

``tests/test_documented_benchmarks_3d.py`` runs a deliberately small subset
(the coarse meshes) so it stays cheap in CI.

Exit status is 1 when any pinned number has drifted, which is what CI keys on.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DOC_PATH = Path("docs/three_dimensional_bem.md")
DEFAULT_RECORD = Path("docs/benchmark_pins_3d.json")

#: Relative tolerance used for every pinned number, with a per-metric absolute
#: floor so that machine-epsilon-scale sanity numbers are not compared relatively.
RELATIVE_TOLERANCE = 5.0e-3


# --------------------------------------------------------------------------- #
# LaTeX number decoding
# --------------------------------------------------------------------------- #

_TIMES = re.compile(r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{?\s*(-?\d+)\s*\}?")
_SCI = re.compile(r"(-?\d+(?:\.\d+)?)[eE]([-+]?\d+)")
_PLAIN = re.compile(r"(-?\d+(?:\.\d+)?)")
_EMPTY_CELLS = {"", "—", "-", "--", "–"}


def decode_number(cell: str) -> float | None:
    """Decode one documentation table cell into a float.

    Handles ``\\(1.4600\\times10^{-1}\\)``, ``1.0171e-1``, ``\\(9.13\\)`` and the
    em-dash placeholder used for "not applicable" cells.
    """
    text = cell.strip().replace("$", "")
    stripped = text.strip("()")
    if stripped in _EMPTY_CELLS or text in _EMPTY_CELLS:
        return None
    match = _TIMES.search(text)
    if match is not None:
        return float(match.group(1)) * 10.0 ** int(match.group(2))
    match = _SCI.search(text)
    if match is not None:
        return float(match.group(1)) * 10.0 ** int(match.group(2))
    match = _PLAIN.search(text)
    if match is not None:
        return float(match.group(1))
    return None


# --------------------------------------------------------------------------- #
# Doc table parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TableSpec:
    """One documentation table and the benchmark that produces it."""

    table_id: str
    """Stable identifier used in the output records."""

    doc_title: str
    """Human-readable location, e.g. ``section 5.1 (interior Laplace)``."""

    header_cells: tuple[str, ...]
    """Distinguishing substring for each header cell, in column order."""

    compute: Callable[[int], Mapping[str, Any]]
    """Run the benchmark for one refinement level (subdivisions)."""

    max_level: int
    """Coarsest level kept out of the default (fast) check is ``level > 1``."""


def _compute_interior(level: int) -> Mapping[str, Any]:
    from tensorbem.benchmarks.three_dimensional_study import run_interior_case

    return run_interior_case(level)


def _compute_constant_dirichlet(level: int) -> Mapping[str, Any]:
    from tensorbem.benchmarks.three_dimensional_study import (
        run_constant_dirichlet_case,
    )

    return run_constant_dirichlet_case(level)


def _compute_added_mass(level: int) -> Mapping[str, Any]:
    from tensorbem.benchmarks.sphere_added_mass import run

    return run(level)


def _compute_velocity(level: int) -> Mapping[str, Any]:
    from tensorbem.benchmarks.sphere_velocity import run

    return run(subdivisions=level)


TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        table_id="interior_laplace",
        doc_title="section 5.1 table (interior linear harmonic field, table 1)",
        header_cells=("h_{\\max}", "E_q", "E_u", "条件数"),
        compute=_compute_interior,
        max_level=3,
    ),
    TableSpec(
        table_id="exterior_added_mass",
        doc_title="section 5.2 table (exterior sphere added mass)",
        header_cells=("m_{A,N}", "相对误差", "通量相对 L2 误差"),
        compute=_compute_added_mass,
        max_level=3,
    ),
    TableSpec(
        table_id="exterior_constant_dirichlet",
        doc_title="section 5.3 table (exterior constant Dirichlet total flux)",
        header_cells=("计算总通量", "观测阶", "均值"),
        compute=_compute_constant_dirichlet,
        max_level=3,
    ),
    TableSpec(
        table_id="exterior_velocity",
        doc_title="section 5.4 table (exterior translating-sphere velocity)",
        header_cells=("E_v", "恢复路径", "表示公式"),
        compute=_compute_velocity,
        max_level=3,
    ),
)


@dataclass(frozen=True)
class Column:
    """One pinned column of one documentation table."""

    symbol: str
    """Column symbol as it appears in the doc header."""

    metric_key: str
    """Key in the benchmark output dictionary."""

    floor: float = 1.0e-12
    """Absolute floor of the drift denominator.

    Epsilon-scale sanity numbers (row-sum errors, linear-solver residuals) are
    only ever claimed to ``1e-15`` order, and their leading digits are a BLAS
    artefact, so comparing them *relatively* would be meaningless.  For those
    columns the floor is raised to ``1e-13``: the check then asks "is the
    quantity still at round-off level", which is the claim the document makes.
    """


COLUMNS: Mapping[str, tuple[Column | None, ...]] = {
    # Aligned with the documented data rows *including* the derived-order
    # columns, which are left as ``None`` holes: an order cell is arithmetic on
    # the pinned errors, so pinning it separately would only add noise.
    "interior_laplace": (
        Column("h_max", "h_max"),
        Column("E_q", "normal_derivative_relative_l2_error"),
        None,  # p_q
        Column("E_u", "field_rmse"),
        None,  # p_u
        Column("kappa", "single_layer_condition_number"),
        Column("row_sum_error", "double_layer_row_sum_error", floor=1.0e-13),
        Column("relative_residual", "relative_residual_l2", floor=1.0e-13),
    ),
    "exterior_added_mass": (
        Column("h_max", "h_max"),
        Column("m_A", "added_mass"),
        Column("rel_err", "added_mass_relative_error"),
        None,  # p_m
        Column("flux_rel_l2", "normal_derivative_relative_l2_error"),
        None,  # p_q
    ),
    "exterior_constant_dirichlet": (
        Column("total_flux", "total_flux"),
        Column("rel_err", "total_flux_relative_error"),
        None,  # observed order
        Column("mean_flux", "normal_derivative_mean"),
    ),
    "exterior_velocity": (
        Column("h_max", "h_max"),
        Column("E_v", "velocity_relative_l2_error"),
        None,  # p
        Column("max_abs", "velocity_max_abs_error"),
        Column("E_surf_recovered", "surface_velocity_relative_l2_error"),
        None,  # p_surf
        Column(
            "E_surf_representation",
            "representation_surface_velocity_relative_l2_error",
        ),
    ),
}

def _row_panels(cell: str) -> int | None:
    try:
        return int(cell.strip())
    except ValueError:
        return None


def parse_doc_tables(doc_text: str) -> dict[str, dict[int, list[float | None]]]:
    """Extract the data rows of every pinned table from the document text.

    Returns ``{table_id: {panel_count: [cell values in column order]}}`` where the
    leading panel-count column has been removed from the value list and order
    matches :data:`COLUMNS`.
    """
    lines = doc_text.splitlines()
    found: dict[str, dict[int, list[float | None]]] = {}
    for spec in TABLES:
        rows: dict[int, list[float | None]] = {}
        in_block = False
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("|"):
                in_block = False
                continue
            if all(marker in stripped for marker in spec.header_cells):
                in_block = True
                continue
            if not in_block:
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            panels = _row_panels(cells[0])
            if panels is None:
                continue
            values = [decode_number(cell) for cell in cells[1:]]
            if len(values) != len(COLUMNS[spec.table_id]):
                in_block = False
                continue
            rows[panels] = values
        if rows:
            found[spec.table_id] = rows
    return found


def _table_header_matches(doc_text: str, spec: TableSpec) -> bool:
    """Check the documentation still has a table with the expected columns."""
    for line in doc_text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        if all(marker in line for marker in spec.header_cells):
            return True
    return False


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #


@dataclass
class PinResult:
    table_id: str
    panels: int
    symbol: str
    declared: float
    measured: float
    drift: float
    passed: bool
    absolute_difference: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": self.table_id,
            "panels": self.panels,
            "column": self.symbol,
            "declared": self.declared,
            "measured": self.measured,
            "drift": self.drift,
            "absolute_difference": self.absolute_difference,
            "passed": self.passed,
        }


def _level_for_panels(spec: TableSpec, panels: int) -> int:
    level = int(round(math.log(panels / 20.0, 4.0))) if panels >= 20 else 0
    if 20 * 4**level != panels:
        raise ValueError(
            f"table {spec.table_id}: panel count {panels} is not 20*4**k"
        )
    return level


def check(
    doc_path: Path = DOC_PATH,
    *,
    max_level: int = 1,
    tolerance: float = RELATIVE_TOLERANCE,
) -> tuple[list[PinResult], list[str]]:
    """Recompute every pinned 3-D number and compare it with the document.

    ``max_level`` bounds the refinement levels that are actually run, so CI can
    check the coarse rows cheaply while ``--levels 3`` runs everything.
    """
    doc_text = doc_path.read_text(encoding="utf-8")
    tables = parse_doc_tables(doc_text)
    results: list[PinResult] = []
    notes: list[str] = []
    for spec in TABLES:
        if not _table_header_matches(doc_text, spec):
            notes.append(f"DRIFT {spec.table_id}: header row not found in {doc_path}")
            continue
        rows = tables.get(spec.table_id)
        if not rows:
            notes.append(f"DRIFT {spec.table_id}: no data rows parsed from {doc_path}")
            continue
        cache: dict[int, Mapping[str, Any]] = {}
        for panels in sorted(rows):
            level = _level_for_panels(spec, panels)
            if level > max_level:
                continue
            if level not in cache:
                cache[level] = spec.compute(level)
            computed = cache[level]
            values = rows[panels]
            for column, declared in zip(COLUMNS[spec.table_id], values):
                if column is None or declared is None:
                    continue
                if column.metric_key not in computed:
                    notes.append(
                        f"DRIFT {spec.table_id}/{panels}/{column.symbol}: "
                        f"metric {column.metric_key!r} missing from the benchmark"
                    )
                    continue
                measured = float(computed[column.metric_key])
                denominator = max(abs(declared), column.floor)
                drift = abs(measured - declared) / denominator
                results.append(
                    PinResult(
                        table_id=spec.table_id,
                        panels=panels,
                        symbol=column.symbol,
                        declared=declared,
                        measured=measured,
                        drift=drift,
                        passed=drift <= tolerance,
                        absolute_difference=abs(measured - declared),
                    )
                )
    return results, notes


def summarise(results: Sequence[PinResult]) -> dict[str, Any]:
    failures = [item for item in results if not item.passed]
    worst = max(results, key=lambda item: item.drift) if results else None
    return {
        "pins": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "worst": None if worst is None else worst.as_dict(),
        "failures": [item.as_dict() for item in failures],
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def plot_pins(
    doc_path: Path, output: Path, *, max_level: int = 3
) -> dict[str, Any]:
    """Declared-vs-measured error curves for every pinned table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doc_text = doc_path.read_text(encoding="utf-8")
    tables = parse_doc_tables(doc_text)
    panels_of: dict[str, list[int]] = {}
    measured_of: dict[str, dict[str, list[float]]] = {}
    for spec in TABLES:
        rows = tables.get(spec.table_id, {})
        cache: dict[int, Mapping[str, Any]] = {}
        panels_of[spec.table_id] = sorted(rows)
        pinned = [column for column in COLUMNS[spec.table_id] if column is not None]
        for column in pinned:
            measured_of.setdefault(spec.table_id, {})[column.symbol] = []
        for panels in sorted(rows):
            level = _level_for_panels(spec, panels)
            if level > max_level:
                continue
            if level not in cache:
                cache[level] = spec.compute(level)
            for column in pinned:
                measured_of[spec.table_id][column.symbol].append(
                    float(cache[level][column.metric_key])
                )

    headline = {
        "interior_laplace": ("E_q", "E_u"),
        "exterior_added_mass": ("rel_err", "flux_rel_l2"),
        "exterior_constant_dirichlet": ("rel_err",),
        "exterior_velocity": ("E_v", "E_surf_recovered"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    markers = {"E_q": "o", "E_u": "s", "rel_err": "^", "flux_rel_l2": "v",
               "E_v": "D", "E_surf_recovered": "P"}
    colours = {"E_q": "C0", "E_u": "C0", "rel_err": "C1", "flux_rel_l2": "C1",
               "E_v": "C2", "E_surf_recovered": "C2"}
    labels = {
        "interior_laplace": "interior Laplace",
        "exterior_added_mass": "exterior added mass",
        "exterior_constant_dirichlet": "exterior constant Dirichlet",
        "exterior_velocity": "exterior velocity",
    }
    drift_points: list[float] = []
    for table_id, symbols in headline.items():
        rows = tables.get(table_id, {})
        panels = sorted(rows)
        plotted = panels[: len(measured_of[table_id][symbols[0]])]
        for symbol in symbols:
            column_index = [
                None if c is None else c.symbol for c in COLUMNS[table_id]
            ].index(symbol)
            declared = [rows[p][column_index] for p in plotted]
            measured = measured_of[table_id][symbol]
            style = dict(
                marker=markers[symbol], color=colours[symbol], markersize=5.0,
                label=f"{labels[table_id]} {symbol} (doc)",
            )
            axes[0].loglog(plotted, declared, linestyle="--", **style)
            axes[0].loglog(
                plotted, measured, linestyle="-", marker=markers[symbol],
                color=colours[symbol], markersize=4.0,
                label=f"{labels[table_id]} {symbol} (recomputed)",
            )
            for d, m in zip(declared, measured):
                drift_points.append(abs(m - d) / max(abs(d), 1.0e-12))
    axes[0].set_xlabel("panels N")
    axes[0].set_ylabel("relative error")
    axes[0].set_title("3-D benchmark errors: documented vs recomputed")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=6.0, ncol=2)

    drift_points = [min(value, 1.0) for value in drift_points]
    axes[1].semilogy(range(len(drift_points)), drift_points, "o", markersize=4.0)
    axes[1].axhline(
        RELATIVE_TOLERANCE, color="C3", linestyle="--",
        label=f"tolerance {RELATIVE_TOLERANCE:g}",
    )
    axes[1].set_xlabel("pinned number index")
    axes[1].set_ylabel("|recomputed - documented| / documented")
    axes[1].set_title("Drift of every number in the 3-D result tables")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7.0)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150.0)
    plt.close(figure)
    return {"figure": str(output), "points": len(drift_points),
            "max_drift": max(drift_points) if drift_points else None}


# --------------------------------------------------------------------------- #
# Record and CLI
# --------------------------------------------------------------------------- #


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - only when git is unavailable
        return None


def build_record(results: Sequence[PinResult], notes: Sequence[str], levels: int) -> dict:
    return {
        "schema": "tensorbem.doc_pins_3d.v1",
        "generated_by": "python -m tensorbem.benchmarks.doc_pins_3d --write",
        "commit": _git_commit(),
        "doc": str(DOC_PATH),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "max_level": levels,
        "summary": summarise(results),
        "notes": list(notes),
        "pins": [item.as_dict() for item in results],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--write", action="store_true", help="rewrite the JSON record")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if any pinned number drifted",
    )
    parser.add_argument(
        "--levels", nargs="+", type=int, default=[3],
        help="highest refinement level to run (1 = coarse/CI, 3 = all rows)",
    )
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="print the record to stdout")
    args = parser.parse_args(argv)

    levels = max(args.levels)
    started = time.time()
    results, notes = check(args.doc, max_level=levels)
    elapsed = time.time() - started
    record = build_record(results, notes, levels)
    record["elapsed_seconds"] = round(elapsed, 3)

    if args.figure is not None:
        record["figure"] = plot_pins(args.doc, args.figure, max_level=levels)
    if args.write:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(record, indent=2))

    summary = record["summary"]
    print(
        f"doc_pins_3d: {summary['passed']}/{summary['pins']} pins match "
        f"(tolerance {RELATIVE_TOLERANCE:g}, {elapsed:.1f} s, levels<= {levels})"
    )
    for note in notes:
        print("  " + note)
    for failure in summary["failures"]:
        print(
            "  DRIFT {table}/{panels}/{column}: documented {declared!r} "
            "recomputed {measured!r} drift {drift:.3e}".format(**failure)
        )
    if summary["worst"] is not None:
        worst = summary["worst"]
        print(
            "  worst: {table}/{panels}/{column} drift {drift:.3e}".format(**worst)
        )
    failed = summary["failed"] > 0 or bool(notes)
    if args.check:
        return 1 if failed else 0
    return 0 if not failed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())