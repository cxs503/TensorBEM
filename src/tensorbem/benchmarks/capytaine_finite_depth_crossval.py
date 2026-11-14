"""Finite-depth cross-validation of TensorBEM against capytaine.

Contract
--------
Read the frozen capytaine reference produced by
``scripts/capytaine_finite_depth_ref.py`` (finite ``water_depth`` per cell),
evaluate the TensorBEM panel ladder on the same ``h/a x k0*a`` grid (see
:mod:`tensorbem.benchmarks.finite_depth_crossval_plan`), Richardson-extrapolate
both sides, and report the code-to-code deviation with the discretisation band
that produced it.

Why extrapolate instead of comparing two numbers
------------------------------------------------
Both codes are low-order constant-strength panel methods on an inscribed
polyhedral approximation of the sphere.  At a single mesh size the difference is
dominated by *two* independent discretisation errors, so a 10 % disagreement at
288 panels says nothing about either solver's correctness.  The plan therefore
measures the observed order of convergence on each side and compares the
extrapolated (``N -> infinity``) values, reporting the order and the
extrapolation distance so that the residual can be judged per cell.

Outputs (all numbers written programmatically, nothing transcribed by hand)
---------------------------------------------------------------------------
``benchmarks/verified/capytaine_finite_depth/tensor_ladder.json``
    Raw TensorBEM ladder: panel count -> heave/surge added mass and damping.
``benchmarks/verified/capytaine_finite_depth/results.json``
    Extrapolated comparison, observed orders, gates, negative control.
``docs/capytaine_finite_depth_crossval.md`` (``--write-doc``)
    Published page: generated tables for every grid cell.
``docs/fig_capytaine_finite_depth_*.png`` (``--write-doc``)
    Frequency response with both codes, convergence ladder, difference map.

Run::

    PYTHONPATH=src python -m tensorbem.benchmarks.capytaine_finite_depth_crossval \
        --out-root benchmarks/verified/capytaine_finite_depth --write-doc
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tensorbem.benchmarks import finite_depth_crossval_plan as plan
from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import submerged_sphere_mesh

CENTER = (0.0, 0.0, -plan.SPHERE_SUBMERGENCE)
MODES = (0, 2)  # surge, heave
MODE_NAMES = ("surge", "heave")

DEFAULT_ROOT = Path("benchmarks/verified/capytaine_finite_depth")
REFERENCE_FILE = "capytaine_ref.json"
LADDER_FILE = "tensor_ladder.json"
RESULT_FILE = "results.json"
DOC_PATH = Path("docs/capytaine_finite_depth_crossval.md")
FIGURE_FREQUENCY = "fig_capytaine_finite_depth_frequency.png"
FIGURE_MAP = "fig_capytaine_finite_depth_difference_map.png"
FIGURE_CONVERGENCE = "fig_capytaine_finite_depth_convergence.png"


# --------------------------------------------------------------------------- #
# TensorBEM side
# --------------------------------------------------------------------------- #
def _panels(level: int) -> int:
    return int(submerged_sphere_mesh(int(level), radius=plan.SPHERE_RADIUS,
                                    submergence=plan.SPHERE_SUBMERGENCE).panel_count)


def tensorbem_values(level: int, cell: dict[str, float]) -> dict[str, Any]:
    """Heave and surge radiation coefficients of one cell on one panel level."""
    mesh = submerged_sphere_mesh(
        int(level), radius=plan.SPHERE_RADIUS, submergence=plan.SPHERE_SUBMERGENCE
    )
    t0 = time.perf_counter()
    result = solve_radiation_finite_depth_3d(
        mesh,
        cell["nu"],
        cell["depth"],
        modes=list(MODES),
        center=CENTER,
        density=plan.DENSITY,
        gravity=plan.GRAVITY,
    )
    values = {}
    for slot, name in enumerate(MODE_NAMES):
        values[name] = {
            "added_mass": float(result.added_mass[slot, slot]),
            "damping": float(result.damping[slot, slot]),
        }
    return {
        "level": int(level),
        "panels": int(mesh.panel_count),
        "mode_count": int(result.mode_count),
        "condition_number": float(result.condition_number),
        "relative_residual": float(result.relative_residual),
        "seconds": round(time.perf_counter() - t0, 3),
        "values": values,
    }


def deep_water_values(level: int, cell: dict[str, float]) -> dict[str, Any]:
    """The depth-blind deep-water solver on the same mesh and frequency.

    This is the negative control: if the finite-depth cross-check cannot
    separate this from the finite-depth answer at small ``k0*h``, the grid is
    not probing depth at all.
    """
    mesh = submerged_sphere_mesh(
        int(level), radius=plan.SPHERE_RADIUS, submergence=plan.SPHERE_SUBMERGENCE
    )
    result = solve_radiation_3d(
        mesh, cell["nu"], modes=list(MODES), center=CENTER,
        density=plan.DENSITY, gravity=plan.GRAVITY,
    )
    values = {}
    for slot, name in enumerate(MODE_NAMES):
        values[name] = {
            "added_mass": float(result.added_mass[slot, slot]),
            "damping": float(result.damping[slot, slot]),
        }
    return {"panels": int(mesh.panel_count), "values": values}


def _worker(cell: dict[str, float], level: int, control_level: int) -> dict[str, Any]:
    """One independent ladder task (level -1 = the deep-water negative control)."""
    if level < 0:
        return deep_water_values(control_level, cell)
    return tensorbem_values(level, cell)


def build_ladder(
    cells: Iterable[dict[str, float]],
    levels: Sequence[int],
    *,
    cache_path: Path | None = None,
    extra_level: int | None = None,
    extra_cells: Sequence[tuple[float, float]] = (),
    control_level: int = 1,
    jobs: int = 1,
) -> dict[str, Any]:
    """Run (or resume) the TensorBEM panel ladder over the grid.

    ``jobs > 1`` evaluates independent ``(cell, level)`` tasks in separate
    processes; each task is a dense ``O(N**2)`` assembly, so process-level
    parallelism (with the BLAS pinned to one thread by the caller) is the
    efficient axis here.
    """
    cache: dict[str, Any] = {}
    if cache_path is not None and cache_path.exists():
        cache = json.loads(cache_path.read_text())
    cache.setdefault("cells", {})

    pending: list[tuple[str, dict[str, float], int]] = []
    extra_set = set(map(tuple, extra_cells))
    for cell in cells:
        key = plan.cell_id(cell["h_over_a"], cell["ka"])
        entry = cache["cells"].setdefault(
            key,
            {
                "cell": key,
                "h_over_a": cell["h_over_a"],
                "ka": cell["ka"],
                "depth": cell["depth"],
                "k0": cell["k0"],
                "k0h": cell["k0h"],
                "nu": cell["nu"],
                "levels": {},
                "negative_control": None,
            },
        )
        wanted = list(levels)
        if extra_level is not None and (cell["h_over_a"], cell["ka"]) in extra_set:
            wanted.append(int(extra_level))
        for level in wanted:
            if str(level) not in entry["levels"]:
                pending.append((key, cell, int(level)))
        if entry["negative_control"] is None:
            pending.append((key, cell, -1))  # -1 = deep-water control

    def _store(key: str, level: int, record: dict[str, Any]) -> None:
        if level < 0:
            cache["cells"][key]["negative_control"] = record
        else:
            cache["cells"][key]["levels"][str(level)] = record
        print(
            f"  tb {key} level={level} panels={record['panels']} "
            f"seconds={record['seconds']}",
            flush=True,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1) + "\n")

    if jobs > 1 and len(pending) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            futures = {
                pool.submit(_worker, cell, level, control_level): (key, level)
                for key, cell, level in pending
            }
            for future in as_completed(futures):
                key, level = futures[future]
                _store(key, level, future.result())
    else:
        for key, cell, level in pending:
            _store(key, level, _worker(cell, level, control_level))
    cache["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "density": plan.DENSITY,
        "gravity": plan.GRAVITY,
        "levels": list(levels),
        "extra_level": extra_level,
        "extra_cells": [list(c) for c in extra_cells],
        "control_level": int(control_level),
    }
    return cache


# --------------------------------------------------------------------------- #
# Richardson extrapolation
# --------------------------------------------------------------------------- #
def _order_from_three(panels: Sequence[float], values: Sequence[float]) -> float | None:
    """Observed order ``p`` from three refinement levels.

    Assumes ``v(N) = v + C N**(-p)`` and solves the resulting scalar equation for
    ``p`` by bisection on ``p in (0, 6]``.  Returns ``None`` when the three
    values are not monotone in the assumed direction (the ladder is then not in
    its asymptotic range and extrapolation would be a fit, not a measurement).
    """
    n1, n2, n3 = (float(v) for v in panels[:3])
    v1, v2, v3 = (float(v) for v in values[:3])
    if not (n1 < n2 < n3):
        return None
    d21 = v2 - v1
    d32 = v3 - v2
    if d21 == 0.0 or d32 == 0.0:
        return None
    if d21 * d32 <= 0.0:
        return None
    ratio = d32 / d21

    def residual(p: float) -> float:
        r1 = n1 ** (-p)
        r2 = n2 ** (-p)
        r3 = n3 ** (-p)
        return (r3 - r2) / (r2 - r1) - ratio

    lo, hi = 1.0e-3, 6.0
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0.0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


def richardson(
    panels: Sequence[float], values: Sequence[float]
) -> dict[str, Any]:
    """Extrapolate a panel ladder to ``N -> infinity``.

    Uses the three finest levels; the order is *measured* from the same three
    levels unless the ladder is non-monotone, in which case the finest value is
    returned and ``extrapolated`` is set to ``False``.
    """
    panels = [float(p) for p in panels]
    values = [float(v) for v in values]
    order = list(np.argsort(panels))
    panels = [panels[i] for i in order]
    values = [values[i] for i in order]
    finest = values[-1]
    out: dict[str, Any] = {
        "panels": panels,
        "values": values,
        "finest": finest,
        "order": None,
        "extrapolated": False,
        "value": finest,
        "extrapolation_shift": 0.0,
    }
    if len(values) >= 3:
        p = _order_from_three(panels[-3:], values[-3:])
        if p is not None:
            n2, n3 = panels[-2], panels[-1]
            v2, v3 = values[-2], values[-1]
            c = (v3 - v2) / (n3 ** (-p) - n2 ** (-p))
            extrap = v3 - c * n3 ** (-p)
            out.update(
                {
                    "order": p,
                    "extrapolated": True,
                    "value": float(extrap),
                    "extrapolation_shift": float(
                        (v3 - extrap) / extrap if extrap else float("inf")
                    ),
                }
            )
    return out


def _extrapolated_entry(levels: dict[str, Any], dof: str, quantity: str) -> dict[str, Any]:
    ordered = sorted(levels.values(), key=lambda rec: rec["panels"])
    panels = [rec["panels"] for rec in ordered]
    values = [rec["values"][dof][quantity] for rec in ordered]
    return richardson(panels, values)


def _reference_entry(cell: dict[str, Any], dof: str, quantity: str) -> dict[str, Any]:
    ordered = sorted(cell["levels"], key=lambda rec: rec["panels"])
    panels = [rec["panels"] for rec in ordered]
    values = [rec["values"][dof][quantity] for rec in ordered]
    return richardson(panels, values)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare(
    ladder: dict[str, Any],
    reference: dict[str, Any],
    *,
    tol_added_mass: float = plan.TOL_ADDED_MASS,
    tol_damping: float = plan.TOL_DAMPING,
) -> dict[str, Any]:
    """Cell-by-cell extrapolated comparison plus gates."""
    ref_cells = {cell["cell"]: cell for cell in reference["cells"]}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    worst: dict[str, float] = {}
    for key, entry in sorted(ladder["cells"].items()):
        ref = ref_cells.get(key)
        if ref is None:
            continue
        row: dict[str, Any] = {
            "cell": key,
            "h_over_a": entry["h_over_a"],
            "ka": entry["ka"],
            "k0h": entry["k0h"],
            "quantities": {},
        }
        tols = {"added_mass": tol_added_mass, "damping": tol_damping}
        for dof in MODE_NAMES:
            for quantity in ("added_mass", "damping"):
                tb = _extrapolated_entry(entry["levels"], dof, quantity)
                cp = _reference_entry(ref, dof, quantity)
                base = cp["value"]
                rel = (tb["value"] - base) / abs(base) if base else float("inf")
                raw_rel = (
                    (tb["finest"] - cp["finest"]) / abs(cp["finest"])
                    if cp["finest"]
                    else float("inf")
                )
                row["quantities"][f"{dof}.{quantity}"] = {
                    "tensorbem": tb["value"],
                    "capytaine": base,
                    "tensorbem_panels": tb["panels"],
                    "tensorbem_order": tb["order"],
                    "tensorbem_extrapolated": tb["extrapolated"],
                    "tensorbem_shift": tb["extrapolation_shift"],
                    "capytaine_order": cp["order"],
                    "capytaine_shift": cp["extrapolation_shift"],
                    "relative": float(rel),
                    "relative_finest_levels": float(raw_rel),
                }
                slot = f"{dof}.{quantity}"
                worst[slot] = max(worst.get(slot, 0.0), abs(rel))
                if abs(rel) > tols[quantity]:
                    failures.append(
                        f"{key} {slot}: {rel:+.2%} > {tols[quantity]:.0%}"
                    )
                if abs(rel) > plan.OWNER_LIMIT:
                    failures.append(
                        f"{key} {slot}: {rel:+.2%} exceeds the owner 15% limit"
                    )
        control = entry.get("negative_control")
        if control is not None:
            finest_key = str(max(int(k) for k in entry["levels"]))
            finite_a33 = entry["levels"][finest_key]["values"]["heave"]["added_mass"]
            deep_a33 = control["values"]["heave"]["added_mass"]
            row["negative_control_gap_added_mass"] = float(
                (finite_a33 - deep_a33) / abs(deep_a33)
            )
        rows.append(row)
    return {
        "rows": rows,
        "worst_relative": worst,
        "failures": failures,
        "tolerances": {"added_mass": tol_added_mass, "damping": tol_damping,
                       "owner_limit": plan.OWNER_LIMIT},
    }


def gates(comparison: dict[str, Any], ladder: dict[str, Any],
          reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Named accept/reject gates for the published study."""
    out: list[dict[str, Any]] = []
    worst = comparison["worst_relative"]
    out.append(
        {
            "gate": "added_mass_agreement",
            "worst": max(worst.get("heave.added_mass", 0.0),
                         worst.get("surge.added_mass", 0.0)),
            "limit": plan.TOL_ADDED_MASS,
            "passed": max(worst.get("heave.added_mass", 0.0),
                          worst.get("surge.added_mass", 0.0)) <= plan.TOL_ADDED_MASS,
        }
    )
    out.append(
        {
            "gate": "damping_agreement",
            "worst": max(worst.get("heave.damping", 0.0),
                         worst.get("surge.damping", 0.0)),
            "limit": plan.TOL_DAMPING,
            "passed": max(worst.get("heave.damping", 0.0),
                          worst.get("surge.damping", 0.0)) <= plan.TOL_DAMPING,
        }
    )
    control = [
        row["negative_control_gap_added_mass"]
        for row in comparison["rows"]
        if "negative_control_gap_added_mass" in row and row["k0h"] <= 2.0
    ]
    out.append(
        {
            "gate": "negative_control_separates_depth",
            "min_gap": min(control) if control else None,
            "limit": plan.NEGATIVE_CONTROL_MIN_GAP,
            "passed": bool(control) and min(control) >= plan.NEGATIVE_CONTROL_MIN_GAP,
        }
    )
    orders = [
        q["capytaine_order"]
        for row in comparison["rows"]
        for q in row["quantities"].values()
        if q["capytaine_order"] is not None
    ]
    out.append(
        {
            "gate": "reference_ladder_in_asymptotic_range",
            "min_order": min(orders) if orders else None,
            "limit": plan.MIN_OBSERVED_ORDER,
            "passed": bool(orders) and min(orders) >= plan.MIN_OBSERVED_ORDER,
        }
    )
    tb_orders = [
        q["tensorbem_order"]
        for row in comparison["rows"]
        for q in row["quantities"].values()
        if q["tensorbem_order"] is not None
    ]
    out.append(
        {
            "gate": "tensorbem_ladder_in_asymptotic_range",
            "min_order": min(tb_orders) if tb_orders else None,
            "limit": plan.MIN_OBSERVED_ORDER,
            "passed": bool(tb_orders) and min(tb_orders) >= plan.MIN_OBSERVED_ORDER,
        }
    )
    residuals = [
        rec["relative_residual"]
        for entry in ladder["cells"].values()
        for rec in entry["levels"].values()
    ]
    out.append(
        {
            "gate": "linear_solve_residuals",
            "max": max(residuals) if residuals else None,
            "limit": 1e-8,
            "passed": bool(residuals) and max(residuals) <= 1e-8,
        }
    )
    finite_depth = all(
        math.isfinite(cell["depth"]) for cell in reference["cells"]
    )
    out.append(
        {
            "gate": "reference_is_finite_depth",
            "passed": finite_depth,
            "limit": True,
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f} %"


def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_document(result: dict[str, Any]) -> str:
    """Markdown page, every number taken from ``result``."""
    comparison = result["comparison"]
    rows = comparison["rows"]
    lines: list[str] = []
    lines.append("# Finite-depth cross-validation against capytaine\n")
    lines.append(
        "**Scope**: code-to-code verification of "
        "`tensorbem.finite_depth_solver3d` in water of *finite* depth against "
        "capytaine's finite-depth Green function (the existing "
        "`docs/capytaine_crosscheck.md` is infinite depth only).\n"
    )
    lines.append(
        "**Plan**: `tensorbem.benchmarks.finite_depth_crossval_plan` — "
        f"submerged sphere `a = {plan.SPHERE_RADIUS} m`, centre "
        f"`z = -{plan.SPHERE_SUBMERGENCE} m`, grid "
        f"`h/a in {list(plan.DEPTH_RATIOS)}` x `k0 a in {list(plan.KA_GRID)}`, "
        "`nu` back-computed per cell from `k0 tanh(k0 h) = nu` so `k0 a` is "
        "exact at every depth.\n"
    )
    worst = comparison["worst_relative"]
    lines.append("## 1. Headline\n")
    lines.append(
        "| quantity | worst extrapolated deviation | pre-registered limit | owner limit |\n"
        "|---|---:|---:|---:|\n"
        f"| heave added mass | {_pct(worst.get('heave.added_mass'))} | "
        f"{_pct(plan.TOL_ADDED_MASS).lstrip('+')} | 15 % |\n"
        f"| surge added mass | {_pct(worst.get('surge.added_mass'))} | "
        f"{_pct(plan.TOL_ADDED_MASS).lstrip('+')} | 15 % |\n"
        f"| heave damping | {_pct(worst.get('heave.damping'))} | "
        f"{_pct(plan.TOL_DAMPING).lstrip('+')} | 15 % |\n"
        f"| surge damping | {_pct(worst.get('surge.damping'))} | "
        f"{_pct(plan.TOL_DAMPING).lstrip('+')} | 15 % |\n"
    )
    lines.append("\n## 2. Gates\n")
    lines.append("| gate | measured | limit | verdict |\n|---|---:|---:|---|\n")
    for gate in result["gates"]:
        measured = gate.get("worst", gate.get("min_gap", gate.get("min_order",
                              gate.get("max"))))
        lines.append(
            f"| `{gate['gate']}` | {_num(measured) if not isinstance(measured, bool) else measured} "
            f"| {gate['limit']} | {'PASS' if gate['passed'] else 'FAIL'} |\n"
        )
    lines.append("\n## 3. Per-cell comparison (Richardson-extrapolated)\n")
    lines.append(
        "| `h/a` | `k0 a` | `k0 h` | A33 TB | A33 capytaine | delta | "
        "B33 TB | B33 capytaine | delta |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for row in rows:
        q = row["quantities"]
        lines.append(
            f"| {row['h_over_a']:.0f} | {row['ka']:.2f} | {row['k0h']:.2f} | "
            f"{_num(q['heave.added_mass']['tensorbem'])} | "
            f"{_num(q['heave.added_mass']['capytaine'])} | "
            f"{_pct(q['heave.added_mass']['relative'])} | "
            f"{_num(q['heave.damping']['tensorbem'])} | "
            f"{_num(q['heave.damping']['capytaine'])} | "
            f"{_pct(q['heave.damping']['relative'])} |\n"
        )
    lines.append("\nObserved convergence orders (same three-level ladder):\n")
    lines.append(
        "| `h/a` | `k0 a` | p (A33, TB) | p (A33, capytaine) | "
        "p (B33, TB) | p (B33, capytaine) |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
    )
    for row in rows:
        q = row["quantities"]
        lines.append(
            f"| {row['h_over_a']:.0f} | {row['ka']:.2f} | "
            f"{_num(q['heave.added_mass']['tensorbem_order'], 2)} | "
            f"{_num(q['heave.added_mass']['capytaine_order'], 2)} | "
            f"{_num(q['heave.damping']['tensorbem_order'], 2)} | "
            f"{_num(q['heave.damping']['capytaine_order'], 2)} |\n"
        )
    lines.append("\n## 4. Negative control (the plan must see depth)\n")
    lines.append(
        "| `h/a` | `k0 a` | `k0 h` | (A33 finite − A33 deep)/A33 deep |\n"
        "|---:|---:|---:|---:|\n"
    )
    for row in rows:
        gap = row.get("negative_control_gap_added_mass")
        lines.append(
            f"| {row['h_over_a']:.0f} | {row['ka']:.2f} | {row['k0h']:.2f} | "
            f"{_pct(gap)} |\n"
        )
    lines.append(
        "\n## 5. Notes, limitations\n"
        "* Both codes discretise the sphere as an inscribed polyhedron, so the "
        "same geometric bias (panel area deficit ~1-2 % at the coarse levels) "
        "enters both sides; the extrapolation removes it to first order, and the "
        "reported order says how much of the residual is discretisation.\n"
        "* capytaine's two finite-depth formulations (`newer` vs `legacy`) agree "
        "to round-off on this geometry, so they do **not** provide an independent "
        "second opinion; the reference uncertainty is the panel ladder, which is "
        "why the ladder is reported per cell instead of a single tolerance.\n"
        "* Submerged closed body only: no waterline, so no irregular-frequency "
        "treatment is exercised here; the surface-piercing finite-depth case "
        "remains out of scope (`docs/finite_depth_solver3d.md`).\n"
        "* Damping involves the radiated-wave amplitude and converges more "
        "slowly than added mass on both sides; the per-cell orders above make "
        "that visible rather than hiding it behind one number.\n"
    )
    return "".join(lines)


def write_figures(result: dict[str, Any], out_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = result["comparison"]["rows"]
    produced: list[dict[str, str]] = []

    ratios = sorted({row["h_over_a"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for index, ratio in enumerate(ratios):
        subset = [row for row in rows if row["h_over_a"] == ratio]
        subset.sort(key=lambda row: row["k0h"])
        x = [row["k0h"] for row in subset]
        for ax, quantity, label in (
            (axes[0], "added_mass", "heave added mass $A_{33}$ [kg]"),
            (axes[1], "damping", "heave damping $B_{33}$ [kg/s]"),
        ):
            ax.plot(
                x,
                [row["quantities"][f"heave.{quantity}"]["tensorbem"] for row in subset],
                "o-", label=f"TensorBEM $h/a={ratio:.0f}$",
                color=f"C{index}",
            )
            ax.plot(
                x,
                [row["quantities"][f"heave.{quantity}"]["capytaine"] for row in subset],
                "x--", color=f"C{index}", alpha=0.8,
            )
            ax.set_xscale("log")
            ax.set_xlabel("$k_0 h$")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Finite-depth cross-validation: markers TensorBEM, crosses capytaine")
    fig.tight_layout()
    path = out_dir / FIGURE_FREQUENCY
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append({"path": str(path),
                     "caption": "Heave added mass and damping vs k0*h for every h/a; "
                                "TensorBEM (solid markers) against capytaine (crosses)."})

    quantities = ("heave.added_mass", "heave.damping")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, quantity in zip(axes, quantities):
        matrix = np.array(
            [
                [
                    row["quantities"][quantity]["relative"] * 100.0
                    for row in sorted(
                        [r for r in rows if r["h_over_a"] == ratio],
                        key=lambda r: r["ka"],
                    )
                ]
                for ratio in ratios
            ]
        )
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-5.0, vmax=5.0)
        ax.set_xticks(range(len(plan.KA_GRID)),
                      [f"{ka:.1f}" for ka in plan.KA_GRID])
        ax.set_yticks(range(len(ratios)), [f"{r:.0f}" for r in ratios])
        ax.set_xlabel("$k_0 a$")
        ax.set_ylabel("$h/a$")
        ax.set_title(quantity.replace(".", " "))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center",
                        fontsize=7)
        fig.colorbar(image, ax=ax, label="TensorBEM vs capytaine [%]")
    fig.tight_layout()
    path = out_dir / FIGURE_MAP
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append({"path": str(path),
                     "caption": "Extrapolated deviation map over the (h/a, k0 a) grid; "
                                "the pre-registered bands are 3 % (added mass) and 5 % (damping)."})

    ladder_cells = result["ladder"]["cells"]
    key = sorted(ladder_cells)[0]
    entry = ladder_cells[key]
    levels = sorted(entry["levels"].values(), key=lambda rec: rec["panels"])
    reference = {c["cell"]: c for c in result["reference"]["cells"]}[key]
    ref_levels = sorted(reference["levels"], key=lambda rec: rec["panels"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, quantity in zip(axes, ("added_mass", "damping")):
        ax.plot([1.0 / rec["panels"] for rec in levels],
                [rec["values"]["heave"][quantity] for rec in levels],
                "o-", label="TensorBEM")
        ax.plot([1.0 / rec["panels"] for rec in ref_levels],
                [rec["values"]["heave"][quantity] for rec in ref_levels],
                "s--", label="capytaine")
        cell_row = [row for row in rows if row["cell"] == key][0]
        for name, colour in (("tensorbem", "C0"), ("capytaine", "C1")):
            ax.axhline(
                cell_row["quantities"][f"heave.{quantity}"][name],
                color=colour, lw=0.8, ls=":",
            )
        ax.set_xscale("log")
        ax.set_xlabel("1 / panels")
        ax.set_ylabel(f"heave {quantity}")
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / FIGURE_CONVERGENCE
    fig.savefig(path, dpi=160)
    plt.close(fig)
    produced.append({"path": str(path),
                     "caption": "Panel ladder of both codes for one cell; dotted lines are the "
                                "Richardson extrapolations that the comparison uses."})
    return produced


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_study(
    ref_path: Path,
    root: Path,
    *,
    levels: Sequence[int] = plan.TENSORBEM_LEVELS,
    extra_level: int | None = plan.EXTRA_TENSORBEM_LEVEL,
    extra_cells: Sequence[tuple[float, float]] = plan.EXTRA_LEVEL_CELLS,
    control_level: int = 1,
    jobs: int = 1,
    write_doc: bool = False,
    doc_path: Path = DOC_PATH,
    figure_dir: Path | None = None,
) -> dict[str, Any]:
    reference = json.loads(Path(ref_path).read_text())
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ladder_path = root / LADDER_FILE
    ladder = build_ladder(
        list(plan.cells()),
        levels,
        cache_path=ladder_path,
        extra_level=extra_level,
        extra_cells=extra_cells,
        control_level=control_level,
        jobs=jobs,
    )
    comparison = compare(ladder, reference)
    gate_rows = gates(comparison, ladder, reference)
    result = {
        "case": {
            "body": "submerged sphere",
            "radius": plan.SPHERE_RADIUS,
            "submergence": plan.SPHERE_SUBMERGENCE,
            "depth_ratios": list(plan.DEPTH_RATIOS),
            "ka_grid": list(plan.KA_GRID),
            "density": plan.DENSITY,
            "gravity": plan.GRAVITY,
        },
        "reference_provenance": reference.get("provenance", {}),
        "tensor_environment": ladder.get("environment", {}),
        "comparison": comparison,
        "gates": gate_rows,
        "reference": reference,
        "ladder": ladder,
    }
    (root / RESULT_FILE).write_text(json.dumps(result, indent=1) + "\n")
    if write_doc:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(render_document(result))
        figures = write_figures(result, figure_dir or doc_path.parent)
        result["figures"] = figures
        (root / RESULT_FILE).write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=Path,
                        default=DEFAULT_ROOT / REFERENCE_FILE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--levels", type=str, default=",".join(map(str, plan.TENSORBEM_LEVELS)))
    parser.add_argument("--extra-level", type=int, default=plan.EXTRA_TENSORBEM_LEVEL)
    parser.add_argument("--control-level", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel worker processes for the TensorBEM ladder")
    parser.add_argument("--write-doc", action="store_true")
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    args = parser.parse_args(argv)
    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    result = run_study(
        args.ref,
        args.out_root,
        levels=levels,
        extra_level=args.extra_level,
        control_level=args.control_level,
        jobs=args.jobs,
        write_doc=args.write_doc,
        doc_path=args.doc,
    )
    worst = result["comparison"]["worst_relative"]
    print(json.dumps({"worst_relative": worst,
                      "gates": [{g["gate"]: g["passed"]} for g in result["gates"]]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())