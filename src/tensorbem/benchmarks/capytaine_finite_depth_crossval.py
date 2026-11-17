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

from tensorbem.benchmarks import crossval_decidability as decidability
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
TRUTH_FILE = "reference_truth_audit.json"
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
    t0 = time.perf_counter()
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
    return {
        "level": int(level),
        "panels": int(mesh.panel_count),
        "mode_count": int(getattr(result, "mode_count", 0)),
        "seconds": round(time.perf_counter() - t0, 3),
        "values": values,
    }


def _worker(cell: dict[str, float], level: int, control_level: int) -> dict[str, Any]:
    """One independent ladder task (level -1 = the deep-water negative control)."""
    if level < 0:
        return deep_water_values(control_level, cell)
    return tensorbem_values(level, cell)


def cells_for_shard(shard: int, shards: int) -> list[dict[str, float]]:
    """The plan cells that belong to ``shard`` of ``shards`` (index % shards).

    Sharding is an index filter over the plan's cell list, so it is reproducible
    without any bookkeeping and every shard is a disjoint subset of the grid.
    """
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError(f"invalid shard {shard} of {shards}")
    return [
        cell for index, cell in enumerate(plan.cells())
        if index % shards == shard
    ]


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
            f"  tb {key} level={level} panels={record.get('panels')} "
            f"seconds={record.get('seconds')}",
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


def merge_ladders(
    paths: Sequence[Path],
    *,
    out: Path | None = None,
    required_levels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Merge sharded ladder caches into one, refusing to lose or double-count work.

    A shard is one ``(cell, level)`` subset of the grid, computed on its own node
    with its own cache.  Two shards that both computed the same rung must agree
    (they are the same deterministic solve), and every requested rung must be
    present afterwards: a study whose ladder silently lost a shard would still
    produce a comparison table, and that table would be wrong.  Both conditions
    raise here instead.
    """
    merged: dict[str, Any] = {"cells": {}, "environment": {}}
    environments: list[dict[str, Any]] = []
    cell_fields = ("h_over_a", "ka", "depth", "k0", "k0h", "nu")
    for path in paths:
        payload = json.loads(Path(path).read_text())
        environments.append(payload.get("environment", {}))
        for key, entry in payload.get("cells", {}).items():
            target = merged["cells"].setdefault(
                key,
                {
                    **{field: entry[field] for field in cell_fields},
                    "cell": entry.get("cell", key),
                    "levels": {},
                    "negative_control": None,
                },
            )
            for field in cell_fields:
                if float(target[field]) != float(entry[field]):
                    raise ValueError(f"{key}: {field} disagrees between shards")
            for level, record in entry.get("levels", {}).items():
                previous = target["levels"].get(level)
                if previous is not None:
                    if int(previous["panels"]) != int(record["panels"]):
                        raise ValueError(
                            f"{key} level {level}: panel count disagrees between shards"
                        )
                    for dof in previous["values"]:
                        for quantity in previous["values"][dof]:
                            a = previous["values"][dof][quantity]
                            b = record["values"][dof][quantity]
                            if abs(a - b) > 1e-9 * max(abs(a), 1e-12):
                                raise ValueError(
                                    f"{key} level {level} {dof}.{quantity}: "
                                    "two shards disagree on the same solve"
                                )
                target["levels"][level] = record
            control = entry.get("negative_control")
            if control is not None:
                previous = target["negative_control"]
                if previous is not None and int(previous["panels"]) != int(control["panels"]):
                    raise ValueError(f"{key}: negative control disagrees between shards")
                target["negative_control"] = control
    if required_levels is not None:
        wanted = {str(int(level)) for level in required_levels}
        for key, entry in merged["cells"].items():
            missing = wanted - set(entry["levels"])
            if missing:
                raise ValueError(f"{key}: missing ladder rungs {sorted(missing)}")
    if environments:
        merged["environment"] = {
            **environments[0],
            "merged_from": [str(path) for path in paths],
            "shard_environments": environments,
        }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=1) + "\n")
    return merged


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
    panels: Sequence[float],
    values: Sequence[float],
    *,
    min_order: float = plan.MIN_OBSERVED_ORDER,
) -> dict[str, Any]:
    """Extrapolate a panel ladder to ``N -> infinity``.

    Uses the three finest levels; the order is *measured* from the same three
    levels unless the ladder is non-monotone, in which case the finest value is
    returned and ``extrapolated`` is set to ``False``.

    The pre-registered rule of :data:`...finite_depth_crossval_plan.MIN_OBSERVED_ORDER`
    is enforced here: a ladder whose *measured* order is below ``min_order`` is
    **not** extrapolated, because the Richardson step is then a fit outside the
    asymptotic range rather than a measurement of the limit.  Such a ladder
    keeps its finest rung as the reported value, records the measured order, and
    additionally reports ``blocked_value``/``blocked_shift`` -- what the
    forbidden fit *would* have done -- so the size of the rejected correction
    stays visible (a shift several times the last measured rung change is the
    signature of a fit blowing up, see ``docs/xval_bem_finite_depth_crossval_plan.md``).
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
        "blocked_by_order": False,
        "blocked_value": None,
        "blocked_shift": None,
        "last_rung_change": None,
    }
    if len(values) >= 2:
        v_prev, v_last = values[-2], values[-1]
        out["last_rung_change"] = (
            float(abs(v_last - v_prev) / abs(v_last)) if v_last else float("inf")
        )
    if len(values) >= 3:
        p = _order_from_three(panels[-3:], values[-3:])
        if p is not None:
            n2, n3 = panels[-2], panels[-1]
            v2, v3 = values[-2], values[-1]
            c = (v3 - v2) / (n3 ** (-p) - n2 ** (-p))
            extrap = v3 - c * n3 ** (-p)
            shift = float((v3 - extrap) / extrap if extrap else float("inf"))
            out["order"] = p
            if p >= min_order:
                out.update(
                    {
                        "extrapolated": True,
                        "value": float(extrap),
                        "extrapolation_shift": shift,
                    }
                )
            else:
                # measured order is below the pre-registered floor: the fit is
                # outside its asymptotic range, so it is reported, not applied
                out.update(
                    {
                        "extrapolated": False,
                        "value": finest,
                        "extrapolation_shift": 0.0,
                        "blocked_by_order": True,
                        "blocked_value": float(extrap),
                        "blocked_shift": shift,
                    }
                )
    return out


def extrapolation_band(
    panels: Sequence[float], values: Sequence[float]
) -> dict[str, Any]:
    """Measure the extrapolation's own uncertainty on a four-rung ladder.

    ``richardson`` fits the observed order to the three *finest* rungs.  With a
    fourth (finest) rung available the same fit can be repeated after dropping
    the *coarsest* rung; the two extrapolants should agree if the ladder is in
    its asymptotic range, and their relative spread is then a measured, per-cell
    uncertainty of the extrapolated value.

    Without four rungs the band is ``None`` -- unknown, not zero.  A two-rung
    ladder is not evidence that the extrapolation converged.
    """
    fine = richardson(panels, values)
    order = list(np.argsort([float(p) for p in panels]))
    panels_sorted = [float(panels[i]) for i in order]
    values_sorted = [float(values[i]) for i in order]
    out: dict[str, Any] = {
        "fine": fine,
        "coarse_value": None,
        "coarse_order": None,
        "band": None,
        "band_basis": None,
    }
    if len(panels_sorted) < 4 or not fine["extrapolated"]:
        return out
    coarse = richardson(panels_sorted[:-1], values_sorted[:-1])
    out["coarse_value"] = coarse["value"]
    out["coarse_order"] = coarse["order"]
    if coarse["extrapolated"] and fine["value"]:
        out["band"] = float(
            abs(fine["value"] - coarse["value"]) / abs(fine["value"])
        )
        out["band_basis"] = "drop-coarsest-rung"
    return out


def _extrapolated_entry(levels: dict[str, Any], dof: str, quantity: str) -> dict[str, Any]:
    ordered = sorted(levels.values(), key=lambda rec: rec["panels"])
    panels = [rec["panels"] for rec in ordered]
    values = [rec["values"][dof][quantity] for rec in ordered]
    out = dict(richardson(panels, values))
    out.update(extrapolation_band(panels, values))
    return out


def load_reference_truth(root: Path) -> dict[str, dict[str, float]]:
    """Load the measured reference error produced by the truth audit.

    The audit (:mod:`scripts.finite_depth_reference_truth_audit`) measures
    capytaine's own error against the closed-form multipole solution on the
    grid cells where the seabed image is negligible.  A missing artifact is not
    an error: the comparison then simply keeps its model bounds.
    """
    path = Path(root) / TRUTH_FILE
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    return {
        row["cell"]: {
            slot: float(quantity["finest_error_abs"])
            for slot, quantity in row["quantities"].items()
        }
        for row in report.get("cells", [])
    }


def _refinement_ratio(panels: Sequence[float]) -> float | None:
    """Panel ratio between the two finest rungs of a ladder."""
    ordered = sorted(float(p) for p in panels)
    if len(ordered) < 2 or ordered[-2] <= 0.0:
        return None
    return float(ordered[-1] / ordered[-2])


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
    reference_truth: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Cell-by-cell extrapolated comparison plus gates.

    ``reference_truth`` (``cell -> quantity slot -> exact relative error``, see
    :func:`load_reference_truth`) marks the cells on which capytaine's own error
    has been *measured* against the closed-form multipole anchor.  On those cells
    the reference bound is replaced by that measurement -- a measurement beats a
    model bound in both directions -- so the deviation there is judged with an
    honest reference budget instead of an optimistic one.
    """
    ref_cells = {cell["cell"]: cell for cell in reference["cells"]}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    worst: dict[str, float] = {}
    without_ladder: list[str] = []
    decisions: dict[str, list[dict[str, Any]]] = {
        "added_mass": [],
        "damping": [],
        "owner_limit": [],
    }
    truth_cells = reference_truth or {}
    truth_decisions: dict[str, list[dict[str, Any]]] = {
        "added_mass": [],
        "damping": [],
        "owner_limit": [],
    }
    truth_covered: list[str] = []
    for key, entry in sorted(ladder["cells"].items()):
        ref = ref_cells.get(key)
        if ref is None:
            continue
        if not entry.get("levels"):
            # A sharded ladder starts life as one entry per planned cell with no
            # rungs: judging those cells raised IndexError, which made a PARTIAL
            # study impossible to re-judge from its own frozen artifacts.
            # Missing is not zero -- the cell is recorded, not compared.
            without_ladder.append(key)
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
                # Admissible values under the pre-registered rule: a ladder whose
                # measured order is below MIN_OBSERVED_ORDER contributes its
                # finest rung (``richardson`` already reports ``value == finest``
                # in that case, and keeps the forbidden fit in ``blocked_value``).
                both_extrapolated = bool(
                    tb["extrapolated"] and cp["extrapolated"]
                )
                basis = "extrapolated" if both_extrapolated else "finest_level"
                base = cp["value"]
                rel = (tb["value"] - base) / abs(base) if base else float("inf")
                raw_rel = (
                    (tb["finest"] - cp["finest"]) / abs(cp["finest"])
                    if cp["finest"]
                    else float("inf")
                )

                def _attempt(entry: dict[str, Any]) -> float:
                    """Value the ungated fit would have produced (provenance only)."""
                    blocked = entry.get("blocked_value")
                    return float(blocked) if blocked is not None else float(entry["value"])

                attempt_base = _attempt(cp)
                rel_attempt = (
                    (_attempt(tb) - attempt_base) / abs(attempt_base)
                    if attempt_base
                    else float("inf")
                )
                row["quantities"][f"{dof}.{quantity}"] = {
                    "tensorbem": tb["value"],
                    "capytaine": base,
                    "relative_basis": basis,
                    "tensorbem_panels": tb["panels"],
                    "tensorbem_order": tb["order"],
                    "tensorbem_extrapolated": tb["extrapolated"],
                    "tensorbem_shift": tb["extrapolation_shift"],
                    "tensorbem_blocked_by_order": tb["blocked_by_order"],
                    "capytaine_order": cp["order"],
                    "capytaine_shift": cp["extrapolation_shift"],
                    "capytaine_extrapolated": cp["extrapolated"],
                    "capytaine_blocked_by_order": cp["blocked_by_order"],
                    "capytaine_blocked_shift": cp["blocked_shift"],
                    "capytaine_truncation_bound": cp["last_rung_change"],
                    "relative": float(rel),
                    "relative_if_extrapolated": float(rel_attempt),
                    "relative_finest_levels": float(raw_rel),
                    "tensorbem_extrapolation_band": tb["band"],
                    "tensorbem_extrapolation_band_basis": tb["band_basis"],
                    "tensorbem_value_without_finest_rung": tb["coarse_value"],
                    "tensorbem_order_without_finest_rung": tb["coarse_order"],
                }
                _ref_panels = [rec["panels"] for rec in ref["levels"]]
                _u_ref = decidability.ladder_uncertainty(
                    last_rung_change=cp["last_rung_change"],
                    order=cp["order"],
                    refinement=_refinement_ratio(_ref_panels),
                    min_order=plan.MIN_OBSERVED_ORDER,
                    safety_factor=decidability.GCI_SAFETY_FACTOR,
                )
                _u_tb = decidability.ladder_uncertainty(
                    last_rung_change=tb["last_rung_change"],
                    order=tb["order"],
                    refinement=_refinement_ratio(tb["panels"]),
                    min_order=plan.MIN_OBSERVED_ORDER,
                    measured_band=tb["band"],
                    safety_factor=decidability.GCI_SAFETY_FACTOR,
                )
                _measured_ref = (truth_cells.get(key) or {}).get(
                    f"{dof}.{quantity}"
                )
                if _measured_ref is not None:
                    _u_ref = decidability.ladder_uncertainty(
                        last_rung_change=None,
                        measured_error=_measured_ref,
                    )
                    row["quantities"][f"{dof}.{quantity}"][
                        "reference_error_measured"
                    ] = float(_measured_ref)
                row["quantities"][f"{dof}.{quantity}"]["uncertainty"] = {
                    "reference": _u_ref,
                    "solver": _u_tb,
                }
                row["quantities"][f"{dof}.{quantity}"]["decidability"] = (
                    decidability.decide(
                        deviation=rel,
                        tolerance=tols[quantity],
                        reference=_u_ref,
                        solver=_u_tb,
                    )
                )
                row["quantities"][f"{dof}.{quantity}"]["decidability_owner_limit"] = (
                    decidability.decide(
                        deviation=rel,
                        tolerance=plan.OWNER_LIMIT,
                        reference=_u_ref,
                        solver=_u_tb,
                    )
                )
                if _measured_ref is not None:
                    if key not in truth_covered:
                        truth_covered.append(key)
                    _dec_truth = decidability.decide(
                        deviation=rel,
                        tolerance=tols[quantity],
                        reference=_u_ref,
                        solver=_u_tb,
                    )
                    _dec_truth_owner = decidability.decide(
                        deviation=rel,
                        tolerance=plan.OWNER_LIMIT,
                        reference=_u_ref,
                        solver=_u_tb,
                    )
                    row["quantities"][f"{dof}.{quantity}"][
                        "decidability_measured_reference"
                    ] = _dec_truth
                    row["quantities"][f"{dof}.{quantity}"][
                        "decidability_measured_reference_owner_limit"
                    ] = _dec_truth_owner
                    truth_decisions[quantity].append(_dec_truth)
                    truth_decisions["owner_limit"].append(_dec_truth_owner)
                decisions[quantity].append(
                    row["quantities"][f"{dof}.{quantity}"]["decidability"]
                )
                decisions["owner_limit"].append(
                    row["quantities"][f"{dof}.{quantity}"]["decidability_owner_limit"]
                )
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
        "cells_without_ladder": without_ladder,
        "tolerances": {"added_mass": tol_added_mass, "damping": tol_damping,
                       "owner_limit": plan.OWNER_LIMIT},
        "decidability": {
            "added_mass": decidability.summarise(
                decisions["added_mass"], tolerance=tol_added_mass
            ),
            "damping": decidability.summarise(
                decisions["damping"], tolerance=tol_damping
            ),
            "owner_limit": decidability.summarise(
                decisions["owner_limit"], tolerance=plan.OWNER_LIMIT
            ),
            "criteria": {
                "budget_share": decidability.BUDGET_SHARE,
                "safety_factor": decidability.GCI_SAFETY_FACTOR,
                "module": "tensorbem.benchmarks.crossval_decidability",
            },
            "measured_reference": {
                "cells": truth_covered,
                "added_mass": decidability.summarise(
                    truth_decisions["added_mass"], tolerance=tol_added_mass
                ),
                "damping": decidability.summarise(
                    truth_decisions["damping"], tolerance=tol_damping
                ),
                "owner_limit": decidability.summarise(
                    truth_decisions["owner_limit"], tolerance=plan.OWNER_LIMIT
                ),
            },
        },
    }


def q_band(quantities: dict[str, Any], dof: str, quantity: str) -> float | None:
    """Extrapolation band of one ``dof``/``quantity`` entry, if measured."""
    entry = quantities.get(f"{dof}.{quantity}") or {}
    return entry.get("tensorbem_extrapolation_band")


def gates(comparison: dict[str, Any], ladder: dict[str, Any],
          reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Named accept/reject gates for the published study."""
    out: list[dict[str, Any]] = []
    worst = comparison["worst_relative"]

    def _agreement(slots: tuple[str, ...], limit: float) -> dict[str, Any]:
        """Worst deviation over ``slots``; unknown (no judged cell) is not zero.

        A comparison with no judged cell used to read as
        ``max(0.0, 0.0) = 0.0 <= limit`` -> PASS, i.e. an empty study published
        as an agreement.  Missing stays missing here.
        """
        seen = [worst[slot] for slot in slots if slot in worst]
        measured = max(seen) if seen else None
        return {
            "worst": measured,
            "limit": limit,
            "passed": measured is not None and measured <= limit,
        }

    out.append(
        {
            "gate": "added_mass_agreement",
            **_agreement(("heave.added_mass", "surge.added_mass"),
                         plan.TOL_ADDED_MASS),
        }
    )
    out.append(
        {
            "gate": "damping_agreement",
            **_agreement(("heave.damping", "surge.damping"), plan.TOL_DAMPING),
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
    truncation: dict[str, float] = {}
    for row in comparison["rows"]:
        for slot, q in row["quantities"].items():
            bound = q.get("capytaine_truncation_bound")
            if bound is None:
                continue
            quantity = slot.split(".")[-1]
            truncation[quantity] = max(truncation.get(quantity, 0.0), float(bound))
    trunc_limits = {
        quantity: plan.extrapolation_limit(quantity) for quantity in truncation
    }
    out.append(
        {
            "gate": "reference_truncation_within_share",
            "worst": max(truncation.values()) if truncation else None,
            "worst_quantity": (
                max(truncation, key=lambda k: truncation[k]) if truncation else None
            ),
            "limit": max(trunc_limits.values()) if trunc_limits else None,
            "per_quantity": truncation,
            "note": "last-rung change of the reference ladder, a conservative "
                    "bound on the reference's own remaining discretisation "
                    "error; limit is 1/3 of the quantity's tolerance "
                    "(plan.EXTRAPOLATION_BAND_FRACTION)",
            "passed": bool(truncation) and all(
                truncation[q] <= trunc_limits[q] for q in truncation
            ),
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
    # Extrapolation self-consistency: on the cells that carry the confirmation
    # rung, dropping the coarsest rung must not move the extrapolant by more
    # than its share of the tolerance.  Without those cells the check is unknown
    # (``worst`` is None), never passed.
    band_ratios: dict[str, float] = {}
    for quantity in ("added_mass", "damping"):
        limit = plan.extrapolation_limit(quantity)
        bands = [
            band
            for row in comparison["rows"]
            for dof in MODE_NAMES
            if (band := q_band(row["quantities"], dof, quantity)) is not None
        ]
        if bands:
            band_ratios[quantity] = max(bands) / limit
    out.append(
        {
            "gate": "extrapolation_self_consistent",
            "worst": max(band_ratios.values()) if band_ratios else None,
            "limit": 1.0,
            "limits": {key: plan.extrapolation_limit(key)
                       for key in ("added_mass", "damping")},
            "note": "worst band as a fraction of its admissible share "
                    f"({plan.EXTRAPOLATION_BAND_FRACTION:.2f} x tolerance)",
            "passed": bool(band_ratios) and max(band_ratios.values()) <= 1.0,
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


def _order_coverage(rows: Sequence[dict[str, Any]]) -> str:
    """How many reference ladders were asymptotic (``s/n``), from the result."""
    entries = [
        quantity
        for row in rows
        for quantity in row["quantities"].values()
    ]
    measured = sum(1 for entry in entries if entry.get("capytaine_order"))
    return f"{measured}/{len(entries)}"


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
    provenance = result.get("reference", {}).get("provenance", {})
    status = str(result.get("case", {}).get("execution_status", "")) or "n/a"
    lines.append(
        "**Reference**: `scripts/capytaine_finite_depth_ref.py` "
        f"({provenance.get('tool', 'capytaine')} {provenance.get('version', 'n/a')}, "
        f"`finite_depth_method={provenance.get('finite_depth_method', 'n/a')}`, "
        f"ladder {[list(r) for r in provenance.get('resolutions', [])]} panels, "
        f"generated {provenance.get('generated_utc', 'n/a')}).\n"
        f"**Execution status**: `{status}`; "
        f"**confirmation rung**: `{result.get('case', {}).get('confirmation_status', 'n/a')}`; "
        "every number below is written by the study driver from the frozen "
        "artifacts, none is transcribed.\n"
    )
    worst = comparison["worst_relative"]
    lines.append("## 1. Headline\n")
    lines.append(
        "| quantity | worst deviation on the pre-registered basis | "
        "pre-registered limit | owner limit |\n"
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
        if measured is None:
            # boolean gates carry no measurement: show the symbol instead of a
            # stray "n/a", otherwise the verdict column says FAIL/PASS about a
            # number the page never printed
            shown = "yes" if gate["passed"] else "no"
        elif isinstance(measured, bool):
            shown = str(measured)
        else:
            shown = _num(measured)
        lines.append(
            f"| `{gate['gate']}` | {shown} "
            f"| {gate['limit']} | {'PASS' if gate['passed'] else 'FAIL'} |\n"
        )
    lines.append(
        "\n## 3. Per-cell comparison (Richardson-extrapolated, gated by the "
        "pre-registered order floor)\n"
    )
    lines.append(
        "| `h/a` | `k0 a` | `k0 h` | A33 TB | A33 capytaine | delta | "
        "B33 TB | B33 capytaine | delta | A11 TB | A11 capytaine | delta | "
        "B11 TB | B11 capytaine | delta |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---:|---:|\n"
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
            f"{_pct(q['heave.damping']['relative'])} | "
            f"{_num(q['surge.added_mass']['tensorbem'])} | "
            f"{_num(q['surge.added_mass']['capytaine'])} | "
            f"{_pct(q['surge.added_mass']['relative'])} | "
            f"{_num(q['surge.damping']['tensorbem'])} | "
            f"{_num(q['surge.damping']['capytaine'])} | "
            f"{_pct(q['surge.damping']['relative'])} |\n"
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
    lines.append(
        "\n## 3b. Extrapolation uncertainty (cells with the confirmation rung)\n"
    )
    lines.append(
        "The extrapolant is recomputed after dropping the coarsest rung; the "
        "spread between the two is the measured uncertainty of the extrapolated "
        "value.  A cell listed as `n/a` has only three rungs, so its "
        "extrapolation uncertainty is unknown (not zero).\n\n"
        "| `h/a` | `k0 a` | band A33 | band B33 | band A11 | band B11 | "
        "admissible (A / B) |\n"
        "|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for row in rows:
        q = row["quantities"]
        cells = [
            _pct(q[f"{dof}.{quantity}"]["tensorbem_extrapolation_band"])
            for dof, quantity in (("heave", "added_mass"), ("heave", "damping"),
                                  ("surge", "added_mass"), ("surge", "damping"))
        ]
        lines.append(
            f"| {row['h_over_a']:.0f} | {row['ka']:.2f} | "
            + " | ".join(cells)
            + f" | {_pct(plan.extrapolation_limit('added_mass')).lstrip('+')} / "
            f"{_pct(plan.extrapolation_limit('damping')).lstrip('+')} |\n"
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
        "* The extrapolation band (section 3b) is a *self-consistency* measure, "
        "not a truth bound: capytaine and TensorBEM share the inscribed-polyhedron "
        "geometry bias, so a common geometric error would sit inside every band "
        "and stay invisible.  What the band rules out is a discretisation error "
        "of the TensorBEM ladder being mistaken for a solver disagreement.\n"
        f"* Measured order available on {_order_coverage(rows)} of the judged "
        "code/cell entries; where the reference ladder is not yet asymptotic "
        "(`n/a` order) the comparison uses the finest reference level, so that "
        "entry's benchmark is the least certain number in this study.\n"
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
    judged = {row["cell"] for row in rows}
    ready = sorted(
        key for key, cell in ladder_cells.items()
        if cell.get("levels") and key in judged
    )
    if not ready:
        # A ladder with no rung anywhere cannot show a convergence trace; the
        # study is then judged on its (empty) comparison alone, not on a figure
        # built from an empty list.
        return produced
    key = ready[0]
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
def execution_status(
    ladder: dict[str, Any],
    levels: Sequence[int],
    extra_level: int | None = None,
    extra_cells: Sequence[tuple[float, float]] = plan.EXTRA_LEVEL_CELLS,
) -> str:
    """Coverage of the *main* cross-code comparison (all planned cells x main levels).

    The status used to be prose written by hand into ``results.json``, so a study
    that had only laddered one cell of twelve could be published with a status
    string that no longer matched the artifact.  It is computed here from the
    ladder itself, and ``tests/test_capytaine_finite_depth_crossval.py`` reads it
    to decide whether a gate may be judged at all.

    The confirmation rung (``extra_level``, the 1280-panel level on the four
    ``EXTRA_LEVEL_CELLS``) is a *stricter* check of the Richardson extrapolation
    band, not a requirement for the cross-code comparison: the comparison is
    decidable from the three pre-registered levels.  Its coverage is therefore
    reported by :func:`confirmation_status` and never downgrades the main verdict.
    """
    planned = [plan.cell_id(c["h_over_a"], c["ka"]) for c in plan.cells()]
    done = [key for key in planned if key in ladder.get("cells", {})]
    wanted = {str(int(level)) for level in levels}
    thin = [
        key for key in done
        if not wanted <= set(ladder["cells"][key].get("levels", {}))
    ]
    parts = []
    if len(done) < len(planned):
        parts.append(f"{len(done)} of {len(planned)} planned cells")
    if thin:
        parts.append(f"{len(thin)} covered cells missing rungs {sorted(wanted)}")
    return "COMPLETE" if not parts else "PARTIAL: " + "; ".join(parts)


def confirmation_status(
    ladder: dict[str, Any],
    extra_level: int | None = None,
    extra_cells: Sequence[tuple[float, float]] = plan.EXTRA_LEVEL_CELLS,
) -> str:
    """Coverage of the confirmation rung (the 4th, 1280-panel level).

    The confirmation rung exists to measure the Richardson extrapolation band
    (the ``extrapolation_self_consistent`` gate) by re-extrapolating after
    dropping the coarsest rung.  Where it is missing the band is *unknown*, not
    zero, so the gate reports ``worst: null`` and does not pass -- but the
    cross-code comparison itself is unaffected and is not marked PARTIAL for it.
    """
    if extra_level is None:
        return "n/a (no confirmation rung planned)"
    wanted_extra = {str(int(extra_level))}
    extra_keys = {
        plan.cell_id(float(ratio), float(ka)) for ratio, ka in extra_cells
    }
    missing = [
        key for key in sorted(extra_keys)
        if key in ladder.get("cells", {})
        and not wanted_extra <= set(ladder["cells"][key].get("levels", {}))
    ]
    return (
        "COMPLETE" if not missing
        else f"PARTIAL: {len(missing)} confirmation cells missing the "
             f"{int(extra_level)}-subdivision rung"
    )


def run_study(
    ref_path: Path,
    root: Path,
    *,
    levels: Sequence[int] = plan.TENSORBEM_LEVELS,
    extra_level: int | None = plan.EXTRA_TENSORBEM_LEVEL,
    extra_cells: Sequence[tuple[float, float]] = plan.EXTRA_LEVEL_CELLS,
    control_level: int = 1,
    jobs: int = 1,
    cache_path: Path | None = None,
    cell_ids: Sequence[str] | None = None,
    write_doc: bool = False,
    doc_path: Path = DOC_PATH,
    figure_dir: Path | None = None,
) -> dict[str, Any]:
    reference = json.loads(Path(ref_path).read_text())
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ladder_path = Path(cache_path) if cache_path is not None else root / LADDER_FILE
    # Only ladder the cells the frozen reference actually covers: a cell with no
    # reference can never enter the comparison, so computing its ladder is pure
    # cost (and on a partially produced reference it is the difference between a
    # minutes-long and an hours-long study).  ``cell_ids`` narrows the set one
    # shard at a time; the merged cache is what the verdict is computed on.
    covered = {cell["cell"] for cell in reference.get("cells", [])}
    cells = [
        cell
        for cell in plan.cells()
        if plan.cell_id(cell["h_over_a"], cell["ka"]) in covered
    ] or list(plan.cells())
    if cell_ids is not None:
        wanted_ids = set(cell_ids)
        unknown = wanted_ids - {plan.cell_id(c["h_over_a"], c["ka"]) for c in cells}
        if unknown:
            raise ValueError(f"cell ids not covered by the reference: {sorted(unknown)}")
        cells = [c for c in cells if plan.cell_id(c["h_over_a"], c["ka"]) in wanted_ids]
    ladder = build_ladder(
        cells,
        levels,
        cache_path=ladder_path,
        extra_level=extra_level,
        extra_cells=extra_cells,
        control_level=control_level,
        jobs=jobs,
    )
    comparison = compare(
        ladder, reference, reference_truth=load_reference_truth(root)
    )
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
            "levels": [int(level) for level in levels],
            "execution_status": execution_status(ladder, levels, extra_level),
            "confirmation_status": confirmation_status(ladder, extra_level),
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
    parser.add_argument("--merge-caches", type=Path, nargs="+", default=None,
                        help="merge these shard ladder caches into --cache instead of "
                             "solving (the shard collector)")
    parser.add_argument("--shard", type=int, default=0,
                        help="ladder-only: this shard index over the plan's cells")
    parser.add_argument("--shards", type=int, default=1,
                        help="ladder-only: total shards (assignment: index %% shards)")
    parser.add_argument("--no-extra-level", action="store_true",
                        help="skip the 1280-panel confirmation rung (budget runs); "
                             "the frozen study then says so instead of pretending")
    parser.add_argument("--control-level", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel worker processes for the TensorBEM ladder")
    parser.add_argument("--cache", type=Path, default=None,
                        help="ladder cache path (default: <out-root>/%s); a shard "
                             "writes its own cache, the merged cache feeds the verdict"
                             % LADDER_FILE)
    parser.add_argument("--cell-ids", type=str, default=None,
                        help="comma-separated subset of plan cell ids to ladder")
    parser.add_argument("--ladder-only", action="store_true",
                        help="produce/resume the ladder cache and stop: no comparison, "
                             "no gates, no page (the shard entry point)")
    parser.add_argument("--write-doc", action="store_true")
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    args = parser.parse_args(argv)
    levels = [int(v) for v in args.levels.split(",") if v.strip()]
    extra_level = None if args.no_extra_level else args.extra_level
    if args.merge_caches:
        merged = merge_ladders(
            args.merge_caches,
            out=args.cache,
            required_levels=levels,
        )
        print(json.dumps({"merged_cells": sorted(merged["cells"]),
                          "cache": str(args.cache)}, indent=1))
        return 0
    cell_ids = (
        [c.strip() for c in args.cell_ids.split(",") if c.strip()]
        if args.cell_ids
        else None
    )
    if args.ladder_only:
        if cell_ids is not None:
            # Explicit ids beat both filters: the ladder and the reference are
            # independent computations, so a shard must not have to wait for (or
            # be reduced by) the other side's progress.
            known = {plan.cell_id(c["h_over_a"], c["ka"]) for c in plan.cells()}
            unknown = set(cell_ids) - known
            if unknown:
                raise SystemExit(f"cell ids not in the plan: {sorted(unknown)}")
            wanted = set(cell_ids)
            cells = [
                c for c in plan.cells()
                if plan.cell_id(c["h_over_a"], c["ka"]) in wanted
            ]
        elif args.shards > 1:
            # A shard is the plan's cell list filtered by index: without this
            # branch the reference filter below would silently hand every shard
            # the whole grid, turning N parallel shards into N full duplicates.
            if not 0 <= args.shard < args.shards:
                parser.error("--shard must satisfy 0 <= shard < shards")
            cells = cells_for_shard(args.shard, args.shards)
        else:
            reference = json.loads(Path(args.ref).read_text())
            covered = {cell["cell"] for cell in reference.get("cells", [])}
            cells = [
                cell for cell in plan.cells()
                if plan.cell_id(cell["h_over_a"], cell["ka"]) in covered
            ] or list(plan.cells())
        ladder = build_ladder(
            cells,
            levels,
            cache_path=args.cache or (args.out_root / LADDER_FILE),
            extra_level=extra_level,
            extra_cells=plan.EXTRA_LEVEL_CELLS,
            control_level=args.control_level,
            jobs=args.jobs,
        )
        print(json.dumps({"ladder_cells": sorted(ladder["cells"]),
                          "cache": str(args.cache or (args.out_root / LADDER_FILE))},
                         indent=1))
        return 0
    result = run_study(
        args.ref,
        args.out_root,
        levels=levels,
        extra_level=extra_level,
        control_level=args.control_level,
        jobs=args.jobs,
        cache_path=args.cache,
        cell_ids=cell_ids,
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