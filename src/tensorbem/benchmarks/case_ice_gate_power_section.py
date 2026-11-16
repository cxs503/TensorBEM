"""Two-dimensional heave added-mass anchor for a surface-piercing rectangular section.

Why this module exists
----------------------
``GAP-ICE-GATE-POWER`` requires every REAL criterion of the ice/structure
verification package to carry information: (a) a *source* for the baseline,
(b) a threshold bound to an observed convergence order or a measured detection
rate, and (c) at least one *rejected* negative control.  The hydrodynamic leg of
that chain is the high-frequency heave added mass of the ice-going vessel
section ``B = 20 m``, ``T = 6 m``, ``rho = 1025 kg/m^3``, declared as
``A_inf = 1.00 * m`` with ``m = rho L B T C_b`` (``A_inf / L = 86.1 t/m``).

Two adversarial audits of that leg disagree by a factor of two:

* report #19234 quotes an "independent potential-flow" 2-D heave added mass of
  ``228.6 t/m`` and calls it ``A_inf`` (the declared value is then 166 % low);
* report #19477 argues the 166 % is a 2-D/3-D *caliber* mismatch and that the
  comparison passes the 15 % gate once converted.

Neither number carries its own convergence order, its spectral limit, or a
rejected negative control, and the limit is exactly where this quantity is
fragile.  The linearised free-surface condition

    d(phi)/dz = (omega**2 / g) phi        at z = 0

degenerates at the two ends of the band,

============================  =========================  ===========================
limit                         free-surface condition     meaning
============================  =========================  ===========================
``omega -> infinity``         ``phi = 0``                pressure release (``A_inf``)
``omega -> 0``                ``d(phi)/dz = 0``          rigid wall
============================  =========================  ===========================

and **only the first is an admissible exterior problem in two dimensions**.  A
surface-piercing section heaving under a rigid wall has to push fluid out along
the free surface to infinity; the doubled-body data then satisfy
``d(phi)/dn = 0`` on the mirror line but carry a non-vanishing net flux
``2 V B``, so no decaying exterior potential exists and the "zero-frequency
added mass" is the finite part of a logarithmically growing field, not a
physical limit.  The module computes both branches and *reports* that
incompatibility (:func:`mirror_diagnostics`) instead of quietly quoting one of
them as "the" 2-D value.

Method
------
One infinite-fluid 2-D boundary-integral solve
(:func:`tensorbem.solver.assemble_matrices`) on the *doubled* body -- the wetted
section mirrored across ``z = 0`` -- with the load integral taken over the
physical (lower) panels only, which halves the kinetic energy exactly.  The
parity convention is the 2-D analogue of
:class:`tensorbem.added_mass_limit.ADDED_MASS_LIMITS`: the image panels carry the
same normal-velocity numbers, and the ``phi = 0`` (``A_inf``) branch is the one
with a *uniform* normal velocity over the closed doubled body.

Verification ladder (closed-form baselines, see
``tests/test_ice_gate_power_section.py``)
--------------------------------------------------------------------------
1. full circle, heave: ``rho*pi*a**2`` (exact);
2. full ellipse, heave: ``rho*pi*a**2`` for the semi-axis across the motion
   (exact conformal-mapping result);
3. half-immersed circle, ``A_inf``: ``0.5*rho*pi*a**2`` (exact -- the heave
   potential of the full circle, ``phi ~ y/r**2``, is odd in ``y``, so its
   restriction to ``y > 0`` already satisfies ``phi = 0`` on the mirror line);
   the same argument is what pins ``(1/3)*pi*rho*a**3`` in the 3-D module, whose
   ``infinite_frequency`` branch reproduces it to 3.6 % at ``nlat=6, nlon=18``;
4. mirror-line residuals: ``max|phi|`` on ``z = 0`` for the ``A_inf`` branch and
   ``max|d(phi)/dz|`` for the rigid-wall branch, plus the net-flux compatibility
   residual that tells the two branches apart physically;
5. mesh ladder: observed order ``p`` and Richardson limit, so a 3 % threshold is
   tied to a measured discretisation error rather than asserted.

Informativeness machinery
-------------------------
``minimum_detectable_defect`` turns a tolerance into the smallest section
defect the gate can see at the criterion mesh, with the Monte-Carlo detection
probability at that defect.  ``negative_controls`` collects the mutants the gate
must reject -- un-halved doubling, coarse-mesh lock, the wrong spectral limit,
waterplane-lid closure and mode-normal mixing.  A control that *passes* is a
criterion defect, not a solver success.

Run::

    python -m tensorbem.benchmarks.case_ice_gate_power_section --write
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tensorbem.geometry import PanelMesh, ellipse_mesh, regular_polygon
from tensorbem.kernels import REFERENCE_LENGTH
from tensorbem.solver import (
    assemble_matrices,
    compute_added_mass_matrix_2d,
    evaluate_exterior_field,
    solve_laplace_exterior_neumann,
)

DENSITY = 1025.0
BEAM = 20.0
DRAFT = 6.0

#: Panel counts per side of the *doubled* rectangle (4 sides -> 4 * n panels).
LADDER: tuple[int, ...] = (8, 16, 32, 64, 128)

LIMIT_INFINITE_FREQUENCY = "infinite_frequency"   # phi = 0      -> A_inf
LIMIT_ZERO_FREQUENCY = "zero_frequency"           # d(phi)/dz=0  -> rigid wall
LIMITS = (LIMIT_INFINITE_FREQUENCY, LIMIT_ZERO_FREQUENCY)

ARTIFACT_ROOT = Path("benchmarks/verified/ice_gate_power_section")
RESULT_FILE = "results.json"
DOC_PATH = Path("docs/ice_gate_power_section.md")

#: Values under dispute, quoted from the reports cited in the module docstring.
DISPUTED_REFERENCES_PER_METRE = {
    "audit_19234_independent_potential_flow": 228.6e3,
    "chain_declared_A_inf_over_L": 86.1e3,
}

TOLERANCES = (0.03, 0.15)


# ---------------------------------------------------------------- mesh helpers


def doubled_rectangle_mesh(
    half_beam: float, draft: float, panels_per_side: int
) -> PanelMesh:
    """Return the doubled (mirrored) rectangular section as one closed polygon.

    The wetted half spans ``x in [-half_beam, half_beam]``, ``y in [-draft, 0]``;
    the doubled body spans ``y in [-draft, +draft]``.  An even
    ``panels_per_side`` keeps every panel strictly on one side of the mirror
    line, which the half-domain load integral relies on.
    """
    for name, value in (("half_beam", half_beam), ("draft", draft)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
    if panels_per_side < 2 or panels_per_side % 2:
        raise ValueError("panels_per_side must be an even integer >= 2")
    xs = np.linspace(-half_beam, half_beam, panels_per_side + 1)
    ys = np.linspace(-draft, draft, panels_per_side + 1)
    ones = np.ones(panels_per_side)
    bottom = np.column_stack((xs[:-1], -draft * ones))
    right = np.column_stack((half_beam * ones, ys[:-1]))
    top = np.column_stack((xs[::-1][:-1], draft * ones))
    left = np.column_stack((-half_beam * ones, ys[::-1][:-1]))
    return PanelMesh(np.vstack((bottom, right, top, left)))


# ------------------------------------------------------------- integral solve


def _mirror_flux(doubled: PanelMesh, limit: str) -> np.ndarray:
    """Fluid-domain normal derivative on the doubled body for unit heave.

    ``base`` is the usual rigid-body heave data ``q = V n_y`` on every panel,
    which for a ``y``-symmetric doubled body is odd in ``y`` and therefore
    already satisfies ``phi = 0`` on the mirror line: that is the ``A_inf``
    branch.  The rigid-wall branch flips the image (upper) half, which makes the
    potential even in ``y`` and leaves a non-vanishing net flux -- the
    2-D pathology reported by :func:`mirror_diagnostics`.
    """
    if limit not in LIMITS:
        raise ValueError(f"limit must be one of {LIMITS}")
    normals = doubled.normals
    upper = doubled.centers[:, 1] > 0.0
    flux = -normals[:, 1].copy()
    if limit == LIMIT_ZERO_FREQUENCY:
        flux[upper] = normals[upper, 1]
    return flux


def half_domain_heave_added_mass(
    doubled: PanelMesh,
    density: float = DENSITY,
    limit: str = LIMIT_INFINITE_FREQUENCY,
) -> float:
    """Heave added mass per unit span of the *half* domain (one wetted section).

    The exterior Neumann problem is solved once for the doubled body; by
    symmetry the half-domain kinetic energy is half of the doubled one, so the
    added mass is ``-rho * sum(phi * n_y * ds)`` over the physical (lower)
    panels -- the same convention as
    :func:`tensorbem.added_mass_limit.added_mass_limit`.
    """
    if not isinstance(doubled, PanelMesh):
        raise TypeError("doubled must be a PanelMesh")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be a positive finite number")
    normals = doubled.normals
    lower = doubled.centers[:, 1] < 0.0
    flux = _mirror_flux(doubled, limit)
    single, double = assemble_matrices(doubled, reference_length=REFERENCE_LENGTH)
    potential = np.linalg.solve(0.5 * np.eye(doubled.panel_count) - double, single @ flux)
    return float(
        -density * np.sum(potential[lower] * normals[lower, 1] * doubled.lengths[lower])
    )


def mirror_diagnostics(
    doubled: PanelMesh, limit: str = LIMIT_INFINITE_FREQUENCY, probes: int = 7
) -> dict[str, Any]:
    """Residuals that identify which free-surface condition the solve satisfies.

    Returns the net flux over the closed doubled body (compatibility: it must
    vanish for a decaying exterior potential), plus ``max|phi|`` and
    ``max|d(phi)/dy|`` on probe points of the mirror line outside the body.
    The ``A_inf`` branch must show ``max|phi| ~ 0``; the rigid-wall branch shows
    ``max|d(phi)/dy| ~ 0`` together with a non-zero net flux.
    """
    flux = _mirror_flux(doubled, limit)
    result = solve_laplace_exterior_neumann(doubled, flux)
    half_beam = float(np.max(doubled.vertices[:, 0]))
    xs = np.linspace(1.05 * half_beam, 4.0 * half_beam, probes)
    points = np.column_stack((xs, np.zeros(probes)))
    field = evaluate_exterior_field(doubled, result, points)
    scale = float(np.max(np.abs(flux)))
    return {
        "limit": limit,
        "net_flux": float(np.sum(flux * doubled.lengths)),
        "mirror_line_max_abs_potential": float(np.max(np.abs(field.potential))),
        "mirror_line_max_abs_normal_derivative": float(
            np.max(np.abs(field.gradient[:, 1]))
        ),
        "flux_scale": scale,
        "mirror_line_potential_ratio": float(
            np.max(np.abs(field.potential)) / max(scale * half_beam, 1e-300)
        ),
        "mirror_line_normal_derivative_ratio": float(
            np.max(np.abs(field.gradient[:, 1])) / max(scale, 1e-300)
        ),
    }


def exact_circle_added_mass(radius: float, density: float = DENSITY) -> float:
    """``rho*pi*a**2``: exact 2-D heave added mass of a circular cylinder."""
    return density * math.pi * radius**2


# ------------------------------------------------------------ verification set


def richardson_limit(
    levels: Sequence[int], values: Sequence[float]
) -> tuple[float, float]:
    """Return ``(observed_order, extrapolated_limit)`` from the finest three levels.

    ``p = log2(|v[1]-v[0]| / |v[2]-v[1]|)`` with ``levels`` sorted ascending, then
    ``limit = v[2] + (v[2]-v[1]) / (2**p - 1)``.  A non-monotone or zero
    difference returns ``(nan, v[-1])`` rather than a fabricated order.
    """
    if len(levels) != len(values) or len(values) < 3:
        raise ValueError("need at least three mesh levels")
    v = np.asarray(values, dtype=float)
    first = abs(v[1] - v[0])
    second = abs(v[2] - v[1])
    if first <= 0.0 or second <= 0.0:
        return float("nan"), float(v[-1])
    order = math.log2(first / second)
    if not math.isfinite(order) or order <= 0.0:
        return float("nan"), float(v[-1])
    limit = v[2] + (v[2] - v[1]) / (2.0**order - 1.0)
    return order, float(limit)


def convergence_ladder(
    half_beam: float,
    draft: float,
    density: float = DENSITY,
    ladder: Iterable[int] = LADDER,
    limits: Sequence[str] = LIMITS,
) -> dict[str, Any]:
    """Mesh ladder for both free-surface branches of one rectangular section."""
    levels = sorted(int(n) for n in ladder)
    out: dict[str, Any] = {
        "half_beam": half_beam,
        "draft": draft,
        "density": density,
        "beam_over_draft": 2.0 * half_beam / draft,
        "levels_panels_per_side": levels,
        "limiting_conditions": {},
    }
    for limit in limits:
        values = []
        for n in levels:
            mesh = doubled_rectangle_mesh(half_beam, draft, n)
            values.append(half_domain_heave_added_mass(mesh, density, limit))
        order, extrapolated = richardson_limit(levels[-3:], values[-3:])
        orders = []
        for i in range(len(levels) - 2):
            triple = values[i : i + 3]
            if triple[1] != triple[0] and triple[2] != triple[1]:
                rate = abs(triple[1] - triple[0]) / abs(triple[2] - triple[1])
                if rate > 0.0:
                    orders.append(math.log2(rate))
        out["limiting_conditions"][limit] = {
            "added_mass_per_metre": values,
            "added_mass_per_metre_tonnes": [v / 1000.0 for v in values],
            "observed_order_finest_three": order,
            "observed_orders_all_triples": orders,
            "richardson_limit": extrapolated,
            "richardson_limit_tonnes": extrapolated / 1000.0,
            "relative_change_finest_pair": (
                abs(values[-1] - values[-2]) / abs(values[-1])
                if values[-1]
                else float("nan")
            ),
            "mirror_diagnostics_finest": mirror_diagnostics(
                doubled_rectangle_mesh(half_beam, draft, levels[2]), limit
            ),
        }
    return out


def analytic_ladder(
    density: float = DENSITY,
    circle_panels: Sequence[int] = (64, 128, 256),
) -> dict[str, Any]:
    """Closed-form baselines the solver is pinned against."""
    radius = 1.0
    circle = []
    for n in circle_panels:
        mesh = regular_polygon(n, radius=radius)
        matrix = compute_added_mass_matrix_2d(mesh, density=density)
        exact = exact_circle_added_mass(radius, density)
        circle.append(
            {
                "panels": n,
                "added_mass": float(matrix.added_mass_matrix[1, 1]),
                "exact": exact,
                "relative_error": abs(float(matrix.added_mass_matrix[1, 1]) - exact) / exact,
            }
        )

    ellipse = []
    for n in circle_panels:
        mesh = ellipse_mesh(n, semi_major=2.0, semi_minor=1.0)
        matrix = compute_added_mass_matrix_2d(mesh, density=density)
        exact = density * math.pi * 2.0**2  # semi-axis across the motion, squared
        ellipse.append(
            {
                "panels": n,
                "added_mass": float(matrix.added_mass_matrix[1, 1]),
                "exact": exact,
                "relative_error": abs(float(matrix.added_mass_matrix[1, 1]) - exact) / exact,
            }
        )

    half_circle = []
    for n in circle_panels:
        mesh = regular_polygon(n, radius=radius)
        value = half_domain_heave_added_mass(mesh, density, LIMIT_INFINITE_FREQUENCY)
        exact = 0.5 * exact_circle_added_mass(radius, density)
        half_circle.append(
            {
                "panels": n,
                "added_mass": value,
                "exact": exact,
                "relative_error": abs(value - exact) / exact,
            }
        )
    return {
        "full_circle_exact_pi_rho_a2": circle,
        "full_ellipse_exact_pi_rho_a2": ellipse,
        "half_immersed_circle_A_inf_exact_half": half_circle,
    }


# ------------------------------------------------------------- informativeness


def gate_verdict(value: float, reference: float, tolerance: float) -> dict[str, Any]:
    """Relative deviation of ``value`` from ``reference`` and the gate decision."""
    deviation = (value - reference) / reference
    return {
        "value": value,
        "reference": reference,
        "relative_deviation": deviation,
        "tolerance": tolerance,
        "rejected": bool(abs(deviation) > tolerance),
    }


def minimum_detectable_defect(
    half_beam: float,
    draft: float,
    density: float = DENSITY,
    mesh_panels_per_side: int = 32,
    tolerance: float = 0.03,
    grid: Sequence[float] | None = None,
    samples: int = 24,
    seed: int = 20261116,
    reference_panels: int = 128,
) -> dict[str, Any]:
    """Smallest section-geometry defect that the gate rejects at ``tolerance``.

    The reference is the Richardson limit of the fine ladder -- the value a
    criterion should be judged against.  A defect scales beam and draft by a
    common factor, is measured at the *criterion* mesh, and the Monte-Carlo
    probability at that defect is reported so the threshold is bound to a
    detection rate instead of an assertion.
    """
    middle = 2 * int(round(math.sqrt(mesh_panels_per_side * reference_panels) / 2))
    ladder = sorted({mesh_panels_per_side, middle, reference_panels})
    limits = convergence_ladder(
        half_beam, draft, density, ladder, (LIMIT_INFINITE_FREQUENCY,)
    )
    reference = limits["limiting_conditions"][LIMIT_INFINITE_FREQUENCY]["richardson_limit"]
    if grid is None:
        grid = (0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12)
    curve = []
    smallest = None
    for defect in grid:
        scale = 1.0 + defect
        mesh = doubled_rectangle_mesh(
            half_beam * scale, draft * scale, mesh_panels_per_side
        )
        value = half_domain_heave_added_mass(mesh, density, LIMIT_INFINITE_FREQUENCY)
        verdict = gate_verdict(value, reference, tolerance)
        curve.append({"defect": defect, **verdict})
        if verdict["rejected"] and smallest is None:
            smallest = defect
    rng = np.random.default_rng(seed)
    detection_probability = None
    if smallest is not None:
        magnitudes = rng.uniform(0.0, 2.0 * smallest, size=int(samples))
        signs = rng.choice((-1.0, 1.0), size=int(samples))
        rejected = 0
        considered = 0
        for magnitude, sign in zip(magnitudes, signs):
            scale = 1.0 + sign * magnitude
            if scale <= 0.0:
                continue
            considered += 1
            mesh = doubled_rectangle_mesh(
                half_beam * scale, draft * scale, mesh_panels_per_side
            )
            value = half_domain_heave_added_mass(mesh, density, LIMIT_INFINITE_FREQUENCY)
            rejected += int(abs((value - reference) / reference) > tolerance)
        detection_probability = rejected / max(considered, 1)
    return {
        "tolerance": tolerance,
        "criterion_mesh_panels_per_side": mesh_panels_per_side,
        "reference_richardson_limit": reference,
        "defect_curve": curve,
        "minimum_detectable_defect": smallest,
        "monte_carlo_detection_probability": detection_probability,
        "samples": int(samples),
        "seed": seed,
    }


def negative_controls(
    limits: dict[str, Any], density: float = DENSITY, tolerance: float = 0.03
) -> list[dict[str, Any]]:
    """Mutants that a correctly written criterion must reject."""
    half_beam = limits["half_beam"]
    draft = limits["draft"]
    infinite = limits["limiting_conditions"][LIMIT_INFINITE_FREQUENCY]
    zero = limits["limiting_conditions"][LIMIT_ZERO_FREQUENCY]
    reference = infinite["richardson_limit"]
    controls: list[dict[str, Any]] = []

    controls.append(
        {
            "name": "doubled_body_not_halved",
            "description": "half-domain mass reported as the full doubled-body mass",
            **gate_verdict(2.0 * reference, reference, tolerance),
        }
    )
    controls.append(
        {
            "name": "coarse_mesh_lock",
            "description": f"coarsest ladder level ({limits['levels_panels_per_side'][0]} panels/side)",
            **gate_verdict(infinite["added_mass_per_metre"][0], reference, tolerance),
        }
    )
    controls.append(
        {
            "name": "wrong_spectral_limit",
            "description": "rigid-wall (omega -> 0) branch used as the A_inf baseline",
            **gate_verdict(zero["richardson_limit"], reference, tolerance),
        }
    )

    mesh = doubled_rectangle_mesh(half_beam, draft, 32)
    controls.append(
        {
            "name": "mode_normal_mixing",
            "description": "heave load integrated against the surge normal component",
            **gate_verdict(
                _half_domain_added_mass_with_normal(mesh, density, 0), reference, tolerance
            ),
        }
    )
    controls.append(
        {
            "name": "waterplane_lid_no_image",
            "description": "wetted section closed by a lid on z = 0 instead of mirrored",
            **gate_verdict(_open_section_added_mass(mesh, density), reference, tolerance),
        }
    )
    return controls


def _half_domain_added_mass_with_normal(
    doubled: PanelMesh, density: float, normal_component: int
) -> float:
    """Same solve as :func:`half_domain_heave_added_mass`, wrong load normal."""
    normals = doubled.normals
    lower = doubled.centers[:, 1] < 0.0
    flux = _mirror_flux(doubled, LIMIT_INFINITE_FREQUENCY)
    single, double = assemble_matrices(doubled, reference_length=REFERENCE_LENGTH)
    potential = np.linalg.solve(0.5 * np.eye(doubled.panel_count) - double, single @ flux)
    return float(
        -density
        * np.sum(
            potential[lower] * normals[lower, normal_component] * doubled.lengths[lower]
        )
    )


def _open_section_added_mass(doubled: PanelMesh, density: float) -> float:
    """Added mass of a section closed by a lid sitting on the mirror line.

    The waterplane lid is the mistake diagnosed in
    :mod:`tensorbem.benchmarks.barge_crosscheck`: the wetted surface is closed
    with a panel on ``z = 0`` instead of being mirrored, which removes the image
    and replaces the free surface by a hull wall.
    """
    half_beam = float(np.max(doubled.vertices[:, 0]))
    draft = float(-np.min(doubled.vertices[:, 1]))
    n = 32
    x = np.linspace(-half_beam, half_beam, n + 1)
    y = np.linspace(-draft, 0.0, n + 1)
    ones = np.ones(n)
    bottom = np.column_stack((x[:-1], -draft * ones))
    right = np.column_stack((half_beam * ones, y[:-1]))
    lid = np.column_stack((x[::-1][:-1], np.zeros(n)))
    left = np.column_stack((-half_beam * ones, y[::-1][:-1]))
    mesh = PanelMesh(np.vstack((bottom, right, lid, left)))
    return float(compute_added_mass_matrix_2d(mesh, density=density).added_mass_matrix[1, 1])


# --------------------------------------------------------------- adjudication


def adjudicate(limits: dict[str, Any]) -> dict[str, Any]:
    """Compare the computed ``A_inf`` with the disputed chain/audit values."""
    infinite = limits["limiting_conditions"][LIMIT_INFINITE_FREQUENCY]["richardson_limit"]
    zero = limits["limiting_conditions"][LIMIT_ZERO_FREQUENCY]["richardson_limit"]
    rows = []
    for name, quoted in DISPUTED_REFERENCES_PER_METRE.items():
        rows.append(
            {
                "reference_label": name,
                "quoted_per_metre": quoted,
                "deviation_vs_A_inf": (quoted - infinite) / abs(infinite),
                "deviation_vs_rigid_wall_branch": (quoted - zero) / abs(zero),
            }
        )
    return {
        "computed_A_inf_per_metre": infinite,
        "computed_A_inf_tonnes_per_metre": infinite / 1000.0,
        "rigid_wall_branch_per_metre": zero,
        "rigid_wall_branch_tonnes_per_metre": zero / 1000.0,
        "rows": rows,
    }


def beam_over_draft_sweep(
    ratios: Sequence[float] = (2.0, 3.0, 4.0, 6.0, 10.0),
    draft: float = DRAFT,
    density: float = DENSITY,
    panels_per_side: int = 64,
) -> list[dict[str, Any]]:
    """``A_inf`` as a function of ``B/T`` -- the trend the literature plots."""
    rows = []
    for ratio in ratios:
        half_beam = 0.5 * ratio * draft
        mesh = doubled_rectangle_mesh(half_beam, draft, panels_per_side)
        a_inf = half_domain_heave_added_mass(mesh, density, LIMIT_INFINITE_FREQUENCY)
        rows.append(
            {
                "beam_over_draft": ratio,
                "half_beam": half_beam,
                "A_inf_per_metre": a_inf,
                "A_inf_per_metre_tonnes": a_inf / 1000.0,
                "A_inf_over_rho_beam_half_squared": a_inf
                / (density * half_beam**2),
            }
        )
    return rows


# ------------------------------------------------------------------- producer


def build_results(
    beam: float = BEAM,
    draft: float = DRAFT,
    density: float = DENSITY,
    ladder: Sequence[int] = LADDER,
) -> dict[str, Any]:
    """Compute every number this module reports."""
    half_beam = 0.5 * beam
    limits = convergence_ladder(half_beam, draft, density, ladder)
    return {
        "module": "tensorbem.benchmarks.case_ice_gate_power_section",
        "section": {
            "beam": beam,
            "draft": draft,
            "half_beam": half_beam,
            "density": density,
            "beam_over_draft": beam / draft,
        },
        "limits": limits,
        "analytic_baselines": analytic_ladder(density),
        "negative_controls": negative_controls(limits, density),
        "informativeness": {
            f"tol_{int(round(tol * 100))}pct": minimum_detectable_defect(
                half_beam, draft, density, tolerance=tol
            )
            for tol in TOLERANCES
        },
        "adjudication": adjudicate(limits),
        "beam_over_draft_sweep": beam_over_draft_sweep(draft=draft, density=density),
    }


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(results: dict[str, Any]) -> str:
    """Render the results dict into the doc's number tables (no hand-typed figures)."""
    section = results["section"]
    levels = results["limits"]["levels_panels_per_side"]
    infinite = results["limits"]["limiting_conditions"][LIMIT_INFINITE_FREQUENCY]
    zero = results["limits"]["limiting_conditions"][LIMIT_ZERO_FREQUENCY]
    lines = [
        f"* Section: B = {_fmt(section['beam'], 1)} m, T = {_fmt(section['draft'], 1)} m,"
        f" rho = {_fmt(section['density'], 0)} kg/m^3, B/T = {_fmt(section['beam_over_draft'], 2)}",
        "",
        "### Mesh ladder (per unit span)",
        "",
        "| panels/side | panels | A(w->inf) [t/m] | rigid-wall branch [t/m] |",
        "|---|---|---|---|",
    ]
    for i, n in enumerate(levels):
        lines.append(
            f"| {n} | {4 * n} | {_fmt(infinite['added_mass_per_metre_tonnes'][i])} |"
            f" {_fmt(zero['added_mass_per_metre_tonnes'][i])} |"
        )
    lines += [
        "",
        "| branch | observed order p (finest three) | Richardson limit [t/m] | finest-pair relative change |",
        "|---|---|---|---|",
    ]
    for name, block in (
        ("omega -> inf (pressure release, A_inf)", infinite),
        ("rigid wall (omega -> 0, not admissible in 2-D)", zero),
    ):
        lines.append(
            f"| {name} | {_fmt(block['observed_order_finest_three'], 3)} |"
            f" {_fmt(block['richardson_limit_tonnes'])} |"
            f" {_fmt(block['relative_change_finest_pair'] * 100.0, 3)} % |"
        )
    lines += [
        "",
        "### Mirror-line diagnostics (which free-surface condition is actually met)",
        "",
        "| branch | net flux d(phi)/dn ds | max abs(phi) on z=0 | max abs(dphi/dz) on z=0 |",
        "|---|---|---|---|",
    ]
    for name, block in (("A_inf", infinite), ("rigid wall", zero)):
        diag = block["mirror_diagnostics_finest"]
        lines.append(
            f"| {name} | {diag['net_flux']:+.4f} |"
            f" {diag['mirror_line_max_abs_potential']:.4e} |"
            f" {diag['mirror_line_max_abs_normal_derivative']:.4e} |"
        )
    lines += [
        "",
        "### Closed-form baselines",
        "",
        "| baseline | mesh | computed | exact | rel. error |",
        "|---|---|---|---|---|",
    ]
    for key, label, digits in (
        ("full_circle_exact_pi_rho_a2", "full circle rho*pi*a^2", 4),
        ("full_ellipse_exact_pi_rho_a2", "full ellipse rho*pi*a^2", 4),
        ("half_immersed_circle_A_inf_exact_half", "half-immersed circle 0.5*rho*pi*a^2", 4),
    ):
        for row in results["analytic_baselines"][key]:
            lines.append(
                f"| {label} | {row['panels']} panels | {_fmt(row['added_mass'], digits)} |"
                f" {_fmt(row['exact'], digits)} | {row['relative_error']:.2e} |"
            )
    lines += [
        "",
        "### Negative controls (each must be REJECTED at the 3 % gate)",
        "",
        "| control | deviation | rejected? |",
        "|---|---|---|",
    ]
    for row in results["negative_controls"]:
        lines.append(
            f"| {row['name']} | {row['relative_deviation'] * 100.0:+.2f} % |"
            f" {'yes' if row['rejected'] else 'NO (criterion defect)'} |"
        )
    lines += [
        "",
        "### Informativeness: smallest detectable geometry defect",
        "",
        "| gate | criterion mesh | reference [t/m] | min detectable defect | MC detection prob. |",
        "|---|---|---|---|---|",
    ]
    for key, block in results["informativeness"].items():
        prob = block["monte_carlo_detection_probability"]
        smallest = block["minimum_detectable_defect"]
        lines.append(
            f"| {key} | {block['criterion_mesh_panels_per_side']} panels/side |"
            f" {_fmt(block['reference_richardson_limit'] / 1000.0)} |"
            f" {('%.4f' % smallest) if smallest is not None else 'not reached on grid'} |"
            f" {('%.2f' % prob) if prob is not None else 'n/a'} |"
        )
    lines += [
        "",
        "### B/T sweep (A_inf, 64 panels/side)",
        "",
        "| B/T | half-beam [m] | A_inf [t/m] | A_inf/(rho*half_beam^2) |",
        "|---|---|---|---|",
    ]
    for row in results["beam_over_draft_sweep"]:
        lines.append(
            f"| {row['beam_over_draft']:.2f} | {row['half_beam']:.2f} |"
            f" {_fmt(row['A_inf_per_metre_tonnes'])} |"
            f" {row['A_inf_over_rho_beam_half_squared']:.4f} |"
        )
    lines += [
        "",
        "### Adjudication of the disputed values",
        "",
        "| quoted value | [t/m] | deviation vs computed A_inf | deviation vs rigid-wall branch |",
        "|---|---|---|---|",
    ]
    for row in results["adjudication"]["rows"]:
        lines.append(
            f"| {row['reference_label']} | {_fmt(row['quoted_per_metre'] / 1000.0)} |"
            f" {row['deviation_vs_A_inf'] * 100.0:+.2f} % |"
            f" {row['deviation_vs_rigid_wall_branch'] * 100.0:+.2f} % |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write results.json + doc tables")
    parser.add_argument("--beam", type=float, default=BEAM)
    parser.add_argument("--draft", type=float, default=DRAFT)
    args = parser.parse_args(argv)
    results = build_results(beam=args.beam, draft=args.draft)
    text = render_markdown(results)
    print(text)
    if args.write:
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / RESULT_FILE).write_text(json.dumps(results, indent=2) + "\n")
        DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOC_PATH.write_text(
            "# 2-D heave added-mass anchor for the ice-going vessel section\n\n"
            "> Numbers below are generated by "
            "`python -m tensorbem.benchmarks.case_ice_gate_power_section --write`; "
            "do not edit them by hand.\n\n" + text + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())