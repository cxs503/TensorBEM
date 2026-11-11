"""Verification of the hydroelastic coupling layer (campaign node IN41-N04).

What is verified, and against what
----------------------------------
The module :mod:`tensorbem.hydroelastic` couples a simply supported Kirchhoff
panel, mounted flush in an infinite rigid baffle, to the zero-frequency
hydrodynamic model of the fluid half space (rigid calm free surface).  Four
independent questions are answered here:

1. **Kernel and quadrature** -- the Galerkin single-layer operator
   ``rho/(2 pi) int int w_i w_j / |x - x'|`` must reproduce the *closed form* of
   the four-fold integral for the uniform mode
   (:func:`tensorbem.hydroelastic.rectangle_self_integral`) at every mesh level,
   and the analytic plane-wave impedance ``m_a = rho/k`` must be recovered as the
   ``k a -> infinity`` limit of the panel result (that limit is derived from the
   fluid, not from the panel discretisation, so it fixes the ``rho/(2 pi)``
   kernel constant and the quadrature of the singular self/near pairs).
2. **Wet modes** -- the first three wet eigenfrequencies of
   ``K phi = omega^2 (M + A) phi``, with the added mass from the panel BEM,
   compared against their mesh-independent (Richardson-extrapolated) values, with
   the modal added-mass ratios ``mu_i = A_ii/M_ii`` quantified against the dry
   frequencies.  The local plane-wave asymptote ``m_a = rho/k`` is followed over a
   mode sweep to show that it is only the high-wavenumber limit.
3. **Wave-excited response** -- vertical (centre deflection per unit wave
   amplitude) and bending (``M_xx`` at the centre) RAO driven by the
   Froude-Krylov pressure of a regular wave, checked against the closed-form
   quasi-static limit, against the coupled eigenfrequency (peak position) and
   with a damping sweep; the local operator resonance is checked against the
   flexural-gravity dispersion root.
4. **Partitioned coupling** -- the staggered displacement/pressure fixed point
   must converge below ``1e-6`` and the converged state must satisfy the
   monolithic generalised eigenproblem to round-off.

Cross-module check
------------------
``tensorbem.ice`` implements the mass-loading flexural-gravity dispersion
independently.  With zero plate rigidity and a finite depth the local impedance
form used here must reproduce it exactly.

What this study does *not* establish
------------------------------------
There is no experimental data in the repository, so the absolute accuracy of the
wet frequencies against a measured panel is ``WITHHELD``.  The hydrodynamic model
is the zero-frequency one (real, frequency independent added mass) with no
radiation damping, the excitation is Froude-Krylov only (a flush panel in an
infinite baffle does not scatter in this model) and the plate is a Kirchhoff
plate.  Every number below is a discretisation/consistency statement about that
model, not a validation of it against reality.

Run ``python -m tensorbem.benchmarks.hydroelastic_verification`` for JSON and
``--figure`` for the comparison plots.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .. import __version__ as tensorbem_version
from .. import hydroelastic as hyd
from .. import ice as ice_module

PANEL_SPEC: dict[str, float] = {
    "length": 1.0,
    "breadth": 0.6,
    "thickness": 0.010,
    "young_modulus": 2.10e11,
    "poisson_ratio": 0.30,
    "plate_density": 7850.0,
    "fluid_density": 1025.0,
}
MESH_LEVELS: tuple[tuple[int, int], ...] = ((8, 5), (16, 10), (32, 20))
PRODUCTION_LEVEL: tuple[int, int] = (32, 20)
MODE_COUNT = 3
ASYMPTOTE_MODES: tuple[tuple[int, int], ...] = (
    (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 3),
)
TOLERANCES: dict[str, float] = {
    "kernel_closed_form_relative": 5.0e-3,
    "asymptote_limit_relative": 0.03,
    "wet_frequency_relative": 0.03,
    "rao_peak_relative": 0.02,
    "quasi_static_relative": 1.0e-4,
    "coupling_residual": 1.0e-6,
    "monolithic_residual": 1.0e-10,
}
#: Residuals that sit on the floating-point floor carry no reproducible digits.
#: Reordering the same summation -- a different BLAS thread count is enough, and
#: the mutation audit pins ``OMP_NUM_THREADS=1`` -- multiplies them by a factor
#: of two or more, so ``1.79e-17`` in the committed document and ``3.47e-16`` in
#: a fresh run are the *same* statement.  The published document is pinned
#: character-for-character by ``tests/test_hydroelastic.py``, so emitting those
#: digits made the reproducibility gate depend on the thread count: green with
#: the default BLAS pool, red with one thread.  Values at or below this floor
#: are reported as ``< 1e-14`` instead of as bare digits; anything above it is
#: still printed exactly, so a genuine regression cannot hide behind the clamp.
ROUNDOFF_FLOOR = 1.0e-14
RAO_DAMPING_RATIOS = (0.005, 0.02, 0.05)
RAO_SWEEP_POINTS = 1200
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "benchmarks" / "verified" / "bem_hydroelastic"
DOC_PATH = ROOT / "docs" / "bem_hydroelastic.md"


def format_residual(value: float) -> str:
    """Format a residual without pinning digits that are pure round-off.

    Below :data:`ROUNDOFF_FLOOR` the stored digits are an artefact of the
    summation order, not of the model, so they are reported as ``< 1e-14``.
    """
    value = float(value)
    if abs(value) <= ROUNDOFF_FLOOR:
        return f"< {ROUNDOFF_FLOOR:.0e}"
    return f"{value:.2e}"


def _rel(value: float, reference: float) -> float:
    """Relative difference, safe at a zero reference."""
    if reference == 0.0:
        return float("inf") if value != 0.0 else 0.0
    return abs(value - reference) / abs(reference)


def _panel(spec: dict[str, float] | None = None) -> hyd.ElasticPanel:
    return hyd.ElasticPanel(**(PANEL_SPEC if spec is None else spec))


def _modes(panel: hyd.ElasticPanel, count: int = MODE_COUNT) -> list[hyd.DryMode]:
    return hyd.simply_supported_modes(panel, count)


# ---------------------------------------------------------------------------
# 1. kernel and quadrature
# ---------------------------------------------------------------------------
def asymptote_fit(sweep: Sequence[dict[str, Any]], min_ka: float = 5.0) -> dict[str, Any]:
    """Power-law fit of the excess over the plane-wave impedance.

    The added mass of a finite panel lies *above* the plane-wave value ``rho/k``
    because the wavenumber spectrum of a confined mode reaches ``k -> 0`` where
    the fluid impedance ``rho/k`` diverges.  The excess is fitted as

        mu_BEM / mu_local - 1 = c * (k a) ** (-p)

    and extrapolated: the kernel constant of
    :func:`tensorbem.hydroelastic.panel_added_mass_matrix` is the physical
    ``rho/(2 pi)`` if and only if the excess vanishes as ``k a -> infinity``.
    """
    entries = [entry for entry in sweep if entry["ka"] >= min_ka]
    ka = np.array([entry["ka"] for entry in entries])
    excess = np.array([entry["ratio"] - 1.0 for entry in entries])
    if np.any(excess <= 0.0):
        raise ValueError("excess must be positive to fit a power law")
    design = np.vstack([np.ones_like(ka), np.log(ka)]).T
    coefficients, *_ = np.linalg.lstsq(design, np.log(excess), rcond=None)
    log_c, negative_p = coefficients
    exponent = -float(negative_p)
    coefficient = float(math.exp(log_c))
    predicted = coefficient * ka ** (-exponent)
    reference_ka = 50.0
    excess_at_reference = coefficient * reference_ka ** (-exponent)
    return {
        "modes_used": [entry["mode"] for entry in entries],
        "ka_used": [float(v) for v in ka],
        "exponent": exponent,
        "coefficient": coefficient,
        "excess_at_ka_50": excess_at_reference,
        "residual_max": float(np.max(np.abs(predicted - excess))),
        "relative_error": excess_at_reference,
    }


def kernel_study(panel: hyd.ElasticPanel) -> dict[str, Any]:
    """Panel quadrature against the closed-form uniform-mode integral."""
    exact = hyd.exact_uniform_added_mass(panel)
    uniform_mode = [hyd.DryMode(1, 1, panel)]
    levels: list[dict[str, Any]] = []
    for nx, ny in MESH_LEVELS:
        _, kernel = hyd.panel_added_mass_matrix(
            panel, uniform_mode, nx=nx, ny=ny, return_kernel=True
        )
        value = panel.fluid_density / (2.0 * math.pi) * float(np.sum(kernel))
        levels.append(
            {
                "nx": nx,
                "ny": ny,
                "panels": nx * ny,
                "uniform_added_mass_kg": value,
                "relative_error": _rel(value, exact),
            }
        )
    return {
        "uniform_mode_closed_form_kg": exact,
        "levels": levels,
        "tolerance": TOLERANCES["kernel_closed_form_relative"],
        "passed": bool(
            all(
                level["relative_error"] < TOLERANCES["kernel_closed_form_relative"]
                for level in levels
            )
        ),
    }


# ---------------------------------------------------------------------------
# 2. wet modes
# ---------------------------------------------------------------------------
def _richardson(values: Sequence[float], ratio: float = 2.0) -> tuple[float, float]:
    """Richardson extrapolation and observed order from three mesh levels.

    ``values`` are ordered coarse -> fine.
    """
    coarse, medium, fine = values
    delta_coarse = medium - coarse
    delta_fine = fine - medium
    if delta_fine == 0.0:
        return fine, float("nan")
    if delta_coarse / delta_fine <= 0.0:
        order = 1.0
    else:
        order = math.log(abs(delta_coarse / delta_fine)) / math.log(ratio)
        order = min(max(order, 0.5), 6.0)
    return fine + delta_fine / (ratio ** order - 1.0), order


def wet_mode_study(panel: hyd.ElasticPanel) -> dict[str, Any]:
    """Wet eigenfrequencies versus mesh-independent values, plus added-mass ratios."""
    modes = _modes(panel)
    dry = np.array([mode.omega_dry for mode in modes])
    per_level: list[dict[str, Any]] = []
    frequency_history: list[list[float]] = []
    added_mass_fine: np.ndarray | None = None
    for nx, ny in MESH_LEVELS:
        added_mass = hyd.panel_added_mass_matrix(panel, modes, nx=nx, ny=ny)
        wet = hyd.wet_modes(panel, modes, added_mass)
        frequency_history.append([float(w) for w in wet.omegas])
        if (nx, ny) == PRODUCTION_LEVEL:
            added_mass_fine = added_mass
        per_level.append(
            {
                "nx": nx,
                "ny": ny,
                "panels": nx * ny,
                "wet_omega": [float(w) for w in wet.omegas],
                "added_mass_ratios": [
                    float(m) for m in hyd.added_mass_ratios(modes, added_mass)
                ],
                "added_mass_matrix_kg": [[float(a) for a in row] for row in added_mass],
            }
        )
    if added_mass_fine is None:
        raise RuntimeError("production mesh level must be part of MESH_LEVELS")

    extrapolated: list[float] = []
    orders: list[float] = []
    for index in range(len(modes)):
        values = [history[index] for history in frequency_history]
        value, order = _richardson(values)
        extrapolated.append(value)
        orders.append(order)

    production = per_level[-1]
    errors = [
        _rel(production["wet_omega"][i], extrapolated[i]) for i in range(len(modes))
    ]
    gci = [
        1.25
        * _rel(frequency_history[-1][i], frequency_history[-2][i])
        / max(2 ** orders[i] - 1.0, 1e-12)
        for i in range(len(modes))
    ]
    mu = hyd.added_mass_ratios(modes, added_mass_fine)
    wet_fine = hyd.wet_modes(panel, modes, added_mass_fine)

    # local plane-wave asymptote over a mode sweep (same panel, production mesh)
    sweep_modes = [hyd.DryMode(m, n, panel) for m, n in ASYMPTOTE_MODES]
    added_sweep = hyd.panel_added_mass_matrix(
        panel, sweep_modes, nx=PRODUCTION_LEVEL[0], ny=PRODUCTION_LEVEL[1]
    )
    added_local = hyd.local_added_mass_matrix(panel, sweep_modes)
    mu_mesh = hyd.added_mass_ratios(sweep_modes, added_sweep)
    mu_local = hyd.added_mass_ratios(sweep_modes, added_local)
    asymptote = [
        {
            "mode": f"({mode.m},{mode.n})",
            "ka": mode.wavenumber * min(panel.length, panel.breadth),
            "mu_mesh": float(mu_mesh[i]),
            "mu_local": float(mu_local[i]),
            "ratio": float(mu_mesh[i] / mu_local[i]),
        }
        for i, mode in enumerate(sweep_modes)
    ]

    asymptote_limit_fit = asymptote_fit(asymptote)

    # cross-module dispersion check against tensorbem.ice (D = 0, finite depth)
    cover = ice_module.IceCover(
        thickness=panel.thickness,
        ice_density=panel.plate_density,
        water_density=panel.fluid_density,
        water_depth=200.0,
    )
    dispersion = []
    for k in (0.05, 0.2, 1.0, 5.0, 20.0):
        mine = hyd.flexural_gravity_omega(
            k, panel, flexural_rigidity=0.0, depth=cover.water_depth
        )
        reference = ice_module.mass_loading_omega(k, cover)
        dispersion.append(
            {
                "wavenumber": k,
                "omega_local_impedance": mine,
                "omega_ice_mass_loading": reference,
                "relative_error": _rel(mine, reference),
            }
        )

    return {
        "dry_omega": [float(w) for w in dry],
        "dry_frequency_hz": [float(w / (2 * math.pi)) for w in dry],
        "mode_labels": [mode.label for mode in modes],
        "levels": per_level,
        "richardson_extrapolated_omega": extrapolated,
        "observed_order": orders,
        "production_wet_omega": production["wet_omega"],
        "production_error_vs_extrapolated": errors,
        "gci_fine": gci,
        "wet_over_dry": [float(r) for r in wet_fine.ratio_wet_over_dry],
        "added_mass_ratios": [float(m) for m in mu],
        "coupling_matrix": [
            [
                float(added_mass_fine[i, j] / added_mass_fine[j, j])
                for j in range(len(modes))
            ]
            for i in range(len(modes))
        ],
        "local_model_wet_omega": [
            float(w) for w in hyd.panel_wet_omegas_analytic(panel, modes)
        ],
        "asymptote_sweep": asymptote,
        "asymptote_limit": asymptote_limit_fit,
        "dispersion_crosscheck": dispersion,
        "dispersion_crosscheck_max_error": max(e["relative_error"] for e in dispersion),
        "tolerance": TOLERANCES["wet_frequency_relative"],
        "passed": bool(max(errors) < TOLERANCES["wet_frequency_relative"]),
    }


# ---------------------------------------------------------------------------
# 3. wave-excited response
# ---------------------------------------------------------------------------
def _peak_location(omega: np.ndarray, amplitude: np.ndarray) -> tuple[float, float]:
    """Peak frequency with a parabolic refinement around the discrete maximum."""
    index = int(np.argmax(amplitude))
    if index == 0 or index == omega.size - 1:
        return float(omega[index]), float(amplitude[index])
    x0, x1, x2 = omega[index - 1], omega[index], omega[index + 1]
    y0, y1, y2 = amplitude[index - 1], amplitude[index], amplitude[index + 1]
    denominator = y0 - 2.0 * y1 + y2
    if denominator == 0.0:
        return float(x1), float(y1)
    shift = 0.5 * (y0 - y2) / denominator
    location = x1 + shift * (x2 - x1)
    height = y1 - 0.25 * (y0 - y2) * shift
    return float(location), float(height)


def rao_study(panel: hyd.ElasticPanel) -> dict[str, Any]:
    """Vertical/bending RAO: quasi-static limit, peak position, damping sweep."""
    modes = _modes(panel)
    added_mass = hyd.panel_added_mass_matrix(
        panel, modes, nx=PRODUCTION_LEVEL[0], ny=PRODUCTION_LEVEL[1]
    )
    wet = hyd.wet_modes(panel, modes, added_mass)
    omega_1 = float(wet.omegas[0])
    sweep = np.linspace(0.02 * omega_1, 1.5 * omega_1, RAO_SWEEP_POINTS)

    base = hyd.modal_rao(
        panel, modes, added_mass, sweep, wet_omegas=wet.omegas, damping_ratio=0.02
    )
    peak_freq, peak_value = _peak_location(np.asarray(base.omega), base.vertical_rao)
    peak_error = _rel(peak_freq, omega_1)
    bending_peak, bending_height = _peak_location(
        np.asarray(base.omega), base.bending_rao
    )

    # quasi-static limit: a *separate* evaluation far below resonance, because the
    # Froude-Krylov pressure already varies across the panel at 2 % of omega_1
    slow = np.array([1.0e-3 * omega_1])
    slow_result = hyd.modal_rao(
        panel, modes, added_mass, slow, wet_omegas=wet.omegas, damping_ratio=0.02
    )
    quasi_static = float(slow_result.vertical_rao[0])
    quasi_static_error = _rel(quasi_static, base.static_vertical_rao)

    modal_peak, modal_height = _peak_location(
        np.asarray(base.omega), np.abs(base.modal_amplitudes[:, 0])
    )
    modal_peak_error = _rel(modal_peak, omega_1)

    damping_sweep = []
    for zeta in RAO_DAMPING_RATIOS:
        result = hyd.modal_rao(
            panel, modes, added_mass, sweep, wet_omegas=wet.omegas, damping_ratio=zeta
        )
        location, height = _peak_location(np.asarray(result.omega), result.vertical_rao)
        damping_sweep.append(
            {
                "damping_ratio": zeta,
                "peak_omega": location,
                "peak_vertical_rao": height,
                "peak_error_vs_wet": _rel(location, omega_1),
            }
        )

    # local operator: the response peak must sit on the flexural-gravity root
    wavenumber = 3.0
    root = hyd.flexural_gravity_omega(wavenumber, panel)
    grid = np.linspace(0.5 * root, 1.5 * root, 4000)
    response = np.array(
        [hyd.infinite_plate_response(w, wavenumber, panel) for w in grid]
    )
    operator_peak, _ = _peak_location(grid, response)
    operator_error = _rel(operator_peak, root)

    return {
        "wet_omega": [float(w) for w in wet.omegas],
        "dry_omega": [float(mode.omega_dry) for mode in modes],
        "sweep": {
            "points": RAO_SWEEP_POINTS,
            "omega_min": float(sweep[0]),
            "omega_max": float(sweep[-1]),
        },
        "peak_frequency_rad_s": peak_freq,
        "peak_vertical_rao_m_per_m": peak_value,
        "peak_error_vs_wet_frequency": peak_error,
        "bending_peak_frequency_rad_s": bending_peak,
        "bending_peak_rao_Nm_per_m": bending_height,
        "bending_peak_error_vs_wet_frequency": _rel(bending_peak, omega_1),
        "fundamental_modal_peak_rad_s": modal_peak,
        "fundamental_modal_peak_error_vs_wet": modal_peak_error,
        "quasi_static_omega": float(slow[0]),
        "static_vertical_rao_closed_form": base.static_vertical_rao,
        "quasi_static_vertical_rao": quasi_static,
        "quasi_static_error": quasi_static_error,
        "damping_sweep": damping_sweep,
        "local_operator": {
            "wavenumber": wavenumber,
            "dispersion_root_omega": root,
            "response_peak_omega": operator_peak,
            "relative_error": operator_error,
        },
        "tolerance": TOLERANCES["rao_peak_relative"],
        "passed": bool(
            peak_error < TOLERANCES["rao_peak_relative"]
            and _rel(bending_peak, omega_1) < TOLERANCES["rao_peak_relative"]
            and quasi_static_error < TOLERANCES["quasi_static_relative"]
            and modal_peak_error < TOLERANCES["rao_peak_relative"]
            and operator_error < 1.0e-3
            and all(
                entry["peak_error_vs_wet"] < TOLERANCES["rao_peak_relative"]
                for entry in damping_sweep
            )
        ),
    }


# ---------------------------------------------------------------------------
# 4. partitioned coupling
# ---------------------------------------------------------------------------
def coupling_study(panel: hyd.ElasticPanel) -> dict[str, Any]:
    """Staggered displacement/pressure fixed point with residual history."""
    modes = _modes(panel)
    history = hyd.staggered_coupling(
        panel,
        modes,
        nx=PRODUCTION_LEVEL[0],
        ny=PRODUCTION_LEVEL[1],
        relaxation=0.7,
        tolerance=TOLERANCES["coupling_residual"],
    )
    added_mass = hyd.panel_added_mass_matrix(
        panel, modes, nx=PRODUCTION_LEVEL[0], ny=PRODUCTION_LEVEL[1]
    )
    wet = hyd.wet_modes(panel, modes, added_mass)
    monolithic = [
        hyd.monolithic_residual(
            panel, modes, added_mass, float(wet.omegas[i]), wet.eigenvectors[:, i]
        )
        for i in range(len(modes))
    ]
    finite = [r for r in history.residuals if math.isfinite(r)]
    return {
        "iterations": history.iterations,
        "converged": bool(history.converged),
        "relaxation": 0.7,
        "omega_residual_history": [float(r) for r in history.residuals],
        "pressure_residual_history": [
            float(r) for r in history.pressure_residuals
        ],
        "final_omega_residual": float(finite[-1]) if finite else float("nan"),
        "final_pressure_residual": float(history.pressure_residuals[-1]),
        "converged_omega": [float(w) for w in history.omegas],
        "monolithic_residual": monolithic,
        "tolerance": TOLERANCES["coupling_residual"],
        "passed": bool(
            history.converged
            and (finite[-1] if finite else 1.0) < TOLERANCES["coupling_residual"]
            and history.pressure_residuals[-1] < TOLERANCES["coupling_residual"]
            and max(monolithic) < TOLERANCES["monolithic_residual"]
        ),
    }


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def run() -> dict[str, Any]:
    """Run every study and return the JSON-able report."""
    panel = _panel()
    modes = _modes(panel)
    report: dict[str, Any] = {
        "study": "bem_hydroelastic",
        "campaign_node": "IN41-N04",
        "generated_by": "tensorbem.benchmarks.hydroelastic_verification",
        "provenance": {
            "tensorbem_version": tensorbem_version,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "panel": {
            "length_m": panel.length,
            "breadth_m": panel.breadth,
            "thickness_m": panel.thickness,
            "young_modulus_Pa": panel.young_modulus,
            "poisson_ratio": panel.poisson_ratio,
            "plate_density_kg_m3": panel.plate_density,
            "fluid_density_kg_m3": panel.fluid_density,
            "flexural_rigidity_Nm": panel.flexural_rigidity,
            "mass_per_area_kg_m2": panel.mass_per_area,
            "plate_mass_kg": panel.plate_mass,
            "dry_modes": [mode.label for mode in modes],
            "dry_frequency_hz": [
                float(mode.omega_dry / (2 * math.pi)) for mode in modes
            ],
            "mode_wavenumbers": [float(mode.wavenumber) for mode in modes],
        },
        "tolerances": TOLERANCES,
        "mesh_levels": [list(level) for level in MESH_LEVELS],
    }
    report["kernel"] = kernel_study(panel)
    report["wet_modes"] = wet_mode_study(panel)
    report["rao"] = rao_study(panel)
    report["coupling"] = coupling_study(panel)
    report["all_criteria_passed"] = bool(
        report["kernel"]["passed"]
        and report["wet_modes"]["asymptote_limit"]["relative_error"]
        < TOLERANCES["asymptote_limit_relative"]
        and report["wet_modes"]["passed"]
        and report["rao"]["passed"]
        and report["coupling"]["passed"]
    )
    report["artifacts"] = {
        "results_json": str((OUTPUT_ROOT / "results.json").relative_to(ROOT)),
        "report_md": str((OUTPUT_ROOT / "report.md").relative_to(ROOT)),
        "documentation_md": str(DOC_PATH.relative_to(ROOT)),
        "figure_kernel": str(
            (OUTPUT_ROOT / "figures" / "bem_hydroelastic_kernel.png").relative_to(ROOT)
        ),
        "figure_rao": str(
            (OUTPUT_ROOT / "figures" / "bem_hydroelastic_rao.png").relative_to(ROOT)
        ),
    }
    return report


def render_document(report: dict[str, Any]) -> str:
    """Markdown report generated from the JSON (single source of truth)."""
    panel = report["panel"]
    kernel = report["kernel"]
    wet = report["wet_modes"]
    rao = report["rao"]
    coupling = report["coupling"]
    lines: list[str] = []
    add = lines.append
    add("# Wet modes and wave-excited response of a hull panel (BEM hydroelasticity)")
    add("")
    add("Campaign node `IN41-N04`.  Module `tensorbem.hydroelastic`; generator")
    add("`python -m tensorbem.benchmarks.hydroelastic_verification`.")
    add("")
    add("## 1. Case")
    add("")
    add(
        "A simply supported Kirchhoff panel of planform "
        f"`a x b = {panel['length_m']:.3g} m x {panel['breadth_m']:.3g} m`, "
        f"thickness `h = {panel['thickness_m'] * 1e3:.3g} mm`, steel"
    )
    add(
        f"(`E = {panel['young_modulus_Pa']:.3g} Pa`, "
        f"`nu = {panel['poisson_ratio']:.2f}`, "
        f"`rho_p = {panel['plate_density_kg_m3']:.4g} kg/m^3`), mounted flush in an"
    )
    add("infinite rigid baffle with seawater on one side")
    add(
        f"(`rho = {panel['fluid_density_kg_m3']:.4g} kg/m^3`): the classical *wet"
        " panel* configuration of ship hydroelasticity (hull / bottom / tank panel)."
    )
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| flexural rigidity `D` | {panel['flexural_rigidity_Nm']:.4e} N m |")
    add(f"| mass per area `m''` | {panel['mass_per_area_kg_m2']:.4e} kg/m^2 |")
    add(f"| plate mass | {panel['plate_mass_kg']:.4g} kg |")
    add(f"| dry modes | {', '.join(panel['dry_modes'])} |")
    add(
        f"| dry frequencies | "
        f"{', '.join(f'{f:.2f} Hz' for f in panel['dry_frequency_hz'])} |"
    )
    add("")
    add("## 2. How it was done")
    add("")
    add("Fluid: inviscid, incompressible, irrotational velocity potential with")
    add("`Laplace(phi) = 0` in the half space, `dphi/dz = 0` on the baffle and")
    add("`dphi/dz = -i omega w` on the panel.  In the zero-frequency limit the calm")
    add("free surface is a rigid wall, so the whole plane `z = 0` is Neumann and the")
    add("half-space Green function evaluated on that plane collapses to")
    add("`1/(2 pi |x - x'|)`; the generalised added mass of the plate modes is the")
    add("symmetric Galerkin single-layer integral")
    add("")
    add("```")
    add("A_ij = (rho / 2 pi) * int_panel int_panel w_i(x) w_j(x') / |x - x'| dA dA'")
    add("```")
    add("")
    add("evaluated by `tensorbem.hydroelastic.panel_added_mass_matrix` on a uniform")
    add(
        "panel grid (levels "
        + ", ".join(f"{nx}x{ny}" for nx, ny in MESH_LEVELS)
        + "): the self-panel term is"
    )
    add("integrated **exactly** (the four-fold integral collapses to a one-dimensional")
    add("autocorrelation integral), separated panels use a 3x3 Gauss product rule and")
    add("panels closer than two panel sizes use a 4x4 subdivided 2x2-point rule.")
    add("Structure: exact dry modes `w_mn = sin(m pi x/a) sin(n pi y/b)`,")
    add("`omega_dry^2 = D k_mn^4 / m''`, `k_mn = pi sqrt((m/a)^2 + (n/b)^2)`.")
    add("Coupling: the generalised eigenproblem `K phi = omega^2 (M + A) phi` in the")
    add("dry-mode basis, and a partitioned displacement/pressure fixed point (structure")
    add("solve -> fluid re-linearisation -> under-relaxation, `omega_r = 0.7`).  Wave")
    add("excitation: Froude-Krylov pressure of a regular wave (deep-water wavenumber")
    add("`k = omega^2/g`); modal damping `C_ii = 2 zeta omega_wet,i (M_ii + A_ii)`.")
    add("")
    add("## 3. Results")
    add("")
    add("### 3.1 Kernel and quadrature against closed forms")
    add("")
    add("| mesh | panels | uniform-mode added mass [kg] | rel. error vs closed form |")
    add("|---|---|---|---|")
    for level in kernel["levels"]:
        add(
            f"| {level['nx']}x{level['ny']} | {level['panels']} | "
            f"{level['uniform_added_mass_kg']:.6f} | {level['relative_error']:.2e} |"
        )
    add("")
    add(
        f"Closed form of `int int dA dA'/|x-x'|` over the panel: "
        f"`{kernel['uniform_mode_closed_form_kg']:.6f}` kg; in dimensionless form "
        f"`A/(rho A_p^1.5) = "
        f"{kernel['uniform_mode_closed_form_kg'] / (panel['fluid_density_kg_m3'] * (panel['length_m'] * panel['breadth_m']) ** 1.5):.4f}`."
    )
    add("")
    limit = wet["asymptote_limit"]
    add(
        "Kernel constant.  The analytic plane-wave impedance `m_a = rho/k` (a plane"
        " wave under a rigid baffle gives `p = rho omega^2 w/k`, independent of the"
        " discretisation) is recovered as `k a -> infinity`: the excess"
        f" `mu_BEM/mu_local - 1` decays as `k a^(-{limit['exponent']:.2f})`"
        f" (modes {', '.join(limit['modes_used'])}, residual"
        f" `{limit['residual_max']:.2e}`), i.e. `{limit['excess_at_ka_50']:.2%}` at"
        " `k a = 50`.  If the `rho/(2 pi)` constant of the kernel were wrong by any"
        " factor, this excess would tend to a finite offset instead of vanishing."
    )
    add("")
    add("### 3.2 Wet modes and added mass")
    add("")
    add("| mode | `k_mn a` | `omega_dry` [rad/s] | `omega_wet` [rad/s] |")
    add("| `mu = A_ii/M_ii` | `omega_wet/omega_dry` | error vs extrapolated | GCI |")
    add("|---|---|---|---|---|---|---|")
    for i, label in enumerate(wet["mode_labels"]):
        add(
            f"| {label} | {panel['mode_wavenumbers'][i] * panel['length_m']:.3f} | "
            f"{wet['dry_omega'][i]:.3f} | {wet['production_wet_omega'][i]:.3f} | "
            f"{wet['added_mass_ratios'][i]:.4f} | {wet['wet_over_dry'][i]:.4f} | "
            f"{wet['production_error_vs_extrapolated'][i]:.2%} | "
            f"{wet['gci_fine'][i]:.2%} |"
        )
    add("")
    add(
        "Mesh-independent (Richardson-extrapolated) values: "
        + ", ".join(f"{v:.3f}" for v in wet["richardson_extrapolated_omega"])
        + " rad/s (observed order "
        + ", ".join(f"{p:.2f}" for p in wet["observed_order"])
        + ")."
    )
    add(
        "Local plane-wave model `omega^2 = D k^4/(m'' + rho/k)`: "
        + ", ".join(f"{v:.3f}" for v in wet["local_model_wet_omega"])
        + " rad/s -- an asymptote, not a reference (see below)."
    )
    add("")
    add("Modal added-mass ratios and the local asymptote (production mesh):")
    add("")
    add("| mode | `k a` | `mu` (BEM) | `mu` (local `rho/k`) | BEM/local |")
    add("|---|---|---|---|---|")
    for entry in wet["asymptote_sweep"]:
        add(
            f"| {entry['mode']} | {entry['ka']:.2f} | {entry['mu_mesh']:.4f} | "
            f"{entry['mu_local']:.4f} | {entry['ratio']:.3f} |"
        )
    add("")
    add("The BEM added mass exceeds the local plane-wave value and the ratio decays")
    add("towards one as `k a` grows: the local impedance is the high-wavenumber")
    add("asymptote and cannot be used as the reference for the fundamental mode.")
    add("")
    add("Cross-module dispersion check (`D = 0`, water depth 200 m) against")
    add("`tensorbem.ice.mass_loading_omega`:")
    add("")
    add("| `k` [1/m] | this module [rad/s] | `tensorbem.ice` [rad/s] | rel. error |")
    add("|---|---|---|---|")
    for entry in wet["dispersion_crosscheck"]:
        add(
            f"| {entry['wavenumber']:.3g} | {entry['omega_local_impedance']:.12f} | "
            f"{entry['omega_ice_mass_loading']:.12f} | {entry['relative_error']:.1e} |"
        )
    add("")
    add("### 3.3 Wave-excited response (vertical and bending RAO)")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(
        f"| sweep | {rao['sweep']['points']} points, "
        f"{rao['sweep']['omega_min']:.2f} .. {rao['sweep']['omega_max']:.2f} rad/s |"
    )
    add(f"| coupled `omega_1` | {rao['wet_omega'][0]:.4f} rad/s |")
    add(
        f"| vertical RAO peak | {rao['peak_frequency_rad_s']:.4f} rad/s, "
        f"{rao['peak_vertical_rao_m_per_m']:.4e} m/m |"
    )
    add(f"| peak position error vs `omega_1` | {rao['peak_error_vs_wet_frequency']:.3%} |")
    add(
        f"| bending RAO peak | {rao['bending_peak_frequency_rad_s']:.4f} rad/s, "
        f"{rao['bending_peak_rao_Nm_per_m']:.4e} N m/m |"
    )
    add(
        f"| bending peak error vs `omega_1` | "
        f"{rao['bending_peak_error_vs_wet_frequency']:.3%} |"
    )
    add(
        f"| fundamental modal peak | {rao['fundamental_modal_peak_rad_s']:.4f} rad/s,"
        f" error vs `omega_1` {rao['fundamental_modal_peak_error_vs_wet']:.4%} |"
    )
    add(f"| quasi-static closed form | {rao['static_vertical_rao_closed_form']:.6e} m/m |")
    add(
        f"| quasi-static evaluation at `omega = {rao['quasi_static_omega']:.3e}` rad/s | "
        f"{rao['quasi_static_vertical_rao']:.6e} m/m "
        f"(error {rao['quasi_static_error']:.2e}) |"
    )
    add("")
    add("Damping sweep (structural modal ratio only; the model has no radiation damping):")
    add("")
    add("| `zeta` | peak [rad/s] | peak RAO [m/m] | error vs `omega_1` |")
    add("|---|---|---|---|")
    for entry in rao["damping_sweep"]:
        add(
            f"| {entry['damping_ratio']:.3f} | {entry['peak_omega']:.4f} | "
            f"{entry['peak_vertical_rao']:.4e} | {entry['peak_error_vs_wet']:.3%} |"
        )
    add("")
    add("Local-operator check: the resonance of the plane-wave response at")
    add(
        f"`k = {rao['local_operator']['wavenumber']:.3g} 1/m` sits at "
        f"{rao['local_operator']['response_peak_omega']:.6f} rad/s against the"
    )
    add(
        f"dispersion root {rao['local_operator']['dispersion_root_omega']:.6f} rad/s "
        f"(rel. error {rao['local_operator']['relative_error']:.2e})."
    )
    add("")
    add("### 3.4 Partitioned fluid-structure coupling")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| sweeps to convergence | {coupling['iterations']} |")
    add(f"| converged | {coupling['converged']} |")
    add(f"| final `omega` residual | {coupling['final_omega_residual']:.3e} |")
    add(f"| final pressure/operator residual | {coupling['final_pressure_residual']:.3e} |")
    add(
        "| monolithic residual (per mode) | "
        + ", ".join(format_residual(r) for r in coupling["monolithic_residual"])
        + " |"
    )
    add("")
    add("Residual history (relative change of the eigenfrequencies and of the")
    add("added-mass operator):")
    add("")
    add("| sweep | `|domega|/omega` | `|dA|/|A|` |")
    add("|---|---|---|")
    for it, (a, b) in enumerate(
        zip(coupling["omega_residual_history"], coupling["pressure_residual_history"]),
        start=1,
    ):
        add(f"| {it} | {a:.3e} | {b:.3e} |")
    add("")
    add("## 4. Figures")
    add("")
    add(f"* `{report['artifacts']['figure_kernel']}` -- (a) quadrature error of the")
    add("  uniform-mode added mass against the closed form versus panel count,")
    add("  (b) BEM/local added-mass ratio versus `k a`.")
    add(f"* `{report['artifacts']['figure_rao']}` -- (a) vertical and bending RAO with the")
    add("  coupled eigenfrequency marked, (b) damping sweep around the coupled resonance.")
    add("")
    add("## 5. Error statement")
    add("")
    add("| criterion | tolerance | measured | verdict |")
    add("|---|---|---|---|")
    add(
        f"| uniform-mode kernel vs closed form | < {kernel['tolerance']:.0e} | "
        f"{max(level['relative_error'] for level in kernel['levels']):.2e} | "
        f"{'PASS' if kernel['passed'] else 'FAIL'} |"
    )
    add(
        f"| kernel constant: `mu_BEM/mu_local -> 1` | < "
        f"{TOLERANCES['asymptote_limit_relative']:.0%} | "
        f"{wet['asymptote_limit']['relative_error']:.2%} | "
        f"{'PASS' if wet['asymptote_limit']['relative_error'] < TOLERANCES['asymptote_limit_relative'] else 'FAIL'} |"
    )
    add(
        f"| wet frequencies vs mesh-independent value | < {wet['tolerance']:.0%} | "
        f"{max(wet['production_error_vs_extrapolated']):.2%} | "
        f"{'PASS' if wet['passed'] else 'FAIL'} |"
    )
    add(
        f"| vertical RAO peak position vs coupled eigenfrequency | < "
        f"{rao['tolerance']:.0%} | {rao['peak_error_vs_wet_frequency']:.2%} | "
        f"{'PASS' if rao['peak_error_vs_wet_frequency'] < rao['tolerance'] else 'FAIL'} |"
    )
    add(
        f"| bending RAO peak position vs coupled eigenfrequency | < "
        f"{rao['tolerance']:.0%} | {rao['bending_peak_error_vs_wet_frequency']:.2%} | "
        f"{'PASS' if rao['bending_peak_error_vs_wet_frequency'] < rao['tolerance'] else 'FAIL'} |"
    )
    add(
        f"| fundamental modal peak vs `omega_1` | < {rao['tolerance']:.0%} | "
        f"{rao['fundamental_modal_peak_error_vs_wet']:.2%} | "
        f"{'PASS' if rao['fundamental_modal_peak_error_vs_wet'] < rao['tolerance'] else 'FAIL'} |"
    )
    add(
        f"| quasi-static RAO vs closed form | < {TOLERANCES['quasi_static_relative']:.0e} | "
        f"{rao['quasi_static_error']:.2e} | "
        f"{'PASS' if rao['quasi_static_error'] < TOLERANCES['quasi_static_relative'] else 'FAIL'} |"
    )
    add(
        f"| dispersion cross-check vs `tensorbem.ice` | < 1e-12 | "
        f"{wet['dispersion_crosscheck_max_error']:.1e} | "
        f"{'PASS' if wet['dispersion_crosscheck_max_error'] < 1e-12 else 'FAIL'} |"
    )
    add(
        f"| partitioned coupling residual | < {coupling['tolerance']:.0e} | "
        f"{coupling['final_omega_residual']:.2e} (omega), "
        f"{coupling['final_pressure_residual']:.2e} (pressure) | "
        f"{'PASS' if coupling['passed'] else 'FAIL'} |"
    )
    add("")
    add("## 6. Limitations")
    add("")
    add("* The hydrodynamic model is the **zero-frequency** one: added mass is real and")
    add("  frequency independent and there is **no radiation damping** (an incompressible")
    add("  fluid bounded by a rigid baffle radiates no acoustic energy).  The RAO damping")
    add("  is structural only, so the peaks are sharper than a physical wetted panel would")
    add("  show; section 3.3 gives the peak position for a range of `zeta` instead of")
    add("  asserting a damping model.")
    add("* Excitation is Froude-Krylov only; the scattered field of a flush panel in an")
    add("  infinite baffle is not represented.  A *floating* plate would additionally")
    add("  require diffraction (see the caveat in `hydroelastic.infinite_plate_response`).")
    add("* Kirchhoff plate: no shear deformation, no rotary inertia, no geometric")
    add("  nonlinearity, small deflections only.")
    add("* The panel is simply supported; clamped or free edges change the wet/dry ratio")
    add("  through the mode shape rather than through the kernel and are not covered.")
    add("* Absolute validation against experiment or against an independent hydroelastic")
    add("  code is **WITHHELD**: every number above is a discretisation/consistency")
    add("  statement for the stated model.")
    add("")
    return "\n".join(lines) + "\n"


def _write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot(report: dict[str, Any], outdir: Path | None = None) -> list[Path]:
    """Produce the two comparison figures (matplotlib, Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = (OUTPUT_ROOT / "figures") if outdir is None else Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # --- figure 1: kernel convergence and asymptote -------------------------
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    levels = report["kernel"]["levels"]
    panels = [level["panels"] for level in levels]
    errors = [level["relative_error"] for level in levels]
    axes[0].loglog(panels, errors, "o-", label="panel quadrature")
    axes[0].axhline(
        report["kernel"]["tolerance"],
        color="r",
        ls="--",
        label=f"tolerance {report['kernel']['tolerance']:.0e}",
    )
    axes[0].set_xlabel("panel count")
    axes[0].set_ylabel("rel. error vs closed form")
    axes[0].set_title("uniform-mode added mass")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    sweep = report["wet_modes"]["asymptote_sweep"]
    ka = [entry["ka"] for entry in sweep]
    ratio = [entry["ratio"] for entry in sweep]
    axes[1].semilogx(ka, ratio, "s-")
    axes[1].axhline(1.0, color="k", ls=":")
    axes[1].set_xlabel(r"$k a$")
    axes[1].set_ylabel(r"$\mu_{\rm BEM} / \mu_{\rm local}$")
    axes[1].set_title("added mass vs plane-wave asymptote")
    axes[1].grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    path = outdir / "bem_hydroelastic_kernel.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    paths.append(path)

    # --- figure 2: RAO curves and damping sweep ----------------------------
    panel = _panel()
    modes = _modes(panel)
    added_mass = hyd.panel_added_mass_matrix(
        panel, modes, nx=PRODUCTION_LEVEL[0], ny=PRODUCTION_LEVEL[1]
    )
    wet = hyd.wet_modes(panel, modes, added_mass)
    omega_1 = float(wet.omegas[0])
    omega = np.linspace(0.02 * omega_1, 1.5 * omega_1, RAO_SWEEP_POINTS)
    base = hyd.modal_rao(
        panel, modes, added_mass, omega, wet_omegas=wet.omegas, damping_ratio=0.02
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].plot(omega / omega_1, base.vertical_rao, "-", label="vertical $w_c/A$")
    axes[0].set_xlabel(r"$\omega/\omega_{\rm wet,1}$")
    axes[0].set_ylabel("vertical RAO [m/m]")
    twin = axes[0].twinx()
    twin.plot(omega / omega_1, base.bending_rao, "--", color="tab:orange",
              label="bending $M_{xx}/A$")
    twin.set_ylabel(r"bending RAO [N m/m$^2$]")
    axes[0].axvline(1.0, color="k", ls=":", label="coupled $\\omega_{\\rm wet,1}$")
    peak = report["rao"]["peak_frequency_rad_s"] / omega_1
    axes[0].axvline(peak, color="r", ls="--", alpha=0.6, label="RAO peak")
    axes[0].set_title("wave-excited response")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    for zeta in RAO_DAMPING_RATIOS:
        result = hyd.modal_rao(
            panel, modes, added_mass, omega, wet_omegas=wet.omegas, damping_ratio=zeta
        )
        axes[1].plot(omega / omega_1, result.vertical_rao, label=f"$\\zeta={zeta}$")
    axes[1].axvline(1.0, color="k", ls=":")
    axes[1].set_xlim(0.7, 1.3)
    axes[1].set_xlabel(r"$\omega/\omega_{\rm wet,1}$")
    axes[1].set_ylabel("vertical RAO [m/m]")
    axes[1].set_title("damping sweep near resonance")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    path = outdir / "bem_hydroelastic_rao.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    paths.append(path)
    return paths


def write_artifacts(report: dict[str, Any], *, figures: bool = True) -> list[Path]:
    """Write ``results.json``, ``report.md`` and the figures."""
    written: list[Path] = []
    _write_json(report, OUTPUT_ROOT / "results.json")
    written.append(OUTPUT_ROOT / "results.json")
    document = render_document(report)
    (OUTPUT_ROOT / "report.md").write_text(document, encoding="utf-8")
    written.append(OUTPUT_ROOT / "report.md")
    DOC_PATH.write_text(document, encoding="utf-8")
    written.append(DOC_PATH)
    if figures:
        written.extend(plot(report))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", action="store_true", help="also regenerate the figures")
    parser.add_argument(
        "--print", action="store_true", help="print the JSON report to stdout"
    )
    args = parser.parse_args(argv)
    report = run()
    written = write_artifacts(report, figures=args.figure)
    if args.print:
        print(json.dumps(report, indent=2, sort_keys=True))
    print(f"all criteria passed: {report['all_criteria_passed']}")
    for path in written:
        print(f"wrote {path}")
    return 0 if report["all_criteria_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())