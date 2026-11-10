"""Campaign: coupled Cummins memory-kernel closure on a real BEM sweep.

This is the multi-degree-of-freedom counterpart of
``tensorbem.benchmarks.cummins_memory_closure`` (which is single-DOF on Hulme's
reference sphere) and of ``cummins_wigley_hull`` (single-DOF on a prescribed
sweep). Here the coefficients themselves come out of *this* repository's own
free-surface boundary-element solver, and both degrees of freedom sweep and
heave are carried together, so the cross coefficients are exercised.

The case is a deeply submerged, tilted elliptical section (:math:`a=1.2` m,
:math:`b=0.6` m, tilt 20 deg, centre 3 m below the free surface, 48 panels).
The tilt is what makes the problem coupled: for a section with a vertical plane
of symmetry, :math:`B_{xz}` vanishes identically and a matrix closure would be
four scalar closures in disguise. The section is the midship section of the
equivalent prismatic hull that the platform geometry gate is applied to.

Gates
-----
G0  geometry  equivalent prism passes the platform Cb/L/B/nabla gate; corrupt
              inputs are rejected.
C0  sampling  the kernel time step satisfies the aliasing budget
              ``eps_tau = max_omega sum_{n>=1} |B(omega + 2 pi n/dt)| / max|B|
              < 1%``, and the kernel is truncated where ``|K|/max|K| < 0.5%``.
C1  transform ``B -> K -> (B, A)`` for all four entries, <= 2%.
C2  harmonic  time-domain convolution under prescribed two-mode motion against
              ``[omega^2 A - i omega B] X e^{i omega t}``, pointwise <= 2%.
C3  causality ``K(t<0) = 0``; ``K_ij(0+) = (2/pi) int B_ij domega`` including
              the *sign* of the cross entry; reciprocity of the sweep.
C4  decay     coupled free decay integrated from the kernel vs the
              frequency-domain quadratic form, <= 5%.
C5  negative  ``B = 0`` must give ``zeta_decay = 0`` (<= 0.5% of the nominal).
C6  envelope  constant-added-mass substitute vs ``omega_p/omega_n``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from tensorbem import memory_kernel_matrix as mkm
from tensorbem.free_surface_solver import solve_radiation_indirect
from tensorbem.geometry import PanelMesh, ellipse_mesh

DENSITY = 1025.0
GRAVITY = 9.80665
PANELS = 48
SEMI_MAJOR = 1.2
SEMI_MINOR = 0.6
TILT_DEGREES = 20.0
SUBMERGENCE = 3.0
PRISM_LENGTH = 12.0

#: radiation-sweep cache; the boundary-element solves are the expensive part and
#: the coefficients are a property of the case, not of the analysis
CACHE = Path(__file__).resolve().parents[3] / "docs" / "cummins_matrix_closure" / "radiation_sweep.json"


def rotated_submerged_section(
    panel_count: int = PANELS,
    semi_major: float = SEMI_MAJOR,
    semi_minor: float = SEMI_MINOR,
    tilt_degrees: float = TILT_DEGREES,
    submergence: float = SUBMERGENCE,
) -> PanelMesh:
    """Counter-clockwise elliptical section, tilted, entirely below ``z = 0``."""
    base = ellipse_mesh(panel_count, semi_major, semi_minor)
    angle = math.radians(float(tilt_degrees))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    vertices = base.vertices @ rotation.T
    vertices[:, 1] -= float(submergence)
    if np.any(vertices[:, 1] >= 0.0):
        raise ValueError("the section must lie entirely below the free surface")
    return PanelMesh(vertices)


def section_geometry(
    panel_count: int = PANELS,
    semi_major: float = SEMI_MAJOR,
    semi_minor: float = SEMI_MINOR,
    tilt_degrees: float = TILT_DEGREES,
    prism_length: float = PRISM_LENGTH,
) -> dict:
    """Equivalent prismatic hull that carries the studied section amidships."""
    angle = math.radians(float(tilt_degrees))
    half_beam = math.hypot(semi_major * math.cos(angle), semi_minor * math.sin(angle))
    half_height = math.hypot(semi_major * math.sin(angle), semi_minor * math.cos(angle))
    beam = 2.0 * half_beam
    depth = 2.0 * half_height
    area = math.pi * semi_major * semi_minor
    displacement = area * float(prism_length) * DENSITY
    wetted = 2.0 * math.pi * math.sqrt(
        0.5 * (semi_major ** 2 + semi_minor ** 2)
    ) * float(prism_length)
    return {
        "length": float(prism_length),
        "beam": float(beam),
        "draft": float(depth),
        "midship_area": float(area),
        "displacement": float(displacement),
        "wetted_surface": float(wetted),
        "block_coefficient": float(area / (beam * depth)),
        "density": DENSITY,
        "section": {
            "semi_major": float(semi_major),
            "semi_minor": float(semi_minor),
            "tilt_degrees": float(tilt_degrees),
            "submergence": float(SUBMERGENCE),
            "panels": int(panel_count),
        },
    }


def geometry_gate(geometry: dict, *, tolerance: float = 0.05) -> dict:
    """Platform D3 contract: ``Cb`` in (0.3, 1] and ``|nabla - L B T Cb| <= 5%``.

    The platform's resistance validator gates geometry exactly this way before
    any flow model runs (``/api/v1/calculations/resistance``); the same contract
    is reproduced here so the campaign carries no service dependency, with the
    live endpoint exercised separately in the report.
    """
    length = float(geometry["length"])
    beam = float(geometry["beam"])
    draft = float(geometry["draft"])
    displacement = float(geometry["displacement"])
    block = float(geometry["block_coefficient"])
    volume = displacement / float(geometry.get("density", DENSITY))
    implied = length * beam * draft * block
    inconsistency = abs(volume - implied) / volume if volume > 0.0 else math.inf
    accepted = bool(0.3 < block <= 1.0) and bool(inconsistency <= tolerance)
    negative_controls = {}
    inflated = dict(geometry)
    inflated["displacement"] = displacement * 1.25
    negative_controls["inflated_displacement"] = _gate_rejects(inflated, tolerance)
    impossible = dict(geometry)
    impossible["block_coefficient"] = 1.42
    negative_controls["impossible_block_coefficient"] = _gate_rejects(impossible, tolerance)
    return {
        "block_coefficient": block,
        "length": length,
        "beam": beam,
        "draft": draft,
        "volume": volume,
        "implied_volume": implied,
        "displacement_consistency": float(inconsistency),
        "tolerance": float(tolerance),
        "accepted": accepted,
        "negative_controls": negative_controls,
        "negative_controls_rejected": bool(all(negative_controls.values())),
    }


def _gate_rejects(geometry: dict, tolerance: float) -> bool:
    block = float(geometry["block_coefficient"])
    volume = float(geometry["displacement"]) / float(geometry.get("density", DENSITY))
    implied = float(geometry["length"]) * float(geometry["beam"]) * float(geometry["draft"]) * block
    inconsistency = abs(volume - implied) / volume if volume > 0.0 else math.inf
    return not (bool(0.3 < block <= 1.0) and bool(inconsistency <= tolerance))


def _radiation_pair(
    mesh: PanelMesh, omega: float, *, panel_nodes: int = 24
) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(A, B)`` 2x2 matrices and solver diagnostics at one frequency."""
    nu = omega ** 2 / GRAVITY
    sway = solve_radiation_indirect(mesh, nu, "sway", density=DENSITY, panel_nodes=panel_nodes)
    heave = solve_radiation_indirect(mesh, nu, "heave", density=DENSITY, panel_nodes=panel_nodes)
    added = np.array(
        [[sway.added_mass[0], heave.added_mass[0]], [sway.added_mass[1], heave.added_mass[1]]]
    )
    damp = np.array(
        [[sway.damping[0], heave.damping[0]], [sway.damping[1], heave.damping[1]]]
    )
    diagnostics = {
        "residual": float(max(sway.relative_residual, heave.relative_residual)),
        "condition": float(max(sway.condition_number, heave.condition_number)),
    }
    return added, damp, diagnostics


def coupled_radiation_sweep(
    omega: Sequence[float],
    *,
    panels: int = PANELS,
    panel_nodes: int = 24,
) -> dict:
    """Solve the coupled radiation problem at every ``omega`` and return matrices."""
    frequencies = np.asarray(omega, dtype=float)
    mesh = rotated_submerged_section(panels)
    added = np.zeros((frequencies.size, 2, 2))
    damp = np.zeros((frequencies.size, 2, 2))
    residual = 0.0
    condition = 0.0
    for index, value in enumerate(frequencies):
        a, b, diagnostics = _radiation_pair(mesh, float(value), panel_nodes=panel_nodes)
        added[index] = a
        damp[index] = b
        residual = max(residual, diagnostics["residual"])
        condition = max(condition, diagnostics["condition"])
    return {
        "omega": frequencies,
        "added_mass": added,
        "damping": damp,
        "provenance": {
            "solver": "tensorbem.free_surface_solver.solve_radiation_indirect",
            "panels": int(panels),
            "panel_nodes": int(panel_nodes),
            "semi_major": SEMI_MAJOR,
            "semi_minor": SEMI_MINOR,
            "tilt_degrees": TILT_DEGREES,
            "submergence": SUBMERGENCE,
            "density": DENSITY,
            "max_linear_residual": float(residual),
            "max_condition_number": float(condition),
        },
    }


def initial_sweep(cache: Path | None = None, *, refresh: bool = False) -> dict:
    """Coarse sweep (log-spaced, 20 nodes from omega=0.44 to 13.3 rad/s)."""
    path = Path(cache) if cache is not None else CACHE
    if path.exists() and not refresh:
        payload = json.loads(path.read_text())
        return {
            "omega": np.asarray(payload["omega"], dtype=float),
            "added_mass": np.asarray(payload["added_mass"], dtype=float),
            "damping": np.asarray(payload["damping"], dtype=float),
            "provenance": payload["provenance"],
            "cache": str(path),
            "refinements": payload.get("refinements", []),
        }
    omega = np.geomspace(0.44, 13.3, 20)
    sweep = coupled_radiation_sweep(omega)
    sweep["cache"] = None
    sweep["refinements"] = []
    return sweep


def refine_sweep(sweep: dict, *, threshold: float = 0.25, budget: int = 8, panels: int = PANELS) -> dict:
    """Insert midpoints where ``|B|`` turns over faster than ``threshold``.

    The criterion that matters downstream is the resolution of the *damping*
    peak, because the kernel and the decay band both live there. Refinement stops
    when no interval violates the threshold or the extra-solve budget is spent;
    every insertion is recorded so the adaptivity is auditable.
    """
    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)
    damp = np.asarray(sweep["damping"], dtype=float)
    refinements = list(sweep.get("refinements", []))
    spent = 0
    mesh = rotated_submerged_section(panels)
    while spent < int(budget):
        scale = float(np.max(np.abs(damp)))
        change = np.abs(np.diff(damp, axis=0)).max(axis=(1, 2)) / scale
        candidates = np.argsort(change)[::-1]
        inserted = False
        for index in candidates:
            if change[index] <= threshold:
                break
            middle = math.sqrt(omega[index] * omega[index + 1])
            if np.any(np.isclose(omega, middle, rtol=1e-6)):
                continue
            a, b, _ = _radiation_pair(mesh, middle)
            omega = np.insert(omega, index + 1, middle)
            added = np.insert(added, index + 1, a, axis=0)
            damp = np.insert(damp, index + 1, b, axis=0)
            refinements.append(
                {
                    "omega": float(middle),
                    "interval": [float(omega[index]), float(omega[index + 2])],
                    "relative_change_before": float(change[index]),
                }
            )
            spent += 1
            inserted = True
            break
        if not inserted:
            break
    return {
        **sweep,
        "omega": omega,
        "added_mass": added,
        "damping": damp,
        "refinements": refinements,
        "refinement_statistics": {
            "insertions": len(refinements),
            "threshold": float(threshold),
            "budget": int(budget),
            "max_interval_relative_change": float(
                np.max(np.abs(np.diff(damp, axis=0)).max(axis=(1, 2))) / float(np.max(np.abs(damp)))
            ),
        },
    }


def save_sweep(sweep: dict, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "omega": np.asarray(sweep["omega"], dtype=float).tolist(),
        "added_mass": np.asarray(sweep["added_mass"], dtype=float).tolist(),
        "damping": np.asarray(sweep["damping"], dtype=float).tolist(),
        "provenance": sweep["provenance"],
        "refinements": sweep.get("refinements", []),
        "refinement_statistics": sweep.get("refinement_statistics", {}),
    }
    path.write_text(json.dumps(payload, indent=1))
    return path

# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def _verdict(identifier: str, title: str, threshold: str, passed: bool, values: dict) -> dict:
    return {
        "id": identifier,
        "title": title,
        "threshold": threshold,
        "passed": bool(passed),
        "values": values,
    }


def fit_added_mass_infinity(sweep: dict, *, tail_fraction: float = 0.3) -> tuple[np.ndarray, dict]:
    """Two-term high-frequency tail fit of ``A_ij``, entry by entry."""
    from tensorbem.causality import fit_added_mass_asymptote

    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)
    infinity = np.zeros((2, 2))
    residual = 0.0
    points = []
    for i in range(2):
        for j in range(2):
            fit = fit_added_mass_asymptote(omega, added[:, i, j], tail_fraction=tail_fraction)
            infinity[i, j] = fit.added_mass_inf
            residual = max(residual, fit.relative_residual)
            points.append(fit.points)
    infinity = 0.5 * (infinity + infinity.T)
    return infinity, {
        "tail_fraction": float(tail_fraction),
        "max_relative_residual": float(residual),
        "points": int(min(points)),
    }


def kernel_grid(sweep: dict, *, refinement: float = 6.0, span: float | None = None) -> tuple[np.ndarray, dict]:
    """Kernel time grid from the alias budget, refined for the oscillator band."""
    omega = np.asarray(sweep["omega"], dtype=float)
    damp = np.asarray(sweep["damping"], dtype=float)
    budget_dt, report = mkm.choose_time_step(omega, damp, threshold=1.0e-2)
    dt = float(budget_dt) / float(refinement)
    span = float(span) if span is not None else float(math.pi / np.min(np.diff(omega)))
    count = int(math.ceil(span / dt))
    return np.arange(count + 1) * dt, {
        "budget_dt": float(budget_dt),
        "dt": dt,
        "refinement": float(refinement),
        "span": float(count * dt),
        "alias_report": report,
    }


def criterion_sampling(sweep: dict, grid: dict, kernel: mkm.CoupledKernelMatrix) -> dict:
    """C0: aliasing budget and kernel truncation."""
    omega = np.asarray(sweep["omega"], dtype=float)
    damp = np.asarray(sweep["damping"], dtype=float)
    budget = mkm.alias_budget(omega, damp, grid["budget_dt"], threshold=1.0e-2)
    used = mkm.alias_budget(omega, damp, grid["dt"], threshold=1.0e-2)
    band = np.geomspace(float(omega[0]), 6.0, 10)
    truncation = mkm.kernel_truncation(kernel, relative=5.0e-3, frequencies=band)
    # The budget is only informative if it is shown to bite somewhere: sweep the
    # time step from the Nyquist limit upwards and report where the fold-back
    # crosses one percent. For this body the damping dies by omega ~ 5 rad/s
    # while the sweep reaches 13.3, so the budget is satisfied with three
    # decades of margin and the *binding* constraint on dt comes from the
    # oscillator (period/120), not from aliasing -- which is exactly what a
    # campaign claiming to have met this criterion has to show.
    scaling = []
    for multiplier in (1.0, 2.0, 4.0, 8.0, 16.0):
        step = math.pi / float(omega[-1]) * multiplier
        report = mkm.check_alias_budget(omega, damp, step)
        scaling.append(
            {
                "dt_over_nyquist_limit": multiplier,
                "dt": float(step),
                "nyquist": float(report["nyquist"]),
                "epsilon_tau": float(report["epsilon_tau"]),
                "passed": bool(report["epsilon_tau"] <= 1.0e-2),
            }
        )
    passed = bool(
        budget["passed"]
        and used["passed"]
        and truncation["worst_tail_fraction_after_cut"] <= 5.0e-3
    )
    return _verdict(
        "C0",
        "sampling contract: alias budget eps_tau < 1% and kernel tail < 0.5%",
        "eps_tau <= 1%, |K(tau_max)|/max|K| <= 0.5%",
        passed,
        {
            "budget_dt": grid["budget_dt"],
            "budget_epsilon_tau": budget["epsilon_tau"],
            "budget_worst_entry": list(budget["worst_entry"]),
            "used_dt": grid["dt"],
            "used_epsilon_tau": used["epsilon_tau"],
            "nyquist_over_omega_max": (math.pi / budget["dt"]) / float(omega[-1]),
            "tail_share_of_budget": max(
                entry["tail_share"] for entry in budget["per_entry"].values()
            ),
            "alias_budget_vs_dt": scaling,
            "truncation": {
                "tau_max": truncation["tau_max"],
                "relative_threshold": truncation["relative_threshold"],
                "worst_tail_fraction_after_cut": truncation["worst_tail_fraction_after_cut"],
                "tail_transfer_fraction": truncation["tail_transfer_fraction"],
                "sampled_fraction": truncation["sampled_fraction"],
            },
        },
    )


def criterion_round_trip(sweep: dict, infinity: np.ndarray, kernel: mkm.CoupledKernelMatrix) -> dict:
    """C1: ``B -> K -> (B, A)`` on all four entries, <= 2%."""
    result = mkm.matrix_round_trip(
        sweep["omega"], sweep["added_mass"], sweep["damping"], infinity, kernel
    )
    return _verdict(
        "C1",
        "frequency->time->frequency round trip on every entry",
        "max relative error <= 2%",
        result["passed"],
        {"band": result["band"], "per_entry": {
            f"{i}{j}": entry for (i, j), entry in result["per_entry"].items()
        }, "worst_relative_error": result["worst_relative_error"], "worst_entry": result["worst_entry"]},
    )


def criterion_harmonic(
    sweep: dict,
    infinity: np.ndarray,
    kernel: mkm.CoupledKernelMatrix,
    *,
    frequencies: Sequence[float] = (0.9, 1.3, 1.8, 2.6, 3.6),
    amplitude: Sequence[float] = (1.0, 0.7),
    phase: Sequence[float] = (0.0, 0.6),
) -> dict:
    """C2: time-domain convolution vs the frequency-domain radiation force."""
    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)
    damp = np.asarray(sweep["damping"], dtype=float)
    rows = []
    worst = 0.0
    for value in frequencies:
        a = np.array([[np.interp(value, omega, added[:, i, j]) for j in range(2)] for i in range(2)])
        b = np.array([[np.interp(value, omega, damp[:, i, j]) for j in range(2)] for i in range(2)])
        result = mkm.harmonic_radiation_force(
            kernel,
            infinity,
            omega=float(value),
            amplitude=amplitude,
            phase=phase,
            added_mass=a,
            damping=b,
        )
        rows.append(
            {
                "omega": float(value),
                "periods_per_second": float(value / (2.0 * math.pi)),
                "worst_relative_error": result["worst_relative_error"],
                "per_mode": {str(k): v for k, v in result["per_mode"].items()},
                "steady_start": result["steady_start"],
            }
        )
        worst = max(worst, result["worst_relative_error"])
    return _verdict(
        "C2",
        "prescribed-motion convolution vs [omega^2 A - i omega B] X e^{i omega t}",
        "pointwise relative error <= 2%",
        worst <= 0.02,
        {"frequencies": list(frequencies), "rows": rows, "worst_relative_error": float(worst)},
    )


def criterion_causality(sweep: dict, kernel: mkm.CoupledKernelMatrix) -> dict:
    """C3: ``K(t<0) = 0``, ``K_ij(0+)`` magnitude and sign, reciprocity."""
    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)
    damp = np.asarray(sweep["damping"], dtype=float)
    dt = kernel.dt
    negative = kernel.at(np.array([-10.0 * dt, -dt, -0.5 * dt, -1.0e-9]))
    violation = float(np.max(np.abs(negative)))
    entries = {}
    worst_scale = 0.0
    sign_mismatch = 0
    for i in range(2):
        for j in range(2):
            # (2/pi) int_0^inf B domega on the tabulated band, plus the low-band
            # closure and the high-band bound the kernel carries.
            trapezoid = np.trapezoid(damp[:, i, j], omega)
            predicted = float(2.0 / math.pi) * (
                trapezoid
                + kernel.damping_scale[i, j] * omega[0] ** 4 / 4.0
                + 0.5 * abs(float(damp[-1, i, j])) * float(omega[-1])
            )
            measured = float(kernel.entry(i, j)[0])
            scale = max(abs(predicted), 1e-12)
            relative = abs(measured - predicted) / scale
            same_sign = bool(np.sign(measured) == np.sign(predicted)) or predicted == 0.0
            if not same_sign:
                sign_mismatch += 1
            entries[f"{i}{j}"] = {
                "kernel_at_zero": measured,
                "predicted_from_sweep": predicted,
                "relative_error": float(relative),
                "sign_agrees": same_sign,
                "diagonal_positive": bool(measured > 0.0) if i == j else None,
            }
            worst_scale = max(worst_scale, relative)
    asymmetry_added = float(
        np.max(np.abs(added - added.swapaxes(1, 2))) / np.max(np.abs(added))
    )
    asymmetry_damping = float(
        np.max(np.abs(damp - damp.swapaxes(1, 2))) / np.max(np.abs(damp))
    )
    diagonal_positive = all(
        entries[f"{i}{i}"]["kernel_at_zero"] > 0.0 for i in range(2)
    )
    cross_negative = entries["01"]["kernel_at_zero"] < 0.0 and entries["10"]["kernel_at_zero"] < 0.0
    passed = bool(
        violation == 0.0
        and worst_scale <= 0.02
        and sign_mismatch == 0
        and diagonal_positive
        and cross_negative
    )
    return _verdict(
        "C3",
        "causality, K(0+) jump sign from the sweep, reciprocity",
        "K(t<0)=0; K_ij(0+) within 2% of (2/pi)int B_ij; signs as predicted",
        passed,
        {
            "kernel_at_negative_time_max_abs": violation,
            "entries": entries,
            "diagonal_positive": bool(diagonal_positive),
            "cross_entry_negative": bool(cross_negative),
            "worst_jump_relative_error": float(worst_scale),
            "reciprocity": {
                "added_mass_asymmetry": asymmetry_added,
                "damping_asymmetry": asymmetry_damping,
            },
        },
    )


def oscillator(
    mass_per_length: float,
    targets: Sequence[float],
    added_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Test-stand oscillator: structural mass and stiffness tuned to ``targets``.

    ``C_ii = (M_ii + A_ii(omega_target)) omega_target^2`` puts each undamped
    resonance on its target frequency, so the free decay really samples the band
    where the memory matters instead of an arbitrary corner of the sweep.
    """
    mass = np.eye(2) * float(mass_per_length)
    stiffness = np.diag(
        [
            (mass[i, i] + float(added_reference[i, i])) * float(targets[i]) ** 2
            for i in range(2)
        ]
    )
    return mass, stiffness, {
        "mass_per_length": float(mass_per_length),
        "target_frequencies": [float(t) for t in targets],
        "added_mass_at_targets": np.asarray(added_reference, dtype=float).tolist(),
    }


def calibrate_mass(
    sweep: dict,
    targets: Sequence[float],
    *,
    target_zeta: float = 0.05,
    bounds: tuple[float, float] = (0.3, 40.0),
    iterations: int = 40,
) -> dict:
    """Structural mass (in units of the displaced mass per metre) for ``zeta``.

    The damping ratio of a two-dimensional section is large at the damping peak,
    so the lightest test stand would be heavily damped and the amplitude envelope
    would not be resolvable. The mass is therefore *chosen* -- by bisection on the
    quadratic form of mode 0 -- so that the nominal damping ratio is
    ``target_zeta``. It is a property of the test stand, not of the ship, and is
    reported as such.
    """
    geometry = section_geometry()
    displaced = DENSITY * geometry["midship_area"]
    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)
    damp = np.asarray(sweep["damping"], dtype=float)

    def added_at(value: float) -> np.ndarray:
        return np.array([[np.interp(value, omega, added[:, i, j]) for j in range(2)] for i in range(2)])

    def damp_at(value: float) -> np.ndarray:
        return np.array([[np.interp(value, omega, damp[:, i, j]) for j in range(2)] for i in range(2)])

    reference = added_at(0.5 * (targets[0] + targets[1]))
    history = []

    def zeta_for(scale: float) -> tuple[float, np.ndarray]:
        mass, stiffness, _ = oscillator(scale * displaced, targets, reference)
        frequencies, vectors = mkm.mode_shape(mass, reference, stiffness)
        value = mkm.modal_damping_ratio(
            frequencies[0], mass, added_at(frequencies[0]), damp_at(frequencies[0]), vectors[:, 0]
        )["zeta"]
        history.append({"mass_scale": float(scale), "zeta": float(value)})
        return float(value), vectors[:, 0]

    low, high = bounds
    for _ in range(int(iterations)):
        middle = math.sqrt(low * high)
        value, _ = zeta_for(middle)
        if value > target_zeta:
            low = middle
        else:
            high = middle
    scale = math.sqrt(low * high)
    mass, stiffness, stand = oscillator(scale * displaced, targets, reference)
    values, vectors = mkm.mode_shape(mass, reference, stiffness)
    zetas = [
        mkm.modal_damping_ratio(
            values[index], mass, added_at(values[index]), damp_at(values[index]), vectors[:, index]
        )["zeta"]
        for index in range(2)
    ]
    stand.update(
        {
            "mass_scale": float(scale),
            "mass_scale_units": "displaced mass per unit length",
            "displaced_mass_per_length": float(displaced),
            "mode_frequencies": values.tolist(),
            "mode_damping_ratios": zetas,
            "target_zeta": float(target_zeta),
        }
    )
    return {"mass": mass, "stiffness": stiffness, "stand": stand}


def decay_study(
    sweep: dict,
    infinity: np.ndarray,
    kernel: mkm.CoupledKernelMatrix,
    *,
    mass: np.ndarray,
    stiffness: np.ndarray,
    targets: Sequence[float],
    duration_periods: float = 8.0,
    samples_per_period: float = 200.0,
) -> dict:
    """C4 core: coupled free decay against the frequency-domain damping ratios.

    Two anchors are reported for every mode:

    * ``zeta_quadratic`` -- ``v^T B v / (2 omega v^T (M + A) v)``, the criterion
      the campaign was posed with;
    * ``zeta_exact`` -- the imaginary part of the complex resonance of the
      frequency-domain system (:func:`tensorbem.memory_kernel_matrix.complex_resonance`),
      which is the exact linear-theory decay rate.

    The time integration is started from the complex mode itself,
    ``x(0) = Re v``, ``x'(0) = Re(i Omega v)``, so the record is a single damped
    sinusoid and the envelope fit measures one decay rate rather than a beat
    between two modes with different phases.
    """
    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)
    damp = np.asarray(sweep["damping"], dtype=float)

    def added_at(value: float) -> np.ndarray:
        return np.array([[np.interp(value, omega, added[:, i, j]) for j in range(2)] for i in range(2)])

    def damp_at(value: float) -> np.ndarray:
        return np.array([[np.interp(value, omega, damp[:, i, j]) for j in range(2)] for i in range(2)])

    roots = mkm.complex_resonances(
        omega,
        added,
        damp,
        infinity,
        mass,
        stiffness,
        omega_window=(0.5 * float(targets[0]), 1.4 * float(targets[1])),
        gamma_window=(0.0, 0.8),
    )
    if len(roots) < 2:
        raise RuntimeError(
            f"only {len(roots)} complex resonances found in the scan window; "
            "refusing to report a mode-by-mode decay comparison"
        )
    kernel_roots = mkm.kernel_resonances(
        kernel,
        infinity,
        mass=mass,
        stiffness=stiffness,
        guesses=[entry["omega"] for entry in roots],
    )
    rows = []
    for index in range(2):
        root = roots[index]
        kernel_root = kernel_roots[index] if index < len(kernel_roots) else None
        shape = np.asarray(root["mode_shape"], dtype=complex)
        initial_displacement = shape.real
        initial_velocity = (1j * root["root"] * shape).real
        period = 2.0 * math.pi / root["omega"]
        duration = duration_periods * period
        target_dt = period / samples_per_period
        measured = mkm.decay_ratio_richardson(
            kernel,
            infinity,
            mass=mass,
            stiffness=stiffness,
            initial_displacement=initial_displacement,
            initial_velocity=initial_velocity,
            duration=duration,
            dt_target=target_dt,
            omega_guess=root["omega"],
        )
        quadratic = mkm.modal_damping_ratio(
            measured["omega"],
            mass,
            added_at(measured["omega"]),
            damp_at(measured["omega"]),
            initial_displacement,
        )
        error_quadratic = abs(measured["zeta"] - quadratic["zeta"]) / abs(quadratic["zeta"])
        error_exact = abs(measured["zeta"] - root["zeta"]) / abs(root["zeta"])
        rows.append(
            {
                "mode": index,
                "complex_resonance": root["root"].real + 0j if False else {
                    "omega": root["omega"],
                    "gamma": root["gamma"],
                    "zeta": root["zeta"],
                    "mode_shape": root["mode_shape"],
                    "residual_determinant": root["residual_determinant"],
                },
                "measured_frequency": measured["omega"],
                "measured_zeta": measured["zeta"],
                "zeta_quadratic": quadratic["zeta"],
                "zeta_exact": root["zeta"],
                "relative_error_vs_quadratic": float(error_quadratic),
                "relative_error_vs_exact": float(error_exact),
                "kernel_root_zeta": None if kernel_root is None else kernel_root["zeta"],
                "relative_error_vs_kernel_root": None
                if kernel_root is None
                else float(abs(measured["zeta"] - kernel_root["zeta"]) / kernel_root["zeta"]),
                "zeta_at_dt": measured["zeta_at_dt"],
                "zeta_at_2dt": measured["zeta_at_2dt"],
                "dt_refinement_drift_over_zeta": measured["drift_over_zeta"],
                "fit_residual_over_amplitude": measured["fit_residual_over_amplitude"],
                "initial_displacement": initial_displacement.tolist(),
                "duration": float(duration),
                "period": float(period),
                "dt": measured["dt"],
                "samples_per_period": float(samples_per_period),
                "time": measured["time"].tolist(),
                "projection": measured["projection"].tolist(),
            }
        )
    worst = max(row["relative_error_vs_quadratic"] for row in rows)
    return {
        "rows": rows,
        "roots_found": len(roots),
        "worst_relative_error_vs_quadratic": float(worst),
        "worst_relative_error_vs_exact": float(
            max(row["relative_error_vs_exact"] for row in rows)
        ),
        "worst_relative_error_vs_kernel_root": float(
            max(
                row["relative_error_vs_kernel_root"]
                for row in rows
                if row["relative_error_vs_kernel_root"] is not None
            )
        )
        if any(row["relative_error_vs_kernel_root"] is not None for row in rows)
        else None,
        "damping_ratios_quadratic": [row["zeta_quadratic"] for row in rows],
        "damping_ratios_exact": [row["zeta_exact"] for row in rows],
        "damping_ratios_measured": [row["measured_zeta"] for row in rows],
    }


def criterion_decay(study: dict, *, threshold: float = 0.05) -> dict:
    cleaned = [
        {key: value for key, value in row.items() if key not in ("time", "projection")}
        for row in study["rows"]
    ]
    return _verdict(
        "C4",
        "coupled free decay vs the frequency-domain damping ratio",
        "|zeta_decay - zeta_fd| / zeta_fd <= 5% (quadratic form); exact complex root reported alongside",
        study["worst_relative_error_vs_quadratic"] <= threshold,
        {
            "rows": cleaned,
            "worst_relative_error_vs_quadratic": study["worst_relative_error_vs_quadratic"],
            "worst_relative_error_vs_exact": study["worst_relative_error_vs_exact"],
            "worst_relative_error_vs_kernel_root": study["worst_relative_error_vs_kernel_root"],
            "damping_ratios_quadratic": study["damping_ratios_quadratic"],
            "damping_ratios_exact": study["damping_ratios_exact"],
            "damping_ratios_measured": study["damping_ratios_measured"],
        },
    )


def damping_approximation_study(
    sweep: dict,
    infinity: np.ndarray,
    kernel: mkm.CoupledKernelMatrix,
    *,
    targets: Sequence[float],
    damping_ratios: Sequence[float] = (0.02, 0.035, 0.05, 0.08),
) -> dict:
    """Where the single-frequency damping anchor stops being true.

    ``zeta = v^T B v / (2 omega v^T (M + A) v)`` samples the damping at one
    frequency, while the decay itself samples a band of width ``~2 zeta omega``.
    This repeats the free-decay comparison at several test-stand damping ratios,
    so the error of the anchor is measured as a function of the bandwidth that
    the anchor ignores. Nothing here is fitted: each row is an independent run.
    """
    rows = []
    for target_zeta in damping_ratios:
        calibrated = calibrate_mass(sweep, targets, target_zeta=float(target_zeta))
        study = decay_study(
            sweep,
            infinity,
            kernel,
            mass=calibrated["mass"],
            stiffness=calibrated["stiffness"],
            targets=targets,
            duration_periods=8.0,
            samples_per_period=120.0,
        )
        for row in study["rows"]:
            rows.append(
                {
                    "target_zeta": float(target_zeta),
                    "mode": int(row["mode"]),
                    "measured_zeta": row["measured_zeta"],
                    "zeta_quadratic": row["zeta_quadratic"],
                    "zeta_kernel_root": row["kernel_root_zeta"],
                    "amplitude_ratio_at_end": float(
                        math.exp(
                            -row["measured_zeta"]
                            * row["measured_frequency"]
                            * 0.73
                            * row["duration"]
                        )
                    ),
                    "relative_error_vs_quadratic": row["relative_error_vs_quadratic"],
                    "relative_error_vs_kernel_root": row["relative_error_vs_kernel_root"],
                }
            )
    worst = max(rows, key=lambda row: row["relative_error_vs_quadratic"])
    return {
        "rows": rows,
        "worst_relative_error_vs_quadratic": float(worst["relative_error_vs_quadratic"]),
        "worst_case": {"target_zeta": worst["target_zeta"], "mode": worst["mode"]},
    }


def criterion_negative_control(
    sweep: dict,
    kernel: mkm.CoupledKernelMatrix,
    *,
    mass: np.ndarray,
    stiffness: np.ndarray,
    nominal_zeta: float,
    threshold: float = 5.0e-3,
) -> dict:
    """C5: zero damping must give zero decay."""
    zero = np.zeros_like(sweep["damping"])
    zero_kernel = mkm.coupled_memory_kernels(
        sweep["omega"], zero, kernel.tau, low_frequency_model="zero"
    )
    frequency, vectors = mkm.mode_shape(mass, np.zeros((2, 2)), stiffness)
    shape = vectors[:, 0]
    period = 2.0 * math.pi / frequency[0]
    duration = 8.0 * period
    result = mkm.coupled_free_decay(
        mkm._refine_kernel(zero_kernel, period / 120.0),
        np.zeros((2, 2)),
        mass=mass,
        stiffness=stiffness,
        initial_displacement=shape,
        duration=duration,
    )
    projection = result["displacement"] @ mass @ shape
    measured = mkm.decay_ratio_from_series(
        result["time"],
        projection,
        window=(0.25 * duration, 0.98 * duration),
        omega_guess=float(frequency[0]),
    )
    ratio = abs(measured["zeta"]) / abs(nominal_zeta)
    identity = float(np.max(np.abs(zero_kernel.values)))
    return _verdict(
        "C5",
        "negative control: B = 0 gives zeta_decay = 0",
        "|zeta_decay| / zeta_nominal <= 0.5%",
        bool(ratio <= threshold and identity < 1e-12 * max(1.0, float(np.max(np.abs(kernel.values))))),
        {
            "kernel_identically_zero": identity,
            "measured_zeta": measured["zeta"],
            "nominal_zeta": float(nominal_zeta),
            "ratio": float(ratio),
            "threshold": float(threshold),
            "amplitude_residual": float(np.max(np.abs(projection))),
        },
    )


def criterion_envelope(
    sweep: dict,
    infinity: np.ndarray,
    *,
    mass: np.ndarray,
    targets: Sequence[float],
    ratios: Sequence[float] = (2.5, 2.0, 1.6, 1.25, 1.0, 0.8, 0.65, 0.5, 0.4),
) -> dict:
    """C6: the constant-added-mass substitute priced against ``omega_p/omega_n``."""
    result = mkm.constant_matrix_envelope(
        sweep["omega"],
        sweep["added_mass"],
        sweep["damping"],
        infinity,
        mass,
        ratios=ratios,
        targets=targets,
        mode=0,
    )
    limit = 0.05
    safe_band = [
        row["omega_p_over_omega_n"]
        for row in result["rows"]
        if row["amplitude_error_at_natural"] <= limit
        and abs(row["phase_error_degrees"]) <= 10.0
    ]
    passed = bool(result["rows"]) and bool(safe_band)
    return _verdict(
        "C6",
        "error envelope of the constant-added-mass model",
        "curve of peak/phase error vs omega_p/omega_n, with the band where the substitute is admissible",
        passed,
        {
            "rows": result["rows"],
            "omega_p": result["omega_p"],
            "worst_peak_amplitude_error": result["worst_peak_amplitude_error"],
            "worst_phase_error_degrees": result["worst_phase_error_degrees"],
            "safe_band": safe_band,
            "criterion": "amplitude error at the design frequency <= 5% and |phase| <= 10 deg; "
            "the peak-based numbers are reported alongside, flagged when the resonance leaves the scan window",
        },
    )


# ---------------------------------------------------------------------------
# the campaign
# ---------------------------------------------------------------------------
def run_campaign(
    *,
    cache: Path | None = None,
    refresh: bool = False,
    panels: int = PANELS,
    target_zeta: float = 0.012,
    targets: Sequence[float] = (1.8, 2.6),
    sweep_damping_ratios: Sequence[float] | None = (0.008, 0.012, 0.02, 0.05),
    refine_budget: int = 8,
    keep_traces: bool = False,
) -> dict:
    """Run every gate and return the machine-readable report."""
    geometry = section_geometry(panels)
    gate = geometry_gate(geometry)
    if not gate["accepted"] or not gate["negative_controls_rejected"]:
        raise RuntimeError("the geometry gate rejects the case; refusing to run BEM")

    sweep = initial_sweep(cache, refresh=refresh)
    if not sweep.get("refinements"):
        sweep = refine_sweep(sweep, budget=refine_budget, panels=panels)
    if sweep.get("cache") is None or refresh:
        saved = save_sweep(sweep, cache)
        sweep["cache"] = str(saved)
    omega = np.asarray(sweep["omega"], dtype=float)
    added = np.asarray(sweep["added_mass"], dtype=float)
    damp = np.asarray(sweep["damping"], dtype=float)
    mkm.validate_coupled_sweep(omega, added, damp)

    infinity, tail = fit_added_mass_infinity(sweep)
    tau, grid = kernel_grid(sweep)
    kernel = mkm.coupled_memory_kernels(omega, damp, tau)
    calibrated = calibrate_mass(sweep, targets, target_zeta=target_zeta)
    mass = calibrated["mass"]
    stiffness = calibrated["stiffness"]
    stand = calibrated["stand"]

    sampling = criterion_sampling(sweep, grid, kernel)
    round_trip = criterion_round_trip(sweep, infinity, kernel)
    harmonic = criterion_harmonic(sweep, infinity, kernel)
    causality = criterion_causality(sweep, kernel)
    study = decay_study(
        sweep, infinity, kernel, mass=mass, stiffness=stiffness, targets=targets
    )
    decay = criterion_decay(study)
    nominal_zeta = float(np.mean(study["damping_ratios_quadratic"]))
    negative = criterion_negative_control(
        sweep, kernel, mass=mass, stiffness=stiffness, nominal_zeta=nominal_zeta
    )
    envelope = criterion_envelope(sweep, infinity, mass=mass, targets=targets)
    validity = (
        damping_approximation_study(
            sweep, infinity, kernel, targets=targets, damping_ratios=sweep_damping_ratios
        )
        if sweep_damping_ratios is not None
        else None
    )
    decay["values"]["anchor_validity_study"] = validity
    geometry_verdict = _verdict(
        "G0",
        "geometry precondition (platform D3 contract on Cb, L, B, nabla)",
        "0.3 < Cb <= 1 and |nabla - L B T Cb| / nabla <= 5%",
        gate["accepted"] and gate["negative_controls_rejected"],
        gate,
    )
    verdicts = [
        geometry_verdict,
        sampling,
        round_trip,
        harmonic,
        causality,
        decay,
        negative,
        envelope,
    ]
    return {
        "campaign": "cummins_matrix_closure",
        "case": {
            "geometry": geometry,
            "panels": int(panels),
            "sweep_points": int(omega.size),
            "omega_range": [float(omega[0]), float(omega[-1])],
            "refinements": sweep.get("refinements", []),
            "refinement_statistics": {
                **sweep.get("refinement_statistics", {}),
                "insertions": len(sweep.get("refinements", [])),
                "max_interval_relative_change": float(
                    np.max(np.abs(np.diff(damp, axis=0)).max(axis=(1, 2)))
                    / float(np.max(np.abs(damp)))
                ),
                "points": int(omega.size),
            },
            "provenance": sweep.get("provenance", {}),
            "cache": sweep.get("cache"),
        },
        "added_mass_infinity": infinity.tolist(),
        "added_mass_infinity_fit": tail,
        "kernel_grid": grid,
        "oscillator": stand,
        "mass": mass.tolist(),
        "stiffness": stiffness.tolist(),
        "sweep": {
            "omega": omega.tolist(),
            "added_mass": added.tolist(),
            "damping": damp.tolist(),
        },
        "verdicts": verdicts,
        "all_passed": bool(all(verdict["passed"] for verdict in verdicts)),
        "decay_study": {key: value for key, value in study.items() if key != "rows"},
        "trace_source": {"kernel": kernel, "study": study} if keep_traces else None,
    }


def write_report(report: dict, path: Path | None = None) -> Path:
    """Write the campaign JSON, sanitising numpy/array/tuple keys first.

    The report is the audit trail: every number in the prose has to be traceable
    to a key here, so the writer refuses to drop anything it does not understand
    (it raises) instead of silently stringifying it.
    """
    def clean(value):
        if isinstance(value, dict):
            return {
                key if isinstance(key, str) else str(key): clean(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, complex):
            return {"real": value.real, "imag": value.imag}
        if isinstance(value, (str, bool)) or value is None:
            return value
        raise TypeError(f"cannot serialise {type(value)!r}")

    payload = clean({key: value for key, value in report.items() if key != "trace_source"})
    target = Path(path) if path is not None else (
        Path(__file__).resolve().parents[3] / "docs" / "cummins_matrix_closure" / "campaign.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1))
    return target


def _figure_dir() -> Path:
    path = Path(__file__).resolve().parents[3] / "docs" / "cummins_matrix_closure" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_figures(report: dict) -> list[dict]:
    """Publication figures, written as PNG next to the campaign report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = _figure_dir()
    figures: list[dict] = []
    omega = np.asarray(report["sweep"]["omega"], dtype=float)
    added = np.asarray(report["sweep"]["added_mass"], dtype=float)
    damp = np.asarray(report["sweep"]["damping"], dtype=float)
    infinity = np.asarray(report["added_mass_infinity"], dtype=float)

    labels = {(0, 0): "sway-sway", (0, 1): "sway-heave", (1, 1): "heave-heave"}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for (i, j), label in labels.items():
        axes[0].plot(omega, added[:, i, j], "-o", ms=2.5, label=label)
        axes[1].plot(omega, damp[:, i, j], "-o", ms=2.5, label=label)
    for i in range(2):
        axes[0].axhline(infinity[i, i], ls="--", lw=0.8, color="k")
    axes[0].set_xlabel(r"$\omega$ [rad/s]")
    axes[0].set_ylabel(r"$A_{ij}$ [kg/m]")
    axes[0].set_xscale("log")
    axes[0].set_title("added mass (dashed: two-term tail fit $A_{ii}(\\infty)$)")
    axes[1].set_xlabel(r"$\omega$ [rad/s]")
    axes[1].set_ylabel(r"$B_{ij}$ [kg/(m s)]")
    axes[1].set_xscale("log")
    axes[1].set_title("damping (the cross entry is negative)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    path = directory / "fig1_radiation_sweep.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    figures.append(
        {
            "name": "fig1_radiation_sweep",
            "caption": "Coupled radiation coefficients of the tilted submerged section from the in-house BEM "
            "sweep (48 panels, adaptively refined). The cross coefficient is large and negative, which is what "
            "makes the case a genuine matrix problem.",
            "path": str(path),
        }
    )

    source = report.get("trace_source")
    kernel = source["kernel"] if source else None
    study = source["study"] if source else None

    if kernel is not None:
        cut = report["verdicts"][1]["values"]["truncation"]["tau_max"]
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.0), sharex=True)
        for i in range(2):
            for j in range(2):
                axis = axes[i, j]
                axis.plot(kernel.tau, kernel.entry(i, j) / kernel.peak[i, j], lw=1.0)
                axis.axvline(cut, color="r", ls=":", lw=0.9)
                axis.axhline(5.0e-3, color="k", ls="--", lw=0.6)
                axis.axhline(-5.0e-3, color="k", ls="--", lw=0.6)
                axis.set_title(f"$K_{{{i}{j}}}(\\tau)/\\max|K_{{{i}{j}}}|$")
                axis.grid(alpha=0.3)
        for axis in axes[-1]:
            axis.set_xlabel(r"$\tau$ [s]")
        figure.suptitle("memory kernel matrix; dotted line = 0.5% truncation time")
        figure.tight_layout()
        path = directory / "fig2_kernel_matrix.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        figures.append(
            {
                "name": "fig2_kernel_matrix",
                "caption": "Normalised memory kernel matrix K_ij(tau). Each entry opens with its K_ij(0+) jump "
                "and falls inside the 0.5% band (dashed) before the truncation time (dotted).",
                "path": str(path),
            }
        )

        value = 1.8
        a_row = np.array(
            [[np.interp(value, omega, added[:, i, j]) for j in range(2)] for i in range(2)]
        )
        b_row = np.array(
            [[np.interp(value, omega, damp[:, i, j]) for j in range(2)] for i in range(2)]
        )
        trace = mkm.harmonic_radiation_force(
            kernel,
            infinity,
            omega=value,
            amplitude=(1.0, 0.7),
            phase=(0.0, 0.6),
            added_mass=a_row,
            damping=b_row,
        )
        time = np.asarray(trace["time"])
        start = int(round(trace["steady_start"] / trace["dt"]))
        mask = slice(start, time.size)
        figure, axes = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True)
        for i in range(2):
            axes[i].plot(
                time[mask] - time[start], trace["force"][mask, i], lw=1.4, label="Cummins convolution"
            )
            axes[i].plot(
                time[mask] - time[start],
                trace["reference"][mask, i],
                "--",
                lw=1.0,
                label="frequency domain",
            )
            axes[i].set_ylabel(f"$F_{i}$ [N/m]")
            axes[i].grid(alpha=0.3)
            axes[i].legend(fontsize=8)
        axes[-1].set_xlabel("time after steady-state start [s]")
        figure.suptitle(
            rf"radiation force under prescribed motion at $\omega={value}$ rad/s "
            f"(worst pointwise error {trace['worst_relative_error'] * 100:.2f}%)"
        )
        figure.tight_layout()
        path = directory / "fig3_harmonic_closure.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        figures.append(
            {
                "name": "fig3_harmonic_closure",
                "caption": "Time-domain convolution (solid) against the frequency-domain radiation force "
                "(dashed) for both degrees of freedom under a two-mode prescribed motion.",
                "path": str(path),
            }
        )

    if study is not None:
        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
        for row in study["rows"]:
            time = np.asarray(row["time"])
            projection = np.asarray(row["projection"])
            normalised = projection / np.max(np.abs(projection))
            axes[0].plot(time, normalised, lw=1.0, label=f"mode {row['mode']} (decay)")
            axes[1].semilogy(time, np.abs(normalised), lw=0.8, color=f"C{row['mode']}")
            axes[1].semilogy(
                time,
                np.exp(-row["measured_zeta"] * row["measured_frequency"] * time),
                "--",
                color=f"C{row['mode']}",
                label=(
                    f"mode {row['mode']}: $\\zeta_d$={row['measured_zeta']:.4f} vs "
                    f"$\\zeta_{{fd}}$={row['zeta_quadratic']:.4f}"
                ),
            )
        axes[0].set_xlabel("time [s]")
        axes[0].set_ylabel("modal coordinate (norm.)")
        axes[0].set_title("coupled free decay, Cummins convolution")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("|modal coordinate|")
        axes[1].set_title("envelope: measured vs frequency-domain decay")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        path = directory / "fig4_free_decay.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        figures.append(
            {
                "name": "fig4_free_decay",
                "caption": "Free decay of the coupled Cummins equation (left) and its envelope against the "
                "frequency-domain damping ratios (right).",
                "path": str(path),
            }
        )

    rows = report["verdicts"][7]["values"]["rows"]
    ratios = [row["omega_p_over_omega_n"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    axes[0].plot(
        ratios, [100.0 * row["peak_amplitude_error"] for row in rows], "-o", ms=3, label="modal peak amplitude"
    )
    axes[0].plot(
        ratios,
        [100.0 * row["dof_amplitude_error"] for row in rows],
        "-s",
        ms=3,
        label="physical DOF peak amplitude",
    )
    axes[0].axhline(5.0, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\omega_p/\omega_n$")
    axes[0].set_ylabel("peak amplitude error [%]")
    axes[0].set_title("constant-coefficient substitute: amplitude")
    axes[1].plot(ratios, [row["phase_error_degrees"] for row in rows], "-o", ms=3, label="phase at $\\omega_n$")
    axes[1].axhline(10.0, color="k", ls="--", lw=0.8)
    axes[1].axhline(-10.0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel(r"$\omega_p/\omega_n$")
    axes[1].set_ylabel("phase error [deg]")
    axes[1].set_title("constant-coefficient substitute: phase")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    path = directory / "fig5_error_envelope.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    figures.append(
        {
            "name": "fig5_error_envelope",
            "caption": "Error envelope of the constant-coefficient substitute against omega_p/omega_n; dashed "
            "lines are the 5% and 10 deg acceptance bands.",
            "path": str(path),
        }
    )
    return figures
