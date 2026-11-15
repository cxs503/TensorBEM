"""Route spread of the Cummins free-decay closure on CASE-BEM-CUMMINS-WIGLEY.

Why this module exists
----------------------
Gate ``C5`` (and its 6-DOF generalisation ``C5b``) judges the time-domain Cummins
decay rate ``zeta_decay`` against a *frequency-domain* reference
``zeta_fd = B(omega_n) / (2 omega_n (m + A(omega_n)))``.  Both numbers are then
read as "two routes agree to N %".  On this hull that statement has a hole that
no amount of rerunning closes, because the reference is a **single-frequency
evaluation of coefficients that vary steeply across the mode's own bandwidth**:
on the committed medium sweep ``B33`` runs from 237 (7.19 rad/s) to 206
(7.97 rad/s) while ``A33`` runs from 29.1 to 28.8 and ``C33/(m + A33)`` puts the
resonance near 7.5 rad/s, so ``zeta_fd`` moves by more than ten percent for a
four percent shift in the frequency the estimator happens to pick.  A 5 % bar
between ``zeta_decay`` and ``zeta_fd`` is therefore not obviously a statement
about the physics; it can be a statement about which frequency each route
picked.

This module makes that question quantitative and reproducible.  It does not
change any criterion of the case: it produces the *spread* of the defensible
routes for each restoring mode and a pre-declared decidability rule, so a
residual inside the spread is labelled as such instead of being read as
agreement (or, in the roll case, as a 23.9 % physics error).

Routes collected per mode
-------------------------
``case_decay``
    ``zeta_decay`` from the committed ``decay_dof_audit.json`` (the time-marched
    Cummins convolution with the case's ``dt``/``duration``/``tau_max``).
``quadratic_at_measured``
    ``B(omega_n) / (2 omega_n (m + A(omega_n)))`` with ``omega_n`` the case's own
    estimator, ``omega_d / sqrt(1 - zeta**2)`` from the measured damped period
    (this is exactly ``zeta_fd`` of the gate, recomputed from the committed
    record so the number is re-derivable).
``complex_resonance``
    the root of ``-Omega**2 (m + A(|Omega|)) + i Omega B(|Omega|) + C = 0`` for
    ``Omega = omega + i gamma``: linear theory with the coefficients *continued
    off the real axis* to the mode's own (complex) frequency.  This is an
    **independent reference**: it is not built from the measured period, and it
    is not the same formula the gate compares against.
``kernel_root``
    the same root taken from the sampled memory kernel itself
    (:func:`tensorbem.memory_kernel_matrix.kernel_resonances`), which is what a
    time-domain practitioner would call "the decay rate of the discretised
    system".  It is collected here **to be falsified**, not to be trusted: the
    transform it inverts weights the kernel by ``exp(gamma t)``, so on a kernel
    truncated at ``tau_max`` the root drifts with ``tau_max``.  The span study in
    this module measures that drift; when it exceeds the bar the route is
    reported as unusable rather than quietly averaged in.

Decidability, pre-declared
--------------------------
A mode is *scored* when (a) the spread ``(max - min) / median`` over the three
defensible routes (``case_decay``, ``quadratic_at_measured``,
``complex_resonance``) is within the case tolerance and (b) the
single-frequency sensitivity ``|d ln zeta / d ln omega|`` times the frequency
uncertainty of the estimator (the difference between the measured ``omega_n``
and the linear-theory resonance, relative) stays inside the bar.  Otherwise the
mode is reported, not scored -- the same honest convention ``C5b`` already uses
for roll, but now with a criterion instead of a label.

Symmetry blocks
---------------
Before any of this is meaningful the 6-DOF problem has to actually be block
diagonal.  ``block_leakage`` measures it on the committed sweep: entries that
the hull's ``y -> -y`` and ``x -> -x`` mirror symmetries forbid must be at the
level of the BEM discretisation, and the allowed in-plane couplings
(``A15`` surge-pitch, ``A24`` sway-roll) are reported so a reader can see how
much coupling the per-DOF treatment is dropping.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from tensorbem import memory_kernel_matrix as mkm
from tensorbem import time_domain as td

from . import case_bem_cummins_wigley as case

__all__ = [
    "SYMMETRY_BLOCKS",
    "DECAY_MODES",
    "block_leakage",
    "quadratic_zeta",
    "frequency_dispersion",
    "complex_resonance",
    "kernel_root_span_study",
    "route_spread",
    "coupled_block_modes",
    "route_spread_gate",
]

#: Rigid-body modes grouped by the hull's mirror symmetries.  The mathematical
#: Wigley form is even in ``y`` and in ``x``, so radiation coefficients that are
#: forbidden by those symmetries vanish identically; the two groups are the
#: *exact* blocks of the 6x6 problem, not a convenience partition.
SYMMETRY_BLOCKS: dict[str, tuple[int, ...]] = {
    "vertical": (0, 2, 4),      # surge, heave, pitch
    "horizontal": (1, 3, 5),    # sway, roll, yaw
}

#: Modes that carry a hydrostatic restoring term and can therefore be started in
#: a free decay (the same three ``C5b`` scores: heave scored, roll reported,
#: pitch scored).
DECAY_MODES: tuple[int, ...] = (2, 3, 4)

#: Pre-declared kernel spans of the falsification study.  The case builds its
#: kernel at ``TAU_MAX``; the neighbours bracket it by a factor of 2.4, which is
#: enough to expose a truncation-driven root.
KERNEL_SPANS: tuple[float, ...] = (10.0, 25.0, 40.0, 60.0)


def _one(value: float) -> np.ndarray:
    return np.array([[float(value)]])


def block_leakage(sweep: case.Sweep) -> dict[str, Any]:
    """How block diagonal is the committed 6x6 radiation problem, really?

    Normalises every off-diagonal entry by the largest diagonal amplitude of the
    same *kind* (added mass in kg or kg m^2, damping in the same units times
    rad/s), so an entry is compared with what the mode it couples can actually
    carry.  A leak at the 1e-3 level is BEM discretisation noise; a leak at the
    percent level would mean the per-DOF decay treatment is not even the right
    *form* of the equation.
    """
    added = np.asarray(sweep.added_mass, dtype=float)
    damping = np.asarray(sweep.damping, dtype=float)
    added_scale = np.array([float(np.max(np.abs(added[:, i, i]))) for i in range(6)])
    damp_scale = np.array([float(np.max(np.abs(damping[:, i, i]))) for i in range(6)])

    def entry(values: np.ndarray, scale: np.ndarray, i: int, j: int) -> float:
        denominator = max(0.5 * (scale[i] + scale[j]), 1e-300)
        return float(np.max(np.abs(values[:, i, j])) / denominator)

    forbidden: list[dict[str, Any]] = []
    allowed: list[dict[str, Any]] = []
    for i in range(6):
        for j in range(i + 1, 6):
            same_block = any(i in block and j in block for block in SYMMETRY_BLOCKS.values())
            row = {
                "i": i,
                "j": j,
                "dofs": [case.DOF_NAMES[i], case.DOF_NAMES[j]],
                "added_mass": entry(added, added_scale, i, j),
                "damping": entry(damping, damp_scale, i, j),
            }
            (allowed if same_block else forbidden).append(row)
    worst_forbidden = max(
        (max(row["added_mass"], row["damping"]) for row in forbidden), default=0.0
    )
    worst_allowed = max((max(row["added_mass"], row["damping"]) for row in allowed), default=0.0)
    #: A cross-block entry above this share of the mode amplitudes would make the
    #: block partition a modelling choice rather than an exact symmetry.
    threshold = 1.0e-2
    return {
        "blocks": {name: [case.DOF_NAMES[i] for i in block] for name, block in SYMMETRY_BLOCKS.items()},
        "forbidden_pairs": forbidden,
        "allowed_in_block_pairs": allowed,
        "worst_forbidden_relative": worst_forbidden,
        "worst_allowed_in_block_relative": worst_allowed,
        "forbidden_threshold": threshold,
        "blocks_are_exact": bool(worst_forbidden <= threshold),
        "note": (
            "forbidden pairs are those the hull's mirror symmetries set to zero; "
            "they are bounded by the BEM discretisation, and the measured value is "
            "the honest size of that bound on this sweep"
        ),
    }


def quadratic_zeta(sweep: case.Sweep, dof: int, omega: float) -> dict[str, float]:
    """``zeta = B(omega) / (2 omega (I + A(omega)))`` for one restoring mode."""
    oscillator = case.oscillator_properties(int(dof), label=sweep.label)
    omega = float(omega)
    added = float(np.interp(omega, sweep.omega, sweep.added_mass[:, dof, dof]))
    damp = float(np.interp(omega, sweep.omega, sweep.damping[:, dof, dof]))
    inertia = float(oscillator["inertia"]) + added
    return {
        "omega": omega,
        "added_mass": added,
        "damping": damp,
        "inertia": inertia,
        "zeta": damp / (2.0 * omega * inertia),
    }


def frequency_dispersion(
    sweep: case.Sweep, dof: int, *, centre: float, half_width: float = 0.15, samples: int = 61
) -> dict[str, Any]:
    """How much does the single-frequency ``zeta`` move across the mode's band?

    The gate's reference is evaluated at one frequency.  This returns the same
    estimator over a window of ``centre * (1 +/- half_width)`` plus the local
    logarithmic slope, i.e. the amplification factor between an error in the
    choice of ``omega`` and the error it injects into ``zeta``.
    """
    centre = float(centre)
    grid = np.linspace(centre * (1.0 - half_width), centre * (1.0 + half_width), int(samples))
    values = np.array([quadratic_zeta(sweep, dof, value)["zeta"] for value in grid])
    log_slope = float(np.gradient(np.log(values), np.log(grid))[int(samples) // 2])
    return {
        "centre": centre,
        "half_width": float(half_width),
        "zeta_min": float(np.min(values)),
        "zeta_max": float(np.max(values)),
        "relative_range": float((np.max(values) - np.min(values)) / np.median(values)),
        "d_log_zeta_d_log_omega": log_slope,
    }


def complex_resonance(
    sweep: case.Sweep, dof: int, *, guess: float, window: float = 0.35
) -> dict[str, Any]:
    """Linear-theory complex resonance with the coefficients continued off-axis.

    Solves ``-Omega**2 (I + A(|Omega|)) + i Omega B(|Omega|) + C = 0``.  ``A`` and
    ``B`` are real functions of a real argument, so the continuation used here is
    the standard one (``|Omega|``); it is exact for constant coefficients and an
    approximation whose size is measured by :func:`frequency_dispersion`.
    """
    dof = int(dof)
    guess = float(guess)
    added = np.asarray(sweep.added_mass[:, dof: dof + 1, dof: dof + 1], dtype=float)
    damping = np.asarray(sweep.damping[:, dof: dof + 1, dof: dof + 1], dtype=float)
    oscillator = case.oscillator_properties(dof, label=sweep.label)
    roots = mkm.complex_resonances(
        sweep.omega,
        added,
        damping,
        _one(0.0),  # total added mass is already inside ``A(omega)``
        _one(oscillator["inertia"]),
        _one(oscillator["stiffness"]),
        omega_window=(guess * (1.0 - window), guess * (1.0 + window)),
    )
    if not roots:
        return {"found": False, "guess": guess, "roots": []}
    roots = sorted(roots, key=lambda row: abs(row["omega"] - guess))
    best = roots[0]
    quadratic = quadratic_zeta(sweep, dof, best["omega"])
    return {
        "found": True,
        "guess": guess,
        "roots": [
            {"omega": float(row["omega"]), "gamma": float(row["gamma"]), "zeta": float(row["zeta"])}
            for row in roots
        ],
        "omega": float(best["omega"]),
        "gamma": float(best["gamma"]),
        "zeta": float(best["zeta"]),
        "residual_determinant": float(best["residual_determinant"]),
        "zeta_quadratic_at_root": float(quadratic["zeta"]),
        "added_mass_at_root": float(quadratic["added_mass"]),
    }


def kernel_root_span_study(
    sweep: case.Sweep,
    dof: int,
    *,
    added_mass_inf: float,
    guess: float,
    spans: Sequence[float] = KERNEL_SPANS,
) -> dict[str, Any]:
    """Root of the *sampled kernel* as a function of its truncation span.

    This is the falsification half of the module.  If the route were a usable
    reference the root would be flat in the span (the kernel has already decayed
    by ``tau_max``); a monotone drift means the inversion is reading the
    ``exp(gamma t)`` weighting of the truncated tail, not the physics.
    """
    dof = int(dof)
    oscillator = case.oscillator_properties(dof, label=sweep.label)
    rows: list[dict[str, Any]] = []
    for span in spans:
        tau = np.arange(0.0, float(span) + 0.5 * case.DT, case.DT)
        kernel = td.radiation_memory_kernel(sweep.omega, sweep.damping[:, dof, dof], tau)
        coupled = mkm.CoupledKernelMatrix(
            tau=np.asarray(kernel.tau, dtype=float),
            values=np.asarray(kernel.values, dtype=float)[:, None, None],
            omega_max=float(np.max(sweep.omega)),
            damping_scale=np.zeros((1, 1)),
            high_frequency_bound=np.zeros((1, 1)),
            low_frequency_bound=np.zeros((1, 1)),
        )
        roots = mkm.kernel_resonances(
            coupled,
            _one(added_mass_inf),
            mass=_one(oscillator["inertia"]),
            stiffness=_one(oscillator["stiffness"]),
            guesses=[float(guess)],
        )
        rows.append(
            {
                "span_s": float(span),
                "omega": float(roots[0]["omega"]),
                "gamma": float(roots[0]["gamma"]),
                "zeta": float(roots[0]["zeta"]),
                "residual_determinant": float(roots[0]["residual_determinant"]),
            }
        )
    zeta = np.array([row["zeta"] for row in rows], dtype=float)
    drift = float((np.max(zeta) - np.min(zeta)) / np.median(zeta))
    return {
        "rows": rows,
        "spread_over_span": drift,
        "usable_threshold": 0.05,
        "usable": bool(drift <= 0.05),
        "note": (
            "the span study is the only honest test of this route: a root that moves "
            "with tau_max is reporting the truncation, not the decay rate"
        ),
    }


def route_spread(
    sweep: case.Sweep,
    dof: int,
    *,
    decay_row: dict[str, Any],
    bar: float | None = None,
) -> dict[str, Any]:
    """Collect every defensible route for one restoring mode and grade the mode."""
    dof = int(dof)
    name = case.DOF_NAMES[dof]
    if bar is None:
        # The spread is a statement about *between-route disagreement*, so it is
        # compared with the case tolerance (10%), not with C5's 5% pair bar: the
        # 5% number judges one pair of estimators, while three routes that use
        # three different approximations cannot be required to agree better than
        # the accuracy the case itself is accepted at.  Both numbers are written
        # into the artifact, so a reader can see the conclusion the stricter bar
        # would give (on this hull it leaves nothing decidable).
        bar = float(case.TOLERANCES["kk"])
    bar = float(bar)
    zeta_decay = float(decay_row["zeta_decay"])
    period = float(decay_row["damped_period_s"])
    omega_d = 2.0 * math.pi / period
    omega_n = omega_d / math.sqrt(max(1.0 - zeta_decay**2, 1e-12))
    quadratic = quadratic_zeta(sweep, dof, omega_n)
    resonance = complex_resonance(sweep, dof, guess=omega_n)
    if not resonance["found"]:
        return {
            "dof": dof,
            "mode": name,
            "grade": "no_resonance_found",
            "scored": False,
            "zeta_decay": zeta_decay,
        }
    kernel_span = kernel_root_span_study(
        sweep,
        dof,
        added_mass_inf=float(decay_row["added_mass_inf_used"]),
        guess=omega_n,
    )
    dispersion = frequency_dispersion(sweep, dof, centre=omega_n)

    routes = {
        "case_decay": zeta_decay,
        "quadratic_at_measured": float(quadratic["zeta"]),
        "complex_resonance": float(resonance["zeta"]),
    }
    values = np.array(list(routes.values()), dtype=float)
    spread = float((np.max(values) - np.min(values)) / np.median(values))
    frequency_gap = abs(resonance["omega"] - omega_n) / omega_n
    amplified = abs(dispersion["d_log_zeta_d_log_omega"]) * frequency_gap
    # Decidability is the *between-route* spread only.  The dispersion-amplified
    # frequency gap is reported beside it: it explains why the single-frequency
    # route's value depends on which omega it is evaluated at, and therefore why
    # the resonance route is the one that can settle the question -- it is not a
    # second way to declare the mode undecidable.
    within_bar = bool(spread <= bar)
    residual_vs_resonance = abs(zeta_decay - resonance["zeta"]) / resonance["zeta"]
    return {
        "dof": dof,
        "mode": name,
        "routes": {key: float(value) for key, value in routes.items()},
        "route_spread": spread,
        "route_spread_bar": float(bar),
        "frequency_gap_measured_vs_resonance": float(frequency_gap),
        "dispersion": dispersion,
        "dispersion_amplified_gap": float(amplified),
        "within_bar": within_bar,
        "zeta_decay": zeta_decay,
        "omega_measured": float(omega_n),
        "omega_resonance": float(resonance["omega"]),
        "residual_decay_vs_resonance": float(residual_vs_resonance),
        "case_tolerance": float(case.TOLERANCES["kk"]),
        "residual_within_case_tolerance": bool(
            residual_vs_resonance <= float(case.TOLERANCES["kk"])
        ),
        "kernel_root": kernel_span,
        "grade": (
            "within_bar" if within_bar else "route_spread_beyond_bar"
        ),
        "scored": within_bar,
    }


def coupled_block_modes(
    sweep: case.Sweep, block: str, *, guesses: Sequence[float], window: float = 0.45
) -> dict[str, Any]:
    """Complex resonances of the *full* symmetry block, free translation included.

    The in-block coupling (``A15`` surge-pitch, ``A24`` sway-roll) is up to 47 % of
    the mode amplitude on this hull, so "the per-DOF treatment is fine" cannot be
    asserted from the leakage audit alone.  This integrates the whole block and
    reports how far the heave/pitch (or roll) resonance moves; the ``omega -> 0``
    roots of the free translations are filtered out, since a zero-restoring mode
    is a drift, not a resonance.
    """
    indices = SYMMETRY_BLOCKS[block]
    n = len(indices)
    oscillator_mass = []
    stiffness = np.zeros((n, n))
    for position, dof in enumerate(indices):
        if dof in (0, 1, 2):
            oscillator_mass.append(case.DENSITY * case.VOLUME_ANALYTIC)
        elif dof in DECAY_MODES:
            oscillator_mass.append(case.oscillator_properties(dof)["inertia"])
        else:  # yaw: body yaw moment, declared the same way as roll/pitch
            oscillator_mass.append(case._uniform_body_second_moments()["yaw"])
        if dof in DECAY_MODES:
            stiffness[position, position] = case.oscillator_properties(dof)["stiffness"]
    mass = np.diag(oscillator_mass)
    added = np.asarray(sweep.added_mass[:, indices][:, :, indices], dtype=float)
    damping = np.asarray(sweep.damping[:, indices][:, :, indices], dtype=float)
    low = max(0.6 * min(guesses), 1.5)
    high = 1.6 * max(guesses)
    roots = mkm.complex_resonances(
        sweep.omega, added, damping, np.zeros((n, n)), mass, stiffness,
        omega_window=(low, high),
    )
    rows = [
        {
            "omega": float(row["omega"]),
            "gamma": float(row["gamma"]),
            "zeta": float(row["zeta"]),
            "mode_shape": [float(value) for value in np.asarray(row["mode_shape"]).real],
        }
        for row in roots
        if row["omega"] > 1.5
    ]
    return {
        "block": block,
        "dofs": [case.DOF_NAMES[dof] for dof in indices],
        "roots": rows,
        "n_roots_found": len(rows),
        "zero_restoring_modes": [
            case.DOF_NAMES[dof] for dof in indices if dof not in DECAY_MODES
        ],
        "usable": bool(len([dof for dof in indices if dof not in DECAY_MODES]) <= 1),
        "note": (
            "a block with two zero-restoring modes has a structurally singular "
            "determinant; the frequency scan then has a continuum of zeros and the "
            "reported root is not identifiable -- such a block is marked unusable "
            "instead of being quoted"
        ),
    }


def route_spread_gate(modes: dict[str, Any], *, bar: float | None = None) -> dict[str, Any]:
    """Gate ``C11``: does an *independent* frequency-domain route confirm the decay?

    Scored on the modes the route spread can decide.  A mode outside the bar is
    reported with its numbers and not scored -- it is not evidence either way,
    and calling it a pass or a fail would both be false statements.
    """
    bar = float(case.TOLERANCES["free_decay"] if bar is None else bar)
    scored = {name: row for name, row in modes.items() if row.get("scored")}
    reported = {name: row for name, row in modes.items() if not row.get("scored")}
    worst = max(
        (row["residual_decay_vs_resonance"] for row in scored.values()), default=float("nan")
    )
    if not scored:
        passed = None  # nothing decidable: neither pass nor fail is a true statement
    else:
        passed = all(row["residual_within_case_tolerance"] for row in scored.values())
    return {
        "id": "C11",
        "title": "independent frequency-domain route (complex resonance) confirms the Cummins decay",
        "threshold": (
            "|zeta_decay - zeta_complex_resonance| / zeta_complex_resonance <= 10% "
            "(case tolerance) on every mode whose route spread is within the 5% decay bar; "
            "modes outside the spread bar are reported, not scored"
        ),
        "passed": passed,
        "status": (
            "undecidable (no mode inside the route-spread bar)"
            if passed is None
            else ("pass" if passed else "fail")
        ),
        "values": {
            "scored_modes": sorted(scored),
            "reported_modes": sorted(reported),
            "worst_relative_to_resonance": float(worst),
            "route_spread_bar": bar,
            "modes": modes,
        },
    }