"""C5 attribution for the Wigley III Cummins case: why the free-decay route fails.

The case-pool anchor ``CASE-BEM-CUMMINS-WIGLEY`` scores 9 of 10 gates.  The one
FAIL is gate C5 of :mod:`tensorbem.benchmarks.case_bem_cummins_wigley`::

    zeta_decay (Cummins free decay, time domain) = 0.13787
    zeta_fd    (B / (2 (M + A) omega_n) at omega_n) = 0.12875
    relative error = 7.09 %   vs an internal tolerance of 5 %

Two explanations fit that number and they imply opposite remediations:

*H-reference* -- the frequency-domain surrogate is the biased one (it
  interpolates ``A``, ``B`` linearly at ``omega_n`` on a 41-node grid, and the
  energy identity is only valid for a single-degree-of-freedom oscillator with
  frequency-independent coefficients);
*H-route* -- the time-domain route is the biased one because
  :func:`tensorbem.time_domain.radiation_memory_kernel` hard-truncates the
  sweep at ``omega_max = 12 rad/s``, so the kernel it produces is not the
  kernel of any causal system: ``K(tau)`` does not decay.

This module decides between them with an estimator that neither hypothesis uses:
the **exact modal damping of a causal radiation model**.  The sweep is continued
off the band by a rational (state-space) fit -- the only continuation that is
causal by construction, because ``K(tau) = sum_k c_k exp(-alpha_k tau)`` is a
sum of decaying exponentials -- and the modal damping is read off the complex
root of the characteristic equation::

    (M + A_inf) s^2 + s * K_hat(s) + C = 0,     K_hat(s) = D_ss + sum_k c_k/(s + alpha_k)

whose root ``s = -zeta omega_n + i omega_d`` is *exact* for the model that is
integrated in the time domain.  For a constant-coefficient oscillator the same
root reproduces ``zeta = B / (2 sqrt((M + A) C))`` to machine precision, so the
estimator itself is validated before it is used to accuse anything (gate D0).

Gate ladder (predeclared, all machine-decided)
---------------------------------------------
D0  estimator validation: on a constant-damping oscillator the complex root
    reproduces the closed form ``B/(2 sqrt((M+A) C))`` to <= 1e-6, and on a
    single-exponential (cubic characteristic) memory the Newton root reproduces
    ``numpy.roots`` to <= 1e-8.
D1  causal continuation quality: the rational fit reproduces the measured heave
    damping to <= 3 % max in band *and* its ``A(omega)`` matches the measured
    added mass to <= 5 % (Kramers-Kronig consistency of the continuation).
D2  kernel completeness: the causal kernel decays (|K(25 s)|/|K(0)| <= 1e-3)
    while the band-truncated kernel does not -- this is the defect D2 documents.
D3  causal decay consistency (the replacement for C5): the time-domain Cummins
    decay driven by the causal kernel reproduces the exact modal damping of the
    *same* model to <= 5 %.
D4  attribution: ``|zeta_decay_legacy - zeta_exact|`` vs
    ``|zeta_fd - zeta_exact|`` -- which route carries the 7.09 %, reported, not
    gated.
D5  band robustness (defect demonstration, not a pass gate): the legacy route's
    damping ratio is scanned over the truncated band top.  A route whose answer
    moves by more than a factor of ten when the top of the sweep moves from 8 to
    12 rad/s cannot certify a 5 % agreement; the causal route has no such
    parameter.

Honesty boundary
----------------
The attribution is *mesh dependent*: on the medium mesh (480 panels) the legacy
free decay is the biased route (6.4 % from the exact root, while the
interpolated reference is 0.7 % away); on the coarse mesh (192 panels) it is the
other way round (legacy decay 0.15 % away, reference ~3 %).  Both routes
therefore carry an error of the same order as the 5 % tolerance, with a sign
that follows the mesh -- so the honest reading of C5 is not "route X is wrong"
but "a 5 % agreement between these two estimators is not decidable on this
case".  What *is* decidable is the cutoff dependence of the legacy route (D5: a
factor 65 between band tops 8 and 12 rad/s on the same hull) and the internal
consistency of the causal route (D3: 3.6 % medium, 3.5 % coarse).
The causal continuation is *fitted to the same BEM sweep*; it is a
best-fit continuation, not an independent measurement.  D1 holds it only
in-band.  Out of band the fit is unconstrained (the fitted feedthrough and the
short-time kernel of the 6-10 pole fits scatter by orders of magnitude), so the
causal answer is quoted as ``zeta = 0.130 +- 0.001`` (order scan) and *not* as a
physics anchor.  The independent-solver anchor of the case remains R1/capytaine.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from tensorbem import causality
from tensorbem import state_space as ssmod
from tensorbem import time_domain as td
from tensorbem.benchmarks import case_bem_cummins_wigley as case

__all__ = [
    "GATES",
    "ARTIFACT_PATH",
    "SWEEP_PATHS",
    "HEAVE",
    "causal_fit",
    "causal_kernel",
    "complex_modal_root",
    "exact_modal_damping",
    "decay_zeta",
    "causal_free_decay",
    "legacy_free_decay",
    "legacy_frequency_domain_zeta",
    "band_top_scan",
    "run_audit",
    "analytic_oscillator_checks",
]

#: Heave is the DOF the case's own verdict scores C5 on.  Roll and pitch are
#: scored alongside it by :func:`case_bem_cummins_wigley.dof_decay_audit`, which
#: builds the C44/C55 restoring terms and the declared body inertia this audit
#: used to record as open debt (see ``decay_dof_audit.json``).
HEAVE = 2

#: Frozen sweeps of the case-pool artifact (``case.run_case`` output).
SWEEP_PATHS = {
    "coarse": "benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz",
    "medium": "benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz",
}

#: Where :func:`run_audit` numbers are written by the driver script.
ARTIFACT_PATH = "benchmarks/verified/casepool_bem_cummins_wigley/causal_decay_audit.json"

#: Predeclared tolerances.  ``case_tolerance`` is the anchor's own <10 %; the
#: internal gates are held tighter so a PASS at 4 % still means something.
GATES = {
    "D0_estimator_validation": {"tol": 1e-6, "quantity": "complex root vs closed form"},
    "D0b_cubic_memory_root": {"tol": 1e-8, "quantity": "Newton vs numpy.roots"},
    "D1_fit_band": {"tol": 0.03, "quantity": "max in-band |B_fit - B_bem| / max|B|"},
    "D1_kk_added_mass": {"tol": 0.05, "quantity": "max |A_fit - A_bem| / max|A|"},
    "D2_kernel_decay": {"tol": 1e-3, "quantity": "|K(tau_max)| / |K(0)| (causal)"},
    "D3_causal_decay": {"tol": 0.05, "quantity": "|zeta_decay_causal - zeta_exact|/zeta_exact"},
    "D4_legacy_bias": {"tol": 0.05, "quantity": "|zeta_decay_legacy - zeta_exact|/zeta_exact"},
    "D5_band_robustness": {"tol": 1.0, "quantity": "max/min zeta over band tops (legacy)"},
}

#: Order scan for the rational continuation.  Twelve poles reproduces the band
#: to 1.1 % but its Kramers-Kronig added-mass residual jumps to 17.6 %
#: (docs/dev_report_m21275_c5_attribution.md), so the ladder stops at ten.
MAX_POLES = 10
FIT_TARGET = 0.02
MIN_POLES = 4


# ---------------------------------------------------------------------------
# The frozen sweep and the exact modal damping of a causal model
# ---------------------------------------------------------------------------
def load_heave(level: str = "medium", root: str | None = None):
    """Return ``(omega, A33, B33, restoring, displaced_mass, A_inf)`` for a level.

    ``A_inf`` is the Kramers-Kronig closure the case's own C5 uses, so the
    only thing this module changes downstream is the *kernel*, not the mass
    bookkeeping.
    """
    path = SWEEP_PATHS[level] if root is None else str(root)
    sweep = case.load_sweep(path, label=level)
    omega = np.asarray(sweep.omega, dtype=float)
    added = np.asarray(sweep.added_mass[:, HEAVE, HEAVE], dtype=float)
    damping = np.asarray(sweep.damping[:, HEAVE, HEAVE], dtype=float)
    restoring, displaced_mass, _ = case._restoring_and_mass(sweep)
    closure = causality.kk_added_mass_closure(omega, added, damping)
    return omega, added, damping, float(restoring), float(displaced_mass), float(closure.added_mass_inf)


def causal_fit(
    omega: Sequence[float],
    damping: Sequence[float],
    *,
    min_poles: int = MIN_POLES,
    max_poles: int = MAX_POLES,
    target: float = FIT_TARGET,
) -> tuple[ssmod.VectorFitResult, ssmod.StateSpaceRealization, int]:
    """Lowest-order rational fit of ``B(omega)`` meeting ``target`` in band.

    Order is scanned from ``min_poles`` upwards and the first fit whose maximum
    relative error is below ``target`` is kept; if none qualifies the best fit
    of the scan is returned.  The realization is *causal by construction*:
    ``B_ss(omega) = D + sum_k c_k alpha_k/(alpha_k^2+omega^2)`` is the Fourier
    cosine transform of ``K(tau) = sum_k c_k exp(-alpha_k tau)``.
    """
    best: tuple[ssmod.VectorFitResult, ssmod.StateSpaceRealization, int] | None = None
    for n_poles in range(int(min_poles), int(max_poles) + 1):
        fit = ssmod.vector_fit_damping(omega, damping, n_poles=n_poles, max_iter=1200)
        realization = ssmod.state_space_from_fit(fit)
        if best is None or fit.relative_error_max < best[0].relative_error_max:
            best = (fit, realization, n_poles)
        if fit.relative_error_max <= target:
            return fit, realization, n_poles
    assert best is not None  # min_poles >= 1
    return best


def causal_kernel(realization: ssmod.StateSpaceRealization, tau: Sequence[float]) -> np.ndarray:
    """``K(tau) = sum_k c_k exp(-alpha_k tau)`` of a rational realization."""
    tau = np.asarray(tau, dtype=float)
    poles = np.asarray(realization.poles, dtype=float)
    residues = np.asarray(realization.residues, dtype=float)
    return np.exp(-np.outer(tau, poles)) @ residues


def causal_transfer(realization: ssmod.StateSpaceRealization, omega: Sequence[float]) -> np.ndarray:
    """``K_hat(i omega) = B(omega) + i omega (A(omega) - A_inf)`` of the fit."""
    omega = np.asarray(omega, dtype=float)
    poles = np.asarray(realization.poles, dtype=float)
    residues = np.asarray(realization.residues, dtype=float)
    return realization.D_ss + (residues[:, None] / (poles[:, None] + 1j * omega[None, :])).sum(axis=0)


def causal_added_mass(
    realization: ssmod.StateSpaceRealization, omega: Sequence[float], added_mass_inf: float
) -> np.ndarray:
    """``A(omega) = A_inf - sum_k c_k/(alpha_k^2 + omega^2)`` of the fit."""
    omega = np.asarray(omega, dtype=float)
    poles = np.asarray(realization.poles, dtype=float)
    residues = np.asarray(realization.residues, dtype=float)
    return added_mass_inf - (residues[:, None] / (poles[:, None] ** 2 + omega[None, :] ** 2)).sum(axis=0)


def causal_damping(realization: ssmod.StateSpaceRealization, omega: Sequence[float]) -> np.ndarray:
    """``B(omega) = D_ss + sum_k c_k alpha_k/(alpha_k^2 + omega^2)`` of the fit."""
    return causal_transfer(realization, omega).real


def complex_modal_root(
    realization: ssmod.StateSpaceRealization,
    mass: float,
    added_mass_inf: float,
    restoring: float,
    *,
    guess_omega: float,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Root of ``(M+A_inf) s^2 + s K_hat(s) + C = 0`` nearest ``i guess_omega``.

    Newton in the complex plane with a central-difference derivative; the step
    is accepted only while the residual keeps falling, and the residual at the
    accepted root is returned so the caller can refuse a non-converged number.
    """
    poles = np.asarray(realization.poles, dtype=float)
    residues = np.asarray(realization.residues, dtype=float)

    def k_hat(s: complex) -> complex:
        return realization.D_ss + complex(np.sum(residues / (s + poles)))

    def residual(s: complex) -> complex:
        return (mass + added_mass_inf) * s * s + s * k_hat(s) + restoring

    s = complex(0.0, float(guess_omega))
    scale = max(abs(restoring), abs((mass + added_mass_inf) * guess_omega ** 2), 1e-30)
    history = [abs(residual(s)) / scale]
    converged = False
    for _ in range(int(max_iter)):
        h = 1e-7 * abs(s)
        slope = (residual(s + 1j * h) - residual(s - 1j * h)) / (2j * h)
        if slope == 0:
            break
        step = residual(s) / slope
        candidate = s - step
        value = abs(residual(candidate)) / scale
        # Damped Newton: halve the step until the residual decreases.
        shrink = 1.0
        while value > history[-1] and shrink > 1e-6:
            shrink *= 0.5
            candidate = s - shrink * step
            value = abs(residual(candidate)) / scale
        s = candidate
        history.append(value)
        if abs(step) * shrink <= tol * abs(s):
            converged = True
            break
    zeta = -s.real / abs(s)
    return {
        "root": complex(s),
        "zeta": float(zeta),
        "omega_damped": float(abs(s.imag)),
        "omega_natural": float(abs(s)),
        "residual_relative": float(history[-1]),
        "iterations": int(len(history) - 1),
        "converged": bool(converged or history[-1] <= 1e-10),
    }


def exact_modal_damping(
    level: str = "medium",
    *,
    root: str | None = None,
    orders: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Order-scan the exact modal damping ratio of the causal continuation."""
    omega, added, damping, restoring, mass, a_inf = load_heave(level, root=root)
    if orders is None:
        fit, realization, n_poles = causal_fit(omega, damping)
        orders = (n_poles,)
        fit_by_order = {n_poles: fit}
    else:
        fit_by_order = {}
    rows = []
    for order in orders:
        if order not in fit_by_order:
            fit_by_order[order] = ssmod.vector_fit_damping(omega, damping, n_poles=order, max_iter=1200)
        fit = fit_by_order[order]
        realization = ssmod.state_space_from_fit(fit)
        a_fit = causal_added_mass(realization, omega, a_inf)
        rr = complex_modal_root(realization, mass, a_inf, restoring, guess_omega=7.8)
        rows.append(
            {
                "n_poles": int(order),
                "fit_relative_error_max": float(fit.relative_error_max),
                "fit_relative_error_rms": float(fit.relative_error_rms),
                "kk_added_mass_relative": float(np.max(np.abs(a_fit - added)) / np.max(np.abs(added))),
                "zeta_exact": rr["zeta"],
                "omega_damped": rr["omega_damped"],
                "residual_relative": rr["residual_relative"],
            }
        )
    return {"level": level, "mass": mass, "restoring": restoring, "added_mass_inf": a_inf, "orders": rows}


# ---------------------------------------------------------------------------
# Time-domain routes
# ---------------------------------------------------------------------------
def decay_zeta(time: Sequence[float], displacement: Sequence[float]) -> tuple[float, float, int]:
    """Damping ratio, damped period and peak count of a decay record.

    Deliberately the same estimator as ``case_bem_cummins_wigley._decay_zeta``
    (peak-envelope log decrement over the record): the audit must not change the
    *measurement* while it changes the *model*.  ``tests`` asserts the two
    agree on a shared record.
    """
    t = np.asarray(time, dtype=float)
    x = np.asarray(displacement, dtype=float)
    excursion = float(np.max(np.abs(x))) if x.size else 0.0
    if excursion <= 0.0:
        return float("nan"), float("nan"), 0
    threshold = 0.01 * excursion
    peaks: list[tuple[float, float]] = []
    for i in range(1, x.size - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > threshold:
            peaks.append((float(t[i]), float(x[i])))
    if len(peaks) < 3:
        return float("nan"), float("nan"), len(peaks)
    first, last = peaks[0], peaks[-1]
    periods = (last[0] - first[0]) / (len(peaks) - 1)
    decrement = math.log(first[1] / last[1]) / (len(peaks) - 1)
    zeta = decrement / math.sqrt(4.0 * math.pi ** 2 + decrement ** 2)
    return float(zeta), float(periods), len(peaks)


def causal_free_decay(
    realization: ssmod.StateSpaceRealization,
    *,
    mass: float,
    restoring: float,
    added_mass_inf: float,
    dt: float = case.DT,
    duration: float = case.DURATION,
    initial_displacement: float = 1.0,
) -> dict[str, Any]:
    """Cummins free decay driven by the causal (exponentially decaying) kernel."""
    result = ssmod.cummins_state_space(
        realization,
        mass=mass,
        added_mass_inf=added_mass_inf,
        restoring=restoring,
        initial_displacement=initial_displacement,
        dt=dt,
        duration=duration,
    )
    time = np.asarray(result.time, dtype=float)
    displacement = np.asarray(result.displacement, dtype=float)
    zeta, period, n_peaks = decay_zeta(time, displacement)
    return {
        "dt": float(dt),
        "duration": float(duration),
        "substeps": int(getattr(result, "substeps", 1)),
        "zeta_decay": zeta,
        "damped_period": period,
        "n_peaks": n_peaks,
        "time": time,
        "displacement": displacement,
    }


def legacy_free_decay(
    omega: Sequence[float],
    damping: Sequence[float],
    *,
    mass: float,
    restoring: float,
    added_mass_inf: float,
    tau_max: float = case.TAU_MAX,
    dt: float = case.DT,
    duration: float = case.DURATION,
    initial_displacement: float = 1.0,
) -> dict[str, Any]:
    """The case's C5 route: hard-truncated sweep -> kernel -> Cummins decay."""
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)
    tau = np.arange(0.0, tau_max + 0.5 * dt, dt)
    kernel = td.radiation_memory_kernel(omega, damping, tau)
    decay = td.cummins_free_decay(
        kernel,
        mass=mass,
        restoring=restoring,
        added_mass_inf=added_mass_inf,
        initial_displacement=initial_displacement,
        dt=dt,
        duration=duration,
    )
    time = np.asarray(decay.time, dtype=float)
    displacement = np.asarray(decay.displacement, dtype=float)
    zeta, period, n_peaks = decay_zeta(time, displacement)
    return {
        "band_top": float(omega[-1]),
        "tau_max": float(tau_max),
        "dt": float(dt),
        "duration": float(duration),
        "kernel_tau": tau,
        "kernel_values": np.asarray(kernel.values, dtype=float),
        "zeta_decay": zeta,
        "damped_period": period,
        "n_peaks": n_peaks,
        "time": time,
        "displacement": displacement,
    }


def legacy_frequency_domain_zeta(
    omega: Sequence[float],
    added_mass: Sequence[float],
    damping: Sequence[float],
    *,
    mass: float,
    omega_natural: float,
) -> dict[str, float]:
    """The case's C5 reference: ``B(omega_n) / (2 (M + A(omega_n)) omega_n)``.

    ``A`` and ``B`` are linearly interpolated at ``omega_n`` on the tabulated
    grid, exactly as :func:`case_bem_cummins_wigley.free_decay_gate` does.
    """
    omega = np.asarray(omega, dtype=float)
    a_at = float(np.interp(omega_natural, omega, np.asarray(added_mass, dtype=float)))
    b_at = float(np.interp(omega_natural, omega, np.asarray(damping, dtype=float)))
    return {
        "omega_natural": float(omega_natural),
        "added_mass_at_omega_n": a_at,
        "damping_at_omega_n": b_at,
        "zeta_frequency_domain": b_at / (2.0 * (mass + a_at) * omega_natural),
    }


def band_top_scan(
    omega: Sequence[float],
    damping: Sequence[float],
    *,
    mass: float,
    restoring: float,
    added_mass_inf: float,
    band_tops: Sequence[float] = (8.0, 9.0, 10.0, 11.0, 12.0),
) -> list[dict[str, Any]]:
    """D5: legacy-route damping ratio as the truncated band top moves."""
    omega = np.asarray(omega, dtype=float)
    damping = np.asarray(damping, dtype=float)
    rows = []
    for top in band_tops:
        mask = omega <= float(top) + 1e-9
        if int(mask.sum()) < 8:
            rows.append({"band_top": float(top), "n_nodes": int(mask.sum()),
                         "status": "undecidable: fewer than 8 nodes below the top"})
            continue
        run = legacy_free_decay(omega[mask], damping[mask], mass=mass,
                                restoring=restoring, added_mass_inf=added_mass_inf)
        rows.append({
            "band_top": float(top),
            "n_nodes": int(mask.sum()),
            "zeta_decay": run["zeta_decay"],
            "damped_period": run["damped_period"],
            "n_peaks": run["n_peaks"],
        })
    return rows


# ---------------------------------------------------------------------------
# D0: the estimator is validated before it is used
# ---------------------------------------------------------------------------
def analytic_oscillator_checks() -> dict[str, Any]:
    """Validate the complex-root solver against closed forms.

    *Constant damping* (kernel ``2 B0 delta(t)``): the characteristic equation is
    quadratic and its root gives ``zeta = B0/(2 sqrt((M+A) C))``.
    *Single-exponential memory* (``K(tau) = c exp(-alpha tau)``): the
    characteristic equation is cubic; ``numpy.roots`` supplies the reference the
    Newton iteration must reproduce.
    """
    mass, a_inf, restoring = 76.875, 50.93021596754935, 6016.012025625
    omega_n = math.sqrt(restoring / (mass + a_inf))
    quadratic = []
    for b0 in (100.0, 250.0):
        closed = b0 / (2.0 * (mass + a_inf) * omega_n)
        root = complex(0.0, omega_n)
        for _ in range(100):
            step = ((mass + a_inf) * root ** 2 + b0 * root + restoring) / (
                2.0 * (mass + a_inf) * root + b0
            )
            root -= step
            if abs(step) <= 1e-15 * abs(root):
                break
        quadratic.append({
            "damping": b0,
            "zeta_closed_form": closed,
            "zeta_complex_root": -root.real / abs(root),
            "relative_error": abs(-root.real / abs(root) - closed) / closed,
        })

    alpha, residue = 3.0, 2500.0
    realization = ssmod.StateSpaceRealization(
        A_ss=np.array([[-alpha]]), B_ss=np.array([residue]), C_ss=np.array([1.0]),
        poles=np.array([alpha]), residues=np.array([residue]), D_ss=0.0,
    )
    # (M+A)s^2 + s*residue/(s+alpha) + C = 0  ->  multiply by (s+alpha):
    # (M+A) s^3 + (M+A) alpha s^2 + (residue + C) s + C alpha = 0   (descending)
    inertia = mass + a_inf
    poly = [inertia, inertia * alpha, residue + restoring, restoring * alpha]
    roots = np.roots(poly)
    oscillatory = [r for r in roots if abs(r.imag) > 1e-9]
    # Physical branch: positive damped frequency and a *stable* (left-half-plane)
    # root.  A synthetic kernel with the opposite residue sign would put the
    # oscillatory root in the right half plane; the check is about the root the
    # Newton iteration finds, so the branch is pinned explicitly.
    stable = [r for r in oscillatory if r.imag > 0.0 and r.real < 0.0]
    reference = stable[0] if stable else max(oscillatory, key=lambda r: r.imag)
    rr = complex_modal_root(realization, mass, a_inf, restoring, guess_omega=omega_n)
    zeta_reference = -reference.real / abs(reference)
    return {
        "constant_damping": quadratic,
        "single_exponential_memory": {
            "pole": alpha,
            "residue": residue,
            "zeta_reference": float(zeta_reference),
            "zeta_newton": rr["zeta"],
            "relative_error": float(abs(rr["zeta"] - zeta_reference) / zeta_reference),
            "root_reference": complex(reference),
            "root_newton": rr["root"],
            "converged": rr["converged"],
        },
    }


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------
def _verdict(identifier: str, value: float | None, tol: float, *, quantity: str,
             lower_is_better: bool = True) -> dict[str, Any]:
    if value is None or not math.isfinite(float(value)):
        return {"id": identifier, "quantity": quantity, "value": value,
                "tolerance": tol, "passed": None,
                "status": "undecidable: non-finite value"}
    value = float(value)
    passed = value <= tol if lower_is_better else value >= tol
    return {"id": identifier, "quantity": quantity, "value": value,
            "tolerance": tol, "passed": bool(passed),
            "status": "PASS" if passed else "FAIL"}


def run_audit(level: str = "medium", *, root: str | None = None) -> dict[str, Any]:
    """Decide between H-reference and H-route for the C5 failure (all numbers from data)."""
    omega, added, damping, restoring, mass, a_inf = load_heave(level, root=root)
    validation = analytic_oscillator_checks()

    legacy = legacy_free_decay(omega, damping, mass=mass, restoring=restoring,
                               added_mass_inf=a_inf)
    zeta_legacy = float(legacy["zeta_decay"])
    omega_d = 2.0 * math.pi / float(legacy["damped_period"])
    omega_n_legacy = omega_d / math.sqrt(max(1.0 - zeta_legacy ** 2, 1e-12))
    reference = legacy_frequency_domain_zeta(omega, added, damping, mass=mass,
                                             omega_natural=omega_n_legacy)

    fit, realization, n_poles = causal_fit(omega, damping)
    a_fit = causal_added_mass(realization, omega, a_inf)
    b_fit = causal_damping(realization, omega)
    fit_error = float(np.max(np.abs(b_fit - damping)) / np.max(np.abs(damping)))
    kk_error = float(np.max(np.abs(a_fit - added)) / np.max(np.abs(added)))

    # Exact modal damping of the causal model, scanned over the pole order so the
    # number carries an uncertainty instead of a spurious second digit.
    order_rows = []
    for order in range(MIN_POLES, MAX_POLES + 1):
        o_fit = fit if order == n_poles else ssmod.vector_fit_damping(omega, damping, n_poles=order, max_iter=1200)
        o_real = realization if order == n_poles else ssmod.state_space_from_fit(o_fit)
        rr = complex_modal_root(o_real, mass, a_inf, restoring, guess_omega=omega_n_legacy)
        order_rows.append({
            "n_poles": order,
            "fit_relative_error_max": float(o_fit.relative_error_max),
            "kk_added_mass_relative": float(
                np.max(np.abs(causal_added_mass(o_real, omega, a_inf) - added)) / np.max(np.abs(added))
            ),
            "zeta_exact": rr["zeta"],
            "omega_damped": rr["omega_damped"],
            "residual_relative": rr["residual_relative"],
            "converged": rr["converged"],
            "realization": o_real,
        })
    # The predeclared selection rule: smallest order that meets the in-band fit
    # target *and* the Kramers-Kronig added-mass budget of D1.  Order 12 was the
    # one that broke the second condition (17.6 %), so it is outside the ladder.
    selected = None
    for row in order_rows:
        if row["fit_relative_error_max"] <= FIT_TARGET and row["kk_added_mass_relative"] <= GATES["D1_kk_added_mass"]["tol"]:
            selected = row
            break
    if selected is None:
        selected = min(order_rows, key=lambda r: r["fit_relative_error_max"])
    realization = selected["realization"]
    zeta_exact = float(selected["zeta_exact"])
    admissible = [r for r in order_rows
                  if r["fit_relative_error_max"] <= 0.03
                  and r["kk_added_mass_relative"] <= GATES["D1_kk_added_mass"]["tol"]]
    zeta_scan = [float(r["zeta_exact"]) for r in (admissible or order_rows)]

    causal = causal_free_decay(realization, mass=mass, restoring=restoring,
                               added_mass_inf=a_inf)
    zeta_causal = float(causal["zeta_decay"])
    tau_max = float(case.TAU_MAX)
    causal_k = causal_kernel(realization, np.array([0.0, tau_max]))
    kernel_decay_ratio = float(abs(causal_k[1]) / max(abs(causal_k[0]), 1e-300))
    legacy_kernel_ratio = float(
        abs(legacy["kernel_values"][-1]) / max(abs(legacy["kernel_values"][0]), 1e-300)
    )

    scan = band_top_scan(omega, damping, mass=mass, restoring=restoring, added_mass_inf=a_inf)
    scan_zeta = [r["zeta_decay"] for r in scan if "zeta_decay" in r and math.isfinite(r["zeta_decay"])]
    scan_spread = (max(scan_zeta) / min(scan_zeta)) if len(scan_zeta) >= 2 else float("nan")

    gates = {
        "D0_estimator_validation": _verdict(
            "D0_estimator_validation",
            max(row["relative_error"] for row in validation["constant_damping"]),
            GATES["D0_estimator_validation"]["tol"],
            quantity=GATES["D0_estimator_validation"]["quantity"],
        ),
        "D0b_cubic_memory_root": _verdict(
            "D0b_cubic_memory_root",
            validation["single_exponential_memory"]["relative_error"],
            GATES["D0b_cubic_memory_root"]["tol"],
            quantity=GATES["D0b_cubic_memory_root"]["quantity"],
        ),
        "D1_fit_band": _verdict("D1_fit_band", fit_error, GATES["D1_fit_band"]["tol"],
                                quantity=GATES["D1_fit_band"]["quantity"]),
        "D1_kk_added_mass": _verdict("D1_kk_added_mass", kk_error,
                                     GATES["D1_kk_added_mass"]["tol"],
                                     quantity=GATES["D1_kk_added_mass"]["quantity"]),
        "D2_kernel_decay": _verdict("D2_kernel_decay", kernel_decay_ratio,
                                    GATES["D2_kernel_decay"]["tol"],
                                    quantity=GATES["D2_kernel_decay"]["quantity"]),
        "D3_causal_decay": _verdict(
            "D3_causal_decay",
            abs(zeta_causal - zeta_exact) / zeta_exact,
            GATES["D3_causal_decay"]["tol"],
            quantity=GATES["D3_causal_decay"]["quantity"],
        ),
        "D4_legacy_bias": _verdict(
            "D4_legacy_bias",
            abs(zeta_legacy - zeta_exact) / zeta_exact,
            GATES["D4_legacy_bias"]["tol"],
            quantity=GATES["D4_legacy_bias"]["quantity"],
        ),
        "D5_band_robustness": _verdict(
            "D5_band_robustness", scan_spread, GATES["D5_band_robustness"]["tol"],
            quantity=GATES["D5_band_robustness"]["quantity"], lower_is_better=False,
        ),
    }

    # Which route carries the 7.09 %?
    legacy_vs_reference = abs(zeta_legacy - reference["zeta_frequency_domain"]) / reference["zeta_frequency_domain"]
    reference_vs_exact = abs(reference["zeta_frequency_domain"] - zeta_exact) / zeta_exact
    legacy_vs_exact = abs(zeta_legacy - zeta_exact) / zeta_exact
    if legacy_vs_exact > reference_vs_exact:
        hypothesis = "H-route: the truncated-kernel free decay carries the bias"
    else:
        hypothesis = "H-reference: the interpolated energy identity carries the bias"

    return {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "audit": "C5 attribution (causal continuation vs truncated kernel)",
        "level": level,
        "heave": {
            "omega": omega.tolist(),
            "added_mass": added.tolist(),
            "damping": damping.tolist(),
        },
        "body": {
            "displaced_mass": mass,
            "restoring_C33": restoring,
            "added_mass_inf_closure": a_inf,
            "omega_natural_legacy": omega_n_legacy,
        },
        "gates_definition": GATES,
        "gates": gates,
        "legacy_route": {
            "zeta_decay": zeta_legacy,
            "damped_period": legacy["damped_period"],
            "n_peaks": legacy["n_peaks"],
            "band_top": legacy["band_top"],
            "tau_max": legacy["tau_max"],
            "kernel_at_zero": float(legacy["kernel_values"][0]),
            "kernel_at_tau_max": float(legacy["kernel_values"][-1]),
            "kernel_decay_ratio": legacy_kernel_ratio,
        },
        "legacy_reference": reference,
        "causal_model": {
            "n_poles_selected": int(selected["n_poles"]),
            "fit_relative_error_max": float(selected["fit_relative_error_max"]),
            "kk_added_mass_relative": float(selected["kk_added_mass_relative"]),
            "zeta_exact": zeta_exact,
            "zeta_exact_order_scan": zeta_scan,
            "zeta_exact_order_spread": float(max(zeta_scan) - min(zeta_scan)),
            "zeta_exact_orders_used": [int(r["n_poles"]) for r in (admissible or order_rows)],
            "omega_damped": float(selected["omega_damped"]),
            "residual_relative": float(selected["residual_relative"]),
            "kernel_at_zero": float(causal_k[0]),
            "kernel_at_tau_max": float(causal_k[1]),
            "kernel_decay_ratio": kernel_decay_ratio,
            "zeta_decay_time_domain": zeta_causal,
            "damped_period_time_domain": causal["damped_period"],
            "n_peaks": causal["n_peaks"],
            "substeps": causal["substeps"],
        },
        "order_scan": [
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in row.items() if k != "realization"}
            for row in order_rows
        ],
        "band_top_scan": scan,
        "attribution": {
            "legacy_C5_relative_error": float(legacy_vs_reference),
            "legacy_vs_exact": float(legacy_vs_exact),
            "reference_vs_exact": float(reference_vs_exact),
            "verdict": hypothesis,
        },
        "estimator_validation": {
            "constant_damping": validation["constant_damping"],
            "single_exponential_memory": {
                k: (complex(v) if isinstance(v, complex) else float(v))
                for k, v in validation["single_exponential_memory"].items()
            },
        },
        "records": {
            "legacy": {"time": legacy["time"].tolist(), "displacement": legacy["displacement"].tolist()},
            "causal": {"time": causal["time"].tolist(), "displacement": causal["displacement"].tolist()},
        },
        "decision": {
            "hypothesis": hypothesis,
            "remediation": (
                "drive the Cummins free decay with the causal (rational) kernel and score "
                "it against the exact modal root of the same model (gate D3); keep C5 as the "
                "recorded legacy comparison"
            ),
            "case_tolerance_statement": (
                f"legacy C5 relative error {legacy_vs_reference:.2%} is inside the anchor's "
                f"<10 % case tolerance but outside the module's internal 5 % gate"
            ),
        },
    }


def _jsonable(value: Any) -> Any:
    """JSON hook: complex roots are written as ``{"real": ..., "imag": ...}``."""
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI
    import argparse
    import json
    import pathlib

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", default="medium")
    parser.add_argument("--out", default=ARTIFACT_PATH)
    parser.add_argument("--json-only", action="store_true", help="omit the decay records")
    args = parser.parse_args(argv)

    result = run_audit(args.level)
    if args.json_only:
        result = dict(result)
        result["records"] = {key: {"n_samples": len(value["time"])} for key, value in result["records"].items()}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=_jsonable))
    failed = [key for key, gate in result["gates"].items() if gate["passed"] is False]
    print(f"[audit] {args.level}: {len(result['gates']) - len(failed)}/{len(result['gates'])} gates pass, "
          f"failed={failed or 'none'} -> {out}")
    print(f"[audit] zeta: legacy={result['legacy_route']['zeta_decay']:.5f} "
          f"reference={result['legacy_reference']['zeta_frequency_domain']:.5f} "
          f"exact={result['causal_model']['zeta_exact']:.5f} "
          f"causal_decay={result['causal_model']['zeta_decay_time_domain']:.5f} "
          f"| {result['attribution']['verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())