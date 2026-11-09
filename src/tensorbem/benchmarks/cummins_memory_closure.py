"""End-to-end closure campaign for the Cummins radiation memory kernel.

A frequency-domain radiation solution answers "what force at this frequency".
A time-domain seakeeping code needs the transient: what force is still acting
after the motion stopped. Cummins' convolution is the bridge, and the bridge
has to carry traffic in both directions:

* frequency -> time, ``B(omega)`` to ``K(tau)`` (and it must be causal:
  ``K(tau < 0) = 0``, no force before the motion);
* time -> frequency, ``K(tau)`` back to ``A(omega)`` **and** ``B(omega)``
  through the Kramers-Kronig pair, on a quadrature network that the forward leg
  never touches;
* time domain against frequency domain, through a free decay whose measured
  decay rate has to match the damping the sweep predicts;
* and a negative control, where the damping is set to zero and the decay rate
  has to follow it to zero, so that "the decay matched" cannot be an artefact of
  the integrator's own numerical damping.

What this module adds to :mod:`tensorbem.time_domain` and
:mod:`tensorbem.causality` (which own the transforms and the principal-value
algebra) is the *campaign*: one body, one sweep, five machine-checkable gates,
and the geometry precondition gate that has to pass before any of the numbers
mean anything.

Gates
-----

``C1`` round trip   ``B -> K -> (B, A)``, per-point relative error ``<= 2 %``.
``C2`` causality    ``K(tau < 0) == 0`` exactly, even parity, ``K(0+) > 0`` with
                    a vanishing relative slope, and the independent
                    Kramers-Kronig reconstruction of ``A`` from ``B``.
``C3`` free decay   ``zeta_decay`` from the Cummins convolution against the
                    frequency-domain damping, ``|delta zeta| / zeta <= 5 %``.
``C4`` negative     ``B(omega) == 0`` must give ``K == 0`` and
     control        ``zeta_decay / zeta_reference <= 0.5 %``.
``C5`` geometry     ``Cb``, ``L``, ``B``, ``T``, ``nabla``, ``S`` must satisfy
     precondition   the platform's geometry contract before C1-C4 run.

Envelope
--------

:func:`error_envelope` measures what the cheap *constant-added-mass* model
(``A = A_inf``, ``B = B(omega_n)``, no convolution) costs as a function of
``omega_p / omega_n`` -- the distance between the radiation band and the
oscillator. It consumes :func:`tensorbem.time_domain.constant_added_mass_error_envelope`.

Case
----

A deeply submerged sphere (``f = 3a``) in deep water: heave ``(A, B)`` from
Hulme's spherical multipole reference, exact to machine precision at every
``ka``, with two independent anchors the case can be checked against -- the
far-field (Haskind) damping, which must equal the pressure-integral damping, and
``A(omega -> inf) = rho * V / 2``, the added mass of the same sphere in an
infinite fluid. A deeply submerged body is deliberately the *hard* case for the
memory transform: its damping is exponentially small above ``ka ~ 2`` and its
added mass varies by only a few percent, so a round trip that closes here is
closing on a kernel three decades smaller than its own value at ``tau = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem import time_domain as td
from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.causality import (
    fit_added_mass_asymptote,
    kkr_added_mass,
)

FloatArray = NDArray[np.float64]

__all__ = [
    "GATE_CB_MAX",
    "GATE_CB_MIN",
    "GATE_TOLERANCE",
    "GEOMETRY_CONTRACT_SOURCE",
    "RadiationSweep",
    "criterion_causality",
    "criterion_free_decay",
    "criterion_geometry",
    "criterion_negative_control",
    "criterion_round_trip",
    "error_envelope",
    "geometry_gate",
    "run_campaign",
    "sphere_geometry",
    "submerged_sphere_sweep",
]

# ---------------------------------------------------------------------------
# geometry precondition (the platform's D3 contract)
# ---------------------------------------------------------------------------
GATE_CB_MIN = 0.30
GATE_CB_MAX = 1.00
GATE_TOLERANCE = 0.05
GEOMETRY_CONTRACT_SOURCE = (
    "ship-performance-platform backend/api/calculations.py D3 geometry gate: "
    "CB_MIN=0.30, CB_MAX=1.00, GEOM_TOL=0.05 on |nabla - L*B*T*Cb|/nabla. "
    "Replicated (not imported) so this repository's tests do not depend on the "
    "platform service; the campaign records the platform HTTP probe separately."
)

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665


def sphere_geometry(radius: float, *, density: float = DEFAULT_DENSITY) -> dict:
    """Geometry of a sphere of radius ``a`` in the platform's gate variables.

    The gate speaks of a box ``L x B x T`` plus a block coefficient, so the
    sphere is described by its circumscribing box: ``L = B = T = 2a``,
    ``nabla = 4/3 pi a**3``, ``Cb = pi/6``, wetted surface ``S = 4 pi a**2``.
    ``Cb`` is therefore an exact property of the shape, not a fitted number.
    """
    a = float(radius)
    if not (a > 0.0 and math.isfinite(a)):
        raise ValueError(f"radius must be positive and finite, got {radius!r}")
    volume = 4.0 / 3.0 * math.pi * a**3
    return {
        "shape": "sphere",
        "length": 2.0 * a,
        "beam": 2.0 * a,
        "draft": 2.0 * a,
        "displacement": volume,
        "wetted_surface": 4.0 * math.pi * a**2,
        "block_coefficient": math.pi / 6.0,
        "displaced_mass": volume * float(density),
        "density": float(density),
    }


def geometry_gate(
    *,
    length: float,
    beam: float,
    draft: float,
    displacement: float,
    wetted_surface: float | None = None,
    block_coefficient: float | None = None,
    tolerance: float = GATE_TOLERANCE,
) -> dict:
    """Check ``Cb``, ``L``, ``B``, ``T``, ``nabla`` (and ``S``) against the contract.

    Two independent failures are possible and both are reported: a block
    coefficient outside the physically realisable window (``0.3 < Cb <= 1``), and
    a displacement that does not match ``L * B * T * Cb`` inside
    ``tolerance``. A declared ``Cb`` is cross-checked against the value implied
    by the box, so a caller cannot pass the window with a number that has
    nothing to do with the geometry. ``S`` is only reported (there is no
    contract on it here) but must be positive when supplied.
    """
    values = {
        "length": float(length),
        "beam": float(beam),
        "draft": float(draft),
        "displacement": float(displacement),
    }
    checks: dict[str, dict] = {}
    for name, value in values.items():
        checks[name] = {
            "value": value,
            "passed": bool(value > 0.0 and math.isfinite(value)),
            "detail": "must be positive and finite",
        }
    box = values["length"] * values["beam"] * values["draft"]
    cb_from_dims = values["displacement"] / box if box > 0.0 else float("nan")
    # The consistency test needs the *declared* block coefficient: comparing
    # nabla with L*B*T*(nabla/(L*B*T)) is a tautology and would pass anything.
    declared = None if block_coefficient is None else float(block_coefficient)
    if declared is None or not math.isfinite(declared):
        consistency = float("inf")
    else:
        consistency = abs(values["displacement"] - box * declared) / values["displacement"]
    checks["block_coefficient"] = {
        "value": cb_from_dims,
        "declared": declared,
        "passed": bool(
            math.isfinite(cb_from_dims)
            and GATE_CB_MIN < cb_from_dims <= GATE_CB_MAX
            and declared is not None
            and math.isfinite(declared)
            and abs(declared - cb_from_dims) <= tolerance * abs(cb_from_dims)
        ),
        "detail": (
            f"required {GATE_CB_MIN} < nabla/(L*B*T) <= {GATE_CB_MAX} and a "
            f"declared Cb consistent with it inside {tolerance:.0%}"
        ),
    }
    checks["displacement_consistency"] = {
        "value_pct": 100.0 * consistency,
        "passed": bool(consistency <= tolerance),
        "detail": f"|nabla - L*B*T*Cb_declared| / nabla must be <= {tolerance:.0%}",
    }
    if wetted_surface is not None:
        surface = float(wetted_surface)
        checks["wetted_surface"] = {
            "value": surface,
            "passed": bool(surface > 0.0 and math.isfinite(surface)),
            "detail": "must be positive and finite (no contract value here)",
        }
    passed = all(entry["passed"] for entry in checks.values())
    return {
        "passed": passed,
        "cb_from_dims": cb_from_dims,
        "displacement_consistency_pct": 100.0 * consistency,
        "block_coefficient": cb_from_dims if math.isfinite(cb_from_dims) else None,
        "tolerance": float(tolerance),
        "cb_window": (GATE_CB_MIN, GATE_CB_MAX),
        "checks": checks,
        "contract_source": GEOMETRY_CONTRACT_SOURCE,
        "geometry": values,
        "wetted_surface": None if wetted_surface is None else float(wetted_surface),
    }


# ---------------------------------------------------------------------------
# the radiation sweep of the case body
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RadiationSweep:
    """Heave ``(A, B)`` of a deeply submerged sphere, with its two anchors."""

    omega: FloatArray
    added_mass: FloatArray
    damping: FloatArray
    radius: float
    submergence: float
    density: float
    gravity: float
    ka: FloatArray
    added_mass_inf: float
    added_mass_inf_anchor: float
    asymptote_residual: float
    far_field_error_max: float
    metadata: dict = field(default_factory=dict)

    @property
    def damping_peak_omega(self) -> float:
        return float(self.omega[int(np.argmax(self.damping))])

    @property
    def damping_peak(self) -> float:
        return float(np.max(self.damping))

    @property
    def displaced_mass(self) -> float:
        volume = 4.0 / 3.0 * math.pi * self.radius**3
        return volume * self.density

    def summary(self) -> dict:
        return {
            "radius": self.radius,
            "submergence": self.submergence,
            "submergence_ratio": self.submergence / self.radius,
            "density": self.density,
            "gravity": self.gravity,
            "points": int(self.omega.size),
            "ka_range": (float(self.ka[0]), float(self.ka[-1])),
            "omega_range": (float(self.omega[0]), float(self.omega[-1])),
            "damping_peak": self.damping_peak,
            "damping_peak_omega": self.damping_peak_omega,
            "added_mass_range": (float(self.added_mass.min()), float(self.added_mass.max())),
            "added_mass_relative_change": float(
                (self.added_mass.max() - self.added_mass.min()) / self.added_mass.max()
            ),
            "added_mass_inf": self.added_mass_inf,
            "added_mass_inf_anchor": self.added_mass_inf_anchor,
            "added_mass_inf_anchor_error_pct": 100.0
            * abs(self.added_mass_inf - self.added_mass_inf_anchor)
            / self.added_mass_inf_anchor,
            "asymptote_relative_residual": self.asymptote_residual,
            "far_field_damping_error_max": self.far_field_error_max,
            "displaced_mass": self.displaced_mass,
        }


def submerged_sphere_sweep(
    *,
    radius: float = 1.0,
    submergence_ratio: float = 3.0,
    ka_min: float = 0.03,
    ka_max: float = 6.0,
    points: int = 150,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    terms: int = 20,
) -> RadiationSweep:
    """Heave ``(A, B)`` sweep of a submerged sphere from the multipole reference.

    The two anchors are computed with the sweep, not asserted afterwards:

    * ``added_mass_inf_anchor`` = ``rho * V / 2`` (sphere in an unbounded fluid);
    * ``far_field_error_max`` = the largest relative gap between the
      pressure-integral damping and the far-field (Haskind) damping returned by
      the same multipole solution.
    """
    a = float(radius)
    f = float(submergence_ratio) * a
    if not (a > 0.0):
        raise ValueError("radius must be positive")
    if not (f > a):
        raise ValueError("the sphere must be fully submerged: submergence > radius")
    count = int(points)
    if count < 24:
        raise ValueError("at least 24 frequencies are needed for a smooth sweep")
    ka = np.geomspace(float(ka_min), float(ka_max), count)
    omega = ka * float(gravity) / a
    added = np.zeros(count)
    damping = np.zeros(count)
    far_field_error = 0.0
    damping_peak_scale = float(density) * float(gravity) * a**2 * 1.0e-12
    for index, value in enumerate(ka):
        result = submerged_sphere_wave_multipole(
            float(value),
            radius=a,
            submergence=f,
            density=density,
            n_terms=terms,
            gravity=gravity,
        )
        added[index] = result.heave_added_mass
        damping[index] = result.heave_damping
        # Haskind/far-field damping is dimensionless for this reference; put it
        # in the same units as the pressure-integral damping before comparing.
        far = (
            result.heave_far_field_damping_dimless
            * float(density)
            * result.omega
            * (4.0 / 3.0 * math.pi * a**3)
        )
        if far > 0.0 and result.heave_damping > 1.0e-12 * damping_peak_scale:
            far_field_error = max(
                far_field_error, abs(result.heave_damping - far) / far
            )
    fit = fit_added_mass_asymptote(omega, added)
    volume = 4.0 / 3.0 * math.pi * a**3
    return RadiationSweep(
        omega=omega,
        added_mass=added,
        damping=damping,
        radius=a,
        submergence=f,
        density=float(density),
        gravity=float(gravity),
        ka=ka,
        added_mass_inf=float(fit.added_mass_inf),
        added_mass_inf_anchor=0.5 * float(density) * volume,
        asymptote_residual=float(fit.relative_residual),
        far_field_error_max=float(far_field_error),
        metadata={"terms": int(terms), "source": "Hulme/Thorne-Wang-Linton multipoles"},
    )


# ---------------------------------------------------------------------------
# helpers shared by the gates
# ---------------------------------------------------------------------------
def _verdict(identifier: str, title: str, threshold: str, passed: bool, values: dict) -> dict:
    return {
        "id": identifier,
        "title": title,
        "threshold": threshold,
        "passed": bool(passed),
        "values": values,
    }


def _kernel(sweep: RadiationSweep, dt: float, tau_max: float) -> td.MemoryKernel:
    tau = np.arange(0.0, float(tau_max) + 0.5 * dt, dt)
    return td.radiation_memory_kernel(sweep.omega, sweep.damping, tau)


# ---------------------------------------------------------------------------
# C1: frequency -> time -> frequency
# ---------------------------------------------------------------------------
def criterion_round_trip(
    sweep: RadiationSweep,
    *,
    dt: float = 0.02,
    tau_max: float | None = None,
    samples: int = 300,
    tolerance: float = 0.02,
) -> dict:
    """``B -> K -> (B, A)`` with the two legs sharing no quadrature rule."""
    record = td.fourier_round_trip(
        sweep.omega,
        sweep.added_mass,
        sweep.damping,
        sweep.added_mass_inf,
        dt=dt,
        tau_max=tau_max,
        samples=samples,
    )
    damping_error = record.damping_relative_error_max
    added_error = record.added_mass_relative_error_max
    values = {
        "damping_relative_error_max": damping_error,
        "added_mass_relative_error_max": added_error,
        "band": [float(record.band[0]), float(record.band[1])],
        "tau_max": float(record.tau_max),
        "dt": float(dt),
        "samples": int(samples),
        "kernel_decay_time_to_one_percent": record.diagnostics["decay_time_to_one_percent"],
        "implied_added_mass_inf_mean": record.implied_added_mass_inf["mean"],
        "implied_added_mass_inf_relative_spread": record.implied_added_mass_inf[
            "relative_spread"
        ],
        "added_mass_inf_used": float(sweep.added_mass_inf),
    }
    return _verdict(
        "C1",
        "frequency -> time -> frequency round-trip closure",
        f"per-point relative error <= {tolerance:.0%} for both A(omega) and B(omega)",
        damping_error <= tolerance and added_error <= tolerance,
        values,
    )


# ---------------------------------------------------------------------------
# C2: causality of the kernel and the Kramers-Kronig pairing
# ---------------------------------------------------------------------------
def criterion_causality(
    sweep: RadiationSweep,
    *,
    dt: float = 0.02,
    tau_max: float = 200.0,
    samples: int = 12,
    tolerance: float = 0.02,
    slope_tolerance: float = 1.0e-2,
) -> dict:
    """``K(tau < 0) = 0``, parity, the ``tau = 0+`` jump, and ``A`` from ``B``."""
    kernel = _kernel(sweep, dt=dt, tau_max=tau_max)
    scale = float(np.max(np.abs(kernel.values)))
    negative = kernel(np.array([-10.0 * dt, -dt, -0.5 * dt, -1.0e-6]))
    negative_violation = float(np.max(np.abs(negative)))
    _, two_sided = td.two_sided_kernel(kernel)
    even_residual = float(np.max(np.abs(two_sided - two_sided[::-1])) / scale)
    jump = float(kernel.values[0])
    # K is even, so K'(0) = 0 analytically; the one-step difference is therefore
    # O((omega_max dt)^2) and must be read on a refined grid, otherwise it
    # measures the grid, not the kernel.
    slope_dt = min(float(dt), 2.0e-3)
    refined = _kernel(sweep, dt=slope_dt, tau_max=1.0)
    relative_slope = float((refined.values[1] - refined.values[0]) / scale)
    slope_grid_bound = 0.5 * (refined.omega_max * slope_dt) ** 2
    jump_is_maximum = bool(
        float(np.max(np.abs(kernel.values))) <= jump * (1.0 + 1.0e-9)
    )
    kernel_zero = (2.0 / math.pi) * float(
        np.trapezoid(sweep.damping, sweep.omega)
    )
    jump_error = abs(jump - kernel_zero) / abs(kernel_zero)

    # Independent Kramers-Kronig leg: A from B through principal-value quadrature.
    grid = sweep.omega
    peak_added = float(np.max(np.abs(sweep.added_mass)))
    sample_frequencies = np.geomspace(
        float(grid[0]) * 3.0, float(grid[-1]) * 0.5, int(samples)
    )
    errors = []
    for frequency in sample_frequencies:
        if np.any(np.abs(grid - frequency) <= 0.0):
            raise ValueError("a Kramers-Kronig sample landed on a tabulated node")
        predicted = kkr_added_mass(
            grid, sweep.damping, float(frequency), added_mass_inf=sweep.added_mass_inf
        )
        exact = float(np.interp(frequency, grid, sweep.added_mass))
        errors.append(abs(predicted - exact) / peak_added)
    kkr_error = float(max(errors))
    # Conditioning of the A-from-B leg: the principal-value integral of A - A_inf
    # is O(nabla A) while the answer can be far smaller, so the achievable
    # relative accuracy has a floor. Report it, do not hide it.
    deficit = sweep.added_mass - sweep.added_mass_inf
    floor = 2.0 / math.pi * abs(float(np.trapezoid(deficit, grid)))

    values = {
        "kernel_at_negative_times_max_abs": negative_violation,
        "even_parity_residual_relative": even_residual,
        "kernel_at_zero": jump,
        "kernel_at_zero_from_damping_integral": kernel_zero,
        "kernel_at_zero_relative_error": jump_error,
        "kernel_relative_slope_at_zero": relative_slope,
        "kernel_slope_grid": slope_dt,
        "kernel_slope_grid_bound": slope_grid_bound,
        "kernel_at_zero_is_global_maximum": jump_is_maximum,
        "kkr_added_mass_error_max_relative_to_peak": kkr_error,
        "kkr_sample_frequencies": [float(v) for v in sample_frequencies],
        "kkr_conditioning_floor_absolute": floor,
        "kkr_conditioning_floor_relative": floor / peak_added,
        "dt": float(dt),
        "tau_max": float(tau_max),
    }
    passed = (
        negative_violation == 0.0
        and even_residual <= 1.0e-9
        and jump > 0.0
        and jump_is_maximum
        and jump_error <= 0.05
        and abs(relative_slope) <= min(slope_tolerance, 2.0 * slope_grid_bound)
        and kkr_error <= tolerance
    )
    return _verdict(
        "C2",
        "causality of K(tau) and the Kramers-Kronig pairing of A and B",
        (
            "K(tau<0) == 0 exactly, even parity, K(0+) > 0 with a positive jump of "
            "(2/pi) int B dw, and A reconstructed from B through the principal "
            f"value to <= {tolerance:.0%} of the added-mass peak"
        ),
        passed,
        values,
    )


# ---------------------------------------------------------------------------
# C3: free decay of the Cummins equation against the frequency domain
# ---------------------------------------------------------------------------
def criterion_free_decay(
    sweep: RadiationSweep,
    *,
    mass_fraction: float = 0.1,
    dt_list: Sequence[float] = (0.02, 0.01, 0.005, 0.0025),
    duration: float = 60.0,
    tau_max: float = 300.0,
    tolerance: float = 0.05,
) -> dict:
    """Damping ratio from the convolution against the frequency-domain value.

    The oscillator mass is ``mass_fraction`` times the displaced mass and the
    stiffness is set to ``(M + A_inf) omega_p**2``, so the natural frequency
    sits at the damping peak and the comparison is not a mistuned-stiffness
    artefact. The time step is refined: the trapezoidal memory integral is
    first order in ``dt`` here (measured ratio ~0.5 per halving), so the
    converged decay rate is the Richardson limit ``2 sigma(dt) - sigma(2 dt)``
    and the raw series is reported next to it.
    """
    omega_p = sweep.damping_peak_omega
    mass = float(mass_fraction) * sweep.displaced_mass
    inertia = mass + sweep.added_mass_inf
    restoring = inertia * omega_p**2
    response = td.half_power_damping(
        sweep.omega,
        sweep.added_mass,
        sweep.damping,
        mass=mass,
        restoring=restoring,
        omega_n=omega_p,
    )
    kernel = _kernel(sweep, dt=float(dt_list[-1]), tau_max=tau_max)
    series = []
    for dt in dt_list:
        decay = td.cummins_free_decay(
            kernel,
            mass=mass,
            restoring=restoring,
            added_mass_inf=sweep.added_mass_inf,
            dt=float(dt),
            duration=float(duration),
        )
        metrics = td.decay_metrics(decay.time, decay.displacement)
        series.append(
            {
                "dt": float(dt),
                "decay_rate": metrics["decay_rate"],
                "frequency": metrics["frequency"],
                "extrema_used": metrics["extrema_used"],
                "fit_r_squared": metrics["fit_r_squared"],
            }
        )
    fine = series[-1]
    coarse = series[-2]
    extrapolated = 2.0 * fine["decay_rate"] - coarse["decay_rate"]
    omega_n = response["omega_n"]
    zeta_decay = extrapolated / omega_n
    zeta_energy = response["zeta_energy"]
    zeta_half_power = response["zeta"]
    deviation_energy = abs(zeta_decay - zeta_energy) / zeta_energy
    deviation_half_power = abs(zeta_decay - zeta_half_power) / zeta_half_power
    values = {
        "mass": mass,
        "mass_fraction_of_displaced": float(mass_fraction),
        "restoring": restoring,
        "omega_n": omega_n,
        "omega_p": omega_p,
        "zeta_decay": zeta_decay,
        "zeta_energy_identity": zeta_energy,
        "zeta_half_power": zeta_half_power,
        "zeta_decay_vs_energy_pct": 100.0 * deviation_energy,
        "zeta_decay_vs_half_power_pct": 100.0 * deviation_half_power,
        "decay_rate_extrapolated": extrapolated,
        "decay_rate_series": series,
        "convergence_ratio": (
            (series[1]["decay_rate"] - series[0]["decay_rate"])
            / (series[2]["decay_rate"] - series[1]["decay_rate"])
            if len(series) > 2
            else float("nan")
        ),
        "measured_frequency": fine["frequency"],
        "duration": float(duration),
        "tau_max": float(tau_max),
    }
    return _verdict(
        "C3",
        "Cummins free decay against the frequency-domain damping",
        f"|zeta_decay - zeta_frequency| / zeta_frequency <= {tolerance:.0%}, "
        "zeta_frequency from the energy identity at the resonance",
        deviation_energy <= tolerance,
        values,
    )


# ---------------------------------------------------------------------------
# C4: negative control -- no damping, no decay
# ---------------------------------------------------------------------------
def criterion_negative_control(
    sweep: RadiationSweep,
    *,
    mass_fraction: float = 0.1,
    dt: float = 0.01,
    duration: float = 60.0,
    tau_max: float = 300.0,
    reference_decay_rate: float | None = None,
    tolerance: float = 0.005,
) -> dict:
    """``B(omega) = 0`` must give a zero kernel and a zero decay rate.

    Without this gate a decay reported as "small" could be the integrator's own
    numerical damping rather than the fluid's. The same oscillator, grid and
    duration are reused with the damping vector zeroed.
    """
    omega_p = sweep.damping_peak_omega
    mass = float(mass_fraction) * sweep.displaced_mass
    inertia = mass + sweep.added_mass_inf
    restoring = inertia * omega_p**2
    zero = np.zeros_like(sweep.damping)
    kernel = _kernel(
        RadiationSweep(
            omega=sweep.omega,
            added_mass=sweep.added_mass,
            damping=zero,
            radius=sweep.radius,
            submergence=sweep.submergence,
            density=sweep.density,
            gravity=sweep.gravity,
            ka=sweep.ka,
            added_mass_inf=sweep.added_mass_inf,
            added_mass_inf_anchor=sweep.added_mass_inf_anchor,
            asymptote_residual=sweep.asymptote_residual,
            far_field_error_max=sweep.far_field_error_max,
        ),
        dt=dt,
        tau_max=tau_max,
    )
    kernel_max = float(np.max(np.abs(kernel.values)))
    decay = td.cummins_free_decay(
        kernel,
        mass=mass,
        restoring=restoring,
        added_mass_inf=sweep.added_mass_inf,
        dt=float(dt),
        duration=float(duration),
    )
    metrics = td.decay_metrics(decay.time, decay.displacement)
    reference = (
        float(reference_decay_rate)
        if reference_decay_rate is not None
        else abs(sweep.damping_peak / (2.0 * inertia) / omega_p * omega_p)
    )
    ratio = abs(metrics["decay_rate"]) / reference if reference else float("inf")
    values = {
        "kernel_max_abs": kernel_max,
        "kernel_is_identically_zero": kernel_max == 0.0,
        "decay_rate_measured": metrics["decay_rate"],
        "decay_rate_reference": reference,
        "decay_rate_ratio": ratio,
        "amplitude_ratio_first_last": metrics["amplitude_ratio_first_last"],
        "extrema_used": metrics["extrema_used"],
        "dt": float(dt),
        "duration": float(duration),
    }
    passed = kernel_max == 0.0 and ratio <= tolerance
    return _verdict(
        "C4",
        "negative control: zero damping gives zero decay",
        f"K == 0 identically and |sigma_measured| / |sigma_reference| <= {tolerance:.1%}",
        passed,
        values,
    )


# ---------------------------------------------------------------------------
# C5: geometry precondition
# ---------------------------------------------------------------------------
def criterion_geometry(
    sweep: RadiationSweep, *, tolerance: float = GATE_TOLERANCE
) -> dict:
    """Gate the case body's geometry, and prove the gate can reject bad input."""
    geometry = sphere_geometry(sweep.radius, density=sweep.density)
    primary = geometry_gate(
        length=geometry["length"],
        beam=geometry["beam"],
        draft=geometry["draft"],
        displacement=geometry["displacement"],
        wetted_surface=geometry["wetted_surface"],
        block_coefficient=geometry["block_coefficient"],
        tolerance=tolerance,
    )
    # Negative controls: the gate must not be a rubber stamp.
    inflated = geometry_gate(
        length=geometry["length"],
        beam=geometry["beam"],
        draft=geometry["draft"],
        displacement=1.25 * geometry["displacement"],
        block_coefficient=geometry["block_coefficient"],
        tolerance=tolerance,
    )
    impossible = geometry_gate(
        length=geometry["length"],
        beam=geometry["beam"],
        draft=geometry["draft"],
        displacement=geometry["displacement"],
        block_coefficient=1.2,
        tolerance=tolerance,
    )
    values = {
        "geometry": geometry,
        "gate": primary,
        "negative_control_inflated_displacement": {
            "passed": inflated["passed"],
            "displacement_consistency_pct": inflated["displacement_consistency_pct"],
        },
        "negative_control_impossible_cb": {
            "passed": impossible["passed"],
            "block_coefficient": impossible["block_coefficient"],
            "cb_from_dims": impossible["cb_from_dims"],
        },
        "cb_window": list(primary["cb_window"]),
        "tolerance": float(tolerance),
    }
    passed = (
        primary["passed"]
        and not inflated["passed"]
        and not impossible["passed"]
    )
    return _verdict(
        "C5",
        "geometry precondition gate (Cb, L, B, T, nabla, S)",
        (
            f"Cb in ({GATE_CB_MIN}, {GATE_CB_MAX}], |nabla - L*B*T*Cb|/nabla <= "
            f"{tolerance:.0%}, S > 0, and both corrupt inputs rejected"
        ),
        passed,
        values,
    )


# ---------------------------------------------------------------------------
# error envelope of the constant-added-mass model
# ---------------------------------------------------------------------------
def error_envelope(
    sweep: RadiationSweep,
    *,
    mass_fraction: float = 0.1,
    ratios: Sequence[float] = (0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5),
    samples: int = 4001,
) -> dict:
    """Peak/phase error of ``A = A_inf``, ``B = B(omega_n)`` versus ``omega_p/omega_n``."""
    mass = float(mass_fraction) * sweep.displaced_mass
    envelope = td.constant_added_mass_error_envelope(
        sweep.omega,
        sweep.added_mass,
        sweep.damping,
        mass=mass,
        ratios=ratios,
        added_mass_inf=sweep.added_mass_inf,
        samples=samples,
    )
    amplitude = np.abs(envelope.peak_amplitude_error)
    frequency = np.abs(envelope.peak_frequency_error)
    # A resonant amplitude error is only informative while the resonance shift
    # stays inside the half-power width; outside it the amplitude ratio is
    # 1/(2 zeta) and a 0.1 % frequency error is a 100 % amplitude error.
    zeta = envelope.zeta_exact
    informative = frequency <= np.maximum(zeta, 1.0e-12)
    curvature = np.abs(envelope.phase_error_degrees)
    rows = []
    for index, ratio in enumerate(envelope.ratios):
        rows.append(
            {
                "omega_p_over_omega_n": float(ratio),
                "omega_n": float(envelope.natural_frequencies[index]),
                "added_mass_relative_change": float(
                    envelope.added_mass_relative_change[index]
                ),
                "peak_amplitude_error": float(envelope.peak_amplitude_error[index]),
                "peak_frequency_error": float(envelope.peak_frequency_error[index]),
                "phase_error_degrees": float(envelope.phase_error_degrees[index]),
                "zeta_exact": float(zeta[index]),
                "zeta_model": float(envelope.zeta_model[index]),
                "damping_ratio_relative_error": float(
                    envelope.damping_ratio_error[index]
                ),
                "amplitude_error_informative": bool(informative[index]),
            }
        )
    informative_amplitudes = amplitude[informative]
    worst_amplitude = (
        float(np.max(informative_amplitudes)) if informative_amplitudes.size else float("nan")
    )
    worst_amplitude_ratio = (
        float(envelope.ratios[informative][int(np.argmax(informative_amplitudes))])
        if informative_amplitudes.size
        else float("nan")
    )
    worst_phase_index = int(np.argmax(curvature))
    return {
        "mass": mass,
        "mass_fraction_of_displaced": float(mass_fraction),
        "added_mass_inf": float(sweep.added_mass_inf),
        "omega_p": float(envelope.omega_p),
        "ratios": [float(r) for r in envelope.ratios],
        "rows": rows,
        "worst_informative_peak_amplitude_error": worst_amplitude,
        "worst_informative_peak_amplitude_ratio": worst_amplitude_ratio,
        "worst_phase_error_degrees": float(curvature[worst_phase_index]),
        "worst_phase_error_ratio": float(envelope.ratios[worst_phase_index]),
        "informative_points": int(np.count_nonzero(informative)),
        "note": (
            "peak-amplitude errors are only reported as worst-case where the "
            "resonance shift is inside the half-power width "
            "(|dw|/w <= zeta); elsewhere the amplitude itself is 1/(2 zeta) "
            "sensitive and the error is a conditioning artefact, not physics"
        ),
    }


# ---------------------------------------------------------------------------
# the campaign
# ---------------------------------------------------------------------------
def run_campaign(
    *,
    radius: float = 1.0,
    submergence_ratio: float = 3.0,
    points: int = 150,
    ka_max: float = 6.0,
    round_trip_dt: float = 0.02,
    mass_fraction: float = 0.1,
) -> dict:
    """Run C5 then C1-C4 and the envelope on one sweep; return the record."""
    started = time.time()
    sweep = submerged_sphere_sweep(
        radius=radius,
        submergence_ratio=submergence_ratio,
        points=points,
        ka_max=ka_max,
    )
    gates = [criterion_geometry(sweep)]
    round_trip = criterion_round_trip(sweep, dt=round_trip_dt)
    gates.append(round_trip)
    gates.append(criterion_causality(sweep, dt=round_trip_dt))
    decay = criterion_free_decay(sweep, mass_fraction=mass_fraction)
    gates.append(decay)
    gates.append(
        criterion_negative_control(
            sweep,
            mass_fraction=mass_fraction,
            reference_decay_rate=decay["values"]["decay_rate_extrapolated"],
        )
    )
    envelope = error_envelope(sweep, mass_fraction=mass_fraction)
    return {
        "case": {
            "title": "Cummins radiation memory kernel closure on a deeply submerged sphere",
            "body": "deeply submerged sphere, heave, deep water",
            "reference": "Hulme (1982) / Thorne-Wang-Linton spherical multipoles",
            "sweep": sweep.summary(),
        },
        "gates": gates,
        "all_passed": all(gate["passed"] for gate in gates),
        "envelope": envelope,
        "runtime_seconds": time.time() - started,
        "tolerances": {
            "round_trip": 0.02,
            "kramers_kronig": 0.02,
            "free_decay": 0.05,
            "negative_control": 0.005,
            "geometry": GATE_TOLERANCE,
        },
    }