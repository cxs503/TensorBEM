"""Replace the first-draft attribution with the evidence-backed version."""
from pathlib import Path

path = Path("src/tensorbem/benchmarks/case_bem_cummins_wigley.py")
text = path.read_text()

start = text.index("def decay_bias_attribution(")
end = text.index("def dof_decay_audit(")
NEW = '''def _tail_extrapolated_sweep(sweep: Sweep, dof: int, *, ratio: float = 10.0):
    """Band top closed with the measured power-law decay ``B ~ c omega^-p``.

    The sweep stops at ``omega_max`` with the damping still at 20-35 % of its
    peak, so the kernel is the transform of a *truncated* spectrum.  Extending it
    to ``ratio * omega_max`` on the local exponent (fitted on the last four
    nodes) is the cheapest way to ask how much of a C5 residual is band
    truncation: if the decay moves, it was truncation, and if it does not, the
    residual is in the estimator.
    """
    omega = np.asarray(sweep.omega, dtype=float)
    damping = np.asarray(sweep.damping[:, dof, dof], dtype=float)
    head = min(4, omega.size)
    w_fit, b_fit = omega[-head:], damping[-head:]
    keep = b_fit > 0.0
    if keep.sum() < 2:
        return None
    slope, intercept = np.polyfit(np.log(w_fit[keep]), np.log(b_fit[keep]), 1)
    tail_omega = np.geomspace(omega[-1], omega[-1] * float(ratio), 120)[1:]
    tail_damping = math.exp(intercept) * tail_omega**slope
    return np.concatenate([omega, tail_omega]), np.concatenate([damping, tail_damping]), -float(slope)


def _decay_zeta_for(sweep: Sweep, dof: int, *, damping, omega, added_mass_inf: float,
                    amplitude_floor: float = None):
    """Run one Cummins free decay and return ``(zeta, period, peaks, record)``."""
    oscillator = oscillator_properties(dof, label=sweep.label)
    tau = np.arange(0.0, TAU_MAX + 0.5 * DT, DT)
    kernel = td.radiation_memory_kernel(omega, damping, tau)
    decay = td.cummins_free_decay(
        kernel,
        mass=oscillator["inertia"],
        restoring=oscillator["stiffness"],
        added_mass_inf=added_mass_inf,
        initial_displacement=1.0,
        dt=DT,
        duration=DURATION,
    )
    kwargs = {} if amplitude_floor is None else {"amplitude_floor": amplitude_floor}
    zeta, period, peaks = _decay_zeta(
        np.asarray(decay.time), np.asarray(decay.displacement), **kwargs
    )
    return zeta, period, peaks, decay


def decay_bias_attribution(sweep: Sweep, *, dof: int) -> dict[str, Any]:
    """Where does a C5 residual come from?  Reported, never judged.

    The gate is untouched: it compares :func:`free_decay_gate`'s two numbers
    against the declared bar.  This function only says what the residual is made
    of, from four independent probes:

    ``numbers``
        the frequency-domain estimate is evaluated at ``A(omega_n)`` (what the
        gate uses) *and* at ``A_inf`` (what the oscillator's constant term is),
        so the two conventions can be told apart;
    ``A dispersion``
        ``(max A - min A) / A(omega_n)`` over the band, plus the added mass
        implied by the *measured* decay period,
        ``A_eff = C / omega_n,decay**2 - I``.  A mode whose ``A`` swings across
        the band is the mode a single-frequency ``zeta`` cannot describe, and
        ``A_eff`` measures the resulting memory contribution directly;
    ``tail closure``
        the same decay with the damping spectrum extended past the band top on
        the measured power law (:func:`_tail_extrapolated_sweep`): the part of
        the residual that band truncation owns;
    ``window``
        the log-decrement re-fitted with a ten-times lower amplitude floor.  A
        residual that moves with the window is an envelope that is *not* a single
        exponential, i.e. the memory, not the numbers.
    """
    oscillator = oscillator_properties(dof, label=sweep.label)
    inertia, stiffness = oscillator["inertia"], oscillator["stiffness"]
    omega = np.asarray(sweep.omega, dtype=float)
    added_mass = np.asarray(sweep.added_mass[:, dof, dof], dtype=float)
    damping = np.asarray(sweep.damping[:, dof, dof], dtype=float)
    row: dict[str, Any] = {
        "dof": int(dof),
        "dof_name": DOF_NAMES[dof],
        "inertia": inertia,
        "stiffness": stiffness,
    }
    if float(np.min(damping)) < 0.0:
        row["status"] = "undecidable: non-physical negative radiation damping"
        return row
    closure = kk_added_mass_closure(omega, added_mass, damping)
    added_mass_inf = float(closure.added_mass_inf)
    omega_n = math.sqrt(stiffness / (inertia + added_mass_inf))
    for _ in range(80):
        omega_n = math.sqrt(stiffness / (inertia + float(np.interp(omega_n, omega, added_mass))))
    a_at = float(np.interp(omega_n, omega, added_mass))
    b_at = float(np.interp(omega_n, omega, damping))
    zeta_with_a_omega = b_at / (2.0 * (inertia + a_at) * omega_n)
    zeta_with_a_inf = b_at / (2.0 * (inertia + added_mass_inf) * omega_n)

    zeta_base, period, peaks, _ = _decay_zeta_for(
        sweep, dof, damping=damping, omega=omega, added_mass_inf=added_mass_inf
    )
    omega_n_decay = (
        2.0 * math.pi / period / math.sqrt(max(1.0 - zeta_base**2, 1e-12))
        if math.isfinite(period) and period > 0.0 and math.isfinite(zeta_base)
        else float("nan")
    )
    a_effective = (
        stiffness / omega_n_decay**2 - inertia if math.isfinite(omega_n_decay) else float("nan")
    )

    tail = _tail_extrapolated_sweep(sweep, dof)
    tail_row: dict[str, Any] = {"available": tail is not None}
    if tail is not None:
        tail_omega, tail_damping, exponent = tail
        zeta_tail, period_tail, peaks_tail, _ = _decay_zeta_for(
            sweep, dof, damping=tail_damping, omega=tail_omega, added_mass_inf=added_mass_inf
        )
        tail_row.update(
            {
                "tail_exponent": exponent,
                "omega_max_extended_rad_s": float(tail_omega[-1]),
                "zeta_decay": zeta_tail,
                "relative_error": (
                    abs(zeta_tail - zeta_with_a_omega) / zeta_with_a_omega
                    if zeta_with_a_omega > 0.0
                    else float("nan")
                ),
                "period_s": period_tail,
                "n_peaks": peaks_tail,
            }
        )

    window_floor = 0.1 * DECAY_AMPLITUDE_FLOOR
    zeta_wide, period_wide, peaks_wide, _ = _decay_zeta_for(
        sweep,
        dof,
        damping=damping,
        omega=omega,
        added_mass_inf=added_mass_inf,
        amplitude_floor=window_floor,
    )
    row.update(
        {
            "status": "reported",
            "natural_frequency_rad_s": omega_n,
            "natural_frequency_from_decay_rad_s": omega_n_decay,
            "added_mass_at_natural": a_at,
            "added_mass_inf": added_mass_inf,
            "added_mass_dispersion_across_band": float((np.max(added_mass) - np.min(added_mass)) / a_at),
            "added_mass_effective_from_decay": a_effective,
            "added_mass_effective_over_band_value": a_effective / a_at if a_at > 0.0 else float("nan"),
            "zeta_frequency_domain": zeta_with_a_omega,
            "zeta_frequency_domain_with_a_inf": zeta_with_a_inf,
            "zeta_decay": zeta_base,
            "relative_error": (
                abs(zeta_base - zeta_with_a_omega) / zeta_with_a_omega
                if zeta_with_a_omega > 0.0
                else float("nan")
            ),
            "tail_closure": tail_row,
            "window": {
                "amplitude_floor": window_floor,
                "zeta_decay": zeta_wide,
                "relative_error": (
                    abs(zeta_wide - zeta_with_a_omega) / zeta_with_a_omega
                    if zeta_with_a_omega > 0.0
                    else float("nan")
                ),
                "period_s": period_wide,
                "n_peaks": peaks_wide,
                "shift_from_declared_window": (
                    abs(zeta_base - zeta_wide) / zeta_base if zeta_base > 0.0 else float("nan")
                ),
            },
            "n_peaks": peaks,
            "period_s": period,
        }
    )
    return row


'''
text = text[:start] + NEW + text[end:]
path.write_text(text)
print("attribution replaced")