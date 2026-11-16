"""Splice the repaired C11 (state-space reduction) gate into the case module."""
import pathlib

path = pathlib.Path("src/tensorbem/benchmarks/case_bem_cummins_wigley.py")
text = path.read_text()

start = text.index("def state_space_gate(")
end = text.index("def case_verdict(gates: dict[str, Any]) -> dict[str, Any]:")
old = text[start:end]

new = '''#: Declared bar of gate ``C11`` (state-space reduction).  ``C5``'s 5% bar
#: compares two routes of the *same* full model; ``C11`` compares a reduced
#: model to it, so its bar is the case-pool acceptance tolerance (10%).
STATE_SPACE_BAR = 0.10


def state_space_gate(
    sweep: Sweep,
    *,
    dof: int = 2,
    target_error: float = 0.03,
    max_poles: int = 12,
    corrupt: bool = False,
) -> dict[str, Any]:
    """C11: a reduced-order state-space realization must reproduce the convolution.

    Why this gate exists
    --------------------
    ``C4``/``M3`` certify ``B -> K -> B`` and ``C5``/``C5b``/``C5c`` certify the
    convolution route against the frequency-domain damping.  None of them
    touches the substitution every operational seakeeping code makes: replacing
    the ``O(N^2)`` memory convolution by a finite sum of decaying exponentials
    (vector fit -> state-space realization).  That substitution is exact only at
    infinite order, so it needs its own acceptance test, with the reduction
    order capped at ``max_poles`` and the fit required to reach ``target_error``
    on the *measured* damping.

    A spectrum the cap cannot represent is reported as *undecidable*, not as a
    physics FAIL: ``adaptive_order_fit`` landing above ``target_error`` at
    ``max_poles`` says the ROM is too small, which is a statement about the ROM
    order and not about the BEM coefficients.

    ``corrupt=True`` is the gate's own negative control: the measured damping is
    modulated by ``1 + 0.3 cos(7 omega)``, which no 12-pole real fit at 3% can
    absorb, so the fit error and the decay comparison must both go bad.
    """
    dof = int(dof)
    if dof not in RESTORING_MODES:
        raise ValueError(
            f"C11 needs a hydrostatic restoring mode {RESTORING_MODES}; "
            f"dof {dof} ({DOF_NAMES[dof]}) has none, so there is no oscillator "
            f"whose decay could be compared"
        )
    omega = np.asarray(sweep.omega, dtype=float)
    measured = np.asarray(sweep.damping[:, dof, dof], dtype=float)
    added = np.asarray(sweep.added_mass[:, dof, dof], dtype=float)
    scale = float(np.max(np.abs(measured))) if measured.size else 0.0
    distorted = measured * (1.0 + 0.3 * np.cos(7.0 * omega)) if corrupt else measured

    identifier = "C11_control" if corrupt else "C11"
    title = (
        f"state-space reduction of the memory kernel ({DOF_NAMES[dof]})"
        + (" [negative control: corrupted damping]" if corrupt else "")
    )
    threshold = (
        f"ROM fit error <= {target_error:.0%} at <= {max_poles} poles AND "
        f"|zeta_ss - zeta_conv|/zeta_conv <= {STATE_SPACE_BAR:.0%} AND "
        f"|T_ss - T_conv|/T_conv <= {STATE_SPACE_BAR:.0%}"
    )
    base_values: dict[str, Any] = {
        "dof": dof,
        "dof_name": DOF_NAMES[dof],
        "bar": STATE_SPACE_BAR,
        "fit_target_error": target_error,
        "max_poles": max_poles,
        "negative_control": bool(corrupt),
        "band_rad_s": [float(omega[0]), float(omega[-1])],
        "reference_route": "C5 convolution (td.cummins_free_decay on the same oscillator)",
    }

    if scale <= 0.0 or float(np.min(measured)) < 0.0:
        return {
            "id": identifier,
            "title": title,
            "threshold": threshold,
            "passed": False,
            "values": {
                **base_values,
                "status": "undecidable: no usable radiation damping in the sweep",
                "min_radiation_damping": float(np.min(measured)) if measured.size else float("nan"),
            },
        }

    closure = kk_added_mass_closure(omega, added, measured)
    added_mass_inf = float(closure.added_mass_inf)
    oscillator = oscillator_properties(dof, label=sweep.label)
    fit = _ss_mod.adaptive_order_fit(
        omega,
        distorted,
        target_error=target_error,
        max_poles=max_poles,
        min_poles=2,
    )
    realization = _ss_mod.state_space_from_fit(fit)
    # The bar is scored against the *measured* damping, so the control cannot
    # hide behind the fit being accurate for the corrupted data it was given.
    fit_error = float(np.max(np.abs(fit.B_reconstructed - measured)) / scale)
    order = int(len(fit.poles))
    poles = np.asarray(fit.poles, dtype=float)
    all_poles_stable = bool(poles.size > 0 and np.all(poles > 0.0))
    nonnegative = bool(np.all(np.asarray(fit.B_reconstructed, dtype=float) >= -1e-9 * scale))
    min_pole = float(np.min(poles)) if poles.size else float("nan")

    ss_decay = _ss_mod.cummins_state_space(
        realization,
        oscillator["inertia"],
        added_mass_inf,
        oscillator["stiffness"],
        initial_displacement=1.0,
        initial_velocity=0.0,
        dt=DT,
        duration=DURATION,
    )
    zeta_ss, period_ss, n_peaks_ss = _decay_zeta(
        np.asarray(ss_decay.time), np.asarray(ss_decay.displacement)
    )
    zeta_conv, period_conv, n_peaks_conv, _decay = _decay_zeta_for(
        sweep, dof, damping=measured, omega=omega, added_mass_inf=added_mass_inf
    )

    def _relative(a: float, b: float) -> float:
        if not (math.isfinite(a) and math.isfinite(b) and b > 0.0):
            return float("inf")
        return abs(a - b) / b

    zeta_error = _relative(zeta_ss, zeta_conv)
    period_error = _relative(period_ss, period_conv)

    if not all_poles_stable:
        status = "failed: the realization has a pole outside the stable half plane"
    elif fit_error > target_error:
        status = (
            f"undecidable: the damping spectrum is not representable with {order} "
            f"<= {max_poles} real poles to {target_error:.0%} (ROM fit error "
            f"{fit_error:.1%}); the reduction order, not the BEM data, is the limit"
        )
    elif not nonnegative:
        status = "failed: the reconstructed damping changes sign inside the band"
    else:
        status = "decided"

    passed = (
        status == "decided"
        and zeta_error <= STATE_SPACE_BAR
        and period_error <= STATE_SPACE_BAR
    )
    return {
        "id": identifier,
        "title": title,
        "threshold": threshold,
        "passed": bool(passed),
        "values": {
            **base_values,
            "status": status,
            "fit_order": order,
            "fit_error": fit_error,
            "fit_error_self": float(fit.relative_error_max),
            "min_pole": min_pole,
            "all_poles_stable": all_poles_stable,
            "damping_nonnegative_in_band": nonnegative,
            "zeta_state_space": zeta_ss,
            "zeta_convolution": zeta_conv,
            "zeta_relative_error": zeta_error,
            "period_state_space_s": period_ss,
            "period_convolution_s": period_conv,
            "period_relative_error": period_error,
            "n_peaks_state_space": int(n_peaks_ss),
            "n_peaks_convolution": int(n_peaks_conv),
            "state_space_substeps": int(ss_decay.substeps),
            "oscillator_inertia": oscillator["inertia"],
            "oscillator_stiffness": oscillator["stiffness"],
            "added_mass_inf_used": added_mass_inf,
            "added_mass_inf_source": "KK closure (least-squares offset), same as C5",
            "dt": DT,
            "duration": DURATION,
        },
    }


'''

path.write_text(text[:start] + new + text[end:])
print("replaced", len(old), "->", len(new))