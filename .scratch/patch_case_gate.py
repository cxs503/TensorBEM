from __future__ import annotations

import pathlib

PATH = pathlib.Path("src/tensorbem/benchmarks/case_bem_cummins_wigley.py")
text = PATH.read_text(encoding="utf-8")

# --------------------------------------------------------------------------
# 1. import the new budget primitive
# --------------------------------------------------------------------------
IMP_OLD = """from tensorbem.causality import (
    fit_added_mass_asymptote,
    fit_damping_power_tail,
    kk_added_mass_closure,
)"""
IMP_NEW = """from tensorbem.causality import (
    fit_added_mass_asymptote,
    fit_damping_power_tail,
    high_band_truncation_budget,
    kk_added_mass_closure,
    local_tail_exponent,
)"""
assert IMP_OLD in text
text = text.replace(IMP_OLD, IMP_NEW)

# --------------------------------------------------------------------------
# 2. tolerance for the truncation budget
# --------------------------------------------------------------------------
TOL_OLD = '''    "panel_kh": 1.0,      # k*sqrt(max panel area) at the top of the band'''
TOL_NEW = '''    "panel_kh": 1.0,      # k*sqrt(max panel area) at the top of the band
    "kk_budget": 0.05,    # half the case tolerance: above this the un-solved band
                          # can produce the residual on its own, so the sample is
                          # not evidence either way'''
assert TOL_OLD in text
text = text.replace(TOL_OLD, TOL_NEW)

# --------------------------------------------------------------------------
# 3. rewrite kk_gate
# --------------------------------------------------------------------------
START = 'def kk_gate(sweep: Sweep, *, corrupt: bool = False) -> dict[str, Any]:'
END = 'def round_trip_gate('
i, j = text.index(START), text.index(END)
assert 0 < i < j

NEW = '''def kk_gate(
    sweep: Sweep,
    *,
    corrupt: bool = False,
    window: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """C3: per-DOF Kramers-Kronig closure of the added mass (the case criterion).

    The residual of a *band-limited* closure cannot be read as a causality
    statement at every sample frequency: the un-solved band ``s > s_max``
    contributes

        ``delta A(omega) = (2/pi) integral_{s_max}^inf B(s)/(s^2-omega^2) ds``,

    which is positive for every ``omega < s_max`` and diverges as
    ``omega -> s_max``.  :func:`tensorbem.causality.high_band_truncation_budget`
    bounds that missing term from the measured local decay exponent of the
    tabulated damping.  A sample therefore counts towards the gate only when the
    bound is below ``TOLERANCES["kk_budget"]`` (half the case tolerance); the
    remaining samples are reported as undecidable instead of as a violation.

    ``window`` freezes the decidable mask, so the corrupted negative control
    (``corrupt=True``) is judged on exactly the samples where the clean sweep was
    judged -- otherwise a corruption could shrink its own window and escape.
    """
    budget_tolerance = float(TOLERANCES["kk_budget"])
    rows = []
    worst = 0.0
    worst_dof = ""
    decidable_all: list[bool] = []
    for index, name in enumerate(DOF_NAMES):
        mass = sweep.added_mass[:, index, index]
        damper = sweep.damping[:, index, index]
        tail = fit_damping_power_tail(sweep.omega, damper)
        damper_used = damper
        if corrupt:
            # Pre-declared negative control: break the *shape* of the damping
            # band.  A constant rescale would be invisible to a closure that
            # normalises by the same scale, so the corruption has to be in shape.
            damper_used = damper * np.where(sweep.omega > 1.0, 1.3, 1.0)
        closure = kk_added_mass_closure(
            sweep.omega, mass, damper_used, damping_tail=tail
        )
        asymptote = fit_added_mass_asymptote(sweep.omega, mass)
        errors = np.abs(closure.relative_residual)
        index_worst = int(np.argmax(errors)) if errors.size else 0
        max_error = float(closure.max_relative_residual)

        # Truncation budget of the *clean* data: the corruption is a statement
        # about the damping shape, not about how much band is missing, and
        # letting it move the window would make the control unfalsifiable.
        budget = high_band_truncation_budget(
            sweep.omega, damper, closure.sample_omegas
        )
        budget_relative = budget.relative(closure.added_mass_scale)
        decidable = np.isfinite(budget_relative) & (budget_relative <= budget_tolerance)
        if window is not None:
            frozen = np.asarray(list(window), dtype=bool).reshape(-1)
            if frozen.size != decidable.size:
                raise ValueError(
                    f"frozen window has {frozen.size} samples, this closure has "
                    f"{decidable.size}"
                )
            decidable = frozen
        decidable_all.append(bool(np.all(decidable)))
        decided_errors = errors[decidable]
        max_decided = (
            float(np.max(decided_errors)) if decided_errors.size else float("inf")
        )
        exponent, _ = local_tail_exponent(sweep.omega, damper)
        rows.append(
            {
                "dof": int(index),
                "dof_name": name,
                "max_relative_residual": max_error,
                "max_relative_residual_decidable": max_decided,
                "n_decidable": int(np.count_nonzero(decidable)),
                "n_samples": int(decidable.size),
                "measured_tail_exponent": float(exponent),
                "truncation_budget_admissible": bool(budget.admissible),
                "truncation_budget_reason": budget.reason,
                "truncation_budget_max_relative": (
                    float(np.nanmax(budget_relative[np.isfinite(budget_relative)]))
                    if np.any(np.isfinite(budget_relative))
                    else float("inf")
                ),
                "truncation_budget_sample_relative": [
                    float("inf") if not np.isfinite(v) else float(v)
                    for v in budget_relative
                ],
                "decidable_mask": [bool(v) for v in decidable],
                "residual_scale": float(closure.added_mass_scale),
                "worst_sample_omega": float(closure.sample_omegas[index_worst]),
                "worst_sample_reference": float(closure.measured_added_mass[index_worst]),
                "worst_sample_reconstructed": float(
                    closure.predicted_added_mass[index_worst]
                ),
                "worst_sample_absolute_error": float(
                    np.abs(closure.residual)[index_worst]
                ),
                "added_mass_inf_closure": float(closure.added_mass_inf),
                "added_mass_inf_closure_source": closure.added_mass_inf_source,
                "added_mass_inf_tail_fit": float(asymptote.added_mass_inf),
                "added_mass_inf_tail_fit_residual": float(asymptote.relative_residual),
                "added_mass_inf_cross_check": float(
                    abs(closure.added_mass_inf - asymptote.added_mass_inf)
                    / max(float(closure.added_mass_scale), 1e-300)
                ),
                "damping_tail_valid": bool(tail.valid),
                "damping_tail_exponent": float(tail.exponent),
                "damping_tail_coefficient": float(tail.coefficient),
                "damping_tail_reason": tail.reason,
                "sample_omega": closure.sample_omegas.tolist(),
                "reference_added_mass": closure.measured_added_mass.tolist(),
                "reconstructed_added_mass": closure.predicted_added_mass.tolist(),
                "relative_residual": closure.relative_residual.tolist(),
            }
        )
        if max_decided > worst:
            worst = max_decided
            worst_dof = name
    tolerance = TOLERANCES["kk"] if not corrupt else TOLERANCES["negative_control"]
    # ``raw`` keeps the pre-existing all-sample statistic (comparable with the
    # artifacts published before this rule existed); the gate itself is decided
    # on the samples where the un-solved band cannot explain the residual.
    raw_worst = float(max(row["max_relative_residual"] for row in rows))
    raw_passed = raw_worst <= tolerance
    any_decidable = any(row["n_decidable"] > 0 for row in rows)
    decided_passed = bool(worst <= tolerance and any_decidable)
    if not any_decidable:
        status = (
            "undecidable: no sample has a truncation budget below "
            f"{budget_tolerance:.0%} of the added-mass scale -- the measured "
            "damping does not decay fast enough at the top of the band for the "
            "omitted integral to be bounded"
        )
    elif decided_passed:
        status = "decided"
    else:
        status = "fail: residual exceeds the tolerance where truncation is bounded"
    return {
        "id": "C3" if not corrupt else "C6b",
        "title": (
            "Kramers-Kronig closure of the added mass (6 DOF)"
            if not corrupt
            else "negative control: corrupted damping shape must fail the KK gate"
        ),
        "threshold": (
            f"max over decidable samples and DOF |A_BEM-A_KK|/max|A_BEM| <= "
            f"{tolerance:.0%} (a sample is decidable when the closed-form bound on "
            f"the omitted s > s_max integral is <= {budget_tolerance:.0%})"
        ),
        "passed": bool(decided_passed) if not corrupt else bool(not decided_passed),
        "raw_gate_passed": bool(raw_passed),
        "status": status,
        "values": {
            "mesh": sweep.label,
            "band": [float(sweep.omega[0]), float(sweep.omega[-1])],
            "worst_dof": worst_dof,
            "worst_relative_residual": worst,
            "worst_relative_residual_all_samples": raw_worst,
            "all_dof_fully_decidable": bool(all(decidable_all)),
            "kk_budget_tolerance": budget_tolerance,
            "per_dof": rows,
            "tail_fit_model": "B(s) = C s^-p on the upper quartile of the sweep",
            "truncation_budget_model": (
                "B continued above s_max as C s^-p with p from the last three "
                "tabulated samples, anchored at the last sample; the omitted "
                "PV integral is then closed in form"
            ),
            "added_mass_inf_estimator": "least-squares closure (one free parameter)",
            "added_mass_inf_cross_check_worst": float(
                max(row["added_mass_inf_cross_check"] for row in rows)
            ),
            "added_mass_inf_cross_check_note": (
                "C3 lets one free parameter (A_inf) absorb a constant offset, so a "
                "closure that only shifts A would still pass. The independent "
                "A_inf + c/omega^2 tail fit is therefore reported next to it: a "
                "closure A_inf far from the tail fit means the residual is being "
                "absorbed, not explained."
            ),
        },
    }


'''

text = text[:i] + NEW + text[j:]

# --------------------------------------------------------------------------
# 4. freeze the decidable window for the negative control
# --------------------------------------------------------------------------
CB_OLD = '        gates["C6b"] = kk_gate(sweep, corrupt=True)'
CB_NEW = (
    '        gates["C6b"] = kk_gate(\n'
    '            sweep,\n'
    '            corrupt=True,\n'
    '            window=[\n'
    '                all(row["decidable_mask"][k] for row in gates["C3"]["values"]["per_dof"])\n'
    '                for k in range(len(gates["C3"]["values"]["per_dof"][0]["decidable_mask"]))\n'
    '            ],\n'
    '        )'
)
assert CB_OLD in text
text = text.replace(CB_OLD, CB_NEW, 1)
CB2_OLD = '        results["gates"]["C6b"] = kk_gate(production, corrupt=True)'
CB2_NEW = (
    '        results["gates"]["C6b"] = kk_gate(\n'
    '            production,\n'
    '            corrupt=True,\n'
    '            window=[\n'
    '                all(\n'
    '                    row["decidable_mask"][k]\n'
    '                    for row in results["gates"]["C3"]["values"]["per_dof"]\n'
    '                )\n'
    '                for k in range(\n'
    '                    len(results["gates"]["C3"]["values"]["per_dof"][0]["decidable_mask"])\n'
    '                )\n'
    '            ],\n'
    '        )'
)
assert CB2_OLD in text
text = text.replace(CB2_OLD, CB2_NEW, 1)

PATH.write_text(text, encoding="utf-8")
print("patched", PATH)