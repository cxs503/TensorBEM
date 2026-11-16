"""Wire the repaired C11 into the replay path and the published verdict."""
import pathlib
import re

path = pathlib.Path("src/tensorbem/benchmarks/case_bem_cummins_wigley.py")
text = path.read_text()

# 1) C11 already listed in C0_DEPENDENT_GATES

# 2) replay path: score C11 on the case's declared free-decay DOF
old_replay = '''    if include_decay_modes:
        gates["C5b"] = decay_modes_gate(sweep)'''
new_replay = '''    if include_decay_modes:
        gates["C5b"] = decay_modes_gate(sweep)'''
assert old_replay in text

anchor_line = '''    gates["C4"] = round_trip_gate(sweep, dof=int(dof_free_decay))
    gates["C5"] = free_decay_gate(sweep, dof=int(dof_free_decay))'''
assert anchor_line in text
text = text.replace(
    anchor_line,
    anchor_line + '''
    gates["C11"] = state_space_gate(sweep, dof=int(dof_free_decay))''',
    1,
)

# 3) publish the ROM audit of every restoring mode beside the gates (reported,
#    not scored: an order-limited ROM is not a physics failure)
old_return = '''    extended_info = attach_extended_band_gates(
        gates,
        sweep,
        Path(path).parent,
        extended_path=extended_path,
    )
    return {
        "case": CASE_SLUG,
        "artifact": str(path),
        "added_mass_inf_anchor": anchor,'''
new_return = '''    extended_info = attach_extended_band_gates(
        gates,
        sweep,
        Path(path).parent,
        extended_path=extended_path,
    )
    return {
        "case": CASE_SLUG,
        "artifact": str(path),
        "state_space_reduction": state_space_audit(sweep),
        "added_mass_inf_anchor": anchor,'''
assert old_return in text
text = text.replace(old_return, new_return, 1)

# 4) the audit helper
helper = '''
def state_space_audit(sweep: Sweep, **kwargs: Any) -> dict[str, Any]:
    """``C11`` on every restoring mode plus its negative control (reported only).

    The verdict tallies ``C11`` on the case's declared free-decay DOF only.  The
    other restoring modes are reported here because a ROM that cannot be built
    at the declared order is information about the reduction, not a failure of
    the BEM coefficients -- but it must not be silently dropped either.
    """
    rows = {}
    for dof in RESTORING_MODES:
        gate = state_space_gate(sweep, dof=int(dof), **kwargs)
        rows[DOF_NAMES[int(dof)]] = {
            "passed": bool(gate["passed"]),
            "status": gate["values"].get("status"),
            "fit_order": gate["values"].get("fit_order"),
            "fit_error": gate["values"].get("fit_error"),
            "zeta_state_space": gate["values"].get("zeta_state_space"),
            "zeta_convolution": gate["values"].get("zeta_convolution"),
            "zeta_relative_error": gate["values"].get("zeta_relative_error"),
            "period_relative_error": gate["values"].get("period_relative_error"),
            "bar": STATE_SPACE_BAR,
        }
    control = state_space_gate(sweep, dof=int(RESTORING_MODES[0]), corrupt=True, **kwargs)
    return {
        "scored_dof": DOF_NAMES[int(RESTORING_MODES[0])],
        "bar": STATE_SPACE_BAR,
        "per_mode": rows,
        "negative_control": {
            "dof": DOF_NAMES[int(RESTORING_MODES[0])],
            "passed": bool(control["passed"]),
            "status": control["values"].get("status"),
            "fit_error": control["values"].get("fit_error"),
        },
    }


'''
text = text.replace("def case_verdict(gates: dict[str, Any]) -> dict[str, Any]:", helper.lstrip("\n") + "def case_verdict(gates: dict[str, Any]) -> dict[str, Any]:", 1)

path.write_text(text)
print("wired")
print("C0 deps:", re.search(r"C0_DEPENDENT_GATES\s*=\s*\(([^)]*)\)", text, re.S).group(1)[:200])
print("C11 occurrences:", text.count('gates["C11"]'), text.count("def state_space_audit"))