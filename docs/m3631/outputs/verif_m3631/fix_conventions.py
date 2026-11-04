"""Fix two tolerance/convention issues found by the first standard-tier run.

1. The B11' observed orders printed in table 10 (1.62 / 1.88) follow the
   m2609 panel-count convention (h ~ N**-1/2), not the h_mean convention of
   the other four order columns.  The runner now also emits the sqrt-N
   convention, and the registry pins the documented numbers under it.
2. The rotational coupling ratios are printed with two significant digits
   (2.2e-4, 2.6e-5); the default 1.5% tolerance is tighter than the printed
   precision, so those two entries get a 2.5% tolerance.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
p = REPO_ROOT / "src" / "tensorbem" / "benchmarks" / "declared_numbers.py"
text = p.read_text()

# --- 1. runner: emit the sqrt-N convention b11 order ----------------------
old = """    if group == "hulme_hemisphere":
        study = floating_hemisphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(s) for s in parameters["subdivisions"]),
            ka=float(parameters.get("ka", 1.0)),
        )
        return _flatten_levels(study["levels"])"""
new = """    if group == "hulme_hemisphere":
        study = floating_hemisphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(s) for s in parameters["subdivisions"]),
            ka=float(parameters.get("ka", 1.0)),
        )
        values = _flatten_levels(study["levels"])
        # The B11' order column of table 10 follows the m2609 panel-count
        # convention (h ~ N**-1/2), not the h_mean convention of the other
        # order columns; the document says so explicitly.
        levels = study["levels"]
        for i in range(1, len(levels)):
            lo, hi = levels[i - 1], levels[i]
            e_lo = float(lo["surge_damping_rel_error"])
            e_hi = float(hi["surge_damping_rel_error"])
            if e_lo > 0.0 and e_hi > 0.0:
                values[
                    f"level{int(hi['subdivisions'])}_observed_order_b11_sqrtN"
                ] = math.log2(e_lo / e_hi) / math.log2(
                    math.sqrt(hi["panel_count"] / lo["panel_count"])
                )
        return values"""
assert text.count(old) == 1
text = text.replace(old, new)

# --- 2. b11 order entries: use the sqrt-N quantity --------------------------
old = """        for name, order in zip(("a33", "b33", "a11", "b11", "fz"), orders):
            add(f"hulme_order_{name}_{panels}",
                f"observed order of the {name.upper()} error at {panels} panels",
                f"§4.10 table 10 row N={panels}: observed order {name.upper()}",
                "hulme_hemisphere", f"level{sub}_observed_order_{name}",
                order, _HULME_TABLE10_PARAMS, 5.0e-3)"""
new = """        for name, order in zip(("a33", "b33", "a11", "b11", "fz"), orders):
            quantity = (
                f"level{sub}_observed_order_b11_sqrtN"
                if name == "b11"
                else f"level{sub}_observed_order_{name}"
            )
            add(f"hulme_order_{name}_{panels}",
                f"observed order of the {name.upper()} error at {panels} panels"
                + (" (panel-count convention, h ~ N**-1/2)" if name == "b11" else ""),
                f"§4.10 table 10 row N={panels}: observed order {name.upper()}",
                "hulme_hemisphere", quantity,
                order, _HULME_TABLE10_PARAMS, 5.0e-3)"""
assert text.count(old) == 1
text = text.replace(old, new)

# --- 3. coupling-ratio tolerance: printed with two significant digits ------
old = """        add(f"sphere_rot_coupling_{panels}",
            f"max rotational added-mass coupling ratio at {panels} panels",
            f"§4.11 {row}: rotational coupling ratio", "submerged_sphere",
            f"level{sub}_rotational_added_mass_max_ratio", rot,
            _SPHERE_TABLE12_PARAMS)"""
new = """        add(f"sphere_rot_coupling_{panels}",
            f"max rotational added-mass coupling ratio at {panels} panels",
            f"§4.11 {row}: rotational coupling ratio", "submerged_sphere",
            f"level{sub}_rotational_added_mass_max_ratio", rot,
            _SPHERE_TABLE12_PARAMS, 2.5e-2)"""
assert text.count(old) == 1
text = text.replace(old, new)

p.write_text(text)
import ast
ast.parse(text)
print("fixes applied, AST valid")
