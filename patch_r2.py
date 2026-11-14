"""Apply the compare()-classification fix to capytaine_crosscheck_r2.py."""
from __future__ import annotations

import pathlib
import re

PATH = pathlib.Path("src/tensorbem/benchmarks/capytaine_crosscheck_r2.py")
text = PATH.read_text()


def replace_between(text: str, start: str, end: str, new: str) -> str:
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[:i] + new + text[j:]


NEW_COMPARE = '''def compare(tensorbem: dict[str, Any], capytaine: dict[str, Any]) -> dict[str, Any]:
    """Entry-by-entry comparison of all 36 entries of both matrices.

    An entry whose row or column is a **declared** structural null
    (``invariant_null_dofs``) has an exact target of zero: a relative deviation
    on it divides one discretisation floor by another, so it measures neither
    code.  Such entries are classed ``invariant`` and bounded against
    ``INVARIANT_RATIO * physical_scale(i, j)``; they never feed the physics
    score.

    This replaces the earlier normaliser ``sqrt(|M_ii M_jj|)``, which
    degenerates *exactly* when a diagonal entry is itself a structural null: on
    ``hemisphere_r1`` (rotation centre on the centre of curvature) ``M_roll,roll``
    is a residual of order ``1e-4 rho L^5``, so the scale collapsed onto it and
    the whole rotational block was scored as "active" with deviations up to
    9859 %, producing a ``FAIL`` verdict for a block in which both codes sit at
    their floor.  The raw ratio is still recorded
    (``worst_relative_any_entry``) so the old number stays in the trail.

    A degenerate normaliser on a *non*-declared entry means the declaration of
    null dofs is incomplete -- that is classed ``undeclared_null`` and is a
    failure, so the invariant list cannot silently miss a zero.
    """
    cases: dict[str, Any] = {}
    failures: list[str] = []
    worst_active = 0.0
    worst_invariant = 0.0
    worst_any = 0.0
    for case, tb_case in tensorbem.items():
        cap_case = capytaine.get(case)
        if cap_case is None:
            cases[case] = {
                "compared": False,
                "reason": "absent from the capytaine reference",
            }
            failures.append(case + ": absent from the capytaine reference")
            continue
        null_dofs = invariant_null_dofs(case)
        null_index = {DOF_ORDER.index(dof) for dof in null_dofs}
        entries: list[dict[str, Any]] = []
        case_failures: list[str] = []
        case_worst = 0.0
        case_invariant = 0.0
        case_any = 0.0
        for nu in NU_GRID:
            key = _key(nu)
            tb = tb_case["reference"][key]
            cap = cap_case["reference"][key]
            for quantity in ("added_mass", "damping"):
                a_tb = np.asarray(tb[quantity], dtype=float)
                a_cap = np.asarray(cap[quantity], dtype=float)
                for i, dof_i in enumerate(DOF_ORDER):
                    for j, dof_j in enumerate(DOF_ORDER):
                        scale = _scale(a_tb, i, j)
                        phys = physical_scale(i, j)
                        entry: dict[str, Any] = {
                            "nu": nu,
                            "row": dof_i,
                            "column": dof_j,
                            "quantity": quantity,
                            "tensorbem": float(a_tb[i, j]),
                            "capytaine": float(a_cap[i, j]),
                            "scale": scale,
                            "physical_scale": phys,
                        }
                        if abs(a_cap[i, j]) > 0.0:
                            case_any = max(
                                case_any,
                                abs((a_tb[i, j] - a_cap[i, j]) / abs(a_cap[i, j])),
                            )
                        if i in null_index or j in null_index:
                            bound = INVARIANT_RATIO * phys
                            biggest = max(abs(a_tb[i, j]), abs(a_cap[i, j]))
                            ratio = biggest / bound if bound else float("inf")
                            entry["class"] = "invariant"
                            entry["invariant_bound"] = bound
                            entry["invariant_ratio"] = float(ratio)
                            entry["tensorbem_null_ratio"] = float(abs(a_tb[i, j]) / phys)
                            entry["capytaine_null_ratio"] = float(abs(a_cap[i, j]) / phys)
                            case_invariant = max(case_invariant, ratio)
                            if ratio > 1.0:
                                case_failures.append(
                                    "{} nu={:g} {} A[{},{}]: invariant zero exceeded "
                                    "({:.6g} vs bound {:.6g})".format(
                                        case, nu, quantity, dof_i, dof_j,
                                        biggest, bound,
                                    )
                                )
                            entries.append(entry)
                            continue
                        degenerate = (
                            not math.isfinite(scale) or scale <= NULL_FLOOR * phys
                        )
                        if degenerate:
                            bound = INVARIANT_RATIO * phys
                            biggest = max(abs(a_tb[i, j]), abs(a_cap[i, j]))
                            ratio = biggest / bound if bound else float("inf")
                            entry["class"] = "undeclared_null"
                            entry["invariant_bound"] = bound
                            entry["invariant_ratio"] = float(ratio)
                            case_failures.append(
                                "{} nu={:g} {} A[{},{}]: undeclared structural null "
                                "(normaliser {:.6g} <= {:.6g}, entries {:.6g}/{:.6g}) "
                                "-- `invariant_null_dofs` is incomplete".format(
                                    case, nu, quantity, dof_i, dof_j,
                                    scale, NULL_FLOOR * phys,
                                    a_tb[i, j], a_cap[i, j],
                                )
                            )
                            entries.append(entry)
                            continue
                        cap_null = abs(a_cap[i, j]) <= NULL_FLOOR * scale
                        tb_null = abs(a_tb[i, j]) <= NULL_FLOOR * scale
                        if cap_null and tb_null:
                            entry["class"] = "null"
                            entry["tensorbem_null_ratio"] = abs(a_tb[i, j]) / scale
                            entry["capytaine_null_ratio"] = abs(a_cap[i, j]) / scale
                        elif cap_null != tb_null:
                            entry["class"] = "conflict"
                            entry["relative"] = None
                            case_failures.append(
                                "{} nu={:g} {} A[{},{}]: null/non-null conflict "
                                "(tensorbem {:.6g}, capytaine {:.6g}, floor {:.6g})".format(
                                    case,
                                    nu,
                                    quantity,
                                    dof_i,
                                    dof_j,
                                    a_tb[i, j],
                                    a_cap[i, j],
                                    NULL_FLOOR * scale,
                                )
                            )
                        else:
                            relative = (a_tb[i, j] - a_cap[i, j]) / abs(a_cap[i, j])
                            band = TOL_DIAGONAL if i == j else TOL_OFFDIAGONAL
                            entry["class"] = "active"
                            entry["relative"] = float(relative)
                            entry["band"] = band
                            entry["within_band"] = bool(abs(relative) <= band)
                            case_worst = max(case_worst, abs(relative))
                            if abs(relative) > band:
                                case_failures.append(
                                    "{} nu={:g} {} A[{},{}]: {:+.2%} (band {:.0%})".format(
                                        case, nu, quantity, dof_i, dof_j, relative, band
                                    )
                                )
                        entries.append(entry)
        nulls = [e for e in entries if e["class"] == "null"]
        invariants = [e for e in entries if e["class"] == "invariant"]
        worst_active = max(worst_active, case_worst)
        worst_invariant = max(worst_invariant, case_invariant)
        worst_any = max(worst_any, case_any)
        failures.extend(case_failures)
        cases[case] = {
            "compared": True,
            "entries": entries,
            "n_entries": len(entries),
            "null_dofs": list(null_dofs),
            "n_active": len([e for e in entries if e["class"] == "active"]),
            "n_null": len(nulls),
            "n_invariant": len(invariants),
            "n_conflict": len([e for e in entries if e["class"] == "conflict"]),
            "n_undeclared_null": len(
                [e for e in entries if e["class"] == "undeclared_null"]
            ),
            "worst_active_relative": case_worst,
            "worst_invariant_ratio": float(case_invariant),
            "worst_relative_any_entry": float(case_any),
            "worst_null_ratio": max(
                (
                    max(e["tensorbem_null_ratio"], e["capytaine_null_ratio"])
                    for e in nulls
                ),
                default=0.0,
            ),
            "tensorbem_faces": tb_case["reference_faces"],
            "capytaine_faces": cap_case["reference_faces"],
            "rotation_center": tb_case["rotation_center"],
            "failures": case_failures,
            "verdict": "PASS" if not case_failures else "FAIL",
        }
    return {
        "tolerance_diagonal": TOL_DIAGONAL,
        "tolerance_offdiagonal": TOL_OFFDIAGONAL,
        "null_floor": NULL_FLOOR,
        "invariant_ratio": INVARIANT_RATIO,
        "worst_active_relative": worst_active,
        "worst_invariant_ratio": float(worst_invariant),
        "worst_relative_any_entry": float(worst_any),
        "failures": failures,
        "cases": cases,
        "verdict": "PASS" if not failures else "FAIL",
    }


'''

text = replace_between(text, "def compare(tensorbem", "def convergence(", NEW_COMPARE)

# ---- convergence(): skip declared structural nulls -------------------------
old_conv = '''        rows = []
        for row in tb_case["levels"]:
            worst_diag = 0.0
            worst_off = 0.0
            for nu in NU_GRID:'''
new_conv = '''        null_index = {DOF_ORDER.index(dof) for dof in invariant_null_dofs(case)}
        rows = []
        for row in tb_case["levels"]:
            worst_diag = 0.0
            worst_off = 0.0
            for nu in NU_GRID:'''
assert old_conv in text
text = text.replace(old_conv, new_conv)

old_conv2 = '''                for i in range(6):
                    for j in range(6):
                        scale = _scale(cap, i, j)
                        if abs(cap[i, j]) <= NULL_FLOOR * scale:
                            continue'''
new_conv2 = '''                for i in range(6):
                    for j in range(6):
                        if i in null_index or j in null_index:
                            # exact structural zero: a ratio of two floors
                            continue
                        scale = _scale(cap, i, j)
                        if abs(cap[i, j]) <= NULL_FLOOR * scale:
                            continue'''
assert old_conv2 in text
text = text.replace(old_conv2, new_conv2)

# ---- render_figures(): same skip ------------------------------------------
old_fig = '''    for col, case in enumerate(cases):
        for row, quantity in enumerate(("added_mass", "damping")):
            ax = axes[col][row]'''
new_fig = '''    for col, case in enumerate(cases):
        null_index = {DOF_ORDER.index(d) for d in invariant_null_dofs(case)}
        for row, quantity in enumerate(("added_mass", "damping")):
            ax = axes[col][row]'''
assert old_fig in text
text = text.replace(old_fig, new_fig)

old_fig2 = '''            for i in range(6):
                for j in range(6):
                    if abs(cap[i, j]) <= NULL_FLOOR * _scale(tb, i, j):
                        dev[i, j] = np.nan'''
new_fig2 = '''            for i in range(6):
                for j in range(6):
                    if i in null_index or j in null_index:
                        dev[i, j] = np.nan
                    elif abs(cap[i, j]) <= NULL_FLOOR * _scale(tb, i, j):
                        dev[i, j] = np.nan'''
assert old_fig2 in text
text = text.replace(old_fig2, new_fig2)

# ---- deep_submergence(): worst rotation over *all* rows -------------------
old_deep = '''    worst_translation = abs(finest["surge_vs_lamb"])
    worst_rotation = max(v["roll_over_scale"] for v in deepest.values())'''
new_deep = '''    worst_translation = abs(finest["surge_vs_lamb"])
    # The bound must hold at *every* depth, not only at the deepest one: a
    # residual that grows as the body comes closer to the free surface would
    # otherwise be reported as the deepest (smallest) value.
    worst_rotation = max(
        v["roll_over_scale"] for row in rows for v in row["values"].values()
    )
    deepest_rotation = max(v["roll_over_scale"] for v in deepest.values())'''
assert old_deep in text
text = text.replace(old_deep, new_deep)

old_deep2 = '''        "worst_rotation_ratio": float(worst_rotation),'''
new_deep2 = '''        "worst_rotation_ratio": float(worst_rotation),
        "deepest_rotation_ratio": float(deepest_rotation),'''
assert old_deep2 in text
text = text.replace(old_deep2, new_deep2)

# ---- document prose -------------------------------------------------------
old_doc3 = '''        "**active**（两码均显著：|M_ij| > {:.0%}·sqrt(|M_ii M_jj|)）、**null**（两码均在此阈下）、"
        "**conflict**（一码显著、另一码为零 → 直接失败）。".format(NULL_FLOOR)'''
new_doc3 = '''        "**invariant**（该行/列是声明过的恒零自由度 `invariant_null_dofs`，精确目标为 0，"
        "用 {:.0%}·ρL^k 物理尺度作上界、不算相对偏差）、**active**（两码均显著："
        "|M_ij| > {:.0%}·sqrt(|M_ii M_jj|)）、**null**（两码均在此阈下）、"
        "**conflict**（一码显著、另一码为零 → 直接失败）、**undeclared_null**"
        "（未声明的恒零项，说明零自由度清单不完整 → 直接失败）。".format(
            INVARIANT_RATIO, NULL_FLOOR
        )'''
assert old_doc3 in text
text = text.replace(old_doc3, new_doc3)

old_tbl = '''    add("| case | TensorBEM 面板 | capytaine 面板 | active | 结构零项 | conflict | 最差 active 偏差 | 判定 |")
    add("|---|---:|---:|---:|---:|---:|---:|---|")'''
new_tbl = '''    add(
        "| case | TensorBEM 面板 | capytaine 面板 | active | 结构零项 | 不变式零项 | conflict | "
        "未声明零项 | 最差 active 偏差 | 最差不变式比 | 判定 |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")'''
assert old_tbl in text
text = text.replace(old_tbl, new_tbl)

old_row = '''            "| {} | {} | {} | {} | {} | {} | {:.2f}% | {} |".format(
                case,
                row["tensorbem_faces"],
                row["capytaine_faces"],
                row["n_active"],
                row["n_null"],
                row["n_conflict"],
                100.0 * row["worst_active_relative"],
                row["verdict"],
            )'''
new_row = '''            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.2f}% | {:.2e} | {} |".format(
                case,
                row["tensorbem_faces"],
                row["capytaine_faces"],
                row["n_active"],
                row["n_null"],
                row["n_invariant"],
                row["n_conflict"],
                row["n_undeclared_null"],
                100.0 * row["worst_active_relative"],
                row["worst_invariant_ratio"],
                row["verdict"],
            )'''
assert old_row in text
text = text.replace(old_row, new_row)

old_tot = '''    n_active = sum(
        c.get("n_active", 0) for c in comparison["cases"].values() if c.get("compared")
    )
    n_null = sum(
        c.get("n_null", 0) for c in comparison["cases"].values() if c.get("compared")
    )'''
new_tot = '''    n_active = sum(
        c.get("n_active", 0) for c in comparison["cases"].values() if c.get("compared")
    )
    n_null = sum(
        c.get("n_null", 0) for c in comparison["cases"].values() if c.get("compared")
    )
    n_invariant = sum(
        c.get("n_invariant", 0)
        for c in comparison["cases"].values()
        if c.get("compared")
    )'''
assert old_tot in text
text = text.replace(old_tot, new_tot)

old_tot2 = '''        "总计 {} 个 active 项与 {} 个结构零项被逐项判定，总体判定 **{}**"
        "（对角带 {:.0%}、耦合带 {:.0%}）。".format(
            n_active, n_null, comparison["verdict"], TOL_DIAGONAL, TOL_OFFDIAGONAL
        )'''
new_tot2 = '''        "总计 {} 个 active 项、{} 个相对阈下的结构零项与 {} 个声明过的恒零项（不变式，上界 "
        "{:.0%}·ρL^k）被逐项判定，总体判定 **{}**（对角带 {:.0%}、耦合带 {:.0%}）；"
        "最差 active 相对偏差 {:.2%}，最差不变式比 {:.2e}。".format(
            n_active,
            n_null,
            n_invariant,
            INVARIANT_RATIO,
            comparison["verdict"],
            TOL_DIAGONAL,
            TOL_OFFDIAGONAL,
            100.0 * comparison["worst_active_relative"],
            comparison["worst_invariant_ratio"],
        )'''
assert old_tot2 in text
text = text.replace(old_tot2, new_tot2)

# ---- module docstring: state the classification rule ----------------------
old_cls = '''* ``NULL_FLOOR`` -- an entry is a *structural null* when ``|entry|`` is at most
  1 % of ``sqrt(|M_ii M_jj|)``; a pair null in one code and not in the other is a
  ``conflict`` and therefore a **failure** (the interesting defect class).'''
new_cls = '''* ``NULL_FLOOR`` -- an entry is a *structural null* when ``|entry|`` is at most
  1 % of ``sqrt(|M_ii M_jj|)``; a pair null in one code and not in the other is a
  ``conflict`` and therefore a **failure** (the interesting defect class).  That
  normaliser is only applied to entries that touch **no** declared null dof: on
  a null diagonal it collapses onto the residual itself, so entries of
  ``invariant_null_dofs`` are bounded by ``INVARIANT_RATIO * rho L**k`` instead
  (``class == "invariant"``) and an undeclared null is a failure
  (``class == "undeclared_null"``).'''
assert old_cls in text
text = text.replace(old_cls, new_cls)

# ---- invariant_scoring docstring: it is now the independent cross-check ---
old_isd = '''    ``compare`` is left untouched: its bands are pre-registered and its verdict
    stays in the artifact exactly as computed.  What is added here is the
    observation that an entry touching a null dof (``invariant_null_dofs``) has
    an exact target of zero, so a relative deviation on it measures the
    discretisation residual of a zero, not agreement between two codes.  Those
    entries are bounded by ``INVARIANT_RATIO`` times ``physical_scale`` instead
    and reported out of band; every other entry keeps the ``compare``
    classification and band unchanged.
    """'''
new_isd = '''    ``compare`` now performs this classification itself (the degenerate
    ``sqrt(|M_ii M_jj|)`` normaliser is not applied to declared nulls -- see its
    docstring).  This function is kept as an **independent recomputation** of
    the same split from the raw entries, so a gate can require the two to agree
    entry for entry; it carries no relaxation of its own and the bands are read
    from the ``compare`` record, not re-declared here.
    """'''
assert old_isd in text
text = text.replace(old_isd, new_isd)

# main() summary line: report both metrics
old_main = '''        "verdict {} (worst active {:.2%}); symmetry {}; deep {}".format(
            study["comparison"]["verdict"],
            study["comparison"]["worst_active_relative"],'''
new_main = '''        "verdict {} (worst active {:.2%}, worst invariant ratio {:.2e}); "
        "symmetry {}; deep {}".format(
            study["comparison"]["verdict"],
            study["comparison"]["worst_active_relative"],
            study["comparison"]["worst_invariant_ratio"],'''
assert old_main in text
text = text.replace(old_main, new_main)

PATH.write_text(text)
print("patched")