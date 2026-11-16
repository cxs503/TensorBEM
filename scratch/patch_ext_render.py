"""Insert the extended-band section into render_markdown (one-shot patch)."""
import pathlib

target = pathlib.Path("src/tensorbem/benchmarks/case_bem_cummins_wigley.py")
text = target.read_text(encoding="utf-8")

anchor = "    lines.extend(_verdict_and_reproduction_lines(verdict, results))\n    return \"\\n\".join(lines) + \"\\n\"\n"
assert text.count(anchor) == 1, text.count(anchor)

helper = '''

def _extended_band_lines(gates: dict[str, Any]) -> list[str]:
    """Per-DOF render of the extended-band gates (empty when they were not run).

    The ``## 2 Gates`` table only carries the verdict of ``C8``/``C9``; the
    numbers that decide them (slope at the old and the new band top, the
    truncation budget that follows, and how many samples became certifiable) are
    per DOF, so they get their own section.  Rendering them here instead of in a
    hand-kept document is what keeps `CASE.md` a render of `case_result.json`.
    """
    extended = gates.get("C9")
    consistency = gates.get("C8")
    anchored = gates.get("C3a_extended")
    if extended is None or consistency is None:
        return []
    lines: list[str] = []
    lines.append("## 4b. Extended band 0.2 .. 20 rad/s (C8 / C9 / C3a_extended)")
    lines.append("")
    committed = consistency["values"]["committed_mesh"]
    widened = consistency["values"]["extended_mesh"]
    lines.append(
        f"Committed mesh {committed['panels']} panels over "
        f"{committed['band'][0]}..{committed['band'][1]} rad/s vs extended mesh "
        f"{widened['panels']} panels over {widened['band'][0]}..{widened['band'][1]} "
        f"rad/s, {consistency['values']['n_shared_frequencies']} shared frequencies."
    )
    lines.append("")
    lines.append(
        "**C8** worst shared-band difference "
        f"{100.0*consistency['values']['worst_relative_difference']:.2f}% "
        f"({consistency['values']['worst_dof']}, bar "
        f"{100.0*consistency['values']['tolerance']:.0f}%) -> "
        f"{'PASS' if consistency['passed'] else 'FAIL'}."
    )
    lines.append("")
    lines.append("| DOF | p at 12 rad/s | p at 20 rad/s | budget admissible at 12 | at 20 | budget @20 | decideable samples 12 -> 20 | worst anchored residual @20 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in extended["values"]["per_dof"]:
        budget = row["budget_max_relative_at_20"]
        budget_text = "inf" if budget != budget or budget in (float("inf"),) else f"{100.0*budget:.2f}%"
        lines.append(
            f"| {row['dof_name']} | {row['exponent_at_12']:.2f} | "
            f"{row['exponent_at_20']:.2f} | "
            f"{'yes' if row['budget_admissible_at_12'] else 'no'} | "
            f"{'yes' if row['budget_admissible_at_20'] else 'no'} | "
            f"{budget_text} | {row['n_decidable_at_12']} -> {row['n_decidable_at_20']} | "
            f"{100.0*row['worst_anchored_residual']:.2f}% |"
        )
    lines.append("")
    lines.append(
        f"**C9** {extended['status']} -> "
        f"{'PASS' if extended['passed'] else 'FAIL'}; certifiable DOF "
        f"{extended['values']['decidable_dof_at_12']} -> "
        f"{extended['values']['decidable_dof_at_20']} of 6."
    )
    if anchored is not None:
        lines.append("")
        lines.append(
            "**C3a_extended** (zero free parameters) worst DOF "
            f"**{anchored['values']['worst_dof']}** at "
            f"{100.0*anchored['values']['worst_relative_residual']:.2f}% "
            f"(criterion {100.0*TOLERANCES['kk']:.0f}%), anchor = "
            f"{extended['values']['anchor_source']}."
        )
    lines.append("")
    return lines
'''

insertion = (
    "    lines.extend(_extended_band_lines(gates))\n"
    + anchor
)
text = text.replace(anchor, insertion)
# append helper just before render_markdown's definition
marker = "def render_markdown(results: dict[str, Any]) -> str:"
assert text.count(marker) == 1
text = text.replace(marker, helper.strip("\n") + "\n\n\n" + marker)
target.write_text(text, encoding="utf-8")
print("patched")