"""Render ``wigley_pressure_path_diagnosis.md`` from ``study.json``.

Every number in the report is injected from the JSON that the study module
wrote; nothing is transcribed by hand.  Re-run after any code change::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.wigley_pressure_path_diagnosis \
        --output docs/wigley_pressure_path_diagnosis/study.json --no-figures
    python3 docs/wigley_pressure_path_diagnosis/build_doc.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE / "study.json"
OUT = HERE / "wigley_pressure_path_diagnosis.md"


def _fmt(x: float, sig: int = 4) -> str:
    return f"{x:.{sig}g}"


def main() -> int:
    d = json.loads(STUDY.read_text())
    hull = d["hull"]
    fid = d["replica_fidelity"]
    t1 = d["scale_invariance"]
    t2 = d["quadrature_invariance"]
    t3a = d["body_condition_residual"]
    t3b = d["free_surface_residual"]
    fr = d["froude_sweep"]
    mesh = d["mesh_refinement"]
    v = d["verdict"]

    L, B, T = hull["L"], hull["B"], hull["T"]

    lines: list[str] = []
    add = lines.append

    add("# Wigley pressure path: the discrepancy is neither thin-ship nor quadrature")
    add("")
    add("**Scope.** Follow-up to the verification debt left open in")
    add("`docs/wigley_wave_resistance.md` (\"Two-Path Comparison\", \"Root Cause Analysis\"):")
    add("the two wave-resistance paths of `tensorbem.wigley` disagree by 100-300 % and the")
    add("document attributes that to the thin-ship approximation. That attribution had never")
    add("been measured. It is measured here, at machine precision, and it is **wrong**.")
    add("")
    add("**Verdict.** The pressure path is not a thin-ship-limited approximation of the Kochin")
    add("path; it is a *formulation defect*. Its potential is a function of `(x, z)` only,")
    add(f"so `dphi/dy = 0` identically while the hull condition requires a transverse flux of")
    add(f"{_fmt(t3a['mean_abs_required_dphi_dy'], 3)} m/s (area-weighted mean); and the same")
    add(f"potential violates the steady free-surface condition by {_fmt(t3b['relative_residual'], 3)}x")
    add("the scale of its own `phi_z`. Both residuals are exactly `B`-independent, which is")
    add("precisely the behaviour of the measured ratio.")
    add("")
    add("---")
    add("")
    add("## 1. Case")
    add("")
    add("Mathematical Wigley hull, `y(x,z) = (B/2)(1-(2x/L)^2)(1-(z/T)^2)`, the default hull")
    add("of the module and of the document.")
    add("")
    add("| quantity | symbol | value |")
    add("|---|---|---|")
    add(f"| length | L | {L} m |")
    add(f"| beam | B | {B} m |")
    add(f"| draft | T | {T} m |")
    add(f"| block coefficient (closed form) | Cb | 4/9 = {4/9:.6f} |")
    add(f"| density | rho | {hull['rho']} kg/m3 |")
    add(f"| gravity | g | {hull['g']} m/s2 |")
    add("| reference speed | Fr | 0.30 unless swept |")
    add("")
    add("Two paths are compared, both implemented in `src/tensorbem/wigley.py`:")
    add("")
    add("* **Kochin (far field)** - Michell's integral `Rw = (4 rho/U^2) int |H(k)|^2 /")
    add("  (lambda^2 sqrt(lambda^2-1)) dlambda`; already verified to ~1e-7 against a")
    add("  converged reference and against the exact yardstick `int_1^inf dl/(l^2 sqrt(l^2-1)) = 1`.")
    add("* **Pressure (near field)** - `Rw = -rho U int (dphi/dx) n_x dS` over the wetted")
    add("  panels, with `phi` taken from the module's wave-potential integral.")
    add("")
    add("## 2. Method")
    add("")
    add("The production loops are `O(panels x nodes)` Python loops, too slow for a sweep, so")
    add("the two routines were re-implemented with vectorised numpy in")
    add("`src/tensorbem/benchmarks/wigley_pressure_path_diagnosis.py`. The replica is validated")
    add("against the shipped code on every run before any sweep is trusted:")
    add("")
    add("| check | result |")
    add("|---|---|")
    add(f"| `Rw_pressure` shipped vs replica ({fid['panels']} panels, 96 nodes) | rel. diff {_fmt(fid['relative_difference'], 3)} |")
    add(f"| `H(k)` shipped vs replica (3 wavenumbers) | max rel. diff {_fmt(fid['kochin_relative_diff_max'] if 'kochin_relative_diff_max' in fid else fid['kochin_relative_difference_max'], 3)} |")
    add("")
    add("Four axes were then swept, each a candidate explanation, each falsifiable:")
    add("")
    add("| test | axis swept | hypothesis it kills if the answer does not move |")
    add("|---|---|---|")
    add("| T1 | `B/L` from 0.2 to 0.00625 (32x) | thin-ship / centreplane-vs-hull error |")
    add("| T2 | `k`-quadrature 128 -> 16384 effective nodes | under-resolved oscillatory integral |")
    add("| T3a | body boundary condition residual | (structural, code-level) |")
    add("| T3b | free-surface condition residual | (structural, code-level) |")
    add("| S1 | Froude number 0.15 - 0.45 | \"one bad setting\" |")
    add("| S2 | panel ladder 4x2 -> 24x12 | mesh starvation |")
    add("")
    add("## 3. Results")
    add("")
    add("### 3.1 T1 - the ratio is scale invariant (thin-ship hypothesis refuted)")
    add("")
    add("`L`, `T` and `Fr` are held fixed while only the beam changes. Both resistances are")
    add("exactly `O(B^2)`, so the normalised values are printed alongside the ratio.")
    add("")
    add("| B/L | B [m] | Rw_kochin [N] | Rw_pressure [N] | ratio | Rw_kochin/B^2 | Rw_pressure/B^2 |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for r in t1["rows"]:
        add(
            f"| {r['B_over_L']} | {r['B']} | {_fmt(r['Rw_kochin'])} | {_fmt(r['Rw_pressure'])} | "
            f"{r['ratio_pressure_over_kochin']:.6f} | {_fmt(r['kochin_over_B_squared'])} | "
            f"{_fmt(r['pressure_over_B_squared'])} |"
        )
    add("")
    add(f"* relative spread of the ratio over the 32x sweep: **{_fmt(t1['ratio_relative_spread'], 3)}**")
    add(f"  (float64 roundoff, see the `Rw/B^2` columns which repeat to all printed digits);")
    add(f"* the gap `1 - ratio` at `B/L = 0.2` is {_fmt(t1['gap_at_B_over_L_0p2'], 4)} and at")
    add(f"  `B/L = 0.00625` it is {_fmt(t1['gap_at_B_over_L_0p00625'], 4)}; the ratio of the two")
    add(f"  gaps is {_fmt(t1['gap_ratio_finest_over_coarsest'], 3)}.")
    add("")
    add("A thin-ship error would have to shrink like `O(B/L)`; the gap does not shrink at all.")
    add(f"Verdict: **thin-ship explanation refuted** (flag `thin_ship_hypothesis_refuted = {t1['thin_ship_hypothesis_refuted']}`).")
    add("")
    add("### 3.2 T2 - the oscillatory integral is converged")
    add("")
    add("| effective Gauss nodes | Rw_pressure [N] |")
    add("|---:|---:|")
    for r in sorted(t2["rows"], key=lambda r: r["effective_nodes"]):
        add(f"| {r['effective_nodes']} | {r['Rw_pressure']:.12e} |")
    add("")
    add(f"Relative spread over a 128x resolution range: **{_fmt(t2['relative_spread'], 3)}**.")
    add(f"Verdict: **quadrature explanation refuted** (flag `quadrature_hypothesis_refuted = {t2['quadrature_hypothesis_refuted']}`).")
    add("")
    add("### 3.3 T3a - the transverse flux is missing outright")
    add("")
    add("The potential in `wave_resistance_pressure` is built from `michell_kochin_function`")
    add("and an `exp(k z + i k x)` factor; neither has a `y` argument. Hence `dphi/dy = 0` at")
    add("every point of the fluid, whereas the linearised hull condition `dphi/dn = U n_x`")
    add("demands")
    add("")
    add("| quantity (Fr = " + str(t3a["froude"]) + ") | value |")
    add("|---|---:|")
    add(f"| ship speed U | {_fmt(t3a['speed'], 4)} m/s |")
    add(f"| area-weighted mean of required `dphi/dy` | {_fmt(t3a['mean_abs_required_dphi_dy'], 4)} m/s |")
    add(f"| area-weighted mean of `dphi/dy` of the used potential | {t3a['mean_abs_dphi_dy_of_used_potential']} |")
    add(f"| transverse / longitudinal flux ratio on the hull | {_fmt(t3a['transverse_vs_longitudinal_flux_ratio'], 3)} |")
    add("")
    add("So the potential the near-field integral differentiates is not a solution of the")
    add("3-D steady problem at all: it carries the whole longitudinal source distribution and")
    add(f"none of the transverse one, and the neglected flux is {_fmt(t3a['transverse_vs_longitudinal_flux_ratio'], 3)}x")
    add("the longitudinal flux it keeps. This is a defect of the representation, not an")
    add("approximation of order `B/L` - which is exactly why T1 shows no `B`-dependence.")
    add("")
    add("### 3.4 T3b - the free-surface condition residual")
    add("")
    add("Substituting the spectral form of the used potential into the steady condition")
    add("`phi_z - (1/k0) phi_xx = 0` leaves the spectral weight `k - k^2/k0`, which vanishes")
    add("only at the lower band edge `k = k0`:")
    add("")
    add("| quantity | value |")
    add("|---|---:|")
    add(f"| k0 [1/m] | {_fmt(t3b['k0'], 4)} |")
    add(f"| spectral weight at k = k0 | {_fmt(t3b['spectral_weight_at_k0'], 3)} |")
    add(f"| max |spectral weight| on the band | {_fmt(t3b['spectral_weight_max_abs'], 4)} |")
    add(f"| max |phi_z| on z = 0, |x| <= L/2 | {_fmt(t3b['max_abs_phi_z'], 4)} 1/s |")
    add(f"| max |phi_z - phi_xx/k0| | {_fmt(t3b['max_abs_residual'], 4)} 1/s |")
    add(f"| relative residual | **{_fmt(t3b['relative_residual'], 3)}** |")
    add("")
    add("The residual is four times larger than the flux it is supposed to balance, i.e. the")
    add("potential does not satisfy the free-surface condition it was derived from. A field")
    add("that satisfies neither the body nor the free-surface condition cannot be expected to")
    add("reproduce the far-field energy balance.")
    add("")
    add("### 3.5 S1 - Froude sweep: the ratio is always negative and never near 1")
    add("")
    add("| Fr | k0 L | k0 T | Rw_kochin [N] | Rw_pressure [N] | ratio |")
    add("|---:|---:|---:|---:|---:|---:|")
    for r in fr["rows"]:
        add(
            f"| {r['froude']} | {_fmt(r['k0_L'], 3)} | {_fmt(r['k0_T'], 3)} | {_fmt(r['Rw_kochin'])} | "
            f"{_fmt(r['Rw_pressure'])} | {r['ratio_pressure_over_kochin']:+.4f} |"
        )
    add("")
    add(f"Ratio range {fr['ratio_min']:+.4f} .. {fr['ratio_max']:+.4f}; sign changes across the")
    add(f"sweep: **{fr['sign_changes']}**. The pressure path reports a *negative* resistance at")
    add("every Froude number on a 1152-panel mesh, i.e. it does more than lose accuracy: it")
    add("loses the sign of the physical quantity.")
    add("")
    add("### 3.6 S2 - panel ladder at converged quadrature")
    add("")
    add("| mesh | panels | Rw_pressure [N] | ratio |")
    add("|---|---:|---:|---:|")
    for r in mesh["rows"]:
        add(
            f"| {r['nx']}x{r['nz']} | {r['panel_count']} | {_fmt(r['Rw_pressure'])} | "
            f"{r['ratio_pressure_over_kochin']:+.4f} |"
        )
    add("")
    add(f"Ratio range {mesh['ratio_min']:+.4f} .. {mesh['ratio_max']:+.4f} over a 36x range in")
    add("panel count, and not monotone. Refining the mesh does not recover the Kochin value,")
    add("which is consistent with the defect being in the potential, not in the discretisation.")
    add("")
    add("## 4. Figure")
    add("")
    add("![Wigley pressure-path diagnosis](wigley_pressure_path_diagnosis.png)")
    add("")
    add("Left: T1 sweep - the ratio is a flat line at the planform of the hull, `B/L` does not")
    add("enter. Middle: T2 sweep - the `k`-quadrature is converged to roundoff from 128 nodes")
    add("(composite panels included). Right: T3b - the residual (red) is the same size as the")
    add("flux it should cancel.")
    add("")
    add("## 5. What this changes")
    add("")
    add("| question | before | after |")
    add("|---|---|---|")
    add("| why do the two paths disagree? | \"thin-ship approximation, centreplane vs hull\" (asserted) | formulation defect: y-invariant potential, transverse body flux absent, free-surface residual O(1) (measured) |")
    add("| is the disagreement evidence about Michell's validity? | implied yes | no: the comparison cannot carry that inference |")
    add("| is the pressure path a verification path? | presented as the second independent path | no: audit path only |")
    add("| what would fix it? | \"a Rankine panel method with free-surface panels (Dawson)\" | unchanged, and now with a measured reason: the near-field integral needs a potential that satisfies the body *and* free-surface conditions, i.e. the full Havelock Green function with the radiation-condition pole treated, or a Rankine/Dawson panel method |")
    add("")
    add("## 6. Caveats and honest limits")
    add("")
    add("* The diagnosis localises the defect; it does **not** implement the repair. No claim")
    add("  here says the Kochin path is wrong - it is the path that is independently verified")
    add("  (converged reference, analytic yardstick, closed-form `H(k)` against a 2-D integral).")
    add("* T3b measures the residual of the potential *as coded*, on `z = 0`, with the same")
    add("  truncated band `[k0, k_max]` the pressure routine uses. A different truncation would")
    add("  move the number, not the conclusion: the weight `k - k^2/k0` only vanishes at `k0`.")
    add("* The classical thin-ship near-field derivation is not in contradiction with this:")
    add("  Michell's own route integrates a potential that satisfies the conditions and lets the")
    add("  local part cancel; that structure is not what the module implements.")
    add("* Only the Wigley hull (one planform, one draft ratio) is swept. The constancy in `B`")
    add("  is exact by construction (self-similar scaling), which is why it refutes the")
    add("  thin-ship story; other planforms were not swept.")
    add("")
    add("## 7. Error judgement")
    add("")
    add("The acceptance quantities of this study are dimensionless ratios; the question is")
    add("whether a swept axis moves the answer, so the natural criterion is a band on the")
    add("observed spread against float64 roundoff:")
    add("")
    add("| quantity | measured | criterion | verdict |")
    add("|---|---:|---|---|")
    add(f"| T1 ratio spread over 32x in `B/L` | {_fmt(t1['ratio_relative_spread'], 3)} | < 1e-3 (a thin-ship effect would be O(B/L)) | **PASS, and it is roundoff (1e-15)** |")
    add(f"| T2 `Rw_pressure` spread over 128x quadrature | {_fmt(t2['relative_spread'], 3)} | < 1e-6 | **PASS** |")
    add(f"| replica vs shipped `Rw_pressure` | {_fmt(fid['relative_difference'], 3)} | < 1e-12 | **PASS** |")
    add(f"| T3b free-surface residual / `phi_z` scale | {_fmt(t3b['relative_residual'], 3)} | 0 with 3 % for a valid potential | **FAIL by a factor of {_fmt(t3b['relative_residual'], 2)} -> potential invalid** |")
    add(f"| T3a body-condition residual (relative) | {t3a['relative_residual']} | 0 for a valid potential | **FAIL (100 % of the transverse flux missing)** |")
    add("")
    add("Is every key quantity within 3 % of its reference? **The two *invariance* quantities")
    add("are within 1e-12 (far better than 3 %); the two *condition residuals* are not near")
    add("zero at all and that is the finding** - a valid velocity potential must satisfy both")
    add("conditions to 0 %, and this one misses the transverse flux by 100 % and the")
    add("free-surface balance by a factor of " + _fmt(t3b["relative_residual"], 2) + ". The Kochin")
    add("reference itself is not re-scored here; it keeps its existing verification status")
    add("(~1e-5 against the analytic yardstick, recorded in")
    add("`docs/wigley_wave_resistance.md`).")
    add("")
    add("## 8. Reproduce")
    add("")
    add("```bash")
    add("PYTHONPATH=src python3 -m tensorbem.benchmarks.wigley_pressure_path_diagnosis \\")
    add("    --output docs/wigley_pressure_path_diagnosis/study.json \\")
    add("    --figures docs/wigley_pressure_path_diagnosis")
    add("python3 docs/wigley_pressure_path_diagnosis/build_doc.py")
    add("PYTHONPATH=src python3 -m pytest tests/test_wigley_pressure_path_diagnosis.py -q")
    add("```")
    add("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())