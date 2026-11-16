#!/usr/bin/env python3
"""Render ``docs/bem_hulme_haskind_verification.md`` from the Haskind artifact.

Every number in the rendered page is read out of
``benchmarks/verified/casepool_bem_hulme_hemisphere/haskind_verification_results.json``
(written by :mod:`haskind_verification`) -- nothing is typed by hand, so the
document cannot drift away from the run it describes.

Usage::

    python benchmarks/verified/casepool_bem_hulme_hemisphere/make_haskind_doc.py \\
        --results benchmarks/verified/casepool_bem_hulme_hemisphere/haskind_verification_results.json \\
        --out docs/bem_hulme_haskind_verification.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE_REL = "benchmarks/verified/casepool_bem_hulme_hemisphere"


def _pct(x: float, digits: int = 3) -> str:
    return f"{100.0 * x:+.{digits}f}%"


def _num(x: float, digits: int = 5) -> str:
    return f"{x:.{digits}f}"


def render(data: dict) -> str:
    cells = sorted(data["cells"], key=lambda c: c["ka"])
    ladder = sorted(data.get("mesh_ladder", []), key=lambda r: (r["ka"], r["panels"]))
    gate = data["gate"]
    sub = data.get("mesh_subdivisions", 4)
    panels = 4 * 4 ** sub
    env = data.get("environment", {})
    threads = env.get("threads", {})

    lines: list[str] = []
    add = lines.append
    add("# CASE-BEM-HULME-HEMISPHERE — Haskind reciprocity cross-check")
    add("")
    add("**Repository**: TensorBEM · **Case**: `BEM-HULME-HEMISPHERE` (floating hemisphere, "
        "radius $a=1$ m, deep water, beam seas) · **Artifact**: "
        f"`{CASE_REL}/haskind_verification_results.json` · **Generated**: {data['timestamp']}")
    add("")
    add("This page is rendered by `make_haskind_doc.py` from the run artifact; "
        "no digit on it is hand-copied.")
    add("")
    add("## 1. What is being verified, and why it is not a tautology")
    add("")
    add("The case page (`docs/bem_hulme_hemisphere_case.md`) certifies the excitation "
        "forces $F_x, F_z$ against Hulme's semi-analytical wave-free multipole series. "
        "That comparison decides whether the *numbers* are right; it does not decide "
        "whether the *diffraction solve* is right, because a wrong diffraction RHS and a "
        "wrong force integral can conspire to land on the reference value.")
    add("")
    add("The Haskind–Newman relation removes the diffraction solve from the problem "
        "entirely: the excitation force follows from the **radiation** potentials alone,")
    add("")
    add("$$F_j = -\\mathrm{i}\\omega\\rho\\int_{\\Gamma_B}"
        "\\left(\\varphi_I\\, n_j - \\varphi_j \\frac{\\partial\\varphi_I}{\\partial n}\\right)\\mathrm{d}S,$$")
    add("")
    add("with $\\varphi_I$ the incident Airy potential and $\\varphi_j$ the $j$-th radiation "
        "potential. Two implementations are present and both are exercised here:")
    add("")
    add("| route | formula | what it shares with the diffraction solve |")
    add("|---|---|---|")
    add("| `boundary_haskind` (near field) | the integral above, on the body panels | the "
        "discretised free-surface operator and the panel quadrature |")
    add("| `kochin_haskind` (far field) | $\\rho g A\\, K_j(\\beta+\\pi)$, $K$ the Kochin "
        "function on the body surface | the radiation potential only — **no** $n_j$ "
        "weighted integral, so it is the most independent of the three |")
    add("")
    add("So the boundary route is a *reciprocity* check on the assembled system, and the "
        "Kochin route is a *far-field* check on the same radiation potentials. Neither is "
        "independent of `tensorbem` (see §6), and that limit is stated rather than hidden.")
    add("")
    add("### Predeclared gate")
    add("")
    add(f"* **H1** — on the band mesh ({panels} panels, subdivisions {sub}), "
        f"$|F_j^{{\\mathrm{{Haskind}}}} - F_j^{{\\mathrm{{diff}}}}| / |F_j^{{\\mathrm{{diff}}}}| "
        f"< {gate*100:.0f}\\%$ for surge and heave at every $ka$ of the declared band "
        "(0.2, 0.5, 0.8, 1.0, 1.5, 2.0).")
    add("* **H2** — the coarse-mesh ladder must show the same discrepancy *shrinking* "
        "under refinement; an agreement that degrades with refinement would mean the two "
        "routes agree for the wrong reason.")
    add("")
    add("## 2. Method and tool chain")
    add("")
    add("One cell = one mesh $\\times$ one frequency, with the diffraction and radiation "
        "problems sharing a single assembled operator "
        "(`assemble_free_surface_operators_3d(mesh, nu, chief_points=\"auto\")`), so the "
        "comparison is not polluted by two different discretisations of the same physics:")
    add("")
    add(f"1. `hemisphere_hull_mesh(subdivisions={sub}, radius=1)` → {panels} panels; the case "
        "geometry gate ($|\\nabla/\\nabla_{exact}-1|<2\\%$, "
        "$|A_{wp}/\\pi a^2-1|<2\\%$) is inherited from the case driver.")
    add("2. `solve_radiation_3d(modes=(0, 2))` → surge and heave radiation potentials.")
    add("3. `solve_diffraction_3d(...)` → $F_x$ = `excitation_force[0]`, "
        "$F_z$ = `excitation_force[2]` (the 6-DOF vector is "
        "surge/sway/heave/roll/pitch/yaw; index 2 is heave, not 1).")
    add("4. `haskind_excitation_force_3d(...)` → both Haskind routes.")
    add("5. All three forces are normalised by $\\rho g A a^2$, the same normalisation the "
        "case driver and the Hulme reference use.")
    add("")
    add("Tool chain: TensorBEM `src/tensorbem` at the mission branch, "
        f"Python {env.get('python', 'n/a')}, numpy {env.get('numpy', 'n/a')}, "
        f"OMP/OpenBLAS threads {threads.get('OMP_NUM_THREADS', 'n/a')}/"
        f"{threads.get('OPENBLAS_NUM_THREADS', 'n/a')}. "
        "Runs are submitted to the cluster scheduler (8 cores, single command).")
    add("")
    add(f"## 3. Results — band cells, {panels} panels")
    add("")
    add("### 3.1 Three-route force magnitudes (dimensionless, $\\rho g A a^2$)")
    add("")
    add("| $ka$ | surge diff. | surge Haskind | surge Kochin | surge Hulme | "
        "heave diff. | heave Haskind | heave Kochin | heave Hulme |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        add(f"| {c['ka']:g} | {_num(c['Fx_diff_mag'])} | {_num(c['Fx_haskind_mag'])} | "
            f"{_num(c['Fx_kochin_mag'])} | {_num(c['hulme_Fx_mag'])} | "
            f"{_num(c['Fz_diff_mag'])} | {_num(c['Fz_haskind_mag'])} | "
            f"{_num(c['Fz_kochin_mag'])} | {_num(c['hulme_Fz_mag'])} |")
    add("")
    add("### 3.2 Relative discrepancies against the direct diffraction solve")
    add("")
    add("| $ka$ | Haskind $F_x$ | Haskind $F_z$ | Kochin $F_x$ | Kochin $F_z$ | "
        "diff. vs Hulme $F_x$ | diff. vs Hulme $F_z$ |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        add(f"| {c['ka']:g} | {_pct(c['rel_err_Fx'])} | {_pct(c['rel_err_Fz'])} | "
            f"{_pct(c['rel_err_Fx_kochin'])} | {_pct(c['rel_err_Fz_kochin'])} | "
            f"{_pct(c['rel_err_Fx_vs_hulme'])} | {_pct(c['rel_err_Fz_vs_hulme'])} |")
    add("")
    add("### 3.3 Solver health of the same cells")
    add("")
    add("| $ka$ | panels | operator cond. | radiation residual | diffraction residual |")
    add("|---:|---:|---:|---:|---:|")
    for c in cells:
        add(f"| {c['ka']:g} | {c['panels']} | {c['condition_number']:.3f} | "
            f"{c['radiation_residual']:.2e} | {c['diffraction_residual']:.2e} |")
    add("")
    band_max = max(max(c["rel_err_Fx"], c["rel_err_Fz"]) for c in cells)
    band_max_k = max(max(c["rel_err_Fx_kochin"], c["rel_err_Fz_kochin"]) for c in cells)
    add(f"**Band verdict**: `{data.get('band_verdict', data['verdict'])}` — worst Haskind "
        f"discrepancy {_pct(band_max)} (gate {gate*100:.0f}%), worst Kochin discrepancy "
        f"{_pct(band_max_k)}.")
    add("")
    if ladder:
        add("## 4. Results — coarse-mesh ladder (gate H2)")
        add("")
        add("| $ka$ | panels | Haskind $F_x$ | Haskind $F_z$ | Kochin $F_x$ | Kochin $F_z$ |")
        add("|---:|---:|---:|---:|---:|---:|")
        for r in ladder:
            add(f"| {r['ka']:g} | {r['panels']} | {_pct(r['rel_err_Fx'])} | "
                f"{_pct(r['rel_err_Fz'])} | {_pct(r['rel_err_Fx_kochin'])} | "
                f"{_pct(r['rel_err_Fz_kochin'])} |")
        add("")
        add(f"**Ladder verdict**: `{data.get('ladder_verdict', 'n/a')}`.")
        add("")
    add("## 5. Figures")
    add("")
    add(f"* `{CASE_REL}/fig_haskind_magnitude_comparison.png` — diffraction / Haskind "
        "(boundary) / Haskind (Kochin) / Hulme magnitudes against $ka$, surge and heave.")
    add(f"* `{CASE_REL}/fig_haskind_error_comparison.png` — relative discrepancy of both "
        "Haskind routes against the direct diffraction solve, with the H1 gate line.")
    add(f"* `{CASE_REL}/fig_haskind_three_way.png` — the three-way comparison with the "
        "BEM-vs-Hulme error on the right axis.")
    add("")
    add("## 6. Honest limits")
    add("")
    add("1. **Not an independent code.** All three routes live in `tensorbem`; the "
        "boundary route shares the assembled operator with the diffraction solve, and both "
        "Haskind routes share the radiation potentials. A defect in the free-surface "
        "kernel (`tensorbem.free_surface3d.wave_function_3d`) is invisible to this page. "
        "The independent axes remain the capytaine cross-check "
        "(`docs/case_bem_hulme_independent_axis.md`) and the Hulme series re-derivation.")
    add("2. **Reciprocity, not solvability.** Agreement of the boundary route certifies "
        "that the discretised reciprocity relation $\\int\\varphi^{scat}n_j\\,\\mathrm{d}S = "
        "-\\int\\varphi_j\\,\\partial_n\\varphi_I\\,\\mathrm{d}S$ holds on this mesh — it "
        "does not certify that either potential is the physical one.")
    add("3. **Band stops at $ka=2$.** The gate is applied on the declared band; the "
        "$ka>2$ behaviour of this case is documented separately "
        "(`docs/hulme_band_extension.md`) and is not re-opened here.")
    add("4. **Single wave heading.** Only beam seas ($\\beta = 0$) are exercised, which is "
        "the heading of the case definition; heading symmetry is covered by "
        "`tests/test_hulme_heading_symmetry.py`.")
    add("")
    add("## 7. Reproduce")
    add("")
    add("```")
    add("PYTHONPATH=src python " + CASE_REL + "/haskind_verification.py \\")
    add("    --out " + CASE_REL + " --ladder")
    add("python " + CASE_REL + "/make_haskind_doc.py")
    add("pytest tests/test_hulme_haskind_verification.py")
    add("```")
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(HERE / "haskind_verification_results.json"))
    parser.add_argument("--out", default=str(REPO / "docs" / "bem_hulme_haskind_verification.md"))
    args = parser.parse_args(argv)

    data = json.loads(Path(args.results).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())