# m16976 — CASE-WAVE-FLUME: linear wavemaker transfer functions (TensorBEM)

**Anchor**: `CASE-WAVE-FLUME` (算例池·规则波生成, acceptance `<10%` vs Biesel-Havelock).
**Branch**: `agent/m16976` · **Deliverables**: `src/tensorbem/wavemaker.py`,
`src/tensorbem/benchmarks/wave_flume_wavemaker.py`, `tests/test_wavemaker_transfer.py`,
`docs/wave_flume_wavemaker{,.md}` (study.json + 2 figures), capability row
`wave-flume-wavemaker` in `docs/audit_baseline.md`.

## 1. Case

Constant-depth flume, `h = 1 m`, `g = 9.80665`. The end wall is panelled from the bed to
the waterline and driven with a prescribed normal velocity; the free-surface condition,
the bed and the radiation condition are carried by the finite-depth kernel. Two classical
wall motions: piston (`u = U0`) and bed-hinged flap (`u = U0 (z+h)/h`), `kh ∈ {0.5, 1, 2, 3}`,
panels `8/16/32/64`.

## 2. Method

1. `wavemaker.py` assembles the finite-depth single-layer normal-derivative matrix for the
   **open** wall curve: exact Rankine row integrals (reused from
   `free_surface_solver._rankine_normal_derivative_matrix`) + Gauss quadrature of
   `d(G_smooth)/dn` (`finite_depth.green_gradient(..., include_rankine=False)`).
   `finite_depth_solver` refuses panels touching `z = 0`/`z = -h`, which a wavemaker does by
   definition, so only the geometry container is new; the operator is proven identical to the
   graded closed-body assembly on a submerged circle (`max |ΔA| / max|A| < 1e-12`).
2. Solve `(A + I/2) σ = u·n`, then read the radiated elevation three ways: Kochin integral
   (`kochin_amplitude`), direct kernel quadrature two wavelengths downstream
   (`free_surface_potential`), and the exact eigenfunction series including evanescent modes
   (`exact_elevation`, derived from the strip modes — independent of any panel algebra).
3. Gauss nodes are asserted to stay strictly inside each panel: at the waterline the image
   term is `log`-singular and the closed-form Rankine part only covers the direct term.

## 3. Results (programmatic, `docs/wave_flume_wavemaker/study.json`)

| quantity | value |
|---|---|
| worst `H/S` error vs Biesel-Havelock, all meshes | **1.0858 %** (8 panels, piston kh=3) |
| worst `H/S` error, 64-panel mesh | **0.0171 %** |
| worst near-field error vs exact eigenfunction series (0.25–4 λ) | **7.93e-5** |
| worst disagreement between the two far-field routes | 0.5836 % (8 panels) |
| convergence | O(h²): piston kh=2 err 0.260 % → 0.0651 % → 0.0163 % → 0.0041 % |

Anchor tolerance `<10 %` → **PASS** (worst 1.09 %, finest mesh 0.017 %); 3 % → **PASS**.

## 4. Scope statement (the honest part)

For a **vertical full-depth wall every cross-panel entry of the operator vanishes identically**
(all panels co-planar ⇒ kernels depend on `|dx|` only ⇒ `d/dx = 0`): measured
`max|A| < 1e-12`, `cond(A + I/2) = 1.0000`. The panel solution therefore *is* the Havelock
projection, and this case grades the **kernel quadrature and mode decomposition**, not the
integral-equation solve. To bound that, an inclined-face wavemaker (same strip, face sloped by
α) is included: the operator becomes non-degenerate (`cond` 1.0000 → 1.29 → 2.22 and max
off-diagonal 2.3e-2 → 7.1e-2 for α = 10° → 30°) and, as α → 0, the computed transfer returns
to the analytic piston value (α = 1e-3° → 0.001 %), which is the quantitative anchor for the
non-degenerate solve.

## 5. Limits

Linear theory: no steepness limit, no breaking, no wavemaker backlash or absorption at the
far end. The flume "wave-height attenuation" appears here only as evanescent-mode decay
(flat to round-off past one wavelength) — propagation loss needs viscous/dissipative physics
that a potential solver does not have.

## 6. Verification run

`PYTHONPATH=src python -m pytest tests/test_wavemaker_transfer.py` → **20 passed** (67 s);
`tests/test_capability_coverage.py` + `tests/test_capability_evidence.py` → **22 passed**
(coverage gate: 0 unowned modules, baseline regenerated with
`python -m tensorbem.capability_evidence --write`).