# Ursell panel-solution verification debt: root cause and correction

**Scope**: closes the open *honest FAIL* recorded in `expand_audit_tensorbem.md` §2.2
(mission m238, 2026-10-24): the 128-panel Ursell added-mass error measured at
\(8.51\times10^{-3}\) while `README.md` line 13 and `docs/marine_bem_benchmarks.md`
§4.6 quoted \(7.7\times10^{-3}\) (+10.6 %).

**Verdict**: the documentation was stale, not the solver, and the audit's two
guessed causes (different panel/node settings, different BLAS/NumPy) were both
wrong. The panel solution changed because the *source-potential evaluation itself*
changed in commit `267d5a3` ("Add marine rigid-body Neumann, diffraction, Helmholtz,
and 3-D Kelvin BEM suites"). The analytical reference did not move at all, which is
what makes the diagnosis unambiguous. The quoted numbers in the README and in §4.6
are now the reproducible ones, and a formulation-level regression test prevents the
same silent drift.

---

## 1. Case

Ursell's (1949) submerged circular cylinder in deep water, radiation problem:

| quantity | value |
|---|---|
| radius \(a\) | 1.0 m |
| axis submergence \(f\) | 2.0 m (\(a/f = 0.5\)) |
| non-dimensional frequency \(ka\) | 0.6 |
| density \(\rho\) | 1025 kg/m³ |
| gravity \(g\) | 9.80665 m/s² |
| \(\omega\) | \(\sqrt{ka\,g/a} = 2.42569\) rad/s |
| reported quantities | added mass \(A\) and damping \(B\) per unit span |
| reference | Ursell multipole solution, truncated closed-form system (no BEM code shared) |
| reference values | \(A = 2545.1672\) kg/m, \(B = 2718.1540\) kg/(m·s) |

The panel solution is the indirect (source-distribution) formulation with piecewise
constant sources. The near-field/far-field damping identity (pressure integral vs
outgoing-wave amplitude) is the check on the radiation condition; it is reported for
every run.

## 2. Method

1. **Bisect by commit.** `git archive 878834a` (the commit that added the Ursell
   benchmark and whose printed table is quoted in the docs) and the branch HEAD were
   both run with identical arguments; `git diff 878834a HEAD -- src/tensorbem/free_surface_solver.py`
   and `free_surface.py` isolated the change.
2. **Separate the two errors.** A quadrature error and a discretisation error are
   distinguishable: sweep the Gauss-Legendre nodes per panel at a fixed mesh, then
   sweep the panel count at fixed nodes. If the answer moves with the node count the
   quoted number is quadrature-limited and cannot be a statement about the method.
3. **Keep the numbers in artefacts.** Every run is stored verbatim in
   `docs/ursell_quadrature_study/raw/`; `docs/ursell_quadrature_study/build_summary.py`
   regenerates `summary.json` / `summary.md` from those files and nothing in this
   report is transcribed by hand. `docs/ursell_quadrature_study/make_figure.py`
   regenerates the figure.
4. **Lock the formulation.** `tests/test_ursell_quadrature_convergence.py` asserts,
   on a coarse mesh, that the panel solution is insensitive to the node count, that
   the convergence is first order, and that the quoted table is regenerated from the
   committed raw runs.

### 2.1 What actually changed in `267d5a3`

| | legacy (`878834a`) | current |
|---|---|---|
| source potential \(\phi(P_i)=\int G\sigma\,ds\) | full Green function, singular `log r` included, handed to a 24-node Gauss-Legendre rule per panel | singular Rankine part integrated **analytically** over each straight panel (closed-form vector log primitive), only the smooth free-surface remainder quadratured |
| smooth-remainder nodes | — | `min(panel_nodes, 6)` |
| free-surface part of the normal-derivative matrix | full kernel quadrature then subtract a numerically differentiated Rankine term | `green_gradient_field(..., include_rankine=False)` + analytic Rankine matrix |

The legacy quadrature therefore contained a genuine singularity error; it happened to
bias \(A\) *downwards*, i.e. towards the reference value for this particular geometry
and frequency.

## 3. Results

### 3.1 Quadrature-node sweep at 128 panels (the decisive experiment)

| panel_nodes | current \(A\) | current \(B\) | legacy \(A\) | legacy \(B\) |
|---|---|---|---|---|
| 6 | 2566.834334 | 2754.470100 | — | — |
| 12 | 2566.834334 | 2754.470100 | 2562.870278 | 2752.582874 |
| 24 | 2566.834334 | 2754.470100 | 2564.813781 | 2753.508146 |
| 48 | 2566.834334 | 2754.470100 | 2565.813896 | 2753.984285 |

* current: spread exactly \(0\) (\(0\) relative) — the matrix entries change by
  \(\le 3\times10^{-17}\) absolute between 6 and 48 nodes, i.e. roundoff;
* legacy: spread \(2.94\) in \(A\) (\(1.16\times10^{-3}\) relative) and \(1.40\) in
  \(B\) (\(5.16\times10^{-4}\) relative), still drifting at 48 nodes, and still
  \(3.98\times10^{-4}\) relative short of the converged value;
* the legacy sequence 2562.87 → 2564.81 → 2565.81 → 2566.83 (converged) is monotone:
  the documented \(7.72\times10^{-3}\) at 24 nodes is an under-resolved *quadrature*
  value, not a more accurate discretisation.

### 3.2 Panel refinement with the current solver (panel_nodes = 24)

| panels | \(A\) | rel. error in \(A\) | \(B\) | rel. error in \(B\) | near/far | \(\kappa\) |
|---|---|---|---|---|---|---|
| 32 | 2627.4210 | \(3.232\times10^{-2}\) | 2841.7492 | \(4.547\times10^{-2}\) | \(2.049\times10^{-2}\) | 2.47 |
| 64 | 2587.7627 | \(1.674\times10^{-2}\) | 2787.2368 | \(2.542\times10^{-2}\) | \(1.054\times10^{-2}\) | 2.45 |
| 128 | 2566.8343 | \(8.513\times10^{-3}\) | 2754.4701 | \(1.336\times10^{-2}\) | \(5.341\times10^{-3}\) | 2.43 |
| 256 | 2556.0936 | \(4.293\times10^{-3}\) | 2736.7493 | \(6.841\times10^{-3}\) | \(2.689\times10^{-3}\) | 2.42 |

Observed order from 64→128→256 in \(A\): \(\log_2(1.674/0.8513)=0.98\),
\(\log_2(0.8513/0.4293)=0.99\); in \(B\): 0.93, 0.97. First order, as documented for
piecewise-constant sources with this kernel.

![Ursell quadrature study](figures/ursell_quadrature_study.png)

Left: added mass at 128 panels versus the number of Gauss-Legendre nodes. The current
solver is flat from 6 nodes on; the legacy sequence climbs towards it. Right: the
remaining error is first order in the panel size, so the quoted number is
discretisation-limited.

### 3.3 Corrected documentation values

| quantity | documented before | reproduced now | deviation |
|---|---|---|---|
| \(A\) relative error, 128 panels | \(7.72\times10^{-3}\) | \(8.513\times10^{-3}\) | +10.2 % |
| \(B\) relative error, 128 panels | \(1.30\times10^{-2}\) | \(1.336\times10^{-2}\) | +2.8 % |
| near/far damping ratio, 128 panels | \(1.0057\) | \(1.00534\) | -0.04 % |
| reference \(A\) | 2545.17 | 2545.1672 | < 1e-6 |
| reference \(B\) | 2718.15 | 2718.1540 | < 1e-6 |
| MFS \(A\) (64-base) | 2544.69 | 2544.6864 | < 1e-6 |

## 4. Error judgement against the benchmark

| key quantity | baseline | reproduced value | relative error | < 3 %? |
|---|---|---|---|---|
| \(A\) at 128 panels | Ursell multipole 2545.1672 | 2566.8343 | \(8.51\times10^{-3}\) | **yes** (0.85 %) |
| \(B\) at 128 panels | Ursell multipole 2718.1540 | 2754.4701 | \(1.34\times10^{-2}\) | **yes** (1.34 %) |
| \(A\) at 256 panels | Ursell multipole 2545.1672 | 2556.0936 | \(4.29\times10^{-3}\) | **yes** |
| near/far damping identity, 128 panels | must vanish | ratio \(1.00534\) | \(5.3\times10^{-3}\) | **yes** |
| reference values vs published zero-damping frequencies | McIver & McIver | reproduced to \(10^{-13}\) | — | yes |

No benchmark quantity exceeds the 3 % threshold; the honest accuracy statement for
this benchmark is now "first order in panel size, 0.85 % in added mass at 128 panels",
with the caveat that the constant-panel indirect formulation is not a high-accuracy
discretisation and that the MFS check (\(1.9\times10^{-4}\)) is the one to quote when a
single high-accuracy number is needed.

## 5. Pitfalls and scope

* **A moved panel number is not a bug by itself.** The analytical reference is the
  invariant; if it moves, the reference code changed, if it does not, the discretisation
  changed. Bisect from that split, not from the error value.
* **Agreement can be accidental.** Here the legacy singular quadrature was *closer* to
  the reference than the converged one. Quoting the more favourable of two
  non-converged numbers is exactly the failure mode the node sweep detects.
* **`--panel-nodes` is inert above 6 for the smooth remainder** (`min(panel_nodes, 6)`),
  and the free-surface gradient quadrature is already at roundoff with 6 nodes. The
  flag is still validated and still drives mesh-independent API calls, but
  `--panel-nodes 24` (the default) means "6 effective nodes" in this solver; this is
  now explicit in the docs.
* **Scope**: only the 2-D deep-water Ursell benchmark was re-verified. The audit's
  second open item (direct reproduction of the ellipsoid at 1280 panels) was not
  touched here; the 3-D benchmarks have their own tables and were not re-run.
* Compute footprint: all runs are 2-D with ≤ 256 panels; the whole study is a few
  minutes of single-core work.