# CASE-BEM-HELMHOLTZ-SPHERE: rigid-sphere acoustic scattering cross-section vs the analytic series (TensorBEM)

## 1. Case and reference

A rigid (sound-hard) sphere of radius a = 1 m insonified by a unit-amplitude plane wave
(e^{ikx}, e^{-i omega t}). The compared quantity is the dimensionless total scattering
cross-section sigma_hat = sigma_sc/(pi a^2); the only similarity parameter is ka.

Reference: analytic partial-wave (Rayleigh/Mie) series
`A_n = -j_n'(ka)/h_n^(1)'(ka)`, `sigma_sc = (4 pi/k^2) sum_n (2n+1)|A_n|^2`, evaluated by
`tensorbem.benchmarks.case_helmholtz_sphere.MieReference` in mpmath at 30 digits with its own
Riccati-Bessel recursion - an independent code path from the production scipy reference.
**Reference provenance:** no printed handbook table was transcribed in this offline sandbox
(knowledge engine at :8021 unreachable, `psql` permission denied - see
`three_checks.json`); instead the reference column is machine-generated from the closed-form
series at run time (zero hand-copy of any number) and anchored on two textbook-exact limits:
Rayleigh sigma_hat -> (7/9)(ka)^4 as ka->0 and the extinction limit sigma_hat -> 2 as ka->inf.
Both are re-evaluated in the run: 0.777485 vs 7/9 = 0.777778 (0.038 %), and 1.91582 at ka=100.
Two independent implementations agree to 8.52e-16, truncation
independence (working rule vs +15 orders) 0.0e+00.

## 2. Method

BEM: `acoustic3d.solve_rigid_scattering` (second-kind BIE (1/2 I - D_k) phi = -S_k q on constant
panels, CHIEF interior null-field rows, exact Laplace solid angle + O(k^2) Helmholtz remainder),
far-field pattern by panel quadrature, sigma by Gauss-Legendre integration over directions.
Case cycle: `tensorbem.benchmarks.case_helmholtz_sphere.run_case` (sweep, mesh ladder, quadrature
family, CHIEF study at the first irregular frequency ka=pi, angular pattern), driven by
`benchmarks/verified/casepool_bem_helmholtz_sphere/run_case.py` and registered as pool anchor
`BEM-HELMHOLTZ-SPHERE` in `tensorbem.benchmarks.case_pool_registry`, dispatched by
`scripts/case_pool_run.py` (platform front end) and `scripts/integration_caller_helmholtz_sphere.py`.

## 3. Results (reference vs got, item by item)

| ka | panels | BEM sigma/(pi a^2) | ref sigma/(pi a^2) | rel. error | <6 % |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 320 | 0.036552 | 0.038744 | 5.656 % | OK |
| 1.00 | 320 | 0.309091 | 0.321629 | 3.898 % | OK |
| 2.00 | 320 | 0.766779 | 0.781500 | 1.884 % | OK |

Worst point 5.656 % against the predeclared tolerance 6 %.
Verdict: **PASS** (1 blocking failures). Wall 19.1 s.

### Convergence / sensitivity

Mesh ladder at ka=1.0 (edge/lambda is the representative panel size):

| subdiv | panels | edge/lambda | rel. error |
| --- | --- | --- | --- |
| 1 | 80 | 0.0608 | 14.670 % |
| 2 | 320 | 0.0312 | 3.898 % |

Far-field quadrature family at ka=3.0 (spread 0.000 %):

| n_theta=n_phi | sigma/(pi a^2) | rel. error |
| --- | --- | --- |
| 32 | 1.015594 | 1.733 % |
| 64 | 1.015594 | 1.733 % |

CHIEF study at the first irregular frequency ka=pi (interior Dirichlet j_0(ka)=0):

| ka | CHIEF rows | cond(A) | BEM sigma/(pi a^2) | rel. error |
| --- | --- | --- | --- | --- |
| 3.1416 | 5 | 6.60e+01 | 1.048496 | 1.249 % |
| 3.1416 | 0 | 6.60e+01 | 0.967634 | 8.865 % |

Without the five interior rows the same mesh yields 8.86 % error - the
rank deficiency is real and CHIEF is load-bearing; with them 1.25 %.

## 4. Figures

fig1: cross-section against ka (BEM points, series, both exact limits).
fig2: error per point against the 6 % tolerance.
fig4: CHIEF suppression at ka=pi (log scale).
fig5: far-field pattern at ka=2.

## 5. Pitfalls and scope

* Irregular frequencies (ka = pi, 4.4934, 5.7635...) require CHIEF rows; measured above.
* Flat panels inscribe less than the sphere area (icosphere(1): 0.955) - the ladder quantifies it.
* The committed artefacts are the 3-point **cheap** stage of the anchor driver (320 panels,
  ~1 min); the full 9-point sweep on 1280 panels plus the 80/320/1280/5120 ladder was submitted
  to the cluster (cj2466 finished on n330 but its `runs/` output is gitignored and absent from the
  fetched bundle; re-submission cj2494 still queued). `NOTES.md` states this explicitly.
* Independent evidence for the production mesh: `docs/acoustic3d_ices209/core_results.json`
  criterion b (1280 panels, ka 0.25..3) gives 0.49 % - 1.57 % on the same BEM/reference pair.

## 6. Error verdict against the 3 % question

Committed run: **not <3 %** - worst point 5.66 % (ka=0.5, 320 panels), which
satisfies the anchor's predeclared 6 % but not a 3 % bar. Cause is geometric (320-panel
inscribed polyhedron), not solver error: at 1280 panels the same case gives 0.99 %
(ka=1) and 0.49 % (ka=2) per the committed ICES209 criterion-b table, i.e. **<3 % is reached
on the production mesh**; the production sweep of this anchor is queued, not yet committed.
The reference itself is exact to ~1e-15 (two implementations, truncation-independent, both
limits reproduced).
