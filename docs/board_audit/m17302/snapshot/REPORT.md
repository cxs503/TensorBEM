# CASE-BEM-CATHODIC-SPHERE — conducting-sphere anchor vs DNV-RP-B401 `4 pi sigma a phi_a`

Case-pool ledger entry: `|BEM-CATHODIC-SPHERE| ref = 4 pi sigma a phi_a (DNV-RP-B401 analytic) | tolerance < 5 %`.

Artifacts (all numbers below are interpolated from them, none copied by hand):

| artifact | content |
|---|---|
| `artifacts/cathodic_sphere/results.json` (committed copy: `docs/cathodic_sphere_dnv_b401/results.json`) | convergence, two-route, scaling, decay blocks + verdict |
| `artifacts/cathodic_sphere/convergence_current.png` | current error vs panels, 2nd-order line, 5 %/3 % bands |
| `artifacts/cathodic_sphere/exterior_decay.png` | exterior potential vs `phi_a a / r` at two resolutions |
| `artifacts/cathodic_sphere/error_budget.png` | convergence levels + non-dimensional scaling points vs tolerance |

Reproduce:

```bash
PYTHONPATH=src python scripts/cathodic_sphere_dnv_b401_study.py \
    --subdiv 1 2 3 --routes-subdiv 2 3 --scaling-subdiv 2
PYTHONPATH=src python -m pytest tests/test_cathodic_sphere_anchor.py -q
```

Environment: python 3.11.6, numpy 2.4.6, `Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38`.

---

## 1. Case introduction and reference

**Physics.** Seawater is a homogeneous conductor (σ). In the steady state the
electrolyte potential solves ∇²φ = 0 in the exterior domain; the metal boundary
carries either a prescribed potential (anode / coating holiday) or zero normal
current (intact coating). A bare metal **sphere of radius `a` at the uniform
potential `phi_a` in an unbounded electrolyte** is the one configuration of this
model with a closed-form solution: with `phi = phi_a a / r` the normal derivative
on the metal is `∂φ/∂n = −phi_a/a`, so Ohm's law gives the total current and the
spreading resistance

```
I = -sigma ∮_S ∂φ/∂n dS = 4 pi sigma a phi_a      R = 1 / (4 pi sigma a) = rho / (4 pi a)
```

**Reference provenance (external anchor).** `R = rho / (4 pi a)` is the classical
spreading resistance of an isolated spherical electrode; DNV-RP-B401 *Cathodic
Protection Design* builds its anode-resistance table on the same `rho·R` scaling
(slender stand-off `R = rho/(2 pi L)(ln(4L/a) − 1)`, flush-mounted, bracelet),
the sphere being the limiting geometry with an exact closed form. The ledger
criterion therefore tests (i) the `sigma` factor of Ohm's law, (ii) the `4 pi` of
the sphere's surface area, (iii) the exterior representation `phi = phi_a a / r`.
Two follow-on limits are stated honestly in §6: the model has no polarization
kinetics, and a uniformly polarised sphere collapses the *mixed* BIE to the pure
single-layer form, so this anchor does not exercise the holiday/anode logic.

---

## 2. Method and implementation

Exterior single/double-layer representation, collocated at panel centroids:

```
S q = (1/2 I - H) phi ,      S_ij = ∫_Tj G(xi,y) dS,   H_ij = ∫_Tj ∂G/∂n dS
```
with `G = 1/(4 pi |x-y|)`; `H` uses exact signed solid angles on flat triangles, the
self term `S_ii` uses exact polar integration of the flat panel, off-diagonal panels
use Duffy quadrature (order 8). Uniform potential on every panel reduces the mixed
system to `S q = (1/2 - H) phi_a`; the physical current is `j_n = sigma q` and the
total is `I = sum_i sigma q_i A_i`.

Code pointers:

| element | location |
|---|---|
| BIE assembly (`S`, `H`) | `src/tensorbem/solver3d.py:96` `assemble_surface_matrices` |
| mixed solver, current integral | `src/tensorbem/cathodic_protection.py` `solve_cathodic_protection` |
| closed-form anchor | `src/tensorbem/cathodic_protection.py` `verify_conducting_sphere` |
| exterior representation | `src/tensorbem/cathodic_protection.py` `evaluate_exterior_potential` |
| **far-field (Gauss-law) route** (new) | `src/tensorbem/cathodic_protection.py` `far_field_current`, `verify_conducting_sphere_routes` |
| benchmark CLI | `src/tensorbem/benchmarks/cathodic_protection_sphere.py` |
| this study | `scripts/cathodic_sphere_dnv_b401_study.py` |
| regression tests | `tests/test_cathodic_sphere_anchor.py` |

**Second current route (new in this round).** Outside the body the multipole
expansion gives `phi = I/(4 pi sigma r) + O(r^-2)`; averaging `phi` over a
quasi-uniform direction set on a sphere of radius `R` isolates the monopole, so

```
I_R = 4 pi sigma R <phi>_{|x|=R} = I_inf + c/R   ->   I_inf  by least squares in 1/R
```
This route never touches the panel normal derivative: it uses the exterior field
only, which is the discrete statement of charge conservation (Gauss law on an
enclosing sphere).

---

## 3. Parameters

| quantity | value | non-dimensional form |
|---|---|---|
| sphere radius `a` | 1 m | scaling length |
| imposed potential `phi_a` | -1 V vs electrolyte | `phi/phi_a` |
| conductivity `sigma` | 3.5 S/m | appears only through `I/(4 pi sigma a phi_a)` |
| surface panels (levels) | 80 / 320 / 1280 | `h/a` = 0.396 / 0.198 / 0.099 |
| target non-dimensional current | `I/(4 pi sigma a phi_a) = 1` | tolerance 5 % |
| quadrature order (off-diagonal) | 8 (Duffy) | self term analytic |
| collocation | panel centroid, piecewise-constant `q`, `phi` | – |

---

## 4. Reference vs computed

Closed form at the stated parameters: `I_exact = 4 pi sigma a phi_a = -43.982297 A`, `R_exact = 1/(4 pi sigma a) = 0.022736420 ohm`.

| sub | panels | h/a | `I` [A] | error vs ref | `R_num` (magnitude) [ohm] | max exterior-field error | relative residual | ref check |
|---|---|---|---|---|---|---|---|---|
| 1 | 80 | 0.396 | -41.820245 | 4.9157 % | 0.023911863 | 4.9474 % | 2.11e-16 | PASS |
| 2 | 320 | 0.198 | -43.410210 | 1.3007 % | 0.023036055 | 1.3088 % | 4.19e-16 | PASS |
| 3 | 1280 | 0.099 | -43.836366 | 0.3318 % | 0.022812110 | 0.3339 % | 7.68e-16 | PASS |

Ledger criterion check list:

| # | criterion | value | verdict |
|---|---|---|---|
| C1 | finest-level current error < 5 % (ledger) | 0.3318 % | PASS |
| C2 | finest-level exterior field error < 5 % | 0.3339 % | PASS |
| C3 | internal stricter target < 3 % | 0.3318 % | PASS |
| C4 | observed order ≈ 2 for the finest pair | 1.971 | PASS |
| C5 | surface vs far-field current agree < 5 % | 1.35e-08 | PASS |
| C6 | non-dimensional scaling `I/(4 pi sigma a phi)=1` over 4 parameter points | max dev 1.3007 % | PASS |

---

## 5. Convergence and parameter sensitivity

Observed orders `p = log(e_coarse/e_fine)/log((N_fine/N_coarse)^1/2)`: 1.918, 1.971. A second-order law in the panel size `h` is what a piecewise-constant collocation on a smooth surface should produce; the finest pair sits at 1.971.

| panel pairs | e_coarse | e_fine | N ratio^(1/2) | observed order |
|---|---|---|---|---|
| 80 -> 320 | 4.9157 % | 1.3007 % | 2.000 | 1.918 |
| 320 -> 1280 | 1.3007 % | 0.3318 % | 2.000 | 1.971 |

**Parameter sensitivity (non-dimensional).** The anchor is exact for every
(σ, a, φ): a solver that double-counts σ, forgets the surface area, or mishandles
the potential scale produces a non-unit ratio at some point.

| σ [S/m] | a [m] | phi_a [V] | panels | `I/(4 pi sigma a phi)` | deviation | `R_num` (magnitude) [ohm] | `R_exact` [ohm] |
|---|---|---|---|---|---|---|---|
| 3.5 | 1 | -1 | 320 | 0.986993 | 1.3007 % | 0.023036055 | 0.022736420 |
| 2 | 1 | -1 | 320 | 0.986993 | 1.3007 % | 0.040313097 | 0.039788736 |
| 3.5 | 0.5 | -1 | 320 | 0.986993 | 1.3007 % | 0.046072111 | 0.045472841 |
| 3.5 | 1 | -2 | 320 | 0.986993 | 1.3007 % | 0.023036055 | 0.022736420 |

All four points share the *same* non-dimensional error — the discretisation error
of the uniformly polarised sphere — which is the expected invariance.

**Exterior decay.** Averaged `phi` on spheres of radius `r = 1.25, 1.5, 2, 3, 5, 8, 12` m, compared with `phi_a a / r`:

| r [m] | exact [V] | BEM sub1 [V] | rel err | BEM sub2 [V] | rel err |
|---|---|---|---|---|---|
| 1.25 | -0.800000 | -0.760685 | 4.9144 % | -0.789588 | 1.3015 % |
| 1.5 | -0.666667 | -0.633885 | 4.9173 % | -0.657992 | 1.3013 % |
| 2 | -0.500000 | -0.475419 | 4.9162 % | -0.493496 | 1.3009 % |
| 3 | -0.333333 | -0.316947 | 4.9158 % | -0.328998 | 1.3007 % |
| 5 | -0.200000 | -0.190169 | 4.9157 % | -0.197399 | 1.3007 % |
| 8 | -0.125000 | -0.118855 | 4.9157 % | -0.123374 | 1.3007 % |
| 12 | -0.083333 | -0.079237 | 4.9157 % | -0.082249 | 1.3007 % |

The relative decay error is flat in `r` (spread 7.93e-06 over the two decades of `r`), i.e. the `1/r` law itself is reproduced to machine precision: the whole residual is a uniform scale error of the boundary solution, identical to the current error. It does not drift outward, which is what would happen if the exterior representation lost part of the single- or double-layer contribution.

---

## 6. Failure modes, limits and artifacts

**What this anchor does not prove.**
1. *Mixed boundary conditions.* Prescribing `phi_a` on every panel removes the
   coating (Neumann) unknowns, so the holiday/anode assembly branch is not
   exercised here. That branch is covered by the hull studies in
   `docs/cathodic_protection_linear/` and `docs/cathodic_protection_galvanic/`.
2. *Polarization kinetics.* The model is linear; it cannot be validated against
   measured protection potentials (that is `tensorbem.cp_galvanic`).
3. *Route independence.* The far-field route shares the solved density `q` with
   the surface route, so their agreement certifies the exterior representation
   formula and the Gauss-law bookkeeping (σ, `4 pi`, double-layer cancellation),
   **not** the boundary quadrature; the latter is certified by the closed form and
   the second-order convergence.
4. *Point-anode DNV tables.* The slender/flush DNV-RP-B401 `R` formulas are not
   reproduced here; only the exact spherical limit is.

**Traps hit while producing these numbers.**
1. `python -m tensorbem...` in a fresh worktree resolved an *editable install
   pointing at another worktree* (`/root/tp-agent/work/m17241/src/...`), so the
   first timing runs measured foreign code. Always `PYTHONPATH=<worktree>/src`
   and assert `tensorbem.__file__` before trusting a number.
2. A single-refinement run of the benchmark CLI crashed on the `None`
   observed-order placeholder (`TypeError: unsupported format string ... NoneType`);
   fixed and pinned by `test_benchmark_cli_handles_single_level`.
3. Cost: a python loop assembles `S`/`H` row by row; level 3 (1280 panels) takes
   ~55 s, level 4 (5120) goes to the cluster queue rather than the control node.

**Verdict.** all ledger criteria PASS: finest-level current error 0.3318 % vs the 5 % ledger band (internal target 3 %).

Files: `artifacts/cathodic_sphere/results.json`, `convergence_current.png`,
`exterior_decay.png`, `error_budget.png` (committed copies of the JSON and the
three figures live next to this report in `docs/cathodic_sphere_dnv_b401/`), this report.

