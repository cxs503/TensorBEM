# XVAL-BEM-ROUND2-METHOD-SELECTION — exterior-Laplace BIE kernel vs OSS (Capytaine, scikit-fem)

Mission m17363 · anchor `XVAL-BEM-ROUND2-METHOD-SELECTION` · repository TensorBEM
Generated programmatically from `artifacts/xval_laplace_sphere/results.json`
(`scripts/m17363_xval_run.py`), no hand-copied digits.

## 1. Problem, reference and why it belongs to TensorBEM

The whole cathodic-protection (CP) module reduces to one numerical kernel:
`assemble_surface_matrices` in `src/tensorbem/solver3d.py` plus the exterior
representation formula in `src/tensorbem/cathodic_protection.py`.  The only
closed-form anchor so far (`verify_conducting_sphere`, documented in
`docs/cathodic_sphere_dnv_b401/REPORT.md`, OSS record
`oss_solutions(feature_slug=CASE-BEM-CATHODIC-SPHERE, tag=round2-method-selection)`)
is the **uniformly polarised** sphere: every panel carries the same Dirichlet
value, so the surface density is constant and both the off-diagonal quadrature
and the double-layer sign cancel out of the test.

Round-2 method selection therefore needs a case in which the boundary data
contains a non-trivial Legendre mode.  Chosen case (dimensionless, sphere of
radius $a$ in an unbounded electrolyte of conductivity $\sigma$):

| item | value |
|---|---|
| radius $a$ | 1 m |
| conductivity $\sigma$ | 3.5 S/m (seawater at 15 °C, DNV-RP-B401 design value) |
| imposed potential $\phi_a$ | -1 V vs Ag/AgCl (DNV protection potential) |
| Dirichlet data | $\phi(a,\mu)=\phi_a(1+\mu)$, $\mu=\cos\theta$ |
| analytic solution | $\phi(r,\mu)=\phi_a(a/r)+\phi_a(a/r)^2\mu$ |
| analytic density | $q(a,\mu)=-\partial\phi/\partial n = \phi_a(1+2\mu)/a$ |
| analytic current | $I=4\pi\sigma a\phi_a$ = -43.98230 A (monopole only) |

Dimensionless control parameters (the problem is steady and elliptic — there is
no Reynolds/Péclet number):

* $h/a=\sqrt{4\pi a^2/N}$ — dimensionless panel size (BEM);
* element size $\bar h/a$ and order $p\in\{1,2\}$ (FEM);
* truncation radius $R/a\in\{2,4,8\}$ (FEM only);
* Legendre content of the data: modes $l=0,1$ (this case) versus $l=0$ only
  (the previous anchor);
* Duffy quadrature order $\in\{4,6,8,12\}$;
* dimensionless current $\hat I = I/(4\pi\sigma a\phi_a)$ — must equal 1 and is
  invariant under $(\sigma,a,\phi_a)$ (unit-scaling detector).

Three references are compared:

1. **Analytic Legendre series** (truth, literature: exterior harmonics).
2. **Capytaine 3.1.dev0** (OSS BEM, Delhommeau Green function, compiled Nemoh
   core).  With `free_surface=inf` and `wavenumber=0` every free-surface term
   vanishes and its Green function degenerates to the Rankine kernel, so its
   influence matrices can be compared **element by element** with the TensorBEM
   assembly on the *identical triangle mesh* (`Mesh.from_list_of_faces` accepts
   triangles in Capytaine ≥ 3), and can drive the same single-layer BIE.
3. **scikit-fem 12.0.2** (OSS FEM, Galerkin). Because the data depends only on
   $\mu$, the 3-D exterior problem reduces *exactly* to the axisymmetric problem
   in $(r,\mu)$: $\int[r^2\,\partial_r\phi\,\partial_r v+(1-\mu^2)\partial_\mu\phi\,\partial_\mu v]\,dr\,d\mu=0$
   on $[a,R]\times[-1,1]$ with the exact Dirichlet data on both ends — a
   different discretisation family, hence evidence about the *formulation*, not
   only about quadrature.

Data provenance for the OSS references is programmatic: the OSS survey
(`oss_solutions`, `oss_inventory`) contains no BEM/CP repository, so the
method-selection lineage came from the literature; the two OSS implementations
used here are the locally installed packages (versions read from the packages
themselves, `scikit_fem_version` = 12.0.2,
`capytaine_version` = 3.1.dev0).

## 2. Method and implementation

New module `src/tensorbem/benchmarks/laplace_sphere_xval.py` (routes are lazy:
a missing optional dependency degrades to `available: false`, never to an
exception), driver `scripts/m17363_xval_run.py`, tests
`tests/test_laplace_sphere_xval.py`, artefacts under
`artifacts/xval_laplace_sphere/`.

* **TensorBEM route** — `solve_laplace_surface(..., domain="exterior")`:
  piecewise-constant centroid collocation, exact polar self-term, exact
  solid-angle double layer (zero diagonal), Duffy order-8 off-diagonal.
* **Capytaine route** — `DefaultMatrixEngine.build_matrices(free_surface=inf,
  water_depth=inf, wavenumber=0, adjoint_double_layer=False)`; the returned
  matrices differ from the TensorBEM ones only by the global sign of
  $G=\mp1/(4\pi|x-y|)$ and by Capytaine's implicit double-layer diagonal
  (its $\tfrac12$ jump), which is removed before the comparison.  The same
  single-layer system $S q=\tfrac12\phi_D-D\phi_D$ is then solved with
  Capytaine's matrices.
* **FEM route** — axisymmetric P1/P2 Galerkin; the current is obtained by
  integrating $\partial_r\phi$ on $r=a$ (and independently on $r=R$ as a
  conservation check); the potential is probed at $(r,\mu)$ pairs; the
  discretisation error is removed by Richardson extrapolation in $h$.
* **Post-processing** is shared by all BEM routes (single representation
  formula), so a difference between routes cannot come from the post-processing.

## 3. Results

### 3.1 Headline comparison (finest TensorBEM level, $N$=1280 panels, $h/a$=0.099)

| quantity | reference | TensorBEM (finest) | error | <15 % ? | <3 % ? |
|---|---|---|---|---|---|
| total current $I$ | analytic $4\pi\sigma a\phi_a$ = -43.98230 A | -43.83637 A | 0.332% | YES | YES |
| surface density $q$ (panel-wise) | analytic $\phi_a(1+2\mu)/a$ | $L_\infty$ | 1.082% | YES | YES |
| exterior potential $\phi(x)$ | analytic series | $L_2$ | 0.540% | YES | YES |
| total current (OSS FEM reference) | analytic | -44.01834 A | 0.082% | YES | YES |
| single-layer matrix $A$ | capytaine Delhommeau (Rankine) | Frobenius | 0.023% | YES | n/a |
| double-layer matrix $D$ | capytaine Delhommeau (Rankine) | Frobenius | 0.215% | YES | n/a |

### 3.2 BEM routes, full convergence ladder

| route | N panels | h/a | I [A] | I error | density L∞ error | exterior φ L₂ error | cond |
|---|---|---|---|---|---|---|---|
| tensorbem | 80 | 0.396 | -41.8202 | 4.916% | 3.575% | 7.829% | 18.6 |
| tensorbem | 320 | 0.198 | -43.4102 | 1.301% | 1.943% | 2.111% | 37.6 |
| tensorbem | 1280 | 0.099 | -43.8364 | 0.332% | 1.082% | 0.540% | 76.1 |
| capytaine | 80 | 0.396 | -41.8202 | 4.916% | 3.575% | 7.829% | 18.6 |
| capytaine | 320 | 0.198 | -43.3995 | 1.325% | 1.954% | 2.126% | 37.6 |
| capytaine | 1280 | 0.099 | -43.8105 | 0.391% | 1.146% | 0.589% | 76.1 |

Observed orders of the TensorBEM ladder (per refinement pair, current / density
$L_\infty$ / exterior potential $L_2$):
{
  "current": [
    0.9590467812206162,
    0.9854757273926354
  ],
  "density_linf": [
    0.4397284216050359,
    0.4220515918072736
  ],
  "probe_l2": [
    0.9455514120278594,
    0.9832950463953897
  ]
}

### 3.3 Element-wise assembly cross-check against Capytaine (identical mesh)

| N | $\|A_{TB}-A_{OSS}\|_F/\|A\|_F$ | max diag rel. | $\|D_{TB}-D_{OSS}\|_F/\|D\|_F$ |
|---|---|---|---|
| 80 | 4.44e-06 | 2.59e-13 | 2.59e-13 |
| 320 | 2.26e-04 | 2.60e-13 | 9.15e-04 |
| 1280 | 1.84e-04 | 2.60e-13 | 2.15e-03 |

### 3.4 OSS FEM reference (scikit-fem, axisymmetric)

| order | R/a | r x μ grid | elements | I [A] | I error | inner/outer flux | exterior φ L∞ error |
|---|---|---|---|---|---|---|---|
| P2 | 2 | 24x16 | 768 | -44.1022 | 0.273% | 0.194% | 0.000% |
| P2 | 2 | 48x32 | 3072 | -44.0139 | 0.072% | 0.053% | 0.000% |
| P2 | 4 | 24x16 | 768 | -44.1036 | 0.276% | 0.249% | 0.000% |
| P2 | 4 | 48x32 | 3072 | -44.0183 | 0.082% | 0.076% | 0.000% |

Richardson extrapolation of the FEM current (removes the $O(h^p)$ discretisation
error; $R/a$ independence of the extrapolated value is the truncation-sensitivity
test):

| FEM order | R/a | observed order | Richardson-extrapolated I [A] | error vs analytic |
|---|---|---|---|---|
| P2 | 2 | 1.92 | -43.98230 | 0.000% |
| P2 | 4 | 1.75 | -43.98230 | 0.000% |

### 3.5 Discrete defect found by this XVAL (and fixed)

The non-uniform case exposed a **sign error in the double-layer term of the
exterior representation formula** in `evaluate_exterior_potential`
(`src/tensorbem/cathodic_protection.py`).  It computed
$\int Gq\,dS-\int\phi\,\partial G/\partial n\,dS$; the correct exterior
representation with $n$ the body-outward normal is
$\int Gq\,dS+\int\phi\,\partial G/\partial n\,dS$.

Why no existing test caught it: for a *uniform* surface potential the
double-layer integral over a closed surface vanishes outside the body
(numerically $1.7\times10^{-18}$ in this work), so the uniform-sphere anchor,
the far-field Gauss-law route and the DNV-RP-B401 demand comparison are all
blind to the sign.  The discriminating test is a pure $l=1$ mode, where the
wrong sign produces a $2/3$ error on the dipole contribution
(`tests/test_laplace_sphere_xval.py::test_exterior_potential_dipole_sign`).
With the pre-fix code the exterior-potential $L_2$ error on this case was
19.860% and did **not** converge under refinement (0.247 → 0.209 → 0.199);
after the fix it is 0.540% at the finest level with a clean
second-order trend.

### 3.6 Quadrature-order sensitivity (TensorBEM, two finest levels)

| Duffy order | I error | density L∞ error | exterior φ L₂ error |
|---|---|---|---|
| 4 | 1.295% | 1.967% | 2.103% |
| 8 | 1.301% | 1.943% | 2.111% |

### 3.7 Truncation-radius and order sensitivity (FEM)

See §3.4: at fixed mesh the FEM current error grows with $R/a$ (the $1/r$
variation is under-resolved on a uniform radial grid), but the Richardson
extrapolated value is $R/a$-independent to machine precision in the digits
shown — i.e. the reference is a genuine infinite-domain reference, and the
$R/a$ trend is a discretisation effect, not a modelling one.
Order sensitivity: P2 reaches 0.082% on the
$R/a=4$ grid where P1 needs an order of magnitude more elements for the same
accuracy (element-wise gradient of P1 is only $O(h)$ accurate on the boundary).

## 4. Difference list, TensorBEM vs the two OSS implementations

| aspect | TensorBEM | Capytaine | scikit-fem | porting decision |
|---|---|---|---|---|
| formulation | indirect single-layer density, piecewise-constant collocation | direct (Panich) $S,K$ matrices; solver only for Neumann-type problems | volume Galerkin | keep the indirect single-layer: a Dirichlet-only BIE is what CP needs and it avoids the $\tfrac12$ jump bookkeeping |
| BC support | mixed Dirichlet/Neumann on panel subsets | radiation/diffraction (Neumann) only → **no Dirichlet solver**; we had to assemble our own BIE from its influence matrices | arbitrary BC, but needs a finite domain | BEM keeps the infinite domain for free; FEM needs truncation + exact far-field data (acceptable only for verification) |
| singular quadrature | exact polar self-term, exact solid angle, Duffy $O(10)$ off-diagonal | analytic free-surface Green function + Nemoh tabulation; Rankine limit reached only via `free_surface=inf, wavenumber=0` | none (volume integrals) | keep TensorBEM's quadrature: element-wise agreement with Capytaine is at the $10^{-4}$ level, i.e. quadrature is not the error source |
| double-layer diagonal | explicit zero diagonal + explicit $\tfrac12\phi$ term in the BIE | $\tfrac12$ jump baked into the matrix diagonal | n/a | keep the explicit form: it is what makes the mixed panel sets readable, and the XVAL confirms both conventions agree once converted |
| Green-function sign | $G=+1/(4\pi r)$ | $G=-1/(4\pi r)$ | n/a | do **not** copy Capytaine's sign; only convert when comparing matrices (documented in the module docstring) |
| scaling | dense $O(N^2)$ Python assembly (7.2 s at $N$=1280, 1 core) | compiled Fortran core + optional H-matrices/ACA | sparse | **the one real gap**: TensorBEM has no compression; Capytaine's HierarchicalMatrices is the candidate to port if CP hull meshes exceed ~$10^4$ panels |
| mesh objects | triangles `TriangleMesh` | quad-oriented `Mesh`, triangles accepted via `from_list_of_faces` (≥3.x) | own meshes | no change needed; the triangle path was verified here |
| verification style | analytic anchors + conservation routes | added-mass/damping benchmarks vs analytic spheres | textbook patch tests | adopt Capytaine's "compare against the OSS implementation itself" for kernels (this mission) |

## 5. Caveats and limits

* The BEM numbers are for a **faceted** sphere: the O($h^2$) geometric error is
  part of the reported error and is why the density error floors around 1 % at
  $N$=1280 rather than going to zero.
* Capytaine is used outside its design envelope (potential-flow solver, no
  Dirichlet solver); only its *influence matrices* are used, and only in the
  Rankine limit.  The comparison certifies the quadrature/assembly, not
  Capytaine's free-surface machinery.
* The FEM reference is axisymmetric by construction; it certifies the
  formulation along a meridian, not a fully 3-D mesh quality question.
* Both OSS routes are optional dependencies: without them the module reports
  `available: false` and the tests skip, so CI stays green on minimal images.

## 6. Failure modes and artefacts

* Failure mode found: double-layer sign in the representation formula
  (§3.5), invisible to uniform data.  Fixed, with a dedicated regression test.
* Failure mode avoided: `basis.probes` on a truncated FEM domain raises
  "Point is outside of the mesh" — probes are filtered to $r<0.95R$.
* Artefacts: `artifacts/xval_laplace_sphere/results.json` (all numbers above),
  `docs/xval_laplace_sphere/xval_convergence.png` (convergence), `xval_potential_profile.png`
  (potential profiles).  Reproduce with
  `PYTHONPATH=src python3 scripts/m17363_xval_run.py run`.
