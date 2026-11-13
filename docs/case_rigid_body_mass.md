# CASE-BEM-RIGID-BODY-MASS — 6×6 rigid-body mass matrix vs the closed-form block form

Anchor `CASE-BEM-RIGID-BODY-MASS` · repo TensorBEM · tolerance `3%` · generated from `runs/m17376_rigid_body_mass/results.json` by `tools/emit_case_rigid_body_mass_doc.py` (every number below is injected from the artifact, none typed by hand).

## 1. Case introduction (physics background, geometry, parameters)

Every frequency-domain seakeeping solve in TensorBEM builds the 6-DOF equation of
motion `(-w^2 (M + A(w)) + i w B(w) + C) xi = F_exc`. `M` is the **rigid-body mass
matrix**: three translations plus three rotations about a reference point at the
calm waterline, coupled through the centre of gravity. Its reference is the
closed-form block form

```
M_RB = [[ m I3 , -m S(r_g) ],
        [ m S(r_g) ,  I_g + m S(r_g) S(r_g)^T ]]
```

Source of the reference: Fossen (2011) Handbook of Marine Craft Hydrodynamics and Motion Control, Wiley, section 3.3.1 (rigid-body kinetics), eqs. (3.44)-(3.53); same block partition in Newman (1977) Marine Hydrodynamics section 4.5 and Journee & Massie (2001) Offshore Hydromechanics section 3.2.
Numbers: evaluated by this module from the cited closed forms and the case geometry below (zero hand-copy).

The case body is a barge with a deck payload (chosen so that all three CG
coordinates and the products of inertia are non-zero — otherwise the coupling
block is zero and the case has no teeth):

| component | length [m] | beam [m] | height [m] | centre [m] | density [kg/m³] | volume [m³] | mass [kg] |
|---|---|---|---|---|---|---|---|
| hull | 120 | 20 | 10 | (0, 0, -3) | 563.75 | 24000 | 1.353e+07 |
| payload | 20 | 10 | 4 | (30, 4, 4) | 1353 | 800 | 1.0824e+06 |

| derived quantity | value |
|---|---|
| total mass m | 14612400.000000 kg |
| centre of gravity r_g | (2.222222, 0.296296, -2.481481) m |
| displaced volume m/rho (rho = 1025 kg/m³) | 14256.0000 m³ |
| waterplane area L·B | 2400.0 m² |
| equilibrium draft (geometry check) | 5.9400 m (< hull depth 10.0 m, bottom at z = -8.0 m) |

The case has no Froude number, wavenumber or other dimensionless group: the
rigid-body mass matrix is a property of the mass distribution only, so the case
is dimensionless-free by construction (the only ratios that appear are geometry
ratios: L/B = 6, B/H = 2, payload/hull mass = 0.08).

## 2. How it was done (method, flow, tool chain)

1. **Route R (reference, analytic).** Solid-box second moments
   (`I_xx = m(b²+c²)/12`, …) plus the parallel-axis theorem are composed over the
   two components to obtain `m`, `r_g`, `I_g`; the block form above then gives the
   reference 6×6 matrix.
2. **Route M (independent geometry route).** The same body is triangulated as a
   closed surface and its mass properties are integrated by the divergence theorem
   (`V = (1/3)∮ r·n dS`, `∫_V r dV = (1/4)∮ r (r·n) dS`,
   `∫_V x_i x_j dV = (1/5)∮ x_i x_j (r·n) dS`) with a degree-3 triangle rule that is
   exact for these integrands. This is the geometric validation step (Cb/L/B style)
   required before any tolerance judgement.
3. **Unit under test (got).** `tensorbem.hydrostatic.compute_mass_matrix(m, r_g, I_g)`,
   the function the RAO solver, the ice-RAO path and the verification harnesses call.
   It is evaluated twice: from the route-R mass properties and from the route-M ones.
4. **Discrimination control.** The same comparison is replayed with the pre-fix
   coupling block (`-m S^T` in the upper-right position, i.e. the transposed
   coupling) — a matrix that is symmetric and positive definite, so it passes every
   symmetry/definiteness test.
5. **Physical identity.** `0.5 x^T M x` is compared with the rigid-body kinetic
   energy `0.5 m |v_o + w × r_g|² + 0.5 w^T I_g w` for 256 random velocity pairs.
6. **Main-solver integration.** `tensorbem.rao.compute_rao_at_frequency` is run on a
   barge cut at the draft implied by the mass model, once as shipped and once with
   the pre-fix block injected at the single call site that assembles `M`.

Environment: python 3.11.6, numpy 2.4.6, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38.

## 3. Results (ref vs got, entry by entry)

Reference matrix entries (kg or kg·m²) and the relative error of the shipped matrix;
criterion `|err| < 3%` per entry:

| mode | surge | sway | heave | roll | pitch | yaw |
|---|---|---|---|---|---|---|
| surge | +1.4612e+07 / 0.00e+00% (PASS) | 0 / 0 (PASS) | 0 / 0 (PASS) | 0 / 0 (PASS) | -3.6260e+07 / 0.00e+00% (PASS) | -4.3296e+06 / 0.00e+00% (PASS) |
| sway | 0 / 0 (PASS) | +1.4612e+07 / 0.00e+00% (PASS) | 0 / 0 (PASS) | +3.6260e+07 / 0.00e+00% (PASS) | 0 / 0 (PASS) | +3.2472e+07 / 0.00e+00% (PASS) |
| heave | 0 / 0 (PASS) | 0 / 0 (PASS) | +1.4612e+07 / 0.00e+00% (PASS) | +4.3296e+06 / 0.00e+00% (PASS) | -3.2472e+07 / 0.00e+00% (PASS) | 0 / 0 (PASS) |
| roll | 0 / 0 (PASS) | +3.6260e+07 / 0.00e+00% (PASS) | +4.3296e+06 / 0.00e+00% (PASS) | +7.3062e+08 / 0.00e+00% (PASS) | -1.2989e+08 / 0.00e+00% (PASS) | -1.2989e+08 / 0.00e+00% (PASS) |
| pitch | -3.6260e+07 / 0.00e+00% (PASS) | 0 / 0 (PASS) | -3.2472e+07 / 0.00e+00% (PASS) | -1.2989e+08 / 0.00e+00% (PASS) | +1.7500e+10 / 0.00e+00% (PASS) | -1.7318e+07 / 0.00e+00% (PASS) |
| yaw | -4.3296e+06 / 0.00e+00% (PASS) | +3.2472e+07 / 0.00e+00% (PASS) | 0 / 0 (PASS) | -1.2989e+08 / 0.00e+00% (PASS) | -1.7318e+07 / 0.00e+00% (PASS) | +1.7724e+10 / 0.00e+00% (PASS) |

- max entry error (shipped, reference mass properties): **0.000e+00**
- max entry error (shipped, *mesh-derived* mass properties): **1.291e-15**
- max entry error (pre-fix control): **2.000e+00**
- kinetic-energy identity, shipped: **4.143e-16**; pre-fix control: **2.177e-01**
- geometry agreement (route R vs route M): mass 0.00e+00 rel, volume 0.00e+00 rel, r_g 4.44e-16 m abs, I_g 6.75e-18 rel

A worked entry: `M[surge, pitch] = -m·S(r_g)[0,1] = +m·z_g`, and with
m = 1.46124e+07 kg, z_g = -2.481481 m this is -3.62604e+07 kg·m — the
number the shipped code returns. The pre-fix code returned its negative.

## 4. Figures

`runs/m17376_rigid_body_mass/rigid_body_mass_error.png` — left: the 6×6 relative-error
map of the shipped matrix (log scale, all entries at round-off); right: the CG-offset
ladder, showing the shipped error staying at round-off while the pre-fix coupling
block stays at its sign-flip error (200 % = exactly `2·|m S|/|m S|`) as `x_g` grows.

## 5. Convergence and sensitivity

**Sensitivity A — mesh refinement.** The body is polyhedral, so the divergence-
theorem route is exact at every refinement; the observed behaviour is the expected
*zero* order (nothing to converge):

| subdivisions | panels | max entry error | control max entry error | energy identity error |
|---|---|---|---|---|
| 1 | 24 | 0.000e+00 | 2.000e+00 | 4.143e-16 |
| 2 | 96 | 0.000e+00 | 2.000e+00 | 4.143e-16 |
| 4 | 384 | 0.000e+00 | 2.000e+00 | 4.143e-16 |

**Sensitivity B — CG station (payload fore/aft).** Sliding the payload moves `x_g`
from 0 to about 4 m, growing the coupling block by a factor
1.912 while the shipped error stays at round-off: the case is not passing on a vanishing
coupling term.

| payload station [m] | x_g [m] | ‖M[0:3,3:6]‖ | shipped max err | control max err |
|---|---|---|---|---|
| 0.0 | 0.0000 | 5.1644e+07 | 0.000e+00 | 2.000e+00 |
| 5.0 | 0.3704 | 5.2208e+07 | 0.000e+00 | 2.000e+00 |
| 15.0 | 1.1111 | 5.6518e+07 | 0.000e+00 | 2.000e+00 |
| 30.0 | 2.2222 | 6.9109e+07 | 0.000e+00 | 2.000e+00 |
| 45.0 | 3.3333 | 8.6093e+07 | 0.000e+00 | 2.000e+00 |
| 55.0 | 4.0741 | 9.8769e+07 | 0.000e+00 | 2.000e+00 |

**Sensitivity D — the platform referee configuration** (solid ellipsoid
a, b, c = [60.0, 12.0, 8.0] m, rho = 1025.0 kg/m³, r_g = [5.0, 0.0, -3.0] m),
which is the configuration recorded in `tensor_benchmark_cases` id=57 for this
anchor. The mesh route here is an independent signed-tetrahedron quadrature, so
the ladder can be compared with that row's published numbers:

| panels | volume error | I_g error | max entry error | control max entry error |
|---|---|---|---|---|
| 20 | 39.4539% | 56.1884% | 55.6477% | 160.5% |
| 80 | 12.6547% | 20.1245% | 19.8832% | 187.3% |
| 320 | 3.3839% | 5.5712% | 5.5005% | 196.6% |
| 1280 | 0.8606% | 1.4299% | 1.4115% | 199.1% |
| 5120 | 0.2161% | 0.3599% | 0.3552% | 199.8% |

At 5120 panels this gives 0.2161 % volume error and 0.3552 % maximum entry error,
and at 320 panels 5.5005 % — the same numbers the referee row publishes
(0.2161 %, 0.3552 %, 5.50 %), reproduced independently from the row text. Note the
residual is *geometry*: an inscribed flat-faceted ellipsoid under-integrates volume
and inertia, so the error falls with panel count while the assembly itself is exact.

The discrimination control sits at 160-200 % on the same ladder: the referee row's
per-entry metric would therefore have failed the pre-fix platform function at every
panel count, even though the diagonal-only numbers were unchanged.

**Sensitivity C — effect on the delivered RAO (main solver).**

| mode | \|RAO\| shipped | \|RAO\| with pre-fix block | change |
|---|---|---|---|
| surge | 0.208546 | 0.320777 | 53.82% |
| sway | 0.0292147 | 0.0199252 | 31.80% |
| heave | 0.425196 | 0.424566 | 0.15% |
| roll | 0.00360765 | 0.00698781 | 93.69% |
| pitch | 0.025013 | 0.0251089 | 0.38% |
| yaw | 0.000594915 | 0.000565211 | 4.99% |

Largest change 93.69% in roll (omega = 0.6 rad/s, 108 wetted panels). Replaying the same
solver with the reference matrix injected at the call site reproduces the shipped
RAO to 0.0e+00, which pins the caller to this matrix.

## 6. Failure modes, caveats, applicability

- **What the defect was.** `compute_mass_matrix` returned `-m S^T` in the upper-right
  block and `-m S` in the lower-left, i.e. the coupling of a body whose CG sits at
  `-r_g`. The matrix stayed symmetric and positive definite, so the existing
  symmetry/definiteness tests and the in-repo "closed form" comparison (which had
  been transcribed with the same transposition, making it a tautology) all passed.
  Only the kinetic-energy identity and this case expose it. Impact: for a barge with
  z_g = -2.481 m the surge–pitch and sway–roll coupling changed sign; the RAO
  numbers moved by up to 93.7% (roll) at the case frequency.
- **Applicability.** The mass matrix is exact for any rigid body given `m`, `r_g`,
  `I_g`; it does not depend on the waterline, the mesh, or the wave frequency. The
  case therefore validates the *assembly*, not the hydrodynamics — the added-mass
  side is covered by the separate added-mass anchors.
- **`inertia` convention.** The argument is the tensor **about the centre of
  gravity**; the parallel-axis term `m S S^T` is added internally. Passing an
  origin-referenced tensor double counts it. The docstrings of
  `compute_mass_matrix`, `rao.compute_rao_at_frequency` and the in-repo
  `run_mass_matrix_check` said "about the origin" before this mission and have been
  corrected.
- **Remaining gap.** `compute_mass_matrix(..., inertia=None)` still falls back to a
  heuristic radius of gyration `k = 0.4·|r_g|`; no anchor covers that fallback and it
  is *not* exercised by this case (all numbers here pass `I_g` explicitly). Treat it
  as a convenience path, not a validated one.
- **Honest scope of the tolerance.** The entry errors are at round-off, i.e. the
  criterion `< 3%` is met by a margin of 14 orders of magnitude. That is expected for
  a closed-form-versus-closed-form comparison and is why the case carries the mesh
  route and the negative control: they are what make the round-off number meaningful.

**Verdict:** shipped matrix within `< 3%` of the reference on all 36 entries → `True`; pre-fix control inside tolerance → `False`.

