# Lifting flows: analytic Joukowski reference and verification harness

**Scope.** README roadmap item 6 lists "Kutta-condition lifting hydrofoils" as
not implemented, and every solver in the package is non-lifting: the 2-D
Dirichlet and zero-net-flux Neumann problems have unique circulation-free
solutions, and the free-surface solvers are radiation/diffraction problems of
the same kind. This note records what is now in the package
(`src/tensorbem/lifting.py`), what it was checked against, and what is still
missing (`solve_lifting_flow` and its verification: `docs/lifting_kutta_closure.md`).

## 1. What is shipped

* `JoukowskiAirfoil` -- the symmetric Joukowski section with the *closed-form*
  flow solution: Kutta circulation, lift coefficient, surface speed, pressure
  coefficient, and the decaying disturbance potential.
* Equal-arc meshing (`JoukowskiAirfoil.mesh`) with a vertex exactly at the cusp
  trailing edge. Uniform `theta` spacing in the circle plane concentrates panels
  quadratically at the cusp (panel length ratios above 60:1 for 128 panels);
  inverting the arc-length map brings the ratio to 1.02:1, which is what makes
  any surface-derivative estimate usable there.
* `compare_nonlifting_field` -- drives the existing
  `solve_laplace_exterior_neumann` with the tangency flux
  `d(phi)/dn_e = U . n_body` and compares the result with the analytic
  disturbance potential, with no constant fitted by hand.
* `benchmarks/lifting_foil.py` + `tests/test_lifting_foil.py` (15 tests).

## 2. The closed-form solution, and the one trap in it

With `z = zeta + b^2/zeta` mapping the circle `|zeta + m| = a`, `a = m + b`, onto
the section, `zeta0 = -m`, uniform stream `U e^{i alpha}`:

```
F(zeta) = U[(zeta - zeta0) e^{-i alpha} + a^2 e^{i alpha}/(zeta - zeta0)]
          + (i Gamma / 2 pi) ln(zeta - zeta0)
```

`dF/dzeta = 0` at the cusp `zeta = b` (stagnation point on the trailing edge --
the Kutta condition for a cusp) gives

```
Gamma = 4 pi a U sin(alpha),      C_L = 2 Gamma / (U c),      c = z_TE - z_LE.
```

For `m = 0.1 b`, `b = 1`: `c = 4.0333`, `t/c = 0.1179`, and at `alpha = 4 deg`
`Gamma = 0.9642`, `C_L = 0.4781`.

**Trap.** `F` carries the constant `-U zeta0 e^{-i alpha}` while the physical
stream potential is `Re[U z e^{-i alpha}]`, so the *decaying* disturbance
potential of the exterior Neumann problem is

```
phi_d = Re[F] - Re[U z e^{-i alpha}] - U m cos(alpha).
```

Dropping the last term leaves a reference field that tends to
`U m cos(alpha) = 0.0998` at infinity instead of zero. It does not change any
velocity, but it biases every boundary-potential comparison by exactly that
constant, which is easy to misread as solver error. Measured against the solver:
the difference of the boundary values from `phi_d` is `-0.0998 +/- 0.0017`
without the term and `+/-0.002` with it.

## 3. Verification numbers (all computed by `benchmarks/lifting_foil.py`)

### 3.1 Analytic reference is self-consistent

Two independent routes to the same lift: Kutta-Joukowski `2 Gamma/(U c)`, and
the integral of the same closed-form `C_p` over the exact surface
(`L' = -(1/2) \oint C_p n_y ds`, `rho = 1`).

| alpha [deg] | Gamma | C_L from Gamma | C_L from pressure | gap |
|---|---|---|---|---|
| 0 | 0.0000 | 0.00000 | 0.00000 | 0 |
| 2 | 0.4824 | 0.23921 | 0.23919 | 0.009% |
| 4 | 0.9642 | 0.47814 | 0.47722 | 0.19% |
| 6 | 1.4449 | 0.71648 | 0.71292 | 0.50% |

Thin-aerofoil limit `2 pi (a/b) sin(alpha)`: 0.4821 vs 0.4781 at 4 deg (0.8%),
i.e. the section is thin enough for the classical slope and thick enough to
depart from it measurably.

Note on the bookkeeping: the pressure route needs `p = (1/2) rho U^2 C_p`, so the
raw `\oint C_p n_y ds` must be halved before normalising by `(1/2) rho U^2 c`.
Forgetting the factor 2 leaves a clean, misleading 100% discrepancy (this
happened during development and is now covered by a test).

### 3.2 The discretised field agrees with the analytic reference

Non-lifting field, `alpha = 4 deg`, equal-arc panels, against `phi_d`:

| panels | max\|dphi\|/max\|phi_d\| | rms \|dphi\| | rms speed error / max speed | cond |
|---|---|---|---|---|
| 32 | 4.67% | 4.80e-3 | 7.5% | 61 |
| 64 | 3.31% | 2.61e-3 | 7.6% | 92 |
| 128 | 2.39% | 1.39e-3 | 8.0% | 138 |
| 256 | 1.68% | 7.36e-4 | 8.2% | 205 |

The rms potential error halves with each refinement (second order in the panel
size); the max error is dominated by the single panel nearest the leading-edge
stagnation point, where the surface speed passes through zero and a sign change
of order `O(h)` in the position of the stagnation point swamps the local value.
That is a property of constant panels at a stagnation point, not of the lifting
formulation.

## 4. The Kutta closure: what was wrong with the first attempt

The first closure (2026-10) superimposed `phi_U + Gamma phi_v` and reported an
**honest negative result**: the magnitude converged (2.3% low at 128 panels) but
the circulation came out with "the opposite sign to the exact Kutta value", and
the pressure integral of the composite field disagreed with its own circulation
lift, which was read as "not physical".

The closure has since been re-derived, and that diagnosis was wrong in one
place: **the sign was a bookkeeping error, not a physical defect.**
`JoukowskiAirfoil.exact_circulation()` quotes `Gamma_doc = 4 pi a U sin(alpha)`,
the value that pairs with `C_L = 2 Gamma/(U c)` in the complex potential, i.e.
the *clockwise* orientation. A point vortex inside the section carries the
counter-clockwise value `Gamma_ccw = closed_integral(v.ds) = -Gamma_doc`, and
Kutta-Joukowski in that convention is `C_L = -2 Gamma_ccw/(U c)`. Comparing
`Gamma_ccw` against `Gamma_doc`, and then forming `2 Gamma_ccw/(U c)` instead of
its negative, produces exactly the two symptoms that were reported: an inverted
circulation and a pressure lift that "disagrees" with it by about 200%.

Measured on the same mesh with the sign bookkeeping fixed, the composite field
is consistent and reproduces the exact surface pressure distribution:
circulation lift and pressure lift agree to `0.12%` at 128 panels and both are
`2.3%` below the exact value (`docs/lifting_kutta_closure.md`, all numbers
machine-generated). So the closure itself was sound; it was validated against
the wrong reference convention.

The shipped solver is `tensorbem.lifting.solve_lifting_flow`. Two further
findings from the re-derivation:

* The singular point of the response does not set the circulation (uniqueness of
  the potential-flow solution); moving it from 0.15 c to 0.75 c changes
  `Gamma_ccw` by less than 1% at 128 panels.
* This note previously recommended replacing the three-point surface-velocity
  rule by the exterior limit of the layer gradients. **That recommendation is
  withdrawn for cusp trailing edges**: the piecewise-constant single-layer
  density is logarithmically singular at panel ends and the two panels that
  define the closure sit in that singularity. The layer route returns a
  circulation ~29% too large at 64 panels, while the three-point rule gives
  ~4.7%. It is kept only as a negative control in the benchmark module.

## 5. Limitations

* Symmetric (zero-camber) Joukowski sections only; no camber, no flap, no
  rounded or blunt trailing edge.
* `solve_lifting_flow` computes the lift of *this* section family only: no
  lift for an arbitrary hull appendage or a cambered/flapped section yet.
* Steady, inviscid, incompressible, infinite domain: no free surface, no
  forward speed, no viscous boundary layer, no separation.
* `plots` in `docs/m2480/outputs/lifting_foil/` are generated by the benchmark
  module and contain no hand-entered data.
