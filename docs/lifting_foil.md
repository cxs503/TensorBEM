# Lifting flows: analytic Joukowski reference and verification harness

**Scope.** README roadmap item 6 lists "Kutta-condition lifting hydrofoils" as
not implemented, and every solver in the package is non-lifting: the 2-D
Dirichlet and zero-net-flux Neumann problems have unique circulation-free
solutions, and the free-surface solvers are radiation/diffraction problems of
the same kind. This note records what is now in the package
(`src/tensorbem/lifting.py`), what it was checked against, and what is still
missing.

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

## 4. What is NOT shipped: the Kutta closure (honest negative result)

The intended closure was exact superposition: represent the disturbance as
`phi_U + Gamma phi_v`, where `phi_v` is the response to an interior unit vortex
(a point vortex of unit strength plus the exterior-Neumann correction that
cancels its normal velocity, which is a zero-net-flux problem), and fix `Gamma`
from the equal-and-opposite tangential velocity at the two panels adjacent to
the trailing edge. A linear solve, no iteration.

Measured status at `alpha = 4 deg` (exact `Gamma = 0.9642`):

| panels | \|Gamma\| | sign | C_L from C_p |
|---|---|---|---|
| 32 | 0.8618 (10.6% low) | inverted | +0.886 |
| 128 | 0.9422 (2.3% low) | inverted | +0.936 |
| 256 | 0.9531 (1.2% low) | inverted | +0.944 |

The magnitude converges, but the returned circulation has the **opposite sign**
to the exact Kutta value, and the pressure integral of the composite field
(`+0.94`) disagrees with its own circulation lift, so the closure as implemented
is not physical. Diagnostics that isolate the cause (recorded here so a follow-up
does not repeat them):

* The base (non-lifting) solve is correct: boundary values match `phi_d` to
  `+/- 2e-3`, and surface speeds match the exact field to ~1-2% away from the
  leading edge.
* The unit-vortex response has the correct *total* circulation (`+0.955` against
  `+1`), but its tangential velocity has the wrong sign at mid-chord
  (`+0.081` against `-0.053` at panel 32), so the error is in the response
  field, not in the free stream or in the base solve.
* The vortex potential must be added to the correction potential *and* its
  branch cut placed along the wake (through the trailing edge), otherwise the
  along-contour potential difference picks up a spurious jump. Both are already
  done in the diagnostic code, and both were necessary but not sufficient.

Shipping a solver that returns a circulation of the wrong sign would have been
worse than shipping nothing, so `solve_lifting_flow` is **not** in the module.
Recommended next step: keep the exact superposition idea but compute the surface
velocity as the *limit from outside* using the layer gradients (single-layer
gradient analytically, double-layer gradient by quadrature on the offset
surface) instead of differencing the boundary potential; that removes both the
piecewise-constant derivative noise and the branch-cut bookkeeping.

## 4b. Status update (m2573): the Kutta closure IS now shipped

The superposition route recommended at the end of section 4 was not the one that
worked. The shipped lifting solve is an independent linear-strength vortex panel
method in `src/tensorbem/lifting_solver.py` (`gamma_TE = 0` imposed as
`gamma_N = -gamma_0`), which reproduces the exact Joukowski circulation to
0.40 % at 256 panels and 0.73 % at 128 panels; see `docs/lifting_kutta.md` for
the tables, the error budget and the two formulations that were rejected
(constant-strength source/vortex assembly: 6.9 % lift error at 128 panels).
The analytic reference and the non-lifting harness in this module are unchanged
and remain the reference the solver is checked against.

## 5. Limitations

* Symmetric (zero-camber) Joukowski sections only; no camber, no flap, no
  rounded or blunt trailing edge.
* Analytic reference plus a verification harness: no lifting boundary-element
  solve, so no lift can be computed for an arbitrary hull appendage yet.
* Steady, inviscid, incompressible, infinite domain: no free surface, no
  forward speed, no viscous boundary layer, no separation.
* `plots` in `docs/m2480/outputs/lifting_foil/` are generated by the benchmark
  module and contain no hand-entered data.
