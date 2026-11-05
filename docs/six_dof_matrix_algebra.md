# Six-degree-of-freedom radiation-matrix algebra: reciprocity, reflection selection rules and reference-centre covariance

Measurement: `python -m tensorbem.benchmarks.six_dof_matrix_algebra --write` | Python 3.11.6, numpy 2.4.6 | ka = 1.0, density = 1025 kg/m^3

## 1. What is checked, and why it was missing

`tensorbem.free_surface_solver3d.solve_radiation_3d` returns the **full** `(6, 6)`
added-mass and damping matrices of a floating body,

```
A_ij + (i/omega) B_ij = -rho * int_{Gamma_B} phi_j n_i dS ,
```

where `phi_j` is the radiation potential of mode `j`. Every other verification in
this repository reads **diagonal** entries only (Hulme's `A11`/`A33` for the
hemisphere, Lamb's added masses for the ellipsoid). The diagonals can be right
while the moment arms, the `i`/`j` order or the sign of the rotational normals are
wrong, and nothing in the suite would notice. Four predictions close that gap.
The first three are theorems about the continuous problem; the fourth is an exact
algebraic identity that must hold at roundoff.

1. **Reciprocity.** `A_ij = A_ji` and `B_ij = B_ji`. In this solver the two are
   different discrete integrals (`phi_j n_i` versus `phi_i n_j`, over different
   radiation solutions), so the check tests the assembled operators.
2. **Reflection selection rule.** The hull is invariant under `S_x: x -> -x` and
   `S_y: y -> -y`. The generalized normal velocity of each mode is even or odd
   under each reflection, with the parity table below; because the free-surface
   Green function and the waterplane are invariant too, the change of variables
   `x -> R x` gives `A_ij = eps_i eps_j A_ij`. Hence `A_ij = B_ij = 0` whenever
   `eps_i eps_j = -1`: 26 of the 30 off-diagonal entries.
3. **Spin-axis identity.** For a body of revolution about the vertical axis the
   azimuthal normal velocity satisfies `(z_hat x r) . n = 0` on the hull, because
   the normal of a surface of revolution has no azimuthal component. Therefore
   `phi_yaw == 0` and `A_66 = B_66 = 0` exactly (this is also the `a = b` limit of
   Lamb's 1932 spheroid formula, s.114).
4. **Reference-centre covariance.** Moving the reference point by `d` maps the
   generalized velocities by `u' = M u`, `M = [[I, -[d]_x], [0, I]]`, and added
   mass is a tensor of that pair of vector spaces, so the same fields must obey
   `A' = M^{-T} A M^{-1}` (and likewise for `B`) for **any** body, wavenumber and
   mesh. This is the only check here whose tolerance is machine precision.

Parity table `(eps(S_x), eps(S_y))` -- derived from `v(Rx) = R v(x)` for polar
vectors and `w -> -R w` for the axial rotation vector, and re-derived numerically
from the assembled normals by the test suite:

| mode | surge | sway | heave | roll | pitch | yaw |
|---|---|---|---|---|---|---|
| `eps(S_x)` | -1 | +1 | +1 | +1 | -1 | -1 |
| `eps(S_y)` | +1 | -1 | +1 | -1 | +1 | -1 |

The four survivors of the reflection rule are the diagonal and the two
*same-parity* off-diagonal pairs, `surge-pitch` and `sway-roll`. They are not
asserted to vanish: see section 5.

## 2. Case

Hulme's floating hemisphere (unit radius, waterline exactly on `z = 0`,
`rho = 1025 kg/m^3`, `g = 9.80665 m/s^2`, wavenumber `nu = k = 1.0 1/m`,
`ka = 1`). Its panel counts are `4 * 4**subdivisions`, the same meshes that back
`docs/marine_bem_benchmarks.md` and the Hulme tables. The centre of rotation is
the waterplane centre `(0, 0, 0)`.

## 3. Measured

| subdiv | panels | h | cond | residual | max|A-A^T|/|A| | max|B-B^T|/|B| | forbidden A | forbidden B | A66/A33 | B66/B33 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 0.8069 | 3.89 | 5.1e-03 | 5.610e-04 | 2.502e-04 | 3.565e-05 | 2.538e-05 | 3.817e-03 | 1.919e-07 |
| 2 | 64 | 0.4322 | 3.05 | 1.2e-03 | 2.041e-05 | 1.267e-04 | 3.385e-06 | 2.494e-06 | 5.302e-04 | 1.246e-08 |
| 3 | 256 | 0.2202 | 2.78 | 1.9e-04 | 3.140e-07 | 1.264e-05 | 2.692e-07 | 1.904e-07 | 6.169e-05 | 8.140e-10 |

| metric | observed order per refinement step |
|---|---|
| `added_mass_block.reciprocity_rel` | n/a, 5.31, 6.19 |
| `damping_block.reciprocity_rel` | n/a, 1.09, 3.42 |
| `added_mass_block.forbidden_rel` | n/a, 3.77, 3.75 |
| `damping_block.forbidden_rel` | n/a, 3.72, 3.81 |
| `spin_axis.added_mass_rel` | n/a, 3.16, 3.19 |
| `spin_axis.damping_rel` | n/a, 4.38, 4.05 |

| permitted coupling | observed order per refinement step |
|---|---|
| `surge-pitch` (symmetric part) | n/a, 1.57, 1.90 |
| `sway-roll` (symmetric part) | n/a, 1.57, 1.90 |

| centre-shift covariance (offset = [0.3, 0.0, 0.25]) | relative residual |
|---|---|
| added mass, 64 panels | 9.72e-17 |
| damping, 64 panels | 1.10e-16 |

`max|A-A^T|/|A|` and `max|B-B^T|/|B|` are normalised by `max|diag|`; `forbidden`
is the largest of the 26 entries the reflection rule kills, normalised the same
way; `A66/A33`, `B66/B33` are the spin-axis identity; `h` is
`sqrt(2 * total area / N)`.

## 4. Results

**Reciprocity holds and tightens.** Added mass is the cleaner of the two:
`3.14e-07` at 256 panels, falling faster than `h**2` (measured
5.3, 6.2). Damping converges too, but with a non-monotone order
(1.1, 3.4) -- the imaginary part is what the wave part of the kernel and
the CHIEF rows act on, so it is the honest number to quote.

**The reflection rule is verified, not by construction.** The twenty forbidden
entries fall from `3.57e-05` to `2.7e-07` (order
3.8, 3.8) and the worst pair is reported per mesh, so a regression names
the coupling that broke. Note the mesh itself is mirror-symmetric to `1e-12`
`1e-12` the parity table is checked against the assembled normals before the
selection rule is applied, so the zeros cannot be explained by a mesh that was
never symmetric.

**The spin-axis identity is approached at third order.** `A66/A33` goes
`3.8e-03 -> 5.3e-04 -> 6.2e-05`; `B66/B33` goes `1.9e-07 -> 1.2e-08 -> 8.1e-10`, i.e. four to five orders below
the damping scale at 256 panels. This is the only check with an *analytical
value* (not merely a symmetry) attached to an off-diagonal-free slot of the
matrix.

**The centre-shift identity is exact.** Residuals of `9.7e-17` (added mass) and
`1.1e-16` (damping) are roundoff. Two negative controls in the test suite (wrong
lever-arm sign, transposed matrix) fail the same check by more than three orders
of magnitude, so the check has teeth rather than passing on a diagonal matrix.

![Decay of the four structural errors with panel size, and the structure of the
measured added-mass matrix at 256 panels](m3630/six_dof_matrix_algebra.png)

Left: every structural residual is normalised by its own diagonal scale and drawn
against `h = sqrt(2A/N)`; the `h**2` and `h**3` guides bracket the measured
orders. Right: `|A_ij| / max|diag A|` at 256 panels on a logarithmic colour scale;
the 26 entries the reflection rule kills are marked and sit at `<= 2.7e-07`,
while the two same-parity pairs (`surge-pitch`, `sway-roll`) are the only visible
off-diagonal entries.

## 5. What is *not* claimed

* **Surge-pitch and sway-roll are not asserted to vanish.** They have
  `eps_i eps_j = +1`, so no reflection forbids them; reciprocity makes them
  *symmetric* (`|antisymmetric part| <= 0.2 |symmetric part|` on every mesh),
  whereas a Timman-Newman reading of the transverse mirror plane would make them
  antisymmetric and hence exactly zero. That relation does not follow from the
  two reflections used here, and the measured magnitude decays like `h**1.57, 1.90`
  towards zero: the data cannot separate "exactly zero, residual is
  discretisation error" from "physical value below the discretisation error".
  The benchmark therefore reports the magnitude and the order. For users of the
  RAO chain the practical statement is the useful one: on a doubly symmetric hull
  the coupling terms are `3.4e-02` of the diagonal at 16 panels and
  `3.6e-03` at 256, the same order as the diagonal's own error against
  Hulme's solution.
* One body, one wavenumber, deep water. The reflection rule would need a third
  mirror plane (or a half-body reduction) to say anything about a hull that is
  fore-aft asymmetric; the covariance identity is body-independent and is the
  check to run for those.
* `B` reciprocity is stated as measured, not as a theorem-derived asymptotic
  order: the 16-to-64-panel step gives order 1.1, the next
  3.4. No order is pinned for it.

## 6. Reproduction

```
python -m tensorbem.benchmarks.six_dof_matrix_algebra --write
python docs/m3630/build_deliverables.py     # regenerates this note and the figure
pytest tests/test_six_dof_matrix_algebra.py
```

The note, the pinned metrics in `tests/data/` and the figure all come from the
same run; `tests/test_six_dof_matrix_algebra.py` re-runs the study, compares the
table above cell by cell with the recomputation, and re-derives the parity table
from the assembled normals.
