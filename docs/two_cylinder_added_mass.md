# Mutual added mass of two circular cylinders (m4721)

Verification case: **the first reference in this repository with mutual
(body-to-body) hydrodynamic coefficients**, and the first that exercises the
exterior Neumann operator on two disjoint boundary loops.

* reference: `tensorbem.analytic.two_cylinder_added_mass` — truncated multipole
  series, no boundary-element code involved;
* numerics: `tensorbem.benchmarks.two_cylinder_added_mass` — constant-panel
  midpoint collocation on a `MultiLoopMesh` whose two loops are both holes;
* tests: `tests/test_two_cylinder_added_mass.py`;
* figures and machine-readable digest, committed next to this page:
  `docs/m4721/refinement.png`, `docs/m4721/field_and_boundary.png`,
  `docs/m4721/interaction.png`, `docs/m4721/figures_digest.json`
  (regenerated into `artifacts/two_cylinder_added_mass/` by
  `docs/m4721/figures.py`; that directory is git-ignored).

Reproduce everything with

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.two_cylinder_added_mass \
    --panels 48 64 96 --separation-ratio 3.0 \
    --json artifacts/two_cylinder_added_mass/benchmark.json
PYTHONPATH=src python3 docs/m4721/figures.py   # writes artifacts/, copied to docs/m4721/
PYTHONPATH=src python3 -m pytest tests/test_two_cylinder_added_mass.py -q
```

## 1. Physical setting

Two equal circular cylinders of radius `a` with axes `d` apart, both immersed in
an inviscid, incompressible fluid at rest at infinity. Per unit span, the
`4x4` added-mass matrix in the modes `(body0 surge, body0 heave, body1 surge,
body1 heave)` is wanted, with body 0 at `(-d/2, 0)` and body 1 at `(+d/2, 0)`.
No free surface and no incident wave are involved: this is the classical
radiation problem of two bodies in an unbounded fluid, the interaction
counterpart of the single-cylinder `rho*pi*a**2` case.

## 2. Reference: truncated multipole series (independent of the BEM)

With `z = x + i*y`, the potential

```
Psi(z) = sum_{k=0,1} sum_{n=1..N} m_{k,n} (z - c_k)^{-n},    phi = Re[Psi]
```

spans all decaying harmonic fields outside the two circles (each body carries
its own singularities; the constant and the logarithm are excluded, so `phi`
decays at infinity). The body condition on circle `j` is `dphi/dr = U_j . e_r`,
and because every term is analytic the radial derivative is closed-form,

```
dphi/dr = Re[ Psi'(z) exp(i*theta) ],   Psi'(z) = -sum n m (z - c)^{-n-1}.
```

The coefficients follow from a real least-squares fit of that Fourier content
on `2N+8` equally spaced collocation points per circle; the added mass is
`A_ij = -rho * closed_integral(phi_j n_i ds)` evaluated with the trapezoidal
rule on 512 points per circle (spectrally accurate for these smooth boundary
functions). `N = 24` is the default.

Internal checks of the reference, all asserted in the test file:

| check | result |
|---|---|
| body-condition residual on an independent, finer angle set (`N=24`) | `3.7e-11` |
| truncation drift, `N = 16` vs `24` vs `32` | `< 1e-8` relative |
| reflection symmetry of the `4x4` matrix (`d/a = 3, 5, 10, 40`) | `< 1e-9` of `rho*pi*a**2` |
| surge-heave blocks and reciprocity | `< 1e-9` of `rho*pi*a**2` |
| far-field limit, `d/a = 40` | self / `rho*pi*a**2` = 1 + 7.8e-7 |
| interaction law, `d/a = 40` | `A12_yy * d**2 / (2*rho*pi*a**2) = 1.0000004` |

Two structural facts the series reproduces to machine precision and which no
single-body reference can express: the mutual coefficient is **negative** in
line with the centres and **positive** across them, with equal magnitude, and
the in-phase heave added mass exceeds the isolated-cylinder value (catamaran
blockage).

## 3. Numerics

`S` and `H` are the single- and double-layer matrices of
`tensorbem.solver.assemble_matrices`; the case introduces no new kernel. Both
circles are marked as holes in the `MultiLoopMesh`, which flips the
domain-outward normals to point into the bodies and therefore flips the sign of
`H`; the assembled system is `0.5*I + H`, whose condition number is 2.2-2.5
(`H (1 on one body) = +1/2` there, so `0.5*I - H` would be singular). The
added mass is `A_ij = -rho * sum(phi_j n_i ds)` with the body-outward normal.

Mesh refinement at `d/a = 3`, `N = 24` reference:

| panels per body | max relative error | self | mutual (in line) | mutual (transverse) |
|---|---|---|---|---|
| 48 | 2.20e-3 | 3322.54 | -728.90 | 728.90 |
| 64 | 1.24e-3 | 3322.72 | -729.60 | 729.60 |
| 96 | 5.51e-4 | 3322.84 | -730.10 | 730.10 |
| reference | — | 3322.94 | -730.51 | 730.51 |

(All entries in units of mass per unit span, `rho = 1025`, `a = 1`. The
`solver`-side numbers are in `artifacts/two_cylinder_added_mass/refinement.png`
and `figures_digest.json`; the reference column is
`two_cylinder_added_mass(1.0, 3.0, 1025.0).added_mass_matrix`.)

Field level, not just integrated coefficients: at 64 panels per body the BEM
boundary potential for `body1_surge` matches the series pointwise with a
relative `L2` error of `8.3e-4` on body 0 and `8.4e-4` on body 1.

## 4. Injected-error control

`--inject layer_sign` assembles `0.5*I - H`, the sign that belongs to normals
pointing into the fluid. On this mesh it is wrong: the mutual coefficients
collapse and the self coefficient shifts, so the gates in
`benchmarks/two_cylinder_added_mass.run` fire:

| quantity | healthy (48 panels) | injected |
|---|---|---|
| self relative error | 1.2e-4 | 3.6e-2 |
| mutual (in line) relative error | 2.2e-3 | 9.94e-1 |
| mutual (transverse) relative error | 2.2e-3 | 9.99e-1 |

The self coefficient alone would barely notice the mistake at coarse meshes;
the interaction term is what makes the gate decisive, which is precisely the
coverage this case adds.

A second, reference-side injection was run by hand for the report: deleting the
leading minus sign in `A_ij = -rho * closed_integral(phi n ds)` (inside
`two_cylinder_added_mass`) turns five tests red --
`test_far_field_limit_recovers_the_single_body_added_mass`,
`test_blockage_raises_the_in_phase_heave_added_mass`,
`test_mode_potential_reproduces_the_added_mass_entry`,
`test_bem_matches_the_multipole_reference`,
`test_bem_converges_towards_the_reference` (`5 failed, 9 passed`). The file was
reverted immediately; the injected variant is not part of the tree.

## 5. Limitations

* Two-dimensional, unbounded fluid, **no free surface**: this verifies the
  exterior Neumann operator and the interaction algebra, not wave radiation or
  ship hydrodynamics.
* Non-touching bodies only (`d > 2a`); the truncated multipole fit degrades as
  the gap shrinks (residual `3.7e-8` at `d/a = 2.5` versus `3.7e-11` at
  `d/a = 3`), so the case is registered at `d/a >= 3`.
* Constant panels with midpoint collocation: the reported convergence is the
  observed ladder of this case, not a claim of a formal order.
* The reference is a truncated series: its truncation drift is asserted
  (`< 1e-8`), and it is validated by the far-field and decay laws above, but it
  is not a closed-form expression.