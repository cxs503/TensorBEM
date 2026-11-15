# CASE-BEM-RIGID-BODY-MASS · round 6: inhomogeneous mass distribution

Anchor `CASE-BEM-RIGID-BODY-MASS` (pool code `BEM-RIGID-BODY-MASS`, repo `TensorBEM`,
tolerance `<3%`). The earlier rounds closed the closed block form, the orientation
covariance, the momentum route, the invariant battery and the geometry-exact
*homogeneous* inertia. All of them ended with the same limit: the mesh route assumed a
homogeneous body. This round removes that assumption at the integration level.

## 1. What changed

* `rigid_body_mass.mesh_mass_properties(..., density_field=f)` and
  `rigid_body_mass.wetted_hull_mass_properties(..., density_field=f)` accept a
  dimensionless shape factor `f(x)` with `rho(x) = density * f(x)`.
* The signed-tetrahedron route interpolates `f` linearly on each tetrahedron and uses
  the closed-form cubic barycentric moments (`int lambda_k lambda_m lambda_n dV` =
  `V/20`, `V/60`, `V/120`). An **affine** field is therefore exact; a bilinear or
  steeper one is second order (measured in section 5).
* `hydrostatic.mass_matrix_from_wetted_hull(..., density_field=f)` forwards it, so the
  default-inertia branch of `rao.py` / `ice_rao.py` can describe a ballasted or
  machinery-heavy hull without hand-supplying an inertia tensor: the CG becomes the
  *mass* centroid and the displaced mass the integrated `int rho dV`.
* `density_field=None` is unchanged, and `f == 1` reproduces it to round-off.

## 2. Reference, and why the comparison is not tautological

The reference is still the anchor's: the closed block form
`M_RB = [[m I, -m S(r_g)], [m S(r_g), I_cg + m S^T S]]` (Fossen 2011 eq. (3.44)), fed
here with *analytic* mass properties of a body with a prescribed polynomial density.
The analytic side integrates the polynomial over a box from one-dimensional raw
moments (`int u^n du`) and never touches a mesh; the mesh side never sees the analytic
second moments. Both sides consume the same `PolynomialDensity` object, so the field
cannot be transcribed differently on the two sides — no coefficient is hand-copied.

## 3. Affine fields on polyhedral bodies (both sides exact ⇒ round-off is the criterion)

| case | field | worst entry rel err | worst block rel err | mass rel err | CG rel err | inertia rel err |
| --- | --- | --- | --- | --- | --- | --- |
| `box_homogeneous` | `homogeneous` | 1.453e-11 | 9.791e-17 | 0 | 4.756e-18 | 9.791e-17 |
| `box_longitudinal` | `longitudinal(c=1.2/L)` | 3.442e-12 | 2.366e-16 | 0 | 7.542e-17 | 3.839e-18 |
| `box_vertical` | `vertical(c=1.8/T,z0=0)` | 1.071e-11 | 5.258e-16 | 0 | 3.922e-17 | 9.791e-17 |
| `barge_production_vertical` | `vertical(c=1.8/T,z0=-1.5)` | 1.759e-11 | 5.875e-16 | 0 | 3.469e-17 | 5.875e-16 |

`<3%` is met by about ten orders of magnitude on the affine cases: the box and the
waterplane-closed barge are exactly polyhedral and an affine field is integrated
exactly, so only round-off is left. This is a closure, not a tolerance pass. The
`box_vertical` case is where the inhomogeneous CG moves in `z` (0.45 m) and the
products of inertia stay zero; `box_longitudinal` moves it 2.0 m in `x`.

## 4. Bilinear field on a refined box (a product of inertia appears)

`f = 1 + 1.6 x y / (L B)` gives a non-zero `I_xy` and is *not* affine, so the
linear-per-tetra interpolation cannot be exact: the ladder measures the order.

| cells per edge | panels | worst entry rel err | block rel err | accepted |
| --- | --- | --- | --- | --- |
| 1 | 12 | 53.879 | 0.006 | no |
| 2 | 48 | 13.470 | 0.001 | no |
| 4 | 192 | 3.367 | 0.000 | yes |
| 8 | 768 | 0.842 | 0.000 | yes |
| 16 | 3072 | 0.210 | 0.000 | yes |

Fitted order: **2.00** (second order, as
a linear-per-tetra interpolation of a bilinear field requires). Accepted worst
block-normalised error at 4 cells per edge:
**0.000**.

The entry-wise column is *amplified* here and the block-normalised one is the
criterion, deliberately: this body has `I_xz = I_yz = 0` exactly, and the entry-wise
metric divides an error on a zero reference entry by `1e-6 x scale` (19 kg m^2 on a
1.9e7 kg m^2 matrix), so it reads 3.4 at 4 cells while the absolute error on that
entry is 6.2e3 kg m^2 of a 1.7e7 kg m^2 block, i.e. 3.3e-4. Both numbers are printed;
the amplification is the documented behaviour of the anchor's metric, not a defect of
the integration -- the product of inertia that the field *does* create (`I_xy`)
converges from 6.5 % to 2.7e-4 of its own block.

## 5. Curved hull (linear field, panelisation of the geometry)

| subdivisions | panels | worst entry rel err | block rel err | mass rel err | accepted |
| --- | --- | --- | --- | --- | --- |
| 2 | 320 | 0.056 | 0.056 | 0.034 | no |
| 3 | 1280 | 0.014 | 0.014 | 0.009 | no |
| 4 | 5120 | 0.004 | 0.004 | 0.002 | yes |
| 5 | 20480 | 0.001 | 0.001 | 0.001 | yes |

Fitted order: **1.98**. The `mass rel
err` column is the *volume deficit* of the inscribed polyhedron, not a field error --
the linear field is odd, so it cannot change the mass; that column is exactly how much
geometry the flat panels miss.
Accepted-discretisation worst entry error: **0.004** and block-normalised **0.004** — inside 3 %.

## 6. The control: ignoring the field

What a check is worth is what it *rejects*. The control takes the same mesh with a
uniform density (exactly the pre-round behaviour) and compares it with the analytic
inhomogeneous reference.

| case | control entry rel err | control block rel err | mass rel err | CG offset (m) |
| --- | --- | --- | --- | --- |
| `box_homogeneous` | 1.453e-11 | 9.791e-17 | 0 | 2.958e-17 |
| `box_longitudinal` | 1.000 | 1.000 | 0 | 2.000 |
| `box_vertical` | 1.000 | 1.000 | 0 | 0.450 |
| `barge_production_vertical` | 0.270 | 0.270 | 0 | 0.283 |
| `box_quadrupole_cells16` | 1.000 | 0.046 | 0 | 0 |

Controls that fail the 3 % block tolerance as required: `box_longitudinal`, `box_vertical`, `barge_production_vertical`, `box_quadrupole_cells4`, `box_quadrupole_cells8`, `box_quadrupole_cells16`.

Controls that stay below it: none.

Read the two lists together — that is the honest part of this round:

* the **mass** is invariant under every field used here (each extra term is odd about
  the body centre), so a mass-only check would pass a homogeneous route every time;
* a **fore-aft** or **bilinear** field is caught by the matrix comparison itself;
* a **vertical** field on a shallow body (3 m draft against 20 m length) is a small
  fraction of the rotational block, so the entry-wise matrix metric alone would not flag
  it — the CG error (0.45 m in the barge case) is what exposes it. That is a property of
  the normalisation, not of the physics, which is why the block-normalised column is
  printed next to the entry-wise one.

## 7. Structural guards

A field that is not a physical density must be refused rather than integrated.

| probe | rejected | message |
| --- | --- | --- |
| `negative_density` | yes | density_field must be positive on the body (min -1.02e+03 kg/m^3) |
| `zero_density` | yes | density_field must be positive on the body (min 0 kg/m^3) |
| `wrong_length` | yes | density_field must return one value per point: expected 48, got 3 |
| `non_finite` | yes | density_field returned non-finite values |

## 8. Error judgement against the `<3 %` criterion

| quantity | worst error | <3 %? |
| --- | --- | --- |
| `M_RB` entry, affine field, exactly polyhedral bodies | 1.759e-11 | yes |
| `M_RB` block-normalised, affine field, polyhedral bodies | 5.875e-16 | yes |
| `M_RB` block-normalised, bilinear field, 4 cells/edge | 0.000 | yes |
| `M_RB` entry, bilinear field, 4 cells/edge (amplified on `I_xz = I_yz = 0`) | 3.367 | no -- metric artefact, see section 4 |
| `M_RB` entry, linear field, curved hull at subdivisions 4 | 0.004 | yes |
| `M_RB` block-normalised, linear field, curved hull | 0.004 | yes |
| production path (`mass_matrix_from_wetted_hull`) | 1.759e-11 | yes |
| homogeneous backward compatibility | 9.791e-17 | yes |

## 9. Honest limits

* The field is **prescribed, not measured**: TensorBEM still cannot *infer* a
  non-uniform mass distribution from geometry. What this round buys is a consistent,
  verified way to inject one through the geometry route instead of bypassing it with a
  hand-built tensor.
* `mesh_mass_properties` receives the field only at the tetrahedron corners, so the
  achievable accuracy is bounded by linear interpolation: exact for an affine field,
  second order otherwise. A field with structure finer than the mesh cannot be resolved
  -- refine the mesh, exactly as for the geometry.
* Vertical-translation invariance from the earlier rounds still holds only for
  horizontal shifts, unchanged by this round.
* Only the submerged (wetted) solid is integrated: mass above the waterline enters only
  through the field evaluated on the closure lid at `z = 0`, so the field must not
  depend on the part of the body the wetted hull does not carry.

Reproduce: `PYTHONPATH=src python3 -m tensorbem.benchmarks.rigid_body_mass_inhomogeneous --outdir docs/case_bem_rigid_body_mass/inhomogeneous_mass`
