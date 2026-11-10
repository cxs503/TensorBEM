# Quadratic panel operator — verification artifacts

Verification record for the quadratic (6-node, isoparametric) panel operator of
`tensorbem.panel_quadratic` (exterior 3-D Laplace BEM, second-kind formulation
`(1/2 I - D) u = -S q`).

Everything in this directory is **generated** — never hand-edited:

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.quadratic_panel_verification
PYTHONPATH=src python -m pytest tests/test_panel_quadratic_operator.py -q
```

The default level set is `(0, 1, 2)` so the command above reproduces the
committed `study.json` bit-for-bit. The run writes:

| Artifact | Content |
|---|---|
| `study.json` | every number of every study + acceptance flags + sha256 of each figure |
| `fig_quadratic_operator.png` | convergence, operator identities, nodal flux deviation map |
| `fig_geometry_generic.png` | geometry-generic convergence (sphere, ellipsoid, sheared ellipsoid, non-affine wavy sphere) and the measured thin-body scope limit |
| `fig_non_affine_geometry_m9712.png` | non-affine geometry axis: flux convergence over the four cases and the quadrature-order invariance of the solid-angle residual (evidence figure of m9712) |

`study["provenance"]` holds the sha256 of each generated figure, so a figure and
the numbers that produced it cannot drift apart silently (the pytest suite
re-hashes them in `test_committed_study_json_matches_its_artifacts`).

## What is verified, against what reference

Exact field of an interior point source `x0` (a decaying harmonic function, i.e.
admissible for the exterior problem):

```
u(x) = 1 / (4 pi |x - x0|)
q(x) = -(x - x0) . n / (4 pi |x - x0|^3)      (exact normal derivative / flux)
```

All references are closed-form; nothing is compared against the solver itself.

1. **Exact-flux recovery** (unit sphere, `x0 = (0, 0, 0.5)`): the physical
   acceptance quantity; exercises `S`, `D` and the singular self-panel
   quadrature in one number (nodal values interpolated to element centroids).
2. **Operator identities**: `D 1 = -1/2` (solid angle of a closed surface),
   `(1/2 I - D) 1 = 1`, `S 1 = 1` on the unit sphere (single-layer potential of
   unit density equals the radius).
3. **Flat-panel self block**: a flat element has `(x - y) . n == 0`, so the
   double-layer self block must vanish identically.
4. **Geometry-generic study** (ellipsoid with three different semi-axes and a
   non-conformal *sheared* ellipsoid, source at the geometric centre): a
   sphere-specific cancellation cannot masquerade as accuracy; the observed order
   and the 3% acceptance must survive. The shear leaves the similarity group
   (`det A = 1`, so scale and source depth are unchanged) and makes the
   parametric mapping non-orthogonal, i.e. the elements genuinely skewed — the
   diagonal axis scalings of the ellipsoid do not. Each level records
   `jacobian_anisotropy_max = max(det J)/min(det J)` per element, so accuracy can
   be read against element quality and the case cannot silently degenerate into a
   rescaled conformal mesh.
   A fourth case is **not an affine image of the unit sphere** at all: the radius
   of the icosphere is modulated by `0.15 P_3(cos theta)` (`wavy_r3_e0p15`).
   Sphere, ellipsoid and sheared ellipsoid are all affine, and the quadratic
   Bezier parametrisation commutes with affine maps, so a Jacobian/normal slip
   that is linear-consistent is invisible to them; the nonlinear map is not
   compatible with the six control nodes to machine precision. Each level
   therefore records `affine_image_residual` (least-squares affine fit between
   corresponding node sets, normalised): round-off (`< 1e-12`) for the affine
   cases, `5.7e-2` at level 1 for the modulated one, which classifies the
   geometry axis instead of asserting it in prose.
5. **Similarity invariance**: translating and uniformly scaling the mesh
   (`x -> 2.7 x + shift`) must leave the *relative* flux error unchanged, since
   the exterior Laplace problem has no length scale.

## Results (from `study.json`, finest level = subdivisions 2, 642 nodes / 320 elements)

| Study | Quantity | Value |
|---|---|---|
| Unit sphere, point source | relative L2 flux error (finest) | **6.07e-03 (0.61 %)** |
| Unit sphere, point source | observed order (levels 0-2) | 2.49 |
| Operator identity `S 1 = 1` | max abs error (finest) | 7.22e-05 (order 4.13) |
| Operator identity `D 1 = -1/2` | max abs error (finest) | 9.61e-04 (order 2.93) |
| Ellipsoid 1.00 : 1.60 : 0.70 | relative L2 flux error (finest) | **6.43e-03 (0.64 %)** |
| Ellipsoid 1.00 : 1.60 : 0.70 | observed order (levels 0-2) | 2.63 |
| Similarity (scale 2.7 + shift) | deviation of the relative error | 2.4e-13 |
| Thin body 1.00 : 2.50 : 0.50 | relative L2 flux error (finest) | 3.55e-02 (3.55 %, **not gated**) |

Every solver run reaches `relative_residual_l2 < 1e-15`, i.e. the linear solve is
not the limiting factor; the reported errors are discretisation error.

## Acceptance gates

`run_study` records and `main()` enforces (exit code 1 on failure, `--allow-fail`
reports without failing):

* finest-mesh flux error on the unit sphere `<= 3 %` (hard limit 15 %);
* same `<= 3 %` on the non-spherical ellipsoid and observed order `>= 1.5`;
* same `<= 3 %` on the sheared (non-similarity affine) ellipsoid and observed
  order `>= 1.5`;
* same `<= 3 %` (and order `>= 1.5`) on the non-affine radially modulated
  sphere, with `affine_image_residual > 1e-2` so the case cannot be a disguised
  affine rescaling;
* similarity deviation `<= 1e-10`.

## Scope limits (honest, measured)

* **Thin bodies.** For a body whose local thickness is comparable to a panel
  size (`1.00 : 2.50 : 0.50`), the recovered flux error stalls at ~3.5 % with a
  degraded order (1.79) and never reaches the 3 % acceptance (`scope_limit`
  block, not gated). This is *field resolution*, not an operator defect: an
  admissible decaying harmonic field with a source inside a thin body is
  necessarily near-singular on the boundary, so no smooth interior-source test
  exists for such a body.
* **Near-singular panels are integrated with the fixed order-8 Gauss rule**
  (`quadrature_order`). Raising it to 32, or recursively subdividing the near
  panels, changed the `D 1` residual of the thin body only from 2.8e-2 to
  1.6e-2 — the residual there is dominated by accumulated cancellation between
  the two sides of the thin section, not by quadrature. The higher-order /
  adaptive machinery is therefore deliberately not enabled (it costs 3-5x
  assembly time for no measurable gain on the meshes in scope).
* **Field-sharpness coupling.** On a sphere the source can be placed at a fixed
  relative depth, so sphere tests are reproducible benchmarks; on a body with
  unequal semi-axes the same relative depth is much sharper somewhere else.
  The geometry study therefore puts the source at the geometric centre, and the
  point-source acceptance on the unit sphere keeps the deeper source.
  Consequence: an interior-point-source test on a strongly anisotropic body
  measures field sharpness and geometry together — do not read a single number
  from such a case as the operator's accuracy.
* **Not covered here.** Helmholtz / free-surface (frequency-domain) quadratic
  panels, and non-Laplace kernels, are out of scope for these artifacts.