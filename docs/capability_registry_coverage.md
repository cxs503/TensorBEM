# Capability-registry coverage: the half of the package the baseline was silent about

Mission **m6280** (TensorBEM, verification-debt wave). Companion artefacts:
`docs/m6280/capability_coverage.json` and `docs/m6280/capability_coverage.md`,
both written by `python -m tensorbem.benchmarks.capability_coverage_study`
(no number in this document is transcribed by hand — the generated table is the
one to read for figures).

## 1. The debt, stated precisely

`docs/audit_baseline.md` §4 publishes a capability table. Since mission m3336
that table is *derived*: `tensorbem.capability_evidence` resolves each row's
modules, graded symbols, test node ids and documentation anchors, and grades the
row `REAL` / `MOCK` / `STUB` / `WITHHELD` from that evidence. The gate is
fail-closed: a stale claim, a stub body, randomness or a constant return all go
red, and the published block must equal the generated block.

That machinery has a blind spot, and it is the same class of error the m308
baseline made when it called finite-depth kernels `NOT_SUPPORTED` long after
they shipped. A row can only be wrong about what it *claims*. A capability with
**no row at all** is not wrong, it is **absent**, and to a reader of the table
absence reads as "nothing to see". Measured on the tree at the start of this
mission: of the 54 core modules under `src/tensorbem/` (excluding the
`tensorbem.benchmarks` drivers), **27 — exactly half — appeared in no row**,
while every one of them shipped with a test suite and a document:

| shipped and silent | what it is |
|---|---|
| `drift`, `drift_force`, `maruo_drift` | second-order mean drift force, near and far field |
| `time_domain` | Cummins radiation memory kernel and free decay |
| `uq`, `recovery_uncertainty` | Richardson/GCI, bootstrap intervals, recovery bands |
| `hydroelastic` | panel modes, wet/dry frequency ratio |
| `ice`, `ice_rao`, `pc6_hull` | ice-cover mass loading, ice-region RAO, ice-class hull |
| `panel_linear`, `panel_smoothing` | order-1 panel operators, tangential recovery smoothing |
| `multiloop`, `multicomponent` | multiply connected 2-D domains |
| `thin_ship_michell`, `ellipsoid_kochin`, `multihull` | general-hullform and tandem thin-ship theory |
| `quadrature_diagnostics`, `quadrature_diagnostics_2d`, `diagnostics_report` | near-singular / conditioning diagnostics |
| `geometry`, `geometry3d` | panel and triangle meshes, closure checks |
| `free_surface_solver(3d)`, `finite_depth_diffraction`, `finite_depth3d_quadrature` | the solvers *behind* rows that named only their benchmark driver |

## 2. What changed

1. **Ten new rows** grade those capabilities: `second-order-drift`,
   `time-domain-memory-kernel`, `uncertainty-quantification`,
   `hydroelastic-dry-modes`, `ice-region-hydrodynamics`,
   `linear-panel-operators`, `multiply-connected-domains`,
   `thin-ship-general-hullforms`, `quadrature-diagnostics`,
   `hull-mesh-geometry`. Each carries modules, graded symbols, named test nodes
   and a documentation anchor, so each is graded by the same fail-closed rule as
   the older rows.
2. **Four existing rows were widened to name the solver they actually describe**
   (`free_surface_solver` under the Ursell row, `free_surface_solver3d` under the
   3-D free-surface row, `finite_depth_diffraction` under the 2-D finite-depth
   row, `finite_depth3d_quadrature` under the 3-D finite-depth row). A row that
   names only its benchmark driver grades the driver, not the physics.
3. **A coverage ratchet**, `capability_evidence.capability_coverage()`. Every
   core module must be named by a row, or carry an entry in
   `AUXILIARY_MODULES` with a reason (there is exactly one: the shared
   closed-form reference library `tensorbem.analytic`). A newly shipped module
   that no row grades now fails `tests/test_capability_evidence.py`, and a
   reason that has become obsolete fails too. The count is rendered into the
   published block, so the baseline now states its own scope instead of leaving
   it implicit.

## 3. Quadrature diagnostics

`tensorbem.quadrature_diagnostics`, `tensorbem.quadrature_diagnostics_2d` and
`tensorbem.diagnostics_report` classify source/target pairs by
`distance / panel size` and assemble the combined report. The row is graded on
three properties that do not depend on the classifier being *reliable*, only on
it being *deterministic and complete*: a self-interaction is always flagged, the
threshold is monotone, and the report round-trips through JSON and Markdown.
These are diagnostics: they say where the quadrature is uncomfortable, they do
not certify accuracy, and no row claims otherwise.

## 4. Live anchors, and what they do not show

Each new row's numerics live in its own tests (`tests/test_time_domain.py`,
`tests/test_ice_mass_loading.py`, `tests/test_panel_linear_order.py`, …), and
`src/tensorbem/benchmarks/capability_coverage_study.py` re-runs one cheap,
independent anchor per row so the association is not merely static:

* icosphere area/volume against `4 pi R^2` and `4/3 pi R^3` over four refinement
  levels (`hull-mesh-geometry`);
* the piecewise-linear rectangular self integral, closed form against its own
  graded Duffy rule (`linear-panel-operators`);
* an exact exponential memory kernel transformed both ways, recovering
  `B(omega) = sum_j c_j a_j / (a_j^2 + omega^2)` (`time-domain-memory-kernel`);
* empirical coverage of the 95 % bootstrap interval for the mean
  (`uncertainty-quantification`).

Three limits are worth stating plainly, because a coverage table invites more
confidence than it earns:

1. **Coverage is not correctness.** "Graded" means a row resolves code, tests
   and documents; the numerical claim of a row rests on its named tests, which
   are analytic references, consistency identities or, where stated, honest
   `WITHHELD`.
2. **Two of the new rows are model-internal.** `hydroelastic-dry-modes` compares
   against the Kirchhoff plate closed form and an exact uniform added-mass limit;
   `docs/bem_hydroelastic.md` registers absolute validation against an
   independent hydroelastic code or experiment as `WITHHELD`, and the row note
   says so. `ice-region-hydrodynamics` is pinned to Weitz-Keller, to independent
   RK4 shooting and to the open-water limit, but the PC6 hull is parametric, so
   absolute RAO magnitudes in ice stay `WITHHELD` in `docs/ice_water_ship.md`.
3. **`thin-ship-general-hullforms` is an asymptotics statement.** The agreement
   is with Michell theory (thin ship), not with measured resistance; the
   experimental anchor for these hullforms does not exist in this work tree.

## 5. Reproduce

```bash
PYTHONPATH=src python -m tensorbem.capability_evidence            # the table
PYTHONPATH=src python -m tensorbem.capability_evidence --write    # refresh §4
PYTHONPATH=src python -m tensorbem.benchmarks.capability_coverage_study \
    --output docs/m6280
PYTHONPATH=src python -m pytest tests/test_capability_evidence.py -q
```