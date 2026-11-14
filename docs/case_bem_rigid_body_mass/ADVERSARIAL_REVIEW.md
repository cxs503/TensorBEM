# CASE-BEM-RIGID-BODY-MASS - independent (adversarial) review

Scope: the committed evidence bundle of the case-pool anchor
`BEM-RIGID-BODY-MASS` (`src/tensorbem/benchmarks/casepool_bem_rigid_body_mass.py`,
`docs/case_bem_rigid_body_mass/{results,verdict}.json`, `CASE.md`) plus the
library it verifies, `src/tensorbem/rigid_body_mass.py`.

## 1. What was checked, and how

| probe | route | independence |
| --- | --- | --- |
| A | `M_RB` as the Hessian of the kinetic energy, symbolically (SymPy) | does not read the block form or the momentum map |
| B | quadrature vs solid-ellipsoid closed form (Lamb 1932) | the reference contains no panelisation |
| C | topology of the quadrature input, pre/post guard | builds corrupted meshes from scratch |
| D | response of the entry-relative metric to injected errors | metric is exercised with *known* errors |

## 2. Results

* **A (symbolic).** `Hessian(T) - block_form` is the zero matrix: `True` (0 non-zero entries). The shipped assembly agrees with the symbolic Hessian to 0.00e+00 on a scale of 1.85e+10 (round-off).
* **B (convergence).** Worst 6x6 entry error at subdivision 4 (5120 panels) is 0.36 % (entry `pitch-pitch`), decreasing with a fitted log-log slope of -2.82. The anchor accepts subdivisions >= 3; the worst entry there is 1.43 %, which is panelisation error of a converging sequence, not an assembly bias, and it is below the 3 % tolerance of the feature ledger.
* **C (topology).** This is the material finding: before the guard, the quadrature route accepted
  any input whose signed volume was positive, while returning volumes wrong by open_patch 0.95 %, flipped_triangle 1.00 %, duplicated_triangle 0.79 % (all inside the anchor's 3 % tolerance); an inward-oriented surface was the only case the
  pre-fix rule caught (negative volume).
  A tolerance comparison can therefore never see the defect: the corrupted reference stays
  plausible. The remediation shipped with this review is `mesh_topology_report()` plus a hard
  guard in `mesh_mass_properties()` (open edges, non-manifold edges, flipped-edge pairs,
  with exact/coincident-vertex welding), tested by `tests/test_rigid_body_mass_topology.py`.
* **D (metric).** The reported value equals the injected absolute error divided by the
  entry magnitude, `eps*scale/max(|M_ij|, 1e-6*scale)`: for every injected case (`roll-roll` diagonal entry/scale 0.07, `surge-pitch` coupling entry/scale 0.0039) the reported and predicted values agree to a relative residual of 2.0e-14. The metric therefore neither dilutes
  an error on a small coupling entry (it amplifies it, e.g. 2.58e-02 for a 1e-04 x scale injection) nor hides one on a large diagonal entry.

## 3. Findings, by severity

1. **High (fixed here).** `mesh_mass_properties` integrated any mesh whose signed volume was
   positive, including open and inconsistent surfaces, and reported no diagnostic. The case
   treats this function as its independent reference, so a broken input would have produced
   a *plausible* verification. Fix: structural guard, 13 new tests, one of which keeps the
   pre-fix wrong volume as a control.
2. **Low (accepted).** The energy-Hessian route of probe A is symmetric by construction, so it
   cannot detect an antisymmetric corruption; the shipped momentum route (route P) already
   covers that case, and the case document states this split of duties explicitly.
3. **Informational.** The acceptance metric is entry-relative with a floor of 1e-6 of the
   matrix scale. Structurally zero entries (e.g. the `sway-roll` coupling when the CG lies in
   the x-z plane) must match to round-off; that is stricter than a global-norm metric, not
   laxer (probe D).

## 4. Checks not run / limitations

* No comparison against an external BEM code was re-run in this audit (capytaine is used by the
  anchor's orientation case, not re-executed here); the audit verifies the *internal* evidence
  chain, not the external one.
* The welding tolerance path uses a KD-tree neighbour search; meshes whose seam nodes are
  separated by more than the requested tolerance are still rejected (by design).
* An inward-oriented but topologically consistent surface is rejected by the pre-existing
  non-positive-volume check, not by the new edge-incidence guard (both are needed).
* Curved higher-order panels are outside the quadrature route's model: it is exact per
  tetrahedron, so only the panelisation of the surface contributes error (probe B).

## 5. Disposition

**Approve with notes.** The anchor's `<3 %` claim survives independent derivation (A), converges
under refinement (B) and is not an artefact of the metric (D). The one material defect found
(C) concerned the validity domain of the reference route and is fixed and regression-tested in
this change set.

## 6. Addendum — the end-to-end claim was not in the verdict (fixed)

This review verified the *closure* route (block form vs quadrature vs symbols) and
the two controls. It did not check that the anchor's **machine** verdict covered
the strongest claim it makes, namely that the seakeeping solver consumes this
matrix. It did not: `verdict.json` listed eight checks, none of them the RAO
caller replay, whose numbers sat unused in `results.json["base_case"]["rao_caller"]`.
A regression at the call site would therefore have left the anchor `PASS`.

Remediation shipped with this change set:

* `casepool_bem_rigid_body_mass.rao_caller_section` promotes the replay to a
  top-level `rao_caller` section with two checks — *reproduction* (injecting the
  closed block form at the solver's single `compute_mass_matrix` call site must
  reproduce the shipped RAO, `< 1e-9`) and *discriminating power* (the pre-fix
  coupling convention must move the delivered RAO by more than the 3 % anchor
  tolerance; measured 93.69 % in roll);
* the driver reports `INCOMPLETE` (never `PASS`) when run with `--no-rao`, and
  lists the unrun checks in `verdict["not_run"]`;
* `CASE.md` §4 now tabulates the replay per mode and §5 states what the zero
  replay residual does and does not prove (the shipped matrix *is* the injected
  form, so the non-triviality comes from the legacy-coupling control);
* the dangling cross-reference in §5 ("见 ... RAO caller 回放一节", a section that
  did not exist in this directory) points at the material that now exists;
* `tests/test_casepool_bem_rigid_body_mass.py` gates the ledger (9 tests): the
  helper's three-state behaviour, the committed artefacts' invariants, the
  premise that the shipped call-site matrix equals the injected form, and the
  `INCOMPLETE`/`PASS` driver semantics.

Residual limitation, unchanged: the probe pins the *call site*, not the
agreement of the block convention with an external BEM code (that remains
`docs/case_rigid_body_mass.md` §Sensitivity C and the capytaine cross-assembly
of the orientation case).
