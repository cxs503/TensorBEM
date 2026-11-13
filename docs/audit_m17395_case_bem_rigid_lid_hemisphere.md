# Audit m17395 — verdict for m17338 `CASE-BEM-RIGID-LID-HEMISPHERE`

Anchor: `CASE-BEM-RIGID-LID-HEMISPHERE` (TensorBEM, pool case, mission m17338, `succeeded`).
This is a **cross-check of the delivered evidence**, not a redo. Every number below was
produced by a command in this work tree; nothing is transcribed from the mission's report.

## 0. Disposition

**approve — level L4 (benchmark validation).** The three-piece evidence exists, the referee
value is confirmed by two routes written independently of the audited code, the committed
"got" numbers reproduce bit-for-bit (modulo wall-clock fields), and the signed 1024-panel
headline of the referee row reproduces (`0.833304`, `+0.2831 %`). Two non-blocking
documentation findings are recorded in §5; neither changes the level.

## 1. What was audited

| item | pointer |
|---|---|
| mission commit | `2630991` in `/root/tp-agent/work/m17338` (parent `ac57fda`) |
| case driver | `src/tensorbem/benchmarks/case_bem_rigid_lid_hemisphere.py` (1146 lines; reference `:194-341`, solver rows `:359-406`, verdict `:561`) |
| artifacts | `benchmarks/verified/case_bem_rigid_lid_hemisphere/{results.json (602 lines), report.md (91 lines), case.json, figures/case_bem_rigid_lid_hemisphere.png}` |
| docs | `docs/case_bem_rigid_lid_hemisphere.md`, `docs/rigid_lid_zero_frequency_anchor/ERRATUM.md` |
| tests | `tests/test_case_bem_rigid_lid_hemisphere.py` |
| repro | `scripts/repro_case_bem_rigid_lid_hemisphere.sh` |
| referee row | `tensor_benchmark_cases` id 55, tolerance `<5% (pool) ; ecosystem <3%` |

## 2. Reference — independent re-derivation (audit route R1)

I re-derived the rigid-lid endpoint from the BVP statement
(`Laplace`, `dphi/dz = 0` on `z = 0`, `dphi/dn = U mu` on the wetted hemisphere) **without
importing the case module**. Even extension about `z = 0` gives a full sphere with
`dphi/dr = -U|mu|` (the mirror hemisphere moves *downward*, i.e. the two halves separate);
projecting on the hemisphere gives, with `J_n = int_0^1 mu P_n dmu` (odd `n` vanish),

```
A33/(rho*nabla) = 3 * sum_{n even} (2n+1) J_n^2 / (n+1)
```

| truncation `degree_max` | 20 | 40 | 60 | 80 | 120 | 240 |
|---|---|---|---|---|---|---|
| `A33/(rho*nabla)` | 0.8309491285 | 0.8309511611 | 0.8309512873 | 0.8309513097 | 0.8309513183 | **0.8309513203** |

* audit value at `degree_max = 240`: **0.8309513203**
* referee row's exact digits `0.830951320` → agreement **4.0e-10** relative
* case's shipped `reference.value_dimless` `0.8309512873` → difference `3.97e-08` relative
  (that is just its `N = 60` truncation; its own ladder shows the `N = 80` value `0.83095131`)
* analytic control in the same code: pressure-release branch `0.5*(3 J_1)^2` = **0.500000000** exactly

**This is the decisive confirmation of the reference value.** The `1.0` "translating image"
reading is not the solution of the stated BVP, because the rigid-lid image of a heaving
hemisphere is the mirrored hemisphere moving *away* (`U -> -U`), so the double-body
boundary condition is `dphi/dr = -U|mu|` (fluid pushed out at both poles, no flux across the
waterplane) and not `-U mu` (a translating sphere, which gives `0.5`). Both endpoints are
reproduced by the audit: `0.8309513203` (rigid lid) and `0.5` (pressure release).

## 3. Reference — independent numerics (audit route R2)

Collocation single-layer BEM on the wetted panels with the rigid-lid Green function
`G = 1/|x-y| + 1/|x-y*|` (`y*` = mirror in `z = 0`; same sign = rigid lid, opposite sign =
pressure release), `A33 = -rho * int_S phi n_z dS`. The free-term sign and the quadrature
were fixed on two **closed-form** controls before the audited quantity was read:

| control | 64 panels | 256 panels | 1024 panels | exact |
|---|---|---|---|---|
| full sphere, unbounded fluid (`A33/(rho*nabla_hemi)`) | 0.958311 | 0.998668 | 1.004513 | 1.0 |
| pressure-release hemisphere | 0.476477 | 0.498700 | 0.502103 | 0.5 |
| **rigid-lid hemisphere (audited)** | **0.828828** | **0.838789** | **0.837228** | ref 0.8309513 |
| deviation from the case reference | −0.255 % | +0.943 % | +0.755 % | — |

Both controls converge to their exact values (deficit tracks the mesh-area deficit
`−4.87 % / −1.26 % / −0.32 %`), and the rigid-lid endpoint sits within **±1 %** of
`0.8309513` on every mesh — never near `1.0` and never near `0.5`. This is an *external*
check in the sense that it shares no code, no series and no discretisation with the case.

## 4. "Got" numbers — reproduction

Referee row (`tensor_benchmark_cases` id 55): got `0.833304` at 1024 panels / `ka = 1e-3`,
tolerance `<5 %` (pool) and `<3 %` (ecosystem).

| evidence | value | err vs ref `0.830951320` | inside 5 % | inside 3 % |
|---|---|---|---|---|
| committed 64 panels, `ka = 1e-3` | 0.807388 | −2.836 % | yes | yes |
| committed 256 panels, `ka = 3e-2` (worst band point) | 0.864358 | +4.020 % | yes | **no** |
| committed 256 panels, `ka = 1e-3` | 0.828152 | −0.337 % | yes | yes |
| row headline 1024 panels, `ka = 1e-3` — **re-run by this audit** | **0.833304** | **+0.2831 %** | yes | yes |

* the 1024-panel re-run reproduces the signed headline to `5.0e-08` relative
  (`artifacts/m17395/row_headline_1024.json`, 93 s solver time, same commit);
* `bash -n` on the repro script passes; step 1 of the repro script (driver with `--check`)
  was re-executed into `artifacts/m17395/repro_artifacts/` and the regenerated
  `results.json` is **identical** to the committed one except the `solve_seconds` fields
  (`diff` shows only timing keys) — determinism confirmed;
* §3b out-of-band sweep (`ka = 0.2 … 1.0`, error up to −48.8 % vs the endpoint) is present
  and is the right kind of scope control, not a defect;
* negative control `G5 = 0.864358 < 0.95` (every band value below `0.95 rho*nabla`) refutes
  the translating-image reference — consistent with §2.

## 5. Findings (adversarial review)

**F1 — medium, documentation (does not change the level).** The committed artifact ladder
stops at 256 panels, where the worst band error is **4.02 %** — *outside* the ecosystem
`<3 %` line. The `<3 %` claim rests on the 1024-panel run, which is **absent from m17338's
artifacts** (the case itself documents "1024-panel level is not part of the predeclared
ladder"). I reproduced it (0.833304, +0.2831 %, 93 s), so the number is real, but it is not
m17338's own committed evidence. *Remediation:* commit the 1024-panel row (or a
`post_ladder` block with its own provenance) so the `<3 %` statement is artifact-backed.
Evidence: `results.json` `levels[].rows`; `report.md:88`; `case.json` friction.

**F2 — low, documentation.** `report.md` §5 answers the pool question ("PASS, tolerance 5 %")
but never states explicitly whether the key quantity is `<3 %`; the reader has to infer it
from §3. The six required sections exist, so this is a completeness nit, not a missing
section. Evidence: `report.md:76-91`.

**F3 — low, evidence independence (self-disclosed).** The "Hulme (1982)" corroboration is an
**in-repo implementation** (`tensorbem.analytic.hulme_hemisphere_radiation`), not a
transcription of Hulme's printed table; `reference_provenance.page_or_article_number`
discloses exactly this. So the mission's two reference routes and its corroboration are all
in-repo code. This audit supplies the missing external legs: R1 (independent derivation,
4.0e-10 agreement) and R2 (independent BEM, ±1 %). With those, the anchor is sound.

**F4 — low, predeclared gate history (handled correctly).** The first geometry gate
(2 %/2 % on every mesh) **failed** (−4.87 % area / −8.86 % volume at 64 panels) and was
restated to "case mesh 2 %/3 % + refinement shrink < 0.5" instead of being deleted. The
residual deficit is carried in §2 of the report, and the coefficient is normalised by the
*exact* `nabla`, so it does not leak into `A33`. Acceptable; noted so the failure is not
mistaken for a passed gate.

**Checks not run / limits of this audit.** The 1024-panel level was reproduced at one `ka`
only (`1e-3`, the row's headline) — the full 1024-panel band was not swept. The audit's R2
BEM is uniformly first-order-ish (0.26 %–0.94 % spread across 64/256/1024 panels), so it
confirms the endpoint to ~1 %, not to 1e-4; it cannot arbitrate the case's third decimal.
No EFD/experimental data is available for this quantity (inviscid linearised potential flow,
`ka -> 0`), so "external" here means analysis and literature, not tank data. Container
per-command budget (120 s) forces the 1024-panel run into a single-row script.

## 6. Level

* L2 real code: yes — `run_case` builds a payload from a solve, no MOCK/STUB path.
* L3 integration: yes — `benchmarks/__init__.py:5`, `capability_evidence.py:1002`
  (`case-bem-rigid-lid-hemisphere`), solver `tensorbem.free_surface_solver3d.solve_radiation_3d`.
* L4 benchmark validation: yes — complete case, analytic reference verified by two
  independent routes, error table + convergence/truncation ladders, mesh precondition,
  negative and scope controls, committed artifacts, one-command repro, 11 tests.
* L5 research/engineering result: no — this validates an existing solver capability in a
  limit; it introduces no new numerical capability, no new publication-grade result.

**Verdict: L4** — recorded in `tensor_maturity_verdicts` (`verdict_by = 'board-audit'`).

## 7. Commands run by this audit

```
# reference (R1) + independent BEM (R2) -> artifacts/m17395/independent_reference_check.json
PYTHONPATH=src python3 scripts/audit_m17395_rigid_lid_hemisphere.py --levels 2 3 4

# reproduction of the committed artifacts
cd /root/tp-agent/work/m17338 && bash -n scripts/repro_case_bem_rigid_lid_hemisphere.sh
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_bem_rigid_lid_hemisphere \
    --outdir <m17395>/artifacts/m17395/repro_artifacts --check      # PASS, identical payload
PYTHONPATH=src python3 -m pytest tests/test_case_bem_rigid_lid_hemisphere.py -q
    -> 11 passed in 57.03 s

# referee-row headline -> artifacts/m17395/row_headline_1024.json
PYTHONPATH=src python3 -c "from tensorbem.benchmarks import case_bem_rigid_lid_hemisphere as c; \
    print(c.solver_rows(4, (1e-3,), c.rigid_lid_reference()))"

# figures
python3 scripts/audit_m17395_figures.py
```

Artifacts produced by this audit (committed under `docs/m17395/`, byte-identical copies in the
git-ignored `artifacts/m17395/`): `independent_reference_check.json`,
`row_headline_1024.json`, `repro_artifacts/` (regenerated case payload),
`audit_m17395_reference_and_convergence.png`, `audit_m17395_independent_controls.png`,
`audit_evidence.json` (the exact JSON stored in `tensor_maturity_verdicts`).

Verdict row: `tensor_maturity_verdicts` id 550, `level = 4`, `verdict_by = 'board-audit'`,
`gate_state = 'pending'` (awaiting the automatic maturity gate).