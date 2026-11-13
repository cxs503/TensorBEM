# AUDIT m17302 — CASE-BEM-CATHODIC-SPHERE (subject m17265, TensorBEM)

Board-audit (交差核验单). Verdict row: `tensor_maturity_verdicts.id=515`
(repo=TensorBEM, scope=`m17265 CASE-BEM-CATHODIC-SPHERE`, level=2, verdict_by=board-audit,
gate pre-check reproduced locally → gate_level=2, why=[]).
Benchmark registration by this audit: `tensor_benchmark_cases.id=47` (anchor_slug=CASE-BEM-CATHODIC-SPHERE,
status=passed, 41 rows existed before, 0 for this anchor → F1).

## 0. Independence and scope

* Audited branch `agent/m17265`, head **912bec2**; ECO main at audit time **89e5a52** (`git merge-base
  --is-ancestor 912bec2 HEAD` → **not** in main).
* Auditor ran in its own sandbox (`/tmp/m17302_subj`, src/tests/scripts copied from the subject worktree,
  `PYTHONPATH` → that copy) and recomputed the closed form from (σ, a, φ) with `tensorbem.geometry3d` only.
* Byte copies of every audited file with sha256 are in `snapshot/`; no subject number was used as an input
  to an auditor computation.

## 1. The three-piece delivery — does it exist?

| piece | audited artefact | status |
|---|---|---|
| run_artifacts | `snapshot/results.json` (sha256 `b1c6f7ef…d91e`, identical to `artifacts/cathodic_sphere/results.json` and to the committed `docs/cathodic_sphere_dnv_b401/results.json`), 3 PNGs (`convergence_current.png` `9cc016c9…`, `exterior_decay.png` `883d4e91…`, `error_budget.png` `d838a0ee…`) | **real, non-empty, byte-identical between the run dir and the docs copy** |
| benchmark ref vs got | ref `I = 4πσaφ_a = -43.982297150257104 A`; got `-41.820245 / -43.410210 / -43.836366 A` at 80/320/1280 panels; rel errors `4.9157 % / 1.3007 % / 0.3318 %`; observed order 1.918 / 1.971 | **real; auditor recomputation of the closed form agrees to 0.0 absolute (σ=3.5, a=1, φ=−1)** |
| repro_cmd | `PYTHONPATH=$PWD/src python scripts/cathodic_sphere_dnv_b401_study.py --subdiv 1 2 3 --routes-subdiv 2 3 --scaling-subdiv 2` + `pytest tests/test_cathodic_sphere_anchor.py -q` | **pytest half auditor-rerun: 10 passed in 69.39 s (first run) and 40.57 s (second run)**; full study re-run launched in the sandbox; **it completed and reproduced the subject `results.json` BYTE-IDENTICALLY** (`cmp` at 03:57, log `artifacts/board_audit_m17302/study_rerun.log`: verdict block `finest_level_error_percent 0.33179433072024567, ledger_pass true, internal_pass true, routes_pass true, scaling_pass true`, 3 figures regenerated) |

Report six-section completeness (`snapshot/REPORT.md`): §1 case + reference provenance, §2 method + code
pointers, §3 parameters, §4 ref-vs-computed table, §5 convergence + sensitivity + decay, §6 limits/traps/verdict —
**all six present**, and the platform report `tensor_reports.id=16272` carries the same six sections plus 3
figures (base64). §6 states the error judgement explicitly: 0.3318 % **is <3 %** (and <5 % ledger).

## 2. What the anchor actually certifies (auditor's own probe)

The uniformly polarised sphere has the *constant* exact single-layer density `q = −φ_a/a`; therefore

    measured current error ≈ k · (facet area deficit of the inscribed icosphere)

| sub | panels | h/a | current rel. error | facet-area deficit `A/(4πa²)−1` | error/deficit |
|---|---|---|---|---|---|
| 1 | 80 | 0.3963 | 4.9157 % | 7.1655 % | 0.686 |
| 2 | 320 | 0.1982 | 1.3007 % | 1.8822 % | 0.691 |
| 3 | 1280 | 0.0991 | 0.3318 % | 0.4765 % | 0.696 |

Least-squares slope through the origin **0.6864**, max relative residual **1.8e-03** → the error is
*proportional* to the geometric deficit over two mesh levels. Consequence: the headline accuracy is the
chord-vs-arc error of the faceted sphere (identical for both current routes), **not** the boundary-quadrature
accuracy. The report's §6.3 sentence "the latter [boundary quadrature] is certified by the closed form and the
second-order convergence" is **overstated** (F3) — no number is affected, but the anchor must not be sold as a
quadrature-accuracy certificate for a general hull.

Other auditor checks that came out **positive**:

* non-dimensional invariance `I/(4πσaφ) = 0.98699278050491` over 4 (σ, a, φ) points, spread **2.2e-16** → the σ
  factor and the 4π area factor are genuinely present in the implementation (a missing σ would show 71.43 %);
* exterior decay error flat in r (spread 7.9e-06 over 1.25–12 m) → the `1/r` law is reproduced to machine
  precision; the residual is the same uniform boundary scale error;
* far-field (Gauss-law) route: extrapolated `I = -43.83636653553718 A`, surface-vs-far-field **3.51e-09**,
  monopole-law max deviation 4.7e-07 at r=20 m; the function raises `ValueError` for enclosing radii inside the
  body and for non-monotone radii (negative controls in `test_far_field_rejects_enclosing_radius_inside_body`).

## 3. Findings (severity-ranked)

| id | severity | finding |
|---|---|---|
| F1 | medium | `tensor_benchmark_cases` had **0 rows** for this anchor_slug before the audit → no machine referee; registered now (id=47) with the audited numbers and the case-pool tolerance. |
| F2 | medium (level cap) | subject commit 912bec2 is **not** in main; `grep -rn 'far_field_current\|verify_conducting_sphere_routes\|cathodic_sphere_dnv_b401'` over the whole main worktree = **0 hits**; `solve_cathodic_protection` production behaviour unchanged → no system integration → **L3 refused**. |
| F3 | info (claim scope) | error ∝ facet-area deficit (slope 0.6864, resid 1.8e-03) → the anchor certifies geometry + σ/4π bookkeeping, **not** the Duffy/self-term quadrature accuracy the report attributes to it. |
| F4 | low | `spreading_resistance_numeric_ohm = −φ_a/I = −0.023036` is negative while the exact value is positive; the report quotes the magnitude and labels it so — cosmetic foot-gun only. |
| F5 | low | the 5120-panel (level-4) rung is absent: cluster job **cj1471 still `queued`** (34 min age) at audit time, no artefacts. The report claims no number for it → honest, but the error table stops at h/a = 0.099. |
| F6 | info | the far-field route reuses the same density `q`, so route agreement (3.5e-09) is a Gauss-bookkeeping/exterior-representation check, not an independent solver. **The subject discloses exactly this** (§6.3 + test docstring) — no credit booked as "second solver". |

## 4. Level: **L2** (not L3/L4/L5)

* L1: `tests/test_cathodic_sphere_anchor.py`, 10 tests, auditor-rerun green.
* L2: real code (197-line far-field delta + 347-line study driver + benchmark-CLI extension), executable entry,
  non-empty run artefacts, numbers reproducible by an independent recomputation of the closed form.
* **L3 refused**: branch unmerged, zero main-side callers (F2).
* **L4 refused**: L3 is a prerequisite that failed; plus the level-4 mesh rung is missing (F5) and the machine
  referee row did not exist before this audit (F1).
* **L5 refused**: the anchor is an analytic identity (no experimental / printed-table / independent-implementation
  comparison) and the model has no polarization kinetics.

## 5. Disposition

`confirmed at L2` + **request changes**:

1. **merge** `agent/m17265` into main (or state explicitly that the far-field route is dropped) — the capability
   is invisible to every main-branch user today;
2. **fix the claim scope** in §6.3: say "the closed form certifies the σ/4π bookkeeping and the exterior
   representation; on the uniformly polarised sphere the density is constant, so the residual error is the facet
   area deficit (auditor slope 0.6864) and general quadrature accuracy is not exercised";
3. **sign-normalise** `spreading_resistance_numeric_ohm` (or document the convention in the API docstring);
4. **finish the level-4 rung** (5120 panels) when the cluster drains, so the error table reaches the ecosystem
   `<3 %` line with margin, not just the boundary.

## Files

`snapshot/` (byte copies + sha256 map), `analysis.json`, `fig_audit_cathodic_sphere.png`,
`audit_analysis.py`, `land_verdict.py`, `AUDIT_m17265.md`; auditor logs
`../../artifacts/board_audit_m17302/{pytest_auditor_rerun.log,study_rerun.log}`.

---

## Addendum (filed after the verdict row) — full reproducibility

The complete `repro_cmd` study re-run (convergence 1/2/3 + two-route 2/3 + scaling + decay) finished in the
auditor sandbox `/tmp/m17302_subj`:

* `artifacts/board_audit_m17302/repro/results.json` is **byte-identical** to the subject
  `artifacts/cathodic_sphere/results.json` (`cmp` → no difference; sha256 `b1c6f7ef…d91e` both),
* its log reprints the same verdict block (`finest_level_error_percent = 0.33179433072024567`,
  `ledger_pass / internal_pass / routes_pass / scaling_pass = true`),
* the three figures were regenerated in the sandbox (not byte-compared: matplotlib metadata/PNG encoder state).

So the delivery is **deterministic and reproducible end-to-end**, not only test-green. This raises confidence in
L2 but does not lift the level: the level blocker is the missing integration face (F2), not reproducibility.
