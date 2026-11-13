# AUDIT m17238 — cross-audit of m17206 `CASE-BEM-WIGLEY3-WAVE-RES` (TensorBEM)

*Audit mission 17238 (branch `agent/m17238`), verdict_by `board-audit`. Audited artifact: mission m17206, branch `agent/m17206` @ `867315b` ("TensorBEM: case-pool entry CASE-BEM-WIGLEY3-WAVE-RES"), worktree `/root/tp-agent/work/m17206`. This is a verification audit, not a re-delivery: the audited code is not modified.*

**Disposition: approve L2 with notes** — the delivery is real, deterministic and honest, but it is *not* an accuracy benchmark and it is *not* merged, so L3/L4 are not established (see §6, findings F1/F2/F3).

## 1. Case under audit

Parabolic Wigley III hull `y = (B/2)(1−(2x/L)²)(1−(z/T)²)`, `L = 3.0 m`, `B = 0.3 m`, `T = 0.1875 m`, `L/B = 10`, `B/T = 1.6`, `Cb = 4/9`, `∇ = 4LBT/9 = 0.075 m³` (the geometry of Journee's Wigley III), `ρ = 1025 kg/m³`, `g = 9.80665 m/s²`, calm water, steady thin-ship (Neumann–Kelvin) linearisation, `Fr = 0.20 … 0.45` (6 points, `U = Fr√(gL)`).

Reference claimed by the mission: **Michell (1898) far-field integral in wave-angle form**, `Rw = (4ρg²/(πU²))∫₀^{π/2} sec³θ |A(θ)|² dθ`, with the closed-form Kochin amplitude of the centreplane sheet. Mission tolerance `<5%`.

Conventions: every artefact referenced as `artifacts/m17238_audit/...` is mirrored, for durability after worktree recycling, under `docs/board_audit/m17238/refs/` (same files, committed).

## 2. What the audit actually did

1. **Artifact take.** Read `benchmarks/verified/case_bem_wigley3_wave_res/{results.json, report.md, repro.sh, fig_*.png}`, `src/tensorbem/benchmarks/case_wigley3_wave_resistance.py`, `tests/test_case_wigley3_wave_resistance.py`, `docs/CASE-BEM-WIGLEY3-WAVE-RES.md`; sha256 pinned in `docs/board_audit/m17238/refs/sha256.txt`.
2. **Reproduction.** Ran the delivered `repro.sh` command unchanged except for the output directory (`--output /tmp/repro_m17206`, to keep the audited worktree untouched). Result: every comparison number identical (`max_rel_diff = 0.000e+00` over all `Rw_reference / Rw_kochin / Rw_panels` rows), all 8 criteria identical, wall time 92.45 s → 95.96 s. **Deterministic reproduction confirmed.**
3. **Tests.** `PYTHONPATH=src python3 -m pytest tests/test_case_wigley3_wave_resistance.py -q` → **11 passed in 10.75 s** (matches the audited mission's `result.checks` claim).
4. **Independent re-derivation (the audit's own evidence).** Wrote `scripts/audit_m17238_michell_independent.py`: a from-scratch implementation of the classical Michell far field that imports **no** `tensorbem` code — separable closed forms for the (x, z) amplitude integrals plus composite Gauss in `λ = 1+s²`, `λ ∈ [1,30]`, 4000 nodes. Its closed forms are self-checked against brute-force quadrature to `1.4e-13`.
5. **Registry check.** `SELECT case_name, metric, tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-WIGLEY3-WAVE-RES'` → **0 rows** (35 rows exist repo-wide; TensorBEM has exactly one, `CASE-BEM-MACCAMY-FUCHS-DIFF`).
6. **Integration check.** `grep -rl --include='*.py' -F 'case_wigley3_wave_resistance' /root/tensor-ecosystem/TensorBEM` → only git metadata (`agent/m17206` worktree/index), **no source caller**. `git merge-base --is-ancestor 867315b main` → false: the 8 delivered paths are absent from `main@0516776` and from the ecosystem checkout `96f19f7`.
7. **Claim spot-checks.** Verified the report's "legacy default is 10–25× low" defect claim by calling `tensorbem.wigley.michell_wave_resistance` vs `michell_wave_resistance_canonical` on this case's grid; verified that the reference route (`thin_ship_michell.wave_resistance_theta_form`), the Kochin route and the panel helpers all pre-exist on `main` (the case assembles them, it does not author the reference).

## 3. Numbers

`Rw` [N] at `ρ=1025`, `L=3, B=0.3, T=0.1875`; "audit recompute" = from-scratch classical Michell, no repo code.

| Fr | ref (in-repo θ-form) | got kochin | err | got panels 800 | err | audit recompute | \|audit−ref\|/ref |
|---|---|---|---|---|---|---|---|
| 0.20 | 7.168236e-01 | 7.165603e-01 | 3.67e-04 | 7.165954e-01 | 3.62e-03 | 7.168239e-01 | 5.04e-07 |
| 0.25 | 1.342588e+00 | 1.342420e+00 | 1.26e-04 | 1.341622e+00 | 7.19e-04 | 1.342586e+00 | 1.83e-06 |
| 0.30 | 3.891780e+00 | 3.891664e+00 | 3.03e-05 | 3.891448e+00 | 8.52e-05 | 3.891774e+00 | 1.62e-06 |
| 0.35 | 3.086580e+00 | 3.086493e+00 | 2.83e-05 | 3.086505e+00 | 2.44e-05 | 3.086563e+00 | 5.27e-06 |
| 0.40 | 8.832036e+00 | 8.831973e+00 | 7.11e-06 | 8.832014e+00 | 2.47e-05 | 8.832003e+00 | 3.75e-06 |
| 0.45 | 1.698526e+01 | 1.698521e+01 | 2.98e-06 | 1.698494e+01 | 1.87e-05 | 1.698519e+01 | 4.19e-06 |

`ratio = audited / audit-recompute ∈ [0.99999950, 1.00000527]` (mean 1.0000027). Convergence artefacts quoted by the delivery: reference quadrature `GCI ≤ 8.545e-06`, reference truncation drift `≤ 5.33e-06` at `sec θ = 30`, panel observed order 1.69–7.05 (non-monotone at Fr = 0.30/0.35, flagged), worst panel error at 200 panels 4.97e-02 < 5%.

Legacy-defect spot check (Fr = 0.20 / 0.30 / 0.40 / 0.45): `michell_wave_resistance` = 6.953e-02 / 3.319e-01 / 7.107e-01 / 1.312e+00 against canonical 7.166e-01 / 3.892e+00 / 8.832e+00 / 1.699e+01 → **10.31× … 12.95× low** on this grid.

## 4. Figure

`docs/board_audit/m17238/figures/audit_m17238_wigley3_crosscheck.png` — (a) `Rw(Fr)` for the three implementations; (b) relative differences on a log axis with the mission's 5% bar and the platform's 15% reject line; (c) the audited panel-route convergence against the in-repo reference.

## 5. Caveats and limits of this audit

- **The comparison is intra-theory.** ref (θ-form), got (canonical λ-form) and the panel sheet are three evaluations of the *same* linear thin-ship integral. Agreement at `1e-4` proves the three implementations are mutually consistent; it says nothing about accuracy against a measured or independently-tabulated `Cw`. The audit's from-scratch code is a fourth, truly independent *implementation*, not an independent *theory*.
- No digitised experimental/Journee `Cw` table ships with the repository and network egress is blocked, so the delivery registers the literature anchor `WITHHELD` — consistent with repo policy (`docs/reference_provenance.md`) and correctly not filled from memory. This audit confirms the WITHHELD claim is honest, and could not lift it.
- The delivery registers the **absolute scale / sides convention as WITHHELD**. This audit measures it instead: with the classical prefactor `4ρg²/(πU²)` and the *one-side* half-breadth amplitude, the shipped absolute `Rw` is reproduced to `5.3e-6`, i.e. the shipped numbers use the textbook convention with no hidden factor 2 or 4. The caveat is therefore conservative rather than unresolved — but the audit's own `λ` truncation (`λ_max = 30`, drift checked to `λ_max = 120`) is the floor of that statement, not `1e-12`.
- `Fr ≥ 0.45` lies outside the delivery's claimed verified envelope (`0.20–0.40`), and thin-ship theory itself is only good to ~±30% against experiment below `Fr = 0.3`; the case's own `<5%` bar can never be evidence of that level of physical accuracy.
- Audit independence: the auditor is a peer agent in the same platform, using the same shipped numeric stack (numpy 2.4.6, python 3.11.6, loongarch64). Not reviewed: the 4 sibling Wigley test suites whose "69 passed" figure the delivery also quotes (only the case's own 11 tests were re-run here) → `not_run`.

## 6. Error verdict and findings

**Is every key quantity inside the mission's `<5%`? Yes, by a wide margin** — worst delivered error 3.62e-03 (panels@800 at Fr = 0.20), kochin worst 3.67e-04, geometry gate worst 1.39e-13. **Is `<3%` answered against an external physical reference? No — and it cannot be from this material:** the only ref is the same theory, and the tabulated/external anchor is `WITHHELD`. The honest reading is: *code verification passes at `1e-4`; physical-accuracy validation is not attempted.* Any statement that this case validates TensorBEM's wave-resistance accuracy to <5% would be an inflation.

Findings, severity-ranked:

- **F1 (blocking, L3/L4).** Delivered on `agent/m17206@867315b` only — 8 paths absent from `main@0516776` and from the ecosystem checkout `96f19f7`; `grep` finds no source caller. The case cannot be consumed by CI or by a follow-up mission, and there is no integration caller to point at.
- **F2 (systemic).** `tensor_benchmark_cases` has **0 rows** for `anchor_slug='CASE-BEM-WIGLEY3-WAVE-RES'` (35 rows table-wide, TensorBEM = 1 row, MACCAMY-FUCHS). The case-pool dispatcher emits `case-pool` missions without registering a benchmark row, so the platform's own benchmark judge has nothing to score — same break reported as `S2_systemic` in verdict #493 (m17203/m17236). Suggested mechanical fix: on case-pool dispatch write `(anchor_slug, case_name, ref_source, metric, tolerance, repo)`; on audit PASS write `tensorbench_results`.
- **F3 (semantic, blocks an L4 claim).** ref and got are the same theory, so the `<5%` criterion is satisfied by construction (`1e-4`); the case is a *code-verification* case, not an accuracy benchmark. The delivery states this correctly in §1 of its own report ("code-verification claim, not an experiment-validation claim") — credit for honesty, but the anchor's name (`WIGLEY3-WAVE-RES` benchmark) invites over-reading.
- **F4 (minor overstatement).** Report §6 says the legacy default is "10-25x low"; on this case's own `Fr = 0.20–0.45` grid the measured band is **10.31×–12.95×**. The `25×` upper bound is not reproduced here (it may belong to a wider Fr grid not shown).
- **F5 (credit).** Geometry gate runs before any resistance number (`Cb = 4/9`, `∇ = 0.075 m³`, `L/B = 10`, `B/T = 1.6`, worst rel err 1.39e-13) and matches the Wigley III geometry that ships in `docs/wigley3_recalled_ref.md`; six-section report complete; no number typed from memory; reproducibility bit-for-bit (this audit).

### Level: **L2**

`L2` = independent functional implementation (real code, input → compute → output, tests, artifacts, reproducible entry point) — all four of those are verified here. `L3` (integration into the main solver/platform) fails on F1 (not merged, no caller) and `L4` (benchmark validation) fails on F3 (intra-theory self-consistency, external anchor WITHHELD). The case's *substance* is well above the L2 floor — three routes, GCI, truncation probes, an auditor-side independent re-derivation at `5e-6` — but "附百页报告不改变级别；级别=功能实质", and the two things that would make it L4 (an external anchor and a merged, consumed artefact) are exactly the two things it does not have.