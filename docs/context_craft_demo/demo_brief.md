# Task brief package

- query: `quadratic panel near singular quadrature tolerance verification of the operator`
- budget: 900 tok · used: 899 tok · density: 34.48 decision-items/1k tok
- kinds: {'TOLERANCE': 21, 'VERDICT': 4, 'NEGATIVE': 3, 'PIN': 2, 'CONTEXT': 1, 'TRAP': 1}
- package sha256: `eebdb85e29ce1e3dc34f8138999273a27f2d8fd902af3742d5b23b2c436900a4`

| # | kind | source:line | sha256[:8] | text |
|---|---|---|---|---|
| 1 | TOLERANCE | `dev_report_m7534_quadratic_panel_verification.md:69` | `e93ea44e` | (a) Quadratic panel flux error < 5% at `subdivisions=2` (320 elements); |
| 2 | TOLERANCE | `docs/drift_second_order/drift_second_order.md:150` | `b8b3811c` | ` F_y < 0.5%` of the surge force at every ka; the near field gives |
| 3 | VERDICT | `docs/audit_baseline.md:179` | `5aa28244` | Quadratic (6-node) curved-triangle panel operator **REAL** `tensorbem.panel_quadratic` 2 node(s) in 1 file(s) `docs/quadratic_near_singular/README.md` |
| 4 | TOLERANCE | `docs/m5240/potential_flow_anchor_maturation.md:161` | `27440e74` | *consistency identity* of the panel decomposition (criterion **W4**), not as an |
| 5 | TOLERANCE | `docs/numeric_evidence_matrix.md:23` | `df8a93dc` | `propeller-kp505-kt-design-point` (propeller-open-water) -- recomputed error 29.1257% exceeds the declared tolerance 3% |
| 6 | TOLERANCE | `docs/numeric_evidence_matrix.md:24` | `fa385a0c` | `propeller-kp505-eta0-worst-physical-point` (propeller-open-water) -- recomputed error 24.3849% exceeds the declared tolerance 5% |
| 7 | NEGATIVE | `docs/lifting_kutta_closure.md:83` | `6f37093c` | `docs/lifting_foil.md` recommended replacing the three-point rule by the exterior limit of the layer gradients. At a cusp that fails: the piecewise-constant single-layer density is logarithmically singular at panel ends, and the two panels that define the closure sit exactly there, so the reconstruction is dominated by the singular term. It is retained only as this negative control. |
| 8 | TOLERANCE | `docs/acoustic3d_pulse.md:4` | `d568a199` | literature solution; tolerance **< 8 %**). |
| 9 | TOLERANCE | `docs/m2970/mutation_gap_closure.md:106` | `6c25fd0f` | round-off \(O(\epsilon/h)\) on the right), so the 1×10⁻⁷ tolerance sits three orders above |
| 10 | TOLERANCE | `dev_report_m7534_quadratic_panel_verification.md:161` | `70033b9b` | **Conclusion**: All key quantities meet the < 3% criterion (or are within expected tolerance for |
| 11 | TOLERANCE | `dev_report_m17021_numeric_evidence.md:115` | `be268190` | threshold). No tolerance was chosen to make a row pass: the two KP505 rows fail |
| 12 | PIN | `docs/finite_depth3d_offaxis_split.md:62` | `cd79fb8e` | pins the *presence* of the quadrature defect (order of magnitude) so a |
| 13 | NEGATIVE | `docs/benchmarks/verified/bem_rankine_fs_panel/report.md:51` | `092a968f` | Evidence for the diagnosis: the wave-free limit is exact (3.3), the kernels are exact (3.2), and the failure scales with resolution rather than with any input parameter — the defect is in the discretisation of the resonant free-surface operator, not in the panel geometry or the kernel integrals. |
| 14 | VERDICT | `docs/bem_hydroelastic.md:159` | `99140f9c` | `benchmarks/verified/bem_hydroelastic/figures/bem_hydroelastic_kernel.png` -- (a) quadrature error of the |
| 15 | TOLERANCE | `docs/numeric_evidence_matrix.md:38` | `2f3ec335` | `propeller-open-water` (item `propeller-kp505-eta0-worst-physical-point`): structural REAL vs numeric WITHHELD -- recomputed error 24.3849% exceeds the declared tolerance 5% |
| 16 | TOLERANCE | `docs/numeric_evidence_matrix.md:37` | `ee90b419` | `propeller-open-water` (item `propeller-kp505-kt-design-point`): structural REAL vs numeric WITHHELD -- recomputed error 29.1257% exceeds the declared tolerance 3% |
| 17 | TOLERANCE | `docs/m17021_numeric_evidence_report.md:57` | `5e5a1546` | `propeller-open-water`: 结构层 REAL vs 数值层 WITHHELD — recomputed error 24.3849% exceeds the declared tolerance 5% |
| 18 | TOLERANCE | `docs/m17021_numeric_evidence_report.md:56` | `d579fdc5` | `propeller-open-water`: 结构层 REAL vs 数值层 WITHHELD — recomputed error 29.1257% exceeds the declared tolerance 3% |
| 19 | PIN | `docs/rao_input_validation.md:116` | `2c404238` | **Symmetry of `inertia` / `extra_damping` is required** (relative tolerance |
| 20 | TOLERANCE | `docs/wave_flume_wavemaker.md:167` | `3c0bbca4` | Transfer function versus Biesel-Havelock: worst **1.0858%**, finest mesh **0.0171%** — **PASS** at the 10% anchor tolerance and **PASS** at 3%. |
| 21 | VERDICT | `dev_report_m7534_quadratic_panel_verification.md:200` | `4eb94165` | **Verdict**: ✅ **PASS** — Quadratic panel convergence order verified against analytical solutions. |
| 22 | TOLERANCE | `docs/assembly_scaling.md:24` | `89a6968c` | Criteria fixed before the run: speedup >= 3.0x, absolute operator deviation < 1e-10, traced peak memory growth <= 10 %. |
| 23 | TOLERANCE | `docs/cathodic_protection_linear/REPORT.md:116` | `6b95e0d7` | C8 sphere current/field **PASS** 0.332 % / 0.334 % at 1280 panels, tolerance 3 % |
| 24 | TOLERANCE | `dev_report_m16976.md:43` | `24674763` | Anchor tolerance `<10 %` → **PASS** (worst 1.09 %, finest mesh 0.017 %); 3 % → **PASS**. |
| 25 | TOLERANCE | `dev_report_m5242_second_order_drift.md:13` | `97ba09ae` | near field, textbook surface+waterline form < 3 % 9.6e-8 yes |
| 26 | TOLERANCE | `docs/context_craft_ab/ab_results.md:17` | `5e7f3184` | arm recall 95%CI tolerance recall gate tokens redundancy decision share |
| 27 | CONTEXT | `docs/quadratic_panel_verification/README.md:3` | `120fea91` | Verification record for the quadratic (6-node, isoparametric) panel operator of |
| 28 | VERDICT | `docs/fmm_acceleration/report.md:109` | `27a97873` | **Status of §3.** Every PASS in the table was produced before two operator |
| 29 | NEGATIVE | `docs/drift_second_order/drift_second_order.md:217` | `0a948ad9` | fails at small `ka`, where the near field cannot resolve the `(ka)^3` residual with |
| 30 | TOLERANCE | `dev_report_m5242_second_order_drift.md:12` | `16bbd737` | near field vs anchor `rho g a^2/2` < 3 % 0.019 % (N=1600) yes |
| 31 | TOLERANCE | `docs/rao_input_validation.md:95` | `3bdd6106` | `estimate_natural_frequency` vs `sqrt(C/(m+A))` closed form rel. error ≤ 1e-12 (test tolerance) yes |
| 32 | TRAP | `docs/m14775/hull_reference_quadrature.md:28` | `266cec0d` | of any quadrature; they are checked against a 10^-5-step trapezoid in |
