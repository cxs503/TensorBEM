# Doc-claim coverage audit（文档数字守卫覆盖审计）

本文件由 `python -m tensorbem.benchmarks.doc_claim_audit` 程序化生成，无手抄数字。

## 1. 覆盖统计

| 指标 | 值 |
|---|---:|
| 被审计文档数 | 94 |
| PINNED（测试直接解析文档） | 40 |
| ARTIFACT（生成模块被测、文档文本未解析） | 32 |
| UNGUARDED（无人守） | 22 |
| 其中论文级 UNGUARDED | 8 |
| 数字 token 总数 | 10868 |
| 已守数字 token | 9461 |
| 数字覆盖率 | 87.1% |
| 文档覆盖率 | 76.6% |

## 2. 论文级未守文档（按数字量降序，前 15）

| 文档 | 数字 token | 行数 | 声明的生成器 | 现状 |
|---|---:|---:|---|---|
| `panel_linear_order.md` | 112 | 124 | `tensorbem.benchmarks.panel_linear_order` | generator never exercised by a test |
| `m3538_propeller_openwater_uncertainty.md` | 101 | 156 | — | no declared generator and no reading test |
| `ursell_doc_reproducibility_report.md` | 95 | 58 | — | no declared generator and no reading test |
| `validation_report_ursell_kochin.md` | 88 | 173 | — | no declared generator and no reading test |
| `ice_mass_loading_verification.md` | 82 | 125 | — | no declared generator and no reading test |
| `interior_field_domain_guard.md` | 73 | 157 | — | no declared generator and no reading test |
| `finite_depth3d_offaxis_split.md` | 44 | 70 | — | no declared generator and no reading test |
| `sphere_flow_study.md` | 27 | 6 | — | no declared generator and no reading test |

## 3. 全部文档分层

| document | tier | numbers | lines | generator |
|---|---|---:|---:|---|
| `panel_linear_order.md` | UNGUARDED | 112 | 124 | `tensorbem.benchmarks.panel_linear_order` |
| `m3538_propeller_openwater_uncertainty.md` | UNGUARDED | 101 | 156 | — |
| `ursell_doc_reproducibility_report.md` | UNGUARDED | 95 | 58 | — |
| `validation_report_ursell_kochin.md` | UNGUARDED | 88 | 173 | — |
| `ice_mass_loading_verification.md` | UNGUARDED | 82 | 125 | — |
| `interior_field_domain_guard.md` | UNGUARDED | 73 | 157 | — |
| `finite_depth3d_offaxis_split.md` | UNGUARDED | 44 | 70 | — |
| `sphere_flow_study.md` | UNGUARDED | 27 | 6 | — |
| `ice_water_ship_numbers.md` | ARTIFACT | 278 | 101 | `tensorbem.benchmarks.ice_ship_verification` |
| `michell_validation.md` | ARTIFACT | 255 | 351 | `tensorbem.benchmarks.michell_validation` |
| `irregular_frequency_study.md` | ARTIFACT | 233 | 180 | `tensorbem.benchmarks.irregular_frequency_study` |
| `near_surface_error_budget.md` | ARTIFACT | 219 | 202 | `tensorbem.benchmarks.near_surface_error_budget` |
| `ice_water_ship.md` | ARTIFACT | 150 | 344 | `tensorbem.benchmarks.ice_ship_verification` |
| `hydrostatic_benchmarks_report.md` | ARTIFACT | 141 | 74 | `tensorbem.benchmarks.hydrostatic_benchmarks` |
| `verification_hydrostatic_rao.md` | ARTIFACT | 126 | 185 | `tensorbem.benchmarks.hydrostatic_rao_verification` |
| `hydrostatic_verification.md` | ARTIFACT | 117 | 174 | `tensorbem.benchmarks.hydrostatic_convergence`, `tensorbem.benchmarks.hydrostatic_figures` |
| `verification_doc_numbers_maccamy_fuchs.md` | ARTIFACT | 107 | 101 | `tensorbem.benchmarks.maccamy_fuchs` |
| `ursell_validation_note.md` | ARTIFACT | 105 | 100 | `tensorbem.benchmarks.ursell_cylinder` |
| `time_domain_pulse.md` | ARTIFACT | 104 | 343 | `tensorbem.benchmarks.time_domain_pulse` |
| `annulus_multicomponent.md` | ARTIFACT | 102 | 232 | `tensorbem.benchmarks.multicomponent_annulus` |
| `finite_depth_diffraction.md` | ARTIFACT | 102 | 145 | `tensorbem.benchmarks.finite_depth_diffraction` |
| `drift_force_report.md` | ARTIFACT | 87 | 167 | `tensorbem.benchmarks.drift_force_study` |
| `multihull_lateral.md` | ARTIFACT | 87 | 85 | `tensorbem.benchmarks.multihull_lateral_study` |
| `ursell_error_budget.md` | ARTIFACT | 65 | 89 | `tensorbem.benchmarks.ursell_error_budget` |
| `long_capsule_2d_limit.md` | ARTIFACT | 62 | 125 | `tensorbem.benchmarks.cylinder_3d_limit` |
| `pressure_forces.md` | ARTIFACT | 54 | 185 | `tensorbem.benchmarks.sphere_flow` |
| `finite_depth3d_gradients.md` | ARTIFACT | 52 | 113 | `tensorbem.benchmarks.finite_depth3d_gradients` |
| `finite_depth_2d.md` | ARTIFACT | 40 | 167 | `tensorbem.benchmarks.finite_depth_kernel` |
| `two_cylinder_added_mass.md` | ARTIFACT | 38 | 145 | `tensorbem.benchmarks.two_cylinder_added_mass` |
| `doc_claim_gate_portability.md` | ARTIFACT | 36 | 122 | `tensorbem.doc_claims` |
| `readme_claim_inventory.md` | ARTIFACT | 20 | 140 | `tensorbem.benchmarks.doc_inventory` |
| `rao_input_validation.md` | ARTIFACT | 19 | 147 | `tensorbem.benchmarks.rao_input_validation` |
| `capability_registry_coverage.md` | ARTIFACT | 0 | 118 | `tensorbem.benchmarks.capability_coverage_study`, `tensorbem.capability_evidence` |
| `marine_bem_benchmarks.md` | PINNED | 701 | 667 | `tensorbem.benchmarks.cylinder_added_mass`, `tensorbem.benchmarks.sphere_added_mass`, `tensorbem.benchmarks.three_dimensional_study`, `tensorbem.benchmarks.ursell_cylinder`, `tensorbem.benchmarks.ursell_convergence_table`, `tensorbem.benchmarks.rigid_body_added_mass`, `tensorbem.benchmarks.dean_diffraction`, `tensorbem.benchmarks.maccamy_fuchs`, `tensorbem.benchmarks.floating_hemisphere`, `tensorbem.benchmarks.declared_numbers`, `tensorbem.benchmarks.submerged_sphere_wave` |
| `hydrostatic_restoring_waterplane_curvature.md` | PINNED | 415 | 199 | `tensorbem.benchmarks.hydrostatic_restoring_pins` |
| `finite_depth_envelope.md` | PINNED | 362 | 172 | `tensorbem.benchmarks.finite_depth_envelope` |
| `panel_operator_order.md` | PINNED | 323 | 240 | `tensorbem.benchmarks.panel_operator_order` |
| `three_dimensional_bem.md` | PINNED | 271 | 570 | `tensorbem.benchmarks.mesh_doc_pins`, `tensorbem.benchmarks.sphere_added_mass`, `tensorbem.benchmarks.three_dimensional_study`, `tensorbem.benchmarks.sphere_velocity`, `tensorbem.benchmarks.surface_velocity_study`, `tensorbem.benchmarks.three_dimensional_doc_pins`, `tensorbem.benchmarks.prose_claims` |
| `lifting_camber.md` | PINNED | 244 | 152 | `tensorbem.benchmarks.lifting_camber` |
| `bem_verified_anchors.md` | PINNED | 228 | 238 | `tensorbem.benchmarks.bem_verified_anchors` |
| `classical_benchmarks.md` | PINNED | 209 | 376 | `tensorbem.benchmarks.declared_numbers`, `tensorbem.benchmarks.interior_harmonics`, `tensorbem.benchmarks.ellipse_added_mass`, `tensorbem.benchmarks.ellipsoid_added_mass`, `tensorbem.benchmarks.classical_study` |
| `bem_hydroelastic.md` | PINNED | 176 | 197 | `tensorbem.benchmarks.hydroelastic_verification` |
| `lifting_kutta_closure.md` | PINNED | 174 | 112 | `tensorbem.benchmarks.lifting_solver` |
| `ursell_convergence_study.md` | PINNED | 171 | 271 | `tensorbem.benchmarks.ursell_convergence_doc` |
| `annulus_laplace.md` | PINNED | 158 | 199 | `tensorbem.benchmarks.annulus_laplace` |
| `verification.md` | PINNED | 132 | 255 | `tensorbem.benchmarks.analytical_suite` |
| `near_surface_field.md` | PINNED | 120 | 243 | `tensorbem.benchmarks.near_surface_field` |
| `kkr_causality_band_study.md` | PINNED | 119 | 176 | — |
| `recovery_robustness.md` | PINNED | 115 | 149 | `tensorbem.benchmarks.recovery_robustness` |
| `finite_depth3d.md` | PINNED | 114 | 128 | `tensorbem.benchmarks.doc_claim_audit`, `tensorbem.benchmarks.finite_depth3d_pins` |
| `mixed_boundary_conditions.md` | PINNED | 107 | 186 | `tensorbem.benchmarks.mixed_boundary` |
| `finite_depth3d_deepwater_arbitration.md` | PINNED | 106 | 265 | `tensorbem.benchmarks.finite_depth3d_deepwater_arbitration` |
| `wigley_wave_resistance.md` | PINNED | 105 | 384 | `tensorbem.benchmarks.wigley_verification` |
| `ursell_verification_debt_2026-11-02.md` | PINNED | 104 | 156 | — |
| `finite_depth3d_offaxis_diagnosis.md` | PINNED | 98 | 268 | — |
| `maruo_drift.md` | PINNED | 95 | 206 | `tensorbem.benchmarks.maruo_drift_verification` |
| `finite_depth3d_regime.md` | PINNED | 81 | 153 | `tensorbem.benchmarks.finite_depth3d_regime` |
| `hydrostatic_restoring_couplings.md` | PINNED | 80 | 137 | — |
| `six_dof_matrix_algebra.md` | PINNED | 78 | 166 | `tensorbem.benchmarks.six_dof_matrix_algebra` |
| `README.md` | PINNED | 58 | 145 | `tensorbem.benchmarks.unit_disk`, `tensorbem.benchmarks.cylinder_added_mass`, `tensorbem.benchmarks.refinement_study`, `tensorbem.benchmarks.analytical_suite`, `tensorbem.benchmarks.sphere_added_mass`, `tensorbem.benchmarks.sphere_velocity`, `tensorbem.benchmarks.three_dimensional_study`, `tensorbem.benchmarks.ellipse_added_mass`, `tensorbem.benchmarks.ellipsoid_added_mass`, `tensorbem.benchmarks.interior_harmonics`, `tensorbem.benchmarks.classical_study`, `tensorbem.benchmarks.ursell_cylinder`, `tensorbem.benchmarks.rigid_body_added_mass`, `tensorbem.benchmarks.dean_diffraction`, `tensorbem.benchmarks.maccamy_fuchs`, `tensorbem.benchmarks.floating_hemisphere`, `tensorbem.benchmarks.submerged_sphere_wave`, `tensorbem.benchmarks.hydrostatic_benchmarks` |
| `kelvin_kernel_reference.md` | PINNED | 54 | 120 | `tensorbem.benchmarks.kelvin_kernel_reference` |
| `rankine_primitive_referee_study.md` | PINNED | 53 | 85 | — |
| `lifting_foil.md` | PINNED | 51 | 161 | `tensorbem.benchmarks.lifting_doc_pins` |
| `symmetry3d.md` | PINNED | 44 | 151 | `tensorbem.benchmarks.symmetry_reduction` |
| `dev_report_m4757_function_level_api_pins.md` | UNGUARDED | 292 | 260 | — |
| `report_m96.md` | UNGUARDED | 124 | 254 | — |
| `report_m57.md` | UNGUARDED | 96 | 214 | — |
| `tensorbem_health_audit_m1621.md` | UNGUARDED | 82 | 285 | — |
| `dev_report_m4291_pr215_ci_red.md` | UNGUARDED | 47 | 88 | — |
| `merge_m3878_pr104_resolution.md` | UNGUARDED | 47 | 118 | — |
| `m3977_pr109_merge_healing_report.md` | UNGUARDED | 39 | 179 | — |
| `ursell_benchmark_discrepancy.md` | UNGUARDED | 38 | 103 | — |
| `m4789_gauss_rule_fallback.md` | UNGUARDED | 16 | 86 | — |
| `m4283_pr186_ci_triage.md` | UNGUARDED | 2 | 75 | — |
| `m4290_ci_fix_report.md` | UNGUARDED | 1 | 44 | — |
| `merge_report_m3929_pr134.md` | UNGUARDED | 1 | 69 | — |
| `PROJECT_CHARTER.md` | UNGUARDED | 0 | 28 | — |
| `report_publish_incident_m2141.md` | UNGUARDED | 0 | 25 | `tensorbem.benchmarks.hydrostatic_figures` |
| `ursell_band_audit.md` | ARTIFACT | 451 | 78 | `tensorbem.benchmarks.ursell_band_audit` |
| `hydrostatic_verification_m2127_zh.md` | ARTIFACT | 258 | 170 | `tensorbem.benchmarks.hydrostatic_analytic` |
