# Doc-claim coverage audit（文档数字守卫覆盖审计）

本文件由 `python -m tensorbem.benchmarks.doc_claim_audit` 程序化生成，无手抄数字。

## 1. 覆盖统计

| 指标 | 值 |
|---|---:|
| 被审计文档数 | 107 |
| PINNED（测试直接解析文档） | 46 |
| ARTIFACT（生成模块被测、文档文本未解析） | 34 |
| UNGUARDED（无人守） | 27 |
| 其中论文级 UNGUARDED | 13 |
| 数字 token 总数 | 12494 |
| 已守数字 token | 10772 |
| 数字覆盖率 | 86.2% |
| 文档覆盖率 | 74.8% |

## 2. 论文级未守文档（按数字量降序，前 15）

| 文档 | 数字 token | 行数 | 声明的生成器 | 现状 |
|---|---:|---:|---|---|
| `wigley_diffraction_radiation_verification.md` | 133 | 183 | — | no declared generator and no reading test |
| `panel_linear_order.md` | 112 | 124 | `tensorbem.benchmarks.panel_linear_order` | generator never exercised by a test |
| `ursell_doc_reproducibility_report.md` | 95 | 58 | — | no declared generator and no reading test |
| `validation_report_ursell_kochin.md` | 88 | 173 | — | no declared generator and no reading test |
| `ice_mass_loading_verification.md` | 82 | 125 | — | no declared generator and no reading test |
| `BEM_FINITE_DEPTH_GREEN.md` | 76 | 156 | — | no declared generator and no reading test |
| `interior_field_domain_guard.md` | 73 | 157 | — | no declared generator and no reading test |
| `neumann_kelvin_verification.md` | 62 | 143 | — | no declared generator and no reading test |
| `kp505_openwater_anchor.md` | 56 | 111 | — | no declared generator and no reading test |
| `cummins_wigley_closure_report.md` | 55 | 246 | — | no declared generator and no reading test |
| `finite_depth3d_offaxis_split.md` | 44 | 70 | — | no declared generator and no reading test |
| `p22_n01_quadratic_panel_report.md` | 34 | 161 | `tensorbem.benchmarks.quadratic_panel_convergence` | generator never exercised by a test |
| `sphere_flow_study.md` | 27 | 6 | — | no declared generator and no reading test |

## 3. 全部文档分层

| document | tier | numbers | lines | generator |
|---|---|---:|---:|---|
| `wigley_diffraction_radiation_verification.md` | UNGUARDED | 133 | 183 | — |
| `panel_linear_order.md` | UNGUARDED | 112 | 124 | `tensorbem.benchmarks.panel_linear_order` |
| `ursell_doc_reproducibility_report.md` | UNGUARDED | 95 | 58 | — |
| `validation_report_ursell_kochin.md` | UNGUARDED | 88 | 173 | — |
| `ice_mass_loading_verification.md` | UNGUARDED | 82 | 125 | — |
| `BEM_FINITE_DEPTH_GREEN.md` | UNGUARDED | 76 | 156 | — |
| `interior_field_domain_guard.md` | UNGUARDED | 73 | 157 | — |
| `neumann_kelvin_verification.md` | UNGUARDED | 62 | 143 | — |
| `kp505_openwater_anchor.md` | UNGUARDED | 56 | 111 | — |
| `cummins_wigley_closure_report.md` | UNGUARDED | 55 | 246 | — |
| `finite_depth3d_offaxis_split.md` | UNGUARDED | 44 | 70 | — |
| `p22_n01_quadratic_panel_report.md` | UNGUARDED | 34 | 161 | `tensorbem.benchmarks.quadratic_panel_convergence` |
| `sphere_flow_study.md` | UNGUARDED | 27 | 6 | — |
| `ice_water_ship_numbers.md` | ARTIFACT | 278 | 101 | `tensorbem.benchmarks.ice_ship_verification` |
| `michell_validation.md` | ARTIFACT | 255 | 351 | `tensorbem.benchmarks.michell_validation` |
| `irregular_frequency_study.md` | ARTIFACT | 233 | 180 | `tensorbem.benchmarks.irregular_frequency_study` |
| `ice_water_ship.md` | ARTIFACT | 150 | 344 | `tensorbem.benchmarks.ice_ship_verification` |
| `cylinder_drift_asymptotic.md` | ARTIFACT | 144 | 218 | `tensorbem.benchmarks.cylinder_drift_asymptotic` |
| `hydrostatic_benchmarks_report.md` | ARTIFACT | 141 | 74 | `tensorbem.benchmarks.hydrostatic_benchmarks` |
| `verification_hydrostatic_rao.md` | ARTIFACT | 126 | 185 | `tensorbem.benchmarks.hydrostatic_rao_verification` |
| `hydrostatic_verification.md` | ARTIFACT | 117 | 174 | `tensorbem.benchmarks.hydrostatic_convergence`, `tensorbem.benchmarks.hydrostatic_figures` |
| `near_wall_envelope.md` | ARTIFACT | 109 | 191 | `tensorbem.benchmarks.near_wall_envelope_study` |
| `verification_doc_numbers_maccamy_fuchs.md` | ARTIFACT | 107 | 101 | `tensorbem.benchmarks.maccamy_fuchs` |
| `ursell_validation_note.md` | ARTIFACT | 105 | 100 | `tensorbem.benchmarks.ursell_cylinder` |
| `time_domain_pulse.md` | ARTIFACT | 104 | 343 | `tensorbem.benchmarks.time_domain_pulse` |
| `annulus_multicomponent.md` | ARTIFACT | 102 | 232 | `tensorbem.benchmarks.multicomponent_annulus` |
| `finite_depth_diffraction.md` | ARTIFACT | 102 | 145 | `tensorbem.benchmarks.finite_depth_diffraction` |
| `drift_force_report.md` | ARTIFACT | 87 | 167 | `tensorbem.benchmarks.drift_force_study` |
| `multihull_lateral.md` | ARTIFACT | 87 | 85 | `tensorbem.benchmarks.multihull_lateral_study` |
| `ursell_error_budget.md` | ARTIFACT | 65 | 89 | `tensorbem.benchmarks.ursell_error_budget` |
| `long_capsule_2d_limit.md` | ARTIFACT | 62 | 125 | `tensorbem.benchmarks.cylinder_3d_limit` |
| `michell_unit_arbitration.md` | ARTIFACT | 57 | 71 | `tensorbem.benchmarks.michell_unit_arbitration` |
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
| `near_surface_error_budget.md` | PINNED | 226 | 217 | `tensorbem.benchmarks.near_surface_error_budget` |
| `classical_benchmarks.md` | PINNED | 209 | 376 | `tensorbem.benchmarks.declared_numbers`, `tensorbem.benchmarks.interior_harmonics`, `tensorbem.benchmarks.ellipse_added_mass`, `tensorbem.benchmarks.ellipsoid_added_mass`, `tensorbem.benchmarks.classical_study` |
| `bem_hydroelastic.md` | PINNED | 176 | 197 | `tensorbem.benchmarks.hydroelastic_verification` |
| `lifting_kutta_closure.md` | PINNED | 174 | 112 | `tensorbem.benchmarks.lifting_solver` |
| `ursell_convergence_study.md` | PINNED | 171 | 271 | `tensorbem.benchmarks.ursell_convergence_doc` |
| `annulus_laplace.md` | PINNED | 158 | 199 | `tensorbem.benchmarks.annulus_laplace` |
| `tank_free_surface_correction.md` | PINNED | 142 | 195 | `tensorbem.benchmarks.tank_free_surface_verification` |
| `verification.md` | PINNED | 132 | 255 | `tensorbem.benchmarks.analytical_suite` |
| `near_surface_field.md` | PINNED | 120 | 243 | `tensorbem.benchmarks.near_surface_field` |
| `kkr_causality_band_study.md` | PINNED | 119 | 176 | — |
| `recovery_robustness.md` | PINNED | 115 | 149 | `tensorbem.benchmarks.recovery_robustness` |
| `finite_depth3d.md` | PINNED | 114 | 128 | `tensorbem.benchmarks.doc_claim_audit`, `tensorbem.benchmarks.finite_depth3d_pins` |
| `mixed_boundary_conditions.md` | PINNED | 107 | 186 | `tensorbem.benchmarks.mixed_boundary` |
| `finite_depth3d_deepwater_arbitration.md` | PINNED | 106 | 265 | `tensorbem.benchmarks.finite_depth3d_deepwater_arbitration` |
| `wigley_wave_resistance.md` | PINNED | 105 | 384 | `tensorbem.benchmarks.wigley_verification` |
| `ursell_verification_debt_2026-11-02.md` | PINNED | 104 | 156 | — |
| `m3538_propeller_openwater_uncertainty.md` | PINNED | 101 | 156 | — |
| `finite_depth3d_offaxis_diagnosis.md` | PINNED | 98 | 268 | — |
| `maruo_drift.md` | PINNED | 95 | 206 | `tensorbem.benchmarks.maruo_drift_verification` |
| `finite_depth3d_regime.md` | PINNED | 81 | 153 | `tensorbem.benchmarks.finite_depth3d_regime` |
| `hydrostatic_restoring_couplings.md` | PINNED | 80 | 137 | — |
| `six_dof_matrix_algebra.md` | PINNED | 78 | 166 | `tensorbem.benchmarks.six_dof_matrix_algebra` |
| `README.md` | PINNED | 58 | 145 | `tensorbem.benchmarks.unit_disk`, `tensorbem.benchmarks.cylinder_added_mass`, `tensorbem.benchmarks.refinement_study`, `tensorbem.benchmarks.analytical_suite`, `tensorbem.benchmarks.sphere_added_mass`, `tensorbem.benchmarks.sphere_velocity`, `tensorbem.benchmarks.three_dimensional_study`, `tensorbem.benchmarks.ellipse_added_mass`, `tensorbem.benchmarks.ellipsoid_added_mass`, `tensorbem.benchmarks.interior_harmonics`, `tensorbem.benchmarks.classical_study`, `tensorbem.benchmarks.ursell_cylinder`, `tensorbem.benchmarks.rigid_body_added_mass`, `tensorbem.benchmarks.dean_diffraction`, `tensorbem.benchmarks.maccamy_fuchs`, `tensorbem.benchmarks.floating_hemisphere`, `tensorbem.benchmarks.submerged_sphere_wave`, `tensorbem.benchmarks.hydrostatic_benchmarks` |
| `kelvin_kernel_reference.md` | PINNED | 54 | 120 | `tensorbem.benchmarks.kelvin_kernel_reference` |
| `drift_mesh_convergence.md` | PINNED | 53 | 138 | `tensorbem.benchmarks.drift_mesh_convergence` |
| `rankine_primitive_referee_study.md` | PINNED | 53 | 85 | — |
| `lifting_foil.md` | PINNED | 51 | 161 | `tensorbem.benchmarks.lifting_doc_pins` |
| `symmetry3d.md` | PINNED | 44 | 151 | `tensorbem.benchmarks.symmetry_reduction` |
| `multibody_frequency_dependence_3d.md` | PINNED | 0 | 147 | — |
| `dev_report_m4757_function_level_api_pins.md` | UNGUARDED | 292 | 260 | — |
| `report_m96.md` | UNGUARDED | 124 | 254 | — |
| `report_m57.md` | UNGUARDED | 96 | 214 | — |
| `tensorbem_health_audit_m1621.md` | UNGUARDED | 82 | 285 | — |
