# Doc-claim coverage audit（文档数字守卫覆盖审计）

本文件由 `python -m tensorbem.benchmarks.doc_claim_audit` 程序化生成，无手抄数字。

## 1. 覆盖统计

| 指标 | 值 |
|---|---:|
| 被审计文档数 | 44 |
| PINNED（测试直接解析文档） | 16 |
| ARTIFACT（生成模块被测、文档文本未解析） | 14 |
| UNGUARDED（无人守） | 14 |
| 其中论文级 UNGUARDED | 7 |
| 数字 token 总数 | 5530 |
| 已守数字 token | 4632 |
| 数字覆盖率 | 83.8% |
| 文档覆盖率 | 68.2% |

## 2. 论文级未守文档（按数字量降序，前 15）

| 文档 | 数字 token | 行数 | 声明的生成器 | 现状 |
|---|---:|---:|---|---|
| `ursell_verification_debt_2026-11-02.md` | 104 | 156 | — | no declared generator and no reading test |
| `ursell_doc_reproducibility_report.md` | 95 | 58 | — | no declared generator and no reading test |
| `ursell_convergence_study.md` | 93 | 201 | — | no declared generator and no reading test |
| `validation_report_ursell_kochin.md` | 88 | 173 | — | no declared generator and no reading test |
| `lifting_foil.md` | 48 | 153 | — | no declared generator and no reading test |
| `finite_depth_2d.md` | 40 | 167 | `tensorbem.benchmarks.finite_depth_kernel` | generator never exercised by a test |
| `sphere_flow_study.md` | 27 | 6 | — | no declared generator and no reading test |

## 3. 全部文档分层

| document | tier | numbers | lines | generator |
|---|---|---:|---:|---|
| `ursell_verification_debt_2026-11-02.md` | UNGUARDED | 104 | 156 | — |
| `ursell_doc_reproducibility_report.md` | UNGUARDED | 95 | 58 | — |
| `ursell_convergence_study.md` | UNGUARDED | 93 | 201 | — |
| `validation_report_ursell_kochin.md` | UNGUARDED | 88 | 173 | — |
| `lifting_foil.md` | UNGUARDED | 48 | 153 | — |
| `finite_depth_2d.md` | UNGUARDED | 40 | 167 | `tensorbem.benchmarks.finite_depth_kernel` |
| `sphere_flow_study.md` | UNGUARDED | 27 | 6 | — |
| `near_surface_error_budget.md` | ARTIFACT | 219 | 202 | `tensorbem.benchmarks.near_surface_error_budget` |
| `hydrostatic_benchmarks_report.md` | ARTIFACT | 141 | 74 | `tensorbem.benchmarks.hydrostatic_benchmarks` |
| `verification.md` | ARTIFACT | 132 | 254 | `tensorbem.benchmarks.analytical_suite` |
| `verification_hydrostatic_rao.md` | ARTIFACT | 126 | 185 | `tensorbem.benchmarks.hydrostatic_rao_verification` |
| `hydrostatic_verification.md` | ARTIFACT | 117 | 174 | `tensorbem.benchmarks.hydrostatic_convergence`, `tensorbem.benchmarks.hydrostatic_figures` |
| `verification_doc_numbers_maccamy_fuchs.md` | ARTIFACT | 107 | 101 | `tensorbem.benchmarks.maccamy_fuchs` |
| `ursell_validation_note.md` | ARTIFACT | 105 | 99 | `tensorbem.benchmarks.ursell_cylinder` |
| `ursell_error_budget.md` | ARTIFACT | 65 | 89 | `tensorbem.benchmarks.ursell_error_budget` |
| `pressure_forces.md` | ARTIFACT | 54 | 185 | `tensorbem.benchmarks.sphere_flow` |
| `doc_claim_gate_portability.md` | ARTIFACT | 36 | 122 | `tensorbem.doc_claims` |
| `marine_bem_benchmarks.md` | PINNED | 453 | 527 | `tensorbem.benchmarks.cylinder_added_mass`, `tensorbem.benchmarks.sphere_added_mass`, `tensorbem.benchmarks.three_dimensional_study`, `tensorbem.benchmarks.ursell_cylinder`, `tensorbem.benchmarks.rigid_body_added_mass`, `tensorbem.benchmarks.dean_diffraction`, `tensorbem.benchmarks.maccamy_fuchs`, `tensorbem.benchmarks.floating_hemisphere`, `tensorbem.benchmarks.submerged_sphere_wave` |
| `finite_depth_envelope.md` | PINNED | 362 | 172 | `tensorbem.benchmarks.finite_depth_envelope` |
| `three_dimensional_bem.md` | PINNED | 252 | 516 | `tensorbem.benchmarks.sphere_added_mass`, `tensorbem.benchmarks.three_dimensional_study`, `tensorbem.benchmarks.sphere_velocity`, `tensorbem.benchmarks.surface_velocity_study`, `tensorbem.benchmarks.three_dimensional_doc_pins` |
| `classical_benchmarks.md` | PINNED | 209 | 376 | `tensorbem.benchmarks.declared_numbers`, `tensorbem.benchmarks.interior_harmonics`, `tensorbem.benchmarks.ellipse_added_mass`, `tensorbem.benchmarks.ellipsoid_added_mass`, `tensorbem.benchmarks.classical_study` |
| `lifting_kutta_closure.md` | PINNED | 174 | 112 | `tensorbem.benchmarks.lifting_solver` |
| `near_surface_field.md` | PINNED | 120 | 240 | `tensorbem.benchmarks.near_surface_field` |
| `finite_depth3d.md` | PINNED | 114 | 123 | `tensorbem.benchmarks.doc_claim_audit`, `tensorbem.benchmarks.finite_depth3d_pins` |
| `mixed_boundary_conditions.md` | PINNED | 107 | 186 | `tensorbem.benchmarks.mixed_boundary` |
| `wigley_wave_resistance.md` | PINNED | 98 | 350 | `tensorbem.benchmarks.wigley_verification` |
| `README.md` | PINNED | 58 | 145 | `tensorbem.benchmarks.unit_disk`, `tensorbem.benchmarks.cylinder_added_mass`, `tensorbem.benchmarks.refinement_study`, `tensorbem.benchmarks.analytical_suite`, `tensorbem.benchmarks.sphere_added_mass`, `tensorbem.benchmarks.sphere_velocity`, `tensorbem.benchmarks.three_dimensional_study`, `tensorbem.benchmarks.ellipse_added_mass`, `tensorbem.benchmarks.ellipsoid_added_mass`, `tensorbem.benchmarks.interior_harmonics`, `tensorbem.benchmarks.classical_study`, `tensorbem.benchmarks.ursell_cylinder`, `tensorbem.benchmarks.rigid_body_added_mass`, `tensorbem.benchmarks.dean_diffraction`, `tensorbem.benchmarks.maccamy_fuchs`, `tensorbem.benchmarks.floating_hemisphere`, `tensorbem.benchmarks.submerged_sphere_wave`, `tensorbem.benchmarks.hydrostatic_benchmarks` |
| `symmetry3d.md` | PINNED | 44 | 132 | `tensorbem.benchmarks.symmetry_reduction` |
| `report_m96.md` | UNGUARDED | 124 | 254 | — |
| `report_m57.md` | UNGUARDED | 96 | 214 | — |
| `tensorbem_health_audit_m1621.md` | UNGUARDED | 82 | 285 | — |
| `audit_baseline.md` | UNGUARDED | 63 | 140 | — |
| `ursell_benchmark_discrepancy.md` | UNGUARDED | 38 | 103 | — |
| `PROJECT_CHARTER.md` | UNGUARDED | 0 | 28 | — |
| `report_publish_incident_m2141.md` | UNGUARDED | 0 | 25 | `tensorbem.benchmarks.hydrostatic_figures` |
| `ursell_band_audit.md` | ARTIFACT | 451 | 78 | `tensorbem.benchmarks.ursell_band_audit` |
| `hydrostatic_verification_m2127_zh.md` | ARTIFACT | 258 | 170 | `tensorbem.benchmarks.hydrostatic_analytic` |
| `audit_m56_final.md` | ARTIFACT | 91 | 214 | `tensorbem.benchmarks.ursell_cylinder`, `tensorbem.benchmarks.floating_hemisphere` |
| `dev_report_tensorbem-ursell-band-audit.md` | ARTIFACT | 65 | 80 | `tensorbem.benchmarks.ursell_band_audit` |
| `project_review_2026-10-22.md` | PINNED | 227 | 522 | `tensorbem.benchmarks.unit_disk`, `tensorbem.benchmarks.cylinder_added_mass`, `tensorbem.benchmarks.refinement_study`, `tensorbem.benchmarks.analytical_suite`, `tensorbem.benchmarks.sphere_added_mass`, `tensorbem.benchmarks.sphere_velocity`, `tensorbem.benchmarks.surface_velocity_study`, `tensorbem.benchmarks.three_dimensional_study` |
| `ursell_provenance_audit.md` | PINNED | 220 | 203 | `tensorbem.benchmarks.ursell_cylinder` |
| `finite_depth_radiation.md` | PINNED | 104 | 142 | `tensorbem.benchmarks.finite_depth_radiation` |
| `ursell_quadrature_provenance.md` | PINNED | 67 | 93 | — |
| `mutation_audit_tensorbem.md` | PINNED | 56 | 112 | `tensorbem.benchmarks.mutation_audit` |
