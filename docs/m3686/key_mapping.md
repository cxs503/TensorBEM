# PR #128 (agent/m3181) 注册表键名 → 合并后规范键名

本表由 `docs/m3686/make_artifacts.py` 从两个分支的模块对象程序化推导，
逐格一一对应（63 格，双射断言在生成时校验通过），无手抄。

| agent/m3181 键 | m3181 组 | 合并后键（规范） | 合并后组 | m3181 声明 | 合并后声明 | m3181 层级 | 合并后层级 |
|---|---|---|---|---:|---:|---|---|
| `hulme_condition_number_16` | `hulme3d` | `hulme_hemisphere_16_condition_number` | `hulme_hemisphere` | 3.893 | 3.893 | heavy | fast |
| `hulme_heave_added_mass_rel_error_16` | `hulme3d` | `hulme_hemisphere_16_heave_added_mass_error` | `hulme_hemisphere` | 0.12 | 0.12 | heavy | fast |
| `hulme_heave_damping_rel_error_16` | `hulme3d` | `hulme_hemisphere_16_heave_damping_error` | `hulme_hemisphere` | 0.122 | 0.122 | heavy | fast |
| `hulme_surge_added_mass_rel_error_16` | `hulme3d` | `hulme_hemisphere_16_surge_added_mass_error` | `hulme_hemisphere` | 0.18 | 0.18 | heavy | fast |
| `hulme_surge_damping_rel_error_16` | `hulme3d` | `hulme_hemisphere_16_surge_damping_error` | `hulme_hemisphere` | 0.402 | 0.402 | heavy | fast |
| `hulme_excitation_force_z_rel_error_16` | `hulme3d` | `hulme_hemisphere_16_vertical_force_error` | `hulme_hemisphere` | 0.0667 | 0.0667 | heavy | fast |
| `hulme_condition_number_64` | `hulme3d` | `hulme_hemisphere_64_condition_number` | `hulme_hemisphere` | 3.052 | 3.052 | heavy | fast |
| `hulme_heave_added_mass_rel_error_64` | `hulme3d` | `hulme_hemisphere_64_heave_added_mass_error` | `hulme_hemisphere` | 0.0297 | 0.0297 | heavy | fast |
| `hulme_heave_damping_rel_error_64` | `hulme3d` | `hulme_hemisphere_64_heave_damping_error` | `hulme_hemisphere` | 0.045 | 0.045 | heavy | fast |
| `hulme_surge_added_mass_rel_error_64` | `hulme3d` | `hulme_hemisphere_64_surge_added_mass_error` | `hulme_hemisphere` | 0.0505 | 0.0505 | heavy | fast |
| `hulme_surge_damping_rel_error_64` | `hulme3d` | `hulme_hemisphere_64_surge_damping_error` | `hulme_hemisphere` | 0.13 | 0.13 | heavy | fast |
| `hulme_excitation_force_z_rel_error_64` | `hulme3d` | `hulme_hemisphere_64_vertical_force_error` | `hulme_hemisphere` | 0.0221 | 0.0221 | heavy | fast |
| `hulme_observed_order_a33_64` | `hulme3d` | `hulme_hemisphere_64_order_a33` | `hulme_hemisphere` | 2.23 | 2.23 | heavy | fast |
| `hulme_observed_order_b33_64` | `hulme3d` | `hulme_hemisphere_64_order_b33` | `hulme_hemisphere` | 1.59 | 1.59 | heavy | fast |
| `hulme_observed_order_a11_64` | `hulme3d` | `hulme_hemisphere_64_order_a11` | `hulme_hemisphere` | 2.04 | 2.04 | heavy | fast |
| `hulme_observed_order_b11_64` | `hulme3d` | `hulme_hemisphere_64_order_b11` | `hulme_hemisphere` | 1.62 | 1.8021 | heavy | fast |
| `hulme_observed_order_fz_64` | `hulme3d` | `hulme_hemisphere_64_order_fz` | `hulme_hemisphere` | 1.77 | 1.77 | heavy | fast |
| `hulme_condition_number_256` | `hulme3d` | `hulme_hemisphere_256_condition_number` | `hulme_hemisphere` | 2.783 | 2.783 | heavy | standard |
| `hulme_heave_added_mass_rel_error_256` | `hulme3d` | `hulme_hemisphere_256_heave_added_mass_error` | `hulme_hemisphere` | 0.00741 | 0.00741 | heavy | standard |
| `hulme_heave_damping_rel_error_256` | `hulme3d` | `hulme_hemisphere_256_heave_damping_error` | `hulme_hemisphere` | 0.0118 | 0.0118 | heavy | standard |
| `hulme_surge_added_mass_rel_error_256` | `hulme3d` | `hulme_hemisphere_256_surge_added_mass_error` | `hulme_hemisphere` | 0.013 | 0.013 | heavy | standard |
| `hulme_surge_damping_rel_error_256` | `hulme3d` | `hulme_hemisphere_256_surge_damping_error` | `hulme_hemisphere` | 0.0353 | 0.0353 | heavy | standard |
| `hulme_excitation_force_z_rel_error_256` | `hulme3d` | `hulme_hemisphere_256_vertical_force_error` | `hulme_hemisphere` | 0.00567 | 0.00567 | heavy | standard |
| `hulme_observed_order_a33_256` | `hulme3d` | `hulme_hemisphere_256_order_a33` | `hulme_hemisphere` | 2.06 | 2.06 | heavy | standard |
| `hulme_observed_order_b33_256` | `hulme3d` | `hulme_hemisphere_256_order_b33` | `hulme_hemisphere` | 1.98 | 1.98 | heavy | standard |
| `hulme_observed_order_a11_256` | `hulme3d` | `hulme_hemisphere_256_order_a11` | `hulme_hemisphere` | 2.01 | 2.01 | heavy | standard |
| `hulme_observed_order_b11_256` | `hulme3d` | `hulme_hemisphere_256_order_b11` | `hulme_hemisphere` | 1.88 | 1.9377 | heavy | standard |
| `hulme_observed_order_fz_256` | `hulme3d` | `hulme_hemisphere_256_order_fz` | `hulme_hemisphere` | 2.02 | 2.02 | heavy | standard |
| `hulme_chief_exact_a33_dimless` | `hulme3d_chief` | `hulme_chief_2_58_exact_a33` | `hulme_chief` | 0.4007 | 0.4007 | heavy | standard |
| `hulme_chief_exact_b33_dimless` | `hulme3d_chief` | `hulme_chief_2_58_exact_b33` | `hulme_chief` | 0.0631 | 0.0631 | heavy | standard |
| `hulme_chief_unregularized_condition_number` | `hulme3d_chief` | `hulme_chief_2_58_unregularized_condition_number` | `hulme_chief` | 145.66 | 145.66 | heavy | standard |
| `hulme_chief_unregularized_a33_dimless` | `hulme3d_chief` | `hulme_chief_2_58_unregularized_a33` | `hulme_chief` | 0.3481 | 0.3481 | heavy | standard |
| `hulme_chief_unregularized_b33_dimless` | `hulme3d_chief` | `hulme_chief_2_58_unregularized_b33` | `hulme_chief` | -0.1042 | -0.1042 | heavy | standard |
| `hulme_chief_chief_regularized_chief_point_count` | `hulme3d_chief` | `hulme_chief_2_58_chief_point_count` | `hulme_chief` | 5 | 5 | heavy | standard |
| `hulme_chief_chief_regularized_condition_number` | `hulme3d_chief` | `hulme_chief_2_58_chief_condition_number` | `hulme_chief` | 4.24 | 4.24 | heavy | standard |
| `hulme_chief_chief_regularized_a33_dimless` | `hulme3d_chief` | `hulme_chief_2_58_chief_a33` | `hulme_chief` | 0.3919 | 0.3919 | heavy | standard |
| `hulme_chief_chief_regularized_b33_dimless` | `hulme3d_chief` | `hulme_chief_2_58_chief_b33` | `hulme_chief` | 0.0687 | 0.0687 | heavy | standard |
| `wang_linton_condition_number_20` | `wang_linton3d` | `submerged_sphere_20_condition_number` | `submerged_sphere` | 2.407 | 2.407 | heavy | fast |
| `wang_linton_heave_added_mass_rel_error_20` | `wang_linton3d` | `submerged_sphere_20_heave_added_mass_error` | `submerged_sphere` | 0.255 | 0.255 | heavy | fast |
| `wang_linton_surge_added_mass_rel_error_20` | `wang_linton3d` | `submerged_sphere_20_surge_added_mass_error` | `submerged_sphere` | 0.268 | 0.268 | heavy | fast |
| `wang_linton_excitation_force_x_rel_error_20` | `wang_linton3d` | `submerged_sphere_20_force_x_error` | `submerged_sphere` | 0.36 | 0.36 | heavy | fast |
| `wang_linton_excitation_force_z_rel_error_20` | `wang_linton3d` | `submerged_sphere_20_force_z_error` | `submerged_sphere` | 0.37 | 0.37 | heavy | fast |
| `wang_linton_rotational_added_mass_max_ratio_20` | `wang_linton3d` | `submerged_sphere_20_rotational_coupling` | `submerged_sphere` | 1e-20 | 1e-20 | heavy | fast |
| `wang_linton_condition_number_80` | `wang_linton3d` | `submerged_sphere_80_condition_number` | `submerged_sphere` | 2.259 | 2.259 | heavy | fast |
| `wang_linton_heave_added_mass_rel_error_80` | `wang_linton3d` | `submerged_sphere_80_heave_added_mass_error` | `submerged_sphere` | 0.073 | 0.073 | heavy | fast |
| `wang_linton_surge_added_mass_rel_error_80` | `wang_linton3d` | `submerged_sphere_80_surge_added_mass_error` | `submerged_sphere` | 0.0802 | 0.0802 | heavy | fast |
| `wang_linton_excitation_force_x_rel_error_80` | `wang_linton3d` | `submerged_sphere_80_force_x_error` | `submerged_sphere` | 0.116 | 0.116 | heavy | fast |
| `wang_linton_excitation_force_z_rel_error_80` | `wang_linton3d` | `submerged_sphere_80_force_z_error` | `submerged_sphere` | 0.119 | 0.119 | heavy | fast |
| `wang_linton_rotational_added_mass_max_ratio_80` | `wang_linton3d` | `submerged_sphere_80_rotational_coupling` | `submerged_sphere` | 0.00022 | 0.0002151 | heavy | fast |
| `wang_linton_observed_order_a33_80` | `wang_linton3d` | `submerged_sphere_80_order_a33` | `submerged_sphere` | 2.12 | 2.12 | heavy | fast |
| `wang_linton_observed_order_b33_80` | `wang_linton3d` | `submerged_sphere_80_order_b33` | `submerged_sphere` | 1.72 | 1.72 | heavy | fast |
| `wang_linton_observed_order_a11_80` | `wang_linton3d` | `submerged_sphere_80_order_a11` | `submerged_sphere` | 2.04 | 2.04 | heavy | fast |
| `wang_linton_observed_order_fz_80` | `wang_linton3d` | `submerged_sphere_80_order_fz` | `submerged_sphere` | 1.91 | 1.91 | heavy | fast |
| `wang_linton_condition_number_320` | `wang_linton3d` | `submerged_sphere_320_condition_number` | `submerged_sphere` | 2.188 | 2.188 | heavy | standard |
| `wang_linton_heave_added_mass_rel_error_320` | `wang_linton3d` | `submerged_sphere_320_heave_added_mass_error` | `submerged_sphere` | 0.0185 | 0.0185 | heavy | standard |
| `wang_linton_surge_added_mass_rel_error_320` | `wang_linton3d` | `submerged_sphere_320_surge_added_mass_error` | `submerged_sphere` | 0.0208 | 0.0208 | heavy | standard |
| `wang_linton_excitation_force_x_rel_error_320` | `wang_linton3d` | `submerged_sphere_320_force_x_error` | `submerged_sphere` | 0.031 | 0.031 | heavy | standard |
| `wang_linton_excitation_force_z_rel_error_320` | `wang_linton3d` | `submerged_sphere_320_force_z_error` | `submerged_sphere` | 0.032 | 0.032 | heavy | standard |
| `wang_linton_rotational_added_mass_max_ratio_320` | `wang_linton3d` | `submerged_sphere_320_rotational_coupling` | `submerged_sphere` | 2.6e-05 | 2.6481e-05 | heavy | standard |
| `wang_linton_observed_order_a33_320` | `wang_linton3d` | `submerged_sphere_320_order_a33` | `submerged_sphere` | 2.06 | 2.06 | heavy | standard |
| `wang_linton_observed_order_b33_320` | `wang_linton3d` | `submerged_sphere_320_order_b33` | `submerged_sphere` | 1.93 | 1.93 | heavy | standard |
| `wang_linton_observed_order_a11_320` | `wang_linton3d` | `submerged_sphere_320_order_a11` | `submerged_sphere` | 2.03 | 2.03 | heavy | standard |
| `wang_linton_observed_order_fz_320` | `wang_linton3d` | `submerged_sphere_320_order_fz` | `submerged_sphere` | 1.98 | 1.98 | heavy | standard |

## 合并时从 m3181 并入 main 的格子（main 侧原本没有）

| 合并后键 | 组 | 声明值 | 判据 | 层级 |
|---|---|---:|---|---|
| `hulme_chief_2_58_exact_a33` | `hulme_chief` | 0.4007 | value | standard |
| `hulme_chief_2_58_exact_b33` | `hulme_chief` | 0.0631 | value | standard |
| `hulme_chief_2_58_chief_point_count` | `hulme_chief` | 5 | value | standard |
| `submerged_sphere_20_rotational_coupling` | `submerged_sphere` | 1e-20 | upper | fast |

## 仅 main(m3150) 有、m3181 未登记的格子（合并后保留）

| 合并后键 | 组 | 声明值 | 判据 | 层级 |
|---|---|---:|---|---|
| `hulme_hemisphere_256_near_far_damping_closure` | `hulme_hemisphere` | 0.000545 | value | standard |
| `hulme_hemisphere_256_haskind_surge` | `hulme_hemisphere` | 0.000209 | value | standard |
| `hulme_hemisphere_256_haskind_heave` | `hulme_hemisphere` | 0.000809 | value | standard |
