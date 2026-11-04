# 合并后三维自由液面注册表逐格复算（66 格）

由 `PYTHONPATH=src python3 -m tensorbem.benchmarks.declared_numbers` 的
严格 JSON 报告（`docs/m3686/measurements.json`）程序化生成。
`upper` 行的相对偏差以声明上界为分母，其 -1.00 仅表示实测远低于上界。

| 键 | 组 | 判据 | 声明 | 实测 | 相对偏差 | 容差 | 判定 |
|---|---|---|---:|---:|---:|---:|---|
| `hulme_chief_2_58_chief_a33` | `hulme_chief` | value | 0.3919 | 0.391908 | +2.15e-05 | 1.5e-02 | PASS |
| `hulme_chief_2_58_chief_b33` | `hulme_chief` | value | 0.0687 | 0.0687079 | +1.15e-04 | 1.5e-02 | PASS |
| `hulme_chief_2_58_chief_condition_number` | `hulme_chief` | value | 4.24 | 4.24497 | +1.17e-03 | 1.5e-02 | PASS |
| `hulme_chief_2_58_chief_point_count` | `hulme_chief` | value | 5 | 5 | +0.00e+00 | 1.5e-02 | PASS |
| `hulme_chief_2_58_exact_a33` | `hulme_chief` | value | 0.4007 | 0.400737 | +9.33e-05 | 1.5e-02 | PASS |
| `hulme_chief_2_58_exact_b33` | `hulme_chief` | value | 0.0631 | 0.063122 | +3.48e-04 | 1.5e-02 | PASS |
| `hulme_chief_2_58_unregularized_a33` | `hulme_chief` | value | 0.3481 | 0.348062 | -1.09e-04 | 1.5e-02 | PASS |
| `hulme_chief_2_58_unregularized_b33` | `hulme_chief` | value | -0.1042 | -0.104155 | +4.35e-04 | 1.5e-02 | PASS |
| `hulme_chief_2_58_unregularized_condition_number` | `hulme_chief` | value | 145.66 | 145.659 | -9.43e-06 | 1.5e-02 | PASS |
| `hulme_hemisphere_16_condition_number` | `hulme_hemisphere` | value | 3.893 | 3.89255 | -1.15e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_16_heave_added_mass_error` | `hulme_hemisphere` | value | 0.12 | 0.120128 | +1.07e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_16_heave_damping_error` | `hulme_hemisphere` | value | 0.122 | 0.1219 | -8.22e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_16_surge_added_mass_error` | `hulme_hemisphere` | value | 0.18 | 0.180237 | +1.32e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_16_surge_damping_error` | `hulme_hemisphere` | value | 0.402 | 0.40241 | +1.02e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_16_vertical_force_error` | `hulme_hemisphere` | value | 0.0667 | 0.0667164 | +2.46e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_condition_number` | `hulme_hemisphere` | value | 2.783 | 2.78288 | -4.37e-05 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_haskind_heave` | `hulme_hemisphere` | value | 0.000809 | 0.000808537 | -5.73e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_haskind_surge` | `hulme_hemisphere` | value | 0.000209 | 0.000208918 | -3.92e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_heave_added_mass_error` | `hulme_hemisphere` | value | 0.00741 | 0.00740544 | -6.15e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_heave_damping_error` | `hulme_hemisphere` | value | 0.0118 | 0.0118119 | +1.01e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_near_far_damping_closure` | `hulme_hemisphere` | value | 0.000545 | 0.000544783 | -3.99e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_order_a11` | `hulme_hemisphere` | value | 2.01 | 2.00598 | -2.00e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_order_a33` | `hulme_hemisphere` | value | 2.06 | 2.05917 | -4.02e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_order_b11` | `hulme_hemisphere` | value | 1.9377 | 1.93773 | +1.33e-05 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_order_b33` | `hulme_hemisphere` | value | 1.98 | 1.98201 | +1.02e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_order_fz` | `hulme_hemisphere` | value | 2.02 | 2.01525 | -2.35e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_surge_added_mass_error` | `hulme_hemisphere` | value | 0.013 | 0.0130384 | +2.95e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_surge_damping_error` | `hulme_hemisphere` | value | 0.0353 | 0.0352745 | -7.22e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_256_vertical_force_error` | `hulme_hemisphere` | value | 0.00567 | 0.00566791 | -3.69e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_condition_number` | `hulme_hemisphere` | value | 3.052 | 3.05155 | -1.48e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_heave_added_mass_error` | `hulme_hemisphere` | value | 0.0297 | 0.0297359 | +1.21e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_heave_damping_error` | `hulme_hemisphere` | value | 0.045 | 0.0450225 | +5.00e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_order_a11` | `hulme_hemisphere` | value | 2.04 | 2.03566 | -2.13e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_order_a33` | `hulme_hemisphere` | value | 2.23 | 2.23418 | +1.87e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_order_b11` | `hulme_hemisphere` | value | 1.8021 | 1.80206 | -2.45e-05 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_order_b33` | `hulme_hemisphere` | value | 1.59 | 1.59384 | +2.42e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_order_fz` | `hulme_hemisphere` | value | 1.77 | 1.76843 | -8.88e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_surge_added_mass_error` | `hulme_hemisphere` | value | 0.0505 | 0.050508 | +1.58e-04 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_surge_damping_error` | `hulme_hemisphere` | value | 0.13 | 0.130492 | +3.79e-03 | 1.5e-02 | PASS |
| `hulme_hemisphere_64_vertical_force_error` | `hulme_hemisphere` | value | 0.0221 | 0.0220941 | -2.68e-04 | 1.5e-02 | PASS |
| `submerged_sphere_20_condition_number` | `submerged_sphere` | value | 2.407 | 2.40747 | +1.93e-04 | 1.5e-02 | PASS |
| `submerged_sphere_20_force_x_error` | `submerged_sphere` | value | 0.36 | 0.36029 | +8.06e-04 | 1.5e-02 | PASS |
| `submerged_sphere_20_force_z_error` | `submerged_sphere` | value | 0.37 | 0.369771 | -6.20e-04 | 1.5e-02 | PASS |
| `submerged_sphere_20_heave_added_mass_error` | `submerged_sphere` | value | 0.255 | 0.254967 | -1.30e-04 | 1.5e-02 | PASS |
| `submerged_sphere_20_rotational_coupling` | `submerged_sphere` | upper | 1e-20 | 2.49163e-32 | -1.00e+00 | 1.5e-02 | PASS |
| `submerged_sphere_20_surge_added_mass_error` | `submerged_sphere` | value | 0.268 | 0.267994 | -2.14e-05 | 1.5e-02 | PASS |
| `submerged_sphere_320_condition_number` | `submerged_sphere` | value | 2.188 | 2.18799 | -2.80e-06 | 1.5e-02 | PASS |
| `submerged_sphere_320_force_x_error` | `submerged_sphere` | value | 0.031 | 0.0310005 | +1.52e-05 | 1.5e-02 | PASS |
| `submerged_sphere_320_force_z_error` | `submerged_sphere` | value | 0.032 | 0.0319723 | -8.66e-04 | 1.5e-02 | PASS |
| `submerged_sphere_320_heave_added_mass_error` | `submerged_sphere` | value | 0.0185 | 0.0185089 | +4.81e-04 | 1.5e-02 | PASS |
| `submerged_sphere_320_order_a11` | `submerged_sphere` | value | 2.03 | 2.03001 | +3.45e-06 | 1.5e-02 | PASS |
| `submerged_sphere_320_order_a33` | `submerged_sphere` | value | 2.06 | 2.06197 | +9.58e-04 | 1.5e-02 | PASS |
| `submerged_sphere_320_order_b33` | `submerged_sphere` | value | 1.93 | 1.92898 | -5.30e-04 | 1.5e-02 | PASS |
| `submerged_sphere_320_order_fz` | `submerged_sphere` | value | 1.98 | 1.97949 | -2.59e-04 | 1.5e-02 | PASS |
| `submerged_sphere_320_rotational_coupling` | `submerged_sphere` | value | 2.6481e-05 | 2.64809e-05 | -3.99e-06 | 1.5e-02 | PASS |
| `submerged_sphere_320_surge_added_mass_error` | `submerged_sphere` | value | 0.0208 | 0.0207657 | -1.65e-03 | 1.5e-02 | PASS |
| `submerged_sphere_80_condition_number` | `submerged_sphere` | value | 2.259 | 2.25871 | -1.26e-04 | 1.5e-02 | PASS |
| `submerged_sphere_80_force_x_error` | `submerged_sphere` | value | 0.116 | 0.115942 | -5.03e-04 | 1.5e-02 | PASS |
| `submerged_sphere_80_force_z_error` | `submerged_sphere` | value | 0.119 | 0.119346 | +2.91e-03 | 1.5e-02 | PASS |
| `submerged_sphere_80_heave_added_mass_error` | `submerged_sphere` | value | 0.073 | 0.0729881 | -1.63e-04 | 1.5e-02 | PASS |
| `submerged_sphere_80_order_a11` | `submerged_sphere` | value | 2.04 | 2.04218 | +1.07e-03 | 1.5e-02 | PASS |
| `submerged_sphere_80_order_a33` | `submerged_sphere` | value | 2.12 | 2.11655 | -1.63e-03 | 1.5e-02 | PASS |
| `submerged_sphere_80_order_b33` | `submerged_sphere` | value | 1.72 | 1.71586 | -2.40e-03 | 1.5e-02 | PASS |
| `submerged_sphere_80_order_fz` | `submerged_sphere` | value | 1.91 | 1.91352 | +1.85e-03 | 1.5e-02 | PASS |
| `submerged_sphere_80_rotational_coupling` | `submerged_sphere` | value | 0.0002151 | 0.000215099 | -5.40e-06 | 1.5e-02 | PASS |
| `submerged_sphere_80_surge_added_mass_error` | `submerged_sphere` | value | 0.0802 | 0.0801643 | -4.45e-04 | 1.5e-02 | PASS |

## 汇总（value 判据行）

| 指标 | 数值 |
|---|---:|
| 复算格数 | 66 |
| PASS | 66 |
| 最大相对偏差 | 3.787e-03 (`hulme_hemisphere_64_surge_damping_error`) |
| 标准差阈值 1.5e-2 | 65 格全部满足 |

