### 4.12 三维自由液面声明数字的注册表复算（m3181）

§4.10–§4.11 的表 10–12 与 CHIEF 对比表共 63 个数字此前**没有任何可执行判据**：只要 `tensorbem.free_surface3d`、`free_surface_solver3d` 或两个多极子参考解任一环节变动，文档数字就会静默腐化。本小节把这些格子逐条登记进 `tensorbem.benchmarks.declared_numbers` 的 `heavy` 层（新增 `hulme3d`、`hulme3d_chief`、`wang_linton3d` 三组，同表共用一次求解），默认 `pytest` 不跑，`TENSORBEM_HEAVY_TESTS=1` 时由 `tests/test_declared_numbers.py::test_documented_hulme_tables_are_reproduced` 与 `::test_documented_wang_linton_table_is_reproduced` 守护。

复算命令（整组 63 格一次跑完，本机约 98 s）：

```bash
PYTHONPATH=src python3 docs/m3181/run_three_d_free_surface.py \
    --output docs/m3181/outputs/three_d_free_surface_registry.json
PYTHONPATH=src python -m tensorbem.benchmarks.declared_numbers --tiers heavy --strict
```

**判决**：63 / 63 格 PASS；等值判据行（62 格）的最坏相对偏差 2.23e-02，容差 5e-3（表格中仅两位有效数字的转动耦合列采用 2.5e-2 舍入带容差；20 面元的转动耦合按机器零上界判，其 -1.00 的“偏差”是相对 1e-20 上界而言，不代表失配）。

**表 10（Hulme 漂浮半球，Ka = 1.0，16/64/256 面元）**

| 注册项 | 文档声明 | 本次复算 | 相对偏差 | 判定 |
|---|---:|---:|---:|---|
| `hulme_condition_number_16` | 3.893 | 3.89255 | -1.15e-04 | PASS |
| `hulme_heave_added_mass_rel_error_16` | 0.12 | 0.120128 | +1.07e-03 | PASS |
| `hulme_heave_damping_rel_error_16` | 0.122 | 0.1219 | -8.22e-04 | PASS |
| `hulme_surge_added_mass_rel_error_16` | 0.18 | 0.180237 | +1.32e-03 | PASS |
| `hulme_surge_damping_rel_error_16` | 0.402 | 0.40241 | +1.02e-03 | PASS |
| `hulme_excitation_force_z_rel_error_16` | 0.0667 | 0.0667164 | +2.46e-04 | PASS |
| `hulme_condition_number_64` | 3.052 | 3.05155 | -1.48e-04 | PASS |
| `hulme_heave_added_mass_rel_error_64` | 0.0297 | 0.0297359 | +1.21e-03 | PASS |
| `hulme_heave_damping_rel_error_64` | 0.045 | 0.0450225 | +5.00e-04 | PASS |
| `hulme_surge_added_mass_rel_error_64` | 0.0505 | 0.050508 | +1.58e-04 | PASS |
| `hulme_surge_damping_rel_error_64` | 0.13 | 0.130492 | +3.79e-03 | PASS |
| `hulme_excitation_force_z_rel_error_64` | 0.0221 | 0.0220941 | -2.68e-04 | PASS |
| `hulme_observed_order_a33_64` | 2.23 | 2.23418 | +1.87e-03 | PASS |
| `hulme_observed_order_b33_64` | 1.59 | 1.59384 | +2.42e-03 | PASS |
| `hulme_observed_order_a11_64` | 2.04 | 2.03566 | -2.13e-03 | PASS |
| `hulme_observed_order_b11_64` | 1.62 | 1.6247 | +2.90e-03 | PASS |
| `hulme_observed_order_fz_64` | 1.77 | 1.76843 | -8.88e-04 | PASS |
| `hulme_condition_number_256` | 2.783 | 2.78288 | -4.37e-05 | PASS |
| `hulme_heave_added_mass_rel_error_256` | 0.00741 | 0.00740544 | -6.15e-04 | PASS |
| `hulme_heave_damping_rel_error_256` | 0.0118 | 0.0118119 | +1.01e-03 | PASS |
| `hulme_surge_added_mass_rel_error_256` | 0.013 | 0.0130384 | +2.95e-03 | PASS |
| `hulme_surge_damping_rel_error_256` | 0.0353 | 0.0352745 | -7.22e-04 | PASS |
| `hulme_excitation_force_z_rel_error_256` | 0.00567 | 0.00566791 | -3.69e-04 | PASS |
| `hulme_observed_order_a33_256` | 2.06 | 2.05917 | -4.02e-04 | PASS |
| `hulme_observed_order_b33_256` | 1.98 | 1.98201 | +1.02e-03 | PASS |
| `hulme_observed_order_a11_256` | 2.01 | 2.00598 | -2.00e-03 | PASS |
| `hulme_observed_order_b11_256` | 1.88 | 1.88727 | +3.87e-03 | PASS |
| `hulme_observed_order_fz_256` | 2.02 | 2.01525 | -2.35e-03 | PASS |
| `hulme_chief_exact_a33_dimless` | 0.4007 | 0.400737 | +9.33e-05 | PASS |
| `hulme_chief_exact_b33_dimless` | 0.0631 | 0.063122 | +3.48e-04 | PASS |
| `hulme_chief_unregularized_condition_number` | 145.66 | 145.659 | -9.43e-06 | PASS |
| `hulme_chief_unregularized_a33_dimless` | 0.3481 | 0.348062 | -1.09e-04 | PASS |
| `hulme_chief_unregularized_b33_dimless` | -0.1042 | -0.104155 | +4.35e-04 | PASS |
| `hulme_chief_chief_regularized_chief_point_count` | 5 | 5 | +0.00e+00 | PASS |
| `hulme_chief_chief_regularized_condition_number` | 4.24 | 4.24497 | +1.17e-03 | PASS |
| `hulme_chief_chief_regularized_a33_dimless` | 0.3919 | 0.391908 | +2.15e-05 | PASS |
| `hulme_chief_chief_regularized_b33_dimless` | 0.0687 | 0.0687079 | +1.15e-04 | PASS |

<sub>来源：docs/marine_bem_benchmarks.md §4.10 表 10；数据由 `docs/m3181/outputs/three_d_free_surface_registry.json` 程序化注入。</sub>

**表 11（Ka = 2.58 免盖板 CHIEF 对比，256 面元）**

| 注册项 | 文档声明 | 本次复算 | 相对偏差 | 判定 |
|---|---:|---:|---:|---|
| `hulme_chief_exact_a33_dimless` | 0.4007 | 0.400737 | +9.33e-05 | PASS |
| `hulme_chief_exact_b33_dimless` | 0.0631 | 0.063122 | +3.48e-04 | PASS |
| `hulme_chief_unregularized_condition_number` | 145.66 | 145.659 | -9.43e-06 | PASS |
| `hulme_chief_unregularized_a33_dimless` | 0.3481 | 0.348062 | -1.09e-04 | PASS |
| `hulme_chief_unregularized_b33_dimless` | -0.1042 | -0.104155 | +4.35e-04 | PASS |
| `hulme_chief_chief_regularized_chief_point_count` | 5 | 5 | +0.00e+00 | PASS |
| `hulme_chief_chief_regularized_condition_number` | 4.24 | 4.24497 | +1.17e-03 | PASS |
| `hulme_chief_chief_regularized_a33_dimless` | 0.3919 | 0.391908 | +2.15e-05 | PASS |
| `hulme_chief_chief_regularized_b33_dimless` | 0.0687 | 0.0687079 | +1.15e-04 | PASS |

<sub>来源：docs/marine_bem_benchmarks.md §4.10 表 11；数据由 `docs/m3181/outputs/three_d_free_surface_registry.json` 程序化注入。</sub>

**表 12（Wang/Linton 潜没球，f/a = 1.5，Ka = 0.8，20/80/320 面元）**

| 注册项 | 文档声明 | 本次复算 | 相对偏差 | 判定 |
|---|---:|---:|---:|---|
| `wang_linton_condition_number_20` | 2.407 | 2.40747 | +1.93e-04 | PASS |
| `wang_linton_heave_added_mass_rel_error_20` | 0.255 | 0.254967 | -1.30e-04 | PASS |
| `wang_linton_surge_added_mass_rel_error_20` | 0.268 | 0.267994 | -2.14e-05 | PASS |
| `wang_linton_excitation_force_x_rel_error_20` | 0.36 | 0.36029 | +8.06e-04 | PASS |
| `wang_linton_excitation_force_z_rel_error_20` | 0.37 | 0.369771 | -6.20e-04 | PASS |
| `wang_linton_rotational_added_mass_max_ratio_20` | 1e-20 | 2.49163e-32 | -1.00e+00 | PASS |
| `wang_linton_condition_number_80` | 2.259 | 2.25871 | -1.26e-04 | PASS |
| `wang_linton_heave_added_mass_rel_error_80` | 0.073 | 0.0729881 | -1.63e-04 | PASS |
| `wang_linton_surge_added_mass_rel_error_80` | 0.0802 | 0.0801643 | -4.45e-04 | PASS |
| `wang_linton_excitation_force_x_rel_error_80` | 0.116 | 0.115942 | -5.03e-04 | PASS |
| `wang_linton_excitation_force_z_rel_error_80` | 0.119 | 0.119346 | +2.91e-03 | PASS |
| `wang_linton_rotational_added_mass_max_ratio_80` | 0.00022 | 0.000215099 | -2.23e-02 | PASS |
| `wang_linton_observed_order_a33_80` | 2.12 | 2.11655 | -1.63e-03 | PASS |
| `wang_linton_observed_order_b33_80` | 1.72 | 1.71586 | -2.40e-03 | PASS |
| `wang_linton_observed_order_a11_80` | 2.04 | 2.04218 | +1.07e-03 | PASS |
| `wang_linton_observed_order_fz_80` | 1.91 | 1.91352 | +1.85e-03 | PASS |
| `wang_linton_condition_number_320` | 2.188 | 2.18799 | -2.80e-06 | PASS |
| `wang_linton_heave_added_mass_rel_error_320` | 0.0185 | 0.0185089 | +4.81e-04 | PASS |
| `wang_linton_surge_added_mass_rel_error_320` | 0.0208 | 0.0207657 | -1.65e-03 | PASS |
| `wang_linton_excitation_force_x_rel_error_320` | 0.031 | 0.0310005 | +1.52e-05 | PASS |
| `wang_linton_excitation_force_z_rel_error_320` | 0.032 | 0.0319723 | -8.66e-04 | PASS |
| `wang_linton_rotational_added_mass_max_ratio_320` | 2.6e-05 | 2.64809e-05 | +1.85e-02 | PASS |
| `wang_linton_observed_order_a33_320` | 2.06 | 2.06197 | +9.58e-04 | PASS |
| `wang_linton_observed_order_b33_320` | 1.93 | 1.92898 | -5.30e-04 | PASS |
| `wang_linton_observed_order_a11_320` | 2.03 | 2.03001 | +3.45e-06 | PASS |
| `wang_linton_observed_order_fz_320` | 1.98 | 1.97949 | -2.59e-04 | PASS |

<sub>来源：docs/marine_bem_benchmarks.md §4.11 表 12；数据由 `docs/m3181/outputs/three_d_free_surface_registry.json` 程序化注入。</sub>

**结论**：三维自由液面层（Kelvin/Havelock Green 函数、6-DOF 辐射–绕射、Kochin/Haskind 守恒、免盖板 CHIEF）的文档数字在四位有效数字上全部可复现，未发现腐化；文档不需要改数。
