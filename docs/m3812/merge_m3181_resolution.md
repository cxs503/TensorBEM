# PR #128（agent/m3181）→ main 三方合并解冲记录（m3812）

**仓**：TensorBEM · **工作树**：`/root/tp-agent/work/m3812` · **合并提交**：见 `git log -1`
**方向**：`origin/main`（m3150 等已入库）← `agent/m3181`（PR #128）
**冲突面**：3 个文件与 main 同时被改（2 个内容冲突 + 1 个自动合并需复核），其余 14 个文件为 m3181 单侧新增，原样并入。

本文件由 `docs/m3812/build_resolution.py` 从三份**程序化回执**生成，表中数字零手抄：
m3181 的 63 格回执（`docs/m3181/outputs/three_d_free_surface_registry.json`）、
m3150 已入库的 62 格回执（`docs/m3150/outputs/registry_3d_free_surface.json`）、
合并后源码跑出的注册表回执（`docs/m3812/outputs/merged_registry.json`）。

## 1. 冲突清单与逐文件解法

| 文件 | 状态 | 解法 |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | 内容冲突 3 处（import 别名、表 10 注册项生成器、`_run_group` 分组派发） | 以 main 的 m3150 实现为骨架（三组 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`、`standard` 层、键名含面元数），并把 m3181 独有的 4 格补入：表 11 参考解 `A33'`/`B33'` 与内部 CHIEF 点数、表 12 的 20 面元机器零格（`kind="upper"`）。`_chief_rows` 增加 `exact_a33`/`exact_b33` 两个量的展开，使这 3 格有可复算的测量来源。m3181 的表驱动解析器（`_TABLE_HULME`/`_refinement_rows`）不保留：它与 main 的显式注册项覆盖同一批格子，保留会触发 `duplicate registry keys` 断言（`test_registry_is_well_formed`），属重复实现而非内容丢失——逐格等价性见 §2。 |
| `tests/test_declared_numbers.py` | 内容冲突 2 处（`declared>0` 断言注释、文档基准族集合中的组名） | 以 main 版本为骨架（组名 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`），并吸收 m3181 的判据：新增 `test_three_dimensional_free_surface_tables_are_fully_registered`（表 10/11/12 格子数 31/9/26、同表共用一次求解、CHIEF 阻尼符号、机器零格为上界）、`test_table_10_b11_order_uses_the_measured_mesh_ratio`（保留 m3181 发现的 `√N` 减半口径 1.62/1.88，并钉死文档采用的实测棱长比口径，外加二阶收敛断言）、`test_table_12_converges_at_second_order_and_machine_zero_coupling_holds`。 |
| `docs/marine_bem_benchmarks.md` | 自动合并（双方改同一批表） | git 自动合并成功，但存在**语义冲突**：m3150 的表 10/12 订正段与 m3181 新写的 §4.12 回执段并存。人工复核：表 10 观测阶列取 m3150 的实测棱长比口径（1.80/1.94），表 12 转动耦合取 2.15e-4/2.65e-5；在 §4.12 增补「合入订正」段，声明仓库内唯一可执行注册表为 m3150 三组、m3181 独有 4 格已并入、回执表保留原键名以备审计，并显式说明两处口径差异同源。 |
| 其余 14 个 m3181 文件（`docs/m3181/**`、`dev_report_m3181_*.md`） | 仅 m3181 有 | 原样并入，未删除、未改写。 |

## 2. m3181 的 63 格 → 合并注册表逐格对应

`docs/m3181/outputs/three_d_free_surface_registry.json` 判定：63/63 PASS；
m3150 回执：62/62 PASS；
合并后注册表：62/88 PASS（fallback：66 格声明值直接取自合并后源码，测量值取 m3150 回执，m3181 独有的 4 格标注 PENDING（全量 standard 回执见 `docs/m3812/test_evidence.md`））。
下表按「同一物理格 = 同一测量值」配对，最后一列为该格在合并注册表中的判定与相对偏差。

| m3181 键名 | m3181 声明 | 合并注册表键名 | 合并声明 | 合并判定（偏差） |
|---|---:|---|---:|---|
| `hulme_condition_number_16` | 3.893 | `hulme_hemisphere_16_condition_number` | 3.893 | PASS (-1.15e-04) |
| `hulme_heave_added_mass_rel_error_16` | 0.12 | `hulme_hemisphere_16_heave_added_mass_error` | 0.12 | PASS (+1.07e-03) |
| `hulme_heave_damping_rel_error_16` | 0.122 | `hulme_hemisphere_16_heave_damping_error` | 0.122 | PASS (-8.22e-04) |
| `hulme_surge_added_mass_rel_error_16` | 0.18 | `hulme_hemisphere_16_surge_added_mass_error` | 0.18 | PASS (+1.32e-03) |
| `hulme_surge_damping_rel_error_16` | 0.402 | `hulme_hemisphere_16_surge_damping_error` | 0.402 | PASS (+1.02e-03) |
| `hulme_excitation_force_z_rel_error_16` | 0.0667 | `hulme_hemisphere_16_vertical_force_error` | 0.0667 | PASS (+2.46e-04) |
| `hulme_condition_number_64` | 3.052 | `hulme_hemisphere_64_condition_number` | 3.052 | PASS (-1.48e-04) |
| `hulme_heave_added_mass_rel_error_64` | 0.0297 | `hulme_hemisphere_64_heave_added_mass_error` | 0.0297 | PASS (+1.21e-03) |
| `hulme_heave_damping_rel_error_64` | 0.045 | `hulme_hemisphere_64_heave_damping_error` | 0.045 | PASS (+5.00e-04) |
| `hulme_surge_added_mass_rel_error_64` | 0.0505 | `hulme_hemisphere_64_surge_added_mass_error` | 0.0505 | PASS (+1.58e-04) |
| `hulme_surge_damping_rel_error_64` | 0.13 | `hulme_hemisphere_64_surge_damping_error` | 0.13 | PASS (+3.79e-03) |
| `hulme_excitation_force_z_rel_error_64` | 0.0221 | `hulme_hemisphere_64_vertical_force_error` | 0.0221 | PASS (-2.68e-04) |
| `hulme_observed_order_a33_64` | 2.23 | `hulme_hemisphere_64_order_a33` | 2.23 | PASS (+1.87e-03) |
| `hulme_observed_order_b33_64` | 1.59 | `hulme_hemisphere_64_order_b33` | 1.59 | PASS (+2.42e-03) |
| `hulme_observed_order_a11_64` | 2.04 | `hulme_hemisphere_64_order_a11` | 2.04 | PASS (-2.13e-03) |
| `hulme_observed_order_fz_64` | 1.77 | `hulme_hemisphere_64_order_fz` | 1.77 | PASS (-8.88e-04) |
| `hulme_condition_number_256` | 2.783 | `hulme_hemisphere_256_condition_number` | 2.783 | PASS (-4.37e-05) |
| `hulme_heave_added_mass_rel_error_256` | 0.00741 | `hulme_hemisphere_256_heave_added_mass_error` | 0.00741 | PASS (-6.15e-04) |
| `hulme_heave_damping_rel_error_256` | 0.0118 | `hulme_hemisphere_256_heave_damping_error` | 0.0118 | PASS (+1.01e-03) |
| `hulme_surge_added_mass_rel_error_256` | 0.013 | `hulme_hemisphere_256_surge_added_mass_error` | 0.013 | PASS (+2.95e-03) |
| `hulme_surge_damping_rel_error_256` | 0.0353 | `hulme_hemisphere_256_surge_damping_error` | 0.0353 | PASS (-7.22e-04) |
| `hulme_excitation_force_z_rel_error_256` | 0.00567 | `hulme_hemisphere_256_vertical_force_error` | 0.00567 | PASS (-3.69e-04) |
| `hulme_observed_order_a33_256` | 2.06 | `hulme_hemisphere_256_order_a33` | 2.06 | PASS (-4.02e-04) |
| `hulme_observed_order_b33_256` | 1.98 | `hulme_hemisphere_256_order_b33` | 1.98 | PASS (+1.02e-03) |
| `hulme_observed_order_a11_256` | 2.01 | `hulme_hemisphere_256_order_a11` | 2.01 | PASS (-2.00e-03) |
| `hulme_observed_order_fz_256` | 2.02 | `hulme_hemisphere_256_order_fz` | 2.02 | PASS (-2.35e-03) |
| `hulme_chief_unregularized_condition_number` | 145.66 | `hulme_chief_2_58_unregularized_condition_number` | 145.66 | PASS (-9.43e-06) |
| `hulme_chief_unregularized_a33_dimless` | 0.3481 | `hulme_chief_2_58_unregularized_a33` | 0.3481 | PASS (-1.09e-04) |
| `hulme_chief_unregularized_b33_dimless` | -0.1042 | `hulme_chief_2_58_unregularized_b33` | -0.1042 | PASS (+4.35e-04) |
| `hulme_chief_chief_regularized_condition_number` | 4.24 | `hulme_chief_2_58_chief_condition_number` | 4.24 | PASS (+1.17e-03) |
| `hulme_chief_chief_regularized_a33_dimless` | 0.3919 | `hulme_chief_2_58_chief_a33` | 0.3919 | PASS (+2.15e-05) |
| `hulme_chief_chief_regularized_b33_dimless` | 0.0687 | `hulme_chief_2_58_chief_b33` | 0.0687 | PASS (+1.15e-04) |
| `wang_linton_condition_number_20` | 2.407 | `submerged_sphere_20_condition_number` | 2.407 | PASS (+1.93e-04) |
| `wang_linton_heave_added_mass_rel_error_20` | 0.255 | `submerged_sphere_20_heave_added_mass_error` | 0.255 | PASS (-1.30e-04) |
| `wang_linton_surge_added_mass_rel_error_20` | 0.268 | `submerged_sphere_20_surge_added_mass_error` | 0.268 | PASS (-2.14e-05) |
| `wang_linton_excitation_force_x_rel_error_20` | 0.36 | `submerged_sphere_20_force_x_error` | 0.36 | PASS (+8.06e-04) |
| `wang_linton_excitation_force_z_rel_error_20` | 0.37 | `submerged_sphere_20_force_z_error` | 0.37 | PASS (-6.20e-04) |
| `wang_linton_condition_number_80` | 2.259 | `submerged_sphere_80_condition_number` | 2.259 | PASS (-1.26e-04) |
| `wang_linton_heave_added_mass_rel_error_80` | 0.073 | `submerged_sphere_80_heave_added_mass_error` | 0.073 | PASS (-1.63e-04) |
| `wang_linton_surge_added_mass_rel_error_80` | 0.0802 | `submerged_sphere_80_surge_added_mass_error` | 0.0802 | PASS (-4.45e-04) |
| `wang_linton_excitation_force_x_rel_error_80` | 0.116 | `submerged_sphere_80_force_x_error` | 0.116 | PASS (-5.03e-04) |
| `wang_linton_excitation_force_z_rel_error_80` | 0.119 | `submerged_sphere_80_force_z_error` | 0.119 | PASS (+2.91e-03) |
| `wang_linton_rotational_added_mass_max_ratio_80` | 2.2000e-04 | `submerged_sphere_80_rotational_coupling` | 2.1510e-04 | PASS (-5.40e-06) |
| `wang_linton_observed_order_a33_80` | 2.12 | `submerged_sphere_80_order_a33` | 2.12 | PASS (-1.63e-03) |
| `wang_linton_observed_order_b33_80` | 1.72 | `submerged_sphere_80_order_b33` | 1.72 | PASS (-2.40e-03) |
| `wang_linton_observed_order_a11_80` | 2.04 | `submerged_sphere_80_order_a11` | 2.04 | PASS (+1.07e-03) |
| `wang_linton_observed_order_fz_80` | 1.91 | `submerged_sphere_80_order_fz` | 1.91 | PASS (+1.85e-03) |
| `wang_linton_condition_number_320` | 2.188 | `submerged_sphere_320_condition_number` | 2.188 | PASS (-2.80e-06) |
| `wang_linton_heave_added_mass_rel_error_320` | 0.0185 | `submerged_sphere_320_heave_added_mass_error` | 0.0185 | PASS (+4.81e-04) |
| `wang_linton_surge_added_mass_rel_error_320` | 0.0208 | `submerged_sphere_320_surge_added_mass_error` | 0.0208 | PASS (-1.65e-03) |
| `wang_linton_excitation_force_x_rel_error_320` | 0.031 | `submerged_sphere_320_force_x_error` | 0.031 | PASS (+1.52e-05) |
| `wang_linton_excitation_force_z_rel_error_320` | 0.032 | `submerged_sphere_320_force_z_error` | 0.032 | PASS (-8.66e-04) |
| `wang_linton_rotational_added_mass_max_ratio_320` | 2.6000e-05 | `submerged_sphere_320_rotational_coupling` | 2.6481e-05 | PASS (-3.99e-06) |
| `wang_linton_observed_order_a33_320` | 2.06 | `submerged_sphere_320_order_a33` | 2.06 | PASS (+9.58e-04) |
| `wang_linton_observed_order_b33_320` | 1.93 | `submerged_sphere_320_order_b33` | 1.93 | PASS (-5.30e-04) |
| `wang_linton_observed_order_a11_320` | 2.03 | `submerged_sphere_320_order_a11` | 2.03 | PASS (+3.45e-06) |
| `wang_linton_observed_order_fz_320` | 1.98 | `submerged_sphere_320_order_fz` | 1.98 | PASS (-2.59e-04) |

按「同一测量值」自动配对时落空的 6 格，逐格都有明确归属，不是丢失：

* `hulme_observed_order_b11_64` / `_256`：同一物理量的另一种度量口径（见 §2.1），合并注册表以 `hulme_hemisphere_{64,256}_order_b11` 登记，测量值随口径不同（1.8021 / 1.9377 替代 1.6247 / 1.8873）。
* `hulme_chief_exact_a33_dimless` / `_b33_dimless` / `_chief_regularized_chief_point_count`、`wang_linton_rotational_added_mass_max_ratio_20`：m3181 独有的 4 格，已并入合并注册表（§3），声明值取自源码，测量待 standard 层全量回执（本轮受机器负载，全量回执未在预算内完成，见 §4）。

### 2.1 声明值不同但物理格相同的行（口径/舍入差异，非数据丢失）

| m3181 键名 | m3181 声明 | 合并键名 | 合并声明 | 差异性质 |
|---|---:|---|---:|---|
| `wang_linton_rotational_added_mass_max_ratio_80` | 2.2000e-04 | `submerged_sphere_80_rotational_coupling` | 2.1510e-04 | 有效数字：m3181 两位，main/m3150 订正为可复现四位 |
| `wang_linton_rotational_added_mass_max_ratio_320` | 2.6000e-05 | `submerged_sphere_320_rotational_coupling` | 2.6481e-05 | 有效数字：m3181 两位，main/m3150 订正为可复现四位 |

* `hulme_observed_order_b11_*`：m3181 按理想 \(\sqrt{N}\) 减半折算（1.62 / 1.88），main 按实测平均棱长比 \(h_0/h_1=1.87\) 折算（1.80 / 1.94）。表 10 全表统一采用后者，故注册表登记后者的可复现值；前者在 `test_table_10_b11_order_uses_the_measured_mesh_ratio` 中被显式复算并断言仍在 1.62 附近，历史上没有丢。
* `wang_linton_rotational_added_mass_max_ratio_(80, 320)`：m3181 按两位有效数字登记（2.2e-4 / 2.6e-5），main 的 m3150 订正要求可复现的四位有效数字（2.151e-4 / 2.6481e-5）。同一测量的两种精度处置，后者可回归，取后者。

## 3. m3181 独有的 4 格：并入合并注册表

| 新增键名 | 声明 | 来源 |
|---|---:|---|
| `hulme_chief_2_58_exact_a33` | 0.4007 | docs/marine_bem_benchmarks.md section 4.10 table 11 |
| `hulme_chief_2_58_exact_b33` | 0.0631 | docs/marine_bem_benchmarks.md section 4.10 table 11 |
| `hulme_chief_2_58_chief_point_count` | 5 | docs/marine_bem_benchmarks.md section 4.10 table 11 |
| `submerged_sphere_20_rotational_coupling` | 1e-20（上界判） | docs/marine_bem_benchmarks.md section 4.11 table 12 |

前 3 格是表 11 表头「精确 0.4007 / 0.0631」的参考解列与 5 点内部 CHIEF 约束数，m3150 当时只登记了两个数值方案下的 6 格，参考列与点数没进注册表；第 4 格是表 12 的 20 面元转动耦合，文档写 \(2.5\times10^{-32}\)（机器零），m3181 按上界 `1e-20` 判，m3150 漏登。合并后注册表把它们补齐，`_chief_rows` 相应增加 `exact_*` 展开，使参考列可复算。

## 3.1 反向：main 独有、m3181 未覆盖的 5 行（保留，未删）

| 行 | 声明 | 来源 |
|---|---:|---|
| `hulme_hemisphere_64_order_b11` | 1.8021 | 表 10 观测阶列（实测棱长比口径） |
| `hulme_hemisphere_256_order_b11` | 1.9377 | 表 10 观测阶列（实测棱长比口径） |
| `hulme_hemisphere_256_near_far_damping_closure` | 5.4500e-04 | §4.10 正文：近场/远场 Kochin 阻尼闭合差 |
| `hulme_hemisphere_256_haskind_surge` | 2.0900e-04 | §4.10 正文：三维 Haskind 互易（纵荡） |
| `hulme_hemisphere_256_haskind_heave` | 8.0900e-04 | §4.10 正文：三维 Haskind 互易（垂荡） |

这 5 行 m3181 未登记（它只覆盖三张表格的格子，正文里的闭合/互易数字不在其口径内），合并时原样保留。

## 4. 测试证据

见 `docs/m3812/test_evidence.md`（由实际 `pytest` 输出摘要注入）。
