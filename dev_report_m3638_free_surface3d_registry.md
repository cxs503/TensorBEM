# TensorBEM 三维深水自由液面声明数字进入注册表（表 10/11/12, m3638）

EDIT_WHITELIST: src/tensorbem/benchmarks/declared_numbers.py, tests/test_declared_numbers.py, docs/marine_bem_benchmarks.md, outputs/m3638/, dev_report_m3638_free_surface3d_registry.md

## 1. 欠账

m3059 在结清 1280 面板验证欠账时明确登记了一条后续项：三维自由液面（Hulme 漂浮半球 / Wang–Linton 潜没球体）虽已由 m2609、m3280 等测试钉桩，但**未纳入 `declared_numbers` 注册表**。注册表是文档数字唯一的机器判据，其覆盖面在三维侧止于无界流椭球与内域谐波；表 10/11/12 的 44 个数只有正文，没有一条能被 CI 复算。

## 2. 做了什么

* `src/tensorbem/benchmarks/declared_numbers.py`：新增三个组 `hulme`（21 条，表 10 + §4.10 正文的近/远场与 Haskind 互易差）、`hulme_chief`（7 条，表 11）、`submerged_sphere`（16 条，表 12），成本层均为 `standard`；runner 复用既有 benchmark 入口 `run_refinement_study` / `run_irregular_frequency_study`，靠注册表自带的 `(group, parameters)` 缓存让一张表只解一次。
* `tests/test_declared_numbers.py`：覆盖面判据加入三组；新增 `test_standard_tier_guards_the_three_dimensional_free_surface_tables`；新增 `test_chief_regularisation_restores_passivity`（判符号翻转 + 条件数降到 1/10）；`standard` 层改为会话内复算一次（`lru_cache`），此前三个测试各重跑一遍 256 面元 Ursell 阶梯。
* `docs/marine_bem_benchmarks.md` §5 第 6 条扩写 + 新增第 7 条，写明登记范围、复算命令与实测偏差。

## 3. 结果（程序化注入，44 条）

| 注册项 | 声明 | 实测 | 相对偏差 | 判定 |
|---|---:|---:|---:|---|
| `hulme_condition_number_16` | 3.893 | 3.89255 | -1.15e-04 | PASS |
| `hulme_heave_added_mass_16` | 0.12 | 0.120128 | +1.07e-03 | PASS |
| `hulme_heave_damping_16` | 0.122 | 0.1219 | -8.22e-04 | PASS |
| `hulme_surge_added_mass_16` | 0.18 | 0.180237 | +1.32e-03 | PASS |
| `hulme_surge_damping_16` | 0.402 | 0.40241 | +1.02e-03 | PASS |
| `hulme_excitation_z_16` | 0.0667 | 0.0667164 | +2.46e-04 | PASS |
| `hulme_condition_number_64` | 3.052 | 3.05155 | -1.48e-04 | PASS |
| `hulme_heave_added_mass_64` | 0.0297 | 0.0297359 | +1.21e-03 | PASS |
| `hulme_condition_number_256` | 2.783 | 2.78288 | -4.37e-05 | PASS |
| `hulme_heave_added_mass_256` | 0.00741 | 0.00740544 | -6.15e-04 | PASS |
| `hulme_heave_damping_256` | 0.0118 | 0.0118119 | +1.01e-03 | PASS |
| `hulme_surge_added_mass_256` | 0.013 | 0.0130384 | +2.95e-03 | PASS |
| `hulme_surge_damping_256` | 0.0353 | 0.0352745 | -7.22e-04 | PASS |
| `hulme_excitation_z_256` | 0.00567 | 0.00566791 | -3.69e-04 | PASS |
| `hulme_order_heave_added_mass_256` | 2.06 | 2.05917 | -4.02e-04 | PASS |
| `hulme_order_heave_damping_256` | 1.98 | 1.98201 | +1.02e-03 | PASS |
| `hulme_order_surge_added_mass_256` | 2.01 | 2.00598 | -2.00e-03 | PASS |
| `hulme_order_excitation_z_256` | 2.02 | 2.01525 | -2.35e-03 | PASS |
| `hulme_far_field_damping_256` | 0.000545 | 0.000544783 | -3.99e-04 | PASS |
| `hulme_haskind_surge_256` | 0.000209 | 0.000208918 | -3.92e-04 | PASS |
| `hulme_haskind_heave_256` | 0.000809 | 0.000808537 | -5.73e-04 | PASS |
| `linton_condition_number_20` | 2.407 | 2.40747 | +1.93e-04 | PASS |
| `linton_heave_added_mass_20` | 0.255 | 0.254967 | -1.30e-04 | PASS |
| `linton_surge_added_mass_20` | 0.268 | 0.267994 | -2.14e-05 | PASS |
| `linton_excitation_x_20` | 0.36 | 0.36029 | +8.06e-04 | PASS |
| `linton_excitation_z_20` | 0.37 | 0.369771 | -6.20e-04 | PASS |
| `linton_rotational_coupling_80` | 0.001 | 0.000215099 | -7.85e-01 | PASS |
| `linton_condition_number_320` | 2.188 | 2.18799 | -2.80e-06 | PASS |
| `linton_heave_added_mass_320` | 0.0185 | 0.0185089 | +4.81e-04 | PASS |
| `linton_surge_added_mass_320` | 0.0208 | 0.0207657 | -1.65e-03 | PASS |
| `linton_excitation_x_320` | 0.031 | 0.0310005 | +1.52e-05 | PASS |
| `linton_excitation_z_320` | 0.032 | 0.0319723 | -8.66e-04 | PASS |
| `linton_rotational_coupling_320` | 0.0001 | 2.64809e-05 | -7.35e-01 | PASS |
| `linton_order_heave_added_mass_320` | 2.06 | 2.06197 | +9.58e-04 | PASS |
| `linton_order_heave_damping_320` | 1.93 | 1.92898 | -5.30e-04 | PASS |
| `linton_order_surge_added_mass_320` | 2.03 | 2.03001 | +3.45e-06 | PASS |
| `linton_order_excitation_z_320` | 1.98 | 1.97949 | -2.59e-04 | PASS |
| `hulme_chief_condition_unregularized` | 145.7 | 145.659 | -9.43e-06 | PASS |
| `hulme_chief_heave_added_mass_unregularized` | 0.3481 | 0.348062 | -1.09e-04 | PASS |
| `hulme_chief_heave_damping_unregularized` | -0.1042 | -0.104155 | +4.35e-04 | PASS |
| `hulme_chief_condition_augmented` | 4.24 | 4.24497 | +1.17e-03 | PASS |
| `hulme_chief_heave_added_mass_augmented` | 0.3919 | 0.391908 | +2.15e-05 | PASS |
| `hulme_chief_heave_damping_augmented` | 0.0687 | 0.0687079 | +1.15e-04 | PASS |
| `hulme_chief_point_count` | 5 | 5 | +0.00e+00 | PASS |

## 4. 图

`outputs/m3638/fig_registry_3d_free_surface.png`：左=对有独立参考解（Hulme 级数 / Wang–Linton 多极子）的收敛阶梯与 $O(h^2)$ 参考斜率；右=各组最坏相对偏差与 1.5e-2 漂移阈值。

## 5. 注意点

* 成本：三组合计 83 s（hulme+submerged_sphere 55.8 s，hulme_chief 27.2 s）。沙箱 shell 有 120 s 墙钟上限，故按组分两次进程复算并合并 JSON（`outputs/m3638/verify_groups.py`）；CI 无此限制。
* `upper` 型注册项（Wang–Linton 转动耦合比）的偏差按「实测−声明」计，数值上会显示很大的负偏差，这是上界声明的语义，不是漂移；正文的「最坏偏差」只在 `value` 型上统计。
* 表 11 的 `chief_point_count=5` 以 0 容差登记（整数），作为自动布点的回归锚。
* 未做：未改任何求解器实现；未把三维自由液面纳入 `fast` 层（成本不允许）；有限水深三维求解器仍是 Roadmap 缺口。

## 6. 误差判定

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---:|---|
| Hulme 垂荡附加质量 256 面元 | Hulme (1982) 级数 | 7.41e-3 | ✅ |
| Hulme 纵荡阻尼 256 面元 | 同上 | 3.53e-2（离散误差，m2609 证明 1024 面元降至 9.0e-3） | ❌（3.53%），已由门槛测试在 1024 面元覆盖 |
| Wang–Linton 垂荡附加质量 320 面元 | Wang/Linton 多极子 | 1.85e-2 | ✅ |
| Wang–Linton 波浪力 320 面元 | 同上 | 3.10e-2 / 3.20e-2 | ❌（3.1–3.2%），二阶收敛、加密可降 |
| 近场 vs 远场 Kochin 阻尼 | 能量恒等式 | 5.45e-4 | ✅ |
| 直接绕射 vs 边界 Haskind | 互易律 | 2.09e-4 / 8.09e-4 | ✅ |
| CHIEF 增广后 $B_{33}'$ | Hulme 精确值 0.0631 | 8.85e-2（幅值），但符号与耗散性恢复 | ❌ 幅值（8.9%），符号结论 ✅ |

结论：物理不变量（能量、互易）与 256/320 面元的附加质量均在 3% 内；**两处超过 3%** 的是同一网格下的阻尼/波浪力，属 $O(h^2)$ 离散误差而非实现偏差，已在正文与门槛测试中分别声明与覆盖，不作粉饰。

## 7. 验证方式

```
PYTHONPATH=src python3 outputs/m3638/verify_groups.py hulme submerged_sphere   # 55.8 s, 37 条 0 DRIFT
PYTHONPATH=src python3 outputs/m3638/verify_groups.py hulme_chief              # 27.2 s,  7 条 0 DRIFT
PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q
```
