# TensorBEM 验证欠账清偿：heavy 层 5 条声明数字直接复现与钉桩

**Mission**: m2845（self-directed，generalist 席）
**Date**: 2026-11-03
**Branch**: `agent/m2845`（基于 main @ 29236f0）
**Status**: ✅ PASS（5/5 heavy 条目直接复现，最坏偏差 8.7e-6）

## 1. 欠账是什么

`tensorbem.benchmarks.declared_numbers` 把文档引用的每个判决数字登记为可执行判据，
分 fast/standard/heavy 三层。m280（`dev_report_tensorbem_declared_numbers.md`）结清了
fast+standard 层 11 条，但 **heavy 层 5 条只登记未复现**——1280 面元三维稠密装配在本机
需数十分钟，当时按容量纪律中止并「留给后续有预算的席位」：

| key | 声明值 | 来源 |
|---|---|---|
| ellipsoid_added_mass_1280_x | 4.1679e-3 | README Milestone 3 / docs/classical_benchmarks.md §3 |
| ellipsoid_added_mass_1280_y | 4.4621e-3 | 同上 |
| ellipsoid_added_mass_1280_z | 4.9359e-3 | 同上 |
| ursell_added_mass_256 | 4.2930e-3 | docs/marine_bem_benchmarks.md §4.6 |
| ursell_damping_256 | 6.8412e-3 | 同上 |

m238 扩编审计（`expand_audit_tensorbem.md` §2.3）对椭球 1280 面元还留过一个
「未直接复现」的存疑（其 16 倍细化外推含算术错误，m280 已订正为 4 倍细化、
观测阶 ~2.0 下声明值自洽）——本单用直接复现终结该存疑。

## 2. 怎么做的

1. 本机（loongarch64，python 3.11.6，numpy 2.4.6，commit 29236f0）跑
   `python -m tensorbem.benchmarks.declared_numbers --tiers heavy --output docs/heavy_tier_evidence/declared-numbers-heavy.json`
   ——内部执行椭球 subdivisions=3（1280 面元）与 Ursell 256 面元两个基准，
   逐条对照声明值判 PASS/DRIFT。
2. 补跑收敛序列供报告图：椭球 320 面元（subdivisions=2）、Ursell 64/128/256 面元
   （`docs/heavy_tier_evidence/ellipsoid-320.json、ursell-{64,128,256}.json）。
3. 复跑 fast+standard 层确认全账本绿：`--tiers fast,standard` → 11/11 PASS。
4. 新增 `tests/test_declared_numbers_heavy.py`（7 测试，秒级）：不重跑重计算，
   而是对**已提交的证据工件**独立重判——工件缺失/漂移/环境溯源缺失都会红。
   未来代码改动若使 heavy 数字漂移，正确动作是重跑 heavy 层刷新工件，漂移即在此显形。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 heavy 层判决（`docs/heavy_tier_evidence/declared-numbers-heavy.json`）

| key | 声明 | 实测 | 相对偏差 | 判决 |
|---|---|---|---|---|
| ellipsoid_added_mass_1280_x | 4.1679e-3 | 4.1679364e-3 | 8.74e-6 | PASS |
| ellipsoid_added_mass_1280_y | 4.4621e-3 | 4.4621215e-3 | 4.82e-6 | PASS |
| ellipsoid_added_mass_1280_z | 4.9359e-3 | 4.9359323e-3 | 6.54e-6 | PASS |
| ursell_added_mass_256 | 4.2930e-3 | 4.2929825e-3 | 4.07e-6 | PASS |
| ursell_damping_256 | 6.8412e-3 | 6.8411641e-3 | 5.25e-6 | PASS |

最坏相对偏差 **8.74e-6**——声明值按 5 位有效数字引用，复现精度达舍入级。
fast+standard 层同环境复跑 11/11 PASS（最坏偏差 0.11，为近/远场闭合上界行，容差 5e-2）。

### 3.2 收敛序列（报告图 `docs/figures/heavy_tier_verification.png`）

| 基准 | 面元 | 量 | 值 |
|---|---|---|---|
| 椭球 x/y/z 轴误差 | 320 | — | 1.6313e-2 / 1.7371e-2 / 1.9080e-2 |
| 椭球 x/y/z 轴误差 | 1280 | — | 4.1679e-3 / 4.4621e-3 / 4.9359e-3 |
| Ursell 附加质量 | 64 / 128 / 256 | kg/m | 2587.76 / 2566.83 / 2556.09（参考 2545.05） |
| Ursell 阻尼 | 64 / 128 / 256 | kg/m | 2787.24 / 2754.47 / 2736.75（参考 2717.55） |

观测收敛：椭球 320→1280（4× 面元）误差降 3.91/3.89/3.87 倍（O(h²) 预期 4×）；
Ursell 64→256 附加质量观测阶 1.96、阻尼 1.89。Ursell 面板解自上方单调逼近
Ursell 多极参考解（256 面元距参考 0.44%/0.71%，128 面元为 0.85%/1.34%）。

## 4. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 椭球 1280 面元三轴附加质量相对误差 | 声明值（Lamb 1932 闭式） | 8.7e-6 / 4.8e-6 / 6.5e-6 | **是** |
| Ursell 256 面元附加质量误差 | 声明值（Ursell 多极闭式） | 4.1e-6 | **是** |
| Ursell 256 面元阻尼误差 | 声明值（Ursell 多极闭式） | 5.3e-6 | **是** |
| 椭球 1280 面元附加质量 vs Lamb 闭式 | 闭式 | 4.2e-3 / 4.5e-3 / 4.9e-3 | **是** |
| Ursell 256 面元附加质量/阻尼 vs 多极参考 | 参考解 | 0.44% / 0.71% | **是** |

全部关键量 <3%；声明数字与当前代码一致（无漂移），m238 审计的椭球存疑关闭。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

* 钉桩测试验证的是**工件**而非代码：若内核改动使 heavy 数字漂移，须重跑
  `--tiers heavy` 刷新 `docs/heavy_tier_evidence/declared-numbers-heavy.json`，否则测试红。
  这是有意的——重计算不该进 CI，但证据必须可查。
* 环境溯源已写入工件（commit 29236f0 / numpy 2.4.6 / loongarch64）；跨架构
  （x86 CI）复现若出现 >1e-3 量级偏差，先查 BLAS/NumPy 版本再谈漂移。
* 集群节点（n315/n330）系统 python 无 numpy 且无外网，本单重计算落在本机
  （load15≈60 < 100，符合容量纪律）；集群跑 BEM 需先补 venv。
* 未做：5120 面元椭球（~16× 1280 成本，超出单干预算）；heavy 层并行化。

## 6. 文件清单

* `docs/heavy_tier_evidence/declared-numbers-heavy.json` —— heavy 层判决工件（证据）
* `docs/heavy_tier_evidence/declared-numbers-fast-standard.json` —— 全账本其余 11 条复跑
* `docs/heavy_tier_evidence/ellipsoid-320.json、ursell-{64,128,256}.json —— 收敛序列
* `tests/test_declared_numbers_heavy.py` —— 7 个秒级钉桩测试
* `docs/figures/heavy_tier_verification.png` —— 收敛图
* `dev_report_tensorbem-heavy-tier-verification.md` —— 本报告
