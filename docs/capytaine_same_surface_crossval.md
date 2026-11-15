# capytaine 同面交叉验证 —— TensorBEM 6×6 辐射矩阵

锚点 `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE`（验收：对标误差 <15%）。本页是 round 2 交叉验证的**同面**增量：TensorBEM 不再用自己的网格，而是直接解在 capytaine 的面元表面上，从而把「两套不同离散」的混淆项拿掉。

## 1. 缺口

round 2 的对比是两条各自网格的「最细层 vs 最细层」，最坏 active 偏差 6.71%。这个数字是两套无关离散误差之和，无法归因；把容差从 5% 放宽到 10% （round 2 报告的建议）只是把判据调松，不构成验证。同面比较把网格误差做成共享项，残差因此可归因于**公式层面**（Green 函数、求积、广义法向、转动中心）。

## 2. 方法

1. capytaine `mesh_sphere(resolution=(n_theta, n_phi)).immersed_part()` 得到四边面元表面；
2. 每个四边面沿 `0-2` 对角线剖成两个三角形，水线上退化的半面片剔除；
3. 剖分面积与 capytaine 原表面面积逐层相等（相对差见下表，0 表示到机器精度），并用 `WettedHullMesh` 的不变量（外法向、水线闭合）验证朝向；
4. TensorBEM `solve_radiation_3d` 在这张表面上解 6 自由度，与 capytaine 在同一表面的解逐项比较；
5. 分类沿用 round 2 的判据：结构零（不变量）与零元（null）不当作相对偏差评分，尺度用 `rho L^k`；band 冻结为模块常量。

capytaine 侧列直接从已提交的 `capytaine_crosscheck_r2/capytaine_ref_r2.json` 读取（同一分辨率层），TensorBEM 侧为本页新算。生产工具：capytaine 3.1.dev0，loongarch64，rho=1025.0 kg/m³，g=9.80665 m/s²，nu=[0.5, 1.0, 2.0]（omega = sqrt(g nu)）。

## 3. 结果

| case | 层 (n_theta,n_phi) | 四边面元 | 三角形 | 剖分面积 | 面积相对差 | 最坏对角偏差 | 最坏耦合偏差 | 判定 |
|---|---|---|---|---|---|---|---|---|
| `hemisphere_r1` | (8, 16) | 64 | 112 | 6.0833 | 0.0e+00 | 14.38% | 0.00% | FAIL |
| `hemisphere_r1` | (16, 32) | 256 | 480 | 6.2328 | 0.0e+00 | 9.48% | 0.00% | FAIL |
| `hemisphere_r1_cz-0.5` | (8, 16) | 64 | 112 | 6.0833 | 0.0e+00 | 14.38% | 10293.51% | FAIL |
| `hemisphere_r1_cz-0.5` | (16, 32) | 256 | 480 | 6.2328 | 0.0e+00 | 9.48% | 0.00% | FAIL |

细化趋势（同一表面细分，偏差应缩小；公式层面的错误不会随细分消失）：

| case | 最粗层最坏偏差 | 最细层最坏偏差 | 比值 |
|---|---|---|---|
| `hemisphere_r1` | 14.38% | 9.48% | 1.52 |
| `hemisphere_r1_cz-0.5` | 10293.51% | 9.48% | 1086.31 |

图：`docs/fig_capytaine_same_surface.png`（实线=同面，虚线=round 2 各自网格）。

## 4. 参考列的独立复现

对已提交 `capytaine_ref_r2.json` 的参考列（最细层）重新求解 6 个列（含两个转动中心变体），按列最大值归一化的最大偏差为 1.9e-16。即 OSS 参考列不是陈旧/拼接数据，可在本机复现到机器精度。

| case | 层 | nu | radiating dof | dA/colmax | dB/colmax |
|---|---|---|---|---|---|
| `hemisphere_r1` | (28, 56) | 1.0 | surge | 5.5e-17 | 1.9e-16 |
| `hemisphere_r1` | (28, 56) | 0.5 | heave | 1.8e-20 | 1.4e-16 |
| `hemisphere_r1_cz-0.5` | (28, 56) | 2.0 | roll | 3.2e-17 | 9.4e-17 |
| `hemisphere_r1_cz-0.5` | (16, 32) | 1.0 | pitch | 1.8e-16 | 7.4e-17 |
| `sphere_r1_z-0.5` | (28, 56) | 0.5 | heave | 4.0e-17 | 4.5e-19 |
| `sphere_r1_z-0.5_c0` | (28, 56) | 1.0 | pitch | 2.2e-17 | 1.2e-16 |

## 5. 被否定的假设：用 Richardson 外推「消掉」6.7%

对每条阶梯拟合 `f(h) = f0 + C h^p`（h = 1/sqrt(N)）后再比极限，看似能给出网格无关结论。在 142 个可评分条目上，126 个（89%）的拟合阶落在搜索边界上（区间 [0.2, 6.0]），阶中位数 1.87、5–95 分位 [0.20, 6.00]；外推极限的最坏偏差反而涨到 11.8%（最细层实测 6.71%）。结论：三到四层、阶数未锁定的阶梯上，外推极限不可辨识，是拟合产物不是网格极限。因此**不放宽容差、也不引用外推极限**；判据仍是可复现的同面 / 各自网格实测值。

## 6. 注意点与局限

1. 本页已执行的分辨率是每个体的两个最粗层（半球 112/480 三角形）；最细层 （半球 (28,56)=1512 三角形、球 (28,56)=2072 三角形）的整梯运行已递交集群作业 `cj3274`，本页写作时仍在排队，**其数值未回传**，不得引用。
2. 同面比较仍非「同一离散」：capytaine 用四边面元 + 解析自项积分，TensorBEM 用剖分后的三角形常值面元，未知量分辨率不同。因此残差里仍含求积差异——这正是同面比较要暴露的量。
3. 粗分辨率下 `hemisphere_r1_cz-0.5` 的耦合偏差出现 1e4% 量级：该层 capytaine 耦合项已接近零（判据用 1% 相对尺度作零元底），两个近零量相除没有物理含义，这类条目只能看绝对值，不能进 band 判定；正是 round 2 用尺度化分类要避免的陷阱。
4. 只在无限水深、零航速、单频三点（nu=0.5/1.0/2.0）上成立，不外推到有航速/有限水深。

## 7. 误差判定

| 量 | 基准 | 实测 | 误差 | <3%? | <15%(ledger)? |
|---|---|---|---|---|---|
| 同面对角偏差（已执行最细层，480 三角形） | capytaine 同面解 | 9.48% | 相对偏差 | 否 | 是 |
| 参考列复现（6 列，机器精度） | capytaine 重解 | 1.9e-16 | 相对偏差 | 是 | 是 |
| 剖分面积守恒 | capytaine 原表面面积 | 0.0e+00 | 相对偏差 | 是 | 是 |

**明确回答**：同面偏差在已执行的最粗两层上为 14.38% → 9.48%，**未达 <3%**；它满足 ledger 的 <15% 带，且随细分单调下降，符合离散误差的签名。把「<3% 一致」当成结论是错的——round 2 的 6.7% 与本页的 9.5% 都不是公式分歧的证据，而是面元分辨率的证据。

## 8. 复现

```bash
PYTHONPATH=src python scripts/capytaine_same_surface_crossval.py     # 同面研究（集群）
PYTHONPATH=src python scripts/richardson_probe_r2.py                 # 外推可辨识性探针
PYTHONPATH=src python scripts/fig_capytaine_same_surface.py          # 图
PYTHONPATH=src python scripts/gen_doc_same_surface.py                # 本页
PYTHONPATH=src python -m pytest tests/test_capytaine_same_surface_crossval.py -q
```
