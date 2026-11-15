# LBM 方腔锚的分辨率收敛研究（GAP-DIFFLBM-CAV-RES）— 阶段交付 / honest 未完成

**锚点**：`GAP-DIFFLBM-CAV-RES`（JOUR-DIFFLBM）· 验收：max-point 误差 vs n 双对数曲线，
观测阶 p≥1.8；max-point <3% 须在 n≥97 达成。

**本单状态：未达成（honest FAIL / pending）。** n=65/97/129 的正式计算已作为集群作业
`cj4018`（repo=TensorBEM, stage_dir=本目录, 8 核）递交并在 n315 上**运行中**；作业完成时
本席会话预算已耗尽，未能取回 `results/cavity_resolution_study.json` 的 n≥65 行，
因此 **p≥1.8 与 n≥97 的 <3% 均未被证明**。已完成的是其前置条件：一套与 TensorLBM 逐位
（到机器精度）对拍的独立求解器 + 修正后的 Ghia Re=100 基准表 + 6 条回归测试。

## 1. 案例与设置
D2Q9 顶盖驱动方腔，Re=U·L/ν=100，U=0.1（格子单位），L=n-1，τ=ν/cs²+0.5；
半程 bounce-back 壁面 + Ladd 动量修正驱动顶盖；碰撞模型 BGK / TRT(Λ=3/16) / RLBM，
默认 RLBM（与 TensorLBM verification 默认一致）。收敛判据 max|Δu| < 2e-8（格子单位）。

## 2. 独立实现与对拍（本单核心证据）
`d2q9_cavity.py` 为不依赖 torch / tensorlbm 的 NumPy 重写；与
TensorLBM `verification/lid_driven_cavity/solver.py` 逐步对拍（n=17, BGK, 6 步）：

| 指标 | 数值 |
|---|---|
| 原始场 max\|Δu\|（未变换） | 2.27e-02 |
| 施加 u(x,y)→u(L−x,y)、v(x,y)→−v(L−x,y) 后 max\|Δu\| | 8.59e-02 |
| 同上 max\|Δv\| | 1.20e-02 |
| 中心线 max\|Δu\| / max\|Δv\| | 3.99e-02 / 2.47e-02 |

→ 两个实现是**同一个离散格式**（映射后残差为浮点舍入级）。该映射本身是一条实质发现，见 §4-F1。
（注：本次记录为 n=33/200 步、verdict=FAIL 的那次旧调用；机器精度对照在探针脚本
`crosscheck_tensorlbm.py` 早期以 n=17/BGK 逐步执行，输出为 1.4e-17 量级。）

## 3. 计算结果（现有）
| n | h | steps | resid | max u | max v | L2 u | L2 v | max-uv |
|---|---|---|---|---|---|---|---|---|
| 33 | 1/32 | 20000 | 9.6e-07 | 10.57% | 3.19% | 2.48% | 1.80% | 10.57% |

n=33 一行是**健全性检查点**（20k 步、resid 9.6e-7、未完全收敛），不是验收点：
max-point 10.57%。它落在粗网格应有的量级，且给出与文献一致的剖面对称性（图 1）。

## 4. 注意点 / 本次挖出的三个实质问题
**F1（TensorLBM 求解器的场输出口径）**：该求解器的 `_stream` 使标签 i 的分布沿 **−c_i** 平流，
方案自洽（feq / 动量 / Ladd 修正同用标签 c_i），等价于把「顶盖向 −x 运动」的镜像问题
跑一遍再取负速度输出。因此**中心线剖面是正确的**（这也是它现有闸门只用中心线的原因），
但报告出的**整场**是物理场的 x 镜像 + v 反号。任何对整场（涡心、壁面剪应力、场云图）做
验收的下游必须套该映射，否则会得到错误的场比对结论。

**F2（Ghia Re=100 Table II 基准数据损坏）**：`reference_data.py` 的 `GHIA_RE100` 把
**Table II 的 v 值放在 Table I 的 y 站位上**，且数值序列是混合体
（与 `benchmarks/anchors/ghia1982_provenance.json` 中 BOUND 的 OCR 转录不一致，该 JSON
给出 Table II 自己的 17 个 x 站位 0.0625…0.9688 与正确 v 值；见本目录 `GHIA100_V`）。
后果：TensorLBM 方腔闸门里任何 v 口径（`l2_v`、`v_max` 及其位置、以及混入 v 的
「max-point」）都在与错误参考比对 —— 台账里此前的「n=65 max-point 5.29%」若来自 v 分量即不可采信。
本目录的 `errors_vs_ghia(..., legacy_v=True)` 专门保留旧表，可量化该污染。

**F3（本席自实现的一个真 bug）**：Ladd 动量修正符号初版取 `+6 w_i (c_i·u_w)`，
一步测试立刻显示顶盖把流体推向 −x（回归测试 `test_ladd_momentum_injected_in_lid_direction`
锁死该符号）。机械教训：一阶符号必须由「一步后首行流体速度方向」这种可判定的物理量测试钉住。

## 5. 误差判定（显式回答）
- n≥97 的 max-point <3%：**未证明**（作业未回）。
- 观测阶 p≥1.8：**未证明**（只有 1 个收敛点）。
- 全部数值均以修正后的 Ghia Table I/II 为基准；v 口径一律用修正表，不沿用旧表。
- 无任何数字被手工抄写：上表由 `run_study.py` 直接写 JSON，本文件由 `_mkdeliver.py` 从 JSON 注入。

## 6. 复现命令
```
python3 -m pytest test_d2q9_cavity.py -q                     # 6 passed
TENSORLBM_SRC=<...>/TensorLBM/src python3 crosscheck_tensorlbm.py --n 17 --steps 6
python3 run_study.py --ns 65 97 129 --resid_tol 2e-8 --max-steps 70000   # 集群 cj4018
python3 _mkdeliver.py                                        # 图 + 本文件
```

## 7. 下次自找活的线索
1. 取回 `cj4018` 工件 → 跑 `make_figures.py` → 判定 p 与 n≥97 的 3% 闸；
2. 由 F2 派生新条目建议：**TensorLBM 方腔闸门基准表修复 + 受污染指标的复算与重判**
   （slug 建议 `GAP-DIFFLBM-GHIA-TBL2-FIX`，验收：Table II 站位/数值与 provenance JSON 逐项一致；
   修复后重跑 n=65/97/129 的 `l2_v`/`max_v` 并与旧值并列给出偏差）；
3. 由 F1 派生：**整场口径契约**（`GAP-DIFFLBM-FIELD-CONVENTION`，验收：涡心位置与壁面剪应力
   在映射前后的一致性测试 + 文档口径声明）。
