# 交叉验证判据复核：r2「invariant zero」失败项的真伪（m21265）

**锚点**: `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE`
**仓库**: TensorBEM  **分支**: `agent/m21265`  **日期**: 2026-11-15
**证据工件**: `benchmarks/verified/capytaine_crosscheck_r2/invariant_residual_audit.json`
**生产者**: `scripts/capytaine_r2_invariant_residual_audit.py`（可复跑，双侧独立测量）

---

## 1. 起点：冻结结果自相矛盾

`results.json`（m19943 冻结）给出：

| 块 | 值 |
|---|---|
| `comparison.verdict` | **FAIL** |
| `invariant_scoring.invariant_verdict` | **FAIL**（8 条 "invariant zero exceeded"） |
| `comparison.worst_active_relative` | **6.71 %**（内部带宽 5 %） |
| 失败项之一 | `sphere_r1_z-0.5 nu=2 damping A[surge,pitch]`：TensorBEM −11.92 vs capytaine −14.83，bound 10.25 |

两个独立求解器都把该量给到 O(10)（ρL³ 的 1.0–1.5 %），而判据声称它「恰为零」。
这就是本单要判的问题：**这是 TensorBEM 的缺陷，还是判据的缺陷？**

## 2. 判据的物理主张与它的适用边界

`invariant_null_dofs()` 的论证是：球面（曲率中心 = 旋转中心）上每点 `(r−c)×n ≡ 0`，
故旋转模态广义法向恒为零，任何含 roll/pitch/yaw 的项目标值恒为 0。

对**光滑**球面这一步无可指摘。错在把它当作**离散网格**的性质：
平面面元上 `n` 是面法向、`r` 是面心，二者不再平行，残差 O(面元张角)。
`compare()` 曾用 `INVARIANT_RATIO = 1 % ρLᵏ` 作为绝对界。

## 3. 独立测量（不依赖被测数值）

定义无量纲几何残差 `Σ A_k |(r_k−c)×n_k| / Σ A_k |r_k−c|`（纯网格性质，解之前即可算），
以及实测残差 `max(|A_ij|,|B_ij|)/ρL^k`：

| 代码 | 网格面数 | 几何残差 mean | 实测阻尼残差 ν=1 | ν=2 |
|---|---|---|---|---|
| TensorBEM（测地 icosphere） | 270 | 1.51 % | **3.87 %** | 4.39 % |
| TensorBEM | 1026 | 0.69 % | **1.17 %** | 1.31 % |
| capytaine（UV 球） | 192 | 0.37 % | 0.57 % | 0.90 % |
| capytaine | 600 | 0.30 % | 2.74 % ⚠ | 3.00 % |
| capytaine | 1120（冻结参考层） | 0.18 % | **1.35 %** | 1.45 % |
| capytaine | 2240（本单补测） | 0.09 % | **0.66 %** | 0.73 % |

三条结论，逐条可验：

1. **参考解自身就在 1 % 之上**：冻结参考层（1120 面）capytaine 残差 1.35 %。
   1 % 的界**用 capytaine 自己的数字就会失败**，与 TensorBEM 无关 → 旧判据无效。
2. **衰减律 ≈ 1/N**：TensorBEM 270→1026 面残差降 3.31×（面数 3.80×）；
   capytaine 1120→2240 面降 2.03×（面数 2.0×）。物理耦合不会随加密衰减，
   facet 伪影会 → 该量在参考分辨率下**不是物理量**，不能做交叉验证靶。
3. **capytaine 的 (20,40) 层非单调**（2.74 %）：面元残差在粗网格上符号抵消强烈，
   噪声主导。如实记录，不做平滑。

TensorBEM 三点衰减律另由模块自带的 `degeneracy_audit` 佐证（hemisphere_r1：
1.465 % → 0.422 % → 0.110 %，面元 64/256/1024）。

## 4. 判据修正（保留缺陷检出力）

`INVARIANT_RATIO`: **1 % → 3 %**，并改称「结构界」：

* 下限理由：必须高于参考解噪声底 1.35 %（3 % = 2.2×），否则检查在参考自己的数上就红。
* 上限理由（保留检出力）：把旋转中心错放在水线原点 (0,0,0) 而非球心 (0,0,−0.5)
  —— 经典缺失 `(r−centre)` 臂 —— 同一项给出 **368 % ρL³**，是 3 % 界的 **123 倍**。
  一个 O(1) 装配缺陷依然被清晰判红。
* 该量**永不做相对误差评分**（`compare()` 早已如此，本单加测试锁死）。

不放松物理带宽（对角 5 % / 非对角 10 % 保持原值），不美化总判决。

## 5. 修正后状态（诚实口径）

| 轴 | 修正前 | 修正后 | 说明 |
|---|---|---|---|
| invariant 轴 | FAIL（8 条假失败） | **PASS** | 8 条残差 ≤1.52 % ≪ 3 % 结构界 |
| physics 轴 | FAIL（8 条） | **FAIL**（8 条） | 5.82–6.71 % vs 内部 5 % 带宽，**保留** |
| 总判决 | FAIL | **FAIL** | 内部 QoS 未达标，不粉饰 |
| 锚点验收（<15 %） | 6.71 % 已满足但未记录 | **`ledger_acceptance.satisfied = true`** | 新增显式块，杜绝含糊 |

physics 轴 8 条的收敛趋势（`convergence` 块）：hemisphere 对角 9.81 % → 4.63 % → 3.30 %（64/256/1024 面，已达标）；
floating sphere 对角 7.25 % → 6.52 %（270/1026 面，收敛慢）。即：**hemisphere 是分辨率不足、sphere 是参考网格本身的
facet 质量限制**，两者都不构成 >15 % 的不可接受误差。

## 6. 局限与注意点

1. TensorBEM 侧本单只补到 1026 面（两点衰减律）；三点律来自模块自带退化审计的另一几何。
   更细的 4098 面点在 `run_compute` 队列里可复跑（`--tensorbem-levels 2,3,4`）。
2. capytaine (20,40) 层的非单调未被解释透（疑与水线切割产生的瘦面元有关），如实记录为噪声。
3. 3 % 结构界是「高于观测噪声底、低于 O(1) 缺陷」的区间取中，不是收敛目标；
   若将来换成真正贴合球面的网格，该界应随噪声底一起下调。
4. 本单**未**证明 floating sphere 的 5 % 带宽可在可行分辨率内达到；若要收紧，
   需换用同样几何的更优参考网格（例如解析 Lamb 类极限或贴体面元）。

## 7. 复现

```bash
# 双侧残差测量（capytaine 需已安装；TensorBEM 侧可分段递交集群）
PYTHONPATH=src python scripts/capytaine_r2_invariant_residual_audit.py --side capytaine
PYTHONPATH=src python scripts/capytaine_r2_invariant_residual_audit.py --side tensorbem
# 用冻结解重算评分与文档（不重新求解）
PYTHONPATH=src python scripts/regen_capytaine_r2_scoring.py
# 判据证据测试
PYTHONPATH=src python -m pytest tests/test_capytaine_r2_invariant_residual.py \
    tests/test_capytaine_crosscheck_r2.py -q
```