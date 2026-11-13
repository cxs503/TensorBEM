# m17299 · TensorBEM `tensorbem.fmm` 代价审计重开第 1 轮：谐波阶数降为 O(p²)、叶容量 regime map

分支 `agent/m17299`（基线 `d41140e`，从当前 main 切出）。原 PR #688（m14931 交付）
已关闭；本单不重复其内容（`docs/fmm_cost_audit/`、`dev_report_m14931_fmm_cost_audit.md`
等文件早已并入 main），而是**接着把仍未闭合的两条验收判据管到底**：台账判据
②「单位交互代价比 ≤1」与 ③「复杂度在 N≥1e4 实测」。重开轮结束后遗留的判据在 §7 如实列出。

**热文件声明（交付纪律）**：`src/tensorbem/fmm.py` 是跨单热点文件，本轮对它的改动
只有一个函数内部实现（`all_real_spherical_harmonics` 的求值顺序），不改任何公开
签名、不改路由判据、不改 `matvec` 语义；数值逐位一致（§2.1）。

## 1. 案例介绍

- 对象：`tensorbem.fmm.FMMOperator`（3-D Laplace 单层算子 S q 的单层 FMM 加速，
  仓库台账记为 REAL）。几何类＝闭合曲面面元质心上的二维源分布（BEM 真实使用面）：
  icosphere subdivisions 2/3/4/5 → 320 / 1280 / 5120 / 20480 面板，单位球，
  源强度＝面板面积；精度靶点＝同网格顶点（避免质心自重合奇异）。
- 参考解＝`benchmarks.fmm_cost_audit.dense_matvec`（numpy 分块稠密求和），
  与 `tensorbem.fmm.direct_laplace_matvec` 同一数学对象。
- 动机：m14931 审出「多极可容许性缺陷 + 单叶空转 + 全 N 慢于稠密」；m15669 修树
  几何并向量化后，`docs/fmm_cost_audit/README.md` §6 测到 N=20480 时 FMM/dense=0.40x，
  但 §1（max_leaf=32）与 §6（max_leaf=128）对**同一个 N** 给出相反结论——说明叶容量
  是决策变量而非噪声参数，且两条台账判据从未按容量维度审过。

## 2. 怎么做的

### 2.1 缺陷：谐波核是 O(p³)，与它自己声明的 O(p²) 不符

`all_real_spherical_harmonics(p)` 逐 `(l, m)` 调用 `real_spherical_harmonic(l, m)`，
而后者每次都从 `P_m^m` 重新启动连带 Legendre 递推 ⇒ 每点工作量
`Σ_{l≤p}(2l+1)·O(l) = O(p³)`，而多极展开本身只需要 `(p+1)²` 项。
修复：每个阶 `m` 只跑一条递推梯子（`P_m^m → P_{m+1}^m → … → P_p^m`），各 `l`
从同一条梯子上读，`O(p²)`。公开签名、列序（`l*l + (m+l)`）、归一化（Schmidt
半归一）全部不变。

**等价性（先证后改）**：以「逐 `(l,m)` 调 `real_spherical_harmonic`」为参考实现，
p=0..8 各 257 个随机方向，最大相对差 **1.8e-16**（=机器精度）；极点
（θ=0/π）同阶一致；空靶集形状 `(0,(p+1)²)` 保持。测试：
`tests/test_fmm_harmonics_ladder.py`（10 项）。

### 2.2 决策实验：叶容量 regime map（预注册判据）

`benchmarks/fmm_cost_audit.py` 新增 `regime_map()`：N ∈ {1280, 5120} ×
max_leaf ∈ {32, 64, 128, 256} × p ∈ {2, 4}，每格测
`t_FMM/t_dense`、交互数比（`target-leaf 对数 / N²`）、单位交互代价比
（`c_fmm/(avg_leaf·c_dense)`）、以及**对稠密解的相对误差**（种子 20260101 的
256 顶点子样本）。判据在跑之前写死：

- **accelerated**：`time_ratio ≤ 0.75`；**parity**：`|ratio-1| ≤ 25%`（噪声带宽，
  由重复测量标定）；**slower**：`ratio ≥ 1.25`；
- **admissible**＝accelerated ∧ `far_fraction ≥ 0.2`（多极路径真的被用到，排除
  「单叶直接求和＝直接求和」的假加速）∧ 误差 ≤ 1e-3（台账判据①）；
- 判决＝admissible 行里 ratio 最小者。

**估计量选择（本轮方法论修正）**：审计机与其它 agent 共享，墙钟在两次独立重算之间
可漂 40%（同一行 5120/max_leaf=128/p=2 先后测到 0.54 / 0.68 / 0.91）。因此改用
`time.process_time` 的 **5 次中位数**（不用最小值：最小值被调度空档系统性低估），
并在 JSON 里同时留 `*_min_s` 供追溯。换估计量后逐行稳定（见 §3.2）。

### 2.3 文件

改：`src/tensorbem/fmm.py`（一个函数体）、`src/tensorbem/benchmarks/fmm_cost_audit.py`
（`harmonic_ladder` / `regime_map` / `_cpu_median_of` / 渲染 §7 §8 / 双图）；
增：`tests/test_fmm_harmonics_ladder.py`；扩：`tests/test_fmm_cost_audit.py`（+4 门）；
重生成：`docs/fmm_cost_audit/{README.md,audit.json,fig_fmm_cost.png,fig_harmonics_ladder.png}`。
复算命令（一条）：

```
PYTHONPATH=src python -m tensorbem.benchmarks.fmm_cost_audit --figures
```

> 坑：本机 `tensorbem` 是 editable 安装且指向**另一个 worktree**
> （`/root/tp-agent/work/m17241`）。直接 `python -m tensorbem...` 会跑别人的代码、
> 把工件写进自己的 docs（本轮先踩了一次，已用 `PYTHONPATH=src` 纠正并重生成）。

## 3. 计算结果（程序化注入，零手抄）

### 3.1 谐波核：同数、更低代价（`audit.json: harmonic_ladder`）

| p | 谐波数 | 参考 O(p³) (s) | 现 O(p²) (s) | 加速 | 最大相对差 |
|---:|---:|---:|---:|---:|---:|
| 2 | 9 | 0.1187 | 0.0498 | 2.38x | 1.8e-16 |
| 4 | 25 | 0.3842 | 0.1639 | 2.34x | 1.3e-16 |
| 6 | 49 | 0.8207 | 0.2703 | 3.04x | 1.1e-16 |
| 8 | 81 | 1.4527 | 0.4094 | 3.55x | 1.9e-16 |

（200 000 点，`_best_of` 3 次；数字取自 `docs/fmm_cost_audit/audit.json` 本轮。）

耦合到 matvec 的收益（`runs/ab_harmonics.py`，同进程 A,B,B,A 交错，4 轮，CPU 时间）：

| 构型 | 改前中位 (s) | 改后中位 (s) | 加速 |
|---|---:|---:|---:|
| N=1280, p=2, max_leaf=128 | 0.0987 | 0.0849 | 1.16x |
| N=1280, p=4, max_leaf=128 | 0.1802 | 0.1229 | 1.47x |
| N=5120, p=2, max_leaf=128 | 0.8089 | 0.8436 | 0.96x（噪声内，近场占主导） |
| N=5120, p=4, max_leaf=128 | 1.1576 | 0.9644 | 1.20x |

→ 谐波核 2.3–3.6x；整算子 p≥4 时 1.2–1.5x，p=2 大 N 时被近场直接求和吃掉（如实）。

### 3.2 叶容量 regime map（`audit.json: regime_map`，CPU 中位数 5 次）

| panels | max_leaf | p | 叶数 | far 占比 | 交互数/稠密 | FMM/dense | 判定 | 误差 |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1280 | 32 | 2 | 56 | 0.74 | 0.044 | 0.89x | parity | 8.00e-04 |
| 1280 | 128 | 2 | 56 | 0.74 | 0.044 | 0.88x | parity | 8.00e-04 |
| 1280 | 128 | 4 | 56 | 0.74 | 0.044 | 1.28x | slower | 2.46e-05 |
| 1280 | 256 | 2 | 8 | 0.00 | 0.006 | 1.45x | slower | 3.27e-15（单叶退化） |
| 5120 | 32 | 2 | 272 | 0.94 | 0.053 | 0.78x | parity | 2.09e-04 |
| 5120 | 64 | 4 | 272 | 0.94 | 0.053 | 1.52x | slower | 1.76e-05 |
| 5120 | 128 | 2 | 56 | 0.74 | 0.011 | **0.58x** | accelerated | 5.84e-04 |
| 5120 | 128 | 4 | 56 | 0.74 | 0.011 | **0.65x** | accelerated | 1.81e-05 |
| 5120 | 256 | 2 | 56 | 0.74 | 0.011 | **0.58x** | accelerated | 5.84e-04 |
| 5120 | 256 | 4 | 56 | 0.74 | 0.011 | **0.71x** | accelerated | 1.81e-05 |

结论（判决行）：**N=5120、max_leaf=128（≈N/40）、p=2 → 0.58x 稠密墙钟，误差 5.8e-4**；
p=4 同一构型 0.65–0.71x，误差 1.8e-5。N=1280 全构型在 parity 带内（0.88–1.45x）；
p=4 且叶容量过小（leaves=272）稳定慢 1.5x——代价花在多极交互数上。

单位交互代价比本轮落在 0.58x–1.52x（p=2 时<1，p=4 时>1，因每次交互要算 25 项展开）：
即台账判据②「单位交互代价比 ≤1」**在 p=2 达标、p=4 不达标**，且该判据本身不是
加速的充分条件——加速取决于「交互数下降倍数（0.011–0.053）是否大于单位代价惩罚」。

## 4. 图

- `docs/fmm_cost_audit/fig_fmm_cost.png`（2×2）：(a) FMM vs 稠密墙钟 log-log；
  (b) 多极精度-阶曲线（分离 / 曲面）；(c) regime map：FMM/dense vs N，按 max_leaf 分线，
  虚线=1；(d) 误差 vs 代价散点，红点线=台账阈值 1e-3。
- `docs/fmm_cost_audit/fig_harmonics_ladder.png`：谐波核 O(p³) vs O(p²) 代价与逐点加速标注。

## 5. 注意点（坑与局限）

1. **editable 安装指向别的 worktree**：不 `PYTHONPATH=src` 会跑他人代码并写自己的 docs。
2. **共享机墙钟不可判 5% 级效应**：同一行三次独立重算 0.54/0.68/0.91；本轮改 CPU 中位数
   并给出 ±25% 噪声带，带宽内的行一律记 parity、不计入加速。
3. **单层 FMM 的渐近复杂度仍是 O(N²)**：叶数随 N 增长（max_leaf 固定时），实测斜率
   1.5（`README.md` §6，dense 1.87），**台账判据③「N∈[1e3,1e5] 斜率∈[0.9,1.2]」仍未达标**
   ——加速来自常数与交互数，不是渐近阶。要达标必须做真正的多级 M2M/M2L/L2L，本轮未做。
4. 精度靶点是 256 顶点的种子子样本（非全顶点），已在文档声明；全顶点版本只在 N=320
   的 §4 用（162 靶）。
5. 谐波核仍是 numpy 级实现；未尝试 numba/C 后端，故 per-interaction 代价比无法降到
   「多极一次 ≪ 叶直接求和」的理论量级。
6. 未改求解器接线：FMM 仍未接入 `solver3d` / `free_surface_solver3d`（沿用 m14931 声明）。

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---:|---|
| 谐波核数值（p=0..8） | 逐 (l,m) 参考实现 | ≤1.8e-16 相对 | ✅ |
| 曲面 matvec（p=4, max_leaf=128, N=5120） | 分块稠密单层求和 | 1.81e-05 | ✅ |
| 曲面 matvec（p=2, max_leaf=128, N=5120） | 同上 | 5.84e-04 | ✅ |
| 分离构型 matvec（p=8） | 同上 | 4.18e-12 | ✅（沿用 §3） |
| FMM/dense 墙钟比 | 稠密 matvec | 不适用（比值量，噪声带 ±25%） | —（判据：加速与否，见 §3.2） |

## 7. 遗留（诚实）

- 台账判据③（渐近斜率 0.9–1.2）**FAIL**：单层实现结构性达不到，需多级翻译算子树，属下一轮范围。
- 台账判据②（单位交互代价比 ≤1）p=2 PASS / p=4 FAIL；已在文档说明它不是加速的充分条件。
- 求解器接线（FMM 进 `solver3d`）未做。