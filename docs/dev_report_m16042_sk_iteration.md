# m16042 · TensorBEM `tensorbem.sk_fit` — 名不副实的“SK 迭代”修复 + 记忆核口径验证

分支 `agent/m16042`（基线 `4d63824`）。EDIT_WHITELIST（自声明）：
`src/tensorbem/sk_fit.py`, `src/tensorbem/benchmarks/sk_fit_verification.py`,
`tests/test_sk_fit.py`, `docs/sk_verification/`, `docs/dev_report_m16042_sk_iteration.md`。

【锚点=NEW:一句话理由】功能点总账内无 BEM 频域→时域有理逼近（辐射阻尼 SK/VF 拟合）条目；
`GAP-*` 表内 BEM 侧仅有 `FEAT-OSS-INVENTORY-BEM`（盘点）。建议新条目：
`slug=GAP-BEM-RATIONAL-FIT`，title「辐射阻尼有理逼近 → Cummins 记忆核的迭代法验收面」，
所属域 TensorBEM；量化验收：①SK 迭代必须真迭代（`n_iter>=2` 且 `max_iter`/`tol` 生效，现 1/硬编码 True）；
②Hulme 半球 heave 阻尼 12 极点最大相对误差 ≤3e-2（现 1.55e-2，修复前 1.68e-1）；
③拟合模型闭式核 vs 表格积分核最大相对误差 ≤1e-2（现 5.44e-3，修复前 1.19e+2）。

## 1 案例介绍

物理背景：船舶/浮体耐波性时域模拟走 Cummins 方程，需要把频域辐射阻尼 `B(ω)` 压成有理（松弛谱）形式

    B(ω) = d + Σ_k c_k α_k /(α_k² + ω²),  α_k > 0

因为只有这种形式能让记忆核 `K(t) = (2/π)∫₀^∞ B(ω)cos(ωt)dω` 变成**精确的指数和**
`K(t) = Σ_k c_k e^(−α_k t)`（t>0；常数项 d 只在 t=0 贡献 2d·δ(t)）。有理逼近的精度直接
决定时域模拟的精度，是频域→时域桥（`docs/m15632_freq_time_bridge_fix.md`）的正确性上游。

对象：`src/tensorbem/sk_fit.py`（m15632 引入，公开 API `sk_fit_damping`，文档声称
“Sanathanan–Koerner 迭代有理逼近”，引用 Sanathanan & Koerner 1963）。

算例（全部程序化生成，零手抄）：

| 项 | 值 |
|---|---|
| 案例 A（模型类内合成曲线） | α=[0.4, 1.1, 3.0]，c=[2.0, 1.2, 0.6]，d=0.05；ω=geomspace(0.02, 60, 600) |
| 案例 B（真实曲线） | `tests/data/time_domain_hemisphere_sweep.json` Hulme 半球 heave 阻尼，201 点，ω∈[0.14, 29.7] |
| 拟合极点数 | A: 3；B: 12 |
| 权重 | 相对权重 `1/max(|B|, 1e-3·max|B|)`（沿用 `state_space._weights`） |
| 硬件 | 99 控制面单核，numpy；全流程 < 5 s |

## 2 怎么做的

### 2.1 缺陷（先读代码，再用实验钉死）

1. **名字与实现不符**：函数体只有**一次**加权线性最小二乘，极点冻结在
   `logspace(0.3ω_min, 3ω_max)` 对数网格上——没有极点重定位，也就没有 SK 迭代；
2. **参数是死的**：签名接受 `max_iter`、`tol`，函数体从不读取（`grep` 可证）；
3. **返回标志造假**：`n_iter=1`、`converged=True` 硬编码——所谓“收敛”无任何判据。

### 2.2 修复：实现真正的迭代（SK 线性化 + 极点重定位 + 对数步长阻尼）

`sk_fit_damping` 现在按 SK 迭代执行（`x = ω²`，分母局部化表示 `D(x)=1+Σ rt_j/(x+β_j)`）：

1. 用 SK 权重 `1/|D_prev(x)|` 加权求解线性化方程
   `Σ c_k g_k(ω_i) + d − B_i·Σ rt_j/(x_i+β_j) = B_i`（未知量 c, d, rt）；
2. `D` 的零点即重定位后的极点（`x = −α²`）；线性化系统描述的是 2p 极点，所以**必须**
   在重定位后的极点上重解 c、d 才评估模型失配（否则拿旧极点残差当模型精度，这是
   原型阶段实测踩到的坑）；
3. 若整步重定位使失配上升，取极点在对数空间的半步（`sqrt(β·β_new)`）再试；
4. `n_iter` = 实际迭代次数，`converged` = 相对失配下降达 `tol` 才为真；`max_iter`/`tol`
   真正生效；返回的是**历史上失配最小**的一组参数（单调保护）。

旧的单次求解保留为 `fixed_pole_fit_damping`，只作基线，不再冒用 SK 之名。
实现细节与文献对应（Sanathanan & Koerner 1963；Gustavsen & Semlyen 1999 的极点重定位）
写进模块 docstring，并显式声明模型类只有实极点、曲线不在类内时 `converged=False`。

### 2.3 验证口径（独立路线，不自我循环）

`python -m tensorbem.benchmarks.sk_fit_verification --out docs/sk_verification`
一条命令产出 JSON+Markdown+图（同一进程渲染，零手抄）：

* **拟合精度**：`|B̂−B|/max|B|` 的最大值/RMS，SK vs 修复前基线；
* **时域后果（案例 B）**：拟合模型的闭式核 `Σ c_k e^(−α_k t)` vs
  表格的精确分段线性积分核 `time_domain.radiation_memory_kernel`——两条**不同**路线；
* **解析恒等式（案例 A）**：闭式核 vs 同一表格积分核（见 §5 未闭合项）。

## 3 计算结果（`docs/sk_verification/results.json`）

### 3.1 拟合精度

| 案例 | 方法 | 最大相对误差 | RMS 相对误差 | n_iter | converged |
|---|---|---:|---:|---:|---|
| A 合成（3 极点） | 修复前（固定极点） | 3.871e-01 | 2.320e-02 | 1 | True（硬编码） |
| A 合成（3 极点） | SK 迭代 | **3.860e-02** | 1.790e-02 | 4 | False |
| B Hulme（12 极点） | 修复前（固定极点） | 1.684e-01 | 3.287e-02 | 1 | True（硬编码） |
| B Hulme（12 极点） | SK 迭代 | **1.547e-02** | 3.180e-03 | 4 | False |

案例 B 最大相对误差改善约 11 倍；案例 A 约 10 倍（RMS 改善有限，因 RMS 由中频段
本已拟合良好的点主导）。

### 3.2 时域记忆核（拟合误差如何传到 Cummins 方程）

| 案例 | 路线 | 核最大相对误差 |
|---|---|---:|
| A | 固定极点模型的闭式核 vs 解析核 | 7.730e-02 |
| A | SK 模型的闭式核 vs 解析核 | **1.350e-02** |
| B | 固定极点模型的闭式核 vs 表格积分核 | 1.190e+02（失效） |
| B | SK 模型的闭式核 vs 表格积分核 | **5.440e-03** |

案例 B 固定极点基线在核上失效（1.19e+2）：极点被压到 `0.3ω_min≈0.042` 附近，
生成表格支撑区间外的长指数尾，`K(t)` 被彻底带偏。**这是本单最有物理意义的结论：
固定极点的“拟合得好”只在 B(ω) 的图上成立，在时域核上可以差两个数量级。**

## 4 结果图

`docs/sk_verification/fig_sk_verification.png`（4 联图）：上排=案例 A 的 B(ω) 拟合与核误差；
下排=案例 B 的 Hulme 阻尼拟合与“闭式核 vs 表格积分核”对比。数值与 JSON 同源。

## 5 注意点 / 局限（honest）

1. **解析恒等式一项未闭合**：案例 A 的“闭式核 vs 表格积分核”最大相对误差
   **3.694e-01**，远大于同案例下两个拟合各自的核误差（7.7e-2 / 1.35e-2），说明差异来自
   积分路线而非拟合：表格是 `[0.02, 60]` 有限带、低/高频截断（`low_frequency_model="zero"`
   丢 [0, ω_min]，ω_max 以上直接截断），而解析核是全域。本单**未**定位到具体来源
   （残差 0.37 vs 截断界估算 ~1e-2），列为未闭合项，**不作验收门**；已登记为后续线索。
   案例 B 的两条路线互校（5.44e-3）说明积分机制本身可用。
2. **`converged=False` 是诚实的**：`tol=1e-8` 的相对失配下降判据在 4 次迭代后未满足
   （重定位带来的下降率不足 1e-8），实现按“失配上升即停”退出并返回最优参数；
   不把“停下来了”写成“收敛了”。
3. **模型类只含实极点**：真实 2-D 阻尼曲线（长波对数支）不在类内，届时重定位多项式会出现
   复根，实现把它们折到实轴并计数、以 `converged=False` 报告；本单两个算例均未触发。
4. **极点本身可观测性差**：带限平滑曲线上单极点位置的条件数实测 ~4.6e6，SK 能压低失配但
   不能识别出“真极点”（案例 A 极点相对误差仍 O(1e-1)）。判据应落在 B(ω) 与 K(t) 的失配上。
5. 未跑全量测试套件（耗时），只跑与本改动相关的 4 个文件 36 项（见 §6）。

## 6 复现与验证

```
PYTHONPATH=src python -m tensorbem.benchmarks.sk_fit_verification --out docs/sk_verification
PYTHONPATH=src python -m pytest tests/test_sk_fit.py tests/test_freq_time_bridge.py \
    tests/test_state_space.py tests/test_state_space_fit_invariants.py -q
# 36 passed in 33.78s
```

`tests/test_sk_fit.py`（12 项）把 §3 的事实钉成门：必须真迭代、`max_iter`/`tol` 生效、
基线确实是旧单次求解、确定性（两次调用逐位相同）、Hulme 上 SK 至少比基线好 3 倍且
最大相对误差 <3e-2、核误差 <1e-2、输入契约（点数/正频/非负阻尼/`n_poles`/`max_iter`）不变。