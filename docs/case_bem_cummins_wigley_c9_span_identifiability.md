# CASE-BEM-CUMMINS-WIGLEY — C9 的「可判性」本身是否可判：斜率对拟合跨度的系统性依赖

**Mission**: m23070 ｜ **Anchor**: `CASE-BEM-CUMMINS-WIGLEY`（feature_ledger: `open`，核对时间 2026-11-16）｜ **仓**: TensorBEM
**对象**: 闸门 `C9`（`truncation_decidability_gate`）在带顶读出的阻尼尾部指数 `p`，以及把频带扩到 20 rad/s 所需的重算管线

---

## 1 本单攻的是哪条验证欠账

算例已发布判据 **16/16 通过**（`benchmarks/verified/casepool_bem_cummins_wigley/case_result.json`）。
欠账在 `extended_probe_report.json` 的 `certified_gates` 节：**`C8` / `C9` = `not_run`**。

这条欠账在账本里挂了多轮，每一轮都以「交给集群、作业还在排队」收尾：

| 轮次 | 做了什么 | 欠账状态 |
|------|---------|---------|
| m22423（round 9） | 断言「扩到 20 rad/s 后 `p(20) ≈ 3.0`，surge/sway/yaw 即可判」 | 断言无出处、无不确定度 |
| m22711 | 递交 55 分片扩展带扫频（`cj4321`），计划 `2–4 CPU·h` | 作业未落盘，`C8/C9` 仍 `not_run` |
| m22756 | 用**已有**扩展带探针工件复核 `p(20)`，否证 `≈3.0` | 否证成立；同时暴露判决随网格翻转 |
| m22881 | C9 改用 `{3,4,5}` 多窗稳健共识（`robust_tail_exponent`） | 只覆盖 ≤0.1 decade 的窗，跨网格翻转未解 |
| m22267 | 计划「执行扩展带、验证 `p(20)`」 | **budget_exhausted，未产出** |

所以本单不再把答案押在「作业跑完自然就知道了」上，而是先问一个更便宜、更根本的问题：

> **`C9` 读的那个 `p`，在这个频带上到底可不可识别？**
> 如果不可识别，那么无论 55 个频点跑不跑得完，`C9` 的逐自由度判决都不是数据的性质。

这个问题**完全可以用仓库里已有的工件回答**，不需要新的 BEM 解。答案（§3）是否定的：
判决在同一件工件内部就随拟合跨度翻转。

---

## 2 怎么做的

### 2.1 计算侧：把「扩展带」真正变成可复现的工件（已递交，未落盘）

原管线依赖 `extended_points/point_*.npz` 落盘后再装配，但集群给**每个作业一份独立检出**，
55 个分片的产物从不落在同一个文件系统上——这正是前几轮「递交了、装不起来」的结构性原因。
本单补上这条链路：

* `scripts/cummins_extended_sweep_job.py` —— 一个作业内用进程池解一段频点（默认 55 点 / 8 进程），
  每个频点打印一行 `NODE <i> <omega> <json>`（`json` 对 `float` 无损往返）。
* `scripts/assemble_cummins_extended_json.py` —— 从作业日志装配 `sweep_extended.npz`，
  缺节点 / 重复节点 / 外来频点网格 / 缺 `solver_relative_residual` / 混合网格 / 未声明网格 **一律拒绝装配**
  （`sweep_extended.npz` 是 `C8`/`C3a_extended`/`C9` 的证据源，静默装配等于静默认证）。
* `scripts/cummins_extended_anchor_job.py` —— `A_inf` 镜像解锚点阶梯（`44x14`、`60x16`）以 JSON 输出，
  `--collect` 重建 `anchor_ladder_extended.json`，使 `C3a_extended` 保持**零自由参数**
  （否则闸门会静默退回「带内拟合偏移」这一自由参数）。
* 装配后：`PYTHONPATH=src python3 scripts/assemble_cummins_extended.py --skip-assembly`（既有脚本）重判并重写
  `case_result.json` / `CASE.md`。

### 2.2 分析侧：斜率跨度敏感性（本单结论的来源，纯本地、零新解）

新增 `scripts/cummins_tail_span_sensitivity.py`：对**已提交**的三件工件
（`sweep_extended_probe_coarse.npz` 192 面板、`sweep_extended_probe_medium.npz` 480 面板、`sweep_medium.npz` 480 面板 / 0.2–12 rad/s），
逐自由度把带顶斜率按**跨度** 3、4、5、7、10 个样本分别拟合，报告 `p`、跨度（decade）、`R²`、
以及同一工件内部判决是否翻转。

### 2.3 测试

| 文件 | 条数 | 断言的是什么 |
|------|------|--------------|
| `tests/test_cummins_extended_assembly.py` | 12 | 装配器对缺节点/重复/外来网格/缺残差/混合网格**确实抛错**；55 点网格与预声明脚本 `grid()` 逐点相等；网格两端恰为 0.2 / 20.0 |
| `tests/test_cummins_tail_dense_probe.py` | 5 | 探针的指数符号与 `tensorbem.causality.local_tail_exponent` 一致（**符号搞反会把每个自由度的判决全部反转**）；跨度（decade）随样本数单调；不完备探针被拒 |

`PYTHONPATH=src python3 -m pytest tests/test_cummins_extended_assembly.py tests/test_cummins_tail_dense_probe.py -q` → **17 passed**。

---

## 3 计算结果（程序化输出，无手抄）

`p` 采用 `B ~ omega^-p`，与 `local_tail_exponent` 同号约定；判据 `p > 2.05`。
跨度（decade）：探针工件 3/4/5/7/10 样本 = 0.049/0.074/0.099/0.148/0.222；`sweep_medium`（41 点）= 0.089/0.133/0.178/0.267/0.400。

### 3.1 `sweep_extended_probe_coarse.npz`（192 面板，12–20 rad/s，10 点）

| DOF | p(3) | p(4) | p(5) | p(7) | p(10) | spread | 判决翻转 |
|-----|------|------|------|------|-------|--------|----------|
| surge | 6.53 | nan | nan | nan | nan | — | **是** |
| sway | 4.79 | −0.13 | −1.30 | 2.89 | 3.19 | 6.08 | **是** |
| heave | 4.87 | 5.83 | 5.76 | 5.41 | 4.59 | 1.24 | 否 |
| roll | 5.49 | 3.04 | 3.35 | 4.51 | 4.51 | 2.45 | 否 |
| pitch | −2.03 | 1.72 | 2.04 | 3.26 | 3.78 | 5.81 | **是** |
| yaw | 1.97 | 0.56 | 1.70 | 2.99 | 2.89 | 2.43 | **是** |

### 3.2 `sweep_extended_probe_medium.npz`（480 面板，12–20 rad/s，10 点）

| DOF | p(3) | p(4) | p(5) | p(7) | p(10) | spread | 判决翻转 |
|-----|------|------|------|------|-------|--------|----------|
| surge | 4.54 | 3.24 | 3.16 | 2.66 | 2.31 | 2.23 | 否 |
| sway | 2.55 | 2.40 | 2.30 | 2.39 | 2.27 | 0.27 | 否 |
| heave | 1.99 | 3.53 | 4.09 | 4.33 | 4.16 | 2.35 | **是** |
| roll | 7.52 | 6.54 | 6.06 | 5.61 | 5.09 | 2.43 | 否 |
| pitch | 5.93 | 5.62 | 5.44 | 5.05 | 4.61 | 1.32 | 否 |
| yaw | 5.58 | 4.27 | 3.73 | 3.20 | 2.75 | 2.83 | 否 |

### 3.3 `sweep_medium.npz`（已提交 0.2–12 rad/s，480 面板，41 点）

| DOF | p(3) | p(4) | p(5) | p(7) | p(10) | spread | 判决翻转 |
|-----|------|------|------|------|-------|--------|----------|
| surge | 1.09 | 0.98 | 0.87 | 0.62 | 0.28 | 0.82 | 否 |
| sway | 1.53 | 1.40 | 1.23 | 0.77 | −0.32 | 1.85 | 否 |
| heave | 2.87 | 2.61 | 2.38 | 1.98 | 1.49 | 1.38 | **是** |
| roll | 3.20 | 2.89 | 2.53 | 1.65 | −0.17 | 3.36 | **是** |
| pitch | 3.07 | 2.83 | 2.60 | 2.20 | 1.62 | 1.45 | **是** |
| yaw | 1.62 | 1.48 | 1.30 | 0.63 | −0.89 | 2.50 | 否 |

### 3.4 结论（honest negative，且是**对闸门本身**的否证）

1. **同一件工件内部，判决就随拟合跨度翻转**：coarse 6 个自由度里 4 个翻转（surge/sway/pitch/yaw），
   medium 1 个（heave），已提交 12 rad/s 网格 3 个（heave/roll/pitch）。
   m22881 的稳健共识只用了 0.099 decade 以内的窗，因此**没看见**这件事。
2. **spread 超过判据阈值 2.05 的自由度**：coarse 4 个、medium 4 个、`sweep_medium` 2 个。
   当 `|Δp| > 2.05` 时，「`p` 是否越过 2.05」这句话没有意义——不确定性比判据本身宽。
3. **surge 在 coarse 上 p(3)=6.53 但 p≥4 直接 `nan`**（尾部出现非正阻尼，`local_tail_exponent` 的守卫生效）：
   闸门若只看 3 窗，会把一个尾部**符号都不稳定**的自由度判成「已进入代数衰减」。
4. 因此 `C9` 当前的逐自由度判决**不能**作为「扩展带使 surge/sway/yaw 升级为带限数据蕴含」的证据。
   `C8`/`C9` 仍为 `not_run`，而本单把它们的**可读性**也否证了：
   闸门要能判，必须先固定并**声明拟合跨度**，且跨度要覆盖足够多的 decade（本仓库现有数据最多 0.30 decade）。

---

## 4 结果图

![C9 带顶斜率 vs 拟合跨度（三件已提交工件，逐自由度）](outputs/m23070/tail_span_sensitivity.png)

图注：横轴=拟合跨度（decade），纵轴=`p`；虚线=`slope_bar=2.05`。同一自由度在同一条工件内曲线穿过虚线即表示判决随跨度翻转
（surge-coarse 的空缺=尾部非正阻尼、拟合不可用）。生成脚本：`scripts/cummins_tail_span_figures.py`（数据源 `tail_span_sensitivity.json`，零手抄）。

---

## 5 注意点与诚实边界

1. **本单没有新增 BEM 解。** 结论不依赖 20 rad/s 的 3840 面板扫频，只依赖已提交工件的**重分析**——
   这是本单能在不落盘任何新数据的前提下给出结论的原因。
2. **扩展带作业仍在排队**：本单递交了 `cj4558`（单节点冒烟）、`cj4559`（55 频点 / 8 进程）、
   `cj4561`（`A_inf` 锚点阶梯）、`cj4564`（10–20 rad/s 密集尾部探针，25 点，用于给出「跨度足够」的 `p`）。
   递交时队列深度 ~74，本单结束时它们**均未落盘**，因此 `C8`/`C3a_extended`/`C9` 仍为 `not_run`，
   不得据此改写 `case_result.json`。**这是 honest-FAIL 的如实记录：管线打通、证据未到。**
3. **判据阈值 `2.05` 本身没被质疑**：`p > 2` 是 `∫^∞ B(s) ds/(s²−ω²)` 收敛的条件，数学上成立；
   被质疑的是「用 3 个样本的带顶斜率去代表 `p`」这个**估计量**。
4. **两件探针工件都不满足 `C2`**（`k·√A_panel ≤ 1`），它们的 `p` 只界定物理量级，不构成认证；
   本单 §3 的结论是「估计量不可识别」，对网格合规性不敏感（粗网格只会更差）。
5. **符号约定是坑**：`local_tail_exponent` 返回的是 `p`（`B ~ ω^-p` 的正指数），
   自写的 `log B` vs `log ω` 最小二乘返回的是 `−p`；两者混用会让每个自由度判决**全部反转**。
   已用单测 `test_exponent_convention_matches_causality` 钉死。

---

## 6 误差判定（是否有基准）

**外部基准**：Wigley 船的辐射阻尼尾部没有实验/解析表，`C9` 无外部锚点。**因此不能回答「vs 基准 <3%」。**
可给的内部量：

| 检验项 | 结果 | 是否 <3% |
|--------|------|----------|
| 合成幂律 `B ∝ ω^-3` 上，探针分析器恢复的 `p`（跨度 3…25 样本） | 误差 ≤ 1e-8 | **是** |
| 同合成幂律在 `C9` 约定下的符号 | 与 `local_tail_exponent` 一致（单测钉死） | 是 |
| 装配器对 6 类坏输入的拒绝 | 6/6 抛错 | 是（定性） |
| **真实带顶斜率的跨度不确定性** | `|Δp|` 最高 6.08（coarse sway），超过判据阈值 2.05 | **否（不可接受）** |

最后一行是本单的核心否定结果：不是「结果误差大」，而是「这个量在当前数据上无法以好于判据的分辨率被测量」。

---

## 7 复现

```bash
cd /root/tp-agent/work/m23070
PYTHONPATH=src python3 -m pytest tests/test_cummins_extended_assembly.py tests/test_cummins_tail_dense_probe.py -q
PYTHONPATH=src python3 scripts/cummins_tail_span_sensitivity.py          # -> tail_span_sensitivity.json
python3 scripts/cummins_tail_span_figures.py                            # -> outputs/m23070/tail_span_sensitivity.png

# 扩展带（作业落盘后）：
PYTHONPATH=src python3 scripts/assemble_cummins_extended_json.py cj4559.log   # -> sweep_extended.npz
PYTHONPATH=src python3 scripts/cummins_extended_anchor_job.py --collect cj4561.log
PYTHONPATH=src python3 scripts/assemble_cummins_extended.py --skip-assembly   # 重判 C8/C3a_extended/C9
```

---

## 8 给下次自找活的线索

1. `cj4559`/`cj4561`/`cj4564` 落盘后：装配 → 重判 → 若 `C9` 过，把判据从 16/16 更新；若不过，按 §3.4 改写 `C9` 的**声明式跨度**判据。
2. **`C9` 判据应改成「声明跨度 + 跨度内一致」**：例如强制跨度 ≥ 0.3 decade 且 `p(span)` 在 `[span_min, span_max]` 上的分歧 < 0.5，
   否则报 `indeterminate`（第三个状态），而不是二值 pass/fail。`truncation_decidability_gate` 只有二值输出，这是设计缺口。
3. 同一套「跨度敏感性」检查可平移到其它带顶斜率判据（`decay_modes_gate`、`robust_decay_gate`、`C5b`），
   它们都用 3 样本带顶斜率。