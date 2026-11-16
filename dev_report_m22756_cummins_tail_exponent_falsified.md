# CASE-BEM-CUMMINS-WIGLEY — C9 尾部指数断言的复核（p(20)≈3.0 被否证）

**Mission**: m22756 ｜ **Anchor**: `CASE-BEM-CUMMINS-WIGLEY`（feature_ledger: open）｜ **仓**: TensorBEM
**对象**: 闸门 `C9`（12 → 20 rad/s 扩展带可判性）所依赖的尾部指数估计器

## 1 本单攻的是哪条验证欠账

算例已发布判据 16/16 通过（`benchmarks/verified/casepool_bem_cummins_wigley/case_result.json`）。
欠账在 `extended_probe_report.json` 的 `certified_gates`：`C8`/`C9` = `not_run`。

round-9 计划（`dev_report_m22423_cummins_wigley_round9.md` §7）把这条欠账的闭合**押在一个断言上**：

> 「把频带扩到 20 rad/s 后，水平模态的阻尼尾部指数 `p(20) ≈ 3.0` > `slope_bar = 2.05`，
>  因此 surge / sway / yaw 的截断预算变为有界，三个自由度从『经验一致』升级为『带限数据蕴含』」。

这个 `3.0` 在仓库里**从未被任何数据检验**；它既没有出处，也没有不确定度，却被用来决定
C9 是否值得花 2–4 CPU·h 去算 55 个 3840 面板频点。本单做的就是把这句断言拿去和数据对质。

## 2 怎么做的

判定 C9 的估计器只有一行：`tensorbem.causality.local_tail_exponent(omega, B_diag, window=3)`，
即对频带顶端**最后 3 个样本**做 log-log 直线拟合，取斜率 `p`；判据 `p > 2.05`
（`high_band_truncation_budget` 的收敛条件，`int^inf B(s) ds/(s^2-ω^2)` 只在 `p > 2` 收敛）。

新增可复现脚本 `scripts/case_cummins_wigley_tail_estimator_robustness.py`：

1. 对仓库里**所有**已存在的扩展带工件重算 `p(20)`：
   `sweep_extended_probe_coarse.npz`（192 面板）、`sweep_extended_probe_medium.npz`（480 面板），
   以及已提交的 `sweep_medium.npz`（0.2–12 rad/s，3840 面板）作为 12 rad/s 参照；
2. 每个自由度取 `window = 3, 4, 5` 三种拟合窗，检验结论对窗口的敏感性；
3. 跨工件比较 `p(20)` 的取值域与「是否可判」判决是否随网格翻转；
4. 结果落盘为 `benchmarks/verified/casepool_bem_cummins_wigley/tail_estimator_robustness.json`。

同时把集群侧的欠账闭合工作重新递交（详见 §6）。

## 3 计算结果（程序化输出，无手抄）

`p` 为频带顶端 log-log 斜率；`adm` = `high_band_truncation_budget(...).admissible`（即 C9 的可判性）。

| 工件 | DOF | p(w3) | p(w4) | p(w5) | adm |
|------|-----|-------|-------|-------|-----|
| probe_coarse (192) | surge | 6.528 | −inf | −inf | True |
| probe_coarse | sway | 4.786 | −0.133 | −1.297 | True |
| probe_coarse | heave | 4.871 | 5.830 | 5.761 | True |
| probe_coarse | roll | 5.493 | 3.040 | 3.354 | True |
| probe_coarse | pitch | −2.033 | 1.717 | 2.043 | **False** |
| probe_coarse | yaw | 1.968 | 0.563 | 1.700 | **False** |
| probe_medium (480) | surge | 4.540 | 3.238 | 3.165 | True |
| probe_medium | sway | 2.547 | 2.396 | 2.303 | True |
| probe_medium | heave | 1.988 | 3.531 | 4.087 | **False** |
| probe_medium | roll | 7.525 | 6.542 | 6.059 | True |
| probe_medium | pitch | 5.932 | 5.621 | 5.443 | True |
| probe_medium | yaw | 5.580 | 4.274 | 3.731 | True |
| sweep_medium (12 rad/s 参照) | surge | 1.095 | 0.978 | 0.870 | False |
| sweep_medium | sway | 1.529 | 1.397 | 1.229 | False |
| sweep_medium | yaw | 1.616 | 1.479 | 1.305 | False |
| sweep_medium | heave | 2.870 | 2.607 | 2.382 | True |
| sweep_medium | roll | 3.197 | 2.893 | 2.533 | True |
| sweep_medium | pitch | 3.070 | 2.829 | 2.603 | True |

跨工件取值域（`window=3`）：

| DOF | p(20) 取值域 | 判决随网格稳定？ |
|-----|--------------|------------------|
| surge | 1.095 – 6.528 | 否 |
| sway | 1.529 – 4.786 | 否 |
| heave | 1.988 – 4.871 | 否 |
| roll | 3.197 – 7.525 | 是（恒 > 2.05） |
| pitch | −2.033 – 5.932 | 否 |
| yaw | 1.616 – 5.580 | 否 |

## 4 结论：断言被否证（honest negative）

1. **`p(20) ≈ 3.0` 不是这条数据的性质，而是一次三样本拟合的抽样**。同一批工件上
   `p(20)` 的取值域跨 O(1)–O(10)：surge 1.10–6.53、sway 1.53–4.79、pitch −2.03–5.93。
   `3.0` 落在带内（例如 probe_medium 的 roll p(w4)=3.040），但没有任何自由度以它为值。
2. **判决本身会随网格翻转**：pitch 在 coarse 上 `adm=False`（p=−2.03）、在 medium 上
   `adm=True`（p=5.93）；heave 反向（4.87 → 1.99）；yaw 反向（1.97 → 5.58）。
   6 个自由度里只有 roll 的判决在三个工件上都稳定。
3. **对拟合窗也不稳定**：coarse 的 surge 在 w=3 给 6.53，w=4/5 直接返回 `-inf`
   （尾部出现非正值，`local_tail_exponent` 的守卫），即该 DOF 的「已进入代数衰减」
   在相邻窗口下不成立。
4. 因此 C9 当前**没有能力**凭 `p(20)` 宣称 surge/sway/yaw「升级为带限数据蕴含」。
   C9 的闸门语义（扩展不得让任何自由度掉档）仍然有效且应当跑，但它的单窗斜率**不能单独作为证据**。

这不是把算例判成 FAIL：已发布 16/16 判据不含 C8/C9；本单否证的是「扩展带必然使水平模态可判」这条**计划断言**。

## 5 注意点与诚实边界

* **两个探针网格都不满足 `C2`**（`k·sqrt(A_panel) ≤ 1`）：coarse 到 ~9.9 rad/s、
  medium 到 ~12.3 rad/s 就失效，而探针带是 12–20 rad/s。所以 §3 的数字**不能认证任何东西**，
  它们只证明**估计器脆弱**，不能证明物理上「尾部没进入衰减」。
* 真正的裁决需要 `sweep_extended.npz`（60×16 = 3840 面板，可判到 20.5 rad/s）。
  在该工件缺失时，C9 必须如实记为 `not_run`，不得写成 "widened"。
* 本单的否证是**关于估计器**的（3 样本斜率 + 网格依赖），不是关于 Wigley 阻尼物理的。
  改用更长窗口或更细网格后，同一批数据可能给出稳定的 `p`；这也是 $6 里 tail-probe 工件的用途。
* 外部锚点（实验/解析辐射系数表）仍为 WITHHELD，与 round-9 一致，本单未改变。

## 6 集群侧：欠账闭合的作业状态（未完成，如实记录）

| 作业 | 内容 | 状态（本单结束时） |
|------|------|--------------------|
| `cj4321` | 扩展带 55 频点 × 3840 面板扫频（m22711 递交，`shard×55` 双节点 detached） | **running**（41 min，节点 n315/n330；`fetch` 取到的 tar 内 `extended_points/` 仍为空 → 尚未落盘） |
| `cj4364` | `anchor_ladder_extended.json`（44×14 / 60×16 镜像 `A_inf` 锚，2 分片） | queued |
| `cj4381` | 认证网格顶端 4 频点（16.86 / 18.37 / 20 rad/s 等）尾部探针 | queued |
| `cj4363` / `cj4373` | 我为本单递交的 55 分片 / 8 分片重复扫频 | **killed**（与 `cj4321` 重复，撤单省算力） |

本单新增 `scripts/case_cummins_wigley_extended_tail_probe.py`：只解频带顶端 4 个节点
（单核 ~3 min/点），在 **C2 可判网格**上给出 `p(20)`，并复用 band 脚本的 `solve_point`
（同名同布局，后续全带装配可直接复用这些节点，不浪费算力）。作业 `cj4381` 即其运行实体；
若它在会话内未返回，则该项仍为欠账。

## 7 误差判定

本单没有外部基准可比（Wigley 船辐射阻尼无实验/解析表，`L5` 仍 WITHHELD），
因此**不能**回答「关键量 vs 基准是否 <3%」——没有基准就给不出误差，这是学科限制而非省略。
可给的内部量是：§3 的跨工件散布本身——同一自由度 `p(20)` 的相对散布达 80%–300%
（例如 sway 1.53 → 4.79，比值 3.1），远超任何 3%/15% 容差，这正是本单的结论：
**该估计量在现有工件上不具备 15% 量级的可复现性**，不能据以放行 C9。

## 8 复现

```bash
cd /srv/tensor/agent/work/m22756
PYTHONPATH=src python3 scripts/case_cummins_wigley_tail_estimator_robustness.py
# -> benchmarks/verified/casepool_bem_cummins_wigley/tail_estimator_robustness.json

# 作业返回后（C2 可判网格的顶端实数）
# run_compute: submit /srv/tensor/agent/work/m22756 scripts/case_cummins_wigley_extended_tail_probe.py --indices 51,52,53,54
# 全带装配 + 重判（需要 sweep_extended.npz 的 55 个节点齐备）
PYTHONPATH=src python3 scripts/assemble_cummins_extended.py
```

## 9 给下次自找活的线索

1. **C9 估计器要加稳健性守卫**：建议把 `truncation_decidability_gate` 的判决改为
   「`window ∈ {3,4,5}` 全一致且预算有界」才算可判，并把窗口分歧写进 `values`
   （否则 C9 会因网格/窗口的运气而放行）。这就是本单否证的直接修复项。
2. `cj4321` 落盘后：`fetch` → `assemble_cummins_extended.py` → 看 `C8`/`C3a_extended`/`C9`；
   若 C9 仍报 `newly_decidable` 为空，按 round-9 §7 P2 转「经验一致」并在 `case_result.json` 显式降级。
3. 若 `cj4381` 的 4 个认证网格节点返回，可直接检验 §3 的网格趋势在 3840 面板上是否收敛
   （这是判「估计器脆弱」是否仅为面板不足的必要一步）。