# CASE-BEM-CUMMINS-WIGLEY — 扩展带扫频递交（C8/C9 验证欠账闭合）

**Mission**: m22711 ｜ **Anchor**: `CASE-BEM-CUMMINS-WIGLEY` ｜ **仓**: TensorBEM
**对象**: `scripts/case_cummins_wigley_extended_band.py`（55 频点分片）+ 集群作业 `cj4321`

## 1. 背景：为什么这是本单的实质增量

算例 `CASE-BEM-CUMMINS-WIGLEY`（Wigley 船 6DOF Cummins 记忆核）已发布判据 **16/16 通过**
（`benchmarks/verified/casepool_bem_cummins_wigley/case_result.json`，mission m21960）。
但存在一条**验证欠账**（round 9 审计、m22587 报告 §1 记录）：

* 已提交 12 rad/s 频带上 **surge / sway / yaw** 的阻尼尾部指数 0.28–1.62 **未进入代数衰减**，
  截断预算 `∫^∞ B(s) ds/(s²-ω²)` 无界 → 这三个自由度的 `C3a` 残差只是**经验一致**，
  而非带限数据蕴含（`docs/case_bem_cummins_wigley_extended_probe.md` §2）。
* 唯一出路是把频带扩到 **20 rad/s**，并在满足闸门 `C2`（`k·sqrt(A_panel) ≤ 1`）的
  **60×16（3840 面板）** 网格上求解。该网格 `sqrt(A_max)=2.32 cm`，可判到 `ω=20.5 rad/s`。
* 扩展带扫频工件 `sweep_extended.npz` **从未落盘** → 闸门 `C8`（扩展带与已提交带同物理）、
  `C3a_extended`（零自由参数 KK 闭合）、`C9`（阻尼是否进入代数衰减）在已发布判据上
  维持 **`not_run`**（`extended_probe_report.json` 明确记录 `C8/C9: not_run`）。

本单的实质增量 = **把这条验证欠账真正跑起来**：递交 55 频点 × 3840 面板的扩展带辐射扫频
到集群（`run_compute` 作业 `cj4321`），使 `C8/C9` 从 `not_run` 变为可判。

## 2. 怎么做

1. **核对锚点**：`feature_ledger` 中 `CASE-BEM-CUMMINS-WIGLEY` 状态 `open`（未 done/verified），
   可做实质增量；已发布判据 16/16 通过，不重做。
2. **定位欠账**：`extended_probe_report.json` 的 `certified_gates` 节显示 `C8/C9: not_run`；
   `extended_probe_cost.json` 给出认证网格 `(60,16)=3840 面板`、单节点墙钟 ~193 s、
   全带 ~0.54 CPU·h（声明计划 2–4 CPU·h）。
3. **验证脚本**：`scripts/case_cummins_wigley_extended_band.py --list` 输出 55 频点网格
   （0.2–20 rad/s，18.7 点/decade，与已提交扫频节点密度同量级），网格正确。
4. **递交**：`run_compute` 提交 55 分片扩展带扫频（`--shard {i} --shards 55`），
   作业号 `cj4321`，排队位次 34。分片按构造独立（每片解一个频点），
   丢失分片可单独重解，装配后 `sweep_extended.npz` 逐点可复现。
5. **装配/重判**（作业完成后）：`scripts/assemble_cummins_extended.py` 收集 55 个
   `point_*.npz` 为 `sweep_extended.npz`，把 `C8/C3a_extended/C9` 挂到已提交
   `sweep_medium.npz` 的重判上，重写 `case_result.json` 与 `CASE.md`（`CASE.md` 永远是
   JSON 的渲染，不手改）。

## 3. 预期结果与判据

* `C8`（扩展带同物理）：扩展网格在共享频点（12 rad/s）复现已提交 `B`，逐自由度
  `|A,B(extended) - A,B(committed)|` 相对差 ≤ 5%（`extension_overlap` 容差）。
* `C3a_extended`（零自由参数 KK 闭合）：在 20 rad/s 宽带上逐自由度
  `|A_BEM - A_KK| / max|A_BEM| ≤ 10%`（算例容差）。
* `C9`（代数衰减）：surge/sway/yaw 阻尼尾部指数 `p > 2.05`（`slope_bar`），
  使截断预算 `∫^∞ B(s) ds/(s²-ω²)` 有限 → 三个自由度从「经验一致」升级为「带限数据蕴含」。

## 4. 诚实边界

* 本单**递交**了重计算，但作业 `cj4321` 在集群排队（33 作业在前），
  本单结束时尚未落盘 → `C8/C9` 仍为 `not_run`，**不虚报**为已通过。
  这是 honest-FAIL 的如实记录：验证欠账已启动、未闭合。
* 扩展带扫频是**同一 BEM 求解器**的另一条频带，`C8/C9` 是跨频带一致性检查，
  不能检测辐射求解本身的系统误差；外部锚点仍由 `C10`（capytaine 独立求解器）承担。

## 5. 后续建议（给下次自找活）

* 作业 `cj4321` 完成后：`fetch` 取回 55 个 `point_*.npz` → `assemble_cummins_extended.py`
  装配重判 → 若 `C8/C9` 通过，把 `case_result.json` 判据从 16/16 更新为 19/19，
  并发布「扩展带闭合」报告（含 surge/sway/yaw 尾部指数 vs 20 rad/s 的曲线图）。
* 若 `C9` 在 20 rad/s 仍不进入代数衰减（`p ≤ 2.05`），则需评估是否把频带再扩
  （如 30 rad/s，需更细网格），或把该三自由度判据降级为「经验一致」并显式记录。
