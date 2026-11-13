# AUDIT m17176 —— CASE-BEM-SPHERE-ADDED-MASS（球体附加质量，TensorBEM）

审核人：board-audit m17232（独立席位；未参与 m17176 实现）
审核对象：mission 17176，分支产物 `/root/tp-agent/work/m17176`（runs/m17176/*, docs/dev_report_m17176_sphere_added_mass.md），
已发布报告 tensor_reports#16204。
基准裁判：`tensor_benchmark_cases` 中 `anchor_slug='CASE-BEM-SPHERE-ADDED-MASS'` 的行 —— **本审计开始时该表 0 行**（F2）。

## 1. 取证（三件套逐项）

| 项 | 结论 | 证据指针 |
|---|---|---|
| run_artifacts 真实存在 | 真 | `refs/sphere_added_mass_sub{1,2,3,4}.json`（本审计 sha256 固定，来自 m17176 worktree runs/m17176/）+ `refs/sensitivity_sub2.json` + `refs/flux_panels_sub2.csv`(320 行逐面板) |
| 实现是真代码 | 真 | `src/tensorbem/benchmarks/sphere_added_mass.py:34 run()` → `solve_laplace_surface(..., domain="exterior")`；输入→计算→输出，无硬编码（负控见 §4） |
| repro_cmd 可执行 | 真 | `PYTHONPATH=src python3 -m tensorbem.benchmarks.sphere_added_mass --subdivisions {0,1,2}`：本审计实跑通过，sub1/sub2 **逐位一致** |
| 报告六段齐全 | 真 | tensor_reports#16204：①问题/ref ②方法 ③参数表 ④对照表 ⑤收敛+敏感性 ⑥失败模式/工件/repro；2 张图内嵌 |
| 报告数字 vs 工件 | 一致 | §4 四行 m_A 与 err 全部命中工件（最大相对偏差 <2e-4，仅四舍五入） |

## 2. 独立复算（auditor rerun）

| sub | panels | m_A 主件 | m_A 复算 | 逐位一致 | 相对误差（主件） | 复算误差 | <3% |
|---|---|---|---|---|---|---|---|
| sub1 | 80 | 2007.732789 | 2007.732789 | 是 | 6.475923e-02 | 6.475923e-02 | FAIL |
| sub2 | 320 | 2109.442900 | 2109.442900 | 是 | 1.738069e-02 | 1.738069e-02 | PASS |
| sub3 | 1280 | 2137.164559 | 2137.164559 | 是 | 4.467404e-03 | 4.467404e-03 | PASS |
| sub4 | 5120 | 2144.329111 | 未跑（见 §6） | - | 1.130017e-03 | - | PASS |

ref = 0.5·ρ·V = 2146.7549799530 kg（Lamb 1932 §6.10；仓内 `analytic.ellipsoid_added_mass((1,1,1),1025)` = 2146.7549799481294，与闭式相对差 2.28e-12）。
**关键量判定：320 面板起全部 <3%，最细网格（5120 面板）误差 1.1300e-03（0.1130%），满足 <3% 判据，且优于生态 3% 线约 26 倍。**

复算数值：sub1 = 2007.7327892588562, sub2 = 2109.4428999421184，与主件逐位相同（abs diff 0.0）。
标度不变性：ρ=1000/U=2 复算 rel_err = 1.738069e-02，与 ρ=1025/U=1 的 1.738069e-02 一致（差值 <1e-15）→ 与主件 sensitivity_sub2.json 自洽。

## 3. 主要发现（按严重度）

**F1（medium，非阻塞，实质）报告收敛阶算错一倍 —— 并被本审计否定其推论。**
主件报告 §5 记「观测阶 0.95 / 0.98 / 0.99」，据此推断「球体附加质量积分继承通量一阶，README 的 h² 声明与实测不符，需后续复核」。
本审计复算：icosphere 每细分一次边长为 1/2，故阶应为 p=log2(e_s/e_{s+1})，实测
**1.90 / 1.96 / 1.98**（依次），即 ≈2，与 README:62「observed order near 2」和椭球/圆柱族 1.99–2.01 一致。
主件的 0.95/0.98/0.99 恰为 log4 空间值（**0.95 / 0.98 / 0.99**）——即除错底数造成的算术滑移。
真正近一阶的量是**逐面板通量**：0.94 / 1.09 / 1.09。
影响：不影响任何 m_A 数值、误差与 <3% 判定；只影响收敛性解读与对 README 的错误质疑。

**F2（medium，系统性）** 本算例在机器裁判表 `tensor_benchmark_cases` 中 0 行 → 判据此前只在任务书文本里。本审计按 m17201 先例登记该 anchor 行（见 insert_verdict.sql），使后续可机器复核。

**F3（low）** 主件称 sub3/sub4 递交集群 cj1448：核 cj1448 终态 = `killed`（提交 11-13 01:55，无集群产物被引用），报告亦如实说明数字由本地 setsid 路先出 → 主张半真但自洽；数值本身已由本审计本地复跑独立证实。

**F4（low，范围上限）** 该算例施加**解析 Dirichlet 迹** u=−½Ua·cosθ 再解 q：验证的是外域单层/DtN 解算子，不是移动体 Neumann 物理 BVP。故 level 封顶 L4，不主张 L5。

**F5（info）** 80 面板误差 6.4759e-2 > 3%（主件已如实判 FAIL）；320 面板起 PASS。

**F6（info）** m17176 交付物只在其 worktree（runs/m17176/），未进仓；本审计已按 sha256 固定副本至 `refs/`。

## 4. 负控（证明"不是硬编码"）

`negcontrol_sphere_added_mass.py`（80 面板，走同一生产入口）：
- 换角向模态（cos2θ）→ m_A = 3524.3648，相对 ref 偏 64.17% → 输出随输入变（非硬编码）。
- 半径 2.0 → m_A = 16061.8623 = 8×（a³ 标度，精确 8.0），相对误差相同（6.475923e-02）。

## 5. 判级

**L4（基准验证）**：真 BEM 求解 + 外部经典闭式锚 + 4 点网格细化误差表 + 可复现（复跑逐位一致）+ 测试绿。
不达 L5：无实验/印刷表格级外部锚；算例为解析迹的半解析验证面（F4）；报告的收敛阶分析有误（F1）。
不因附百页文档升/降级；级别=功能实质。

## 6. 未做的检查（honest: not_run）

| 检查 | 状态 |
|---|---|
| sub3/sub4 本审计本地复跑 | 提交时仍在跑（大网格稠密解 + 条件数估计）；sub1/sub2 已逐位复现，sub3/sub4 仅主件数值 + 集群 cj1466 递交在途 |
| 集群 cj1466 结果 | 排队中（queued_ahead 数百），未取回 |
| 与实验/CFD 数据对比 | 无（球体附加质量无实验锚，经典闭式为唯一基准） |
| 自由面/船舶水动力适用性 | 不在本算例范围（主件已声明） |

## 7. 处置

**approve with notes**：数值与工件成立，判 L4；F1 要求把收敛阶改正（本审计报告已给正确值），README 无需修改（README 原本正确）。

## 复现命令

```bash
# 主件算例
PYTHONPATH=src python3 -m tensorbem.benchmarks.sphere_added_mass --subdivisions 2
# 本审计复算 + 负控
PYTHONPATH=src python3 docs/board_audit/m17232/negcontrol_sphere_added_mass.py
python3 docs/board_audit/m17232/analyze.py && python3 docs/board_audit/m17232/build_audit.py
```
