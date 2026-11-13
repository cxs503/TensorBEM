# m17327 交差核验：CASE-BEM-MARUO-DRIFT（TensorBEM · m17276）

**裁定：L3**（board-audit；非重做，只核验）。核心结论：真代码 + 真求解器链路，闭式 ref 经独立重推验证，但**生产分辨率下 ka=2 单元 5.331% > 5% 判据未过**，且 benchmark 登记表中本锚点 **0 行**——L4 双阻塞，诚实降级到 L3。

## 1. 取证（三件套是否存在）

| 件 | 结论 | 证据 |
|---|---|---|
| run_artifacts | ✅ 全部存在 | `artifacts/maruo_drift_far_field_case/study.json`（16 cells，sha256 见 existence.json）；docs 侧同 sha256 副本 + 3 张 png |
| benchmark ref vs got | ✅ 数字自洽 | 16 cells 全部按 `|got-ref|/ref` 重算，与工件 `rel_error` 逐一吻合=True |
| repro_cmd | ✅ 可执行（但为 merge-only） | 重放 rc=1（与 gate outcome 一致：True），study.json 除墙钟外逐字节一致=True；四个生产单元另做集群重跑（见 §4） |

## 2. 基准裁判（唯一裁判）

`SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-MARUO-DRIFT'` → **0 行**。

因此机器裁判缺席，本次判据取任务 goal 原文：**「容差判据：<5%」**（0 行不是「无判据」，是「登记未落地」——与 m17204/CASE-BEM-CYL-DRIFT-ASYMP 同一个生态缺口，该锚点后来才补登 id=43）。

| ka | 生产网格 panels | got [N] | ref [N] | 重算误差 | 报告声明 | <5% |
|---|---|---|---|---|---|---|
| 0.2 | 800 | 233.295 | 240.591 | 3.032% | 3.032% | ✅ |
| 0.5 | 800 | 2779.98 | 2875.66 | 3.327% | 3.327% | ✅ |
| 1.0 | 800 | 6480.77 | 6685.99 | 3.069% | 3.069% | ✅ |
| 2.0 | 800 | 5970.22 | 6306.4 | 5.331% | 5.331% | ❌ |

生产网格最坏单元：ka=2.0（800 panels）**5.331%**，声明 5.331%，一致；`G3_bem_vs_ref_production` measured=0.0533082 passed=False，工件 `all_passed`=False。**交付自己就没说过通过**——mission 头行写的是 「G3 FAIL(5.33% @ ka=2.0)」，auto-approve 仍把 status 置 succeeded，这是平台验收口径问题，不是交付方粉饰。

## 3. ref 的独立重推（关键：不 import tensorbem）

`scripts/audit/m17327_ref_independent.py` 只用教科书定义 + scipy 重写：MacCamy–Fuchs 余弦型散射系数 `A_m=J_m'(ka)/H_m'^(1)(ka)`、`S(θ)=-Σ ε_m A_m cos mθ`、Maruo 远场积分，然后**暴力梯形求积**（21000 点，对有限三角多项式精确）算 `∫|S|²(1-cosθ)dθ`。结果：

| ka | 交付 ref [N] | 独立重推 [N] | 相对差 |
|---|---|---|---|
| 0.2 | 240.5906823 | 240.5906823 | 2.96e-13 |
| 0.5 | 2875.656993 | 2875.656993 | 4.27e-13 |
| 1.0 | 6685.993519 | 6685.993519 | 7.63e-13 |
| 2.0 | 6306.398054 | 6306.398054 | 2.23e-12 |

外部锚点（唯一真正外部的）：ka→0 极限独立重推得 3.08433929，vs 经典 `5π²/16 = 3.08425138`，相对差 2.85e-05（残差为 O(ka²) 外推误差）。**注意踩坑**：本审计脚本第一版用全整数 m 的复指数求和、相位因子写成 `(-i)^m`，奇 m 相对相位翻转，长波极限变成 π²/16 = 1/5 的经典值——内部求积自洽（modal vs quad 1e-11）却整体错 5 倍，**只有外部锚点抓得住**。这也是为什么「闭式 vs 仓内求积 2.4e-16」不能算外部验证：两条路线共用同一套系数。

## 4. 可复现性：merge 重放 + 生产单元真重算

* 交付 repro_cmd 是 `--merge artifacts/.../cell_ka_*.json`：重放 rc=1，输出与交付 study.json 除 `elapsed_seconds` 外**逐字节一致** = True。但它**不重跑 BEM 求解**，所以只能证明 gate/ref 派生逻辑可复现，证明不了求解器可复现。
* 集群重跑已递交（**cj1489**，TensorBEM 检出 + 四个生产单元 + 两个越档网格），但审计落卷时作业仍在排队（队列 611）→ 本项记 **not_run**，不记 pass。这是本单唯一的未跑项，不影响定级：L4 的第二阻塞（判据未过）用的是**交付自己的**最细档数字 5.04%（1200 panels），比 5% 判据仍差 0.04 个百分点。

## 5. 报告六段与数字溯源

六段齐全 = True（①案例+ref出处 ②方法 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式）。④ 表解析 16 行，可溯源 16 行，漂移 0 行。

**不可溯源的正文数字（material，不是 L 级阻断）**：
* '(32,12,3.0)=832 三角形' (⑤-1) but study.json records 800 panels for that mesh
* '(32,6,2.5)=416 三角形' (⑤-1) is not a rung of the delivered study (panel counts: [312, 544, 800, 1200])
* 'ka=1.0 ... 降到 fine 的 3.01%' (⑤-1) quotes the *finer* value (3.01%); fine is 3.07%

## 6. 定级（L0–L5，级别=功能实质）

| 级 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元验证 | 闭式 + 单元测试 | ✅ 13 tests pass（审计重跑 3.28 s）；闭式经审计独立重推 1e-12 一致 |
| L2 独立功能实现 | 输入→计算→输出真代码 | ✅ `bem_case`：建网格→`solve_diffraction_3d`→`bem_kochin`→`maruo_far_field_drift`→误差，569 行，无硬编码 |
| L3 系统集成 | 接入主求解器 | ✅ 直接调用仓内三条主求解器 API，16 单元端到端跑通，残差 ~1e-15 |
| L4 基准验证 | 完整算例+误差分析+可复现+**判据通过** | ❌ 双阻塞：登记表 0 行 + 生产网格 ka=2 5.331% > 5% |
| L5 科研工程成果 | — | ❌ |

**裁定：L3。** 不是 L2——链路是真集成不是孤立脚本；不到 L4——判据没过 + 裁判缺席。百页报告不改变级别。

## 7. findings（按严重度）

* **[blocking]** blocking-for-L4: tensor_benchmark_cases has 0 rows for anchor CASE-BEM-MARUO-DRIFT -> no machine referee; criterion only in mission-goal text
* **[blocking]** blocking-for-L4: declared <5% criterion FAILS at the production mesh (ka=2.0, 800 panels, 5.331%)
* **[material]** prose numbers with no artifact source (3 items)
* **[note]** delivered repro_cmd is merge-only (no BEM re-solve) -- reproducibility of the solver rests on the audit cluster replay

## 8. 未跑 / 不知道的

* 与**实验或文献表格**的对照没有跑：沙箱无外网，交付与审计都只能对闭式定律（这是真实的限制，不是选择）。因此「外部锚点」只有长波 `5π²/16` 一条。
* 有限水深、非圆截面、多体遮蔽均不在本算例范围（交付 ⑥ 已声明）。
* 集群重跑的架构是 loong64，与 GitHub CI 的 x86 不同；本单只核数值一致性，未核跨架构一致性。

## 9. 工件

* `artifacts/audit_m17327/evidence.json`
* `artifacts/audit_m17327/existence.json`
* `artifacts/audit_m17327/fig_m17327_error_vs_panels.png`
* `artifacts/audit_m17327/figure.json`
* `artifacts/audit_m17327/finding.json`
* `artifacts/audit_m17327/independent_ref.json`
* `artifacts/audit_m17327/judge.json`
* `artifacts/audit_m17327/merge_state.json`
* `artifacts/audit_m17327/referee.json`
* `artifacts/audit_m17327/report_check.json`
* `artifacts/audit_m17327/repro.json`

图：`artifacts/audit_m17327/fig_m17327_error_vs_panels.png`（sha256 728ac75514390ad3fc556160677ce82e2931eee8221c92990c3dd4568e4777a4）

> 工件镜像：`artifacts/audit_m17327/` 在 .gitignore 内（git 忽略），因此全部 JSON + 图已提交到 `docs/board_audit_m17327/evidence/`（同 sha256，见 evidence.json）。
