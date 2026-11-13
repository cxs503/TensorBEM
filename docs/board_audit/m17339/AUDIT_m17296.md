# AUDIT m17339 — CASE-BEM-CYL-DRIFT-ASYMP（交差核验单，对象 = m17296）

- **锚点**: `CASE-BEM-CYL-DRIFT-ASYMP`（算例池 id=140 / `tensor_benchmark_cases.id=43`）
- **被审对象**: mission 17296（`agent/m17296`，status=succeeded，返修轮1，commit `cc3c087`）
- **同案前史**: mission 17204（真实 BEM 算例，branch `agent/m17204` @ `3fe683d`）；mission 17263 审计判 L4 → `tensor_maturity_verdicts.id=509` 被 maturity_gate **reopened 至 gate_level=1**（原因：evidence 指针不可机器核，impl/entry/run_artifacts 均判缺失）
- **本审计**: mission 17339，`verdict_by='board-audit'`，scope=`m17296 CASE-BEM-CYL-DRIFT-ASYMP`
- **纪律**: 基准唯一裁判 = `tensor_benchmark_cases`；报告好看不算数；auditor 只交可机器复核的指针 + 声明级别

---

## 1. 机器裁判（先于一切判据）

```
SELECT case_name, metric, tolerance, status, passed_at
FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-CYL-DRIFT-ASYMP';
```

- `id=43`, `repo=TensorBEM`, **`status=passed`**, `passed_at=2026-11-13 03:18:54Z`
- `ref_source`: MacCamy & Fuchs (1954) 绕射势（经 Chakrabarti & Gupta 2007 Eq.(3) 转录，pdf_sha256 记录在案）+ Maruo (1960) 远场动量定理（经 McIver 1986 Eqs.(5.1)-(5.2) 转录）+ 长波系数 `5π²/16`（无逐字页码 → 对 L5 记 WITHHELD，但由 m17263 独立重推 3.0842513989 @ ka=1e-4，差 7.7e-9）
- `tolerance`: `<5%`（算例池 id=140 tol_policy + mission 17204 判据）；生态 `<3%` 在主远场门上要求满足
- `metric` 逐项（本审计逐条复核，见 §4）：800 面板远场误差 2.028/1.068/1.317 %、长波律 0.011684 %、斜率 2.9964、近-远场闭式互证 2.84e-15、幅值² 标度 4.000000、面板阶梯 208/456/800/1776 = 远 7.396/3.179/2.028/1.406 % / 近 30.202/13.350/6.082/2.123 %

## 2. 取证：m17296 三件套逐项核（工作树 /root/tp-agent/work/m17296）

| 声称 | 实际（本审计实测） | 判定 |
|---|---|---|
| run_artifacts：`artifacts/cylinder_drift_asymptotic/{study.json,benchmark.json,fig_*.png}` | 存在且非空：study.json 8564 B、benchmark.json 3097 B、两张 PNG 53842/63268 B；sha256 `ef52eac4…`/`6859c468…`/`192cce05…`/`b5c7c332…`；**已固化快照**到 `docs/board_audit/m17339/snapshots/` | **真** |
| benchmark：ref vs got 数值对照 | `benchmark.json` 有 8 个 ka 的 near/far 数值 + 长波系数 ref `5π²/16=3.0842514` vs got `3.0838911`（rel 1.168e-4）+ 斜率 2.977 + 容差 5 % | **真（但只覆盖解析半）** |
| repro_cmd 可执行 | `PYTHONPATH=src python3 repro_cylinder_drift_asymptotic.py` 实测 **3.39 s** 退出 0，5 门全 PASS；重跑产物与交付件**逐字节一致**（JSON 相等，PNG sha256 相同） | **真** |
| 报告六段 | `tensor_reports.id=16292`（13601 字符 + 2 图）§1 问题/ref、§2 方法、§3 结果表、§4 图、§5 坑/局限、§6 误差判定 六段齐全，数字可回溯 study.json | **齐** |
| 提交/持久性 | `git show --stat cc3c087` 只含 2 个脚本（repro + post_report）；`artifacts/` 被 `.gitignore` 忽略，**工件未入库**；分支 `agent/m17296` **非** `origin/main` 祖先 | **缺陷（见 D1）** |

## 3. 独立复算（不做重做，只做 authorized reproduction）

1. **解析半一键重跑**：把 m17296 的 repro 脚本拷到 `/tmp/m17339_repro` 重跑（`PYTHONPATH=/root/tp-agent/work/m17296/src`）：3.39 s，5 门全 PASS；`study.json` 除 figures 绝对路径外**完全相等**，`benchmark.json` **完全相等**，两张 PNG **sha256 相同**（matplotlib 输出确定性）。
2. **BEM 半独立复算（本审计新增）**：只用 **main 树代码** `src/tensorbem/benchmarks/drift_far_field_3d_study.py::run_case`（面板 BEM → Kochin → Maruo）+ `tensorbem.cylinder_drift` 闭式，重算最便宜的一档 **208 面板 @ ka=0.5**：
   - 复算远场误差 `0.07395553351990684` vs 交付 `0.07395553351990668` → |Δ| = **1.7e-16**
   - 复算近场误差 `0.3020235303885376` vs 交付 `0.3020235303885386` → |Δ| = **1.0e-15**
   - 求解器相对残差 1.16e-15，耗时 8.9 s；脚本 `docs/board_audit/m17339/replay_bem_cell.py`，工件 `replay/bem_cell_208.json`
   - 结论：BEM 能力本体在 main 树上、可从 main 代码逐位复现；m17204 的 611 行 case driver 只是编排层（未合入 main，见 D1）

## 4. ref vs got 逐项误差表（裁判 = row 43 metric 逐条）

| # | 量 | ref | got | 误差 | 判据 | 结论 |
|---|---|---|---|---|---|---|
| 1 | BEM 远场 vs 闭式，800 面板，ka=0.5 | 2875.6570 N | 2817.327 N | **2.0284 %** | <5 % / <3 % | ✅✅ |
| 2 | 同上 ka=1.0 | 6685.9935 N | 6614.5977 N | **1.0678 %** | <5 % / <3 % | ✅✅ |
| 3 | 同上 ka=2.0 | 6306.3981 N | 6223.3267 N | **1.3173 %** | <5 % / <3 % | ✅✅ |
| 4 | 长波律 `5π²/16 ρgA²a(ka)³` @ ka=0.02 | 0.24810335 N | 0.24807436 N | **0.0116824 %** | <5 % / <3 % | ✅✅ |
| 5 | 长波指数斜率（ka≤0.1） | 3.0 | **2.99643** | 0.12 % | 3±0.05 | ✅ |
| 6 | 近-远场两条闭式路线（8 个 ka） | — | — | **2.84e-15** | <1e-6 | ✅ |
| 7 | 幅值² 标度 F(2A)/F(A) | 4.0 | **4.000000** | 0 | 1e-9 | ✅ |
| 8 | BEM 阶梯远场 208/456/800/1776 | 闭式 | 7.396/3.179/2.028/1.406 % | 单调、阶 ~2.15 | 收敛 | ✅ |
| 9 | BEM 阶梯近场 208/456/800/1776 | 闭式 | 30.202/13.350/**6.082**/2.123 % | 单调 | <5 % | ⚠️ 800 面板 **6.08 % 超 5 %**（ka=0.5；同网格 ka=1.0 4.12 %、ka=2.0 4.85 % 合格），1776 面板 2.12 % 合格 |
| 10 | 方位分辨率 180→360（ka=1.0） | — | 6.17e-5 | — | <1 % | ✅ |

第 9 项是本案唯一未达 <5 % 的单元，且被 row 43 与 m17204 报告**如实披露**（属网格不足而非公式错误；h^(-1/2) 收敛到 1776 面板即合格）。主判据（第 1–4 项）全部满足 `<5 %`，第 1–3 项同时满足生态 `<3 %`。

## 5. 判级

| 级 | 判据 | 本审计证据 | 结论 |
|---|---|---|---|
| L0 | 文档 | — | 超 |
| L1 | 数学/单元验证 | `tests/test_cylinder_drift_asymptotic.py` **29 passed**（本审计在 main 树重跑 1.91 s）；墙锚 `ρgA²/2` 精确、能量残差 ≤9.6e-15 | 过 |
| L2 | 独立功能实现（真代码 输入→计算→输出） | main 树 `src/tensorbem/cylinder_drift.py`（MF 精确场 + 近/远场漂移 + 纯 NumPy Bessel）、`src/tensorbem/benchmarks/cylinder_drift_asymptotic.py:279 main()`；真实工件(§2) | 过 |
| L3 | 系统集成 | `src/tensorbem/diffraction_drift_maturity.py:48 from . import cylinder_drift`，`:467/:583` 调 `near_field_cylinder_drift/far_field_cylinder_drift`，`:597` 判 (ka)³ 律；BEM 路由 `drift_far_field_3d_study.py::run_case` 由本审计独立复算 | 过 |
| L4 | 基准验证（完整算例+误差分析+可复现） | 见 §2/§4/§3：完整算例（8 ka + 长波带 + BEM 800 面板 ×3 + 4 档阶梯 + 方位灵敏 + 幅值标度 + 能量平衡）、逐项误差分析（10 项判据，9 合格 + 1 披露）、可复现（解析半 3.39 s 逐字节复现；BEM 208 档 8.9 s 逐位复现；750 s 全案命令在案且 m17263 已复跑 208/456 两档） | **过** |
| L5 | 科研工程成果 | 外部锚为解析闭式（MacCamy-Fuchs / Maruo），**无实验或印刷数表锚**：`5π²/16` 无逐字页码（WITHHELD），Wigley RAO vs 拖水池实验如实声明未做 | 不过（天花板 L4） |

**判级结论：L4。** 判级对象是本案的**功能实质**（能力本体在 main 树、真实 BEM 求解、逐项误差分析、可逐位复现），不是 m17296 的报告篇幅。m17296 单看自身交付只到 L3（它的 repro 只跑解析半），L4 的 BEM 半证据来自 m17204 工件 —— 本审计以三件事把它变成**可核**：①固化 sha256 快照入库（`snapshots/m17204_results.json`，sha256 `ac14e4e5…` 与 verdict 509 记录一致）；②用 main 树代码独立复算 208 面板档（|Δ|=1.7e-16）；③逐条对齐 row 43 的 10 项数值。

## 6. 发现（按严重度）

- **D1 (material, 非阻塞 L4)**: 工件不可持续 —— `artifacts/` 在 `.gitignore` 内、`cc3c087` 只提交 2 个脚本、分支未并入 `origin/main`；工作树一旦回收，BEM 数值证据将只剩本审计快照。**本审计已缓解**：`docs/board_audit/m17339/snapshots/` 固化 study/benchmark/results.json + 2 PNG（含 sha256），`replay/` 存复算 JSON。**遗留返修**：把 case driver `case_cyl_drift_asymptotic.py`（sha256 `7cd5cfd9…`，branch `agent/m17204`）与一份窄化 results 摘要提交到 `benchmarks/verified/cylinder_drift_asymptotic/`（不要放被忽略的 artifacts/ 目录）。
- **D2 (material, 非阻塞)**: m17296 的 `repro_cmd` **只重放解析半**（3.39 s，零 BEM 求解），与 row 43 的主动量（BEM 800 面板）不同源；"L4 delivered" 的措辞在报告 §Return-round-1 里强于其自身命令能证的强度 → 已在 §5 与本 verdict 中按实收窄（BEM 半由 m17204 工件 + 本审计 208 档复算背书）。
- **D3 (minor)**: 报告 §6 斜率写 2.977（拟合带 ka∈[0.02,0.30]），row 43 写 2.9964（拟合带 ka≤0.1）；两者各自合规但报告未标注拟合带差异，易被读成打架。
- **D4 (minor)**: 800 面板近场 6.08 %（ka=0.5）超 5 % 门，报告 §Return-round-1 的机器可判表未列该行（只列了 5 门全过），披露只在 row 43 / m17204 报告中 → 建议在报告加强调表格里显式列出。
- **D5 (minor)**: 六段报告无独立参数表段落（ρ/g/A/a/ka/kd 分散在 §2/§3.1 行文），无量纲数（ka, kd=5）虽齐全但不成表 —— 与算例池流程要求的"参数表（无量纲数完整）"有格式差距。
- **D7 (minor)**: row 43 的 metric 文字把长波误差写作 `0.011684 %`，工件实测 `0.0116824 %`（rel 1.6e-5 的四舍五入/转录差），量级远低于容差但登记文本应与工件对齐。
- **D6 (bookkeeping)**: `tensor_case_pool` 该案状态与 row 43 `passed` 的时间戳早于本轮返修（03:18:54 vs 04:19），即 row 43 的 "passed" 记录的是 m17204 那一轮，不是 m17296 —— 结榜时应以本 verdict 为准。

## 7. 未做 / 局限（auditor independence limits）

- **未重跑 750 s 全案**（800 面板 ×3 + 1776 档 + 方位 180/360）：只复算 208 面板档（8.9 s）。全案复现依据 m17204 `run.repro_cmd`（在案）+ m17263 对 208/456 两档的逐位复跑记录（`artifacts/independent_checks.json`，far |Δ|≤1.7e-16）；本审计对 800/1776 档**只核了数值一致性，未核求解器重放**。
- **未取 MacCamy-Fuchs (1954) / Maruo (1960) 原文页码**：ref_source 里的转录链（Chakrabarti & Gupta 2007、McIver 1986，含 pdf_sha256）为 m17204 声明，本审计在沙箱内无法独立验 PDF。
- **未审 PNG 像素内容**：只核 sha256 与生成路径（重跑后 sha256 一致即为确定性生成）。
- **未审 m17204 case driver 的代码质量**（611 行未合入 main 的编排层）；本审计用 main 树 `run_case` 旁路复算，故 L2/L4 结论不依赖该 driver。

## 8. 处置

**approve with notes → level 4（confirmed 依据由 maturity_gate 机器复核 gate_level）。**
返修建议（不阻塞本案结榜，可开廉价 successor）：`benchmarks/verified/cylinder_drift_asymptotic/` 入库（driver + 窄化 results + 复算脚本），并在报告 §6 表内补 800 面板近场行与拟合带标注。