# 交差审计 — m17203 CASE-BEM-DEAN-DIFFRACTION（TensorBEM）

审计单 m17236 · 审计对象 m17203（succeeded，branch `agent/m17203` @ `3c0310e`）
· 独立复跑树 `/tmp/aud_m17236` = `git archive main@96f19f7`（= m17203 的 parent）+ 仅 24 个交付路径
· 判级声明 **L4**（详见 §2/§6）· 结论 **PASS**

## ① 案例介绍

二维深水线性势流：单位宽度 Airy 波（A=1 m）沿 +x 越过完全潜没圆柱（a=1 m，轴深 f=2 m，f/a=2），
求一阶激励力 F_x、F_z（N/m）、反射 R、透射 T 与透射相位 δ。

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| 半径 a | 1.0 m | 波数 k = ka/a | 0.6 1/m |
| 轴深 f | 2.0 m（f/a=2） | 深水波长 λ | 10.4720 m |
| 入射波幅 A | 1.0 m | 周期 T | 2.5903 s |
| 密度 ρ | 1025 kg/m³ | 面元数 N | 128（扫 32–384） |
| 重力 g | 9.80665 m/s² | 每面元求积点 n_q | 24（扫 6–48） |

判据（**先于运行声明**）：每个力指标相对误差 < 5%（任务 goal 原文）；
定理量按绝对量判：|R| < 1e-9（Dean 1948 零反射）、||T|−1| < 1e-9、
∠(F_z/F_x) = −90°±1e-3°（深水圆偏振）、相位误差 < 1°。

> **基准裁判核验（本单第一项必查）**：`tensor_benchmark_cases` 中
> `anchor_slug='CASE-BEM-DEAN-DIFFRACTION'` 命中 **0 行**（案例池有
> `tensor_case_pool.case_code=BEM-DEAN-DIFFRACTION`，mission 17203，status=dispatched；
> 基准表却无登记行）；`tensorbench_results` mission 17203 **0 行**；
> 平台 `GET /tensor/bench/l4-verdict/17203` = `CLAIMED_NOT_PROVEN`。
> 即：**机器基准裁判缺席**（登记链断点在平台侧，见 §5 摩擦）。本审计因此以
> mission goal 的 5% 声明 + 文献定理锚点为准，并逐项独立复算，不采信报告自述。

## ② 怎么做的（审计方法与工具链）

1. **取证**：读 m17203 工作树与 `tensor_reports` id=16214（9140 字符 markdown，六段齐全）。
   交付件 24 个路径全部 sha256 双取（原件 vs 审计副本），`23/23` 一致 →
   `verify.hash_integrity_all_equal = True`。
2. **独立复跑**：新建干净树（parent 96f19f7 + 仅交付路径），跑
   `pytest tests/test_case_dean_diffraction_m17203.py -q` → `5 passed, 1 warning in 16.33s`；
   重算标称算例 `one_run(panels=128, panel_nodes=24)`，与交付 `stage_core.json` 逐字段比对：
   **36 个数值字段最大相对差 0.00e+00**（仅 `runtime_seconds` 变化）。
3. **重算判据**：从 `all_stages.csv` 28 行重导误差/不变量（不引用报告文字）；capytaine 外部软件
   L=40 单点复跑：交付 0.8633% / 0.6081%，
   复跑 0.8633% / 0.6081%。
4. **图-数一致性**：用 `scripts/m17203_figures.py` 从交付 JSON 重画 5 图，
   5/5 与交付 PNG **sha256 逐字节相同**（图非手绘/非改数）。
5. **报告零手抄**：用 `scripts/m17203_build_report.py` 从工件重生成 markdown，与库内发布版
   （tensor_reports#16214）比对：225 行 vs 224 行，**仅差 1 个空行** → 表内数字均为程序注入。

## ③ 计算结果（ref vs got，审计侧重算）

标称算例 ka=0.6, f/a=2, N=128, n_q=24：

| 量 | ref（多极子闭式，`analytic.py:703`） | got（面元，`free_surface_solver`） | 误差 | 判据 | 判定 |
|---|---|---|---|---|---|
| F_x [N/m] | 2422.7475 -10226.9181i | 2455.2918 -10262.1559i | 0.4564 % | <5% | ✅ |
| F_z [N/m] | -10226.9181 -2422.7475i | -10262.1559 -2455.2918i | 0.4564 % | <5% | ✅ |
| 透射相位 δ | 26.6552° | 26.9109° | 0.2557° | <1° | ✅ |
| \|R\| | 0（Dean 1948） | 5.492e-17 | 5.492e-17（绝对） | <1e-9 | ✅ |
| \|T\| | 1 | 1.000000000000000 | 1.110e-16 | <1e-9 | ✅ |
| 能量 \|R\|²+\|T\|²−1 | 0 | -2.22e-16 | 2.22e-16 | <1e-9 | ✅ |
| 圆偏振 ∠(F_z/F_x) | −90° | -90.0000000000° | 1.42e-14° | <1e-3° | ✅ |
| Haskind 互易 vs 直接 | — | — | 0.0005 % | <0.1% | ✅ |
| MFS 参考 vs 多极子 | — | — | 0.0377 % | <5% | ✅ |

28 行扫描最坏值：力 1.4190%（N=32），相位 0.8375°；
定理不变量全域 max|R|=1.966e-16、max||T|−1|=2.220e-16、
max 能量残差=4.44e-16、max 线性残差=1.84e-15、
max 条件数=3.23。
capytaine 3-D→2-D 极限 5 点最坏 2.3425%（L/λ=1.91）。
网格收敛拟合阶 -0.88（N=32→384：1.4190%→0.1596%）。

图：`figures/audit_dean_error_and_convergence.png`（(a) 28 行误差 vs 5% 门；(b) 网格收敛）。

## ④ 结果图

* 审计图 `figures/audit_dean_error_and_convergence.png`。
* 交付 5 图（`refs/figures/fig1..fig5.png`）经重画 **sha256 逐字节一致**，
  说明 fig1（|F_x| vs ka）、fig2（网格收敛）、fig3（误差扫描）、fig4（capytaine 极限）、
  fig5（自由面升高包络 + 总势场）均由其 JSON 数据驱动，无手绘改数。

## ⑤ 注意点 / 局限

1. **基准登记链断点（平台侧，S2 级系统性）**：案例池有 `BEM-DEAN-DIFFRACTION` 行且已派单，
   但 `tensor_benchmark_cases` 无对应 anchor 行、`tensorbench_results` 无平台重算行 →
   L4 的"机器容差裁判"实际未生效。本审计以 mission goal 声明的 <5% + 定理锚点替代，
   与既有 board-audit（verdict#490，m17174）判法一致：登记缺口记为 finding，不因此把交付降级。
2. **参考值是文献闭式程序化生成，不是抄表**：Ogilvie 1963 原表原页不可达（交付件 §1/§6 已如实声明）。
   `dean_cylinder_diffraction`（`src/tensorbem/analytic.py:703`）在 main 中**先于本单存在**，
   本单未改 src（`git diff main -- src/` 为空），故 ref 与 got 并非同一实现：
   ref=多极子级数解（另加 capytaine 独立软件佐证 2.34%），got=面元积分。
   但仍有一处残余风险：ref 与 got 同在 TensorBEM 仓内，若 `analytic.py` 的多极子实现本身错，
   二者不会互相发现——本审计只做了"capytaine 独立实现一致 + 定理不变量到机器精度"的三角锁定，
   未逐行数学复核 `_ursell_multipole_system`。
3. **交付分支未入 main**：`agent/m17203` 24 个路径在 `main@96f19f7` 均不存在
   （`git branch --contains 3c0310e` 只有本分支），故"系统集成"一侧仅成立到
   "算例驱动主求解器公共 API + 案例池登记"；库内主能力（`src/tensorbem/benchmarks/dean_diffraction.py`、
   `capability_evidence.py:417` 登记、`tests/test_dean_diffraction.py`）属先行既有集成。
4. **报告图未随文入库**：tensor_reports#16214 的 `figures` 只有 name/caption，
   无 `png_base64`（发布 payload 里其实带了 base64，库内被剥离；全库仅 5 条报告有 base64）→
   不阻塞（图在仓内可核），记为低危平台观察。
5. **范围**：仅完全潜没圆柱（f>a）；越浪/半潜需 MacCamy–Fuchs 分支，不在本算例。
   未做集群 loong64 重算（单次标称算例 13 s，本地复跑已足证）。

## ⑥ 误差判定（逐项，显式回答"是否<3%"）

| 指标 | 误差 | <3%？ | 依据 |
|---|---|---|---|
| 标称 F_x | 0.4564 % | **是** | 审计侧重算，ref=`analytic.py:703` |
| 标称 F_z | 0.4564 % | **是** | 同上 |
| 标称相位 δ | 0.2557° | **是**（0.256°） | 相位差按度判 <1° |
| \|R\|（Dean 定理） | 5.492e-17 | **是**（远小于 3%） | 绝对量 <1e-9 |
| 最坏力（N=32） | 1.4190 % | **是** | 28 行扫描上界，仍 <5% |
| capytaine 外部软件最坏 | 2.3425 % | **是** | 独立软件 L=40 复跑复现 |

**结论：全部关键量 vs 基准误差 < 3%（最坏 1.42%），且独立复跑逐位复现 → 判定 PASS。**

## 复现命令（审计侧实际执行）

```bash
# 干净树：parent main@96f19f7 + 仅 m17203 的 24 个交付路径
cd /tmp/aud_m17236 && PYTHONPATH=src python3 -m pytest tests/test_case_dean_diffraction_m17203.py -q
# -> 5 passed, 1 warning in 16.33s
PYTHONPATH=src python3 scripts/m17203_capytaine_crosscheck.py --lengths 40 --panels-per-length 1.5
PYTHONPATH=src python3 scripts/m17203_figures.py          # 5/5 PNG sha256 与交付一致
PYTHONPATH=src python3 scripts/m17203_build_report.py     # 重生成 markdown 与库内发布版仅差 1 空行
```
