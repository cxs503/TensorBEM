# 交差审计报告 · m17373 `CASE-BEM-ICE-DISPERSION`（TensorBEM）

审计单 **m17429**（board-audit）· 对象 m17373（原单 m17274 返修轮1）· 锚点 `CASE-BEM-ICE-DISPERSION`
被审 commit `26cbde4516fcf6756105ec6584dcd097b51b399d`（分支 `agent/m17373`）· 审计日期 2026-11-13
**结论：判级 L3（声明 3，机器门槛见 §6）** — 真实 BEM 实现 + 真实验证数字 + 可复现；差 L4 的两项均写明。

---

## 1. 对象与判据（审计契约）

| 项 | 内容 |
|---|---|
| 算例 | 冰覆水域质量加载色散（Weitz–Keller 1950），深水系列 H=200 m（kH=4..80）/ 浅水系列 H=2 m（kH=0.04..0.4） |
| 参数 | h_i=1 m, ρ_i=917, ρ_w=1025, g=9.80665, μ=0.894634, ω_cutoff=3.31084 rad/s |
| 基准判据（唯一裁判） | `tensor_benchmark_cases` WHERE `anchor_slug='CASE-BEM-ICE-DISPERSION'`：1 行，tolerance **<5%**；metric 为散文汇总「deep-water k rel err 4.50e-14; shallow k rel err 2.87e-02 (kH=0.426…, omega 3.14e-14); thickness 0.0%; RK4 1.03e-13; Robin 4.47e-15」——**该行无逐项 ref/got 数值对，容差只有一条**（见 F4） |
| 验收三件套 | run_artifacts / benchmark ref vs got / repro_cmd 可执行 |
| 审计方法 | 独立复算：脚本 `docs/audit_m17429/audit_m17373.py`（不 import 被审方的 benchmark driver，只 import 交付代码本身）→ `audit_metrics.json`、`logs/*.log` |

审计工作树与证据：`docs/audit_m17429/`（脚本 / 指标 JSON / 日志 / 3 图）。

---

## 2. 取证结果（三件套逐项核）

| 项 | 核验方式 | 结论 |
|---|---|---|
| impl_files 真代码 | 读 `src/tensorbem/ice_mass_loading_dispersion.py`（1047 行，sha256 `9d48b378…`） | **真实现**：周期条带常值面元网格（`build_periodic_strip`）、单层/双层积分 `assemble_strip_matrices`、周期连接 `tie_operators`、稠密 BIE 求解 `_solve_tied_system`、DtN 装配 `dirichlet_to_neumann`、L2 Rayleigh 商取 Λ。自由面 Dirichlet `cos(kx)` 给定后**解出**通量再取商 —— 不是把解析解抄回来 |
| entry | `src/tensorbem/benchmarks/ice_mass_loading_dispersion.py`（739 行）+ `__init__.py:230-244` 导出 | 存在，CLI 入口实测可跑 |
| run_artifacts | `docs/case_bem_ice_dispersion/{numbers.json, table.md, REPORT.md, figs×4}` | 均非空；PNG 头校验：896×588 / 1540×588 / 1820×532 / 1456×560 |
| benchmark ref vs got | 审计方独立复算 ref（闭式在审计脚本里从报告附录重写，不 import 被审方）→ 9 行逐行对照 | **max \|ω_got/ω_ref−1\| = 0.3850 % < 5 %**（逐行见 §3）；mission 自报 C1=0.003846 与之一致 |
| repro_cmd 可执行 | 在隔离副本 `/tmp/aud17373`（m17373 @26cbde4 的 src/tests 字节拷贝）重跑 `PYTHONPATH=src python -m tensorbem.benchmarks.ice_mass_loading_dispersion --figure`（30.8 s）+ `pytest tests/test_ice_mass_loading_dispersion.py` | **exit 0；31 passed（15.1 s）**；重生成 numbers.json 与被审工件**逐字段相同**（除 `environment`），摘要 sha256 `a101fa49b9aa1cb6…` 一致（`logs/repro_*.log`、`numbers_regenerated.json`） |
| 报告六段 | 读 `docs/case_bem_ice_dispersion/REPORT.md`（230 行）与 `tensor_reports` id=16372（markdown 5737 字符、figures 4 张） | 六段齐全：①案例+ref出处 ②方法与实现 ③参数表 ④对照表 ⑤收敛+敏感性 ⑥失败模式与工件；另附推导附录 |

---

## 3. 数字（程序化复算，零手抄）

### 3.1 算例 9 行 ref vs got（审计方独立 ref）

| 系列 | k [1/m] | H [m] | kH | ω_got | ω_ref(审计复算) | 误差 | Λ_BEM | Λ_ref=k·tanh(kH) | Λ 误差 | 面元 |
|---|---|---|---|---|---|---|---|---|---|---|
| deep | 0.02 | 200 | 4 | 0.4391210 | 0.4388147 | 0.0698 % | 0.020015 | 0.019987 | 0.1421 % | 154 |
| deep | 0.05 | 200 | 10 | 0.6855525 | 0.6850826 | 0.0686 % | 0.050072 | 0.050000 | 0.1434 % | 164 |
| deep | 0.1 | 200 | 20 | 0.9493843 | 0.9487549 | 0.0663 % | 0.100145 | 0.100000 | 0.1446 % | 172 |
| deep | 0.2 | 200 | 40 | 1.2906232 | 1.2898283 | 0.0616 % | 0.200291 | 0.200000 | 0.1454 % | 182 |
| deep | 0.4 | 200 | 80 | 1.7005790 | 1.6996670 | 0.0537 % | 0.400583 | 0.400000 | 0.1458 % | 190 |
| shallow | 0.02 | 2 | 0.04 | 0.0888590 | 0.0885185 | 0.3846 % | 0.000806 | 0.000800 | 0.7712 % | 144 |
| shallow | 0.05 | 2 | 0.10 | 0.2210627 | 0.2205755 | 0.2209 % | 0.005006 | 0.004983 | 0.4442 % | 144 |
| shallow | 0.1 | 2 | 0.20 | 0.4366132 | 0.4361199 | 0.1131 % | 0.019783 | 0.019738 | 0.2304 % | 144 |
| shallow | 0.2 | 2 | 0.40 | 0.8358278 | 0.8353256 | 0.0601 % | 0.076087 | 0.075990 | 0.1285 % | 144 |

### 3.2 审计方自己的收敛阶梯（不用被审方的 convergence_study）

| 系列 | 8 ppw | 16 | 32 | 64 | 128 | 观测阶 |
|---|---|---|---|---|---|---|
| deep (k=0.1,H=200) | 3.6334 % | 1.2941 | 0.4301 | 0.1446 | 0.0517 | 1.49 / 1.59 / 1.57 / 1.48 |
| intermediate (H=20) | 2.3101 % | 0.8688 | 0.5178 | 0.1370 | 0.0491 | 1.41 / 0.75 / 1.92 / 1.48 |
| shallow (H=2) | 5.9085 % | 2.9660 | 0.9091 | 0.2304 | 0.0595 | 0.99 / 1.71 / 1.98 / 1.95 |

单调下降（三系列都 True），观测阶 ~1.5 —— 与被审方报告「常值面元、角点通量、阶 ~1.5」的自述一致（独立复现，非抄）。

### 3.3 注册基准行（散文 metric）逐项复算（`tensorbem.ice` 闭式族）

| metric 项 | 注册值 | 审计复算 | 判定 |
|---|---|---|---|
| deep-water k 求解一致性 | 4.50e-14 | 2.20e-13（Newton vs 自写二分） | 同量级 PASS |
| shallow k 相对长波极限 | 2.87e-02 @ kH=0.426 | kH=0.426472 ✓、ω 回收 3.138e-14 ✓（注册 3.14e-14）但长波偏差复算 **4.13 %**（h_i=0.9,H=6,T=12） | **部分不可复现**（F4），仍 <5 % 但余量仅 0.9 pp |
| RK4 shooting | 1.03e-13 | 1.43e-13 | 同量级 PASS |
| Robin 残差 | 4.47e-15 | 4.02e-15 | 同量级 PASS |
| 冰厚研究 | 0.0 % | 1.11e-14 | PASS |

---

## 4. 方法要点（被审方实际做了什么）

单个周期单元 `0≤x≤2π/k, −H≤z≤0`，四边常值面元；混合边值 BIE（自由项 1/2）+ 周期连接 `φ_R=φ_L, q_R=−q_L`：
1. **driven**：给定自由面 `φ=cos(kx)`，解出 `q=(dφ/dz)`，取 L2 商 `Λ=Σq φ l/Σφ² l`；
2. **DtN**：装配自由面稠密 Dirichlet-to-Neumann 矩阵，经同一商取 Λ（独立代码路径）。
由 Robin 条件 `Λ=ω²/(g−μω²)` 反解 `ω² = gΛ/(1+μΛ)`；参考 `Λ_ref=k·tanh(kH)`，`ω_ref` 即 Weitz–Keller 闭式。
边界值复原检查：底/壁电位未被给定，解出后与精确模态 `cos(kx)cosh(k(z+H))/cosh(kH)` 对比（bv_error≈3e-3）。

---

## 5. 发现（按严重度）

| # | 严重度 | 发现 | 证据 | 影响 |
|---|---|---|---|---|
| **F1** | **blocking（对 L3 生效性）** | 分支 `agent/m17373` **未合并、且无远程引用**（`git branch -r` 无 m17373；main `fc205ca` 无该文件）→ 生态主检出里根本没有这套能力 | `git log --oneline -1 agent/m17373`=26cbde4；`ls /root/tensor-ecosystem/TensorBEM/src/tensorbem/ice_mass_loading_dispersion.py` 不存在；gate 的 `grep -r --include='*.py' -F strip_bem_mass_loading_omega /root/tensor-ecosystem/TensorBEM` 无输出 | 交差审计的机器门槛（L3 integration_callers）在 main 上必然失败 → gate 会降到 L2；且交付物仅存在于本地工作树，**有丢失风险** |
| **F2** | major | 「集成」只在库 API 层：新增 `tensorbem.ice.mass_loading_omega_bem`（ice.py:239）与包级导出，但**没有任何既有求解路径调用它**（`ice_rao.py`、`hydroelastic.py` 仍用闭式 `mass_loading_omega`）；唯一 caller 是测试 `test_ice_module_callers_use_the_bem` | `grep -rn mass_loading_omega_from_bem\|mass_loading_omega_bem src` 仅 ice.py/`__init__`/tests | 不能称「接入主求解器」；L3 只在「库导出 + 域模块包装」意义上成立 |
| **F3** | major（对 L4） | benchmark 的**头条量 ω 与 ref 共用同一条解析 Robin 映射**：`ω_got=f(Λ_BEM)`、`ω_ref=f(Λ_ref)`，f 单调 → ω 误差只是 Λ 误差的重新表达。BEM 真正**独立**算出并被验证的是深度因子 Λ（0.14 %@64 ppw） | §3.1 表：ω 误差 0.0537–0.3846 %，Λ 误差 0.1285–0.7712 %；两列单调对应 | 「BEM 验证了色散关系」是过强表述；诚实表述为「BEM 验证了 Λ，(Λ→ω) 由解析 Robin 换算」。L4 的 benchmark 因此只部分独立（非 Bessel 案那般同常数恒等式，故不判 tautology） |
| **F4** | minor | 注册基准行（`tensor_benchmark_cases`）是**散文 metric、无逐项 ref/got**，其中「shallow k rel err 2.87e-02」不可精确复现（同 kH=0.426 下审计得 4.13 %；h_i 扫描 0.3→1.0 给 3.29 %→4.27 %，无 2.87 % 的落点） | §3.3；`audit_metrics.json:R1_registered_case_metric.shallow_thickness_scan` | 基准判据部分机器不可裁；容差 <5 % 仍被满足但浅水项余量薄（0.9 pp） |
| **F5** | minor | 外部锚 **WITHHELD**：ref 是程序内闭式（Fox & Squire 1994 eq.(3) / Weitz & Keller 1950），未对原表原页；被审报告已自曝（REPORT.md §1.1） | REPORT.md:40-46 | L5 不予考虑（external_anchor 不成立） |
| **F6** | non-blocking | C2（两路线一致到 1.5e-14）说服力有限：driven 与 DtN 共用同一套 `assemble_strip_matrices`，非完全独立；真正独立的是「精确模态边界值复原」（bv_error≈3e-3）与「加密收敛」 | ice_mass_loading_dispersion.py:521-596；tests:154-159 | 报告未把它当独立证据夸大，仅作一致性检查，可接受 |
| **F7** | non-blocking | 测试断言实质（非空跑）：解析 Λ、DtN vs driven、底/壁复原、域内场、加密单调、负例输入、driver 6 判据齐全；容差 2–3 % 与 32 ppw 匹配 | tests/test_ice_mass_loading_dispersion.py:146-285 | 支持 L1/L2 |

**独立性限制（本条审计的）**：审计在同一台机器、同一 Python/numpy 环境（3.11/numpy 2.4.6）复算；参考值同为解析闭式（无法访问 1950 年原表）；未做二阶面元或三维冰盖等替代离散的交叉验证；集群（loong64 双节点）未使用（本算例 31 s 单核，无需）。

---

## 6. 判级与差异说明

| 级别 | 判据 | 本单 |
|---|---|---|
| L0 文档 | — | 不止 |
| L1 数学/单元验证 | tests 非空 | ✅ 31 passed（审计亲自重跑） |
| L2 独立功能实现 | 真代码 + 入口 + 运行工件 | ✅（1047 行真实 BEM；driver 出 numbers/table/REPORT/4 图） |
| L3 系统集成 | 主求解器/平台侧真调用 | **⚠ 有条件成立**：库导出 + `tensorbem.ice` 包装存在，但**主检出无此代码（F1）**、**无既有求解路径调用（F2）**。记为「分支内库级集成」 |
| L4 基准验证 | 完整算例+误差分析+可复现 | **数值实质基本达到**（0.385 %<5 %、独立收敛阶、byte 级可复现），但 **L3 前置在 main 不成立（F1）** + ref 与头条量共用解析映射（F3） + 原表锚 WITHHELD（F5）→ **不判 L4** |
| L5 科研工程 | + external_anchor | ✗ WITHHELD |

**声明级别 = 3**；若把「分支合并进 main」与「换用外部数值锚（原表或实验）」补上，L4 是下一个可达级（工作量小：合并 + 一个独立锚）。
机器门槛预期：gate 以 main 检出 grep `integration_callers`，会判 L2 并注记 —— 这是 F1 的机制后果，不是数字问题（confirmed L2+ 仍足以消账）。

---

## 7. 复现命令（审计方实跑）

```bash
# 隔离副本 = m17373 @26cbde4 的 src/tests 字节拷贝
cd /tmp/aud17373
PYTHONPATH=src python -m pytest tests/test_ice_mass_loading_dispersion.py -q        # 31 passed, 15.1 s
PYTHONPATH=src python -m tensorbem.benchmarks.ice_mass_loading_dispersion --figure  # exit 0, 30.8 s, all criteria passed
# 审计独立复算（本工作树）
cd /root/tp-agent/work/m17429 && python3 docs/audit_m17429/audit_m17373.py          # R1–R4
python3 docs/audit_m17429/make_figures.py                                            # 3 图
```

工件：`docs/audit_m17429/{audit_metrics.json, numbers_regenerated.json, logs/{independent_check,repro_benchmark,repro_tests}.log, fig_ref_vs_got.png, fig_convergence_independent.png, fig_depth_factor.png}`

---

## 8. 误差判定（逐项，显式回答）

| 关键量 | 值 | 基准 | 是否 <3 % | 是否 <5 %（注册容差） |
|---|---|---|---|---|
| 9 行 ω 最大相对误差 | 0.385 % | Weitz–Keller 闭式（审计方独立求值） | **是** | 是 |
| 64 ppw 下 Λ 相对误差（deep） | 0.145 % | k·tanh(kH) | **是** | 是 |
| 64 ppw 下 Λ 相对误差（shallow，kH=0.04） | 0.771 % | k·tanh(kH) | **是** | 是 |
| 128 ppw 下 Λ 相对误差（shallow） | 0.060 %（审计阶梯） | k·tanh(kH) | **是** | 是 |
| 注册行 shallow 长波偏差 | 4.13 %（复算） | √(gH)·k | 否 | 是（余量 0.9 pp） |

**结论：数值实质诚实达标（远好于 3 % 线）**；本单不给 L4，不是因为数字差，而是因为集成在 main 不生效（F1/F2）与外部锚缺失（F5），以及头条量非独立预测（F3）。