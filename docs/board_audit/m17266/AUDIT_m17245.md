# AUDIT m17245 — CASE-BEM-KP505-GCI（KP505 三密度 r=2 Roache GCI 收敛，TensorBEM）

- 交差核验单（非重做）：审计任务 m17266 / 审计者 `board-audit`
- 被审对象：mission 17245，分支 `agent/m17245`，head commit `ab3f9c1960995ca03cba2812125cfe9033c01fc4`
- 审计工作树：`/root/tp-agent/work/m17266`（证据快照 `docs/board_audit/m17266/snapshot/`，机器可核证据 `evidence.json`，独立重算 `independent_recompute.json`）
- 独立性限制：审计者是同平台同 harness 的智能体，共享同一容器与同一份二进制；对参考值只能做**转录级**核对（PDF sha256 + 抽文行），无法取得一次实验原表。

## 1. 三件套核验（逐项，带指针）

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 真存在 | ✔ | `benchmarks/verified/bem_kp505_gci/{case.json(26 KB),gci.csv,levels.csv,figures/kp505_gci_ladder.png,kp505_gci_indices.png,cache/13×json}`；审计快照 sha256 记于 `evidence.json.impl_origin.sha256` |
| 实现真代码 | ✔ | 驱动 831 行（`snapshot/kp505_three_density_gci.py`），调主线求解器 `tensorbem.propeller.open_water_point`（driver:299 / propeller.py:680）；`TODO|NotImplementedError|np.random` 命中 0 |
| tests | ✔ 8 passed | `tests/test_case_bem_kp505_gci.py`，审计侧复跑 0.78 s |
| benchmark ref vs got | ✘ 未达 3% | 见 §2 |
| repro_cmd 可执行 | ✔ **重跑过**（不是只读） | `PYTHONPATH=src python -m tensorbem.benchmarks.kp505_three_density_gci <outdir>`，165 s exit 0；`gci.csv` 逐字节一致，`levels.csv` 仅 `wall_seconds` 列不同，`case.json` 的 verdict/gci/criteria/anchor_comparison 全等，两图 md5 一致（`e1c322b7…` / `625299d5…`） |
| 平台机器裁判 | ✘ 缺席 | `tensor_benchmark_cases` 中 `anchor_slug='CASE-BEM-KP505-GCI'` **0 行**（全表 39 行无一 TensorBEM 行）；`tensor_case_pool` id=164 仍 `status='dispatched'`、`linked_anchor` 为 NULL；`GET /tensor/bench/l4-verdict/17245` → `CLAIMED_NOT_PROVEN, l4_benchmark_pass=false, rows=0` |
| 报告六段 | ✔ | `tensor_reports` id=16245（五/六节含判据逐条 + 工件 + friction），2 张内嵌图，`scenario_key=propeller_open_water_verification` |

## 2. 基准判据（唯一裁判）——honest FAIL 复核成立

算例自声明的容差 = GCI_fine ≤ 3%（case-pool `tol_policy`，动手前预声明）。

| 量 @J=0.70 | 验收线 | got GCI_fine | 观测阶 p | <3% |
|---|---|---|---|---|
| KT | ≤3% | **10.074%** | −3.967 | ✘ |
| 10KQ | ≤3% | **8.474%** | −1.125 | ✘ |
| eta0 | ≤3% | **11.249%** | −0.774 | ✘ |

审计者从 `levels.csv` 独立重算 p / GCI_fine / 渐近比（**不 import tensorbem**）：9 个 (J, 量) 单元与交付 `gci.csv` 最大偏差 < 1e-6 → 算术与工件自洽，没有事后改数。
判据 5/8 通过（G1,G2,G7,G8 ✔；G3 单调、G4 p≥1、G5 容差、G6 渐近准入 ✘），`fail_stage=G3-monotone`。

锚值对照（EFD 二手转录 @J=0.70：KT 0.193 / 10KQ 0.342 / eta0 0.6287）：

| 量 | D3 细网 got | 误差 | 外推 | 误差 |
|---|---|---|---|---|
| KT | 0.15824 | **−18.01%** | 0.17100 | −11.40% |
| 10KQ | 0.27465 | **−19.69%** | 0.29326 | −14.25% |
| eta0 | 0.64190 | +2.10% | 0.69967 | +11.29% |

→ 6 个对照里 **2 个越过平台 15% 不可接受线**，仅 1 个在 3% 内。
**陷阱（写进结论）**：eta0 的 +2.10% 是 KT/10KQ 同向偏低（−18%/−20%）在比值里互相抵消的结果，不能当作"精度达标"的证据。

## 3. 参考值侧核对

- `benchmarks/anchors/sources/kp505_marine2023_iftikhar.pdf` sha256 = `161ef626…5bc7c` = 锚文件声明 `pdf_sha256`；
- p.10 Table 2 抽文第 21 行 `0.7 0.173 0.356 0.193 0.342 10.31% 4.01%` 与锚 `source_line` 逐字节相等；`eta0 = J·KT/(2π·KQ)` 残差 0.0；
- 但该锚为 **SECONDARY_REPRODUCTION**（原文未引 KRISO/NMRI 一次实验表），故锚只能作报告项，不得作精度宣称 → 这是 L5 被拒的硬理由。

## 4. 判级：L2（声明 2 / 机器 gate 复核 2）

| 级 | 判 | 依据 |
|---|---|---|
| L1 | 过 | 测试非空且通过（8/8），G7 制造解自验 1e-15 级 |
| **L2** | **过（本审级）** | 831 行真代码 + 入口 + 非空运行工件 + 审计复跑逐位一致 |
| L3 | 拒 | 生态仓 `grep -rl kp505_three_density_gci /root/tensor-ecosystem/TensorBEM` = **0 命中**；main 无该驱动、无该测试、无 `benchmarks/verified/bem_kp505_gci/`；驱动只是在既有求解器上加算例壳，未扩展求解器 |
| L4 | 拒 | 3% 判据未达（8.47–11.25%，p<0 反收敛）；锚值 −18.0%/−19.7% 越 15% 线；平台机器 L4 裁定 `CLAIMED_NOT_PROVEN` |
| L5 | 拒 | 参考值仅二手转录 |

机器复核：`maturity_gate.judge(2, evidence, 'TensorBEM', 17266) -> (2, [])`；`judge(3)/judge(4)` 均降级到 2，理由 `integration_callers`。

**关于"是否虚标"**：交付自报 honest-FAIL，`failure_reason` 与 criteria 表一致，数字与工件逐项可复算 → **无虚标**。判 L2 不是罚造假，而是功能实质只到 L2 且基准未通过（附百页报告不改变级别）。

## 5. 严重度分级发现

| # | 严重度 | 发现 | 证据/复现 |
|---|---|---|---|
| F1 | medium（平台账） | `tensor_benchmark_cases` 无 `CASE-BEM-KP505-GCI` 行，`tensor_case_pool` id=164 仍 `dispatched`、`linked_anchor` NULL → 机器裁判永久缺席，这类算例永远只能"自证" | 见 §1 表格两条 SQL/API 结果 |
| F2 | medium（物理结论） | 展向面元数**不是**该 BEM 的有效收敛旋钮：n_radial 8→32 扫描 KT 相对极差 19.32% 且非单调（8:0.169749, 12:0.165844, 16:0.170180, 20:0.155237, 24:0.139722, 32:0.145370）；展向隔离阶梯 p=−5.845 | `case.json.axis_isolation_spanwise` + `fail_mode_localisation`；报告 §5 |
| F3 | low（结论表述） | eta0 在细网上 +2.10% 与 KT/10KQ −18/−20% 并存，若单看 eta0 会误判"精度达标" | §2 表；审计在 verdict 中显式锁定该陷阱 |
| F4 | low（工程卫生） | 交付悬在 `agent/m17245` 未并入 main；`case.json.env.git_dirty=true` 且 `git_commit=0516776`（= 交付提交的父提交），工件与提交身份未绑定 | `git log --oneline -1 origin/main`；`case.json.env` |
| F5 | info（方法学加分） | 报告主动指出 Roache 渐近比在单一三元组上代数恒等于 f_fine/f_medium、不足以检验 p，并据此把 G6 门控在"单调且 p>0"——这一点是对的，且被本审计认可 | `case.json.method_notes[0]` |

## 6. 未做/做不到的检查（如实声明）

- KRISO/KCS-workshop 一次实验原表：not_run（沙箱不可达）→ 锚仍是二手。
- 第四密度级（再 r=2）：not_run（1024 束缚面元 × 192 尾涡步，直接稠密解，D3 已 6.4 GB/52 s，本容器不可承载）→ 渐近区只能"准入"不能"验收"。
- 叶片几何重建（拱度/径向螺距 vs KP505 IGES）：not_run（只读交付自述的几何模型）。
- 集群复跑：not_run（单次 167 s < 10 min 集群阈值，无需递交）。

## 7. 处置（disposition）

**approve with notes at L2**（gate 应判 confirmed，消账但不升格）+ 三项返修建议：

1. 平台侧登记 `tensor_benchmark_cases` 行（`anchor_slug=CASE-BEM-KP505-GCI`，metric=GCI_fine/KT/10KQ/eta0，tolerance=`<3%`）并把 `tensor_case_pool` id=164 置终态——否则 F1 复发；
2. 合并或明确废弃 `agent/m17245`，并把 `case.json.env` 的 commit 与工件绑定（消除 F4）；
3. 提高精度的方向是**模型形式**（尖部 √(1−r) 载荷退化、拱度/径向螺距分布、尾涡模型），不是继续加密展向面元（F2 已证展向加密不收敛）；锚值升格前须拿到 KRISO 原表。