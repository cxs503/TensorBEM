# 交差审计 m17177 · CASE-BEM-ELLIPSOID-ADDED-MASS（TensorBEM，椭球三轴附加质量）

审计单 m17233 · 审计者席位 board-audit · 基准裁判 = `tensor_benchmark_cases` + 独立重算参考值
审计工作树 `/root/tp-agent/work/m17233`（TensorBEM @ 0516776 "Merge #761 (m17158) via merge-master"）
被审交付 m17177（succeeded，报告 ID 16175，head_commit=null，changed_files 全为 `??` 未跟踪文件）

## 0. 结论（disposition）

| 项 | 结论 |
|---|---|
| 裁定级别 | **L4 基准验证**（完整算例 + 误差分析 + 可复现 + 独立参考核对） |
| 声明级别（审计者） | L4，非 L5（理由见 §5） |
| 功能性缺陷 | 无。三轴附加质量数值真实、非硬编码、可复现，误差 1.63/1.74/1.91% < 3% < 5%（判据） |
| 阻塞性缺陷 | 无（BEM 数字不因文档缺陷失效） |
| 非阻塞缺陷 | ①报告③参数表 h_max/h_mean 三行全错；②§2 声称使用 CHIEF 点，源码无 CHIEF；③"零手抄"自证与事实不符（ref 由仓库自身 analytic 模块算出）；④交付工件未入库、repro_cmd 指向可回收工作树；⑤§6"生态摩擦：无"低估了④ |
| 系统缺陷 | `tensor_benchmark_cases` 对该 anchor_slug 有 **0 行**（机器裁判缺失）；本次按独立重算参考值补登一行 |

## 1. 取证：三件套核验（逐项）

### 1.1 run_artifacts（真实工件）

被审工作树 `/root/tp-agent/work/m17177` 内实存：

| 工件 | 大小 | 说明 |
|---|---|---|
| `artifacts/ellipsoid_added_mass_sub2.json` | 4697 B | sub=2（320 面板）完整数值结果 |
| `artifacts/ellipsoid_sub1.json` | 4678 B | sub=1（80 面板） |
| `artifacts/ellipsoid_sub0.json` | 4668 B | sub=0（20 面板） |
| `artifacts/figures/{ellipsoid_geometry,added_mass_comparison,convergence_study,shape_factors}.png` | 384/73/157/57 KB | 四张报告图 |

→ 工件**存在且非空**。但 m17177 `result.git_status` 显示其全部为未跟踪（`??`）、`head_commit=null`：**没有进入仓库**，只活在可回收工作树里（`worktree_recyclable`）。审计已把它们原样复制到本分支 `docs/board_audit/m17233/refs/` 固化。

### 1.2 benchmark ref vs got（数值对照）

`tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-ELLIPSOID-ADDED-MASS'` → **返回 0 行**（机器裁判不存在；全表仅 20 余行，其中 BEM 域仅 1 行 `CASE-BEM-MACCAMY-FUCHS-DIFF`）。故按任务判据（m17177 goal：容差 <5%）+ 仓库 pin（`docs/benchmark_pins.json` `ellipsoid_added_mass_320`，rtol 1e-6）作裁判，并**独立重算参考值**：

| 轴 | got（320 面板，BEM） | ref（审计独立重算，Lamb §114 积分，mpmath 40 位自适应） | 误差 | <3% | <5% |
|---|---|---|---|---|---|
| x (surge) | 1191.2955950090695 kg | 1211.0519952419288 kg | **1.6313 %** | ✓ | ✓ |
| y (sway) | 2394.8177826981614 kg | 2437.1535015128825 kg | **1.7371 %** | ✓ | ✓ |
| z (heave) | 4104.549811866565 kg | 4184.389189761621 kg | **1.9080 %** | ✓ | ✓ |

参考值的三重独立核对（脚本 `docs/board_audit/m17233/independent_ref_check.py`，输出 `rerun/independent_reference.json`）：

1. **定义式**（Lamb 1932 §114）：α_i = a₁a₂a₃∫₀^∞ dλ/[(a_i²+λ)√∏(a_j²+λ)]，A_ii = ρVα_i/(2−α_i)。mpmath 自适应求积（λ=t/(1−t) 替换）得 α=(0.400927574554924, 0.670712917798092, 0.928359507646984)，**α₁+α₂+α₃=2 恒等式残差 2.2e-18**。
2. **已发表闭式**（长椭球 a>b=c）：α_axial=2(1−e²)/e³[atanh(e)−e]（e=√(1−b²/a²)）。在 (2,1,1) 上与同一积分吻合到 **1.3e-18**（轴向 0.347127995067928、横向 0.826436002466036）。
3. **球极限**：α_i=2/3 精确到 **6.7e-19**，A=0.5ρV 精确成立。
4. 与仓库 `tensorbem.analytic` 交叉核对：α 最大相对差 **1.1e-12**，ref 附加质量最大相对差 **2.0e-12** → 仓库参考实现本身正确（"ref 由仓库自己算"这一自指风险已被外部重算消解）。

### 1.3 repro_cmd（可执行性）

报告给出的命令把工作树写死为 `/root/tp-agent/work/m17177`（可回收 → 会失效）。仓库内等价命令经审计实跑：

```
PYTHONPATH=src python -m tensorbem.benchmarks.ellipsoid_added_mass \
    --subdivisions 2 --semi-axes 1.5 1.0 0.75 --density 1025.0 --output <out>.json
```

CLI 真实存在（`src/tensorbem/benchmarks/ellipsoid_added_mass.py:180-206`，argparse：`--subdivisions/--semi-axes/--speed/--density/--output`）。审计从分支 HEAD 重跑 sub=0/1/2 三档，JSON **与 m17177 工件逐键完全相同**（sha256(排序 JSON) 217a000414edc64d / 8e5a8faccc2a2820 / 1c4ff31dcd367329）。→ 命令可执行、结果可复现，且复现不依赖被审工作树。

## 2. 数值复核（含收敛性）

| sub | 面板 | got x / y / z (kg) | 相对误差 x / y / z |
|---|---|---|---|
| 0 | 20 | 958.485995 / 1922.280060 / 3241.204877 | 20.8551 % / 21.1260 % / 22.5405 % |
| 1 | 80 | 1136.423798 / 2279.400297 / 3890.642146 | 6.1623 % / 6.4728 % / 7.0201 % |
| 2 | 320 | 1191.295595 / 2394.817783 / 4104.549812 | **1.6313 % / 1.7371 % / 1.9080 %** |

误差随细分的下降比（sub1→sub2）：3.78× / 3.73× / 3.68×，面板数 ×4 → 观测阶 ≈ 0.94–0.96（近一阶，与 flat-panel 只在体积/迹上二阶、通量近一阶的理论一致；与仓库 `docs/marine_bem_benchmarks.md` 声明的"通量近一阶、附加质量二阶"一致）。体积比 0.6055 → 0.8735 → 0.9662 单调趋近 1。

**负对照（非硬编码证据）**：density 1025→2050，三轴附加质量精确 ×2.000000000000；改几何为球 (1,1,1) 时 α=2/3、exact=0.5ρV、BEM 值随网格收敛（sub=1 时 0.9352·ρV）。输出随输入变化，非查表/硬编码。

## 3. 报告六段核验（平台报告 ID 16175，qa_status=pending）

| 段 | 是否齐 | 核验结果 |
|---|---|---|
| ①问题+ref 出处 | 齐 | 引用 Lamb (1932) §114 pp.152-156 + Newman (1977) §4.10；但**"零手抄声明"自证与任务要求相悖**：ref 数值显式声明由 `tensorbem.analytic` 算出（自指），不是"从原文/手册查得原表原页" |
| ②方法与实现 | 齐 | 基本准确（单层+双层势、外域 Dirichlet、边界能量积分 A=(ρ/U²)∮φ∂φ/∂n dS）；**但声称"CHIEF 点：自动 interior points（抑制虚假特征值）"**——`src/tensorbem/solver3d.py:128-171 solve_laplace_surface` 无任何 CHIEF/interior point 逻辑，属无据陈述；求解器模块名应为 `solver3d` 而非 `solver` |
| ③参数表 | 齐 | 无量纲参数（Re=∞、Fr=0、a/b=1.5、b/c=1.333）正确；**但网格表 h_max/h_mean 三行全部与实跑不符**（见 §4 缺陷 D1） |
| ④对照表 ref vs got + 误差打勾 | 齐 | 数值与实跑一致（1.631/1.737/1.908 %，通量 L2 2.729/2.842/3.348 %，κ(S)=44.67） |
| ⑤收敛性/敏感性（≥2 点） | 齐 | 网格收敛 + 形状因子积分节点敏感性 + 拓扑敏感性；其中"32 节点时残差增至 ~1e-6"**无工件背书**（未列入工件清单） |
| ⑥失败模式与工件 | 齐 | 局限（faceting、无自由面、无粘、O(N³)）如实；**"生态摩擦：无"低估事实**——交付未入库、repro_cmd 指死路径即为摩擦 |

## 4. 缺陷清单（按严重度）

- **D1（中，报告事实错误，已复核）**：③参数表网格列错。实跑（本审计 sub=0/1/2）：h_max = **1.577193 / 0.927051 / 0.487380 m**，h_mean = **1.167340 / 0.646594 / 0.332389 m**；报告写 0.9747/0.4874/0.2437 与 0.7123/0.3562/0.1781。仅体积比列（0.6055/0.8735/0.9662）正确。仓库 pin（`docs/benchmark_pins.json` 320 行 h_max=0.487379…）与实跑一致，故错在报告表、不在代码。判：不改级别（不改功能实质），但**报告不得作为数字来源引用**。
- **D2（中，无据陈述）**：§2 CHIEF 点（源码无）；§2 求解器模块名（solver → solver3d）。
- **D3（中，证据不可持续）**：交付工件全部未跟踪、head_commit=null、repro_cmd 写死可回收工作树路径。审计已重跑并固化副本。
- **D4（低，未背书断言）**：§5 "32 节点求积残差 ~1e-6" 无工件。
- **D5（系统，跨案例）**：`tensor_benchmark_cases` 对本 anchor 0 行 → 任务书要求的"唯一裁判"缺失。审计按独立重算参考值补登（本次同时给出 got 与实测误差），使机器裁判从此可得。
- **D6（范围声明）**：本算例 Dirichlet 迹取解析线性迹（`u=-U·k·x_i`，k=α_i/(2−α_i)），BEM 只承担通量与能量积分 → 验证的是"离散化 + 外域积分方程"，不是完整外域解；算例无自由面/无粘/非船体。故 L4 上限由范围决定，与 L5 无关。

## 5. 为什么是 L4 而不是 L5

- L4（基准验证）= 完整算例 + 误差分析 + 可复现 +（本案）独立参考重算 → **全部满足**。
- L5（科研工程成果）要求 L4 + 外部锚（文献/实验的**具体表值/页值**或前后对照的问题记录）+ 科研实质增量。本案：①参考值是**公式现场求值**（虽然审计用独立实现验证到 1.1e-12），未从 Lamb 原表原页转录，正是 m17177 goal 明文要求却未做到的一项；②该能力与其 pin（320/1280 面板）早于本单入库，本单是"复跑+报告"，无新能力/新算法/新物理；③算例本身剔除自由面与粘性。故按"级别=功能实质"定 **L4**。

## 6. 复现（审计自用命令，一键）

```bash
cd <TensorBEM worktree>            # 本次：/root/tp-agent/work/m17233 @ 0516776
bash docs/board_audit/m17233/repro_cmd.sh
```

脚本内含：三档 BEM 重跑 → 独立参考重算（mpmath）→ 与 pin/被审工件比对 → 打印结论。

注：整脚本耗时约 2–3 min（sub=2 单档约 60 s + mpmath 参考重算），超过本地 shell 120 s 上限；审计按阶段分别验证（sub=0/1 经脚本重写、sub=2 与独立参考此前各自实跑通过），未把整脚本一次跑完，如实记此限。

## 7. 工件与出处

| 工件 | 路径 |
|---|---|
| 审计重跑（sub=0/1/2） | `docs/board_audit/m17233/rerun/ellipsoid_added_mass_sub{0,1,2}_rerun.json` |
| 独立参考重算 | `docs/board_audit/m17233/rerun/independent_reference.json`、脚本 `independent_ref_check.py` |
| 负对照（密度×2 / 球极限） | `docs/board_audit/m17233/rerun/negctl_{density,sphere}.json` |
| 被审工件事后固化 | `docs/board_audit/m17233/refs/`（3 JSON + 4 PNG + 报告 md + 原复现脚本） |
| 本审计报告 | `docs/board_audit/m17233/AUDIT_m17177.md` |
| 平台报告 | tensor_reports（scenario_key=CASE-BEM-ELLIPSOID-ADDED-MASS，mission 17233） |

## 8. 未做的检查（honest）

- 未在 x86 GitHub CI 上跑 `tests/test_ellipsoid_1280_verification.py`（1280 面板 ×3 轴，单机 >120 s；本机 shell 超时 120 s 截断；已改投集群 cj1464，排队 592 未回，结果未计）。已跑：`tests/test_analytic_references.py -k ellipsoid` → 2 passed。
- 未取 Lamb (1932) 原书扫描页做 OCR 逐字比对（无该版权文本可用）；外部锚由公式定义 + 已发表闭式 + 球极限三重独立核对代替，如实标注为"公式级外部锚，非表值级"。