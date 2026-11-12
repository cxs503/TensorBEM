# FMMOperator 未 setup 静默错答（fail-open）→ fail-closed 契约（m16109）

**仓库**：TensorBEM · **分支**：agent/m16109 · **基线**：823e989 (main @ 2026-11-12)
**锚点**：NEW（总账无 FMM 锚点；本单收编 m16000 报告「局限 #2」登记的后续项）

## ① 案例介绍（物理背景/几何/参数）

`tensorbem.fmm.FMMOperator` 是 3-D Laplace 单层算子 `S q = Σ_j G(x_i,y_j) q_j`
（`G = 1/(4πr)`）的 FMM 加速算子：octree 叶内近场直和 + 叶间远场实球谐多极展开。
公开入口两个：`matvec(targets, sources, charges)` 与诊断方法 `far_fraction(targets)`
（多极对占比）。

复现几何（与 m16000 线性化基准同一套，seed=42）：

| 参数 | 值 |
|---|---|
| 源点数 | 200（`N(0, 0.5²)` 三维簇） |
| 靶点 | `[4,0,0]`, `[8,0,0]`, `[0,6,0]` |
| 展开阶 p | 4（诊断）/ 1..8（m16000 A/B 复核） |
| max_leaf | 25 |

## ② 怎么做的（方法/流程/关键设置）

1. **复现**（`outputs/m16109_repro.py`，修复前运行）：对**从未 `setup()`** 的算子
   调 `far_fraction` 与 `matvec`，记录静默返回值；对照已 setup 算子。
2. **根因**：`FMMOperator` 是 dataclass，未 setup 时 `leaves == []`；
   `matvec` 的早退条件 `if n == 0 or not self.leaves: return result` 把「未 setup」
   与「空靶点」混为一谈 → 返回全零；`far_fraction` 的 `if ... or not self.leaves:
   return 0.0` 同理 → 返回 0.0。m16000 的 bench（`scripts/fmm_linearity_bench.py:74`）
   恰好在未 setup 的新算子上调 `far_fraction`，得到 0.0，与「远场缓存被使用」
   （2.46 误差）矛盾——矛盾的解释就是这条静默路径。
3. **修复**（`src/tensorbem/fmm.py`，最小改动）：新增 `_require_setup(method)`，
   `tree is None or not leaves` 时抛 `RuntimeError`（消息含方法名）；`matvec` 与
   `far_fraction` 入口调用；空靶点的合法早退保留（`n == 0` 单独判断）。
   类 docstring 声明 setup 前置条件。
4. **bench 误用修正**：`scripts/fmm_linearity_bench.py` 的 `far_fraction` 调用
   先 `setup(sources)`。
5. **回归测试**：`tests/test_fmm_setup_contract.py`（9 例：4 例未 setup 必抛 +
   4 例已 setup 行为不变 + 1 例错误消息点名方法）。
6. **证据**：`evidence.json` 全部数字由 `outputs/m16109_evidence.py` 程序化注入。

## ③ 计算结果（数字表，程序化注入）

| 量 | 修复前（未 setup） | 修复后（未 setup） | 修复后（已 setup，对照） |
|---|---|---|---|
| `far_fraction` | **0.0**（静默，误导） | `RuntimeError: ...far_fraction() called before setup()` | 0.173913 |
| `matvec` 相对误差（vs 直和） | **1.0**（全零向量，静默 100% 错） | `RuntimeError: ...matvec() called before setup()` | 1.333308e-06 |
| `matvec` max\|·\| | 0.0 | — | 0.128168（= direct_scale） |

m16000 A/B 复核（`error_versus_order()`，修复后重跑，与 m16000 evidence 一致）：
prefix（旧缓存语义）second_err 恒 2.4639（p=1..8）；fixed second_err 随 p 降至
2.94e-10（p=8）——本单改动不影响 m16000 的修复语义。

`far_fraction` 诊断曲线（已 setup，靶点沿 +x 从 d=0.5 到 10）：
d=0.5 → 0.0（全近场）；d=4 → 0.2609（单靶）；d≥~6 → 1.0（全远场）。
诊断现在报告真实占比，不再是恒 0。

## ④ 结果图

- `fig_far_fraction_vs_distance.png`：修复后 `far_fraction` 随靶距单调上升
  （0→1），红点=m16000 bench 靶点（d=4，0.261）。
- `fig_failopen_vs_failclosed.png`：左=matvec 相对误差（log 轴），右=far_fraction；
  斜纹柱=修复后该静默值不再出现（改抛 RuntimeError）。

## ⑤ 注意点（坑/局限/适用范围）

1. **行为变更（有意）**：未 setup 调 `matvec`/`far_fraction` 从静默错答变为
   `RuntimeError`。仓内全部既有调用点（`fmm_cost_audit.py` 5 处、`test_fmm.py`
   7 处、`test_fmm_cost_audit.py` 1 处、`test_fmm_operator_linearity.py` 5 处）
   均先 `setup` 后使用，已逐一核对，无依赖静默零答的代码。
2. 本单只改 `FMMOperator` 的两个公开入口；`fmm_laplace_matvec`（无状态入口）
   内部自建算子并 setup，不受影响。
3. 局限：`_require_setup` 判据是 `tree is None or not leaves`；若未来出现
   「合法的空 octree」语义需再议（当前 `build_octree` 对非空点集必产生叶）。
4. 容量纪律：本机 load15≈100（≥100），本单全程轻量（200 源点级计算），未起重算力。

## ⑥ 误差判定（逐项 vs 基准）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 已 setup `matvec` vs 直和（p=4） | `direct_laplace_matvec` | 1.33e-06 | ✅ 是 |
| 已 setup `matvec` vs 直和（p=6，m16000 测试） | 同上 | <1e-6（测试断言） | ✅ 是 |
| `far_fraction`（d=4 单靶） | 本单诊断自洽（近/远对计数） | 与 `matvec` 近/远掩码同源，无独立基准 | 声明：诊断量，无外部基准；自洽性=与 `_near_mask` 同一谓词 |
| 修复前后首次调用数值 | m16000 evidence（p=1..8 first_err） | 逐位不变（A/B 行一致） | ✅ 是（0 偏差） |

**honest 声明**：本单无外部文献基准（属 API 契约缺陷修复，非物理量预报）；
物理精度锚点沿用 m16000 已验证的直和对照，未引入新物理声明。

## 测试

- 新文件红→绿：修复前 `4 failed, 4 passed`（git stash 回退 fmm.py 验证）；
  修复后 `tests/test_fmm_setup_contract.py + test_fmm_operator_linearity.py +
  test_fmm.py + test_fmm_cost_audit.py` = **57 passed**。

## 工件

- `docs/fmm_setup_contract/evidence.json`（机读证据）
- `docs/fmm_setup_contract/fig_far_fraction_vs_distance.png`
- `docs/fmm_setup_contract/fig_failopen_vs_failclosed.png`
- `outputs/m16109_repro.py` / `m16109_repro_prefix.json` / `m16109_repro_postfix.json`
- `outputs/m16109_evidence.py`（证据生成脚本）
