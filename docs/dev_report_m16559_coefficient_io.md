# m16559 · GAP-BEM-COEFF-IO：BEM 系数互通层（WAMIT .1/.3 + Nemoh tec + 原生容器）

**仓库** TensorBEM · **分支** `agent/m16559` · **基线** `69b5ae7`（main）
**锚点** `NEW: GAP-BEM-COEFF-IO`（由 m16421 `FEAT-OSS-INVENTORY-BEM` 波1 提议、`docs/oss_inventory_bem/registry.json` 记 `gap_status=verified_absent`（探针 5 词 0 命中）；feature_ledger 无此条目 → 本单按 NEW 处置，finish 给出建议入账行）

## 1 案例介绍（为什么做这件事）

不是流场算例，而是**数据通路**算例。TensorBEM 的波载结果（附加质量 `A_ij`、阻尼 `B_ij`、波浪激励力 `F_i`）**
只能被写成各自的 benchmark JSON，没有任何读回路径**：于是全部对标参考值都是手抄常量
（`benchmarks/wigley3_complete_verification.py:_WAMIT_A33_MASS_RATIO = 0.42`），
既违反零手抄纪律，又让第三方参考值无法跨算例复用。`docs/oss_inventory_bem` 的缺口探针
把这条缺口判为 P0 / `verified_absent`。

对象：一个容器 + 四种方言：

| 方言 | 内容 | 谁在写 |
|---|---|---|
| 原生 `.json` | A/B/F + 环境 + provenance + payload SHA-256 | 本库 |
| WAMIT `.1` | A/B（非量纲 `A/(rho L^k)`、`B/(omega rho L^k)`，k=3/4/5） | capytaine / BEMRosetta 链 |
| WAMIT `.3` | F 的 `|F|`/相位（非量纲 `F/(rho g A L^m)`，m=2/3） | 同上 |
| Nemoh `.tec` | `RadiationCoefficients.tec` / `ExcitationForce.tec` | Nemoh / BEMRosetta |

## 2 怎么做的（方法/工具链）

1. `src/tensorbem/coefficients.py`（新增，无求解逻辑）：
   `HydrodynamicCoefficients` 冻结容器（形状/有限性/频率严格递增/航向与激励一致性全部构造时校验）+
   四个 reader/writer 对 + `from_radiation_3d()` / `stack_frequencies()` 从既有求解器结果直连。
2. **完整性**：`fingerprint()` 只哈希数字载荷（`json` 的 float `repr` 是跨平台确定的最短精确表示），
   `write_json` 落盘时盖章，`read_json` 默认复核，篡改即 `CoefficientIntegrityError` 失败关闭。
3. **互操作判据不靠自证**：`.1/.3` 由**独立实现** capytaine 3.1 的 `io.wamit.export_wamit_1/3` 写文件，
   本模块读；并用 capytaine 的 `DOF_INDEX/DOF_TYPE/K_LOOKUP` 校准 k 指数、用其 exporter 校准
   「第2列=radiating、第3列=influenced」的列序（该列序与直觉相反，靠往返+对拍确认，非猜测）。
4. **端到端**：真 3-D BEM 辐射求解（潜没球 `submerged_sphere_mesh(subdivisions=1)`，
   `solve_radiation_3d` modes=(surge, heave)，4 个 ω）→ 容器 → 每种方言往返 → 重新导出无量纲 A33/ρ。
5. 证据脚本 `scripts/coefficients_io_evidence.py` 一次性产出
   `docs/coefficient_io/{evidence.json,README.md,fig_roundtrip.png}`，README 表格由图脚本从 JSON 注入，零手抄。

## 3 计算结果（数字表，来自 `docs/coefficient_io/evidence.json`）

| 检查 | 实测最大相对误差 | 判据 | 判 |
|---|---:|---|---|
| 原生 JSON：A / B / F | **0.0 / 0.0 / 0.0** | < 1e-12 | PASS |
| WAMIT `.1`：A、B（A 阵故意非对称，禁止静默对称化） | **7.5e-10** | < 1e-6 | PASS |
| WAMIT `.3`：F | 7.5e-10 | < 1e-6 | PASS |
| Nemoh `.tec`：A、B、F | 7.5e-10 | < 1e-6 | PASS |
| JSON 被改动 1e-7 相对后被检出 | 篡改检出=是 | 必须检出 | PASS |
| capytaine 写 `.1` → 本模块读（A/B/ω） | **5.9e-7** | < 1e-6 | PASS |
| capytaine 写 `.3` → 本模块读（模数） | 5.9e-7 | < 1e-6 | PASS |

求解器算例（潜没球 heave，`n_panels` = 80，条件数与残差见 `docs/coefficient_io/evidence.json`）：

| ω [rad/s] | A33/ρ [m³] | B33 [kg/s] |
|---:|---:|---:|
| 1.9806 | 0.030490 | 0.227619 |
| 2.8009 | 0.030496 | 1.157464 |
| 3.4304 | 0.030292 | 2.140066 |
| 3.9611 | 0.030041 | 2.616635 |

A33 经每种方言往返后的传播误差（同一 evidence 文件）：原生 0.0e+00、`.1` 1.3e-13、`.tec` 6.0e-11；三方言往返最大相对误差 7.5e-10（判据 1e-6），原生 JSON 相对误差 0.0e+00（判据 1e-12）；capytaine 互操作最差 5.9e-07。

## 4 结果图

`docs/coefficient_io/fig_roundtrip.png`：左=各容器方言的往返最大相对误差（对数轴，标 1e-12/1e-6 判据线）；
右=真解算例的 A33/ρ 曲线 + 三方言传播误差。图由脚本生成，无手绘数字。

## 5 注意点（坑/局限/适用范围）

1. **列序反直觉**：WAMIT `.1` 行内两列 DOI 索引是 (radiating, influenced)，不是 (i, j) 的位置序；
   首版按位置序实现，与 capytaine 对拍立刻暴露整矩阵**转置**（A[0,1]=2000 vs 3000），改为语义列序后归零。
   这条只在与外部实现对拍时才可能发现——自证往返会双双通过。
2. **频率网格精度**是方言固有的：`.1` 存周期，7 位有效数字 ⇒ ω 只能回到 ~4e-7 相对；振幅量本身可到 1e-9。
3. **`.3` 相位列 7 位有效数字**（度）⇒ 复数值回读约 1e-5 相对，模数仍 1e-6；WAMIT 的 `e^{+iωt}` 约定
   用显式 `conjugate=True` 开关吸收，不做隐式默认（否则静默把相位翻号）。
4. **未实现**：WAMIT `.1` 的**块格式**（6×6 行块）——仓内无权威样本，写了也无法验证，明确不写。
5. **未验证**：BEMRosetta 打开本模块输出的文件（本机离线无 BEMRosetta）。格式一致性只对 capytaine 成立，
   报告不声称 BEMRosetta 兼容已核验。
6. **锚点验收只完成一半**：把 Wigley 驱动的硬编码基准改走文件注入、并证明验证误差不劣化 <1%，
   需要跑 Wigley 3-D 扫频与改造既有 driver，本单未做（诚实未完成）。
7. 容量纪律：全程轻量（单算例 4 频点、subdivisions=1 的潜没球），未起重算力进程。

## 6 误差判定（逐项，显式回答）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---:|---|
| 原生 JSON 往返 A/B/F | 原值 | 0.0 | 是（且 0） |
| `.1/.3/.tec` 往返 A/B/F | 原值 | 7.5e-10 | 是 |
| capytaine 文件解析 A/B/ω、`|F|` | capytaine exporter 源值 | 5.9e-7 | 是 |
| capytaine `.3` 复数值 | 源复数值 | ~8e-6（相位列位数所限） | 是 |
| A33/ρ 经方言传播 | 求解器原值 | ≤7.5e-10 | 是 |

**结论**：判据 <3% 大幅满足；其中「是否 <1e-6（方言）」与「是否 <1e-12（原生）」两档也逐项满足。
未达成的部分（BEMRosetta 核验、Wigley 文件注入改造）如实列为未完成，不计入 PASS。

## 7 测试

`tests/test_coefficients_io.py` 32 例全绿（含 capytaine 互操作 2 例、篡改检出、校验拒绝 9 例、
求解器直连 3 例）：

```
PYTHONPATH=src python3 -m pytest tests/test_coefficients_io.py -q   # 32 passed
```