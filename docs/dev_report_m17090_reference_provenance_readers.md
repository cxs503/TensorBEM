# 回忆常数的读图 + 判据回读：把"这条债务没人读"从散文变成闸门（m17090 / TensorBEM）

**仓** TensorBEM · **分支** `agent/m17090` · **类型** 能力扩展（`reference_provenance` 闸门第二维度）
**工件** `src/tensorbem/reference_provenance.py`、`tests/test_reference_provenance.py`、
`docs/reference_provenance.md`（块重生）、`docs/audit_baseline.md`（能力行重生）、
`docs/fig_m17090_reference_provenance_readers.png`、`make_report_figure_m17090.py`
**复现** `PYTHONPATH=src python -m pytest tests/test_reference_provenance.py -q`；
`PYTHONPATH=src python -m tensorbem.reference_provenance`（巡查，非零退出=有债）

## 1. 案例介绍

### 1.1 背景：百分比的价值等于分母的价值

TensorBEM 的验证结论全是误差百分比（某附加质量 vs WAMIT、某 RAO 峰 vs Journee 1992）。
m14725 建了 `tensorbem.reference_provenance`：扫描 `src/tensorbem/**` 里名字带文献标记的**模块级**
数字常量，要求全部登记分档——`CITED`（本仓交付文档发表了这些数字，登记 `(路径, 数字针)` 证据对，
每次审计重读文件）或 `UNSOURCED`（回忆值，必须写 note，公开为债务，永不算证据）。

闸门当时管了两件事：**新出现未登记的常数**红灯、**引文腐烂**红灯。它**不**管第三件事：**谁在读这条债务**。
于是债务表的每一行都靠散文承诺"Ungraded: no test and no capability row reads this constant"——
而这句承诺在 2026-11-12 之后已经不成立了：`tensorbem.benchmarks.wigley3_recalled_ref`（m16892）
把 5 个回忆常数 import 进来做判据。散文漂移、机器不知道，正是本单要堵的洞。

### 1.2 本单被判据的物理案例（Wigley III 频域水动力）

判据 artefact 的对象是标准 Wigley III 方尾数学船型（本仓 `tensorbem.wigley3_hull`，几何先经
Cb/排水体积验证）：

| 量 | 值 |
|---|---|
| 垂线间长 L | 3.0 m |
| 型宽 B | 0.3 m |
| 吃水 T | 0.1875 m |
| 丰满度参数 alpha | 1.1087500000000008 |
| 解析排水体积 | 0.09163125 m³（artefact metadata 记 0.09163125 m³，两者一致） |
| 方形系数 Cb | 0.543 |
| 记录频率 | omega = 1.0 rad/s |
| 判据带宽 BAND | ±0.1（即 ±10%） |
| 独立第二码 | capytaine（冻结件 `capytaine_ref.json`，机器 loongarch64） |

判据把每个回忆常数放到两条独立路径（TensorBEM 两种网格构造 + capytaine 冻结件，各自三种面板数）算出的
同量纲值上，输出 `REPRODUCED` / `FREQUENCY_MISMATCH` / `REFUTED` 逐自由度判定。

## 2. 怎么做的

### 2.1 读图（`readers`）：谁读了这条债务

* AST 遍历 `src/`、`tests/`、`scripts/` 三个根（`docs/` 故意排除：叙述文本提到名字不是消费者），
  只认两种读法：`from ... import NAME` 与属性访问 `something.NAME`。
* **定义模块本身不算读者**；**自己赋值同名常量的模块也不算**（避免把重复抄写误判成消费）。
  匹配按名字、故意粗粒度：假阳是一次对话，假阴是回忆数字悄悄进了结论。
* 结果按登记 key 归集为 `Reader(consumer, line, how)`，进入 `Audit.readers`。

### 2.2 允许清单（`ADJUDICATED`）：回忆常数只许被它的判定器读

`ADJUDICATED: dict[key, consumer_module]` 是跨模块读者的**唯一**白名单。三个方向全部 fail-closed：

| 方向 | `Audit` 字段 | 含义 |
|---|---|---|
| 出现未登记读者 | `leaked` | 一条回忆常数被白名单外的模块读了 → 回忆数字可通向结论 |
| 判定器不再读它 | `unread_adjudicator` | 白名单腐烂（登记了一个不存在的读法） |
| 判定器不给判据 | `unadjudicated` | `ADJUDICATION_ARTEFACT` 里没有这条常数的判定 |

三者任一非空 → `Audit.ok is False` → 测试红 / CLI 非零退出。

### 2.3 判据回读（`adjudication`）：债务表说出被测量过什么

`adjudication()` 读 `benchmarks/verified/wigley3_recalled_ref/results.json`，按 `constant` 聚合 `(verdict, factor)`，
`Adjudication.describe()` 渲染成 `REFUTED 6/6 (computed/recalled 0.27-12.8x)` 直接进债务表。
数字**不落人手**：文档块由 `--write` 从 artefact 生成，`tests/test_reference_provenance.py::test_published_block_matches_the_tree`
保证手改即红。

### 2.4 文档与能力台账

* `docs/reference_provenance.md` 的生成块新增两列（**read by** / **adjudicated**）与一段说明，
  fail-closed 行补上 `leaked` 与 `unadjudicated` 计数。
* `capability_evidence.py` 既有能力行「Machine-checked provenance of reference constants」扩写 claim、
  增补 `readers` / `adjudication` 符号与两条新测试，`docs/audit_baseline.md` 块用
  `python -m tensorbem.capability_evidence --write` 重生（**热文件改动已在报告声明**）。

## 3. 计算结果（全部来自程序读取，零手抄）

### 3.1 读图与闸门现状（`rp.audit()` 本地实测）

| 量 | 值 |
|---|---|
| 扫到的文献常数 | 15 |
| CITED（引文可解析） | 1 |
| UNSOURCED（债务） | 14 |
| 跨模块读者边 | 7 |
| 有判据的负债常数 | 5 |
| undeclared / stale / unverified | 0 / 0 / 0 |
| leaked / unadjudicated | 0 / 0 |

跨模块读者全表：

| 常数 | 读者（文件:行 / 读法） |
|---|---|
| `tensorbem.benchmarks.wigley3_complete_verification._WAMIT_A33_MASS_RATIO` | `tensorbem.benchmarks.wigley3_recalled_ref`:235 (import) |
| `tensorbem.benchmarks.wigley3_freq_verification._WAMIT_ADDED_MASS_DIAGONAL` | `tensorbem.benchmarks.wigley3_recalled_ref`:236 (import) |
| `tensorbem.benchmarks.wigley3_freq_verification._WAMIT_DAMPING_DIAGONAL` | `tensorbem.benchmarks.wigley3_recalled_ref`:236 (import) |
| `tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_A33_REF` | `tensorbem.benchmarks.wigley3_recalled_ref`:237 (import) |
| `tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_B33_REF` | `tensorbem.benchmarks.wigley3_recalled_ref`:237 (import) |
| `tensorbem.qtf_drift.SEMISUB_PUBLISHED_DISPLACEMENT` | `tensorbem.benchmarks.qtf_drift_study`:32 (import), `tests/test_qtf_drift.py`:37 (import) |

唯一无读者的 9 条债务（`_JOURNEE_HEAVE_RAO_PEAK`, `_JOURNEE_HEAVE_RAO_PEAK_OMEGA`, `WAMIT_REF`, `JOURNEE1992_RAO`, `_JOURNEE_EXCITATION`, `_JOURNEE_RAO_PEAKS`, `WAMIT_F3_BETA0_REF`, `JOURNEE_HEAVE_RAO_REF`, `JOURNEE_PITCH_RAO_REF`）保持"零读者"，
它们因此不可能进入任何结论——这一条现在是**机器事实**而非散文。

### 3.2 判据回读（读 `benchmarks/verified/wigley3_recalled_ref/results.json`）

| 常数 | 定义模块 | 记录数 | 判定 | computed/recalled 幅度 |
|---|---|---|---|---|
| `_WAMIT_A33_MASS_RATIO` | `tensorbem.benchmarks` | 1 | FREQUENCY_MISMATCH 1/1 | 4.3 – 4.3 |
| `_WAMIT_ADDED_MASS_DIAGONAL` | `tensorbem.benchmarks` | 6 | FREQUENCY_MISMATCH 1/6, REFUTED 5/6 | 0.27 – 12.8 |
| `_WAMIT_DAMPING_DIAGONAL` | `tensorbem.benchmarks` | 6 | REFUTED 6/6 | 3.17e-06 – 1.91 |
| `WAMIT_A33_REF` | `tensorbem.benchmarks` | 1 | REFUTED 1/1 | 446 – 446 |
| `WAMIT_B33_REF` | `tensorbem.benchmarks` | 1 | REFUTED 1/1 | 120 – 120 |

### 3.3 负控与红灯方向（合成树，不依赖真实仓库状态）

`tests/test_reference_provenance.py` 新增的合成树用例把四种状态逐一走通：
未登记读者→`leaked`；登记判定器→干净（且判据被读回）；判定器停读→`unread_adjudicator`；
artefact 无该常数→`unadjudicated`。四种都验证 `Audit.ok` 的取值，证明闸门**会区分**而不只是"存在"。

## 4. 结果图

`docs/fig_m17090_reference_provenance_readers.png`（`make_report_figure_m17090.py` 生成，数据全部
从 `results.json` 与 `rp.audit()` 读取）：

* **左图**：15 条判据记录的 `computed/recalled` 因子（对数轴），灰带为判定器自己的 band = 0.10（±10%）容差；
  红点 `REFUTED`（13 条）、橙点 `FREQUENCY_MISMATCH`（2 条）。因子分布 3.17e-06 – 446×，
  **没有一条在标称频率落入 ±10% 带宽**（`|factor-1| <= 0.10` 的记录数 = 0）。
* **右图**：闸门测得的表面——扫描 15 条、CITED 1、债务 14、
  有判据 5、无判据 9、读者边 7、
  白名单外读者 0。

## 5. 注意点（踩过的坑 / 局限）

1. **散文会腐烂**：原债务表对 5 条常数写的是"no test and no capability row reads this constant"，
   m16892 的判定器引入后该句已成假话。本单把它改成指向判定器 + 机器可核的读者列；
   教训是"没人读"这类性质必须落到图上，不能写进 note。
2. **名字匹配是粗的**：属性访问按 attr 名字匹配，跨包同名常量会互相点亮（当前 0 例）。
   选择粗匹配是 fail-closed 取向，代价是可能要求一次人工确认。
3. **仍有 9 条债务没有判据**（其余 Journee 表与 F3 激励等）：本单只登记、不裁决。它们零读者，
   因此不会污染结论，但"回忆值对不对"未知，**不得**在任何报告里当参照值使用。
4. **副作用成本**：`audit()` 现在要解析 `src/`+`tests/`+`scripts/` 全部 py 文件，单次审计从 ~0.5 s 升到
   ~1.5 s；`tests/test_reference_provenance.py` 从 23 s 升到 ~100 s。没有加全局缓存——测试里会改写
   合成树，缓存会让负控变假绿。
5. **热点文件**：本单改了 `capability_evidence.py`（能力台账，多单共用热文件），
   改动仅限既有行 claim/符号/测试三元组扩写，未动结构与其它行；`docs/audit_baseline.md` 由 `--write` 重生。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 结果 | 是否 <3% |
|---|---|---|---|
| 债务分档完整性（15 条文献常数是否全登记） | 扫描结果 vs 登记表，机器全等 | undeclared = 0, stale = 0 | **是**（0/15，恒等比较，无浮点误差） |
| 白名单外读者 | 读图 vs `ADJUDICATED` | leaked = 0 | **是**（0 条） |
| 判定器/artefact 一致性 | 登记 key ⊆ 读图 ∧ key ∈ artefact | unadjudicated = 0 | **是**（0 条） |
| 回忆常数本身 vs 两条独立码 | capytaine 冻结件 + TensorBEM 两构造，三种面板数（`results.json` 15 条记录） | 见 3.2：REFUTED 13 条、FREQUENCY_MISMATCH 2 条，`|factor-1| <= 0.10` 的记录 **0** 条；因子 3.17e-06 – 446× | **否，且这正是结论**：回忆值与真值相差 3.2e5 倍（高估）到 446 倍（低估），远超 3%；登记表说它们不是证据，判据证明它们确实错 |
| 唯一 CITED 常数的几何一致性 | `docs/qtf_drift_oc4.md` 发表的 13473 m³ | 平滑几何积分 vs 发表值 +1.8%（见该文档） | **是**（1.8% < 3%） |

结论：本单的"被测量"是闸门自身的判别力（计数类，机器恒等，0 误差）与回忆常数的真值偏差
（因子 3.17e-06 – 446×，**远超 3%，但方向与登记表一致**：登记表说它们不是证据，判据证明它们确实是错的）。
被 CITED 的那条常数其几何一致性 +1.8% < 3%，可继续作为证据使用。

## 7. 验证清单

* `PYTHONPATH=src python -m pytest tests/test_reference_provenance.py -q` → 17 passed（12 存量 + 5 新增）
* `PYTHONPATH=src python -m pytest tests/test_capability_evidence.py -q` → 18 passed（能力台账块与新 claim 一致）
* 集群全量：`python -m pytest -q` 递交集群（`cj1416`），相关子集（reference_provenance + capability_coverage + capability_registry_debt + wigley3_recalled_ref）递交 `cj1415`；
  递交时队列前有 ~557 个作业，两单在控制面 120 s 窗口内仍处 `queued`，**结果未回传**——本报告不代填它们的通过与否，本地已过的是上两条加 5 条新用例（`-k` 选中 5 passed）。
* `PYTHONPATH=src python -m tensorbem.reference_provenance` → 退出码 0，`leaked=0 unadjudicated=0`
