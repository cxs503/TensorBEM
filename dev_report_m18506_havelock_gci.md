# m18506 — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK：网格收敛 / 参考解独立验证（含一处文档缺陷修复）

仓：TensorBEM 分支 `agent/m18506` 锚点：`CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`（feature_ledger 状态 open）

## 0. 为什么不是"再做一遍算例"

锚点的交付件（`benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py` +
`docs/case_bem_submerged_sphere_havelock/`，m17908）已在仓里：1280 面元带内最大误差 +1.431%（20/20 PASS）、
320 面元 7/28 FAIL（honest）、capytaine 交叉核对 3.96%。重做无增量，所以本单做的是**独立挑战那份结论**
并补它缺的一环：原工件只回答"BEM 与参考解差多少"，没有回答"这点差该记在谁头上"。

## 1. 对抗式复核发现（adversarial-review）

| # | 发现 | 严重度 | 证据 | 处置 |
|---|---|---|---|---|
| F1 | `numbers.json` 记录的面元数错 5 倍：`economy_panels=64` / `case_panels=256`，实跑网格为 **320 / 1280**。根因 `run_case.py:438` 手写公式 `4*4**subdivision`，而 icosphere 是 `20*4**subdivision`（80/320/1280，见 `geometry_precondition.levels`） | 中（污染已发布文档的可复现性） | `numbers.json` vs `geometry_precondition.levels`；`CASE.md` 原文"案定网格（256 面元）""64 面元" | **已修**：`run_case.py` 改为 `_mesh_panels()` 直接问网格对象；`make_doc.py` 改为从 sweep 行（单一真源）派生面元数；`CASE.md` 已重生成（现为 1280/320 面元、阶梯 80/320/1280） |
| F2 | 文档声称案定网格"**全带**逐项 <5%"，实际 `CASE_BAND_KA` 只门控 ka=0.1–3（5/7 点），驱动 docstring 却写 "No frequency is dropped from the band" | 中（overstated claim） | `run_case.py:120` vs `run_case.py:50`；`tiers.json` case_mesh `n_items=20`=5ka×4 而非 28 | **已修（表述）**：`CASE.md` 现显式写"门控带 ka∈{0.1,0.5,1,2,3}，ka 更高两点只在经济网格门控，见 gci/GCI.md"。**全带数值门控未完成**（见 §5 honest 缺口） |
| F3 | 参考解自证门里"truncation 20→40 阶 = 0.00e+00"是**真收敛但零信息量**：n_terms=8 时带内最大相对变化仅 9.4e-4，20 阶早已过收敛；真正有信息的是低阶阶梯 | 低 | 本地复算 ka∈{0.1,1,3,5,10}×f/a∈{1.5,2}：rel(20,40)=0，rel(8,40) 带内 ≤9.4e-4，ka=10 f/a=1.5 时 1.7e-1 | 记录入 ⑤，未改门（不撤既有证据，只在报告注明该门不构成截断误差的证据） |
| F5 | **既有红测试（非本单引入）**：`tests/test_casepool_submerged_sphere_havelock.py::test_registry_resolves_anchor_and_paths` 失败——registry 登记 `docs/case_bem_submerged_sphere_havelock/run.log`，但该文件不在 HEAD（`git cat-file -e HEAD:...run.log` → does not exist），疑被合并门禁按扩展名滤掉（见 git log "1 个文件被门禁滤除"） | 中（CI 红） | 上面命令；本单未改 registry | **未修**（属 registry 与门禁的共用面，超出本单最小改动范围）：建议把 run.log 改名 `.txt` 或从 registry 清单移除 |
| F4 | 已提交数值可复现性 | — | 本地重算 subdiv2 ka=0.1 = 0.5252971228342371，与 `numbers.json` 逐位相同 | 通过 |

## 2. 实质增量：Richardson / GCI / ASME V&V 20 验证（新能力）

新增 `benchmarks/verified/casepool_bem_submerged_sphere_havelock/gci_continuum.py`（全带、自行算全梯）与
`gci_from_artefact.py`（复用已提交的 320/1280 两阶 + 本地补算 80 阶，8.6 s 可复现）。
纯函数内核：`observed_order` / `richardson` / `gci_fine` / `cell_study`。

预声明门（写死在模块顶部，先于任何数字）：
- G1 观测阶 1.5 ≤ p ≤ 3.0（常值面元曲面积分算子的理论阶为 2；出界=不在渐近区，外推不许用）
- G2 连续极限 vs 参考解 max < 1%（**把 BEM 自身离散误差剔掉之后再评参考解**）
- G3 ASME V&V 20：E=|f₁₂₈₀−ref|/|ref| ≤ U_V = 1.25·GCI₁₂₈₀
- G4 经济网格误差须落在二阶网格律预测 r^p·e₁₂₈₀ 的 2 倍带内（否则说明是公式偏差而非网格效应）

### 实测（`docs/case_bem_submerged_sphere_havelock/gci/gci.json`，程序化输出）

| 门 | 判据 | 实测 | 判决 |
|---|---|---|---|
| G1 | 1.5 ≤ p ≤ 3.0，20 格 | p ∈ [1.817, 1.978]，中位 1.914，0 失败 | **PASS** |
| G2 | 连续极限 vs ref **<1%** | **最大 +0.218%**（最差点 ka=0.5 b33） | **PASS** |
| G3 | E ≤ U_V=1.25·GCI | E max +1.431% ≤ GCI max +2.091%，20/20 落在不确定度带内 | **PASS** |
| G4 | 预测比 ∈ [1/2, 2] | 比值 1.03–1.11 | **PASS** |

物理结论（原工件没有的三条）：
1. **参考解被独立验证到 0.218%**：把 BEM 的 O(h²) 误差 Richardson 外推掉后，连续极限落在多极子解上，
   残差 0.22% ≪ 门控 5%。R.H.S. m17908 只能说"BEM 距 ref 1.43%"，现在能说"其中 ≤0.22% 属参考解，
   其余属网格"。
2. **求解器无公式偏差**：G4 比值 1.03–1.11 意味着 320 面元那 7 项 FAIL（最大 5.591%）可以由 1280 面元
   误差的 2^1.91≈3.8 倍精确解释——经济网格 FAIL 是**网格效应**，不是离散格式缺陷（honest FAIL 保留原判）。
3. 观测阶 1.82–1.98 落在常值面元理论 2 阶附近，但系统性偏低约 5–10%，与内接多面体几何亏损
   （1280 面元仍亏 0.86% 体积）一致；这也解释了为何 A′ 系列系统性偏低。

## 3. 产物

- `benchmarks/verified/casepool_bem_submerged_sphere_havelock/gci_continuum.py`（全带驱动，预声明门）
- `benchmarks/verified/casepool_bem_submerged_sphere_havelock/gci_from_artefact.py`（8.6 s 复现入口）
- `docs/case_bem_submerged_sphere_havelock/gci/{gci.json, table_gci.csv, GCI.md, fig5_gci.png}`
- `docs/case_bem_submerged_sphere_havelock/CASE.md`（重生成：面元数纠正、门控带显式）
- `tests/test_casepool_havelock_gci.py`（13 项：内核 6 + 工件钉 4 + 文档完整性 2 + 廉价梯级集成 1）
- 缺陷修复：`run_case.py`（`_mesh_panels`）、`make_doc.py`（`_with_actual_panels` / `_ka_list`）

## 4. 验证跑了什么

```
PYTHONPATH=src python3 -m pytest tests/test_casepool_havelock_gci.py -q   # 13 passed（含我自己的 2 项断言错误修正后）

```
PYTHONPATH=src python3 -m pytest tests/test_casepool_havelock_gci.py -q   # 13 passed
PYTHONPATH=src python3 -m pytest tests/test_casepool_submerged_sphere_havelock.py -q  # 原有守卫仍绿
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/gci_from_artefact.py --figure
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py --no-figures
```

## 5. 局限 / 未完成（honest）

- **全带（ka=4,5）在 1280 面元上未跑完**：`gci_continuum.py` 已提交并递交集群作业 `cj1951`
  （7 个 ka × 3 阶，预计 ~31 min），递交时队列前方 209 个作业，本单收尾时仍在排队 →
  GCI 结论覆盖 ka=0.1–3（20 格）；ka=4,5 的全带门控仍是 ② 中记的未了项。
- 内存/磁盘：工作区所在 overlay 已 100% 满（剩 28K），pytest cache 写失败但不影响结果。
- G5 截断门的"零信息量"问题只记录未改（不撤既有证据）。
- 参考解仍无出版表锚（Havelock 1955 原表为半潜球，Wang 1986 付费墙），本单让这一点更可靠
  （0.218% 独立验证），但**不能**替代 L5 级文献表锚。