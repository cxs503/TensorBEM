# m22876 — CASE-BEM-CUMMINS-WIGLEY 算例池说明的闭环（原 PR#1081 重开第 1 轮）

分支 `agent/m22876`，基于 `origin/main`（合并 `#1113 (m22843)`）。域：船舶总体性能 / 耐波（TensorBEM）。

## 1. 重开时树上的实际状态（先核对，再动手）

原 PR#1081 的六件交付物**已经在 main 上**（`b715a32e` / `1d50e3dd` 一路合入）：

| 文件 | 现状 |
| --- | --- |
| `src/tensorbem/benchmarks/case_bem_cummins_wigley.py` | 3557 行，C0 `solver_validity_gate` 已在（`C0` / `C0_coarse`） |
| `tests/test_case_bem_cummins_wigley.py` | 1311 行，73 例；本机实测 **73 passed, 0 skipped**，无 skip |
| `tools/rerender_casepool_bem_cummins_wigley.py` | 在场 |
| `docs/casepool_bem_cummins_wigley/` | `README.md`、`evidence_cluster_n315_sweep_coarse.npz`、`fig_node_vs_local.png` 全在 |
| `benchmarks/verified/casepool_bem_cummins_wigley/` | `case_result.json` 判定 **16/16 通过**，`sweep_medium.npz`（480 面板）已入库 |

所以这一单**不需要**重解 BEM，也不需要重做闸门：功能侧已经被后续 mission（m21960 在 loongarch64 节点重算）补齐，
`test_production_artifact_replays_to_the_published_verdict` 已经从 skip 变 green。

## 2. 但发现一个真实缺陷：说明文档在断言一个已不存在的状态

`docs/casepool_bem_cummins_wigley/README.md`（m20419 写的）仍在说：

* 「已提交 `case_result.json`」= band 0.15–8.0 / n_omega 13 / coarse 生产网格 / m17375；
  实测现态是 **0.2–12.0 / 41 / medium 20x6（480 面板）/ m21960**；
* 「提交件的判定 5/7 gates passed (FAILED C3, C5)」；实测现态是 **16/16 通过**；
* 「生产网格的 sweep 从来没提交过，闸门一直空转」；实测 `sweep_medium.npz` 已在库，回放测试 green。

这正是本算例自己那条测试的判词所防的事：*"A generated document that is not regenerated is a second,
silent source of truth."* 文档与工件互相矛盾，而且矛盾的方向是**把已经修好的东西说成坏的**——比反过来更
难被发现（没人会去复核一个「已知欠账」）。

## 3. 本单改动（3 个功能文件 + 1 图 + 本报告）

1. **`tools/render_casepool_bem_cummins_wigley_readme.py`（新）**：把 `README.md` 变成**渲染产物**。
   所有数字在渲染时从工件读出：`benchmarks/verified/.../case_result.json`、两份 `sweep_*.npz`、
   `docs/casepool_bem_cummins_wigley/evidence_cluster_n315_sweep_coarse.npz`、`docs/m20021/sweep_medium.npz`。
   两份负控的 C3（Kramers-Kronig 闭合）也不是抄的运行记录，而是调 `case.kk_gate(case.load_sweep(...))`
   在 `.npz` 上**重算**（单次 0.8 s，读者可复现）。
2. **`docs/casepool_bem_cummins_wigley/README.md`（重写，70 行）**：从「欠账清单」改成「欠账 → 闸门 →
   复算 → 现态 → 仍开放项」的闭环记录；历史数字只以「历史现场」出现，现态表由工件渲染。
3. **`tests/test_case_bem_cummins_wigley.py`（+3 例）**：
   * `test_casepool_readme_is_the_render_of_the_artifacts`：`render() == 磁盘上的 README`（渲染等式）；
   * `test_casepool_readme_does_not_restate_the_pre_fix_state`：断言三条「现态」式陈词不在文中，
     且现态数字（16/16、n_omega）与工件一致、文中点名的四张图都真实存在；
   * `test_casepool_readme_render_tracks_the_artifact`（**咬合测试**）：把工件换成改坏的副本
     （band 0.15–8 / 13 节点 / 5-7 判否）后渲染必须**变**，否则上面的等式测试是「因为 prose 没动」
     而过，防自欺。
4. **`tools/make_m22876_figures.py` + `docs/casepool_bem_cummins_wigley/fig_c0_negative_controls.png`（新）**：
   C0 前置条件的分离度图——(a) 逐频相对残差、(b) 逐频最小对角辐射阻尼；生产网格 vs 两份负控。

## 4. 验证（跑了什么）

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 本算例全套 | `python -m pytest tests/test_case_bem_cummins_wigley.py -q` | **76 passed**（73 旧 + 3 新），0 skipped，114 s |
| 新增三例 | `-k casepool_readme` | 3 passed，0.9 s |
| 文档闸门 | `pytest tests/test_doc_claim_audit.py tests/test_doc_claims.py tests/test_doc_inventory.py tests/test_documented_benchmarks.py` | 54 passed, **3 failed**；已在**干净树上复跑同样 3 failed**，且审计计数改前/改后完全相同（documents 317 / pinned 84 / artifact 91 / unguarded 142 / coverage 62.0 %）⇒ 与本单无关，未被我恶化 |
| 相邻算例测试（15 文件，分三批跑，均 <120 s 本地墙钟） | `pytest` 分三批：`added_mass_limit + xval + state_space_kk_bridge + verdict_closure + band_edges`；`decay_dof + g4 + lowtail_bound + kk_tail_certificate + kk_truncation_budget`；`matrix_audit + route_spread + kk_gate_calibration` | **39 passed/6 skipped**、**61 passed/2 skipped**、**33 passed**，无 failed（skip 均为未生成的缓存件） |
| 渲染可复现 | `python tools/render_casepool_bem_cummins_wigley_readme.py` 与 `--write` 后再跑渲染等式测试 | 文本一致 |

## 5. 集群全仓测试

`cj4456`：`python -m pytest tests -q -p no:cacheprovider`，8 核，`stage_dir=/root/tp-agent/work/m22876`
（递交即读自本工作树，非 GitHub main）。

**终态：截至本单收尾仍在排队**（递交后 34 min 仍 `state=queued`，同期队列 in-flight 86–88 个作业，
本单未拿到 `rc`）。按「不谎报」纪律：全仓测试**未**取得集群回执。替代证据是本节的本地分批判证——
本单涉及的 16 个测试文件（算例本体 + 15 个相邻 Cummins/KK 测试）全部本地跑过且无 failed；
本单未改动任何被这些文件测试的模块（只改 `docs/`、`tests/test_case_bem_cummins_wigley.py` 尾部、`tools/` 两个新文件）。

## 6. 诚实边界

* **没有**新增物理能力：C0 闸门、voided 语义、重渲染工装都是 m20419 已合的；本单做的是
  「让算例池说明与工件一致，并把它钉死」，属于文档/证据链缺陷修复（`defect_fix`），不是新算法。
* 图中 (b) 面板把阻尼下限截到 −4：`docs/m20021/sweep_medium.npz` 的 −129.9 落在轴外，
  这是**有意的**——否则其余三条曲线被压成一条直线；数值见 README 2.1 表。
* `NODE-N315-NUMERICAL-VALIDITY` 仍未归因（n315 对同输入给 residual 1.25 / 负阻尼）；
  本单不碰节点，只保证这类运行永远过不了 C0。
* 本单未触发任何 BEM 求解，也不改动 `case_bem_cummins_wigley.py`，与其它开 PR 的热文件路径无重叠。