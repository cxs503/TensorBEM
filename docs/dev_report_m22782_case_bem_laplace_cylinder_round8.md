# CASE-BEM-LAPLACE-CYLINDER 第 8 轮（记录重生成被队列阻塞：main 已红的门禁 + 本机不可行的实测证据）

**锚点** CASE-BEM-LAPLACE-CYLINDER（算例池 `BEM-LAPLACE-CYLINDER`，ref 解析复势，容差 `<3%`）
**仓** TensorBEM · **分支** agent/m22782 · **基线** main `aa72b8d9`

## ① 本轮结论（先给判决）

1. **main 当前是红的**，红的正是本锚点的记录陈旧欠账：
   `tests/test_near_wall_tolerance_boundary.py::test_committed_record_gates_the_tolerance_boundary`
   在 `aa72b8d9` 上 `FAILED`（实测，见 §2.1）。这不是本轮引入的，是本轮**首次定位到具体门禁**。
2. **重生成是本轮唯一正确的修复**，但它在**本机不可行**（实测，见 §3），且**集群作业至今未拿到节点**
   （`cj4375` 递交 02:34，排队 88 min 仍在 `queued`，见 §4）。因此本轮**不落提交记录**
   ——用陈旧数据手写/拼接 `results.json` 会制造比陈旧更坏的证据。
3. 本轮交付的是**可执行、可复现的阻塞证据 + 已备好的补丁文本**（§5），下一轮拿到一次节点即可一次收口。

> honest-FAIL：本轮**未**完成记录重生成，锚点仍 `open`。不粉饰。

## ② 缺陷定位

### 2.1 main 上已红的门禁（本轮实测）

```text
$ python3 -m pytest tests/test_case_pool_verdict.py::test_committed_cylinder_record_re_derives_cleanly \
    tests/test_case_cylinder_reference_independent.py::test_committed_record_names_the_potential_this_audit_checks \
    tests/test_near_wall_tolerance_boundary.py -q
...
E   AssertionError: the committed record predates the tolerance-boundary study and is now cited by the rendered document; re-run: PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir benchmarks/verified/case_bem_laplace_cylinder --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check
E   assert None is not None
tests/test_near_wall_tolerance_boundary.py:124: AssertionError
1 failed, 5 passed, 1 warning in 40.77s
```

同批 5 条通过（说明求解器与记录的可复算部分都健康，缺的只是**近壁容差边界研究那一节**）。

### 2.2 提交记录与当前模块的差

| 项 | 提交记录 `benchmarks/verified/case_bem_laplace_cylinder/results.json` | 当前模块 `src/tensorbem/benchmarks/case_cylinder_stream.py` |
| --- | --- | --- |
| 判据数 | **38** | 53（含近壁线性密度、刻面、方位角、斜来流 `heading_sweep`） |
| `provenance` | `null` | `{"generator": {..., "combined_sha256": ...}}` |
| `near_wall` 子键 | `window, exclusion, guard, ladder, factors, levels, probe_directions, note` | 另含 `floor`（容差边界研究） |
| 环境 | `python 3.11.6 / numpy 2.4.6 / loongarch64` | — |

直接后果有两条（都已由现有测试表达，不是新提法）：

- `tests/test_near_wall_tolerance_boundary.py::test_committed_record_gates_the_tolerance_boundary`
  → **红**（记录缺 `near_wall.floor`）。
- `tests/test_case_cylinder_stream.py::test_committed_record_was_rendered_from_the_current_case_module`
  → **skip**（记录无 `provenance`，无法证明它描述当前模块）。本轮实测：
  `1 passed, 1 skipped, 38 deselected in 0.74s`，skip 原因文本即 `the committed record carries
  no generator provenance ...`。

一个 **skip + 一个 fail 指向同一笔欠账**：fail 说明门禁是硬的，skip 说明还有一处门禁在装绿。
两者都只能由「用当前模块重生成记录」同时消除，故 `docs/dev_report_m22521_*.md` §2.3 预先写下的
`skip → pytest.fail` 强化**必须与重生成同批落地**；本轮没有重生成，因此**没有**启用它
（启用只会让 main 在同一笔欠账上多红一条）。

### 2.3 提交记录的数值本身是好的（不是精度问题）

| 关键量（N=128） | 基准 | 相对误差 | `<3%` |
| --- | --- | ---: | --- |
| 路线 D 外部速度 $L_2$ | $dW/dz$ | 1.366e-07 | 是 |
| 路线 D 扰动势 $L_2$ | $Ua^2\cos\theta/r$ | 3.578e-07 | 是 |
| 路线 N 物面速度 $L_2$ | $2U\lvert\sin\theta\rvert$ | 1.008e-04 | 是 |
| 路线 N 外部速度 $L_2$ | $dW/dz$ | 1.532e-04 | 是 |
| 路线 N $C_p$ $L_2$ | $1-4\sin^2\theta$ | 2.849e-04 | 是 |
| 环量路线升力 | Kutta–Joukowski | 4.014e-04 | 是 |

38 条判据在提交记录内自洽（`test_committed_record_criteria_are_all_counted` 绿，
`test_committed_cylinder_record_re_derives_cleanly` 绿）。**缺的是覆盖面与指纹，不是精度。**

## ③ 本机重行为什么不可行（实测，不是推测）

- 本机 shell 单次窗口 120 s（超时被收割）。实测 `--quick --no-controls`
  在 120 s 内**没有写出任何工件**（`TIMEOUT after 120s`）。
- 单点实测：**一次** `route_neumann(128)` 在 100 s 窗口内**未返回**
  （宿主 load average ≈ 46，128 核共享，浮点吞吐被邻居挤压）。
- 据此，完整入口点（3 条路线 × 3 级 + 近壁/多极/斜来流/环量扫描 + 5 张图）在本机是
  **10 min 级**任务，**超出 owner 2026-11-13 计算调度令允许的本机窗口**，本机跑属于违规；
  后台化（`nohup`/`&`）已被本机计算门硬拦（实测拦截回执）。

结论：**重生成只能走集群**，没有第二条路。

## ④ 集群阻塞证据（作业队列）

| 作业 | 用途 | 核 | 递交 | 状态 |
| --- | --- | ---: | --- | --- |
| `cj4374` | 探针：节点 `python/matplotlib` 可用性 + 记录目录 | 1 | 02:33 | `queued`（>90 min 未派发） |
| `cj4375` | **重生成主作业**：`python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir artifacts/... --doc ... --check`（`stage_dir` = 本单工作树） | 4 | 02:34 | `queued`（88 min 仍未派发，`queued_ahead` 递交时 55） |

队列态势（同源回执）：`jobs_in_flight` 73 → 87（观察窗内**只进不出**），
近 2 日 `core_min_occupied` 8.2e4，节点 `n315`/`n330` 均健康（`load` 42/40、`mem_free` 925 G、`disk_free` 3.0 T）。
即：**不是节点坏了，是队列饱和**。这是本锚点第 5、6、7 轮反复卡在同一处的根因
（前序轮 cj3470 / cj3943 / cj4188 / cj4219 均止于队列或取件）。

对 owner 的建议（不改验收口径，只改可维护性）：
1. 给「记录重生成」类作业一个**小核数、短超时**的优先通道（单进程、无 MPI，实占 1 核），
   避免与 8–16 核的大作业同池排队；
2. 或允许此类作业在**低负载窗口**（load15 < 8）使用 ≤10 min 的本机窗口（实测该窗口足够）；
3. 若两者都不可行，则按 `docs/dev_report_m22521_*.md` §5 的建议把阶梯预算从 N=128 降到 N=64
   并在文档显式声明——**这属验收口径变更，需 owner 同意，本单不擅自改**。

## ⑤ 已备好的补丁（拿到节点后一次收口）

拿到 `cj4375` 工件后，按下列顺序落地（3 个文件、单一功能点）：

1. `benchmarks/verified/case_bem_laplace_cylinder/results.json` / `report.md` / `case.json`
   ← 由 `--outdir` 工件直接覆盖（判据数应为 53、`provenance.generator` 非空、
   `near_wall.floor` 存在、`overall == "PASS"`）。
2. `docs/CASE-BEM-LAPLACE-CYLINDER.md` ← 同一次运行的 `--doc` 输出（与 `results.json` 逐字一致）。
3. `tests/test_case_cylinder_stream.py`：把
   `test_committed_record_was_rendered_from_the_current_case_module` 的
   `if "provenance" not in payload: pytest.skip(...)` 换成：

```python
    rerun = (
        "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream "
        "--outdir benchmarks/verified/case_bem_laplace_cylinder "
        "--doc docs/CASE-BEM-LAPLACE-CYLINDER.md"
    )
    assert "provenance" in payload and payload["provenance"], (
        "the committed record carries no generator provenance, so it cannot be "
        "shown to describe the current case module: the document, the "
        "``solver.evaluate_exterior_field`` docstring and the near-wall "
        f"tolerance-boundary gate all cite it as evidence. Re-run: {rerun}"
    )
```

验收（下一轮必跑，本机即可，均 < 60 s，不触发重计算门）：

```text
python3 -m pytest tests/test_near_wall_tolerance_boundary.py \
  tests/test_case_cylinder_stream.py -k "not quick_ladder" -q
```

预期：`test_committed_record_gates_the_tolerance_boundary` 由红转绿、
`test_committed_record_was_rendered_from_the_current_case_module` 由 skip 转**真检**并通过。
`test_case_acceptance_passes_on_the_quick_ladder` 属重计算，随 `cj4375` 或其后续作业在节点上跑。

## ⑥ 局限与适用范围

- 本轮**没有任何数值被重新测量**：§2.3 的数字全部复自**提交记录**（N=32/64/128 的三级梯级），
  它们证明「记录里的物理是对的」，**不证明**「当前模块在完整阶梯上 53 条判据全过」。
  后者只有重生成能证明，本轮如实记为**未完成**。
- §3 的本机耗时是在 load ≈ 46 的共享宿主上测的，属**上界**；低负载下 `--quick` 可能落进 120 s，
  但完整阶梯仍需 10 min 级——不足以改变结论。
- 本轮**没有**改任何阈值、容差、判据或验收口径；也没有为了变绿去动测试。

## ⑦ 复核命令清单（本轮实际跑过）

```bash
# 红门禁定位（40.77 s）
python3 -m pytest tests/test_case_pool_verdict.py::test_committed_cylinder_record_re_derives_cleanly \
  tests/test_case_cylinder_reference_independent.py::test_committed_record_names_the_potential_this_audit_checks \
  tests/test_near_wall_tolerance_boundary.py -q
# → 1 failed, 5 passed

# skip 门禁确认（0.74 s）
python3 -m pytest tests/test_case_cylinder_stream.py \
  -k "committed_record_was_rendered or committed_record_criteria_are_all_counted" -q
# → 1 passed, 1 skipped

# 本机不可行性（120 s 窗口被收割 / 100 s 内未返回）
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream --quick --no-controls --outdir /tmp/ccs_quick
PYTHONPATH=src python3 -c "from tensorbem.benchmarks import case_cylinder_stream as c; c.route_neumann(128, c.RADIUS, c.SPEED)"
```