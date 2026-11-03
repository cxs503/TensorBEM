# TensorBEM 三维基准表验证欠账结清：表 2/表 4 全行钉桩 + 1280 面元行直接复现（m2638）

**任务**：m2638（data 席，自找活） · **分支**：agent/m2638 · **基线**：`fc1c094`（origin/main，含 PR#29）
**仓**：TensorBEM · **日期**：2026-11-03

---

## 1. 案例介绍

`docs/marine_bem_benchmarks.md` 的无自由液面三维部分有两个发布表格：

- **表 2**（§4.2）平移球体：面元数 20 / 80 / 320 / 1280，列 = `h_max`、数值附加质量、相对误差、逐面元通量相对 L2 误差、附加质量观测阶；解析值 `ρV/2 = 2146.75498`。
- **表 4**（§4.4）三轴平移椭球（半轴 1.5/1.0/0.75）：同样四行，列 = `h_max`、`V/V_exact`、三个方向的相对误差与观测阶；解析值 `m_{A,i}=ρV α_i/(2-α_i) = (1211.052, 2437.1535, 4184.3892)`。

**欠账来源**：`expand_audit_tensorbem.md` §2.3（2026-10-24）判定「1280 面元行未直接复现」，§4.2 列为需关注项；此后 `docs/benchmark_pins.json` 只钉了 2-D 圆柱与 Ursell 六个算例（`doc_pins.CASES` 原 6 条），**整个三维章节仍是无判据的散文**。这类「已合并功能对文献/解析基准复核」正是本单优先项。

## 2. 怎么做的

1. **先复算、后钉桩**：单线程（`OMP_NUM_THREADS=1`）串行跑 `sphere_added_mass` 与 `ellipsoid_added_mass` 的 `subdivisions 0..3`，输出落 `artifacts/m2638/bench_{sphere,ellipsoid}_sub{0..3}.json`；1280 面元两行分别耗时 25.4 s 与 74.7 s。
2. **把表变成判据**：`src/tensorbem/benchmarks/doc_pins.py` 新增 8 个 `PinCase`（球体/椭球 × 20/80/320/1280，度量 `h_max`、附加质量、相对误差、通量 L2 误差、`volume_ratio`、三轴误差），`subdivisions ≥ 2` 标 `slow=True` 不占 CI 默认时间；`docs/benchmark_pins.json` 由 `--write --include-slow` 程序化重建（14 个算例）。
3. **从正文反解比对**：`tests/test_three_dimensional_doc_tables.py` 用正则把表 2/表 4 的 markdown 数字（含 `\(4.4674\times10^{-3}\)` 形式）解析出来，与钉死值比对；观测阶列不单独钉，而是用相邻两行 `log(e_粗/e_细)/log(h_粗/h_细)` 重算后比对，防止「钉住派生量掩盖误差抵消」。
4. **慢用例守最贵的一行**：`TENSORBEM_SLOW=1` 时现场重跑 1280 面元行，与正文数字比到 `rel=1e-5`。

## 3. 计算结果（零手抄，见 `artifacts/m2638/three_dimensional_table_audit.json`）

表 2 球体（解析 `2146.75498`）：

| N | `h_max` | 数值 m_A | 相对误差（实测=发布） | 观测阶（重算=发布） |
|---:|---:|---:|---:|---:|
| 20 | 1.05146 | 1690.18039 | 2.1268e-1 | — |
| 80 | 0.61803 | 2007.73279 | 6.4759e-2 | 2.238 |
| 320 | 0.32492 | 2109.44290 | 1.7381e-2 | 2.046 |
| 1280 | 0.16465 | 2137.16456 | 4.4674e-3 | 1.999 |

表 4 椭球三轴相对误差：20 → 2.0855e-1/2.1126e-1/2.2541e-1；80 → 6.1623e-2/6.4728e-2/7.0201e-2；320 → 1.6313e-2/1.7371e-2/1.9080e-2；1280 → **4.1679e-3/4.4621e-3/4.9359e-3**（观测阶 2.007/1.999/1.989）。`V/V_exact` 四行为 0.605461/0.873453/0.966161/0.991394。逐位与发布表一致。

图：`docs/figures/m2638_three_dimensional_pin_audit.png`（左：球体误差-面元尺寸与斜率 2 参考线；右：椭球三轴同图）。

## 4. 注意点

- **容量纪律**：初版一次性并发 8 个进程且未限 BLAS 线程，把本机 load1 从 44 推到 418（nproc=128）；已全部收敛为单线程串行，代价是 1280 面元行 25–75 s。教训：三维基准的 `np.linalg.solve` 会自动吃满线程，重算必须显式 `OMP/OPENBLAS/MKL_NUM_THREADS=1`。
- **既有失败如实保留**：`tests/test_documented_benchmarks.py::test_analytic_rankine_panel_integral_is_exact` 在**基线 commit 的干净 worktree 上同样失败**（-1.693139665 vs 精确 -1.693147180，相对 4.4e-6），与本单改动无关，未修，留作下一条欠账。
- 同文件 `test_pinned_cases_reference_existing_docs` 亦在基线为红（文档未指向 pin 文件）；本次已在 §4.2/§4.4 补上 `docs/benchmark_pins.json` 指引，转为绿。
- 局限：钉桩只覆盖无自由液面三维章节；表 3（二维椭圆柱）、表 8/9（MacCamy-Fuchs）与 Hulme/Wang-Linton 仍未纳入同一注册表。

## 5. 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| 球体 1280 面元附加质量 | 解析 ρV/2 | 4.4674e-3 | **是** |
| 椭球 1280 面元三轴附加质量 | 解析 ρVα/(2-α) | 4.1679e-3 / 4.4621e-3 / 4.9359e-3 | **是** |
| 发布表数字 vs 当前代码重算 | 文档正文 | 全部 < 5e-5 相对 | **是** |
| 观测阶列 vs 重算 | 文档正文 | 相对 <1e-3（三维）/ <5e-3（椭球） | **是** |

## 6. 验证

- `PYTHONPATH=src pytest tests/test_three_dimensional_doc_tables.py tests/test_documented_benchmarks.py tests/test_declared_numbers.py -q` → **33 passed, 3 skipped, 1 failed**（唯一失败为上述基线既红用例）。
- `python -m tensorbem.benchmarks.doc_pins --check`（默认，不含 slow）→ 全通过；`--write --include-slow` 重建 14 个算例，既有 6 个算例度量漂移 ≤ 5e-13（BLAS 非确定性量级），无实质变化。