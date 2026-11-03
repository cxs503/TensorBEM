# 三维文档数字的可执行化：`doc_pins_3d` 验证门禁

**范围**：TensorBEM，`docs/three_dimensional_bem.md` 的四张结果表（内域 Laplace、外域常值 Dirichlet、外域球体附加质量、外域球体速度）。
**问题类别**：验证欠账 —— 已合并功能/已发表数字没有机器判据。
**判据**：文档写的每个数字，都由 `src/tensorbem/benchmarks/doc_pins_3d.py` 从文档正文反解出声明值、重算、比对；漂移超过容差即测试红。

---

## 1. 为什么这件事值得做

`docs/three_dimensional_bem.md` 是这个仓库三维线的验收面：第 5.1–5.4 节的四张表既被 README 引用，也承载了"速度场表示公式在表面失效、恢复路径替代"这一关键结论的量化证据（53–65% → 1.38e-2）。这些数字此前是**手写 LaTeX 散文**，CI 里没有任何东西盯着它们。

同一类失效在本仓库已经真实发生过：自由液面源势求积改动（commit `267d5a3`）把 128 面元 Ursell 附加质量误差从 `7.72e-3` 挪到 `8.51e-3`，文档静默过期若干提交才被审计发现（`docs/ursell_quadrature_provenance.md`、`expand_audit_tensorbem.md` §2.2）。二维线的补丁是 `tensorbem.benchmarks.doc_pins`（`docs/benchmark_pins.json`）；三维线一直没有对应的东西。本单补上。

## 2. 怎么做的

1. **不从文档抄数字。** 新增模块只登记"每个数字住在哪一格"：表 id、面元数、列符号、对应的计算函数与输出字段。声明值在检查时由 `parse_doc_tables()` 直接读 Markdown、用 `decode_number()` 反解 `\(1.4600\times10^{-1}\)` / `1.0171e-1` / `\(9.13\)` / `—` 这几种写法。
2. **表按块解析。** 初版按"列数相同即同表"的启发式归行，结果内域表和附加质量表的行互相串台（附加质量 `m_A` 被拿去和内域 `E_q` 比），28 个 pin 里出现 16 个假漂移。改为按表头标记定位块、块内逐行取列，误配消失。这条教训写进了模块注释。
3. **重算走库内原有基准。** `three_dimensional_study.run_interior_case` / `run_constant_dirichlet_case` / `sphere_added_mass.run` / `sphere_velocity.run`，一行代码不改，因此 pin 检验的是**发布代码**而不是一份并行实现。
4. **容差与量级。** 误差类数字用相对容差 `5e-3`（远小于历史 10.6% 那种真实漂移，又能吸收文档 4–5 位有效数字的舍入）。行和误差、线性系统相对残差这类 `1e-16` 量级数字另设 `1e-13` 的绝对下限：文档对它们的断言只是"到 `1e-15` 量级"，首位数字是 BLAS 舍入的产物，做相对比较没有意义。
5. **派生量不单独钉。** 各表的观测阶列（`p_q`、`p_u`、`p_m`、`p`）是相邻两行误差的算术结果，单独 pin 只会引入噪声；它们在 `COLUMNS` 里留 `None` 占位，保证列对齐。
6. **负控。** `tests/test_documented_benchmarks_3d.py` 把文档里 20 面元内域通量误差 `1.46e-1` 改成 `1.96e-1`，断言检查器必须报漂移——防止"检查器睡着了也全绿"。
7. **CI 只跑粗网格。** 默认 `--check` 跑 level ≤ 1（5.7 s）；`--levels 3`（含 1280 面元行，本机约 8 min）用于发布前全量核对。

## 3. 产物

| 文件 | 作用 |
|---|---|
| `src/tensorbem/benchmarks/doc_pins_3d.py` | pin 登记 + 文档解析 + 重算比对 + 记录写出 + 图 |
| `docs/benchmark_pins_3d.json` | 机器可读记录：每个 pin 的声明值、重算值、漂移、产生该记录的 commit |
| `tests/test_documented_benchmarks_3d.py` | 结构 / 正控 / 负控 / 记录与文档一致性 |
| `docs/figures/doc_pins_3d.png` | 左：文档值 vs 重算值的误差收敛曲线；右：全部 pin 的漂移散点 |

复现：

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.doc_pins_3d            # 快速核对
PYTHONPATH=src python -m tensorbem.benchmarks.doc_pins_3d --levels 3 --write
PYTHONPATH=src python -m tensorbem.benchmarks.doc_pins_3d --levels 3 --figure docs/figures/doc_pins_3d.png
```

## 4. 注意点与适用范围

1. **pin 的是数字，不是物理。** 通过只说明"文档与当前代码一致"，不说明三维基线本身正确；物理论证仍靠各基准自带的解析参考（球体附加质量 `0.5 ρV`、常值 Dirichlet 总通量 `4πa` 等）。
2. **文档解析器有前提。** 它要求四张表的列数与 `COLUMNS` 对齐、行首单元格是面元数。改表结构（加减列）会立刻红——这是有意的，但意味着**改表必须同时改 `COLUMNS`**。
3. **1280 面元行的成本。** 稠密直接法 `O(N³)`，单次重算在本机约数十秒到数分钟；因此 CI 只钉粗网格，全量核对是发布前动作。
4. **非确定性来源。** 所有 pin 都是确定性数值；跨 BLAS/NumPy 版本的差异在相对容差 `5e-3` 之内（本单实测远小于此，见第 5 节），但 `1e-16` 量级的两个数字依赖舍入，已按下限规则处理。
5. **未覆盖**：`docs/pressure_forces.md`、`docs/finite_depth_radiation.md`、`docs/hydrostatic_verification.md` 的数字仍未 pin，属于后续同构工作。

## 5. 结果（程序化注入）

记录：`docs/benchmark_pins_3d.json`（commit `71e51fcb7d62`，max_level=2，相对容差 0.005，重算耗时 61.555 s）。

| 表 | N | 列 | 文档值 | 重算值 | 漂移 | 判定 |
|---|---:|---|---:|---:|---:|---|
| interior_laplace | 20 | h_max | 1.05146 | 1.05146 | 2.12e-06 | PASS |
| interior_laplace | 20 | E_q | 0.146 | 0.146004 | 2.74e-05 | PASS |
| interior_laplace | 20 | E_u | 0.014321 | 0.0143212 | 1.20e-05 | PASS |
| interior_laplace | 20 | kappa | 9.13 | 9.13482 | 5.28e-04 | PASS |
| interior_laplace | 20 | row_sum_error | 1.1e-16 | 1.66533e-16 | 5.65e-04 | PASS |
| interior_laplace | 20 | relative_residual | 1.46e-16 | 2.7874e-16 | 1.33e-03 | PASS |
| interior_laplace | 80 | h_max | 0.61803 | 0.618034 | 6.45e-06 | PASS |
| interior_laplace | 80 | E_q | 0.043804 | 0.0438045 | 1.04e-05 | PASS |
| interior_laplace | 80 | E_u | 0.0036992 | 0.00369917 | 8.36e-06 | PASS |
| interior_laplace | 80 | kappa | 18.62 | 18.6215 | 8.02e-05 | PASS |
| interior_laplace | 80 | row_sum_error | 2.2e-16 | 2.22045e-16 | 2.04e-05 | PASS |
| interior_laplace | 80 | relative_residual | 2.69e-16 | 3.04294e-16 | 3.53e-04 | PASS |
| interior_laplace | 320 | h_max | 0.32492 | 0.32492 | 9.35e-07 | PASS |
| interior_laplace | 320 | E_q | 0.012334 | 0.0123338 | 1.52e-05 | PASS |
| interior_laplace | 320 | E_u | 0.00092055 | 0.000920547 | 3.10e-06 | PASS |
| interior_laplace | 320 | kappa | 37.57 | 37.5656 | 1.17e-04 | PASS |
| interior_laplace | 320 | row_sum_error | 5.6e-16 | 5.55112e-16 | 4.89e-05 | PASS |
| interior_laplace | 320 | relative_residual | 3.69e-16 | 7.48256e-16 | 3.79e-03 | PASS |
| exterior_added_mass | 20 | h_max | 1.05146 | 1.05146 | 2.12e-06 | PASS |
| exterior_added_mass | 20 | m_A | 1690.18 | 1690.18 | 1.11e-09 | PASS |
| exterior_added_mass | 20 | rel_err | 0.21268 | 0.212681 | 6.01e-06 | PASS |
| exterior_added_mass | 20 | flux_rel_l2 | 0.033338 | 0.0333381 | 4.43e-06 | PASS |
| exterior_added_mass | 80 | h_max | 0.61803 | 0.618034 | 6.45e-06 | PASS |
| exterior_added_mass | 80 | m_A | 2007.73 | 2007.73 | 3.69e-10 | PASS |
| exterior_added_mass | 80 | rel_err | 0.064759 | 0.0647592 | 3.49e-06 | PASS |
| exterior_added_mass | 80 | flux_rel_l2 | 0.027884 | 0.0278842 | 6.45e-06 | PASS |
| exterior_added_mass | 320 | h_max | 0.32492 | 0.32492 | 9.35e-07 | PASS |
| exterior_added_mass | 320 | m_A | 2109.44 | 2109.44 | 2.74e-11 | PASS |
| exterior_added_mass | 320 | rel_err | 0.017381 | 0.0173807 | 1.79e-05 | PASS |
| exterior_added_mass | 320 | flux_rel_l2 | 0.014542 | 0.0145416 | 3.06e-05 | PASS |
| exterior_constant_dirichlet | 20 | total_flux | 10.4904 | 10.4904 | 4.68e-08 | PASS |
| exterior_constant_dirichlet | 20 | rel_err | 0.1652 | 0.165201 | 4.66e-06 | PASS |
| exterior_constant_dirichlet | 20 | mean_flux | 1.09566 | 1.09566 | 2.26e-07 | PASS |
| exterior_constant_dirichlet | 80 | total_flux | 11.9486 | 11.9486 | 3.70e-08 | PASS |
| exterior_constant_dirichlet | 80 | rel_err | 0.049157 | 0.0491573 | 6.44e-06 | PASS |
| exterior_constant_dirichlet | 80 | mean_flux | 1.02603 | 1.02603 | 4.85e-07 | PASS |
| exterior_constant_dirichlet | 320 | total_flux | 12.4029 | 12.4029 | 5.93e-09 | PASS |
| exterior_constant_dirichlet | 320 | rel_err | 0.013007 | 0.0130072 | 1.69e-05 | PASS |
| exterior_constant_dirichlet | 320 | mean_flux | 1.00674 | 1.00674 | 1.54e-07 | PASS |
| exterior_velocity | 80 | h_max | 0.61803 | 0.618034 | 6.45e-06 | PASS |
| exterior_velocity | 80 | E_v | 0.10171 | 0.10171 | 2.49e-06 | PASS |
| exterior_velocity | 80 | max_abs | 0.029189 | 0.0291895 | 1.62e-05 | PASS |
| exterior_velocity | 80 | E_surf_recovered | 0.15027 | 0.150267 | 2.03e-05 | PASS |
| exterior_velocity | 80 | E_surf_representation | 0.65314 | 0.653139 | 2.23e-06 | PASS |
| exterior_velocity | 320 | h_max | 0.32492 | 0.32492 | 9.35e-07 | PASS |
| exterior_velocity | 320 | E_v | 0.027635 | 0.0276346 | 1.63e-05 | PASS |
| exterior_velocity | 320 | max_abs | 0.0078167 | 0.00781675 | 5.96e-06 | PASS |
| exterior_velocity | 320 | E_surf_recovered | 0.043731 | 0.043731 | 1.11e-06 | PASS |
| exterior_velocity | 320 | E_surf_representation | 0.56878 | 0.568785 | 8.23e-06 | PASS |

**合计 49 个 pin：49 通过、0 漂移**；最大漂移 3.79e-03（interior_laplace/320/relative_residual）。
