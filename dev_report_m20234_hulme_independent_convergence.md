# Dev report m20234 — CASE-BEM-HULME-HEMISPHERE 独立轴自身收敛性审计

**锚点**：CASE-BEM-HULME-HEMISPHERE（算例池·Hulme 浮半球辐射/绕射，容差 <5%）
**仓**：TensorBEM　**类型**：验证欠账复核（adversarial review 式）+ 工具化

## 1. 发现的问题

已发布的独立裁定（`docs/case_bem_hulme_independent_axis.md`）报告「参考解与第三方 BEM（capytaine）最坏码间差 3.36%」，
其支撑判据 `I3_capytaine_mesh_converged` 只证明「最后一档网格漂移（1024→1600 面板）< 1%」。
**网格步长小 ≠ 收敛**：一列每档仍移动 0.67% 的序列可以离自己的连续极限好几个百分点，
而该残差与 3.36% 同量级时，这 3.36% 究竟说明参考解有误还是第三方码没算准，没有答案。

## 2. 做了什么

1. `scripts/capytaine_hulme_ladder_extend_partial.py`：把 capytaine 阶梯从 4 档延拓到 7 档
   （256/576/1024/1600/2304/3136/4096 面板，全 ka 频带 0.2–2，6 个量），
   前 4 档逐位沿用已发布冻结值（`levels_provenance` 标注 `reused-frozen`），后 3 档本次现算
   （工件 `benchmarks/verified/casepool_bem_hulme_hemisphere/capytaine_ladder_extended.json`）。
2. `src/tensorbem/benchmarks/hemisphere_independent_convergence.py`：对每（量, ka）拟合三参数收敛模型
   `v(N)=v_inf+C·N^-p`（阶 p 由数据估），并用 4 档窗口 / 7 档窗口互换、最细档残差、
   「参考 vs 冻结档」与「参考 vs v_inf」两列做裁定；全部数字由工件程序化渲染进
   `docs/case_bem_hulme_independent_convergence.md`（零手抄）。
3. `scripts/hulme_hemisphere_inhouse_ladder.py`：本仓求解器在 ka=2 的延拓阶梯（64→4096 面元）。
   **未产出**：4096 面元档在集群节点上以 `malloc(): invalid size (unsorted)` 崩溃（见 §5）。

## 3. 结果（honest FAIL）

| 判据 | 实测 | 裁定 |
|---|---|---|
| `E1_convergence_model_credible` | 0.08% | PASS |
| `E2_observed_order_identified` | 878.46% | FAIL |
| `E3_extrapolation_stable` | 18.29% | FAIL |
| `E4_frozen_level_residual` | 21.02% | FAIL |
| `E5_adjudicated_spread` | 16.33% | FAIL |
| `E6_extrapolation_moves_toward_ref` | 16.33% | FAIL |

- 最坏格：**B33, ka=2** —— 冻结 1600 档距其外推极限 **21.02%**，
  参考 vs 外推极限 **16.33%**（冻结档口径 3.36%）。
- 全局幂律模型在多个格子上**不可辨识**（观测阶贴边界、窗口互换不一致），
  因此「把 3.36% 归因于第三方码离散误差」这一假设**本轮未能证实**——按诚实裁定记 FAIL，而不是放宽判据。

## 4. 结论与建议

- `I3` 的措辞应改为「最后一档步长 < 1%」，不得读作「独立码已收敛」。若要支撑 5% 级裁定，
  独立码必须先给出连续极限（更高档 + 可辨识的收敛阶，或换成解析级数/更细离散）。
- 后续（建议新条目）：`AUDIT-BEM-HULME-INDEPENDENT-AXIS-LIMIT`——用局部（逐格）阶估计 + 逐格 4/5/6/7 档窗口交叉验证取代全局幂律，
  并为 capytaine 换取水线加密（`immersed_part()` 切出多边形水线）后的对照，量化水线离散对残差的贡献。

## 5. 生态摩擦（如实登记）

- **集群节点无 capytaine**：作业 cj2583 以 `ModuleNotFoundError: No module named 'capytaine'` 失败；
  延长阶梯只能在控制面按 ≤115 s 时间片完成（本次一次跑完 7 档，95 s）。
- **本仓求解器 4096 面元崩溃**：作业 cj2582 在 subdivisions=5 时以 glibc `malloc(): invalid size (unsorted)` 终止，
  这是真实的规模上限缺陷（1024 → 4096 的 16 倍矩阵装配路径），建议单独立项。
- 该作业在 ka=2 打印的 A33/B33（0.3666/0.0508，1024 面元）与已发布 results.json 同格（0.38634/0.10425）不一致，
  **本单未及核实**，可能是驱动侧或算例侧缺陷，建议下一次自找活优先查证。

## 6. 验证

- `PYTHONPATH=src python -m tensorbem.benchmarks.hemisphere_independent_convergence` → 生成证据/文档/图，裁定 FAIL（诚实）。
- `PYTHONPATH=src python -m pytest tests/test_casepool_hulme_independent_convergence.py -q` → **7 passed, 1 skipped**。
