# TensorBEM 补全 bem_finite_depth 验证算例缺失的 4 张结果图

**任务**：m13784（selfstart·data 席） · **分支**：`agent/m13784` · **日期**：2026-11-11
**仓**：TensorBEM
**归仓理由**：求解器仓的验证算例工件补全（ECOSYSTEM-MAP：求解器→对应 Tensor* 库）

EDIT_WHITELIST: benchmarks/verified/bem_finite_depth/figures/

---

## 1. 问题

`tests/test_bem_finite_depth_anchors.py::test_case_artifacts_and_documentation_exist` 断言
`benchmarks/verified/bem_finite_depth/figures/` 目录下至少有 4 张 PNG 图，但该目录从未入库，
导致测试在 clean main 上必败。该问题在 PR#306（m8788）和 PR#312（m8799）的 CI 日志中均被记录为
"与本 PR 无关的既有失败"，但一直未被修复。

`src/tensorbem/benchmarks/bem_finite_depth_anchors.py` 的 `make_figures()` 函数可生成 4 张图：
- `fig_energy_identities.png`：各工况的反射/透射能量守恒残差
- `fig_kramers_kronig.png`：Ursell 解析阻尼与面板扫描的 KK 因果性对比
- `fig_deep_water_limit.png`：有限深→无限深渐近收敛
- `fig_section_matrix.png`：截面×深度×频率矩阵的附加质量/阻尼热力图

## 2. 修复

运行基准程序生成图：

```bash
PYTHONPATH=src python -m tensorbem.benchmarks.bem_finite_depth_anchors --stage merge
```

输出：
```
figure -> benchmarks/verified/bem_finite_depth/figures/fig_energy_identities.png
figure -> benchmarks/verified/bem_finite_depth/figures/fig_kramers_kronig.png
figure -> benchmarks/verified/bem_finite_depth/figures/fig_deep_water_limit.png
figure -> benchmarks/verified/bem_finite_depth/figures/fig_section_matrix.png
```

## 3. 验证

```
$ PYTHONPATH=src python -m pytest tests/test_bem_finite_depth_anchors.py -v
======================== 17 passed, 1 warning in 3.98s =========================
```

全部 17 条测试通过（此前 `test_case_artifacts_and_documentation_exist` 为唯一失败项）。

## 4. 局限

- 本单仅补全缺失工件，未修改任何求解器代码或测试逻辑。
- 图的生成依赖 `result.json` 中的已有数据；若未来基准参数变更，需重新生成。
