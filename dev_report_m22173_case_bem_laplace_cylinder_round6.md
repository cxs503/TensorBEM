# CASE-BEM-LAPLACE-CYLINDER 第 6 轮（记录刷新+测试固化）— TensorBEM m22173

**锚点** CASE-BEM-LAPLACE-CYLINDER（算例池·圆柱绕势流速度场，容差 <3%）· **仓** TensorBEM · **分支** agent/m22173

## ① 问题陈述

锚点 `CASE-BEM-LAPLACE-CYLINDER` 在 feature_ledger 中状态为 `open`，验收标准为：
「算例池|BEM-LAPLACE-CYLINDER｜ref: 解析复势｜容差: <3%」。

前序工作（m21490 第 5 轮）已实现完整的算例模块（`case_cylinder_stream.py`），
覆盖 44 条预置判据（路线 D/N/Γ/M + 近壁线性密度 + 方位角扫描 + 几何刻面残差归因）。

但存在**记录陈旧**欠账：
- `benchmarks/verified/case_bem_laplace_cylinder/results.json` 仅含 38 条判据（缺 linear/azimuth/faceting 节）
- `provenance` 字段缺失（`"NONE"`），无法通过 `test_committed_record_was_rendered_from_the_current_case_module`
- `figures/` 目录不存在（5 张图件未提交）
- `report.md` 与当前 `results.json` 不同步

第 5 轮曾递交集群作业 cj3470 重生成，但工件未取回/未落盘。

## ② 本轮工作

### 2.1 集群重生成递交

已递交集群作业 `cj3943`（8 核，3600s 超时）：
```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream \
  --outdir benchmarks/verified/case_bem_laplace_cylinder \
  --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check
```

该命令将生成：
- `results.json`（含 44 条判据 + provenance）
- `report.md`
- `case.json`
- `figures/` 目录（5 张图：field.png, convergence.png, near_wall.png, multipole.png, circulation.png）

### 2.2 源码指纹记录

当前 case 模块的 generator provenance（用于 cj3943 完成后验证）：

| 文件 | SHA-256 |
| --- | --- |
| benchmarks/case_cylinder_stream.py | ef85a5acac0e4bb4aa47dbb8731f8b027a04f7f5c594618db0236b7bf59cdb67 |
| solver.py | 4f242d5e2f4c679c128c64415485aa4ce1898846f215e0997d3a3e758e9d895d |
| kernels.py | 0fc56f795b8d9e77795e878c26d68be079f4746f6fb4938a62858a5292140dec |
| near_singular.py | 5b1b91d197f91d4ead7e899cff0062a851ad9b722ea28c0fc3b99beda8b26e23 |
| geometry.py | fcb890818a6f77b5bf178d3b3f2d771b03ff26b3981ccddd15bb99c8e9daf463 |
| multipole2d.py | 44e4ee6b9e9de899daaefae08fa210d408a974025a1480d9b868ec05e5309b7c |
| lifting_cylinder.py | e7a55df017a3858e802002dd4f99db936ed0a33235b886183dace27418a50f08 |
| lifting.py | 3fc5c057af3272111933c0cd6dc5b779348a9370186f532be8d29847ff294203 |

Combined SHA-256: `a66b390ec135ee19045f0678a9b0beefd8e84894ae394fcd6bd30e4f324a82fc`

### 2.3 测试覆盖确认

`tests/test_case_cylinder_stream.py` 含 33 条测试（含参数化），覆盖：
- 单层/双层面板梯度 vs 中心差分（12+ 组参数）
- 外部场求值器梯度 vs 势差分
- 外部场 vs 解析复势
- 内部解拒绝/边界点拒绝
- 物面速度法向性+匹配参考
- 快速阶梯验收（QUICK_LADDER = [32, 64]）
- 提交记录与文档一致性（含图件可重现性）
- 提交记录与当前模块指纹一致性（provenance gate）
- 近壁适用窗测量（含线性密度/方位角扫描）
- 守门拒绝（太近的点）

### 2.4 既有数值复核（基于已提交 results.json, N=128）

| 关键量 | got | ref（解析复势）| 误差 | <3% |
| --- | --- | --- | --- | --- |
| 路线 D 外部速度 $L_2$ | — | $dW/dz$ | 1.366e-07 | 是 |
| 路线 D 扰动势 $L_2$ | — | $Ua^2\cos\theta/r$ | 3.578e-07 | 是 |
| 路线 N 物面速度 $L_2$ | — | $2U|\sin\theta|$ | 1.008e-04 | 是 |
| 路线 N 外部速度 $L_2$ | — | $dW/dz$ | 1.532e-04 | 是 |
| 路线 N $C_p$ $L_2$ | — | $1-4\sin^2\theta$ | 2.849e-04 | 是 |
| 零法向流动 $\max|v\cdot n|/U$ | 6.025e-04 | 0 | 同上 | 是 |
| $C_p$ 极小值 | -2.999194 | -3.000000 | 8.060e-04 | 是 |
| 远场 $r=8a$ 速度 | — | $dW/dz$ | 5.591e-09 | 是 |
| 环量路线升力相对误差 | — | Kutta-Joukowski | 4.014e-04 | 是 |
| 驻点角误差 | — | $\sin^{-1}(\Gamma/4\pi Ua)$ | 0.006° | 是 |
| 路线 M 偶极矩 $A_1$ | — | 1.0 | 3.578e-07 | 是 |
| 路线 M 四极矩 $A_2$ | — | 1.0 | 1.048e-04 | 是 |

38 条既有判据全部 PASS，最差误差 2.849e-04（<< 3%）。

## ③ 待完成项（集群作业 cj3943）

作业完成后需：
1. `run_cluster fetch cj3943` 取回工件
2. 验证 `overall=PASS`、44 条判据、provenance 与当前指纹一致
3. 将 `results.json`、`report.md`、`case.json`、`figures/` 落盘到 worktree
4. 运行 `test_committed_record_was_rendered_from_the_current_case_module` 由 skip 转 pass
5. 运行 `test_committed_case_artifact_matches_the_committed_document` 确认图件可重现

## ④ 结论

- 算例模块功能实质已就绪，数值结果与解析复势基准误差远优于 3% 容差（最差 2.849e-04）
- 记录陈旧是本轮要解决的唯一欠账：cj3943 重生成后，44 条判据 + provenance + 5 张图件将全部就位
- 测试覆盖完整（33 条），含 provenance gate 确保记录不再落后于代码
- 所有关键量误差均 <3%，收敛阶二阶符合理论预期
