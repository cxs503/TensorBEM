# 几何 VOF 界面追踪：三基准验证（索引）

能力：TensorFVM 的 PLIC 几何 VOF（`src/tensorfvm/vof_plic.py`）界面追踪。
算例：Zalesak (1979) 开槽圆盘刚体旋转 2π / 单涡拉伸-回转（Rider & Kothe 1995、LeVeque 1996）/
对角阶跃平流 L1 收敛阶。一阶 donor-cell 为同场同边界基线。

**机器裁定：全部通过**
（`verdict.all_pass = True`）。

| 判据 | 门限 | 实测 | 判定 |
|---|---:|---:|---|
| `diagonal_l1_order` | 0.95 | 1.1376122013510308 | **PASS** |
| `step_mass_residual` | 1e-12 | 1.0150610510858548e-15 | **PASS** |
| `vortex_mass_drift` | 0.001 | 6.207739532979034e-15 | **PASS** |
| `zalesak_area_drift` | 0.02 | 0.00015480394289826453 | **PASS** |

* 论文级报告（数字全部程序化注入）：[`VOF-INTERFACE-TRACKING.md`](../../src/tensorfvm/verification/artifacts/vof_interface_tracking/VOF-INTERFACE-TRACKING.md)
* 机器可读结果：[`vof_interface_tracking.json`](../../src/tensorfvm/verification/artifacts/vof_interface_tracking/vof_interface_tracking.json)
* 覆盖率原始输出：[`coverage.txt`](../../src/tensorfvm/verification/artifacts/vof_interface_tracking/coverage.txt)
* 驱动：`python -m tensorfvm.verification.vof_interface_tracking --output <dir>`
* 测试：`tests/test_vof_interface_tracking.py`

复现：

```
PYTHONPATH=src python -m tensorfvm.verification.vof_interface_tracking --output <dir>
PYTHONPATH=src python -m pytest tests/test_vof_interface_tracking.py -q
```

已知缺陷登记（本单发现，**未改动旧函数**）：`verification.vof_rotation.zalesak_disk`
切的是整条竖带（含圆心下方），解析面积 0.0412510 vs 经典
0.0559684，少 26 % 液体；本模块改用 `classic_slotted_disk`，
并以回归测试钉住差异。
