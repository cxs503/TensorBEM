# XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE — 验证欠账根因定位

## 结论（一句话）

TensorBEM 的 r2 全 6×6 辐射矩阵对 capytaine 交叉验证 **FAIL**（verdict=FAIL）的根因
不是求解器精度缺陷，而是基准网格生成器 `floating_sphere_hull_mesh` 在自由面
（z=0）处产生**退化水线 sliver 面板**（Sutherland-Hodgman 裁剪 icosphere 所致）。

## 证据链

### 1. 同一物理体、两种网格，收敛行为不同

`hemisphere_r1`（`hemisphere_hull_mesh`，半径 1、平底在 z=0）与
`sphere_r1_z-0.5`（`floating_sphere_hull_mesh`，半径 1、球心 z=-0.5）是**同一
物理体**（半径 1 的半球，平底在 z=0），仅网格构造不同。r2 收敛数据：

| 网格 | level | 面板数 | worst_diagonal |
|------|-------|--------|----------------|
| hemisphere (clean) | 4 | 1024 | **3.3%**（入 5% 带） |
| floating_sphere (clipped) | 3 | 1026 | **6.5%**（出 5% 带） |

### 2. 网格质量诊断（`diagnose_mesh_quality`）

| 网格 | max_AR | panels_above_5 | min_angle |
|------|--------|----------------|-----------|
| hemisphere_hull_mesh(3) | **1.41** | **0** | **45.3°** |
| floating_sphere_hull_mesh(3) | **7.35** | **16** | **4.7°** |

`floating_sphere_hull_mesh` 在 level 3 有 **16 个 max_AR>5、min_angle≈4.7°** 的
退化面板，全部位于水线（含 z=0 顶点）——即裁剪产生的 sliver。

### 3. 失败分类

r2 交叉验证失败分两类：
- **对角带违规**（~5-7%，均在 nu=2 高频）：由上述水线 sliver 网格质量导致；
  同一物理体用干净网格收敛到 3.3%（入带）。
- **不变量零违规**（surge-pitch/sway-roll 耦合，on-center 球）：两代码均给
  ~10-14（≈1% 物理尺度），是**平板离散化**的固有产物（`(r-centre)×n` 在平板
  上非严格零），非 TensorBEM 特有。

## 处置

- 新增 `tests/test_mesh_quality.py` 三个回归测试，**钉住**该缺陷：
  - `test_floating_sphere_hull_mesh_reports_waterline_slivers`：诊断必须报告
    max_AR>5、min_angle<30°；
  - `test_floating_sphere_slivers_are_on_the_waterline`：退化面板必须位于水线；
  - `test_hemisphere_hull_mesh_is_clean_control`：干净网格（对照）必须无 sliver。
- 网格修复（使 `floating_sphere_hull_mesh` 生成干净球冠）为**后续项**：需重跑
  BEM 交叉验证确认对角误差入带，属重计算，本单（容量纪律）未执行。

## 适用范围

本结论适用于 r2 交叉验证锚点的浮球/半球算例（半径 1、球心 z∈(-1,0)）。
不改变求解器本身；`hemisphere_hull_mesh` 的干净网格已证明求解器对角精度
（3.3%<5%）成立。
