# m8580 — Panel-Mesh Geometry Validation: Watertightness, Orientation, Sphere-Area Convergence

**Repo**: TensorBEM · **Branch**: agent/m8580 · **Date**: 2026-11-09
**Task**: 面元几何校验单元测试补强（NEW:0731×qw38 通任务 A/B 真实增量——测试面/知识面硬化）

## 1. 背景

`tensorbem.geometry3d.TriangleMesh` 是 3D BEM 的几何基座：collocation 在三角形
质心，要求网格闭包（watertight）、绕向一致（外法向）。既有测试
（`tests/test_geometry3d.py`）只覆盖了构造器**拒绝**坏网格的负向路径，缺少
对**合法网格**的三项正向机器断言。本单补齐：

1. **水密性**：每条无向边恰被 2 个面元共享，闭流形网格 0 孤边，
   `E = 3F/2`，Euler 特征 `V − E + F = 2`；同一检查对"删一个面元"的负向
   对照必须检出恰好 3 条孤边（`V − E + F = 1`）。
2. **绕向/法向一致性**：相邻面元共享边逆指向（每条有向边 `(a→b)` 的逆
   `(b→a)` 必属于另一个面元）；外法向统计为正（球体 `n·c > 0` 逐面元、
   符号体积 > 0；椭球用 `n·∇(x²/a²+y²/b²+z²/c²) > 0`）。
3. **单位球面积收敛**：icosphere 离散面积 vs 解析 `4π`，预声明门槛
   1280 面元处相对误差 < 1%、随加密单调下降、误差比落在二阶带 (3, 5)。

## 2. 交付物

| 路径 | 内容 |
|---|---|
| `tests/test_panel_geometry_validation.py` | 16 个新 pytest（4 组参数化 + 负向对照 + 确定性） |
| `tools/m8580_panel_geometry_validation.py` | 数字/图程序化生成器（报告零手抄来源） |
| `tools/m8580_post_report.py` | 论文级报告构建+发布（数字全部注入自 results.json） |
| `artifacts/m8580_panel_geometry_validation/` | `results.json` + 3 张图 + `report.json` |

## 3. 结果（全部由 `tools/m8580_panel_geometry_validation.py` 程序化生成）

### 水密性（icosphere levels 0–3）

| 网格 | 边数 E | 恰 2 面共享 | 孤边 | >2 面 | V−E+F |
|---|---|---|---|---|---|
| icosphere(0) | 30 | 30 | 0 | 0 | 2 |
| icosphere(1) | 120 | 120 | 0 | 0 | 2 |
| icosphere(2) | 480 | 480 | 0 | 0 | 2 |
| icosphere(3) | 1920 | 1920 | 0 | 0 | 2 |
| icosphere(1)−1 面（负向对照） | 120 | 117 | **3** | 0 | 1 |

### 绕向/法向

levels 0–3 共 5100 条有向边：逆指向违规 **0**；`n·c > 0` 面元占比 **100%**
（level 3 最小值 0.99547）；符号体积全正（level 3: 4.152741）。

### 单位球面积收敛（vs 4π = 12.5663706144）

| level | 面元 F | 面积 A | 相对误差 | 误差比 |
|---|---|---|---|---|
| 0 | 20 | 9.5745413833 | 2.380822e-01 | — |
| 1 | 80 | 11.6659313917 | 7.165468e-02 | 3.3226 |
| 2 | 320 | 12.3298485952 | 1.882182e-02 | 3.8070 |
| 3 | 1280 | 12.5064927340 | **4.764930e-03** | 3.9501 |
| 4 | 5120 | 12.5513538801 | 1.194994e-03 | 3.9874 |

单调下降 ✓；1280 面元处 0.4765% < 1% 门槛 ✓；误差比 → 4（二阶 `O(h²)`）✓。

## 4. 验证

```
python -m pytest tests/test_panel_geometry_validation.py -v   # 16 passed
python -m pytest tests/test_geometry3d.py -q                  # 14 passed（既有面不回归）
python tools/m8580_panel_geometry_validation.py              # 重新生成 artifacts
```

## 5. 误差判定（关键量 vs 基准，逐项）

| 量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| 面积 @1280 面元 | 4π | 0.4765% | **是**（且 <1% 门槛） |
| 面积 @5120 面元 | 4π | 0.1195% | **是** |
| 体积 @1280 面元 | 4π/3 | 0.8606% | **是** |
| 水密性/绕向 | 组合恒等式（精确） | 0 违规 | 精确成立 |

## 6. 注意点

* 检查器直接操作原始 face 数组（不经过构造器），因此能对构造器会拒绝的
  开网格做孤边计数——这是负向对照可行的关键。
* 收敛表 level 4（5120 面元）仅用于记录渐近误差比，不在验收表内。
* 本单纯几何/测试面，无求解器运行；"场量"= 边关联组合量与法向场，
  比较曲线对解析 `4π`。
* 报告已发布（POST /tensor/reports → 200），scenario_key=
  `tensorbem_panel_geometry_validation`。
