# 刚体质量矩阵的「无惯量张量」默认值审计（锚点 CASE-BEM-RIGID-BODY-MASS）

## 发现（缺陷）

`hydrostatic.compute_mass_matrix(mass, cg, inertia=None)` 的历史回退是
`I_cg = m*(0.4*|r_g|)^2 I3`，**`|r_g|` 是坐标原点到质心的距离**——于是同一刚体的惯量成了
原点放在哪里的函数。实测（`benchmarks/rigid_body_mass_inertia_default.py`）：
同一体在 x 偏移 0 → 1000 m 上，回退惯量变化 **111112 倍**；对锚点用的椭球
(60,12,8) m、m=1e6 kg，回退值与闭式值差 **-99.9%**，roll 固有频率因此被高估 **28.5×**
（pitch 134×、yaw 137×）。`rao.py` / `ice_rao.py` 在调用方不传 `inertia` 时正是走这条路，
所以缺陷会静默进入每一次「未显式给惯量」的 RAO。

## 修复

1. `rigid_body_mass`：新增 `equivalent_half_ellipsoid` /
   `equivalent_inertia_from_hydrostatics` / `waterplane_central_moments`——
   用**排水体积 + 水线面二阶矩**重建等效半椭球，惯量由几何导出，
   因而与原点无关（换原点不变量实测 2.4e-11）。
2. `hydrostatic.compute_mass_matrix`：新增 `radius_of_gyration`（显式且不变的回退）；
   两者都缺省时保留历史回退但发 `RuntimeWarning`（不静默）。
3. 新入口 `hydrostatic.mass_matrix_from_hull_geometry`，`rao.py` 与 `ice_rao.py`
   在 `inertia=None` 时改走它；水线面非椭圆（`pi*a*b/A != 1`）时额外告警。

## 误差判定（vs 闭式）

| 量 | 基准 | 误差 | <3% |
|---|---|---|---|
| 等效惯量（球面体族 60:12:8，sub=4） | 半椭球闭式 | roll -0.24% / pitch -0.16% / yaw -0.16% | 是 |
| 同上 sub=2（未收敛档） | 同上 | -3.70% / -2.59% / -2.54% | 否（收敛阶梯） |
| hull 几何装配 vs 显式闭式张量装配 | 闭式 | 最大入口 0.19% | 是 |
| 换原点不变量 | 同体 I_cg | 2.4e-11 | 是 |
| 矩形驳船（20×8×3 m，声明为近似） | 长方体闭式 | -11.8% / -18.5% / -20.0% | **否**（诚实声明：二阶矩不足以定形体惯量，此域需显式张量） |
| 历史回退（锚点椭球） | 闭式 | -99.9% | 否（缺陷，已被 RAO 链弃用） |

## 注意点

- `waterplane_moments` 是**关于水线面形心**的二阶矩（实测：平移 (7,-4) m 不变），
  `compute_hydrostatic_restoring_matrix` 文档里「about the coordinate origin」的措辞是教材记号，
  不是返回值的约定；已在 `waterplane_central_moments` 的 docstring 与测试中钉住。
- 二阶矩只有 2 个数，无法唯一决定实体惯量；盒形体的 ~20% 偏差是该近似的固有上限，
  故以 `waterplane_area_ratio` 与其告警显式暴露，而不是假装精确。
- 平行轴定理的张量形式必须在 `C - P` 方向上加（本单开发中曾两次把符号/参考点写错，
  由 `tests/test_rigid_body_mass_equivalent_inertia.py` 与基准的换原点检查钉住）。

产物：`default_inertia_audit.json`、`inertia_default_audit.png`、基准模块
`src/tensorbem/benchmarks/rigid_body_mass_inertia_default.py`、测试 10 项。
