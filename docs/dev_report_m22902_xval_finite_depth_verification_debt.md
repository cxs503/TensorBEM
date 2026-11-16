# XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN — 验证欠账闭环（m22902）

**锚点**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN` · 仓 TensorBEM · 分支 `agent/m22902`
**日期**: 2026-11-16

## 1. 开工核对

- 总账：`slug=XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`，`status=open`，`acceptance=<15%` → 可做。
- 仓内现状（HEAD `50d39d61`）：`benchmarks/verified/capytaine_finite_depth/`
  - `capytaine_ref.json`：**12/12 格齐**（capytaine 参考已冻结，provenance 完整）；
  - `tensor_ladder.json`：**12 格占位，仅 1 格（h/a=3, ka=0.5）三级齐、1 格（h/a=3, ka=1.0）两级**，
    其余 10 格 `levels={}` 空；
  - `results.json`：`execution_status = "PARTIAL: 1 of 12 planned cells"`。
- 全量 12 格 TensorBEM 梯级已在飞（`cj4422`，m22843 递交，`--ladder-only --shards 6`，排队中）。
  本单**不重复**该在飞计算，只做轻量验证欠账闭环。

## 2. 本单实质增量（轻量，不压重算）

### 2.1 失效测试修复（defect_fix）

`tests/test_capytaine_finite_depth_crossval.py::test_frozen_ladder_reproduces_under_a_live_recompute`
在 PARTIAL 工件上**失效**：

```python
key, entry = sorted(ladder["cells"].items())[0]   # 取字典序第一格
level = min(int(name) for name in entry["levels"])  # -> ValueError: min() arg is an empty sequence
```

字典序第一格是 `h_over_a=10.0|ka=0.50`（`"10.0" < "3.0"`），该格 `levels={}` 空——
对**未计算格**做"活重算复现"不是对任何东西的复现，测试必然崩。

**修复**：改为取**首个已完成的格**（`levels` 非空）：

```python
key, entry = next((k, e) for k, e in sorted(ladder["cells"].items()) if e["levels"])
```

修复后：`17 passed, 6 xfailed`（原 `16 passed, 1 failed`）。6 个 `XFAIL` 是 PARTIAL 态
的诚实占位（梯级未外推、单格无深度趋势等），不覆盖。

### 2.2 深水负控求解器精度复核（falsified_hypothesis）

计划用**深度盲深水求解器**（`solve_radiation_3d`）作负控，证明有限水深求解器在
`k0h=1.5` 格上确实在探水深。本单用**独立解析锚**（Thorne/Wang/Linton 多极子，
`tensorbem.analytic.submerged_sphere_wave_multipole`）复核该负控的精度。

**关键陷阱（可复用）**：负控以**有限水深格的 `nu`**（`nu = k0 tanh(k0 h)`）调用深水
求解器，而深水色散 `k0 = nu`。若拿深水求解器在 `k0=0.4526` 的结果去对**解析解在
`k0=0.5`** 的值，会得到 ~20% 的"阻尼偏差"——**这是波数错配的伪影，不是缺陷**。

**同波数正确对比**（深水求解器 `k0=nu=0.4526` vs 解析解 `ka=0.4526`）：

| 面板 | A11 | B11 | A33 | B33 |
|---|---:|---:|---:|---:|
| 20 | −29.78% | −57.21% | −32.70% | −58.75% |
| 80 | −9.29% | −20.74% | −10.50% | −21.67% |
| 320 | −2.45% | −5.75% | −2.80% | −6.05% |
| 1280 | **−0.62%** | **−1.47%** | **−0.71%** | **−1.55%** |

**结论**：深水负控求解器在 1280 面板上对解析锚**全部 <2%**（A11 −0.62%、B11 −1.47%、
A33 −0.71%、B33 −1.55%），**收敛正确**。所谓"~20% 阻尼缺陷"是**波数错配伪影**，
被否定（falsified）。负控作为交叉验证的独立锚是**可信**的。

## 3. 误差判定（逐项，是否 <15% owner 线）

| 关键量 | 基准 | 误差 | <3%? | <15%? |
|---|---|---|---|---|
| 深水 A11（1280 面板） | 解析 2435.727 | −0.62% | ✅ | ✅ |
| 深水 B11（1280 面板） | 解析 567.363 | −1.47% | ✅ | ✅ |
| 深水 A33（1280 面板） | 解析 2732.955 | −0.71% | ✅ | ✅ |
| 深水 B33（1280 面板） | 解析 1257.212 | −1.55% | ✅ | ✅ |

**显式回答**：深水负控求解器对解析锚**全部 <3%**（最差 B33 −1.55%），且全部 <15%。
交叉验证的负控腿**验证通过**。

## 4. 注意点与局限

1. **全量 12 格交叉验证仍在飞**（`cj4422`）：本单只闭环了负控腿与失效测试，
   主比较（TB 有限水深 vs capy 有限水深）的 12 格外推判定待 `cj4422` 回填后由
   `make_report_m22474.py` 零改动复用。
2. **波数错配是负控复核的头号陷阱**：深水求解器与解析锚必须**同 `k0`** 对比，
   否则得到的是伪影不是偏差。已在本单文档固化。
3. **6 个 `XFAIL` 是 PARTIAL 态诚实占位**：不覆盖，待全量梯级回填后转真。

## 5. 复现

```bash
# 失效测试修复验证
PYTHONPATH=src python3 -m pytest tests/test_capytaine_finite_depth_crossval.py -q
#  -> 17 passed, 6 xfailed

# 深水负控同波数复核（轻量，<8 核 <10min）
PYTHONPATH=src python3 - <<'EOF'
from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import submerged_sphere_mesh
nu = 0.4525741268224332          # 有限水深格 h/a=3, ka=0.5 的 nu
an = submerged_sphere_wave_multipole(ka=nu, radius=1.0, submergence=1.5)
mesh = submerged_sphere_mesh(3, radius=1.0, submergence=1.5)
res = solve_radiation_3d(mesh, nu, modes=[0,2], center=(0,0,-1.5))
print((res.added_mass[0,0]-an.surge_added_mass)/an.surge_added_mass*100)  # -0.62%
EOF
```

## 6. 给下一次自找活的线索

1. **全量 12 格交叉验证回填**：`cj4422` 完成后跑 `make_report_m22474.py --publish`，
   把 6 个 `XFAIL` 转真，给出 12 格 <3%/<15% 的完整判定——这是本锚的最终收口。
2. **绕射问题交叉验证**：本锚仅覆盖辐射；绕射（含能量守恒）跨代码比对是自然续作。
3. **复杂船体有限水深**：球体验证通过后，Wigley III / KCS 船体的有限水深跨代码验证。
