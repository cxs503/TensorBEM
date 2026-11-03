# 有限水深二维辐射问题的端到端验证（`finite_depth` 核接入 BEM）

**状态**：完成。`docs/finite_depth_2d.md` §7 局限 1（"未接入 BEM 求解器"）与局限 2（"无水动力系数精度声明"）
由此结清。**产物**：`docs/finite_depth_radiation/finite_depth_radiation.json`（程序生成）、
`fig_finite_depth_radiation.png`、`tests/test_finite_depth_radiation.py`（14 条）、
`src/tensorbem/benchmarks/finite_depth_radiation.py`。

---

## 1. 案例（物理 / 几何 / 参数）

Ursell（1949）潜没圆柱，与深水基准 `benchmarks/ursell_cylinder.py` 同一几何，只是加了平底：

| 量 | 值 |
|---|---|
| 圆柱半径 `a` | 1.0 m |
| 轴心潜深 `f` | 2.0 m（`a/f = 0.5`） |
| 无因次频率 `nu*a` | 0.6（`nu = omega^2/g`），`omega = 2.42569 rad/s` |
| 密度 `rho` | 1025 kg/m³，`g = 9.80665 m/s²` |
| 水深 `h` | 8 / 16 / 32 m（`nu*h = 4.8 / 9.6 / 19.2`） |
| 离散 | 64 个等长直线面元、常数源、质心配点 |
| 输出量 | 单位波峰长附加质量 `A`、阻尼 `B`（升沉） |

深水参考值 `A = 2545.1672 kg/m`、`B = 2718.1540 kg/(m·s)` 取自 Ursell 多重极解（截断解析系统），
**只**用于检验 `h -> inf` 极限，不进任何有限水深数字的构造。

## 2. 方法

`finite_depth` 核此前只做了核级验证（逐点极限、边界条件残差、奇点强度），没有进过组装矩阵。
本次改动全部在 `free_surface_solver`（`depth=None` 仍是深水路径，逐位不变）：

1. `_kernel_providers(depth)` 返回 `(势, 梯度)` 一对可调用对象，把"核选择"收敛到一个地方；
   深水核与有限水深核在同一符号约定下可互换（`finite_depth` 文档已声明，`tests/test_finite_depth.py` 已钉）。
2. `assemble_indirect_matrix` / `_evaluate_source_potential` / `_free_surface_only_normal_derivative` /
   `solve_radiation_indirect` / `radiation_reference` 增加 `depth` 关键字。鞍点结构不变：
   奇异的 `log r/(2*pi)` 部分仍逐面元解析积分（平面面元上 `(Q-P)·n = 0`，该核部分对角项为零），
   只把光滑余项换成有限水深核。
3. `finite_depth.green_function` 新增 `include_rankine` 开关（与深水核语义一致：减去 `log(r)/(2*pi)`），
   这正是组装需要的"光滑余项"，此前该模块没有这个入口。
4. **远场阻尼的常数**（m2274 明确留给调用方的那一步）。有限水深下辐射功率
   `P = 0.5*rho*g*c_g*|eta|^2`（单位波峰长），`eta = i*omega*phi(x,0)/g`，由此

   ```
   B = rho*omega^2*c_g*|a_w|^2 / (2*g*k0^2*N0hat^2),
   N0hat = N0/cosh(k0*h)^2 = (h/2)*(sech^2(k0*h) + tanh(k0*h)/(k0*h))
   ```

   远场幅值 `a_w = i*oint [phi0hat*q - phi*d(phi0hat*e^{-i k0 x})/dn] ds`，
   `phi0hat(z) = cosh(k0(z+h))/cosh(k0 h)`。`depth=None` 时该式逐项退化为 `omega*rho*|a_w|^2`。

## 3. 结果

（下表由 `docs/finite_depth_radiation/make_artifacts.py` 从 JSON 工件注入，非手抄。）

### 3.1 远场常数的深度无关性（不含任何 BEM 求解）

把**同一个深水势**喂给 `far_field_damping`，只改 `depth` 参数：

| h [m] | nu*h | 远场阻尼 | 与深水值相对偏差 |
|---|---|---|---|
| 50 | 30.0 | 2758.177419909637 | 1.09e-14 |
| 200 | 120.0 | 2758.177419909641 | 1.09e-14 |
| 800 | 480.0 | 2758.177419909637 | 1.09e-14 |

同一场、同一网格、同一速度，四个深度与深水值相差 `~1e-14`。**常数没有拟合、没有自由参数**：
`c_g`、模态模方、`k0` 三处任何一处写错都会在这里留下一个与 `h` 无关的固定偏移。

### 3.2 `h -> inf` 极限回到 Ursell 深水值（MFS，独立于面元法）

| h [m] | nu*h | A/MFS [kg/m] | 相对 Ursell 误差 | B/MFS [kg/(m·s)] | 相对 Ursell 误差 |
|---|---|---|---|---|---|
| 8 | 4.80 | 2653.535 | 4.26e-02 | 2809.017 | 3.34e-02 |
| 16 | 9.60 | 2563.424 | 7.17e-03 | 2728.229 | 3.71e-03 |
| 32 | 19.20 | 2547.712 | 1.00e-03 | 2713.179 | 1.83e-03 |

`nu*h = 4.8` 处的 4% 差是**真实有限水深效应**（浅水/阻塞使附加质量上升），不是误差：
随 `h` 增大它按 ~`1/h` 收敛到 Ursell 值。方法为基本解（MFS）：源点置于体内、无面元、无共享矩阵元。

### 3.3 近场/远场能量恒等式（压力积分 vs 出射波幅）

| 情形 | 近场 B | 远场 B | 面元法残差 | MFS 残差 |
|---|---|---|---|---|
| 深水 | 2787.237 | 2758.177 | 1.04e-02 | 8.04e-04 |
| h = 8 m | 2892.677 | 2862.532 | 1.04e-02 | 7.69e-04 |
| h = 16 m | 2807.795 | 2778.539 | 1.04e-02 | 8.51e-04 |
| h = 32 m | 2791.822 | 2762.544 | 1.05e-02 | 7.03e-04 |

残差**与深度无关**（1.04e-2 → 1.05e-2），且 MFS 光场的残差小一个量级（~8e-4）：
说明残差来自常数源的表示误差（面元法本身），恒等式在有限水深下**同样成立**。
这一点是本次判定的核心——若常数或 `c_g` 写错，残差会在有限水深列为深水列的若干倍。

### 3.4 面元收敛（h = 16 m，对 MFS 参考）

| 面元数 | A [kg/m] | B [kg/(m·s)] | A 相对误差 | B 相对误差 |
|---|---|---|---|---|
| 32 | 2648.291 | 2862.935 | 3.57e-02 | 6.09e-02 |
| 64 | 2608.311 | 2807.795 | 1.75e-02 | 2.92e-02 |

关联条件数 2.4（深水 64 面元同量级），线性系统相对残差 `< 1e-10`。

![finite depth radiation](finite_depth_radiation/fig_finite_depth_radiation.png)

## 4. 误差判定（逐项）

| 关键量 | 基准 | 实测相对误差 | 是否 < 3% |
|---|---|---|---|
| `h -> inf` 附加质量（nu*h=19.2, MFS） | Ursell 2545.1672 | 1.00e-03 | **是** |
| `h -> inf` 阻尼（nu*h=19.2, MFS） | Ursell 2718.1540 | 1.83e-03 | **是** |
| 远场常数（4 个深度 vs 深水公式） | 深水 `omega*rho*|a_w|^2` | 1.09e-14 | **是** |
| 能量恒等式（h=8/16/32） | 深水同网格残差 1.04e-2 | 1.04e-2 / 1.04e-2 / 1.05e-2（比值 1.00–1.01） | **是**（残差与深度无关） |
| 面元解 vs MFS（h=16, 64 面元） | MFS | A 1.75e-02、B 2.92e-02 | A 是；**B 否（2.9%）**——常数源表示误差，非有限水深问题；深水 64 面元同量级（A 1.7%、B 2.9% 对 Ursell） |
| 有限水深**独立文献系数** | 无 | — | **无基准可判**：未复现任何有限水深发表的 A/B 表，只做了极限 + 恒等式 + 独立表示法互证 |

最后一行是本次最诚实的一条：**有限水深区的绝对精度没有文献锚**。能声明的是"在 `h -> inf` 时回到
Ursell 深水值、在任意 `h` 下满足能量恒等式、且与 MFS 独立表示法一致到面元表示误差量级"，
不能声明"对任意 `nu*h` 的 A/B 有 3% 精度"。要关闭这一条，需要 Yu & Ursell（1961）/Wehausen &
Laitone 的有限水深表格，本次未做（无可靠数字化来源，拒绝手抄）。

## 5. 注意点 / 局限

1. **绕射与 3-D 未接入**。`solve_diffraction_indirect`、`free_surface3d` 仍是深水专用；
   有限水深绕射需要重做入射波（有限水深色散 `k0 tanh(k0 h) = nu`）与反射/透射能量平衡，超出本单。
2. **代价**：核是模态和，模数 `max(256, 24*nu*h)`。64 面元的有限水深组装约 13 s（深水 < 1 s），
   `nu*h = 19.2` 约 28 s。深水区（`nu*h > 100`）应继续用 `depth=None`，那是闭式且更快。
3. **数值细节（踩过的坑）**：`cosh(k0*h)` 在 `h = 800` 处平方即溢出；模态模方必须写成
   `(h/2)*(sech^2 + tanh/x)`、模态形状写成 `(e^{k0 z} + e^{-k0(z+2h)})/(1+e^{-2k0 h})`。
   两个改写在深水极限逐位复现 `exp(nu*z)`（`test_amplitude_coefficient_is_overflow_free`）。
4. **`include_rankine` 的语义**：有限水深核的奇点藏在 Kummer 尾项里，减掉 `log(r)/(2*pi)` 之后
   余项才可跨配点高斯积分；该减法在 `r -> 0` 处不做（回退 0），但实际上从不取到（高斯节点在面元内部）。
5. 常水深、无流、线性势流；变水深需自由面面元。
6. **顺手发现的仓内既有缺陷（非本单引入，未修）**：`src/tensorbem/benchmarks/hydrostatic_benchmarks.py:58`
   `from ..geometry3d import box_hull_mesh, spheroid_hull_mesh` 在 main 上不可导入
   （`geometry3d` 无这两个名字，疑似 m2228 提交时被门禁滤除），导致
   `tests/test_hydrostatic_verification.py` 在收集阶段 ImportError、整套 pytest 中断。
   本单验证时用 `--ignore=tests/test_hydrostatic_verification.py` 跑其余全部测试。
7. **第二处既有缺陷（未修）**：`tests/test_doc_claims.py` 需要的 `artifacts/ursell_doc_claims.json`
   在 main 上不存在（m2186 提交时被门禁滤除），3 条断言在收集后 FileNotFoundError。
   两者都与本单改动无关（`git log`/`git show --stat` 可核）。

## 6. 复算

```bash
PYTHONPATH=src pytest tests/test_finite_depth_radiation.py -q     # 14 passed, 48 s
PYTHONPATH=src python3 docs/finite_depth_radiation/make_artifacts.py   # 重生成 JSON + 图（~2.5 min）
PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_radiation  # 表格
```