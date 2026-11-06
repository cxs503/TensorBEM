# m4741：静水恢复矩阵文档数字的机器化守卫 + 曲壁水线面第三条独立路径（交叉验证 m4750）

**任务**：m4741（platform 席自找活）· **仓**：TensorBEM · **分支**：`agent/m4741`
**类型**：验证欠账（先锚：`docs/doc_claim_audit/audit.json` 的 UNGUARDED 清单 + m3196 文档 §5 自陈欠账）
**平台报告**：id 5209（首次）、id 5453（m4883 重开落地，含曲壁对比图与偏差柱状图）
**工件**：`src/tensorbem/benchmarks/hydrostatic_restoring_pins.py`、
`tests/test_hydrostatic_restoring_doc_pins.py`（14 条）、
`docs/hydrostatic_restoring_waterplane_curvature.md`（新，415 个数字自动注入）、
`docs/figures/fig_restoring_curvature.png`、`docs/hydrostatic_restoring_doc_pins.json`（逐数台账）

---

## 1. 选题：两条账对上了

`docs/doc_claim_audit/audit.json` 把 `docs/hydrostatic_restoring_couplings.md`
判为 **UNGUARDED**（80 个数字无人读，且是 paper-level 文档）：它的数字由
`docs/m3196/outputs/restoring_couplings/make_artifacts.py` 注入，但这个脚本既不可
import、也没有测试；`tensorbem/hydrostatic.py` 改一个符号，文档不会红。
而该文档 §5 第 3 条自己记着一条欠账：「弧度曲壁船体的耦合项未用能量 Hessian 复核」。

于是本单做两件事：**把文档数字变成可执行判据**，并**为那条欠账补第三条独立取值路径**。

> **重开说明（m4883，PR#317 关闭后重启）**：本单成果从当前 `main` 重新落地。落地时
> 那条弧壁欠账已由 m4750（PR#320，`hydrostatic_curved_waterplane.py`：逐三角形裁剪的
> 能量 Hessian + 精确几何闭式、O(h²) 收敛）独立结清，因此本文的定位修正为
> **第三条不共享代码的路径**（Green 定理水线面环积分）与**文档数字的机器化**，
> 不再声称「首结欠账」。另外 m4750 的测试只在 docstring 里提到
> `hydrostatic_restoring_couplings.md`，按 audit 的文件名匹配规则足以把它判成 PINNED，
> 但**没有任何测试真的读它的数字**——正是 audit 文档警告的「洗白」。本单的 pins 模块
> 逐数解析该文档 78 个数字，把这个 PINNED 变成名副其实。

## 2. 做了什么

1. **`hydrostatic_restoring_pins.py`**（新增，~700 行）
   - 复算 m3196 主算例（驳船 2×1×1 m，LCF=1.30 m、TCF=0.45 m）：六个非零元的
     code / 矩形闭式解 / 能量 Hessian / **pre-fix 旧实现**（按形心惯矩解析重建，
     不依赖 `patch_hydrostatic.py`）+ 对称性、yaw 行列、全矩阵最大相对偏差；
   - 复算 LCF 扫描表（6 行 × 4 列）与 §6 误差表（7 行）+ 正文标量；
   - **从 markdown 里把每个数字解析回来**，容差按**打印精度**取
     （`2.010363e+04` → 量子 5e-4，容差 1.5×量子；多出的 50% 只吸收 `0.125`→`0.12`
     这类进位边界），所以判据失败一定是真数字动了，不是舍入勉强；
   - 新验证族：椭圆截面柱体（n=16/32/64，形心 (0,0) 与 (0.30,0.10)）与浮半球
     （sub=2/3，形心 (0,0) 与 (0.25,0.15)），每条 3 个独立取值路径：
     生产代码 / 能量 Hessian / **水线面环 Green 积分**（模块内自实现，不调用
     `compute_waterplane_properties`）。
2. **`docs/hydrostatic_restoring_waterplane_curvature.md`**：新文档，全部数字由
   `--write` 注入；中英混排但公式与数字零手抄。
3. **`tests/test_hydrostatic_restoring_doc_pins.py`**：14 条，含**两条负控**
   （改一位数字 → 必须报 DRIFT），以及「两文档被 audit 判为 PINNED」的一致性断言。

## 3. 结果

| 文档 | 解析到并逐数复算的数字 | 状态 |
|---|---|---|
| `hydrostatic_restoring_couplings.md` | 78 | **PASS** |
| `hydrostatic_restoring_waterplane_curvature.md` | 350 | **PASS** |

曲壁族逐案（分母为 $\max|C|$）：

| case | max rel dev vs energy Hessian | vs ring integration | 偏移构型耦合项非零 |
|---|---|---|---|
| `elliptic-16-centred` | 1.22e-09 | 2.96e-17 | False |
| `elliptic-16-offset` | 5.87e-09 | 1.48e-17 | True |
| `elliptic-32-centred` | 4.23e-09 | 1.16e-16 | False |
| `elliptic-32-offset` | 3.15e-09 | 5.80e-17 | True |
| `elliptic-64-centred` | 9.10e-09 | 8.85e-18 | False |
| `elliptic-64-offset` | 1.57e-08 | 5.77e-17 | True |
| `hemisphere-2-centred` | 1.29e-05 | 0.00e+00 | False |
| `hemisphere-2-offset` | 1.29e-05 | 2.96e-17 | True |
| `hemisphere-3-centred` | 6.32e-06 | 4.45e-18 | False |
| `hemisphere-3-offset` | 6.32e-06 | 5.80e-17 | True |

最差项 `hemisphere-2-centred` = 1.29e-05，比 3 % 判据小
2329 倍。形心在原点构型的三个耦合项
全部 < $10^{-6}\max|C|$，偏移构型全部非零——耦合项确实由水线面一次矩驱动。

**覆盖审计变化**（`docs/doc_claim_audit/audit.json`，本单在当前 `main` 上重算）：
`hydrostatic_restoring_waterplane_curvature.md` 新增并判为 **PINNED**（350 个表格数字
逐数解析），`hydrostatic_restoring_couplings.md` 保持 PINNED 但现在由**真解析**支撑
（m4750 只贡献文件名匹配）；baseline 无需放松（`unguarded` 22 项、`unguarded_paper_level`
8 项均未增长），全仓 guarded number fraction = 82.9 %。

## 4. 注意点（坑 / 局限）

1. **NC 语义**：能量 Hessian 是物理判据；环积分与生产代码代数等价（同一组多边形矩 +
   平行轴），它对 code 的一致（~1e-17）只证明记账没写错，**不构成物理验证**。文档 §2 已写清。
2. **半球族残差地板更高**（~1e-5 vs 椭圆族 ~1e-9）：多曲率三角面上的裁剪浮点噪声 +
   $h=2\times10^{-4}$ 差分截断不再被平面壁的精确代数抵消。随离散下降，是方法噪声不是缺陷。
3. **环顶点必须按质心极角排序**：两个 `*_hull_mesh` 的 $z=0$ 顶点数组不是环序，
   直接连边会自交、面积算错（本单实测过）。
4. **未做**：三维自由液面（浮球辐射/衍射、Hulme、Wang–Linton）的耦合项复核；
   外飘（flare）水线面仍按 $z=0$ 截面处理；半球 subdivisions 封顶 3（能量 Hessian 的
   24 次重积分代价）。
5. **仓库清单里其余 11 个 paper-level UNGUARDED 文档仍在**（`kkr_causality_band_study.md`、
   `ursell_verification_debt_2026-11-02.md` 等），本单只吃掉一个；照 `pins` 模块的骨架
   可以逐个搬（解析 + 复算 + 负控三段式）。

## 5. 复现

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.hydrostatic_restoring_pins --write   # 重生成文档/图/台账
PYTHONPATH=src python3 -m tensorbem.benchmarks.hydrostatic_restoring_pins --check   # 漂移即非零退出
PYTHONPATH=src python3 -m pytest tests/test_hydrostatic_restoring_doc_pins.py -q     # 14 passed
```

测试实测：`tests/test_hydrostatic_restoring_doc_pins.py` 14 passed in 22.6 s；
回归 `test_doc_claim_audit.py + test_hydrostatic_restoring_couplings.py +
test_hydrostatic_convergence.py + test_doc_claims.py` 56 passed in 8.5 s。

**平台报告 id**：5209（scenario_key `tensorbem-hydrostatic-restoring-doc-pins`）。
