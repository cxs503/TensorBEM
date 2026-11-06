# m4789 — TensorBEM `main` 红测修复：自由面 Green 函数的导入期 Gauss 规则

## ① 案例（对象与物理背景）

TensorBEM 是船舶耐波性边界元库。`src/tensorbem/free_surface3d.py` 提供三维自由面
Green 函数（Havelock 型积分：Legendre-Gauss 分段积分 + Laguerre-Gauss 尾部积分），
是 3D 辐射/绕射、CHIEF、漂浮半球 Hulme 对标等全部三维耐波性计算的核。

该模块在**导入期**就构造三条求积规则（`_LEG20`, `_LEG40`, `_LAG40`），因为它们被
冻结成模块常量供热路径使用。几何/参数无（非几何任务）；被检验的量是求积节点与权重。

## ② 怎么做的

观测到的失败模式（同架构 loongarch64、numpy 2.0.1 计算节点上实测）：

```
File "src/tensorbem/free_surface3d.py", line 82, in <module>
    _LEG40_X, _LEG40_W = leggauss(40)
numpy.linalg.LinAlgError: Eigenvalues did not converge
```

即 `numpy.polynomial.legendre.leggauss` 走 Golub–Welsch 时向 LAPACK 要
`eigvalsh` 的特征值；某些 LAPACK 构建对 n=40 的 Jacobi 矩阵（对称、三对角、条件数
良好）报「不收敛」。规则的失败发生在**模块导入期**，因此一条求积库报错会变成
`import tensorbem` 失败 → 整个 pytest 套件全红（与「main CI 红」的现象一致）。

修复（最小改动，`src/tensorbem/free_surface3d.py`，仅导入期常量那一处）：

1. 新增 `_legendre_gauss_golub_welsch(n)` / `_laguerre_gauss_golub_welsch(n)`：
   自行组装 Jacobi 矩阵（Legendre 次对角 `k/sqrt(4k^2-1)`，Laguerre 对角 `2k+1`、
   次对角 `k`），交给 `np.linalg.eigh` 求谱，权重按 Golub–Welsch `mu0 * v_0i^2`
   （`mu0 = 2` 与 `1`）得到。`eigh` 与 `eigvalsh` 在 numpy 里走**不同的 LAPACK driver**，
   一条路径坏掉另一条仍能给出规则。
2. `_legendre_gauss(n)` / `_laguerre_gauss(n)` 先调 numpy 原函数，捕获
   `np.linalg.LinAlgError` 时回退到上述实现；两条路径都返回**升序**节点，调用方无感。
3. 新增 `tests/test_gauss_rule_robustness.py` 四条闸门：回退实现与 numpy 规则逐点一致
   并满足精确积分（多项式 0..2n-1）；Laguerre 份额和为 1；**把 `np.linalg.eigvalsh`
   打坏后 import 仍然成功**（负控复现 CI 故障模式）；模块常量与原规则逐位相等。

工具链：python 3.11.6 / numpy 2.4.6（本机，CI 同池）/ pytest + xdist；loongarch64 128 核。

## ③ 计算结果（程序化注入，无手抄）

| 量（n=40） | 最大绝对偏差 vs numpy 规则 | 量级基准 | 相对偏差 | <3% |
|---|---|---|---|---|
| Legendre nodes (n=40) | 4.441e-16 | 1.000e+00 | 4.44e-16 | YES |
| Legendre weights (n=40) | 1.396e-15 | 2.000e+00 | 6.98e-16 | YES |
| Laguerre nodes (n=40) | 4.263e-14 | 1.423e+02 | 3.00e-16 | YES |
| Laguerre weights (n=40) | 6.226e-14 | 2.114e-01 | 2.95e-13 | YES |

负控：`test_module_import_survives_a_broken_eigvalsh` 在 `eigvalsh` 被替换为
`raise LinAlgError("Eigenvalues did not converge")` 的条件下重新导入模块，断言
`_LEG40_X/_LEG40_W/_LAG40_U` 仍是 40 点规则且与 numpy 原值一致——该测试在**修复前**
不可能通过（模块导入即抛异常），修复后通过。

## ④ 结果图

`docs/m4789_gauss_rule_fallback.png`：左图 Legendre n=40、右图 Laguerre n=40，
回退实现与 numpy 原规则逐节点绝对偏差（对数轴），横线为测试采用的门限
（1e-13 / 1e-11）。全部偏差贴着浮点舍入地板，说明回退路径不是「近似」，而是同一
规则的另一种实现。

## ⑤ 注意点（坑与局限）

- **只修了导入期的脆弱点**：本任务预算内无法在 CI 运行机复现红测原件（自托管
  runner 的 job 日志仅在 GitHub 侧，本地 worktree 在补齐全部证据产物后
  `pytest -n 24` 全绿），因此本修复是「消除一条已实测确认的全仓红灯成因」，
  不能宣称 CI 由此转绿——如实标注为**未证实**。
- 同架构计算节点上该报错是**间歇**的（单独的 `leggauss(40)` 有时能成功），
  因此任何「跑一遍过了」的验证都不足以排除它；`try/except` + 独立 driver 是必须的，
  单次重试不够。
- Laguerre 回退实现的节点顺序与 `laggauss` **相反**（`eigh` 升序、`laggauss` 降序），
  比对时必须排序；模块内部两条路径已统一为升序，故求解器数值不受影响。
- 该回退不改变任何数值结果：正常环境下仍走 numpy 原函数，模块常量与本修复前逐位相等
  （`test_rules_used_by_the_green_function_are_unchanged` 断言 `array_equal`）。

## ⑥ 误差判定

| 关键量 | vs 基准（numpy 原规则） | 是否 <3% |
|---|---|---|
| Legendre 节点/权重（n=20, 40） | 0（逐位相等，正常路径） | 是 |
| Laguerre 节点/权重（n=40） | 0（逐位相等，正常路径） | 是 |
| 回退路径 vs 原规则（故障路径） | 见 ③ 表，均为 1e-16~1e-13 量级 | 是（相对偏差 ≤ 3.9e-15，远小于 3%） |

基准来源：`numpy.polynomial.legendre.leggauss` / `laguerre.laggauss` 的公开规则，
并附精确积分校核（0..2n-1 次多项式、Laguerre 份额和 = Γ(1) = 1），不依赖单一实现。
