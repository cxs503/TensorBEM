# CASE-BEM-LAPLACE-CYLINDER — 第 3 轮补充：远场多极矩验证（route M, m20567）

被补充的分支基线：`agent/m20029` 的算例（审计 m20100，L4）。
本轮不改求解器数值路径，只新增**远场内容**判据与一个可复用的分解工具。

## 1. 为什么补
审计 m20100 的全部判据都是**点值**比较（外部 64 探针的 L2/最大偏差、Cp、近壁窗口）。
点值对不说明远场结构对：外部扰动的衰减内容
$\phi'=\sum_n (A_n\cos n\theta+B_n\sin n\theta)/r^n$ 才是远场方法（远场附加质量/漂移力/辐射能量）
真正读取的量，而它此前在算例里只由 `far_field` 单点（$r=8a$）间接体现。

## 2. 新增内容
- `src/tensorbem/multipole2d.py`（新）：`fit_ring`（单环谐波最小二乘，含基条件数与样本不足校验）、
  `decompose_rings`（多环合成：矩的半径无关性、以「环均值 vs $\log r$ 斜率」分离二维源项）。
  约定 $\phi=c+m\log r+\sum_n r^{-n}(\cdot)$；`amplitude` 是幅值（丢符号），`cos_strength`/`sin_strength` 保留符号。
- 算例新增 route M：两个解析场交替检验——
  均匀来流（只有 $A_1=Ua^2$）与纯应变流 $u=\gamma x,\ v=-\gamma y$（只有 $A_2=\gamma a^4/2$），
  环 $r/a=1.5,2.5,4.0$，每环采样 $\max(64,2N)$ 点，阶 4。
- 新增 9 项判据（合计 29 项）：目标矩误差、目标矩收敛阶、谐波纯度、源项泄漏、环间半径无关性、
  拟合残差、镜像（$A_1\to-A_1$）负对照的判别比。

## 3. 实测（预算 N=128）
| 量 | 实测 | 阈值 | 判定 |
| --- | --- | --- | --- |
| 偶极 $A_1/(Ua^2)$ 相对误差 | 3.578e-07 | < 0.03 | PASS |
| 四极 $A_2/(\tfrac{\gamma}{2}a^4)$ 相对误差 | 1.048e-04 | < 0.03 | PASS |
| 目标矩收敛阶（偶极 / 四极） | [3.138, 3.069] / [2.111, 2.059] | ≥ 1.5 | PASS |
| 谐波纯度（非激发谐波 / 尺度） | 1.83e-14 | < 1e-9 | PASS |
| 源项泄漏（$\log r$ 斜率 / 尺度） | 5.52e-16 | < 1e-9 | PASS |
| 环间半径无关性 | 3.63e-13 | < 1e-9 | PASS |
| 拟合相对残差（阶 4 截断） | 3.41e-07 | < 1e-5 | PASS |
| 镜像负对照判别比 | 5.589e+06 | ≥ 10 | PASS |

## 4. 踩到的坑（诚实记录）
1. **幅值丢符号**：初版负对照用 `amplitude`（$=\sqrt{A^2+B^2}$），镜像数据得到 $+1.000000$，
   判别比形式上有 5.6e6× 但**证据是错的**。改为 `cos_strength`（保留符号）后镜像给出
   $-1.0000004$，`signed_error` 3.6e-07 才真正证明「矩带符号」。同时补了单元测试锁住该行为。
2. **环采样混叠**：每环 32 点时，正 $N$ 边形的面元谐波（$n=kN\pm1$）会混叠到低阶，
   N=32 时偶极被污染到 2.6e-5、N=128 时污染消失，读数看起来"越加密越超收敛"。
   采样数改为 $\max(64,2N)$ 后，偶极误差呈干净的 O(h^3) 而非虚假的机器精度。
3. 单环无法把二维源项与常数项分开（二者在环上都是常数），故必须 ≥2 个环。

## 5. 发布
- 平台报告 tensor_reports.id=18711（六段 + 2 内嵌图：multipole.png、field.png）
- 复现：`PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream --outdir benchmarks/verified/case_bem_laplace_cylinder --doc docs/CASE-BEM-LAPLACE-CYLINDER.md --check`（102 s，PASS，29 判据 0 失败）
- 测试：`PYTHONPATH=src python3 -m pytest tests/test_case_cylinder_stream.py tests/test_multipole2d.py -q` → 36 passed, 2 skipped（跳过为参数化落在对数奇异面板/面板本身，主值情形）
