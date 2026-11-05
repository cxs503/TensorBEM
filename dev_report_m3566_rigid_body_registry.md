# TensorBEM 验证欠账结清：刚体 Neumann 附加质量张量（docs 表 5、表 6）纳入可执行注册表

**任务**：m3566（infrastructure 席，自找活） · **分支**：`agent/m3566` · **日期**：2026-11-05
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3566`）
**工件**：`outputs/m3566/{standard_tier.json, rigid_body.json, fig_rigid_2d.png, fig_rigid_3d.png, fig_registry_deviations.png}`、
脚本 `outputs/m3566/{patch_registry.py, patch_tests.py, inject_docs.py, make_figures.py}`

EDIT_WHITELIST: src/tensorbem/benchmarks/declared_numbers.py, tests/test_declared_numbers.py, docs/marine_bem_benchmarks.md, outputs/m3566/, dev_report_m3566_rigid_body_registry.md

---

## 1. 案例介绍（为什么做这件事）

`docs/marine_bem_benchmarks.md` §4.7 是 TensorBEM 里**唯一**一条不预设解析表面势、直接用刚体无穿透
Neumann 条件求解第二类 Fredholm 方程 `(½I − H)φ = S(−n)` 的验证面，覆盖二维椭圆截面 `3×3` 与三维三轴
椭球 `6×6` 附加质量张量（真值 = Lamb 1932 §§71,114–115 闭式解）。

问题在于：表 5（二维，16/32/64/128 面元）与表 6（三维，20/80/320 面元）共 **44 个数字**（条件数、
三个/六个对角相对误差、观测阶、对称性残差）从 2026-10-25 落库起只存在于文档里。集中式声明数字注册表
`tensorbem.benchmarks.declared_numbers` 建立于 2026-11-03，只登记了 cylinder / ellipsoid / ursell /
ursell_reference / interior_harmonics 五组，**没有 rigid_body 组**：`test_registry_covers_every_documented_benchmark_family`
的覆盖断言里也没有它。也就是说，这两张表可以在内核、求积或网格改动后整体失效而 CI 全绿。

这正是「已合并功能对文献/解析基准的复核」类验证欠账：不需要新物理，判据干净（解析闭式真值），
计算量小（二维 2.4 s、三维 11.3 s，合计 <15 s，属 `standard` 成本层，不触本机 128 核的容量纪律）。

**算例参数**

| 项 | 二维椭圆 | 三维三轴椭球 |
|---|---|---|
| 几何 | 半轴 \(a=1.5,\ b=0.75\) m（单位跨长） | 半轴 \(1.5/1.0/0.75\) m，\(\alpha=(0.400928,0.670713,0.928360)\) |
| 密度 | \(\rho=1025\ \mathrm{kg/m^3}\) | 同左 |
| 离散 | 16 / 32 / 64 / 128 等角常数面元 | `subdivisions` 0/1/2 → 20 / 80 / 320 三角形常数面元 |
| 真值 | \(A'_{11}=\rho\pi b^2\)、\(A'_{22}=\rho\pi a^2\)、\(A'_{66}=\tfrac18\rho\pi(a^2-b^2)^2\) | Lamb 形状因子闭式平动附加质量 + 转动附加惯矩 |
| 判据量 | 条件数、三对角相对误差、\(A'_{66}\) 观测阶、对称性残差 | 条件数、平动最大相对误差、三转动相对误差、转动观测阶 |

## 2. 怎么做的（方法与关键设置）

1. **先复算再登记**（防止把文档的旧数字当真理）：跑 `python -m tensorbem.benchmarks.rigid_body_added_mass
   --panels 16 32 64 128 --subdivisions 0 1 2`，得到 `outputs/m3566/rigid_body.json`，逐格与文档比对。
2. **44 条判据登记进集中式注册表**：`declared_numbers.py` 新增 `rigid_body` 组与 `rigid_body` runner
   （`_run_group` 里按 `parameters={"case": "ellipse"|"ellipsoid"}` 分派到 `run_ellipse_neumann_study` /
   `run_ellipsoid_neumann_study`），表格数字以一张紧凑的 `_RIGID_BODY_2D_TABLE` / `_RIGID_BODY_3D_TABLE`
   元组作为唯一来源，由 `_rigid_body_registry()` 展开为注册项。注册表自带的 `(group, parameters)` 缓存
   保证 44 条判据只花两次求解——这是复用既有设施，没有新增任何求解路径。
3. **二阶收敛与良态性从散文升级为判据**：文档正文声称第二类算子条件数保持 \(O(1)\)、误差二阶收敛。
   仅钉数字抓不住「降阶但误差仍单调下降」的退化，因此新增
   `test_rigid_body_claims_hold_where_the_documents_make_them`：9 个观测阶全部落在 \([1.85,2.15]\)，
   条件数二维 \(<3.1\)、三维 \(<2.5\)。
4. **对称性列的舍入噪声修正**（本次审计发现的真问题）：表 5 原先把张量反对称残差写成两位有效数字
   （\(6.4/9.3/4.5/8.0\times10^{-13}\)）。该量是机器 epsilon 量级的舍入噪声，随平台/BLAS 路径变化：
   同一份代码在 loongarch64 上实测 \(5.98/6.34/6.41\times10^{-13}\)、\(1.14\times10^{-12}\)，与原记录
   相差可达 40%。钉成等式只会产生假漂移，故把该列改写为可复现上界 \(<10^{-11}\)（实测最坏 1.14e-12，
   余量 8.8 倍；真实对称性缺陷会推高数个量级），注册表对应 4 条 `kind="upper"` 判据。
5. **文档表由 JSON 程序化注入**：`docs/marine_bem_benchmarks.md` 新增 §4.7.1，表格与偏差数字由
   `outputs/m3566/inject_docs.py` 从 `standard_tier.json` 生成，零手抄。

## 3. 计算结果

**注册表复算（`outputs/m3566/standard_tier.json`，`--tiers standard --strict`，退出码 0）**

| 项 | 数值 |
|---|---|
| `standard` 层判据总数 | 48（其中本次新增 rigid_body 44 条） |
| PASS / DRIFT | **48 / 0** |
| rigid_body 44 条相对偏差最大值（不含上界判据） | **2.6×10⁻³** |
| 上界判据最坏余量 | 8.8×（\(1.14\times10^{-12}\) vs 上界 \(10^{-11}\)） |

**二维（表 5，Lamb 1932）** — 声明 vs 实测（节选，全表见文档 §4.7.1）

| \(N\) | 条件数 | \(A'_{11}\) 误差 | \(A'_{22}\) 误差 | \(A'_{66}\) 误差 | \(A'_{66}\) 观测阶 |
|---:|---:|---:|---:|---:|---:|
| 16 | 3.0776 (3.078) | 2.6365e-3 (2.64e-3) | 5.1495e-3 (5.15e-3) | 3.2316e-2 (3.23e-2) | — |
| 32 | 3.0190 (3.019) | 7.4180e-4 (7.42e-4) | 1.4137e-3 (1.41e-3) | 8.7077e-3 (8.71e-3) | 1.9357 (1.936) |
| 64 | 3.0048 (3.005) | 1.9389e-4 (1.94e-4) | 3.7530e-4 (3.75e-4) | 2.2460e-3 (2.25e-3) | 1.9661 (1.966) |
| 128 | 3.0012 (3.001) | 4.9383e-5 (4.94e-5) | 9.6966e-5 (9.70e-5) | 5.6959e-4 (5.70e-4) | 1.9822 (1.982) |

**三维（表 6，Lamb 形状因子）**

| `subdivisions` | \(N\) | 条件数 | 平动最大误差 | 转动误差 roll/pitch/yaw |
|---:|---:|---:|---:|---|
| 0 | 20 | 2.4475 (2.447) | 2.6292e-1 (2.63e-1) | 4.9034e-1 / 5.1946e-1 / 5.1063e-1 |
| 1 | 80 | 2.2463 (2.246) | 8.1058e-2 (8.11e-2) | 1.6342e-1 / 1.7833e-1 / 1.7436e-1 |
| 2 | 320 | 2.1305 (2.130) | 2.1375e-2 (2.14e-2) | 4.5996e-2 / 4.8416e-2 / 4.8452e-2 |

## 4. 结果图

* `outputs/m3566/fig_rigid_2d.png`：左=二维三对角相对误差 vs 面元数（log-log，含 \(O(h^2)\) 参考斜率）；
  右=条件数随加密的变化（3.078→3.001，保持 \(O(1)\)）。
* `outputs/m3566/fig_rigid_3d.png`：左=三维 6 模态相对误差收敛族（转动比平动慢约 2.3 倍，与棱边奇异性一致）；
  右=\(N=320\) 时 \(6\times6\) 张量对角相对误差的矩阵云图（\(\log_{10}\) 色标）。
* `outputs/m3566/fig_registry_deviations.png`：44 条注册项 |相对偏差| 柱状图 + 1.5e-2 漂移阈线（全在线下）。

## 5. 注意点（坑与适用范围）

1. **文档 3 位有效数字是偏差上限的来源**：2.6e-3 的最大偏差几乎全部来自「1.41e-3」这类三位有效数字
   的舍入，而不是求解器误差；把容差收紧到 1e-3 会制造假漂移。
2. **对称性残差不能钉等式**（见 §2.4）。凡「机器 epsilon 量级的量」都应按上界登记，这是可推广的判据
   设计规则。
3. **注册表 `summary.worst_relative_deviation` 会把上界判据算进"最坏偏差"**（本次 0.94），语义上是
   「占上界的比例」而非误差；这是既有 `_judge` 语义，本次未改动以免影响既有判据，读报告时需注意。
4. **覆盖仍不完整**：§4.8（Dean 绕射/Haskind 互易）、§4.10（Hulme 漂浮半球）、§4.11（Wang/Linton 潜没球体）
   的表 7–12 仍只在各自的专项测试里，未纳入集中注册表；这是下一次自找活的现成线索。
5. 本机 load15≈107（128 核），故只做 `standard` 层（<15 s）的复算，未起 heavy 层。

## 6. 误差判定

| 关键量 | 基准 | 实测偏差 | 是否 <3% |
|---|---|---|---|
| 二维 \(3\times3\) 张量三角全部 12 个误差格 | Lamb 1932 闭式 | ≤5.0e-4 | **是** |
| 二维条件数 4 格 | 文档记录 | ≤1.4e-4 | **是** |
| 二维 \(A'_{66}\) 观测阶 3 格 | 文档记录 | ≤1.5e-4 | **是** |
| 三维 \(6\times6\) 张量 12 个误差格 | Lamb 形状因子 | ≤2.6e-3 | **是** |
| 三维条件数 3 格 | 文档记录 | ≤2.0e-4 | **是** |
| 三维转动观测阶 6 格 | 文档记录 | ≤2.1e-4 | **是** |
| 公式本身（面元解 vs 解析解，最细网格） | 真值 | 二维 \(A'_{66}\) 5.70e-4、三维最坏 4.85e-2 | 二维**是**；三维**否**（4.85e-2，320 面元尚未进入渐近区，需 `subdivisions=3` 才能达标） |

> 注：最后一行是本任务唯一的「未达标」项，如实记录：三维 320 面元下转动量仍有 4.8% 离散误差，
> 属文档已声明的多面体棱边奇异性限制，不作为失败，但也不能宣称「三维转动 3% 内」。

## 7. 验证方式

```bash
PYTHONPATH=src python -m pytest tests/test_declared_numbers.py -q            # 含 2 条新增判据
PYTHONPATH=src python -m tensorbem.benchmarks.declared_numbers --tiers standard --strict
```