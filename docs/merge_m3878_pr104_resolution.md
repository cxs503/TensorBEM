# PR #104 治愈记录：`agent/m2930` → `main` 三方合并

**任务** m3878（heal，TensorBEM）· **PR** #104 · **分支** `agent/m2930`
（PR head `ed248e3`，本地旧镜像 `71c2a5f`，两棵树的 blob 完全相同）· **main** `ecba798`
**合并方式** `git merge --no-ff`，人工逐文件解冲（禁用机械 overlay）

---

## 1. 冲突面（merge-base = `29236f0`，两侧各 78 / 1 提交）

| 文件 | 类型 | main 侧 | m2930 侧 |
|---|---|---|---|
| `src/tensorbem/finite_depth3d.py` | **add/add** | 465 行，John **级数解**（eigenseries），含 `bessel_k0/j0/y0/h0`、`green_gradient_z_3d`、`deep_water_limit_error`，被 `tensorbem/__init__.py` 导入 | 488 行，**波数积分解**（kernel-subtraction），API 为 `green_function_finite_depth_3d` / `_z_derivative_3d` / `pole_data` / `dispersion_derivative` |
| `tests/test_finite_depth3d.py` | **add/add** | 305 行，26 道门（含深水极限、浅水极限、远场出射、Bessel 对标 SciPy） | 137 行，19 测试 + 1 个 `xfail(strict=True)` 的深水极限红门（m2930 的核心证据） |
| `docs/finite_depth3d_offaxis_diagnosis.md` | 纯新增 | — | 268 行离轴失效诊断（自动合并，无冲突） |
| `src/tensorbem/benchmarks/finite_depth3d_verification.py` | 纯新增 | — | 224 行验证 harness（自动合并，无冲突） |

冲突根源：**同一个模块路径上两条独立演化的物理实现**。m2930 从 `b408d0d` 分支出来，main 侧
经 m3015（`04872ad`）→ m3409（`b462190`/`54a2f85`/`867fdbb`）把该核重写成级数解并加了
对数缩放与 11 道 regime 门。二者 API 不兼容，不可能文本并集，只能**语义并集**：一条占
canonical 路径，另一条**原样保留**在并列路径上，任何一侧的内容（代码/测试/文档）零删除。

## 2. 逐文件解法

1. **`src/tensorbem/finite_depth3d.py` ← main 侧原文**（一字未改）。
   canonical 判据不是"谁新"，而是可执行约束：`tensorbem/__init__.py` 第 138 行按级数解
   `import`；main 侧有 26+11 道门与已发表的数字（`docs/finite_depth3d.md`）；而 m2930 侧
   文件自带 `STATUS: UNVERIFIED -- do not use in a solver` 横幅。
2. **`src/tensorbem/finite_depth3d_quadrature.py` ← m2930 侧原文**，仅加一段
   *Provenance* 头（说明来源、为何并列保留、不被 `tensorbem` 导入）。包含 m2930 的实质
   改动：`_wave_integral` 面板宽度 `bump/3.0 → bump/6.0`、`0.25*k0 → 0.1*k0`。
3. **`tests/test_finite_depth3d.py` ← main 侧 26 门原文**，另追加 §17 两道门（见 §4）。
4. **`tests/test_finite_depth3d_quadrature.py` ← m2930 侧 19+1 测试原文**，仅把 import 指向
   并列模块 `tensorbem.finite_depth3d_quadrature`，并在 `xfail` 理由里补本次复测证据。
5. **`benchmarks/finite_depth3d_verification.py`**：import 改指并列模块（它写的就是积分解
   API），否则虽然自动合并成功、一旦运行就 ImportError——**自动合并"成功"≠可用**。

## 3. 合并期实测（`docs/m3878/kernel_merge_ledger.json`，图 `fig_m3878_kernel_merge.png`）

参考基准 = `tensorbem.free_surface3d.green_function_3d`（深水 Kelvin 核）。

**(a) 离轴深水极限，两层核各自 vs 水深（`nu=0.6`, 源 `(0,0,-1)`）**

| h [m] | 级数解（canonical, main） | 积分解（m2930） |
|---:|---:|---:|
| 8 | 2.200e-3 | 3.198e+0 |
| 20 | 6.172e-5 | 3.197e+0 |
| 60 | 1.840e-6 | 3.197e+0 |
| 200 | 4.670e-8 | 3.197e+0 |
| 800 | 7.0e-10 | 3.197e+0 |

级数解随水深干净收敛（h=800 时 7e-10），m2930 报告的离轴失效在 main 线上**已经不存在**；
积分解误差随 h 恒定 ≈3.2（**系统性**，不是分辨率）。这也解释了为何两条线必须都留下：一个是
可用核，一个是"反例账本"。

**(b) m2930 面板加密是否真的起作用**（同一 m2930 代码，`b408d0d` 原版 vs 加密版）

| h | 加密前后最大差 `max|G_post - G_pre|` | 深水极限相对误差（加密后） |
|---:|---:|---|
| 8 | 6.8e-12 | 0.347 / 3.198 / 2.278e-3 |
| 20 | 1.0e-11 | 0.345 / 3.197 / 6.4e-5 |
| 60 | 1.1e-11 | 0.345 / 3.197 / 1.9e-6 |

结论（诚实的负面证据）：加密只改变核 `≤1.1e-11`，深水极限误差**一位不变** —— 积分早已收敛，
m2930 提交说明里的假设"分辨率不足"在实测下**不成立**，残余误差是公式/正则化层面的系统性偏差。
这条事实以注释形式写进并列测试的 `xfail` 理由，不粉饰。

**(c) 合并期新发现的未覆盖区域：源轴 R=0**

| R [m] | 级数解相对误差 | 积分解相对误差 |
|---:|---:|---:|
| 0 | 2.94e+1（单点批 2.4e+2，见注） | 6.43e-5 |
| 1e-4 | 3.19e-2 | 6.43e-5 |
| 1e-3 | 1.03e-4 | 6.49e-5 |
| 1e-2 | 6.43e-5 | 1.32e-4 |
| 0.5 | 7.09e-5 | 1.6e-1 |
| 2.0 | 1.14e-4 | 1.5e+0 |

`R→0` 时级数解的消散模求和 `Σ_n K_0(μ_n R)` 对数发散、被截断，`G_fd` 出现虚假实部
（单点 `-4.434e1` vs 参考 `1.136e-1`）；`R ≥ 1e-3` 即回到 1e-4。**注**：该误差依赖同批
其他点决定的模数（批内 `R_min` 越小模数越多），故批次不同数值在 3e1~2.4e2 间变化 —— 这也是
它一直没被 main 的"全局范数"门发现的原因。已在 §17 注册为 `xfail(strict=True)` + 一道
`R ≥ 1e-3` 的通过门，修复方向（闭式对数减除）写入理由，留给后续核任务，本次不越权改核。

## 4. 测试证据

```
# main 基线（origin/main ecba798，PYTHONPATH=src）
tests/test_finite_depth3d.py + test_finite_depth3d_regime.py + test_finite_depth.py
+ test_free_surface3d.py + test_free_surface_kernel.py            -> 66 passed

# 合并结果
同一 5 文件 + tests/test_finite_depth3d_quadrature.py            -> 86 passed, 2 xfailed
  （= 66 + m2930 的 18 passed + §17 的两道新门；2 个 xfail 均为 xfail(strict=True)，
    分别记录 m2930 离轴失效与 main 的 R=0 缺陷，任何一侧被修好都会 XPASS 变红）
```

## 5. 注意点 / 适用范围

* **机械 overlay 的教训**：两个文件都对 main 是 add/add 时，`--theirs` 会让 `__init__.py`
  的 `import` 崩、26 道门连同 main 已发表数字一起消失；`--ours` 会删掉 m2930 的全部证据。
  可用的判据只有一条：**哪一侧被包内其他代码引用/测试钉住**。
* 自动合并成功的 `benchmarks/finite_depth3d_verification.py` 其实是**断的**（import 指向
  不存在的级数解符号），必须人工复核新增文件的可运行性，"无冲突"不等于"正确"。
* 并列保留的积分模块**不进包导出**、保留 UNVERIFIED 横幅，避免被误用于求解器。
* 本记录只覆盖核层面；3-D 有限水深 BEM 求解器（面板积分/层跳变）仍未实现，不在本单范围。
## 6. 交付 SHA 与推回状态

| 项 | SHA |
|---|---|
| main（first parent） | `ecba798` |
| PR #104 head（second parent，远端旧值） | `ed248e3`（本地镜像 `71c2a5f`，树相同） |
| **合并提交** | `65e4cbf` |
| 本分支最终 tip（合并 + 报告脚本） | `d38199f` |

沙箱拒绝推送与本地 ref 改写命令，故远端 `agent/m2930` 仍为 `ed248e3`。`ed248e3` 是合并提交的
父提交，因此平台侧把 `d38199f` 推到 `agent/m2930` 为**快进（无需 force）**，PR #104 随即变干净，
`merge_governor` 下一周期可复验。
