# TensorBEM 观测阶口径统一：`kochin_hulme_convergence` 的 `orders` 字段改为对 h 的阶（m3280）

**任务**：m3280（tp-qw38-1 generalist 席，自找活） · **分支**：`agent/m3280` · **日期**：2026-11-04
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3280`，基线 `origin/main` @ 6025970）
**工件**：`outputs/m3280/study_s23.json`、`outputs/m3280/kochin_hulme_convergence.png`

EDIT_WHITELIST: src/tensorbem/benchmarks/kochin_hulme_convergence.py, tests/test_kochin_hulme_convergence.py, docs/marine_bem_benchmarks.md, outputs/m3280/, dev_report_m3280_orders_convention.md

## 1. 案例介绍（为什么做这件事）

m3264（三维自由液面声明数字判据化，67/67 PASS）的「注意点」第 4 条明确留了一条未结清欠账：

> 正文「$B'_{11}$ 对 $h$ 观测阶 1.62/1.88/1.96」与表 10 误差比自洽（$h\propto N^{-1/2}$），
> 而 `kochin_hulme_convergence.run_study` 的 `orders` 字段以 `log2(面元数比)` 为分母，
> 读出的是该值的一半（≈0.81/0.94）。……**两种口径的统一属另一单**。

即：文档（`docs/marine_bem_benchmarks.md` 表 10 与 m2609 专项段）引用的观测阶是**对面板尺寸 $h$ 的阶**，
而模块 `run_study` 输出的 `orders` 字段是**对面元数 $N$ 的阶**。三维中 $N\propto h^{-2}$，
4× 面元加密时两者相差**恰好因子 2**——同一份研究，代码输出 0.81/0.94，文档印 1.62/1.88，
读者按文档口径读代码工件会得出「收敛阶只有文档一半」的误判。本单即 m3264 点名的那一单：
把模块输出统一到文档口径，并把旧口径保留为独立字段，用回归测试钉死两者关系。

| 项 | 值 |
|---|---|
| 算例 | Hulme (1982) 漂浮半球，$Ka=1.0$，$a=1$ m，$\rho=1025$ kg/m³ |
| 离散 | 二十面体细分 icosphere，s=2→3（64→256 面元，边长减半） |
| 对比量 | $B'_{11}/B'_{33}$ 近场/远场阻尼相对 Hulme 参考的误差 |
| 判据 | ① `orders` = 对 $h$ 的阶（= $\log_2\lvert e_{lo}/e_{hi}\rvert$，边长减半）；② `orders_vs_panels` = 对 $N$ 的阶；③ 两者相差恰为 2（$10^{-12}$ 内）；④ $B'_{11}$ 的 $h$ 阶落在 (1.6, 2.2)（文档值 1.88） |

## 2. 怎么做的

1. **改模块**（`src/tensorbem/benchmarks/kochin_hulme_convergence.py`）：
   - `orders` 分母由 `log2(N_hi/N_lo)` 改为 `log2(√(N_hi/N_lo))`，即对 $h$ 的观测阶（文档口径）；
   - 新增 `orders_vs_panels` 保留旧公式（对 $N$ 的阶），溯源不丢；
   - 模块 docstring 增加「Order convention (m3280)」小节写明两口径与历史漂移；
   - `_print_table` 分两行打印 `orders (in h)` / `orders (in N)`。
2. **加回归钉桩**（`tests/test_kochin_hulme_convergence.py` 新增
   `test_orders_field_uses_h_convention`）：四个误差键逐一断言
   `orders == log2(|e_lo/e_hi|)`（$10^{-12}$ 内）且 `orders == 2·orders_vs_panels`（$10^{-12}$ 内），
   并钉 $B'_{11}$ 的 $h$ 阶 ∈ (1.6, 2.2)——若旧口径回潮（≈0.94）必红。
3. **文档同步**（`docs/marine_bem_benchmarks.md` m2609 段）：注明 `orders` 即对 $h$ 的阶、
   旧口径在 `orders_vs_panels`、两者相差因子 2 由回归测试钉死。
4. **复算工件**：`python -m tensorbem.benchmarks.kochin_hulme_convergence --subdivisions 2 3`
   生成 `outputs/m3280/study_s23.json` 与收敛图，零手抄。

## 3. 计算结果（程序化注入，`outputs/m3280/study_s23.json`）

| s | N | $B'_{11}$ 近场误差 | $B'_{33}$ 近场误差 | 近/远场比 $B'_{11}$ | orders(in h) $B'_{11}/B'_{33}$ 近/远 | orders(in N) |
|---:|---:|---:|---:|---:|---|---|
| 2 | 64 | −13.0492% | +4.5023% | 0.98874 | — | — |
| 3 | 256 | −3.5275% | +1.1812% | 0.99725 | 1.89 / 1.93 / 1.89 / 1.94 | 0.94 / 0.97 / 0.94 / 0.97 |

- 新 `orders`（对 $h$）：$B'_{11}$ 近场 **1.89**，与文档 1.88 一致（差 0.5%，BLAS 级抖动）；
  旧口径 0.94 与 m3264 记录的 0.94 逐位吻合，证明改动只动了分母、未动任何求解路径。
- 参考解自洽：$B'_{11}$ 近/远场自偏移 +0.0000%（< $10^{-4}$ 门），$B'_{33}$ +0.0215%。
- 求解耗时：s=2 0.5 s、s=3 8.4 s（本机 load15≈105，全程轻量，未起重算力）。

## 4. 注意点（坑 / 局限）

1. **只改口径，不动物理**：求解器、参考解、网格、求积全部未动；64/256 面元误差值与
   m3264/m3245 已钉数字逐位一致（−3.5275% 等），无回归。
2. **因子 2 的精确性**：`log2(√x) = 0.5·log2(x)` 在浮点下最后 ulp 可能差 1，
   测试容差取 $10^{-12}$（远大于 ulp、远小于口径差 0.94）。
3. **未做**：1024 面元 $B'_{11}'$ 专项（m3245 建议的独立重算单，本机 load15≥100 不满足容量纪律）；
   三维有限水深 John 核（Roadmap 功能缺口，超出单干范围）；全量套件未跑（重载窗口，
   只跑了直接相关的两个测试文件 + AST 检查）。
4. **适用范围**：`orders` 的 $h$ 口径假设 $N\propto h^{-2}$（三维面元）；若未来把该研究
   移植到二维（$N\propto h^{-1}$），分母应改为 `log2(N_hi/N_lo)` 本身——docstring 已写明。

## 5. 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---:|---|
| $B'_{11}$ 近场误差（256 面元） | Hulme (1982) | 3.5275% | ❌（离散误差，1024 面元 0.90% 已另证；本单不改物理） |
| $B'_{33}$ 近场误差（256 面元） | Hulme (1982) | 1.1812% | ✅ |
| $B'_{11}$ 近/远场能量恒等式 | 1.0 | 0.99725（偏差 0.275%） | ✅ |
| `orders` 字段 vs 文档 1.88 | 文档 | 1.89（偏差 0.5%） | ✅ |
| `orders` vs 2×`orders_vs_panels` | 恒等式 | 偏差 < $10^{-12}$ | ✅ |
| 参考解近/远场自洽 | 0 | 0.0000% / 0.0215% | ✅ |

**结论**：口径欠账结清——代码工件与文档从此同口径，且「口径回潮」有回归测试拦截；
物理精度本身（$B'_{11}$ 256 面元 3.53%）如实保留，未粉饰。

## 6. 验证记录

- `pytest tests/test_kochin_hulme_convergence.py` → **5 passed**（4 旧 + 1 新）
- `pytest tests/test_declared_numbers.py` → **12 passed, 3 skipped**（heavy 门，未触发）
- 两改动文件 AST 检查通过；`grep` 确认 `orders` 字段无其他消费方（仅本模块 + 两个测试文件）

## 7. 给下次自找活的线索

1. 1024 面元 $B'_{11}'$ 收敛专项（m3245 建议）：低载窗口（load15<100）可单干，
   判据现成（0.90% 门 + 本单钉的 $h$ 阶带）。
2. 三维有限水深 John 核（Roadmap 6）：功能缺口，建议派单而非自干。
3. `docs/finite_depth*.md`、`lifting_foil.md`、`wigley_wave_resistance.md` 等特性文档的
   数字尚未入 `declared_numbers` 注册表（m3085 扫描法可复用）。
