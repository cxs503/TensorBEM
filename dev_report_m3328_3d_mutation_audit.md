# TensorBEM 三维核的变异审计：三维验证套件到底能不能发现缺陷？

**任务**：m3328（paper 席，自找活） · **分支**：`agent/m3328` · **日期**：2026-11-05
**仓**：TensorBEM · 基线 `6025970`（main 合并点）
**EDIT_WHITELIST**: src/tensorbem/benchmarks/mutation_audit.py, tests/test_mutation_gaps_3d.py,
outputs/m3328/, docs/m3328/, dev_report_m3328_3d_mutation_audit.md

---

## 1. 案例介绍（为什么做这件事）

TensorBEM 的三维部分（`kernels3d` 三角形面元核 + `free_surface3d` 深水 Kelvin 核 +
`finite_depth3d` 有限水深 John 本征级数核）已有 39 个专门的测试用例、几百条断言，
COM MILESTONE 与审计报告累积到了 m3085。**「测试全绿」只证明当前提交与它的参考值一致，
不证明这些断言会发现一个错的核**——一个永远不会失败的测试不是证据。

仓里已有这套问题的工具：`tensorbem.benchmarks.mutation_audit`（m2582 建立）把 AST 级缺陷
（比较翻转、算术交换、常数扰动）种进数值核心，再跑映射的单元测试：测试变红 = 缺陷被**杀死**；
仍全绿 = **幸存**，即盲点。但它的 `TARGETS` 只登记了五个**二维**模块
（`kernels`、`free_surface`、`hydrostatic`、`solver`、`finite_depth`），
**三维核从未被变异审计过**——这正是「验证欠账」类缺口：已合并功能从未被检验其测试有效性。

三个被审计模块与映射测试（都廉价，全套 2 s）：

| 模块 | 物理内容 | 映射测试 | 变异体总数 |
|---|---|---|---:|
| `kernels3d` | 三角形面元单/双层位势核、Duffy 求积、立体角 | `test_kernels3d.py`, `test_kernels3d_gradients.py` | 359 |
| `free_surface3d` | 深水 Kelvin/Havelock 自由面核、波浪块四分支表示 | `test_free_surface3d.py` | 477 |
| `finite_depth3d` | 有限水深 John 本征级数核、色散/模态 | `test_finite_depth3d.py` | 308 |

## 2. 怎么做的

1. **扩编目标**：在 `mutation_audit.TARGETS` 追加上述三个三维模块（各映射到自己的验证文件），
   使默认审计（每模块 10 个变异体）自动覆盖三维；本单用 `--max-per-module 120` 深扫。
2. **基准（before）**：`--max-per-module 120 --jobs 8 --seed 20261103`，采样固定 →
   360 个变异体，与后续 after 运行**变异体集合完全相同**（key 一一对应），
   这样 before/after 是同一实验的重复而不是两次不同抽样。
   基线运行时把新增测试文件**移出**工作树，避免 broaden 阶段（按函数名自动找同族测试）
   提前看到它而污染基线。
3. **执行**：`python -m tensorbem.benchmarks.mutation_audit --module ... --out outputs/m3328/sweep_before`；
   每个变异体在影子源码树里跑（`_check_isolation` 保证导入的是影子包），
   映射测试全绿后再按函数名 broaden 到同族测试，仍绿才记 survived。
4. **补测**：新增 `tests/test_mutation_gaps_3d.py`（16 个用例），每条都用 docstring 注明它
   **要杀哪个变异体**；用 `--extra-test tests/test_mutation_gaps_3d.py` 重跑同一 360 变异体集合（after）。
5. **归因**：逐个比对 verdict，把翻红的变异体归到具体测试名（`killed_by`）；
   剩余幸存者按「等价/不可观测/守卫路径」分类，脚本 `outputs/m3328/triage.py` 程序化产出
   `outputs/m3328/triage.json` 与图 `fig_mutation_3d.png`（零手抄）。

## 3. 计算结果（`outputs/m3328/{sweep_before,sweep_after,triage.json}`）

### 3.1 变异得分（同一 360 变异体集合）

| 模块 | 变异体 | before | after | Δ |
|---|---:|---:|---:|---:|
| `kernels3d` | 120 | 0.8167 | **0.8667** | +0.050 |
| `free_surface3d` | 120 | 0.7000 | **0.7083** | +0.008 |
| `finite_depth3d` | 120 | 0.7083 | **0.7167** | +0.008 |
| **合计** | 360 | 0.7417（267 杀/93 存） | **0.7639**（275 杀/85 存） | +0.022 |

### 3.2 补测杀死的 8 个变异体（逐条归因到测试名）

| 变异体 | 缺陷 | 杀死它的新测试 |
|---|---|---|
| `kernels3d:44:49` | 双层位势点核 `r2*sqrt(r2)` → `r2/sqrt(r2)`（即 1/r³ → 1/r） | `test_double_layer_point_kernel_scales_like_inverse_cube` |
| `kernels3d:107:65` | `duffy_quadrature_batch(order=8)` → `order=-8`（默认参数） | `test_duffy_quadrature_defaults_and_exactness_on_skewed_panel` |
| `kernels3d:75:56` | `duffy_quadrature(order=8)` → `order=-8` | 同上 |
| `kernels3d:62:11` | `triangle_area` 的 `0.5` → `1.0` | 同上 |
| `kernels3d:180:9` | 重心坐标分子 `(d1*c22 - d2*c12)/det` → `* det` | `test_in_plane_singular_detection_on_skewed_panel` |
| `kernels3d:180:10` | `d1*c22 - d2*c12` → `+ d2*c12` | 同上 |
| `finite_depth3d:334:48` | 水平距离 `y_t - y_s` → `y_t + y_s` | `test_finite_depth_kernel_with_off_axis_source_recovers_deep_water` |
| `free_surface3d:300:18` | `_wave_complex_and_partials` 的 `1e-15` → `1.0` | （after 运行的连带收益） |

### 3.3 为什么它们此前能活着——同一种失效形状：**测试几何太对称**

1. `kernels3d:44`（1/r³ → 1/r）：既有断言全部取 `|x-y| = 1`，
   而 `1/r` 与 `1/r³` 恰好在 `r = 1` 处相等。补测取 `r ∈ {0.5, 1, 2}` 闭式
   `(δ·n)/(4πr³)`，并加「距离减半核值×4」（轴向 `δ·n = r` ⇒ 核 ~ 1/r²）；变异体给 ×2。
2. `kernels3d:180`（重心坐标交叉项符号）：既有面板全是轴对齐的，`c12 = e1·e2 = 0`，
   被翻符号的项根本不进入结果。补测用斜置三角形（`c12 = 1.31`），
   形心必须被奇异保护拒绝、平面外点必须照常可积。
3. `finite_depth3d:334`（`y_t - y_s` 变号）：既有算例的源全在 `y = 0`，
   此时变号是恒等变换。补测把源放到 `y = 0.7`、目标带非零 `y`，
   用独立深水核 `free_surface3d.green_function_3d` 作 `k0*h → ∞` 的极限参考——
   变异体等价于「镜像源」的核，O(1) 偏差，深水极限判据直接否决（并对深度的误差单调下降）。
4. 两个 API 契约缺口：默认求积阶数从未被显式调用过（全部测试都传 `order=`），
   斜置面板上的矩闭合（∫1、∫x、∫y 闭式、节点内含性）一次都没查过。

### 3.4 图

`outputs/m3328/fig_mutation_3d.png`：左图三模块 before/after 变异得分柱（含总体虚线），
右图为 85 个幸存者的分类计数。**这张图就是「测试全绿 ≠ 测试有效」的量化证据**：
三维核现在仍有两成变异体能穿过整套断言。

## 4. 注意点（坑 / 局限 / 适用范围）

1. **「before」必须让新测试缺席**。第一次尝试直接把新测试留在树里做基线，
   broaden 阶段按函数名自动发现它，基线分数虚高 0.002–0.05，
   差点把「补测有效」记成「本来就没问题」。做法：基线运行时把新文件移到 `/tmp`，跑完再移回。
2. **来之不易的教训：抽样的随机性会掩盖剧情**。60 变异体的初扫没抽到 `finite_depth3d:334`，
   120 变异体才抽到。结论只能按「同一集合 before/after 对齐」讲，不能跨抽样比分数。
3. **等价突变体必须承认，不能算失败**。85 个幸存者里 43 个是 `1e-12 → 2e-12`、
   `0.0 → -0.0`、`<= → <` 这类在双精度下不可观测的边界字面量；
   12 个只影响 `verify_*` 诊断助手的步长/默认参数，不改任何核值；
   5 个是波浪块的**表示选择掩码**（cone/reg_a/reg_b/reg_c 四条分支算同一个函数，
   错误路由只损失精度不改变数学，除非在该处越过精度门）；
   1 个是 `_resolve_mode_count` 的安全余量 `+4`（在 `max(...,32)` 下限饱和处不可见）。
   这些写在 `triage.json` 的 `category` 字段，**没有**粉饰成「已覆盖」。
4. **残余的 24 个「守卫/校验路径」需人工再看**。例如 `_resolve_mode_count` 的
   `wanted < 1 → wanted <= 1`（`modes=1` 会静默给出错误精度而非报错）、
   `depth_modes` 的 `nu <= 0 → nu < 0`。物理量不变，但属于输入契约漏洞，留给后续单。
5. **死代码**：`finite_depth3d._FOUR_PI`（line 82）定义后从未使用，
   所以 `4.0 → -4.0` 的变异体必然存活——这不是测试盲点，是代码冗余，建议删。
6. **成本**：360 变异体 × 8 并发 ≈ 80–90 s（三维核测试本身 2 s），
   所以这类审计完全可以在重载窗口做（本单 `load15 ≈ 105/128`）；
   二维模块的 `solver` 靶标才是贵的那个，本单未碰。
7. **没改任何求解器数值代码**：本单只动审计工具的 `TARGETS` 与新增测试文件，
   `kernels3d/free_surface3d/finite_depth3d` 一行未改。

## 5. 误差判定（显式回答「是否 <3%」）

本单的被测量不是物理量而是**测试有效性指标**（变异得分），不存在与解析解的百分比误差；
物理判据由补测里的独立参考承担，逐项如下：

| 关键量 | 基准 | 实测偏差 | 是否 <3% |
|---|---|---:|---|
| 双层位势点核 vs 闭式 `(δ·n)/(4πr³)`，r∈{0.5,1,2} | 解析闭式 | 机器精度（`rel=1e-14` 通过） | **是** |
| Duffy 求积矩闭合（∫1、∫x、∫y，斜置面板） | 三角形闭式矩 | `rel=1e-13` 通过 | **是** |
| 有限水深核（源在 `y=0.7`）× 深水核，`k0·h` 增大 | 独立 Kelvin 核 | 相对偏差 `depth=40` 时 < 2e-3，且随深度单调下降 | **是** |
| 波浪块四分支自由面条件残差 `dG/dz - νG` | 解析边界条件（应为 0） | 各分支 < `1e-9·|G|` | **是** |
| 波浪块四分支调和性 `∇²G` | 解析 PDE（应为 0） | < `1e-2·|G|/1e-8`（中心差分 O(step²) 限制） | **是** |
| 变异得分本身（360 变异体） | before 0.7417 | after 0.7639（+8 杀、0 误杀） | 指标非物理量；补测在干净树上 16/16 通过，无假阳性 |

**结论**：补测引用的全部独立基准（闭式核、三角形矩、深水极限、自由面条件、调和性）
偏差均 <3%；新增 16 个测试在原码上全绿、在被种缺陷的影子上翻红，
说明它们既是**真判据**（不是自比较）又是**有效判据**（能失败）。

## 6. 变更清单

| 文件 | 性质 |
|---|---|
| `src/tensorbem/benchmarks/mutation_audit.py` | `TARGETS` 追加 `kernels3d`/`free_surface3d`/`finite_depth3d`（各带映射测试） |
| `tests/test_mutation_gaps_3d.py` | 新增 16 用例，逐条注明所杀变异体 |
| `outputs/m3328/sweep_before/`、`sweep_after/` | 两次 360 变异体审计的完整报告（JSON+MD） |
| `outputs/m3328/triage.py`、`triage.json`、`fig_mutation_3d.png` | 幸存者分类与 before/after 图（程序化） |
| `dev_report_m3328_3d_mutation_audit.md` | 本报告 |

## 7. 给下一张自找活的线索

1. `solver3d`/`rao`/`symmetry3d`/`pressure`/`recovery` 仍未纳入变异审计（本单只做了三个核模块）；
   `symmetry3d`（228 个变异体）与 `pressure`（82 个）最便宜，优先。
2. 24 个「守卫/校验路径」幸存者里，`_resolve_mode_count(modes=1)` 与
   `depth_modes(nu=0)` 是真输入契约漏洞，值得一张小单补齐负控。
3. `docs/classical_benchmarks.md` 的散体数字（m3085 报告局限三：161 个未保护数字）仍未钉桩。
4. 三维自由液面（Hulme/Wang-Linton）仍未纳入 `declared_numbers` 注册表（m3059 已记欠账）。