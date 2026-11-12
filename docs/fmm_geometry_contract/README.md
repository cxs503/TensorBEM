# FMMOperator 几何契约缺陷：`matvec(sources≠setup 点集)` 的静默错答（fail-open）→ 失败关闭

**仓库**：TensorBEM · **分支**：agent/m16265 · **基线**：main @ 13ea06b
**工件**：`src/tensorbem/fmm.py`（修补）· `scripts/fmm_geometry_contract.py`（证据生成）·
`docs/fmm_geometry_contract/evidence.json`（全部数字，零手抄）·
`docs/fmm_geometry_contract/fig_geometry_contract.png` · `tests/test_fmm_geometry_contract.py`（13 例）

## 1 案例介绍

`tensorbem.fmm.FMMOperator` 是 Laplace 单层算子 `S q` 的快速多极加速器（八叉树 +
实球谐多极）。算子对象在一次 `setup(points[, charges])` 调用里固定三样**几何派生量**：

| 派生量 | 来源 | 使用者 |
|---|---|---|
| 八叉树 + 叶中心 `_leaf_centers` | `setup(points)` | 目标路由 `_route` |
| 逐叶源点索引 `_leaf_indices` | `setup(points)` | 近场直和 `_near_contributions` |
| 逐叶源半径 `_radii_vec` | `setup(points)` | 可容性判据 `_ADMISSIBILITY` |

而公开入口是 `matvec(targets, sources, charges)` —— **`sources` 是每次调用的实参**。
m16000 修掉了「电荷向量换了但多极系数不失效」的静默错答，m16109 修掉了「未 `setup`
就用算子返回全零」的 fail-open。两者都围绕 *电荷/生命周期*，**几何**这一维仍被信任：
`sources` 与 `setup` 点集不一致时，代码把**新点集**塞进**旧点集的叶划分与叶中心**里算。

本单要回答的问题是：这个未定义用法在数值上是否真的错、错多少、以及最小学的修法。

## 2 怎么做的

- **判别性 A/B**：`_PrefixGeometryOperator` 继承出修复前的语义，唯一差异是
  `_require_setup_geometry()` 被覆写为空操作（其余代码路径逐位相同），因此 A/B 的单变量
  就是本单新增的契约。
- 几何自动生成（`scripts/fmm_geometry_contract.py`）：`points ~ N(0, 0.5)`，N=200，
  目标点 3 个位于 6–8 距离外，`p=4, max_leaf=25`；真值用同模块的
  `direct_laplace_matvec`（稠密直和）。
- 五个「点集被改动」场景：平移 +10x、缩放 ×3、缩放 ×20、置换顺序、1e-9 级微移；
  加一组「源点数不同」（50 / 199 / 201）以检查异常类型。
- 修复（`src/tensorbem/fmm.py`，最小改动 3 处）：
  1. `setup()` 末尾存档 `_setup_points = np.array(points, copy=True)`；
  2. 新增 `_require_setup_geometry(sources)`：形状不符或数组不等 → `ValueError`，
     并显式说明「八叉树/叶索引/叶半径属于 setup 几何」；未 `setup` → `RuntimeError`；
  3. `matvec` 在 `_require_setup("matvec")` 之后立即调用它。
  电荷（多右端项/迭代求解）与目标点**照旧可变**，不设限。

## 3 计算结果（全部来自 `evidence.json`，程序化生成）

同几何基线（p=4 的 FMM 截断误差上限）：**1.775e-07**。

| 场景 | 修复前（无几何校验）相对误差 | 修复前结局 | 修复后结局 |
|---|---|---|---|
| 平移 +10x | **9.777e-01（97.8%）** | 静默返回错值 | `ValueError` |
| 缩放 ×3 | **6.454e-03（0.65%）** | 静默返回错值 | `ValueError` |
| 缩放 ×20 | **1.747e+02（17470%）** | 静默返回错值 | `ValueError` |
| 置换源点顺序 | 2.653e-04 | 静默返回错值 | `ValueError` |
| 1e-9 微移 | 1.775e-07（=基线） | 静默返回错值 | `ValueError` |

源点数不一致（同一 setup 几何下换点数）：

| n_sources | 修复前 | 修复后 |
|---|---|---|
| 50 | `IndexError`（近场循环深处的裸异常） | `ValueError`（显式契约消息） |
| 199 | `IndexError` | `ValueError` |
| 201 | **静默接受**（错值） | `ValueError` |

## 4 结果图

`docs/fmm_geometry_contract/fig_geometry_contract.png`：五个场景的修复前相对误差（对数
纵轴）与 p=4 同几何基线（虚线）。图由 `scripts/fmm_geometry_contract.py` 从同一
`evidence.json` 数据直接绘制（`matplotlib`，非手摆数字）。

## 5 注意点

1. **误差量级取决于错法**：置换顺序只错 2.65e-4（多极总和近似不变，只是叶分配错），
   而缩放 ×20 错 174 倍——因为源点被扔进远离其自身的叶盒，多极展开的收敛前提
   （`|x−c| > |y_i−c|`）被破坏。**不能说「几何错也问题不大」**：最温和的场景已超出
   p=4 截断误差 3 个量级（2.65e-4 vs 1.775e-7）。
2. 修复是**契约收紧**，不是数值算法改动：同几何的答案逐位不变（基线 1.775e-07，
   `tests/test_fmm_geometry_contract.py::test_setup_geometry_answers_match_the_dense_sum`
   与 m16000 的 49 例回归均绿）。
3. 微移 1e-9 的用例说明选择的是**精确相等**判据：任何不完全相同的点集都要求重新
   `setup()`，以避免「看起来一样、叶分配其实可能翻转」的灰区。代价是每调用一次
   `np.array_equal`（O(N)，相对于一次 matvec 可忽略）。
4. 未做：不自动重建八叉树（那需要把 `points/charges` 的语义并入一个可变状态机，
   超出最小改动）；若将来需要「同算子跨几何复用」，应在 `setup` 之外提供显式的
   `rebuild_geometry()`。

## 6 误差判定

| 关键量 | 基准 | 修复前误差 | 修复后 | 是否 <3% |
|---|---|---|---|---|
| 平移 +10x 场景的 matvec | 稠密直和 | 97.8% | 拒绝执行（`ValueError`） | 修复前 **否**；修复后不适用（无错答可报） |
| 缩放 ×3 场景 | 稠密直和 | 0.645% | 拒绝执行 | 修复前在 3% 内但**静默**（不可接受） |
| 缩放 ×20 场景 | 稠密直和 | 17470% | 拒绝执行 | 修复前 **否** |
| 合法几何（同点集） | 稠密直和 | 1.775e-07 | 1.775e-07 | **是**（<3%，且比 3% 小 5 个量级） |

结论：合法用法的精度未受影响（1.8e-7 ≪ 3%）；非法用法从「静默 0.6%–17470% 错答」
变为「显式拒绝」。本单是缺陷修复 + 失败关闭契约，不是精度改进。

## 7 复现

```bash
PYTHONPATH=src python scripts/fmm_geometry_contract.py     # 写 evidence.json + 图
PYTHONPATH=src python -m pytest tests/test_fmm_geometry_contract.py \
    tests/test_fmm.py tests/test_fmm_setup_contract.py tests/test_fmm_operator_linearity.py -q
```

实测：`13 passed`（新文件）+ 既有 FMM 三文件 61 passed（合计 61 passed / 0 failed，
`-q` 30.5 s，load15≈110 下只跑定向文件，未跑全量套件）。