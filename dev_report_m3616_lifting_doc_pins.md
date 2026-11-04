# docs/lifting_foil.md 的 48 个声明数字判据化：文档文本 → 实测，零中间工件（并揪出两处真实文档错误）

**任务**：m3616（infrastructure 席，自找活） · **分支**：`agent/m3616` · **日期**：2026-11-04
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3616`）
**工件**：`src/tensorbem/benchmarks/lifting_doc_pins.py`（判据）、
`tests/test_lifting_doc_pins.py`（18 条测试，含 4 条负控）、
`docs/m3616/lifting_doc_pins_ledger.json`（48 条逐项判决）、
`docs/m3616/figures/lifting_doc_pins.png`、生成器 `docs/m3616/make_ledger_and_figures.py`

EDIT_WHITELIST: src/tensorbem/benchmarks/lifting_doc_pins.py, tests/test_lifting_doc_pins.py, docs/lifting_foil.md, docs/m3616/

---

## 1. 案例介绍（为什么做这件事）

`docs/lifting_foil.md` 是 TensorBEM 二维升力线的验证记录：解析 Joukowski 参考解（Kutta 环量 vs 同场压强积分自洽性）与 2-D 外域 Neumann 离散化的一致性，外加 Kutta 闭合、驻点敏感性、以及被否决的层梯度闭合负控。

它 §3 标题写着「所有数字由 `benchmarks/lifting_foil.py` 计算」，然后列出两张表（4 个攻角 × 4 列、4 个网格 × 4 列），§2/§4/§5 又用散文给出 9 个数。

**问题**：`grep -rn '4\.67\|2\.39\|7\.36e-4' tests/ src/` **零命中**。围绕 lifting 代码的测试（`tests/test_lifting_foil.py`、`tests/test_lifting_flow.py`，共 30 条）全是**定性**断言（「细化降低误差」「升力在 3% 以内」），因此这张表里每一位数字都可以在构建仍绿的情况下漂走——本仓已被这个失效模式咬过三次（Ursell `7.7e-3→8.51e-3`、MFS 旧列、Wigley 体积 `2LBT/3→4LBT/9`），每次都是后来的审计发现，而不是构建发现。

**算例参数**（全部沿用文档，未改物理设置）

| 项 | 值 |
|---|---|
| 剖面 | 对称 Joukowski 翼型，圆平面偏心/变换半径 0.1/1.0 |
| 弦长 / 厚比 | c = 4.0333 m，t/c = 0.1179 |
| 攻角 | 0°, 2°, 4°, 6°（参考攻角 4°） |
| 网格 | 32/64/128/256 等弧长面元，尾缘为顶点（尖尾） |
| 求解器 | `solve_laplace_exterior_neumann`（外域 Neumann，二维）；解析参考 `tensorbem.lifting.JoukowskiAirfoil`（与 BEM 不共享代码） |
| 判据 | 逐数字：印刷精度决定容差（HULP） |

## 2. 怎么做的

1. **先造判据**：把「文档里的数字」本身当作 claim 源——`parse_document()` 从 markdown 反向解析两张表 + 5 段散文，`4.80e-3` 是 3 位有效数字的主张、`61` 是整数主张，容差一律取**半个末位单位（HULP）**，不引入任何手工转录的数字。
2. **实测**：`measure()` 用当前代码重算全部 48 个量：解析参考（`analytic_reference`）、细化梯度（`compare_nonlifting_field`）、Kutta 闭合（`lifting_solver._row`）、驻点敏感性（0.15c vs 0.75c，两个分数**从文档里读**）、负控（`_layer_gradient_probe`，网格数也**从文档里读**）。没有任何中间产物文件，因此数字没有第二个可以腐化的地方——改文档就是改主张。
3. **判决**：`compare()` 逐条 PASS/DRIFT，`--check` 出口码 0/1，`--json` 落盘完账。
4. **负控证明闸门不是哑的**：改文档一个印刷位（`4.67%→4.68%`）必须 DRIFT；改一个实测值必须 DRIFT；把散文里的「first order」改回「second order」必须 DRIFT；删掉表格一行必须使 claim 清单少那一行（否则 parse 静默变宽松）。
5. **CI 接线**：`tests/test_lifting_doc_pins.py` 18 条，含 2 条 subprocess CLI 契约（真文档 → 出口 0；污染文档 → 出口 1 且 drift 项唯一）。

## 3. 计算结果

**判决：48/48 PASS，0 DRIFT**（`docs/m3616/lifting_doc_pins_ledger.json`）。

### 3.1 §3.1 解析自洽性（4 攻角 × 4 量）

| α [deg] | Γ | C_L(环量) | C_L(压强) | gap 文档 | gap 实测 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.00000 | 0.00000 | 0 | 0（精确） |
| 2 | 0.4824 | 0.23921 | 0.23919 | 0.009% | 0.0090% |
| 4 | 0.9642 | 0.47814 | 0.47722 | 0.19% | 0.192% |
| 6 | 1.4449 | 0.71648 | 0.71292 | 0.50% | 0.496% |

### 3.2 §3.2 离散场细化梯度（4 网格 × 4 列）

| panels | max\|δφ\|/max\|φ_d\| | rms \|δφ\| | rms 速度误差/max | cond |
|---:|---:|---:|---:|---:|
| 32 | 4.67% | 4.80e-3 | 7.5% | 61 |
| 64 | 3.31% | 2.61e-3 | 7.6% | 92 |
| 128 | 2.39% | 1.39e-3 | 8.0% | 138 |
| 256 | 1.68% | 7.36e-4 | 8.2% | 205 |

全部在 HULP 内复现。rms 的观测阶数（本次判据化的新量）：**0.88 / 0.91 / 0.92**（均值 0.90）。

### 3.3 §2/§4/§5 散文数字

| 声明 | 文档 | 实测 | 判定 |
|---|---|---:|---|
| 衰减常数 U m cos α | 0.0998 | 0.0997564 | PASS |
| 边界值偏差（未去项） | −0.0998 | −0.099791 | PASS |
| 该偏差残差谱宽（1σ, 128 面元） | ±0.0014 | 0.001393 | PASS |
| 薄翼极限 C_L 4° | 0.4821 | 0.482122 | PASS |
| 剖面 C_L 4° | 0.4781 | 0.478138 | PASS |
| 薄翼极限偏差 | 0.8% | 0.826% | PASS |
| Kutta 闭合：环量升力 vs 压强升力（128） | 0.12% | 0.11696% | PASS |
| 升力低于精确值（128） | 2.3% | 2.2578% | PASS |
| 驻点位置 0.15c→0.75c 环量变化 | <1% | 0.6822% | PASS |
| 层梯度闭合误差（64） | ~29% | 29.124% | PASS |
| 三点格式误差（64） | ~4.7% | 4.7375% | PASS |

### 3.4 两处**真实文档错误**（本判据的审计产出，已修文档并由 claim 钉住）

1. **阶数错标**：原文「The rms potential error halves with each refinement (**second order** in the panel size)」。梯度表做的正是「面元尺寸减半」，二阶应当**四分之一**，但印刷列只减半。实测阶数 0.88/0.91/0.92 → **一阶**。已改为一阶并写出三个实测值；「first/second」这个词本身现在是 claim（容差 0.15），改回二阶即红。
2. **统计量无定义**：原文「`-0.0998 +/- 0.0017` without the term and `+/-0.002` with it」——既没说网格也没说统计量，且谱宽与同一场的 `±0.002` 自相矛盾。**任何网格都复现不出 0.0017**（64/128/256 的 1σ 分别 0.0026/0.0014/0.00074）。已改为「128 面元、1σ = ±0.0014」并纳入判据。

## 4. 结果图

`docs/m3616/figures/lifting_doc_pins.png`（两联图，程序化生成，零手抄）：
左＝§3.2 细化梯度，线段为**重算值**、× 为**文档印刷值**，两者逐点重合；右＝48 条 claim 的 |实测−声明|/容差 排序条形图（对数轴），红色虚线为容差上限 —— 全部在 1 之下，即构建绿的判据。

## 5. 注意点（坑与适用范围）

* **导入路径坑**：本机 `site-packages` 里装了一份 `tensorbem`，直接 `python3` 会静默导入**安装版**而非 worktree。所有命令必须 `PYTHONPATH=$PWD/src`，否则「实测」验的是别的代码。
* **不要把 `lifting_solver.main()` 拿来当测量函数**：它默认 `write_doc=True`，会重写 `docs/lifting_kutta_closure.md`。本模块只调用 `_row`/`_layer_gradient_probe` 这类无副作用私有函数。
* **markdown 表格里的转义竖线**：§3.2 表头含 `max\|dphi\|`，按裸 `|` 切分会把表头切碎 → 用 `(?<!\\)\|` 负向后视，并有单测钉住。
* **适用范围**：只覆盖 `docs/lifting_foil.md`；`docs/lifting_kutta_closure.md` 是**生成式文档**（`_render_document` 直接写盘），风险类型不同（不是漂移而是生成器本身），本次未纳入。
* 容量纪律：本机 load15 ≥ 100，全部计算单进程 `OPENBLAS_NUM_THREADS=1`，整套测试 74 s（三个模块）。

## 6. 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

基准 = 文档印刷值 / 解析闭式解；误差 = |实测 − 声明| / 声明（上界类 claim 用「实测 ≤ 声明」判定）。

| 关键量 | 基准来源 | 相对误差 | <3%？ |
|---|---|---:|---|
| Γ @ α=2/4/6° | 解析闭式 4πaU sinα | 0（HULP 内） | 是 |
| C_L 环量 vs 压强（4°） | 同场压强积分 | 0.192% | 是 |
| rms \|δφ\| @ 256 面元 | 文档 7.36e-4 | <1e-3（HULP） | 是 |
| Kutta 闭合自洽（128） | 文档 0.12% | 0.11696% vs 0.12 → 2.5% | 是（在容差内；该项本身是对自洽性残差的量化，不是物理量精度） |
| 升力 vs 精确（128） | 精确 C_L 0.478138 | 2.2578% | 是 |
| 环量 vs 精确（128） | 精确 Γ 0.964244 | 2.3719% | 是 |
| 升力 vs 精确（256） | 精确 C_L | 1.32%（`docs/lifting_kutta_closure.md` 的生成值，本单未复算） | 未判定（本单范围外） |
| 驻点敏感性 | 同网格 0.15c vs 0.75c | 0.6822%（<1% 上界） | 是 |

**结论**：全部数字在印刷精度内复现（47 条数值 + 1 条上界，48/48 PASS）；物理精度侧 128 面元升力误差 2.26%、256 面元 1.32%，均 <3%，与文档声明一致。两处**非数值**缺陷（阶数措辞、谱宽统计量未定义）已如实记录并修正，不作粉饰。

## 7. 复现

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python3 -m tensorbem.benchmarks.lifting_doc_pins --check
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/test_lifting_doc_pins.py -q   # 18 passed
python3 docs/m3616/make_ledger_and_figures.py                                               # 重建 ledger + 图
```