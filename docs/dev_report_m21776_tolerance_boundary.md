# CASE-BEM-LAPLACE-CYLINDER 第 6 轮：<3% 的容差边界测量 + 阻断性渲染缺陷修复 — TensorBEM m21776

**锚点** `CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，容差 <3%）· **仓** TensorBEM · **分支** `agent/m21776`

## ① 本轮要解决的问题

前 5 轮把该算例的判据从 0 做到 45 条，数值误差远优于 3%（最差 2.85e-04）。但总账锚点仍 `open`，
原因不在数值，而在**证据链**：已提交记录 `results.json` 仍是 38 条判据、无 `provenance`、无
`figures/`，而 `docs/CASE-BEM-LAPLACE-CYLINDER.md` 引用了 5 张并不存在的图。第 4、5 轮都只把
"记录重生成"递交到集群（cj3433 / cj3470）然后结束，**没有一轮验证它是否真的能重生成**。

本轮先回答"为什么两轮都重生成失败"，再补一处实质验证欠账。

## ② 阻断性缺陷（本轮新发现，均已修复）

### 2.1 `render_markdown()` 必然抛 `NameError`（这就是两轮重生成空手而归的根因）

`render_markdown()` 的"误差归因（第 3 轮修订）"段引用了 `finest_key` 与 `exclusion_key`
两个**从未绑定**的名字：

```
NameError: name 'finest_key' is not defined. Did you mean: 'finest_c'?
  case_cylinder_stream.py:2829  in render_markdown
```

任何 artifact 走到渲染就死，`results.json` / `doc` 都写不出来——与"集群没排上"无关，
是纯代码缺陷。修复：在引用前由最细层与排除区因子构造这两个键
（`finest_key = str(near["levels"][-1]["panels"])`，`exclusion_key = f"{NEAR_WALL_EXCLUSION_FACTOR:g}"`），
与被引用的 `attribution` 表的键构造方式一致。

端到端验证（本地两层级 harness，`near_wall` 强制 `(32,64)`）：
`run_case → criteria(49 条) → render_markdown` 输出 12517 字符，`RENDER_OK`，不再抛异常。

### 2.2 `near_wall_study(factors=...)` 对调用方抛裸 `KeyError`

`factors` 读起来是采样选择，但函数内部有 3 个 clearance 水平是**契约**（窗口 2.0、
一阶矩窗口 0.5、排除区 0.1），缺任一个就在汇总处抛 `KeyError: 0.5`。第 5 轮是在**测试里**
把 `factors` 补全来绕过它，等于把这个坑留给下一个调用方。修复：在函数入口把契约水平并入
`factors`（集合并集 + 降序），调用方传短元组不再崩。

## ③ 实质增量：`<3%` 到底从多远开始成立（本轮新增判据）

### 3.1 缺口

记录用两个**离散** clearance 钉住了两端：`δ = 2 h_max` 误差 ≤1%（`near_wall_window_error`），
`δ = 0.1 h_max` 误差 >3%（`near_wall_exclusion_zone_is_real`）。**两者之间的容差边界从未测量**，
而调用方要的正是这个数。既有扫掠已经显示 `δ = 0.25 h_max` 处误差约 9%——即**已在 3% 容差之外，
而当时每一条判据都是 PASS**。这正是"判据通过 ≠ 接受成立"的证据链漏洞。

### 3.2 做法

在 `δ/h_max` 上按 14 点对数网格（4.0 → 0.05）逐层测量最差方向相对速度误差（仅常值密度，
与被闸门的表示一致），取穿越 `TOLERANCE = 3%` 的相邻两点做对数-对数插值，得到
`δ*/h_max`；再要求三个性质：

1. **可定位**：每层都在网格内跨越 3%（否则可用区无界=假设，或为空）→
   `near_wall_tolerance_boundary_is_located`
2. **加密使边界向内**：可用包络不得随加密缩小 → `near_wall_tolerance_boundary_moves_inward_with_refinement`
   （容差 5%；单层梯子**不得**声称该性质，`shifts` 为空时判据取 False）
3. **窗口离边界够远**：`2 h_max` 至少是实测边界的 2 倍 →
   `near_wall_window_clears_measured_tolerance_boundary`

### 3.3 实测结果（两层，本地验证；三层见 ⑤）

| 层 | $h_{max}/a$ | 3% 边界 $\delta^*/h_{max}$ | 夹逼区间 | 区间端点误差 | 区内幂次 |
| --- | --- | --- | --- | --- | --- |
| 32 | 1.960e-01 | 0.577 | [0.45, 0.6] | 4.293e-02 / 2.834e-02 | 1.44 |
| 64 | 9.814e-02 | 0.473 | [0.45, 0.6] | 3.301e-02 / 1.885e-02 | 1.95 |

- 粗/细层边界比 **1.220** → 加密使边界向内移动（可用包络不缩小），判据 2 PASS。
- 最细层边界 0.473 → 守门的 1% 窗口 `2 h_max` 距边界 **4.23×**（判据要求 ≥2×），判据 3 PASS。
- 给消费方的分辨率规则：要评价物理 clearance $\delta$，需 $h_{max} \le \delta/\delta^*$
  （$\delta^*$ 随加密缓慢内移，故按最细层取值偏保守）。
- 路线 D/N 的探针环最近在 $r/a = 1.25$（$\delta = 0.25a$）；最细层 $h_{max}=0.049a$ 处
  $\delta/h_{max}\approx 5.1$，远在边界之外 → **既有路线判据不受影响**，本轮增量只是把
  "远在边界之外"从隐含改为实测。

## ④ 注意点 / 局限

- `δ*` 的被测量是"径向、过面板中点"方向的最差误差，与既有两个闸门同构，不是全方向上确界。
- 边界随加密**缓慢**内移（1.22×/每减半），所以"按最细层 $\delta^*$ 取网格"是保守侧，不是最优侧。
- 判据 2 的语义要求至少两层；单层梯子下该判据取 False（已用测试钉住这个语义，防止未来被
  "空 `all()` 恒真"糊过去）。
- **两个既有判据在 (32,64) 两层梯子上 FAIL**：`near_wall_linear_window_extends_to_half_panel`、
  `near_wall_exclusion_floor_is_faceting`。本轮**未改动**这两个判据及其输入的任何一个数字；
  它们在完整三层梯子 `(32,64,128)` 上是否 PASS 需记录重生成后确认（见 ⑤）。此处如实记录，
  不推定为通过。

## ⑤ 未完成项（如实）

**记录重生成尚未落盘**。集群作业 `cj3714`（8 核，`stage_dir` 指向本工作树）已递交并排队，
排队位次 63，等待 25 min 仍在 `queued`（同窗口内 11 分钟前的作业同样未派发）。
本机 `nohup/setsid` 已被计算调度门硬拦（owner 2026-11-13 令），前台跑超 `render`/阶梯
时长（>120 s）会被工具收割，因此**无法在本单内合法完成**全三层重生成。

已提交的代码使重生成**可达**（此前任何 artifact 都无法渲染）。解除阻塞只需一条命令：

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream \
  --outdir benchmarks/verified/case_bem_laplace_cylinder \
  --doc docs/CASE-BEM-LAPLACE-CYLINDER.md
```

它会一次产出：49 条判据的 `results.json`（含 `provenance`）、`report.md`、`case.json`、
`figures/{field,convergence,near_wall,multipole,circulation}.png`，并重写文档。
`tests/test_near_wall_tolerance_boundary.py::test_committed_record_gates_the_tolerance_boundary`
在记录未刷新时 SKIP、刷新后转为断言（判据 3 条 ok + 5 张图齐 + provenance 存在），
`test_committed_record_was_rendered_from_the_current_case_module` 同理。

## ⑥ 验证（本轮实际跑过的）

| 命令 | 结果 |
| --- | --- |
| `pytest tests/test_near_wall_tolerance_boundary.py -q` | **3 passed, 1 skipped**（78 s；skip=记录待刷新，符合设计） |
| `run_case(ladder=(32,64)) → criteria → render_markdown` | 49 条判据，渲染 12517 字符，`RENDER_OK`（修前 `NameError` 必崩） |
| `near_wall_study(ladder=(32,))` + `factors=(2.0,0.1)` | 不再 `KeyError`；边界 0.577；单层不声称不变性 |

新增判据 3 条在两层实测下均 PASS（边界可定位 0.473、向内移动 1.220×、窗口余量 4.23×）。

## ⑦ 交接线索

1. 排队中的 `cj3714` 一旦派发即产出上述全部工件，取回后本锚点可从 `open` 转 `verified`
   （仅需确认 49/49 PASS；若 `near_wall_linear_window_extends_to_half_panel` /
   `near_wall_exclusion_floor_is_faceting` 在完整梯子上仍 FAIL，则那是**这两个判据的**
   honest FAIL，与本轮无关，应另开条目而非降低判据）。
2. 更普遍的可复用教训：**"递交给集群后结束"不算交付**。第 4、5 两轮把重生成外包出去而没有
   验证命令能跑，掩盖了一个单行 `NameError`。凡"记录刷新/工件重生成"类任务，至少要本地跑到
   渲染成功（可缩阶梯）再交集群跑全量。