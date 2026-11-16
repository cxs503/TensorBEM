# CASE-BEM-LAPLACE-CYLINDER：近壁判据复核——2 条前置声明被实测否证并改写（m22884）

**锚点** `CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，ref 解析复势，容差 `<3%`，
总账 `open`，本单认领做实质增量）· **仓** TensorBEM · **分支** `agent/m22884`

## 1. 起手对账

- 总账机检：`SELECT ... WHERE slug='CASE-BEM-LAPLACE-CYLINDER'` → `open`，未被他人完成，故认领。
- 防重复：m22235（cj3969 重生成，6/49 判据红）、m22521（复核多极 3 条为不可复现 + 入口点写入顺序修复）、
  m22782/m22820（记录陈旧定位）均已读。m22521 明确留下的 `not_run` 欠账是**近壁 3 条红判据**，
  且注明「需完整阶梯才能测」。本单用模块内的近壁子研究把这条欠账机检掉了。

## 2. 结论（三条红判据）

| 判据 | m22235 结论 | 本单机检 | 定性 |
| --- | --- | --- | --- |
| `near_wall_guard_rejects_inside_window` | false | 32/64/128 三层均 `ValueError` 拒绝 | **不可复现**，判据保留 |
| `near_wall_linear_window_extends_to_half_panel` | 可用窗口 2.0（要求 ≤0.5） | 跨层口径 2.0 vs 常值 2.0；**最细层**线性 0.5、常值 1.0 | 判据口径错（把单层结论写成跨层断言）→ 拆两条实测判据 |
| `near_wall_exclusion_floor_is_faceting` | 0.131 > 0.126/3 | 折面参考使误差**升高** 1.17/1.08/1.04 倍 | 判据**方向写反**（折面在圆周内侧，同一名义 δ 真实间隙更小）→ 改为增长率不变量 |

数字、复现命令、几何推导见 `docs/CASE-BEM-LAPLACE-CYLINDER-NEAR-WALL-CRITERIA-AUDIT.md`。

## 3. 代码改动（最小面）

`src/tensorbem/benchmarks/case_cylinder_stream.py`

1. 新增常量 `NEAR_WALL_FACETING_PER_OFFSET_BAND = (0.3, 1.2)`，并把 `NEAR_WALL_LINEAR_FACTOR`
   的注释改成「最细层口径」，附否证数字。
2. `near_wall_study` 新增 `_usable_window(key, subset)`（远→近扫描、遇第一个越界即停），
   产出 `usable_clearance_over_h_max`（线性/常值两基）与 `finest_level_usable_clearance_over_h_max`。
3. `faceting` 每层新增 `offset_over_delta`、`amplification_{constant,linear}`、
   `amplification_per_offset_{constant,linear}`；汇总新增 `amplification` 块
   （`grows_on_every_level` / `decays_with_refinement` / `per_offset_band`）。
4. 判据：`near_wall_linear_window_extends_to_half_panel` 拆为
   `near_wall_linear_window_extends_the_finest_window` + `near_wall_linear_window_does_not_widen_the_worst_case_window`；
   `near_wall_exclusion_floor_is_faceting` 断言改为「每层 surface>circle ∧ 单位偏移放大落在
   [0.3,1.2] ∧ 放大随 h 单调下降」。
5. `render_markdown` 两处判读文字按实测改写（原「窗口扩到 0.5h」「误差塌缩」两处）；
   折面段落显式写出方向。

净效果：删掉的只有被实测推翻的两句，新增三条原来没有的实测不变量。

## 4. 验证

- `pytest tests/test_case_cylinder_stream.py -k "not quick_ladder and not committed_case_artifact and not committed_record"`
  → **34 passed, 2 skipped**（skip 均为目标点落在对数奇异面板上的预期用例）。
- 近壁子研究 `near_wall_study(ladder=(32,64))` 本机跑通，新字段数值：`usable_linear=2.0`、
  `usable_const=2.0`、`finest_linear=0.5`、放大 1.170→1.075 单调下降、单位偏移放大 0.694/0.614
  （带内）、守门三层拒绝、3% 边界三层全部定位、窗口余量 4.23×。
- 端到端完整阶梯重生成：已递交 **cj4455**（旧码复现）与 **cj4460**（本工作树新码），
  递交时队列位次 69/71、在飞 87 作业，**本单窗口内均未起跑**；提交记录仍是 38 条无 `provenance`
  的旧版，记录陈旧这笔账**仍然开着**（不粉饰）。

## 5. 下次自找活线索

1. cj4460 起跑后 `fetch` 取 `runs/m22884/regen2/results.json`，若 `overall=true` 则逐位拷入
   `benchmarks/verified/case_bem_laplace_cylinder/`（含 `provenance` 与 5 张图）并改写
   `docs/CASE-BEM-LAPLACE-CYLINDER.md`，同批启用 m22521 §2.3 备好的 `skip → fail` 硬化，
   一次性关掉记录陈旧账。
2. 34 份 `benchmarks/verified/*/results.json` 中仅 4 份带 `provenance`：把截断不变量 +
   重生成门推广到全池是下一笔结构性欠账。
3. 近壁窗口若要真正做到 `δ = 0.5 h_max` 跨层成立，需要曲边面板或更高阶迹（本轮只把
   可行性边界测清，未做实现）。
