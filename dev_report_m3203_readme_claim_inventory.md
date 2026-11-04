# TensorBEM 验证欠账：README 能力声明与已合并实现的矛盾（3 处）及其可执行闸门

**任务**：m3203（verifier 席，自找活） · **分支**：`agent/m3203` · **日期**：2026-11-04
**仓**：TensorBEM（worktree `/root/tp-agent/work/m3203`，基线 `6025970`）

EDIT_WHITELIST: src/tensorbem/benchmarks/doc_inventory.py, tests/test_doc_inventory.py,
README.md, docs/near_surface_field.md, docs/readme_claim_inventory.md,
outputs/verif_m3203/, dev_report_m3203_readme_claim_inventory.md

---

## 1. 为什么选这件活（选题依据）

扫描所得：`declared_numbers` 注册表（22 条声明数字，m3059 刚复算完毕）、
`doc_number_ledger`（MacCamy-Fuchs 表 8/9）、`doc_pins`、`three_dimensional_doc_pins`
已经覆盖了「文档里引用的**数值**」。**没有被覆盖的是「文档里的能力声明」**——即
「X 已实现」与「Y 仍是缺口」这两类句子。这类句子一旦与代码脱节，代价不是读者被误导，
而是把舰队（含 AI agent）派去重做已合并的工作。

本单实测到 3 处矛盾，其中 2 处位于**同一个文件内**：

| 编号 | 位置 | 原句（2026-11-04 实测） | 与代码的冲突 |
|---|---|---|---|
| D1 | README「Numerical conventions」3-D 段 | "near-boundary near-singular quadrature is **not implemented**" | `tensorbem.near_singular` + `solver3d.evaluate_surface_field_near_surface` / `evaluate_surface_velocity_near_surface` 已发布，并有 `docs/near_surface_field.md` 与两组测试 |
| D2 | README 路线图第 5 条 | "graded meshes, higher-order/curved panels, and **near-singular field quadrature remain**" | 同 D1；同一句里另两项（渐变网格、高阶面板）确实是缺口 |
| D3 | README 路线图第 6 条 | "**Next, needs new physics:** finite-depth 3-D John eigenseries Green function…" | 同一 README 上方 24 行处已声明该核 "now implemented and verified"（`tensorbem.finite_depth3d`，26 条测试门） |

D3 是最贵的一条：只读路线图的人（或 agent）会把已验证的 John 特征级数核当成待开发项重做一遍。
没有任何可执行对象对此提出异议，因为**句子不是测试**。

## 2. 做了什么

1. **`src/tensorbem/benchmarks/doc_inventory.py`**：把 README 的 19 条可判定能力声明做成表
   （16 条 shipped、3 条 absence），每行携带判定证据：
   - `modules`：shipped 必须可导入；**absence 必须不可导入**（缺口真开着）；
   - `test_files`：必须存在且真的含 `def test_`（"已实现" 意味着 "有守护"）；
   - `docs`：报告文件必须存在；
   - `probes`：数值证明（测得的误差 + 验收上限），让声明带数字而非只有文件名；
   - `phrase`（absence）：缺口句子必须仍在 README 里；
   - `stale_phrases` / `stale_sources`：一旦 shipped，这些句子**不得再出现**——即把 D1/D2/D3
     变成回归钉桩，任何人写回旧句都会让闸门变红。
   CLI：`--strict`（drift 时非零退出）、`--markdown`（表格注入）、`--output`（严格 JSON）。
   `check(overrides={...})` 允许测试用替换文本评估"假设的树"而不写盘。
2. **修正 README 三处 + `docs/near_surface_field.md` 一处**（该报告开头原本也复述了
   "README 把近场求积列为唯一剩余缺口"）：按**实测**改写为
   - 近场求积已发布（两个 entry point + 自适应细分规则），并保留诚实限制：「近壁
     *速度/压力* 残差来自分段常数密度的微分，不是求积误差」；
   - 路线图第 5 条剩余缺口收窄为渐变网格与高阶/曲率面板；
   - 路线图第 6 条缺口收窄为 **有限水深 3-D 边界元求解器层**（核已验证），保留 Garrett
     截断浮筒、有航速 Rankine 兴波阻力、三维升力尾流；
   - `docs/near_surface_field.md` 改为历史时态并注明 2026-11-04 被本闸门纠正。
3. **`tests/test_doc_inventory.py`（19 条测试）**：正向 + 六种负控（写回旧句、缺测试文件、
   模块不可导入、缺口实为已关闭、absense 句子被删、探针不达标），并钉住探针数值。
4. **`docs/readme_claim_inventory.md`** + `outputs/verif_m3203/{readme_claim_inventory.json,
   fig_claim_inventory.png,build_report.py}`：表格与图全部由 JSON 程序化生成（零手抄）。

## 3. 结果

**闸门状态**：修正前 **17/19 PASS**，2 行 DRIFT（`near_singular_field`、`finite_depth3d_kernel`），
共命中 4 条旧句（README 3 条 + `docs/near_surface_field.md` 1 条）；修正后 **19/19 PASS，
drift 为空**，`--strict` 退出码 0。（"修正前"不是回忆：`build_report.py` 用 `overrides`
把三条旧句重放进同一 README 再跑一遍闸门，4 条命中是重新测出来的。）

四个可执行探针（都能独立复现文档里的数字）：

| claim | 探针 | 实测 | 文档值 | 上限 | 判定 |
|---|---|---|---|---|---|
| `finite_depth3d_kernel` | 深水极限 vs Kelvin 核（`k0h=76.8`） | 2.9079e-7 | 2.908e-7 | < 1e-5 | PASS |
| `finite_depth3d_kernel` | 自由面残差 `\|∂G/∂z−νG\|/(ν\|G\|)`（`h=40`） | 2.4304e-12 | 2.430e-12 | < 1e-10 | PASS |
| `near_singular_field` | 近奇异单层积分（1e-3 面板尺度） | 1.886e-14 | 机器精度级 | < 1e-10 | PASS |
| `declared_number_registry` | 注册表条目数 | 22 | ≥20 | > 20 | PASS |

## 4. 注意点与局限

- **表是策展的，不是抓取的**：覆盖 milestone/roadmap/numerical-conventions 三段中所有
  「证据可判定」的能力声明；任意散文不解析，未被任何行提及的句子是未受守护的（行数已报告，
  使"README 说了什么"与"被钉住了什么"的差距保持可见）。
- **"测试文件存在且含测试" 是存在性闸门，不是覆盖率闸门**：那些测试的数值内容由仓内既有
  套件守护（`test_declared_numbers.py`、`test_finite_depth3d.py` 等），本模块刻意不重复。
- **absence 行只认行内声明的模块名**（`tensorbem.finite_depth3d_solver`、`tensorbem.rankine`、
  `tensorbem.lifting3d`）：将来换个名字实现，缺口会关闭而不触发行——但仍需人工改 README，
  这正是闸门的目的：句子与代码必须一起动。
- **一个容易踩的坑**（本单实测，已写进报告局限）：深水极限探针只在**固定物理几何**下有
  意义。`k0R ≫ 1` 时源与其镜像近抵消，把几何整体缩放后在同样的 `k0h` 下比较，同一对核
  会给出 2e-1 量级的相对差——那是比较本身的抵消病态，不是任一核的误差。故探针钉死了
  `docs/finite_depth3d.md` §3.3 的原始点集（`ν=0.6, h=128`）。
- 未做：未改任何求解器实现；未把库内其它报告的散文纳入闸门（如 `docs/finite_depth_*.md`），
  建议下一个自找活把 `docs/` 内的能力声明也纳入同一张表。

## 5. 误差判定

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---|---|
| 有限水深核深水极限 | 独立 Kelvin 核（`free_surface3d`） | 2.908e-7（vs 文档 2.908e-7，偏差 <1e-4） | ✅ |
| 有限水深核自由面残差 | 边界条件本身 | 2.430e-12 | ✅ |
| 近奇异求积误差（1e-3 面板尺度） | 同规则收敛实例（order 14 / split 2） | 1.886e-14 | ✅ |
| README 声明 vs 代码 | 19 条策展声明 | 修正后 0 条误述（修正前 3 条） | ✅ |

结论：本次涉及的全部关键量对基准误差均 <3%；三条误述已按实测定写并由闸门回归钉桩。