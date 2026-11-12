# m16488 · TensorBEM 能力台账「静默降级」缺陷：7 行证据被自己抹掉 → 71/71 REAL

**仓库** TensorBEM · **分支** `agent/m16488` · 基线 `b200ae0`（main @ 2026-11-12）
**产物** `src/tensorbem/capability_evidence.py`、`docs/audit_baseline.md`（生成块）、
`tests/test_capability_evidence.py`、`docs/m16488/{audit_before,audit_after,pin_probe,evidence}.json`、
`docs/m16488/fig_evidence_drift_repair.png`

## 1 案例介绍（问题不是物理，是证据本身）

`docs/audit_baseline.md` §4 的能力表由 `tensorbem.capability_evidence` 从「代码+测试节点+文档锚」
**推导**，`tests/test_capability_evidence.py` 保证发布块与树一致。本单发现该推导器有一处
**静默降级**：只要一行声明的测试节点解析不出来，等级就从 `REAL` 悄悄变成 `WITHHELD`，而
`ok=True`、套件全绿、发布表照抄——**台账自己把自己的证据抹掉，没有任何告警**。

复算（`python -m tensorbem.capability_evidence --json`，修复前 b200ae0）：

| 项 | 修复前 | 修复后 |
|---|---|---|
| REAL | 64 | **71** |
| WITHHELD | **7** | **0** |
| 其余（STUB/MOCK/MISSING/STALE_CLAIM） | 0 | 0 |
| 合计 | 71 | 71 |

7 行 WITHHELD 里 6 行的原因是 `named test node missing`：它们的测试**存在且全绿**，
只是名字写法（`Class::test`）解析器不认识；第 7 行（KCS 几何锚）指向的文档
`docs/m6656/README.md` **整份文件不存在**。

## 2 怎么做的（根因与修法）

1. **根因 A — 解析器只认裸函数名。** `_test_nodes()` 用 AST 收集 `test_*` 函数名，
   于是注册表里按 `pytest --collect-only` 写法登记的 `TestCLI::test_all_cases`
   永远解析不到。修法：同时登记 `Class::method` 与裸名两种形式（唯一性规则不变，
   仍拒绝退化成「这个文件里有某个测试」）。
2. **根因 B — 引用失效被当成「还没做」。** `grade()` 原本对「测试文件不存在」和
   「文件在、节点没了」不作区分，两者都并入 `missing_tests` → `WITHHELD`。
   新增等级 `STALE_PIN` 并纳入 `FAILING_LEVELS`：**文件存在但节点解析不到** = 声明失效 → 红；
   **文件不存在** = 测试确实没写 → 仍 `WITHHELD`（诚实）。文档同理：
   声明了却没有该文件/该锚点 = `STALE_PIN`；压根没声明文档 = `WITHHELD`。
3. **修 3 处真错引**（不是改名解析能救的）：`TestEncounterFrequency` 类不存在，实际归属
   `TestZeroSpeedDegeneration`；`TestSeries60::test_series60_hull_properties` 不存在，实际是
   `test_series60_Cw_positive` / `test_series60_Cw_zero_at_zero_speed`；KCS 行的文档锚改到
   `docs/m14775/hull_reference_quadrature.md`（其中记录 KCS-class Cb=0.651098 vs 发布 0.651，
   0.015 %）。
4. **独立核验「真有断言」**（防「仅 import 就算钉住」的假证据）：用仓库既有静态扫描器
   `scripts/audit_module_pins.py` 重算这 9 个被提升模块的断言级节点，结果全部 ≥2：
   `figure_integrity 17`、`finite_depth_green 15`、`haskind2d 2`、`net_source_reg 3`、
   `neumann_kelvin 18`、`rankine_wave 33`、`series60 5`、`stf_fullspeed 12`、`kcs_hull 6`。
5. 文档块用 `--write` 重新生成（禁止手改）：`docs/audit_baseline.md` 8 行 8 改，全部是
   WITHHELD→REAL 与锚点替换。

## 3 计算结果 / 证据（零手抄，全部程序化）

- `docs/m16488/audit_before.json` — 修复前（b200ae0 代码）矩阵，7 行 WITHHELD，逐行列出
  `missing_tests` / `missing_docs`；
- `docs/m16488/audit_after.json` — 修复后矩阵，71 REAL；
- `docs/m16488/pin_probe.json` — 断言级节点扫描（模块 × 测试节点）；
- `docs/m16488/evidence.json` — 上面三份的汇总 + 跑测计数；
- `docs/m16488/fig_evidence_drift_repair.png` — 左：等级分布前后对比；右：9 个被提升模块的
  断言级节点数（证明它们从来就被钉住，是台账没看见）。

## 4 测试（先红后绿）

- `tests/test_capability_evidence.py`：**18 passed**。新增 4 个负面控制/守卫：
  `test_class_qualified_node_ids_resolve`（`Class::test` 必须解析）、
  `test_unresolvable_node_on_a_supported_row_fails_instead_of_degrading`
  （文件存在节点不在 → `STALE_PIN` 且 `ok=False`；文件不存在 → 仍 `WITHHELD`）、
  `test_documentation_anchor_must_exist_to_count`（文件缺失/锚点缺失均 `STALE_PIN`）、
  `test_no_published_row_carries_an_unresolvable_citation`（发布注册表零失效引用）。
- 被提升行引用的 6 个测试文件：`test_stf_haskind_reg.py` `test_finite_depth_green.py`
  `test_neumann_kelvin.py` `test_rankine_wave_resistance.py` `test_figure_integrity.py`
  `test_wigley_kcs_kochin_energy.py` → **170 passed in 114.50s**。
- `python -m tensorbem.capability_evidence` 退出码 0（`failing rows == []`）。

## 5 注意点（局限）

- 本单**不新增物理能力**：被提升的 7 行原本就有代码、绿测试与文档，本单只修「台账看不见」。
  因此这些行的物理结论（含 KCS 行的 Cwp 欠缺、Neumann-Kelvin 与 STF 的适用范围）维持原状，
  `note` 里的 WITHHELD 声明（KCS 预报 vs EFD）**没有**被本单解除。
- `STALE_PIN` 只判「引用能否解析」。它**不**证明被引节点断言的是该模块的物理量——后者仍需
  `scripts/audit_module_pins.py` 这类 AST 扫描（本单对 9 个模块跑了，见上）。
- 等级全部 REAL ≠ 精度全部合格：`docs/audit_baseline.md` §5 的 Ursell 10.6 % honest-FAIL 仍在，
  本单不粉饰。

## 6 误差判定

| 检查量 | 基准 | 实测 | 误差 | <3 %? |
|---|---|---|---|---|
| 发布等级 = 推导等级 | `document_drift() is None` | None | 0 | ✅ 是 |
| 失效引用数（发布注册表） | 0 | 0 | 0 | ✅ 是 |
| 修复前 6 行「缺失节点」实为存在且绿 | 6/6 行可解析 | 6/6 | 0 | ✅ 是 |
| `capability_evidence` 失败行 | 0 | 0（退出码 0） | 0 | ✅ 是 |
| 断言级钉住模块数 | 9/9 有 ≥1 断言节点 | 9/9（最少 2） | 0 | ✅ 是 |