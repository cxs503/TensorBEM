# 文献参考常数的出处闸门 + 能力台账红灯修复（m14725 / TensorBEM）

**仓** TensorBEM · **分支** `agent/m14725` · **类型** 缺陷修复（主分支门禁红灯）+ 新增结构性防造假能力
**工件** `src/tensorbem/reference_provenance.py`、`tests/test_reference_provenance.py`、
`docs/reference_provenance.md`、`docs/audit_baseline.md`（第 4 节块重生）
**复现** `PYTHONPATH=src python -m pytest tests/test_reference_provenance.py tests/test_capability_evidence.py -q`

## 1. 案例介绍（问题背景与仓库几何）

TensorBEM 的验证结论全是**误差百分比**：某附加质量 vs WAMIT、某 RAO 峰值 vs Journee (1992)、
某排水量 vs 已发表 OC4 值。百分比的价值等于其分母的价值；仓库的铁律是"零手抄"。

但现有的能力台账（`docs/audit_baseline.md` 第 4 节，由 `tensorbem.capability_evidence` 派生）
只审查**代码 / 测试 / 文档**三件套，**从不检查驱动脚本拿去比较的常数来自哪里**。这是一个
结构性盲区：把常数改大改小，报告里的误差数字跟着走，测试一条都不会红。

本单开工时的机器可核事实（两处红灯 + 一处盲区）：

| 项 | 状态（本单开工时 `origin/main` @ b8194ed） |
|---|---|
| `pytest tests/test_capability_evidence.py` | **2 failed / 12 passed**（主分支红灯） |
| 失败 1 | `test_uncapability_baseline_has_no_untested_module_without_a_reason`：13 个模块未被任何测试引用且无备案理由 |
| 失败 2 | `test_published_block_matches_the_tree`：文档块陈旧（84 core modules vs 实际 89） |
| 能力台账覆盖率 | 45 rows → 实际 46 rows；m14571 / m14560 / m14514 三单新增的 5 个核心模块**一行都没有** |
| 文献常数来源 | 全仓 **15** 个模块级文献常数，**0** 个被任何门禁检查；其中 14 个是回忆值（WAMIT/Journee 表） |

物理背景一句话：这不是新求解器，而是**证据基础设施**。它管的是"验证脚本凭什么相信自己的参照值"。

## 2. 怎么做的（方法 / 流程 / 关键设置）

### 2.1 新能力：`tensorbem.reference_provenance`（文献常数的出处闸门）

* **扫描**：AST 遍历 `src/tensorbem/**`，收集**模块级**、名字里带文献标记
  （`WAMIT|JOURNEE|EFD|EXPERIMENT|LITERATURE|PUBLISHED|MEASURED|OBSERVED|REFERENCE_DATA`，
  大小写不敏感）且值为数字字面量 / 数字映射的常量。函数体内局部量不扫——局部量属于算法
  （Bessel 零点、Gauss 横标），模块级量才是"驱动脚本的参照答案"。
* **分档**只有两种：
  - `CITED`：数值由本仓交付的文档发表，且登记 `(路径, 数字针)` 证据对；每次审计**重新读文件**，
    引文腐烂=红灯，不是脚注。
  - `UNSOURCED`：回忆值。留在登记表里并**必须**写 note，在生成文档里公开为债务；
    这条状态**永远不算证据**。
* **闸门方向**：`undeclared`（新出现未登记的文献常数）、`stale`（登记的常数已消失）、
  `unverified`（引文不再含该数字）任一非空 → 退出码 1 / 测试红。即**新增一条回忆参照表会被
  机器拒绝**，这正是四份遗留 Wigley III 驱动当初能悄悄累积的方式。
* **文档生成**：`docs/reference_provenance.md` 内嵌 `<!-- reference-provenance:begin/end -->` 块，
  由 `--write` 程序化生成；手改文档=红灯（与能力台账同款棘轮）。

### 2.2 修复主分支红灯（能力台账）

1. 为 13 个"未被测试引用且无理由"的驱动/命名空间补备案理由（每条都指明其物理由哪个**已评分**
   核心模块 + 哪份测试钉住，`test_every_untested_reason_is_still_true` 会校验理由里点名的
   `tests/*.py` 真实存在）。
2. 为 m14571/m14560/m14514 新交付的 4 个模块补能力行（`diffraction-drift-maturity`、
   `qtf-slow-drift`、`fmm-acceleration`、`flexural-gravity-numeric`），每行给出模块、符号、
   具体测试节点 id、文档锚点；等级仍由代码/测试/文档证据**派生**，不是自报。
3. `tensorbem.pekeris_acoustics` **不**补行：仓内没有任何文档锚点（`grep Pekeris docs/**.md` 为空），
   按"没有出处就不评分"的纪律留在 unowned 清单里，如实记为剩余欠账。
4. `python -m tensorbem.capability_evidence --write` 重生文档块（84 → 89 core modules）。

### 2.3 工具链

Python 3.11 + numpy；纯静态分析（AST）+ 一次 pytest，无重计算（本机 load15 ≈ 116 ≥ 100，
按容量纪律只做轻量活）。未起任何求解器作业。

## 3. 计算结果（机器生成，零手抄）

| 量 | 本单开工（b8194ed） | 本单交付 | 来源命令 |
|---|---|---|---|
| `test_capability_evidence.py` | 2 failed / 12 passed | **12 passed** | pytest |
| 过渡态（新增自有审计后合计） | — | **26 passed / 0 failed** | pytest 两文件 |
| 能力台账行数 | 46（含 1 WITHHELD） | **50（46 REAL + 4 新 REAL）**，failing rows = 0 | `capability_evidence.audit()` |
| 未引用且无理由模块 | 13 | **0** | `untested_modules()['unexplained']` |
| 陈旧备案理由 | 若干（文档 84 vs 89） | **0** | `untested_modules()['stale_reasons']` |
| 文献常数扫描 | 无此机制 | **15 声明 / 1 CITED / 14 UNSOURCED / 0 undeclared / 0 stale / 0 unverified** | `reference_provenance.audit().summary()` |

新增的 4 行等级（派生结果，非自报）：

| capability key | level | 解析到的测试节点 | 缺失项 |
|---|---|---|---|
| `diffraction-drift-maturity` | REAL | 2 | 无 |
| `qtf-slow-drift` | REAL | 2 | 无 |
| `fmm-acceleration` | REAL | 2 | 无 |
| `flexural-gravity-numeric` | REAL | 2 | 无 |

14 条 UNSOURCED 债务全部落在四份遗留 Wigley III 驱动
（`wigley3_complete_verification` / `wigley3_freq_verification` / `wigley3_freq_domain_verification` /
`wigley_diffraction_radiation`）里，驱动自身注释就写着 "approximate values from experimental data"；
仓内无任何数字化 WAMIT/Journee 表，故一律记债务、**不作证据**。唯一 CITED 是
`tensorbem.qtf_drift.SEMISUB_PUBLISHED_DISPLACEMENT = 13473 m³`（DeepCwind/OC4），
由 `docs/qtf_drift_oc4.md` 发表该数字并给出 Robertson et al. (2014) 出处与 1.8% 比值。

## 4. 结果图

**本单不产图，理由**：本单是证据基础设施（AST 静态审计 + 台账派生），没有场量、没有随参数变化的
物理曲线；唯一的量化对象是"红灯数/债务数"这类离散计数，已在上表逐项列出。故无云图、无对照曲线。
闸门是否真的会咬人，由负控测试给出（`tests/test_reference_provenance.py` 的 6 条负控：
合成仓里新增 `_WAMIT_A33 = 0.42` → 被抓；`JOURNEE_RAO` 映射 → 被抓；注释/字符串里的标记 →
不误报；引文不含该数字 → 拒收；登记的常数消失 → stale；无 note 或 CITED 无证据 → 构造即抛错）。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **坑（已修）**：在测试文件里直接写出 `tensorbem.benchmarks.<driver>` 这样的完整模块路径，
   会被 `capability_evidence.module_reference_index` 判为"该驱动已被本测试引用"，从而让我刚补的
   备案理由立刻变成 `stale_reason`。修法：回归钉读登记表，不在测试里拼模块路径。这条对后续
   写负控的人同样成立。
2. **坑（已修）**：`tests` 节点名解析只认**裸函数名**（类方法取 `test_<name>`，不带类限定），
   写成 `Class.test_name` 会解析失败并把行降级为 WITHHELD。
3. **局限**：闸门只看**名字**。故意起名 `X_REF_TABLE` 的回忆值能绕过——这是命名约定的代价，
   换来的是零误报与零维护语义表（无领域词条硬编码）。真正的补强方向是把"驱动报告的误差数字"
   与"该数字被哪个测试/工件复算"绑定（`doc_claims` 已在 Ursell 表上做了同样的事，尚未覆盖驱动级）。
4. **局限**：14 条债务**没有被消除**，只是被显式登记并排除出证据面。消除需要数字化 WAMIT/Journee
   原始表并落 `benchmarks/verified/`，本机无网络，属下一单的实打实工作量。
5. **适用范围**：本模块只管模块级文献常数；算法内部的表化常量（Bessel 零点、Gauss 横标）不在其列。

## 6. 误差判定

本单的判据是**离散的门禁状态**，不是物理量误差，故逐项给出"是否<3%"的物理基准不适用，
按纪律声明原因：

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| `test_capability_evidence.py` 通过率 | 100% | 12/12 | PASS（开工时 12/14=86%） |
| 能力台账 failing rows | 0 | 0 | PASS |
| 未引用且无理由模块 | 0 | 0 | PASS |
| 陈旧备案理由 / 未解析节点 | 0 | 0 | PASS |
| 文献常数 undeclared / stale / unverified | 0 | 0 / 0 / 0 | PASS |
| 已登记债务 | 如实登记 | 14 UNSOURCED，逐条 note，公开于文档 | PASS（honest FAIL：债务未消除） |
| 物理量误差 <3% | 不适用 | 本单未产生任何物理量 | — 声明原因：纯证据基础设施单 |

**未完成**：`tensorbem.pekeris_acoustics` 无文档锚点，未入台账（如实留 unowned）；14 条回忆常数
仍是债务。