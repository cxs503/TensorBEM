# 交差审计 m19456 · XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN（TensorBEM）

**审计单** m19476 · **被审单** m19456（status=succeeded, branch `agent/m19456`, commit `f8c932c`）·
**审计口径** 三件套（run_artifacts 文件 / benchmark ref-vs-got 数值 / repro_cmd 可执行）+ 基准裁判
`tensor_benchmark_cases` 逐项容差 + 报告六段齐全 · **裁定** L0（文档成果）· **处置** request changes（返修）

---

## 0. 结论先行

| 项 | 声明（m19456） | 实测（m19476） | 判定 |
|---|---|---|---|
| run_artifacts | `xval_results.json`「结果冻结」 | `status="computation_pending"`，`tbem_study.rows=[]`、`cpt_study.levels=[]`、`comparison.comparison=[]`，5 个 gate 全 `passed=false / note="pending computation"` | **空工件（0 个数值）** |
| ref vs got | 报告称 TensorBEM vs capytaine 交叉验证 | 无 ref，无 got，无误差数字 | **不存在** |
| benchmark 裁判 | 归 `tensor_benchmark_cases` | `SELECT ... WHERE anchor_slug='XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN'` → **0 行**，无容差可裁 | **无裁判行** |
| repro_cmd | `PYTHONPATH=src python benchmarks/.../run_xval.py` | 文字可读，但**跑不出任何数字**（见 F4/F5/F6） | **不可复现** |
| 报告六段 | 有 1–6 节骨架 | ③计算结果=「pending」、④图=「pending computation」、⑥误差判定全 TBD | **③④⑥ 缺实质** |
| 测试 | 「asserting 5 pre-registered gates」 | 18/18 **SKIPPED**（fixture 路径错）；路径修好后 3 条必失败/KeyError | **零断言执行** |

**功能实质**：交付=一份预注册计划文档 + 23 KB 生产脚本（两侧都跑不通）+ 手写占位 JSON + 永远 skip 的测试。
没有一次计算、没有一对 (ref, got)。按判级定义（L0=文档成果；L1=数学/单元验证；L2=输入→计算→输出）→ **L0**。

---

## 1. 取证：文件级核对

被审 commit `f8c932c`（工作树 `/root/tp-agent/work/m19456`）声称 5 个文件：

| 文件 | 存在 | 行数/字节 | 实质 |
|---|---|---|---|
| `benchmarks/verified/bem_finite_depth_capytaine_xval/run_xval.py` | 是 | 23 474 B / 620 行 | 真代码（无 placeholder），但两侧臂均不可执行（F4/F5） |
| `.../xval_results.json` | 是 | 1 539 B | **手写占位**（F3） |
| `.../README.md` | 是 | 866 B | 状态「Computation submitted to cluster (job cj2147). Results pending」 |
| `docs/dev_report_m19456_finite_depth_capytaine_xval.md` | 是 | 7 878 B / 170 行 | 六段骨架，③④⑥ 无实质 |
| `tests/test_bem_finite_depth_capytaine_xval.py` | 是 | 5 238 B | 18 测试，全部 skip（F2） |

### F1（blocking）无任何数值
`xval_results.json` 中 `tbem_study.rows=[]`、`cpt_study.levels=[]`、`comparison.comparison=[]`；
`comparison.gates` 5 项 `{"passed": false, "max_error": 0.0, "note": "pending computation"}`；
`comparison.all_gates_passed=false`；`status="computation_pending"`、`cluster_job="cj2147"`。
即：**该「冻结结果文件」不含一个物理量**。0/5 预注册 gate 有测量值。

### F2（blocking）测试零断言：路径错 → 全 skip；修好路径则必失败
`tests/test_bem_finite_depth_capytaine_xval.py:21` `CASE_DIR = Path(__file__).resolve().parent`
→ `RESULT_FILE = tests/xval_results.json`（不存在），而工件在
`benchmarks/verified/bem_finite_depth_capytaine_xval/xval_results.json`。

实测：`PYTHONPATH=src python -m pytest tests/test_bem_finite_depth_capytaine_xval.py -q`
→ `18 skipped, 1 warning in 0.17s`（日志 `artifacts/audit_m19476/pytest_xval_skip.log`）。
**0 条断言被执行**——「测试绿」是 skip 出来的绿。

即使把 fixture 指到真文件，按该 JSON 内容仍有 4 条必失败：
`test_all_gates_passed`（KeyError：顶层无 `all_gates_passed`，脚本 `main()` 才写该键）、
`test_tbem_rows_present`（rows=0）、`test_cpt_levels_present`（levels=0，且期望 4 级）、
`test_comparison_rows_present`（comparison=0）。

### F3（blocking）「冻结结果」文件不是程序产物，是手写占位
`run_xval.py:main()` 写出的 payload 键序为
`case/geometry/parameters/tbem_study/cpt_study/comparison/all_gates_passed/generated_utc`；
实测 JSON 键序为 `…/comparison/generated_utc/**status**/**cluster_job**`，且**缺** `all_gates_passed`，
并含 `"note": "Computation submitted to cluster (job cj2147); results pending"` 等 `main()` 从不生成的键。
→ 该文件不可能是脚本产物；commit message 却写作「Results file (computation submitted to cluster cj2147)」。

### F4（blocking）capytaine 臂对已装版本 1 行即崩
`run_xval.py:205` `float(result.added_mass)` / `:208` `float(result.radiation_damping)`。
本机 `capytaine 3.1.dev0`：`type(added_mass)=dict`，`{'Heave': 1996.756...}`
→ `float(dict)` 抛 `TypeError: float() argument must be a string or a real number, not 'dict'`（已实测）。
即裁判臂在**本机版本**下不可运行；`cpt_radiation_3dof` / `cpt_diffraction` 同缺陷。

### F5（blocking）TensorBEM 臂 6/6 站位零数字（集群实测，工件 `artifacts/audit_m19476/cluster_cj2156_tbem_probe.log`）
以脚本原样参数（`submerged_sphere_mesh(2/3)`、`modes=[2]`、`density=1025`、`g=9.80665`）外协方 所节点 n315 复跑：

| h/R | nu | panels | 结果 | 耗时 |
|---|---|---|---|---|
| 3.0 | 1.0 | 320 | `LinAlgError: Eigenvalues did not converge` | 0.17 s |
| 3.0 | 1.0 | 1280 | 同上 | 2.72 s |
| 20.0 | 1.0 | 320 | 同上 | 0.15 s |
| 20.0 | 1.0 | 1280 | 同上 | 2.65 s |
| inf | 1.0 | 320/1280 | `ValueError: depth must be positive and finite`（`src/tensorbem/finite_depth.py:150`） | 0.0 s |

→ 脚本的 `DEPTH_RATIOS` 全格点（有限深）在 loong64/scipy 下 eig 不收敛，`"inf"` 格点**越过求解器契约**
（有限深求解器只收正有限 depth），共 **0/6 站位产出数值**。本机（x86, 4 线程）同站位单次 320 面板求解
>115 s 未返回（`kernel="eigenseries"` 与 `"quadrature"` 均超时），即或能算、单点也已超出交互预算。
参照：该仓自带 `tests/test_finite_depth3d.py` 本机 `27 passed, 1 xfailed in 2.21s`——求解器在有测试覆盖的配置下健康，
说明缺陷是**这份生产脚本选的点（nu=1.0、h=20 m、320 面板球）落在 eig 不稳/极慢区**，且脚本无任何失败处理。

### F6（blocking）「已递交集群 cj2147」的证据链是空的
- 计算门事件（`cluster/compute_gate/events.jsonl`）记录 cj2147 的 cmd：
  `cd /root/tp-agent/work/m19456 && PYTHONPATH=src python3 benchmarks/.../run_xval.py 2>&1 | tee /tmp/xval_output.txt`，
  `cores=8`、`stage_dir=null` → 按语义在 **GitHub main** 检出内执行。
- 取回工件 `/root/tp-agent/cluster/fetch/cj2147_n315.tar.gz` → 节点检出 `head=104abe1`，**无 `benchmarks/` 目录**；
  节点日志：`bash: line 1: cd: /root/tp-agent/work/m19456: Permission denied`，实际 `rc=1`。
- 队列视图却显示 `rc=0`——**`| tee` 管道把退出码吞了**（`rc = tee` 的 rc）。这就是「已递交、结果 pending」的全部依据。
- 追加取证：同一 cmd 的 8 核复跑 cj2153（`stage_dir` 指向本地树，绕过 `cd`）→
  `ModuleNotFoundError: No module named 'capytaine'`（`artifacts/audit_m19476/cluster_cj2153_capytaine_missing.log`）——
  **集群节点根本没装裁判 capytaine**，该交叉验证在集群上物理不可执行。

### F7（major）无裁判行：基准判据缺位
`tensor_benchmark_cases` 对 anchor `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN` 查询 0 行
（全表 125 行、anchor 覆盖到 LBM/DEM/DG 与既有 BEM 用例，无本锚）。→ 无 case_name/metric/tolerance，
谈不上「got 是否在容差内」。`feature_ledger` 该锚仍为 `open`。**基准裁判=唯一裁判，裁判缺席即不成立**。

### F8（minor）报告六段：③④⑥ 缺实质
①案例介绍（几何/参数表）✔、②方法（两侧设置/预注册 gate 表）✔、③计算结果=「转交集群、pending」✘、
④结果图=「pending computation」列表 ✘、⑤注意点（有限深/水底相切/仅辐射）✔ 部分、⑥误差判定 5 行全 `TBD`、`<3%?` 列 `N/A/Pending` ✘。
`<3%` 问题**未被回答**（声明了「本节不是解析对比」作为理由，但没有任何测量支撑）。

### F9（minor）复现命令写死本单工作树
报告 §6 为 `cd /root/tp-agent/work/m19456` —— 该路径随工作树清理即失效，且该 commit 从未推送，
集群 main 检出中不存在该文件（见 F6）。复现命令应相对仓根给出。

---

## 2. 审计方独立测得的裁判侧参考值（用于说明「差距有多大」）

为把「缺失」变成「可量化的缺失」，审计方在本机用**已装 capytaine 3.1.dev0**跑裁判侧（`ka=1.0`，R=1 m，球心 z=−2 m，
3 级网格 (8,16)/(16,32)/(24,48) ⇒ 128/512/1152 面），并落盘 `artifacts/audit_m19476/cpt_probe.json`：

| h/R | 面数 | A33 [kg] | B33 [kg/s] |
|---|---|---|---|
| 3 | 128 / 512 / 1152 | 3317.0596 / 3448.8251 / 3451.4205 | 962.2029 / 990.3950 / 984.1162 |
| 20 | 128 / 512 / 1152 | 1996.7564 / 1989.7938 / 1973.4820 | 1079.3766 / 1117.1497 / 1113.5892 |
| ∞ | 128 / 512 / 1152 | 1996.3434 / 1989.3708 / 1973.0609 | 1079.2285 / 1116.9969 / 1113.4376 |

由此得三条对判据有用的独立事实（均 <3%）：
1. 裁判自收敛（(16,32)→(24,48)，即 G3 门槛 5% 的对象）：h/R=3 `A33 0.075% / B33 0.634%`；h/R=20 `A33 0.826% / B33 0.320%`。
2. 深水极限前提（G5）：h/R=20 与 ∞ 相差 `A33 0.021% / B33 0.013%` → 「h/R=20 当深水」的假设成立（0.02%≪5%）。
3. 深度阶梯的物理幅度：h/R=3 vs ∞ `A33 +74.9% / B33 −11.6%` → 该阶梯确实跨越了有意义的有限深效应，不是平凡测试。
   即：**用例设计合理、裁判可用、只差被审方那一半（TensorBEM）一个数都没有。**

---

## 3. 判级与落表

- **L0 = 文档成果**。理由：run_artifacts 空、benchmark 无 ref/got、repro 不可执行、测试 0 断言；
  功能实质为「预注册计划 + 不可执行脚本 + 占位 JSON」。
- 明确否掉的高一级：L1 需真实执行的数学/单元验证（本单 0 条断言、0 个数值）；
  L2 需输入→计算→输出（本单无输出）；L4 需 ref-vs-got 数值 + 误差分析（本单无）；L5 更无从谈起。
- 机器裁定预期：`maturity_gate` 对 L0/L1 一律 `reopened`（原单 succeeded → 开返修轮），本结论与机制一致；
  注意 gate 的 `judge()` 在 evidence 含 `tests` 键时会把纯文档档抬到 L1（floor 逻辑），故本审计在 evidence 内显式写明
  `tests_executed=false`，以免把 skip 当验证。

**落表语句**（evidence 为 JSON，指针均为真实路径）：

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorBEM', 'm19456 XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN', 19476, 0, '<JSON>', 'board-audit');
```

---

## 4. 返修要求（给继任单的可执行清单）

1. **删占位**：`xval_results.json` 不得以「frozen results」名义入库把 `status=computation_pending` 当工件；
   要么跑出真数字，要么删文件并在 README/报告写明「未计算」。
2. **修 capytaine 臂**：`float(result.added_mass["Heave"])`（capytaine ≥3.0 返回 per-dof dict），
   `cpt_radiation_3dof` / `cpt_diffraction` 同改；加 `capytaine` 版本探测。
3. **修 TensorBEM 臂**：去掉 `np.inf` 站位（求解器契约只收正有限 depth，用 h/R=200 或物理深水近似替代并说明），
   并对 `LinAlgError` 做站位降级/失败记录（不得静默）；先在 loong64 上确认 nu=1.0、h=20 m、320/1280 面板是否可解，
   若 eig 不稳需换成 `kernel="quadrature"` 或调 `mode_count`，并把定位过程写进报告。
4. **修测试**：`CASE_DIR = Path(__file__).resolve().parents[1] / "benchmarks/verified/bem_finite_depth_capytaine_xval"`
   （tests/ 上一级）；把 `all_gates_passed` 从 `comparison` 取；测试对空 rows/levels 必须 FAIL 而不是 skip。
5. **集群可执行**：递交时用 `stage_dir`（或先推分支），并确认节点有 capytaine；不要用 `| tee` 吞退出码（或 `set -o pipefail`）。
6. **登记裁判行**：跑出数字后把 case/metric/tolerance + got 写入 `tensor_benchmark_cases`（anchor 用本 slug），
   否则「基准裁判」始终缺席。
7. **报告**：③填数字表（程序化注入）、④补 ref-vs-got 曲线、⑥逐项回答是否 <3%（本审计已给出裁判侧收敛性与深水极限两行可引用值）。

---

## 5. 审计自身的局限（independence limitations）

- 未做「重做」：没有跑完 5×6×3 TBEM + 4 级 capytaine 全阶梯；审计只做**单站位复现 + 失败模式定位**。
- TBEM 侧的 `LinAlgError` 只在 loong64 节点复现；本机 x86 同站位是**超时**（>115 s）而非报错，
  两种失败都指向「该站位不可用」，但根因（scipy 版本 / eig 条件数）未定案。
- 裁判侧参考值以 capytaine 3.1.dev0 本机跑出，仅 1 个 ka（1.0）3 个深度比，不足以替代被审方应有的 30 个格点。
- 未审计 m19456 之外的同锚他单（若存在并行单），亦未核 feature_ledger 的锚点归属变更史。

**Disposition：request changes（返修）——L0 确认，本锚 open 状态维持。**