# 文档-代码一致性闸门的可移植性修复（m2918）

**仓**：TensorBEM　**分支**：`agent/m2918`　**日期**：2026-11-03
**工件**：`docs/m2918/outputs/doc_claim_gate/{results.json, doc_claim_gate.png, make_figure.py}`

## ① 案例背景

TensorBEM 有一条「发布数字必须机器可查」的闸门：`src/tensorbem/doc_claims.py`
把 `docs/marine_bem_benchmarks.md` §4.6 与 `README.md` 里印出的 23 个 Ursell
基准数字解析出来，和 `artifacts/ursell_doc_claims.json`（由
`python -m tensorbem.doc_claims --write` 实测生成）逐个比对。
设计意图写在模块 docstring 里：文档漂移要让构建立刻变红，而不是留到下一次审计的脚注里。

这条闸门有两个结构性缺陷，**它自己却检测不到**：

1. **根目录靠数父级得到**：`REPO_ROOT = Path(__file__).resolve().parents[2]`。
   在 `src/` 布局的源码树里这正好是仓库根；但一旦包以**非 editable** 方式安装
   （本机 `site-packages` 就是这种情况），同一行会解析到
   `/usr/lib/python3.11`，于是 `MARINE_DOC` 变成
   `/usr/lib/python3.11/docs/marine_bem_benchmarks.md` —— 闸门直接
   `FileNotFoundError`，根本跑不到比对那一步。
2. **比对基准是 gitignore 的产物**：`.gitignore` 里 `artifacts/` 被排除，
   干净 clone 里没有 `ursell_doc_claims.json`，`test_doc_claims.py` 里三条
   测试无条件 `read_text()` 该文件 → 新克隆与 CI 的 `pytest` 步骤必然红。
   （本仓已有 `test_finite_depth_energy_convergence.py` 的 `skipif` 范式，
   但 `skip` 等于**放弃验证**，不是修法。）

再加上闸门真正跑起来后暴露的两条**真实文档欠账**：README 第 17 行仍写
`\(7.7\times10^{-3}\)`（实测 8.513e-3，偏 9.55%），以及近场/远场阻尼比这一列的
**容差推导错误**（比值印到 `1.0053`，其偏差只有两位有效数字，半 ULP = 5e-5，
相对偏差应为 9.4e-3，原用 5e-3 太紧，把正确舍入的 `1.0053` 判成 FAIL）。

## ② 方法

改动集中在 `src/tensorbem/doc_claims.py`、`tests/test_doc_claims.py`、
`tests/data/ursell_doc_claims.json`、`README.md`，**求解器数值内核一行未动**。

1. **根目录发现**（`repo_root()`）：候选序列 =
   `$TENSORBEM_REPO_ROOT` → 模块所在目录及其所有父级 → 当前工作目录及其所有父级；
   判据是同时存在 `pyproject.toml` 与 `docs/marine_bem_benchmarks.md`
   （`ROOT_MARKERS`）。找不到时抛
   `RuntimeError` 并指明要设哪个环境变量——**显式失败胜过在系统路径上
   FileNotFoundError**。
2. **产物回退**（`resolve_artifact()`）：优先用实测的
   `artifacts/ursell_doc_claims.json`，缺失时回退到**入库的**
   `tests/data/ursell_doc_claims.json`；两者都没有才报错并提示去跑 `--write`。
   入库副本由本次实测生成（`--write`，112 s），并由原有的
   「64 面元 live 复现（rel 1e-9）」测试保证它没和代码脱钩。
3. **容差按印刷位数推导**：近场/远场比值一列改为 `1.0e-2`，并在代码注释里
   写出推导（两位有效数字 → 半 ULP/量值 = 9.4e-3）。
4. **README 头条数字改实测值**：`7.7e-3` → `8.51e-3`，与本仓 2026-11-02 已在
   `docs/marine_bem_benchmarks.md` §4.6 完成的复测口径一致
   （见 `docs/ursell_benchmark_discrepancy.md`：该偏差是文档陈旧，不是求解器缺陷，
   两者都 < 1e-2 门槛）。
5. **新增 5 条回归测试**（12 → 14 条全绿）：根目录发现、环境变量覆盖、
   非 checkout 目录不得被误认、干净 clone 的产物回退链、印刷位数容差
   （`5.3e-3` 通过 / `5.6e-3` 必须被拒），以及一条**子进程级**端到端测试：
   把 `src/tensorbem` 拷到临时目录模拟 wheel 安装，断言无环境变量时报错信息里
   出现 `TENSORBEM_REPO_ROOT`，设上之后 `REPO_ROOT`、`MARINE_DOC.exists()` 均正确。

## ③ 计算结果（程序化注入，见 `results.json`）

| 项 | 值 |
|---|---|
| 发现的根目录 | `/root/tp-agent/work/m2918`（发现方式：模块父级扫描） |
| 使用的产物 | `artifacts/ursell_doc_claims.json`（实测生成；入库回退副本同时在场） |
| 比对数字个数 | 23 |
| FAIL / missing | 0 / 0 |
| 全表最大相对偏差 | 7.746e-3（近场/远场比值列，容差 1e-2） |
| 修复前闸门状态 | 2 FAIL（README 9.55% 超 1e-2 容差；比值列 7.75e-3 超 5e-3 容差） |
| 修复后闸门状态 | `all published numbers reproduce`（退出码 0） |

修复前（实测输出，2026-11-03）：

```
checked 23 published numbers
FAIL table row 128: near/far-field damping ratio: document says 0.0053, code gives 0.00534137 (relative difference 0.00775 > 0.005)
FAIL README/abstract: 128-panel A error: document says 0.0077, code gives 0.00851303 (relative difference 0.0955 > 0.01)
```

## ④ 结果图

`doc_claim_gate.png`：左图 = 23 条发布数字的 |文档值 − 实测值|/实测值（对数横轴）
与各自的容差刻度，全部落在容差内侧；右图 = 本次闸门抓出的两处红点
（README 头条 7.7e-3 vs 实测 8.513e-3；比值列容差推导错误）修复前后的数值对照。

## ⑤ 注意点 / 局限

1. **只修「闸门能否运行」，没修全部红测试**：本仓当前还有 3 条**与本单无关的
   既存红**，本单在 HEAD 的纯净解包（`git archive`）里复现了同样的三条，
   确认不是本次改动引入，按纪律**只留证不改**：
   - `tests/test_documented_benchmarks.py::test_pinned_cases_reference_existing_docs`
     —— 文档没有指向 `docs/benchmark_pins.json`；
   - `tests/test_documented_benchmarks.py::test_analytic_rankine_panel_integral_is_exact`
     —— 数值判据失败（待独立定性）；
   - `tests/test_ursell_doc_reproducibility.py::test_document_contains_the_generated_rows`
     —— 文档里 MFS 两行的数值（7.57e-4 / 1.89e-4）与当前生成值不一致。
   这三条建议作为下一单（「文档表 vs 生成值」清账）的入口。
2. **入库产物是快照，不是真理**：`tests/data/ursell_doc_claims.json` 只在
   无人跑过 `--write` 时提供基准；一旦数值内核改动，`test_live_benchmark_reproduces_the_artifact`
   会以 rel 1e-9 报红，逼人重测。这也是选「回退」而非 `skip` 的原因：`skip` 会让
   CI 永远沉默。
3. **112 s 的重测成本**：`measure()` 跑 64/128/256 三个面元（256 面元单点 88 s）。
   回退副本的存在让常规 `pytest` 只需 4 s（64 面元 live 复现）。
4. **发现逻辑的边界**：若把包装进 wheel 且仓库不在 cwd 的任何父级上，必须显式设
   `$TENSORBEM_REPO_ROOT`；子进程测试覆盖了这条路径。
5. 本机的 `site-packages/tensorbem` 是一份**非 editable 的陈旧副本**，直接
   `pytest` 会导入它而不是工作树代码；本次所有验证都用 `PYTHONPATH=src` 强制
   指向工作树（这也正是缺陷 1 会真实发生的环境，值得后续在 `AGENTS.md` 里提示）。

## ⑥ 误差判定（逐项回答「是否 <3%」）

| 判据 | 基准 | 实测 | 相对偏差 | <3%？ |
|---|---|---|---|---|
| README 头条附加质量误差（128 面元） | 代码实测 8.513030e-3 | 修复后文档 8.51e-3 | 3.6e-4 | 是 |
| 表 row128 近场/远场比值偏差 | 5.341368e-3 | 5.3e-3 | 7.75e-3 | 是 |
| 全表 23 条最大相对偏差 | 各条代码实测 | 文档值 | 7.746e-3 | 是（且全部在各自印刷位数容差内） |
| 入库产物 vs live 重算（64 面元，5 个字段） | live 运行 | 入库副本 | ≤1e-9（断言） | 是 |

**结论**：本次所有关键量相对偏差均 < 3%，闸门从「跑不起来 + 2 条 FAIL」变为
23/23 复现、退出码 0。未达标项不存在，但**既存 3 条无关红测试如实登记、未修**，
不计入本单达标。