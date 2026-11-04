# m3541：把两个「永跳」的验证门救活 —— 产物未入库 ⇒ 守卫永久空转

**仓**：TensorBEM（worktree `/root/tp-agent/work/m3541`，基线 `4f5b73e`） · **分支** `agent/m3541`
**任务类型**：验证欠账（自找活） · **日期**：2026-11-04
**工件**：`outputs/finite_depth_energy_identity/{convergence.json,*.png}`、
`outputs/ursell_reproducibility_study.json`、`outputs/ursell_reproducibility_table.md`、
`docs/figures/ursell_reproducibility.png`、`docs/m3541/figures/gate_liveness.png`

EDIT_WHITELIST: tests/, docs/, outputs/, src/tensorbem/benchmarks/

---

## 1. 欠账是什么（问题的类，不是两行坏数据）

TensorBEM 有 7 个测试模块用「产物存在则断言、不存在则 `skip`」的写法守护文档里的物理结论：

```python
@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated")
```

`skip` 在 CI 汇总里与 `pass` 无法区分。审计发现：`outputs/` 目录在 HEAD 上**零个被跟踪文件**
（`git ls-tree -r HEAD | grep -c '^outputs/' → 0`），却有 5 个 test function 的门指着它：

| 测试模块 | 常量 | 产物 | 基线 HEAD 上的命运 |
|---|---|---|---|
| `test_finite_depth_energy_convergence.py` | `ARTIFACT` | `outputs/finite_depth_energy_identity/convergence.json` | 3 个 test 永跳 |
| `test_ursell_reference_pinning.py` | `STUDY_JSON` | `outputs/ursell_reproducibility_study.json` | 2 个 test 永跳 |

`test_ursell_reference_pinning.py:101` 里甚至写着注释 `# pragma: no cover - artifact is committed`
——注释声称已入库，事实为零跟踪。被这两道门守的物理是：有限水深辐射能量恒等式 1% 偏移的收敛归因、
以及 Ursell 潜没圆柱文档数字（7.72e-3）与实测（8.51e-3）的 honest-FAIL 归因。两条都是有文献/解析基准的
验证欠账，只是**守卫从没运行过**。

## 2. 怎么做的

1. **先造判据（负控）**：新增 `tests/test_artifact_gate_liveness.py`，用 AST 遍历 `tests/*.py`，
   不执行模块地求值模块级 `Path` 常量链（`Path(__file__).resolve().parents[1] / ...`），
   要求每个指向 `docs/ outputs/ tests/data/ examples/` 的产物常量 **(a) 盘上存在、(b) 被 git 跟踪**。
   在修复前运行 → 恰好点名上述两条（负控成立），其余 5 个已入库产物的门不误报。
2. **补产物**：用仓库自带的生成器重跑两项研究（2-D、单进程、`OMP_NUM_THREADS=1`：
   `python -m tensorbem.benchmarks.finite_depth_energy_convergence`、`python examples/ursell_reproducibility_study.py`），
   产物落到测试期望的路径并入 git。
3. **修一处真实的复算漂移**：`docs/finite_depth_energy_identity/report.md` 由 `build_tables.py --inject`
   从 JSON 程序化注入，重跑后 `test_report_tables_match_the_artifact_verbatim` 红：深水工况的
   `offset_limit` 在浮点噪声里符号翻转（`+0.000e+00` → `-4.337e-19`）。修法不是改数，而是在
   `build_tables.py` 加 `_NOISE_FLOOR=1e-12`，|v|<1e-12 一律打印为无符号零——**1e-19 的符号不是物理量**。
   修后重新注入，report.md 与修复前**逐字节相同**（`diff` 空，rc=0），说明这不是数值改动而是可复现性修复。
4. **把「永跳」变成可执行断言**：两道门恢复后无需改断言即全绿，未放宽任何判据。

## 3. 结果

### 3.1 守卫存活率

| 状态 | `test_finite_depth_energy_convergence` | `test_ursell_reference_pinning` | 合计 |
|---|---:|---:|---:|
| 干净检出 `git archive HEAD`（修前，`docs/m3541/baseline_skip_run.log`） | 3 skipped | 2 skipped | **9 passed, 5 skipped** |
| 本分支（修后） | 0 skipped | 0 skipped | **14 passed, 0 skipped** |

干净检出复现命令（零）：
`git archive HEAD | tar -x -C /tmp/m3541_base2 && cd /tmp/m3541_base2 && PYTHONPATH=$PWD/src python -m pytest tests/test_finite_depth_energy_convergence.py tests/test_ursell_reference_pinning.py -q -rs`

### 3.2 被救活的门给出了什么（重跑结果）

有限水深能量恒等式偏移（垂荡，`ka=0.6`，每面元 8 点）随面元收敛：

| 水深 | 观测阶 $p$（中位） | 外推偏移极限 | 近/远场极限相对差 |
|---|---:|---:|---:|
| h/a=3.2 | 0.976 | −4.9e-5 | 1.02e-4 |
| h/a=5.0 | 0.982 | −2.9e-5 | 6.9e-5 |
| h/a=20.0 | 0.970 | 0（噪声内） | 3.7e-5 |

即 64 面元时 +1.06% 的偏移是**一阶离散误差**，`N=256` 时降到 0.27%，外推到 ~5e-5。
`B=2ρg c_g|η_a|²` 的系数无需修改——原报告的结论被独立复算确认。

Ursell 潜没圆柱（`ka=0.6`, `a=1`, `f=2`）：

| 面元 | 附加质量误差 | 阻尼误差 | 近/远场比 |
|---:|---:|---:|---:|
| 64 | 1.674e-2 | 2.542e-2 | — |
| 128 | 8.513e-3 | 1.336e-2 | 1.00534 |
| 256 | 4.293e-3 | 6.841e-3 | — |

收敛阶 0.975 / 0.988（附加质量），0.928 / 0.966（阻尼），一阶；节点数 6/12/24 结果三位一致，
复现了 `min(panel_nodes, 6)` 的截断结论（光滑项不再细化）。

## 4. 误差判定（逐项 vs 基准，是否 <3%）

| 关键量 | 基准 | 本分支实测 | 相对误差 | <3% |
|---|---|---:|---:|---|
| 有限水深垂荡阻尼（N=256，h/a=3.2） | 能量恒等式近场值 2265.382 | 2265.151（远场） | 1.0e-04 | ✅ |
| 有限水深外推偏移（h/a=3.2） | 0（恒等式） | −4.9e-5 | — | ✅ |
| Ursell 附加质量（N=128） | Ursell 多极解析解 2545.17 | 2566.83 | 8.51e-03 | ✅ |
| Ursell 阻尼（N=128） | Ursell 多极解析解 2718.15 | 2754.47 | 1.34e-02 | ✅ |
| 守卫存活率 | 期望 0 skipped | 0 skipped | 0 | ✅ |

（解析基准值取脚本写入 JSON 的 `reference_*` 字段，非手抄。）

## 5. 注意点与局限

- **`outputs/` 不是被忽略目录**，只是历史上无人提交；本次把产物落到测试**已经期望的路径**，
  以避免改测试路径这种更 invasive 的改动。若将来有人把 `outputs/` 加进 `.gitignore`，
  `test_artifact_gate_liveness.py` 会立刻变红——这正是它存在的意义。
- AST 扫描只覆盖「模块级 `Path` 常量」这一种写法；在函数体内动态拼路径的门（本仓目前没有）不会被发现。
- 门只在 `git` 可用时检查「是否入库」；无 git 环境退化为只检查存在性。
- 有限水深与 Ursell 两项研究都是 2-D、单进程、分钟级；未引入任何新物理实现。
- 未做的事：没有审计 `artifacts/`（git-ignored，按设计不入库）下 CI 运行时生成的门。

## 6. 复现

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_energy_convergence \
    --json outputs/finite_depth_energy_identity/convergence.json \
    --figures outputs/finite_depth_energy_identity
PYTHONPATH=src python3 examples/ursell_reproducibility_study.py
PYTHONPATH=src python3 docs/finite_depth_energy_identity/build_tables.py --inject
PYTHONPATH=src python3 docs/m3541/make_figures.py
PYTHONPATH=src python3 -m pytest tests/test_artifact_gate_liveness.py \
    tests/test_finite_depth_energy_convergence.py tests/test_ursell_reference_pinning.py -q
```