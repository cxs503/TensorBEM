# CASE 池图件欠账修复：annulus / ellipse 已复原，cylinder 定位到 schema 漂移（m21302）

**分支** `agent/m21302` · **仓** TensorBEM · **只读扫描**：/srv/tensor/agent/proposals、tensor/reports 局限节、feature_ledger、各仓 dev_report_*。
**锚点核对**：`CASE-BEM-RIGID-BODY-MASS` 的 feature_ledger 状态虽为 `open`，但仓内该锚的算例机（`benchmarks/casepool_bem_rigid_body_mass.py` + `results.json`/`verdict.json`/`CASE.md` + 三张图 + `ADVERSARIAL_REVIEW.md`）**已完备且 verdict=PASS**（worst accepted entry 相对误差 **1.415%** < 3% 容差），再跑一遍只是复现、不是增量，故**不重做**（建议平台侧把该锚状态推 `verified`）。

## 1. 选什么活：CI 红 + 文档承诺的图件缺失

`pytest tests/test_case_pool_registry.py` 在本次开工时**红**：

```
assert {'CASE-BEM-ANNULUS-LAPLACE': [...3 张], 'CASE-BEM-ELLIPSE-2D-ADDED-MASS': [...3 张],
        'CASE-BEM-LAPLACE-CYLINDER': [...5 张]} == {}
```

即注册表（`case_pool_registry.CASES`）为 3 个锚声明了 11 张图，磁盘上 0 张；而三个 case 的 `report.md`/`docs/CASE-*.md` 正文里**已经画着这些图的 markdown 链接**——数字在、图没了，报告读起来是破图。这是「验证欠账」里最廉价也最硬的一类：不涉及物理判据，纯证据链完整性，却直接把主 CI 钉红，掩盖真回归。

## 2. 怎么做的

| 锚 | 做法 | 结果 |
|---|---|---|
| `CASE-BEM-ANNULUS-LAPLACE` | 全量重跑驱动 `python -m tensorbem.benchmarks.eccentric_annulus`（73 s，单进程） | verdict `true`，37 checks / 0 failed；**`results.json`、`report.md` 逐字节不变**（git diff 为空）→ 图件来自与判决同一批数字，且判决可复现；`figures/{convergence,clearance_sweep,field}.png` 复原 |
| `CASE-BEM-ELLIPSE-2D-ADDED-MASS` | 全量重跑 `--output benchmarks/verified/case_bem_ellipse_2d_added_mass`（39 s） | overall `PASS`，33 checks，预算网格最差相对误差 **8.998e-4**；`results.json` 仅 `generated_at_utc` 与 `wall_seconds` 变化，**数值全同**；3 张图复原 |
| `CASE-BEM-LAPLACE-CYLINDER` | 本地 120 s 被 harness 收割 → 递交 `run_compute` cj3326（节点 n315，rc=0，4.7 min，8 核） | **未复原**：fetch 取回的是运行中快照（同一 tar 二次 fetch 未刷新），节点上生成的 `figures/` 未回到本工作树。见 §4 |
| 新增守卫 | `tests/test_case_pool_registry.py::test_registered_figures_are_real_png_files` | 注册图必须是**真 PNG**：签名 + `IEND` 尾块 + `IHDR` 宽高 ≥ 200×150 + 体积 > 5 kB。原测试只查 `is_file()`，一个 0 字节占位文件即可绿 |

新增维护工具 `scripts/regen_case_pool_figures.py`：从**已提交** `results.json` 调驱动自带的 `plot_*` 纯函数重渲染图件（`--all`），使图不能与判决脱钩。**该工具在 cylinder 上暴露了真缺陷**（§4），这是本次最可复用的产出。

## 3. 实测（复现命令）

```
PYTHONPATH=src python -m tensorbem.benchmarks.eccentric_annulus            # 73 s, verdict true, 37/37
PYTHONPATH=src python -m tensorbem.benchmarks.case_ellipse_2d_added_mass     --output benchmarks/verified/case_bem_ellipse_2d_added_mass            # 39 s, PASS, 33 checks
PYTHONPATH=src python scripts/regen_case_pool_figures.py --all              # 图件重渲染
PYTHONPATH=src python -m pytest tests/test_case_pool_registry.py -q
```

开工时 `test_every_registered_case_resolves_its_paths` 红（3 锚 / 11 图缺失）；收工时同一用例**只剩 1 锚 / 5 图**（cylinder）。新增 PNG 守卫对图件完整的 4 个锚真跑、对图件不全的锚 `skip`（缺图由既有用例唯一裁决，不制造第二个红灯）。

## 4. 定位到的真缺陷（cylinder，未修完，honest）

`scripts/regen_case_pool_figures.py CASE-BEM-LAPLACE-CYLINDER` 抛：

```
File "src/tensorbem/benchmarks/case_cylinder_stream.py", line 3141, in plot_near_wall
    [row["max_relative_velocity_error_linear"] for row in level["sweep"]]
KeyError: 'max_relative_velocity_error_linear'
```

即**已提交的 `results.json` 与驱动的绘图代码 schema 漂移**：`sweep` 行里没有 `max_relative_velocity_error_linear` 这个键（后续单改过 near-wall 扫掠的字段名或结构，results 未随之重生成）。后果不只是图缺失：说明该锚**提交的数值工件与其渲染/文档层已经不同步**，而注册表把这份工件当判决源（`artifact_payload()` 读它）。因此 cylinder 的图**不能**由「已提交工件」忠实复原，只能由一次全量重跑复原；本次重跑在节点上成功（cj3326 rc=0），但取件通道给的是运行中快照，图未落地本树。

**未完成**：cylinder 的 5 张图 + 其 `results.json`/绘图代码的 schema 归一（应由一次全量重跑 + 逐键对比完成）。**不得**用占位图或改注册表删声明来「修绿」。

## 5. 注意点 / 局限

- 本机 `run_shell` 硬顶 120 s，超过即整组收割：椭圆 39 s、annulus 73 s 能本地跑完，cylinder 全量不行 → 计算递交 `run_compute`；**取件须等终态后 fetch，运行中 fetch 的快照会被复用**（本次踩到）。
- `import tensorbem` 在本机解析到 site-packages 安装快照，全部命令前置 `PYTHONPATH=src`。
- 图件复原后 `results.json` 的时间戳会更新（ellipse），这是驱动重跑的自然结果，数值未变，已逐键确认。
- 本单未触碰 TensorPlatform、未动 main。
