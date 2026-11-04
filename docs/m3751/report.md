
# PR #128 治脏入库：agent/m3181 → main 的三方合并（语义并集，零静默丢失）

## ① 案例介绍
TensorBEM 的注册表 `src/tensorbem/benchmarks/declared_numbers.py` 把文档里每一个"声明数字"变成可执行断言。
两条独立工作线同时扩展了同一批三维自由液面基准（Hulme 半球的表 10、CHIEF 表 11、Wang/Linton 潜没球的表 12）与同一份文档
`docs/marine_bem_benchmarks.md`，于是 PR #128（分支 `agent/m3181`，旧 SHA `55db3976dc283e9522334f2ab7309fc655f3ae27`）
对 main（`b420b2ee`）变脏（mergeable=false）。

| 量 | PR 分支 | main |
|---|---|---|
| 注册项行数 | 85 | 84 |
| 分组 | hulme3d / hulme3d_chief / wang_linton3d | hulme_hemisphere / submerged_sphere / hulme_chief |
| 冲突文件 | `declared_numbers.py`、`tests/test_declared_numbers.py` 文本冲突；`docs/marine_bem_benchmarks.md` 自动合并 | — |

## ② 怎么做的
1. 三方合并：`git merge` 以 `origin/agent/m3181` 为 ours、`origin/main` 为 theirs，基线 `4f5b73e7`。
2. 三个文本冲突逐文件按**语义并集**解冲（脚本 `docs/m3751/resolve_conflicts.py`，幂等且带断言，缺一不静默通过）：
   - 导入别名：两侧给同一模块起了不同别名，合并体两侧都会用到 → 统一为 main 的 `hemisphere_benchmark` / `sphere_benchmark`，并把分支侧 3 处调用点改名；
   - `DECLARED_NUMBERS`：保留 main 的内联表 10–12 行，并把分支的生成块 `_free_surface_3d_rows()` 串接上去；
   - `_run_group` 分派：两侧的 group 处理器**全部保留**（键互不重叠）。
3. **抓出 git 静默合并的语义碰撞**：两侧都定义了模块级 `_chief_rows`（分支＝无参注册行工厂，main＝有参 study 展平器）。行级合并无冲突标记，后者会覆盖前者 → 把展平器改名 `_chief_study_rows`。
4. 两侧对**同一张表**各自注册了一份数字：分支生成族仍带 m3150 之前的旧口径（B11 观测阶按理想网格 `log 2`、转动耦合比旧舍入）。为不复活被订正的值、也不让合并后测试转红，把 4 个单元对齐到 main 的 m3150 复算口径（脚本 `docs/m3751/reconcile_table_cells.py`），并把分支的 B11 观测阶改为按**实测平均棱长比** `log(h0/h1)` 求。
5. 机械证明：`docs/m3751/verify_merge.py` 做三项离线核验——全树 `ast.parse`、**合并引入的重复定义**（与两个父提交对比，继承性重复只报 INFO）、**内容并集**（逐行检查任一侧新增的行是否仍在合并结果中，仅允许脚本内显式登记的改名映射）。

## ③ 计算结果
| 检查项 | 结果 |
|---|---|
| 注册项行数（main / 分支 / 合并） | 84 / 85 / 147（= 84 + 85 − 22 共有） |
| 分组行数 | 每个分组计数与父提交**逐一相等**（见下图 a） |
| 重复 key | 0 |
| 静态解析 | 231 个 py 文件全部通过 |
| 合并引入的重复定义 | 0（继承性重复 6 处，均为父提交既有） |
| 内容并集 | 除登记的 7 处改名/订正外，**零缺失行、零丢失文件** |

## ④ 结果图
![registry union](fig_registry_union.png)
(a) 合并后的注册表分组计数＝两父提交的并集：`hulme3d` 28 / `hulme3d_chief` 9 / `wang_linton3d` 26 来自分支，`hulme_hemisphere` 31 / `submerged_sphere` 25 / `hulme_chief` 6 来自 main，共有组（cylinder 2、ellipsoid 3、interior_harmonics 6、ursell 9、ursell_reference 2）计数不变。
(b) 只有 4 个单元两侧说法不一致，全部对齐到 main 的 m3150 复算值（B11 观测阶 1.62→1.80、1.88→1.94；转动耦合比 2.2e-4→2.151e-4、2.6e-5→2.648e-5）。

## ⑤ 注意点
- 合并后注册表**同时存在两套指向同一批文档的表**（分支的生成族 + main 的内联族）。语义并集要求两者都不丢，代价是同一 study 会算两遍（cache key 含 group）——建议后续单独立项收敛成一套，而不是在治脏单里删内容。
- 分支的 B11 观测阶原先用 `log 2`（理想棱长减半）推导，与 main 的 `log(h0/h1)`（实测比 1.87）口径不同，两者相差 10.1% / 3.0%；本单按 main 口径统一，并在文档第 4.10 节已有订正说明。
- 全量测试在交付时**仍在跑最后的多分钟三维 refinement 项**（见⑥），因此"全套 980 项通过"这一条我**没有**拿到最终回执，不能声称。

## ⑥ 误差判定（关键量 vs 基准）
| 关键量 | 合并前偏差 | 合并后 | 是否 <3% |
|---|---|---|---|
| B11 观测阶（16→64） | 分支声明 1.62 vs 实测口径 1.8021 | 声明 1.80（相对偏差 0.12%） | **是** |
| B11 观测阶（64→256） | 分支声明 1.88 vs 1.9377 | 声明 1.94（0.12%） | **是** |
| 转动耦合比（80 面元） | 2.2e-4 vs 2.151e-4（2.3%） | 2.151e-4（0.0%） | **是** |
| 转动耦合比（320 面元） | 2.6e-5 vs 2.648e-5（1.8%） | 2.648e-5（0.0%） | **是** |
| 注册表内容完备性 | 机械 overlay 有静默丢失前科 | 0 缺失行 / 0 丢失文件 / 0 重复 key | **是**（严格 0） |
| 全量测试 | main 基线 980 项 | 合并树执行至 95%（≈931 项）时 0 failed、3 skipped，尾部三维 refinement 项仍在运行 | **未取得最终回执（honest）** |
