# PR #162 (agent/m3566) Re-Merge Report — Mission m4019

## 1. 案例介绍（物理背景 / 几何 / 参数表）

PR #162（分支 `agent/m3566`）为 TensorBEM 注册表新增 **44 个刚体 Neumann 附加质量张量**（`docs/marine_bem_benchmarks.md` §4.7，表 5/表 6）的声明数字（`rigid_body` 组），使这些此前仅存在于文档中的数字变为可执行、可复算的判据。与此同时，main 侧新增了 **3-D 自由液面** 的 Hulme 漂浮半球（§4.10）与 Wang/Linton 潜没球体（§4.11）基准及其守卫测试。

两方同时编辑了三个文件，产生冲突：
- `src/tensorbem/benchmarks/declared_numbers.py`
- `tests/test_declared_numbers.py`
- `docs/marine_bem_benchmarks.md`（自动合并，无冲突）

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. 建工作树，拉取 `main`（`648af21`）与 `agent/m3566`（`887c000`）。
2. 三方合并 `agent/m3566 → main`，逐文件解冲。
3. 冲突解法采用 **语义并集**（参考 m3985-merge 已验证的解析）：
   - **import 块**：同时保留 `rigid_body_added_mass as rigid_body_benchmark`（m3566 侧）与 `floating_hemisphere as hemisphere_benchmark`、`submerged_sphere_wave as sphere_benchmark`（main 侧）。
   - **DECLARED_NUMBERS**：保留 main 侧全部 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere` 行，随后追加 m3566 侧的 `*_rigid_body_registry()`。
   - **测试**：covered-group 集合同时含 `rigid_body` 与 `submerged_sphere`；main 侧 3-D 自由液面守卫测试与 m3566 侧刚体复算/散文声明测试全部保留。
4. 本地跑核心测试（结构测试通过）。

## 3. 计算结果（数字表）

| 检查项 | 结果 |
|---|---|
| `test_registry_is_well_formed` | PASS |
| `test_registry_covers_every_documented_benchmark_family`（含 rigid_body + submerged_sphere） | PASS |
| `test_3d_free_surface_rows_are_guarded_by_the_default_tiers`（main 侧 3-D 守卫） | PASS |
| 冲突标记残留 | 0（无） |
| 合并后 SHA | `987be21903ad27189a71738b151700ef2bcc1987` |
| 旧 m3566 SHA | `887c000f773020a55bc71220d6c6134da3de4b84` |

## 4. 结果图

纯文档/合并任务，无场量云图。理由：本任务为 git 三方合并与冲突解析，不涉及数值场量计算；合并正确性由结构测试与逐行审计证明。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

- **机械 overlay 有静默丢内容前科**：本仓历史上有 overlay 合并静默丢内容的记录，故本任务采用 `verify_merge_no_loss.py` 逐行审计，确认双方新增行均保留。
- **main 已大幅前进**：main 自 m3566 的合并基（`317bb03`）已前进 91 个文件（含 m3100/m2904 等其它任务），本合并仅针对 PR #162 涉及的三个文件，其余 main 内容由 `git merge` 自动带入。
- **push 被安全策略拦截**：`git push` 被平台安全策略拒绝，分支由平台自动推送；merge-governor 下一周期自动复验合并。

## 6. 误差判定

本任务为代码合并/冲突解析，无数值基准可比对。合并正确性由以下证明：
- 双方新增行均保留（逐行审计，无冲突标记残留）；
- 结构测试全部通过（`test_registry_is_well_formed`、`test_registry_covers_every_documented_benchmark_family`、`test_3d_free_surface_rows_are_guarded_by_the_default_tiers`）；
- 无任何一侧内容被静默丢弃。
