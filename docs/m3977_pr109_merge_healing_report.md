# PR #109 合并治愈报告：agent/m2966 → main

**任务ID**: m3977  
**日期**: 2026-11-05  
**仓库**: TensorBEM  
**PR**: #109  

## 1. 问题描述

PR #109（分支 `agent/m2966` → `main`）状态为 `mergeable=false`，需要三方合并解冲突。

**分支状态分析**：
- `origin/main` = `ecba798`（Merge pull request #103）
- `origin/agent/m2966` = `ca4ba73`（m2966: matured mission deliverables）
- 合并基线 = `d22b6ad`
- `main` 领先 `agent/m2966` 76 个 commit
- `agent/m2966` 领先 `main` 1 个 commit

**冲突根源**：分支与 main 在两处文本上重叠：
1. `docs/marine_bem_benchmarks.md` §4.6 Ursell 表格与精度声明
2. `tests/test_documented_benchmarks.py` 对数板块闭式积分裁判测试

## 2. 合并状态验证

### 2.1 本地分支已解冲突

检查发现本地 `agent/m2966` 分支（`9a87109`）已经包含完整的合并解决：

```
9a87109 Add the merge-resolution evidence artifacts and report builder
8fa1b57 Merge main into agent/m2966: keep both sides of the Ursell provenance conflict
ca4ba73 m2966: matured mission deliverables
```

**关键合并 commit** `8fa1b57` 的解决策略：
- **语义并集**：保留双方内容，不静默丢失任何一侧的新增/删除
- **逐文件裁决**：每个冲突文件都有明确的合并决策
- **程序化复核**：所有争议数字都从代码重算验证

### 2.2 冲突文件清单与解法

| 文件 | 冲突类型 | 解决方案 |
|------|----------|----------|
| `docs/marine_bem_benchmarks.md` §4.6 MFS 行 | 表格行数与标签格式 | 保留 3 行（64/128/256 面元），标签统一为生成式「MFS，N 面元、S 源」；两条注释并存 |
| `docs/marine_bem_benchmarks.md` §4.6 尾注 | 两段不同主题的补注 | 两段都保留（数字溯源 vs 适用范围，内容互不重叠） |
| `tests/test_documented_benchmarks.py` | 裁判测试断言 | 两套断言都在：先做阶数钉扎，再用初等闭式做独立第二裁判，最后跑三配置循环 |
| `tests/test_ursell_doc_reproducibility.py` | MFS 标签正则（间接冲突） | 正则改为匹配生成式标签，保留"按列计数不按标签信任"原意；增加 256 面元行 |

### 2.3 合并文档

完整的合并解决记录位于：
- `docs/m2966/merge_resolution.md`：逐文件裁决说明
- `docs/m2966/merge_resolution_checks.py`：程序化复核脚本
- `docs/m2966/outputs/merge_checks/merge_checks.json`：复核结果数据
- `docs/m2966/figures/mfs_rows_vs_reference.png`：MFS 行 vs 参考解对比图
- `docs/m2966/figures/panel_referee_resolution.png`：裁判分辨率极限图

## 3. 测试验证

### 3.1 关键测试通过

**test_ursell_doc_reproducibility.py**（7 项测试）：
```
✓ test_documented_rows_still_reproduce[64]
✓ test_documented_rows_still_reproduce[128]
✓ test_reference_coefficients_are_stable
✓ test_mfs_rows_still_reproduce
✓ test_radiation_condition_identity_is_satisfied
✓ test_document_contains_the_generated_rows
✓ test_every_panel_column_has_a_generated_mfs_row
```

**test_documented_benchmarks.py**（6 项测试）：
```
✓ test_pin_file_exists_and_is_well_formed
✓ test_documentation_pins_match_current_code
✓ test_pinned_cases_reference_existing_docs
✓ test_analytic_rankine_panel_integral_is_exact
✓ test_uniform_gauss_referee_cannot_resolve_the_coplanar_log_singularity
✓ test_legacy_log_singular_quadrature_bias_is_quantified
```

### 3.2 合并解决复核

运行 `python docs/m2966/merge_resolution_checks.py`，输出：

**MFS 行复核**（Ursell 淹没圆柱，ka=0.6, a/f=0.5）：

| 面元数 | 源数 | 附加质量 A | A 相对误差 | 文档值差 | 阻尼 B | B 相对误差 | 文档值差 |
|-------:|-----:|-----------:|-----------:|---------:|-------:|-----------:|---------:|
| 64 | 24 | 2543.2406 | 7.57e-4 | 2.4e-7 | 2708.5687 | 3.53e-3 | 4.9e-7 |
| 128 | 24 | 2544.6864 | 1.89e-4 | 1.4e-6 | 2715.7556 | 8.82e-4 | 1.6e-6 |
| 256 | 24 | 2545.0471 | 4.72e-5 | 1.1e-6 | 2717.5542 | 2.21e-4 | 1.6e-6 |

**判定**：
- ✓ 三行文档值差均 ≤1.6e-6（文档保留 2 位小数所致）
- ✓ 全部行对 Ursell 闭式解的相对误差 ≤3% 验收线
- ✓ 256 面元行（main 侧没有、分支有）成功保留并验证

**对数板块积分裁判复核**：

| 量 | 值 | 判定 |
|---|---|---|
| 闭式板积分（ξ=0.5, h=0, L=1） | -1.6931471805599454 | ✓ 等于 2log(1/2)-1 |
| 均匀分片 Gauss 观测阶（100→800 片） | 1.000 | ✓ 符合理论预期 |
| 400 片残差 | >10⁻⁶ | ✓ 远未达旧容忍度 10⁻⁸ |
| 单片跨奇异点偏差 | 5.31e-2 | ✓ 历史缺陷量化 |
| 初等闭式 vs 板积分一致性 | 0.0 | ✓ 机器精度 |
| 三配置闭式 vs 收敛裁判 | 0.0 / 1.33e-15 / 1.11e-16 | ✓ 全部通过 |

**输出**：`all merge-resolution numbers reproduce`（8 项判据全部通过）

## 4. 合并策略评估

### 4.1 遵循的原则

✓ **语义并集**：保留双方内容，不静默丢失任何一侧的新增/删除  
✓ **逐文件裁决**：每个冲突文件都有明确的合并决策文档  
✓ **程序化复核**：所有争议数字都从代码重算验证，不手抄  
✓ **测试覆盖**：受影响的测试全部通过，无回归  

### 4.2 风险点

**唯一真实语义风险**：`tests/test_ursell_doc_reproducibility.py` 的 MFS 标签正则
- main 改了文档标签格式（从「配点网格 N 面元」到「N 面元、S 源」）
- 分支新增的守卫测试按旧的标签正则解析
- **解决**：正则改为匹配生成式标签，双方原意同时成立

**风险评估**：低。合并后的测试覆盖了所有争议点，程序化复核验证了所有数字。

## 5. 完成标准核查

| 标准 | 状态 | 证据 |
|------|------|------|
| 三方合并完成 | ✓ | commit 8fa1b57 |
| 逐文件解冲 | ✓ | docs/m2966/merge_resolution.md |
| 核心测试通过 | ✓ | 13 项关键测试全部通过 |
| 不低于 main 基线 | ✓ | 无回归，新增 256 面元行验证 |
| 合并解决文档 | ✓ | merge_resolution.md + checks.py + JSON +  figures |
| 推回分支 | ⏳ | 本地 agent/m2966 = 9a87109，待平台推送 |
| PR #109 合并进 main | ⏳ | 待 pr-gate 记账 |

## 6. 下一步行动

1. **平台推送**：将本地 `agent/m2966`（9a87109）推送到 `origin/agent/m2966`
   - 旧 SHA: `ca4ba73`
   - 新 SHA: `9a87109`
   
2. **PR 合并**：merge-governor 自动复验并合并 PR #109 进 main

3. **验证闭环**：确认 PR #109 在 GitHub 上状态变为 `merged`

## 7. 结论

**PR #109 的合并冲突已经正确解决**，采用语义并集策略，保留双方内容，所有争议数字都经过程序化复核。关键测试全部通过，无回归风险。

**建议**：平台立即推送本地 `agent/m2966` 分支到 origin，触发 merge-governor 完成 PR #109 合并。

---

**验证命令**：
```bash
# 检查本地分支状态
git log --oneline agent/m2966 -3

# 运行关键测试
python -m pytest tests/test_ursell_doc_reproducibility.py -v
python -m pytest tests/test_documented_benchmarks.py -v

# 运行合并解决复核
python docs/m2966/merge_resolution_checks.py
```

**产物清单**：
- `docs/m2966/merge_resolution.md`：合并解决记录
- `docs/m2966/merge_resolution_checks.py`：程序化复核脚本
- `docs/m2966/outputs/merge_checks/merge_checks.json`：复核结果
- `docs/m2966/figures/mfs_rows_vs_reference.png`：MFS 行对比图
- `docs/m2966/figures/panel_referee_resolution.png`：裁判分辨率图
