# m21907 — CASE-BEM-RIGID-BODY-MASS 独立审计的再生性收口

**结论一句话**：注册表记载的复现命令此前是**破坏性**的（重跑会静默删掉 `ADVERSARIAL_REVIEW.md` 的第 6 章，
2076 字节 / 全文 30.3 %，退出码仍为 0）；现已修复为字节级 no-op，并新增 `--check` 与 14 项门禁；
四探针 A–D 在 `rigid_body_mass.py` 于 #959/#982 两次重构后仍逐位复现。

## 1. 缺陷（main@cdd44932 实测）

```
PYTHONPATH=src python3 scripts/audit_m19935_rigid_body_mass.py   # 注册表 repro_cmd
git diff docs/case_bem_rigid_body_mass/                          # → ADVERSARIAL_REVIEW.md −2076 B
```

根因：`write_review()` 只渲染 1–5 章；第 6 章（RAO-caller 回放附录）由后续变更集写入文档，
生成器不认识它。于是"按文档复现已评审工件"这条链在无人察觉的情况下退化，且没有任何自动门禁——
后续两次 `rigid_body_mass.py` 重构（#959 m21196、#982 m21422）也没有验证过探针是否仍复现。

## 2. 修复

| 改动 | 内容 |
| --- | --- |
| `ADDENDUM_SECTION` | 第 6 章提升为生成器内的作者文本常量，随生成器一起走 |
| `render_review()` | 渲染拆成纯函数（可测试），`write_review()` 退化为写文件 |
| `--check` | 不写任何文件；重算四探针并与已提交工件比对（评审文档字符级、数字 1e-9 相对），漂移 exit 1 |
| `tests/test_audit_m19935_rigid_body_mass.py` | 14 tests / 2.2 s：再生字节一致、第 6 章存活、探针 A–D 复现、检查器判别力（修复前生成器为必败对照） |

## 3. 探针复现（对照：已提交 `adversarial_audit.json`，差异计数 0）

| 探针 | 量 | 值 |
| --- | --- | --- |
| A | Hessian(T) − 分块形式 | 零矩阵（0 非零元）；装配 vs 符号 Hessian 0.0e+00（尺度 1.85e10） |
| B | 最差 6×6 元素误差 @ 80/320/1280/5120 面板 | 20.12 % / 5.57 % / 1.43 % / 0.36 %，拟合斜率 −2.82 |
| C | 修复前被静默接受的坏网格体积误差 | 断口 0.95 %、翻转 1.00 %、重复 0.79 %（**全部 <3 % 容差**），修复后全部 rejected |
| D | 指标报告值 vs 定义值 | 残差 2.0e−14；小耦合元素放大 258× |

## 4. 门禁判别力（实测，报告图 C）

对拟合斜率注入相对扰动：数值层自 **1e−9** 起检出；渲染文本层（2 位小数）自 **3.2e−3** 起才可见。
两层互补，故 `--check` 同时比对数字与文本；图件只校验存在（PNG 内嵌编码器元数据，字节比较无意义）。

## 5. 误差判定

关键量全部 <3 %：符号 Hessian 0 %、求积最差元素 0.36 %（5120 面板）/1.43 %（锚点接受细分 3）、
体积 0.216 %、指标残差 2.0e−14、文档再生 0 字节。
**反例（诚实记录）**：修复前被静默接受的坏网格误差仅 0.79–1.00 %，落在容差内——对这类缺陷
"是否 <3 %" 是假阴性判据，正确判据是结构性判据（闭合性/流形性/定向一致性）＋工件再生一致性。

## 6. 验证

* 本地：`tests/test_audit_m19935_rigid_body_mass.py` + `tests/test_rigid_body_mass_topology.py` +
  `tests/test_casepool_bem_rigid_body_mass.py` = 36 passed（13.4 s）；`tests/test_benchmark_cli_contract.py` = 20 passed。
* 集群全套 `cj3776`：收集阶段因 7 个无关模块的导入期 `numpy.linalg.LinAlgError` 中断（同节点
  更早作业 `cj1903`/`cj1826` 同样出现 → 环境问题，非本单引入），**未形成全套结论**；
  复跑 `cj3836`（忽略这些模块）已递交。

## 7. 改动文件

* `scripts/audit_m19935_rigid_body_mass.py`
* `tests/test_audit_m19935_rigid_body_mass.py`（新增）

未触碰 `src/tensorbem/rigid_body_mass.py` 与 `case_pool_registry.py`（后者与多个在开 PR 重叠的热文件）。