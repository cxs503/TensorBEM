# m20485 · CASE-BEM-RIGID-BODY-MASS 复核 + 算例池证据指针修复

锚点：`CASE-BEM-RIGID-BODY-MASS`（算例池·刚体质量矩阵，ledger 状态 `open`）
仓：TensorBEM　分支：`agent/m20485`

## 1. 锚点核对结论（先做，避免重复劳动）

开工第一件事是核对总账与仓内实际状态。核对结果是：**该锚的求解与对标内容已经在仓内落地**
（`src/tensorbem/benchmarks/case_rigid_body_mass.py`、`rigid_body_mass.py`、
`casepool_bem_rigid_body_mass.py`、registry 条目、9 个测试、`docs/case_bem_rigid_body_mass/`
证据包），ledger 之所以仍为 `open`，缺的不是物理结果而是**可机器复核的闭环**。因此本单不做重复求解，
而是做两件增量：

1. 对锚点做一次**独立复算复核**（干净目录重跑，逐位比对提交的 `results.json` / `verdict.json`）；
2. 修复锚点所在的**算例池证据链契约**——`tests/test_case_pool_registry.py` 在基线
   commit `cc685699` 上是红的（4 failed / 8 passed），注册表承诺的路径在树里不存在、
   两个锚的 verdict 抽取返回 `None`。这不是锚点本身的问题，但它让锚点（以及其余 6 个锚）
   的"最近一次判定"无法被机器回答。

## 2. 锚点复算复核（独立、可复现）

```
PYTHONPATH=src python3 -m tensorbem.benchmarks.casepool_bem_rigid_body_mass --outdir runs/m20485/rigid_body_mass
```

与提交证据 `docs/case_bem_rigid_body_mass/{results,verdict}.json` 逐项比对（仅忽略
`environment` 段：python/platform/wall_time），**完全一致**。物理量：

| 项 | 数值 | 判据 |
|---|---|---|
| 闭式参考 | Fossen (2011) §3.3.1 eq.(3.44)-(3.53) 分块形式 `[[mI, -mS(r_g)],[mS(r_g), I_g+mS(r_g)S(r_g)^T]]` | — |
| 基础算例（盒体，解析 vs 闭式） | 最大入口相对误差 `0.0`（24/96/384 面元三级） | <3% ✅ |
| 基础算例（网格求积 vs 闭式） | `4.3e-16 / 4.3e-16 / 1.3e-15` | 机器精度 ✅ |
| 旋转体（全惯量张量，接受层级 sub=3） | `1.415e-2`（体积误差 8.6e-3） | <3% ✅ |
| 旋转体 sub=1 / sub=2（未接受层级） | `1.993e-1 / 5.514e-2` | 收敛阶梯，未被接受 |
| 动量定义路线 P（4 个构型） | `0 / 1.1e-16 / 0 / 1.1e-16` | <1e-9 ✅ |
| 负对照：翻转耦合块符号 | `2.0`（100% 偏离） | 必须被拒 ✅ |
| 负对照：反对称污染 | 能量恒等式 `<1e-9` 通过但动量 `>0.1` 失败 | 分辨力证明 ✅ |

结论：**验收（全部 36 个入口 <3%）通过**，最坏接受值 1.42% < 3%，且负对照证明判据有分辨力。
证据：`docs/case_bem_rigid_body_mass/{results,verdict,adversarial_audit}.json`、`runs/m20485/anchor_rerun.log`。

## 3. 缺陷修复（本单实际增量）

### 3.1 `_verdict_of` 对两种 canonical schema 返回 `None`

`src/tensorbem/benchmarks/case_pool_registry.py` 的 verdict 抽取循环：

```python
for key in ("overall", "verdict", "status"):
    if not isinstance(verdict.get(key), dict):
        return verdict.get(key)      # 第一个缺失的 key 直接 return None
```

只要 `verdict` 块里没有 `overall` 这个键（例如
`{"passed": true, "tolerance": 0.03, "checks": [...]}` 或
`{"criteria": {...}, "verdict": "PASS"}`），函数在第一个 key 上就返回 `None`，
后面的 `criteria` 折叠根本走不到。后果是注册表的**唯一用途**（回答"最近一次判定是什么"）
对 `CASE-BEM-ANNULUS-LAPLACE` 和 `CASE-BEM-HULME-HEMISPHERE` 静默失效。

修复后的抽取顺序（`_verdict_of` + 新增 `_verdict_from_criteria` / `_as_bool`）：

1. 顶层标量 `overall/status/level`；
2. 嵌套 `verdict` 块：先取**有明文措辞**的标量（`overall/verdict/status/level`），
   再取布尔旗标（`passed/pass/ok/all_pass/overall_passed`），
   否则把 `criteria{name:{pass|passed}}` 折叠成 `{"passed":…, "failed_checks":[…], "check_count":…}`；
3. 顶层 `acceptance` / `criteria` 同样折叠；
4. `tiers` 分档；
5. 兜底字符串 `"see artifact"`——**任何情况都不再返回 `None`**。

关键安全性质：折叠只在条目带**显式布尔**时计数，无布尔的 criteria 块不会凭空造出 verdict
（`test_criteria_block_without_booleans_is_not_fabricated`）；带 `False` 的块折叠为
`passed=False` 并列出失败项（`test_nested_verdict_block_without_scalar_still_reports_failure`），
即**不会把 FAIL 洗成 PASS**。

修复后 7 个锚全部可回答：

| case | verdict |
|---|---|
| ANNULUS-LAPLACE | `{"passed": true, "failed_checks": []}` |
| ELLIPSE-2D-ADDED-MASS | `PASS` |
| HULME-HEMISPHERE | `PASS` |
| LAPLACE-CYLINDER | `PASS` |
| NEUMANN-KELVIN | `see artifact`（该证据包本身无 verdict 字段，如实兜底） |
| RIGID-BODY-MASS | `PASS` |
| SUBMERGED-SPHERE-HAVELOCK | `PASS`（`economy` 档 FAIL 写在 `verdict.honest_note` 里，未掩盖） |

### 3.2 注册表承诺了 10 个不存在的文件（悬空证据指针）

`registry.missing_paths()` 在基线上返回 4 个 case、10 条缺失：

* `CASE-BEM-ANNULUS-LAPLACE`：eccentric 环的 3 张图（`convergence/clearance_sweep/field.png`）；
* `CASE-BEM-LAPLACE-CYLINDER`：3 张图（`field/convergence/near_wall.png`）；
* `CASE-BEM-ELLIPSE-2D-ADDED-MASS`：3 张图（`convergence/aspect_ratio/boundary_flux_error.png`）；
* `CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`：`run.log`。

这些是**派生文件**：数字（`results.json` / `numbers.json`）都在树里，丢的是渲染/落盘那一步。
处置不是删指针（那会抹掉证据主张），而是新增 `scripts/repair_case_pool_evidence.py`，
用**各自模块自己的绘图函数**以已提交的 `results.json` 为唯一输入重新生成，并写审计
`docs/case_pool_records/evidence_repair.json`（每个文件的路径/sha256/字节数/生产者）。
`run.log` 从 `docs/case_bem_submerged_sphere_havelock/numbers.json["log"]`（同一次运行记录在证据里的原文）
物化，不是新跑一次的数据。无生产者的路径会以非零退出码显式报 `UNREPAIRABLE`，不允许悄悄抹平。

结果：`missing_paths() == {}`，10 个文件全部恢复（PNG 像素方差 0.12–0.27，非空白图）。

## 4. 验证

```
PYTHONPATH=src python3 -m pytest tests/test_case_pool_registry.py \
    tests/test_case_pool_verdict_extraction.py tests/test_rigid_body_mass_momentum.py \
    tests/test_case_ellipse_2d_added_mass.py -q     -> 51 passed
PYTHONPATH=src python3 -m pytest tests/test_rigid_body_mass*.py tests/test_case_rigid_body_mass.py \
    tests/test_doc_numbers_rigid_body.py -q         -> 73 passed
PYTHONPATH=src python3 scripts/case_pool_run.py --check   -> exit 0, 7/7 status=OK, missing=[]
PYTHONPATH=src python3 scripts/repair_case_pool_evidence.py --check -> 修复前 exit 1 / 修复后 exit 0
```

基线红：`tests/test_case_pool_registry.py` 4 failed / 8 passed → 现在 0 failed。

## 5. 注意点与局限

* `CASE-BEM-NEUMANN-KELVIN` 的证据包没有 verdict 字段，注册表只能答 `see artifact`；
  这是**如实兜底**，不是修复遗漏——要让机器可答，需要该 case 的驱动写出 verdict 块（下一件活的线索）。
* `_verdict_of` 的 criteria 折叠只看布尔，不看阈值/量级；它回答"pass 与否"，不替代证据审阅。
* 重新生成的 PNG 与丢失的原图**不保证逐位相同**（matplotlib 版本/后端差异），
  但它们由同一份提交数据驱动，审计里记录了 sha256 以便后续比对。
* 锚点的接受层级是 sub=3（误差 1.42%）；sub=1/2 明显超差（19.9% / 5.5%），
  说明该算例的容差判据只在收敛后成立——引用结论时不可脱离网格层级。

## 6. 给下一次自找活的线索

1. `CASE-BEM-NEUMANN-KELVIN` 补 verdict 块（让注册表对 7/7 都能回答），并把它纳入
   `repair_case_pool_evidence.py` 的可修复集合。
2. 锚点侧：`M_RB` 的**量纲/尺度不变性**尚无显式测试（M 随 m 线性、角块随 m·L²），
   可作为刚体质量矩阵的下一个验证欠账。
3. `registry` 目前没有"artefact 与驱动版本一致性"检查（驱动改了而证据没重生成），
   可加 `results.json["environment"]` 之外的内容哈希门。
## 7. 报告

平台进展报告 id `18640`（`POST /api/v1/tensor/reports`，标题「m20485 · 刚体质量矩阵锚点复核与算例池证据链修复（TensorBEM）」，含两张内嵌图）。
