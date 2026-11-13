# 交差审计报告：m17179 CASE-BEM-URSELL-CYL-RAD

**审计人**: board-audit (m17200)  
**日期**: 2026-11-13  
**对象**: m17179 — Ursell 潜没圆柱辐射（深水）· TensorBEM  
**锚点**: CASE-BEM-URSELL-CYL-RAD

---

## 1. 取证清单

### 1.1 收三件套核验

| 件 | 状态 | 路径/证据 |
|---|---|---|
| run_artifacts | ✅ 存在 | `runs/ursell_cyl_rad_ka06.json` (6.7KB, 3 panel counts), `runs/ursell_cyl_rad_ka_sweep.json` (11.2KB, 5 ka values), `runs/ursell_figures_base64.json` (352KB, 3 figures) |
| benchmark ref vs got | ✅ 存在 | JSON 内含 `reference_added_mass`/`reference_damping` 与 `panel_added_mass`/`panel_damping` 逐项对比 |
| repro_cmd | ✅ 可执行 | `PYTHONPATH=src python -m tensorbem.benchmarks.ursell_cylinder --panels 128 --ka 0.6` — 实测通过 |

### 1.2 基准裁判：tensor_benchmark_cases

**⚠️ 表不存在**：遍历 /root/tp-agent 下所有 .db/.sqlite 文件，未找到 `tensor_benchmark_cases` 表。无法按任务令 `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-URSELL-CYL-RAD'` 执行逐项核验。

**替代裁判**：以 JSON 工件内 `reference_*` 字段（Ursell 多极子解析解）为基准，以 5% 为通用容差（平台精度标准：>15% 不可接受，本算例属经典基准取 5% 为合理门槛）。

### 1.3 数值核验（ka=0.6, 128 面元）

| 关键量 | 基准 (ref) | 实测 (got) | 相对误差 | <5%? |
|---|---|---|---|---|
| 附加质量 A [kg/m] | 2545.1672 | 2566.8343 | 0.8513% | ✅ |
| 阻尼 B [kg/(m·s)] | 2718.1540 | 2754.4701 | 1.3361% | ✅ |
| 近/远场阻尼比 | 1.0 | 1.005341 | 0.5341% | ✅ |
| 垂荡/横荡对称性 | 0 | 1.77e-16 | — | ✅ (机器精度) |

### 1.4 ka 扫描核验（128 面元）

| ka | A 误差 | B 误差 | <5%? |
|---|---|---|---|
| 0.2 | 1.1362% | 1.8142% | ✅ |
| 0.4 | 0.9329% | 1.5466% | ✅ |
| 0.6 | 0.8513% | 1.3361% | ✅ |
| 0.8 | 0.8580% | 1.2239% | ✅ |
| 1.0 | 0.8855% | 1.1675% | ✅ |

**结论**：所有关键量误差均 < 2%，远低于 5% 容差。

### 1.5 测试核验

| 测试文件 | 通过/总数 | 备注 |
|---|---|---|
| test_ursell_benchmark.py | 9/10 | 1 fail: `test_cli_writes_strict_json` — 模块未 pip install，PYTHONPATH 可用 |
| test_ursell_convergence.py | 6/6 | 收敛阶、单调性、细网格精度 |
| test_ursell_reference_pinning.py | 6/6 | 钉桩值、解析解一致性、求积收敛 |
| test_ursell_quadrature_convergence.py | 5/5 | 求积节点无关性、一阶收敛、文档一致性 |
| **合计** | **26/27** | 1 个环境问题，非代码缺陷 |

### 1.6 报告六段核验

| 段 | 状态 | 备注 |
|---|---|---|
| ① 案例介绍 | ✅ | 物理背景、几何参数表完整 |
| ② 方法 | ✅ | 间接源公式、离散设置、工具链 |
| ③ 计算结果 | ✅ | 数字表程序化注入（JSON→MD），零手抄 |
| ④ 结果图 | ✅ | 3 图：收敛性、误差 vs ka、系数 vs ka |
| ⑤ 注意点 | ✅ | 一阶收敛为常数面元固有性质、求积收敛已钉桩 |
| ⑥ 误差判定 | ✅ | 逐项列出 vs 基准误差，显式回答 <5% |

---

## 2. 判级

**L4 — 基准验证（完整算例 + 误差分析 + 可复现）**

理由：
- ✅ 完整算例：Ursell 潜没圆柱深水辐射，含解析参考解
- ✅ 误差分析：面元收敛研究（32/64/128）+ ka 参数扫描（0.2-1.0）
- ✅ 可复现：CLI 入口 + 钉桩测试 + JSON 工件
- ✅ 多路交叉验证：面板 BEM vs MFS vs 近/远场恒等式 vs 垂荡/横荡对称性
- ✅ 报告六段齐全
- 不达 L5：无科研新贡献（经典基准验证，非新方法/新发现）

**降级风险项**（不构成降级但需记录）：
1. `tensor_benchmark_cases` 表不存在 → 无法按任务令逐项核验容差
2. `test_cli_writes_strict_json` 失败 → 模块未 pip install（PYTHONPATH 可用，非代码缺陷）
3. 报告未成功推送至平台（500 错误）

---

## 3. 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorBEM', 'm17179 CASE-BEM-URSELL-CYL-RAD', 17200, 4,
'{"impl_files":"src/tensorbem/benchmarks/ursell_cylinder.py:1-232;src/tensorbem/analytic.py:ursell_cylinder_added_mass","entry":"src/tensorbem/benchmarks/ursell_cylinder.py:main()","run_artifacts":"runs/ursell_cyl_rad_ka06.json;runs/ursell_cyl_rad_ka_sweep.json;runs/ursell_figures_base64.json","tests":"tests/test_ursell_benchmark.py:9/10 pass;tests/test_ursell_convergence.py:6/6;tests/test_ursell_reference_pinning.py:6/6;tests/test_ursell_quadrature_convergence.py:5/5;total=26/27","integration_callers":"src/tensorbem/benchmarks/__init__.py","benchmark":"ursell_submerged_circular_cylinder_deep_water ka=0.6 128-panel A_err=0.85% B_err=1.34% both<5%","repro_cmd":"PYTHONPATH=src python -m tensorbem.benchmarks.ursell_cylinder --panels 128 --ka 0.6","external_anchor":"Ursell 1949 multipole closed-form (zero code sharing with BEM)","gaps":"tensor_benchmark_cases table not found;test_cli_writes_strict_json fails (env issue not code);report post 500 error"}',
'board-audit');
```

---

## 4. 发现与建议

### 4.1 阻塞项（无）

所有关键量误差 < 2%，远低于 5% 容差。核心功能完整。

### 4.2 非阻塞建议

1. **tensor_benchmark_cases 表缺失**：平台应建立此表以支持自动化容差核验。当前依赖人工比对 JSON 工件。
2. **CLI 测试环境**：`test_cli_writes_strict_json` 需模块安装后才能通过。建议在 CI 中添加 `pip install -e .` 前置步骤，或改用 `PYTHONPATH` 方式调用。
3. **报告推送**：平台 `/tensor/reports` 端点返回 500，需排查。
4. **benchmarks/verified/ursell**：未在 verified 目录下建立锚点子目录，建议补充以与其他基准（ellipsoid, wigley 等）对齐。
