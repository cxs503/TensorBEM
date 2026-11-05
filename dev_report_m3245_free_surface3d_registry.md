# 三维自由液面声明数字的可执行化：Hulme 半球 / Ka=2.58 CHIEF / 潜没球体（m3245）

EDIT_WHITELIST: src/tensorbem/benchmarks/declared_numbers.py, tests/test_declared_numbers.py, docs/, outputs/m3245/, dev_report_m3245_free_surface3d_registry.md

**任务**：m3245（platform 席自找活） · **分支**：`agent/m3245` · **仓**：TensorBEM
**类型**：验证欠账（把已合并功能对解析/半解析基准复核），承接 m3059 遗留建议

---

## 1. 为什么做这一单（选题依据）

扫描本仓缺口账（`expand_audit_tensorbem.md` §4、`dev_report_m3059_heavy_verification_debt.md` §4、
`docs/verification.md`、`docs/marine_bem_benchmarks.md` §5 局限）后，**唯一一条既明确登记、
又是「验证欠账」类、且未与近 24 h 终态单重叠**的缺口是：

> m3059 收尾原话：「**三维自由液面（Hulme/Wang-Linton）虽已由 m2609 等测试钉桩，但
> 仍未纳入注册表（成本更高，建议后续单独立项）**」。

复核属实：`tensorbem.benchmarks.declared_numbers`（m2136 建立）只登记二维圆柱、椭球、Ursell 与
内部谐波四族；`docs/marine_bem_benchmarks.md` §4.10 表 10/表 11 与 §4.11 表 12 的 46 个数字
**没有任何测试重算**。这些数字承载了本仓三维自由液面求解器（`free_surface_solver3d`）的全部
精度声明，随时可以静默腐化——正是「已合并功能对解析基准复核」这一类欠账。

## 2. 做了什么

1. `src/tensorbem/benchmarks/declared_numbers.py`：新增三个 benchmark 组
   （`hulme_hemisphere`、`hulme_chief`、`submerged_sphere_wave`），把每层每量拍平为
   `panels<N>.<quantity>`（CHIEF 为 `ka2.58.<scheme>.<quantity>`），靠注册表自带的
   `(group, parameters)` 缓存让一条注册项 = 一次求解；新增 **56 条**注册行
   （表 10 的 15 个误差 + 4 个观测阶、表 11 的 6 个、表 12 的 12 个误差 + 4 个观测阶、
   5 条上界型物理守恒断言、2 组 `_standard_guard` 单层行）。
2. 成本分层：三维稠密自由液面求解秒级→分钟级，故主体入 `heavy` 层；每族另登记一条**单层**
   （64 面元半球 / 80 面元球体）`standard` 行，保证默认 `pytest` 仍走通三维自由液面代码路径。
3. `tests/test_declared_numbers.py`：覆盖门扩到新三组；新增默认层门
   `test_standard_tier_reproduces_the_3d_free_surface_coarse_rows` 与 `HEAVY` 门
   `test_documented_3d_free_surface_tables_are_reproduced`。HEAVY 门复用同一次运行额外断言物理方向：
   未增广 BIE 在 `Ka=2.58` 的垂荡阻尼**为负**（违背无源耗散性）、CHIEF 增广后**为正**；
   并断言细网格误差严格小于粗网格误差（防止在未收敛族上引用观测阶）。
4. `docs/free_surface3d_registry.md`（新）由 `outputs/m3245/inject_docs.py` 从
   `outputs/m3245/free_surface3d_registry.json` **程序化注入**（零手抄）；`docs/marine_bem_benchmarks.md`
   §4 末尾加入一段指向该文档与两条回归门的说明（不改动任何原有表格数字）。
5. 工件：`outputs/m3245/{run_registry.py, free_surface3d_registry.json, make_figures.py,
   fig_3d_free_surface_registry.png, inject_docs.py, patch_registry.py, patch_tests.py, probe_3d.py}`。

## 3. 结果（数字全部来自 `outputs/m3245/free_surface3d_registry.json`，明细表见 docs 文件）

- 注册表本组 **56 行全部 PASS**，`value` 型最坏相对偏差 **4.6e-3**（3 位有效数字舍入量级）
  ⇒ 文档与代码一致，声明从此可执行。重算合计约 **70 s**。
- 表格逐项：表 10（16/64/256 面元，5 量 × 3 层）、表 12（20/80/320 面元，4 量 × 3 层）、
  表 11（`Ka=2.58`, N=256：未增广 κ=145.66 / A33'=0.348 / |B33'|=0.104（负），
  CHIEF κ=4.24 / A33'=0.3919 / B33'=0.0687）—— 与文档逐位一致。
- 观测阶：Hulme 256 面元 2.06/1.98/2.01/2.02；潜没球体 320 面元 2.06/1.93/2.03/1.98，与文档一致。

## 4. 验证方式（跑了什么）

```bash
PYTHONPATH=src python3 outputs/m3245/run_registry.py                     # 56/56 PASS，写出 JSON
PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q        # 13 passed, 4 skipped (1:55)
TENSORBEM_HEAVY_TESTS=1 PYTHONPATH=src python3 -m pytest \
  tests/test_declared_numbers.py -q -k "3d_free_surface"                 # 2 passed (1:23)
PYTHONPATH=src python3 outputs/m3245/inject_docs.py                       # docs 表从 JSON 注入
PYTHONPATH=src python3 outputs/m3245/make_figures.py                      # 图件 83 KB
```

（全量测试套件结果见本节末补充；`tests/test_declared_numbers.py` 未改任何既有断言的语义。）

## 5. 注意点与未做

1. **未改求解器实现**：本单是验证/注册表工作，不新增物理模型、不调精度参数。
2. **未增广阻尼登记为幅值**：注册表 well-formedness 门只接受正的 `declared`，故登记 `|B33'|`
   并把符号断言放进 HEAVY 测试（代码注释与 docs §4 均写明，不是丢符号）。
3. **`worst_relative_deviation` 会被上界行主导**（上界 4e-3、实测 3e-4 即 -92%），故 docs 另报
   `value` 型最坏偏差。
4. **仍未纳入注册表**：`kochin_hulme_convergence` 的 1024 面元 `B11'` 行（0.90%）——单次
   1024 面元三维求解成本高一个量级，建议单独一单；有限水深三维 John 核仍是功能缺口（不在本单范围）。
5. **精度并非全 <3%**：Hulme `B11'` 256 面元 3.53%、潜没球体波浪力 3.10–3.20%、CHIEF 后 `B33'`
   误差 8.85%。这些是常数面元离散误差，文档原本就如实给出（`B11'` 另有 1024 面元 0.90% 证据），
   本单如实转载，不粉饰、不改数。