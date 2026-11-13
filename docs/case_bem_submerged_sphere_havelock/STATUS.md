# STATUS — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK (m17908, 完成)

交付件（本目录）：
- `numbers.json`：**完整运行工件**（`--stage all`，含 case_mesh 1280 面元 + economy_mesh 320 面元全部分级）。
  - case_mesh（1280 面元）：20/20 项 PASS，最大相对误差 1.431%（ka=0.5 b33）→ **PASS**（<5% 容差）。
  - economy_mesh（320 面元）：21/28 项 PASS，7 项 FAIL，最大相对误差 5.591%（ka=0.5 b33）→ **FAIL**（honest FAIL）。
  - overall verdict: **PASS**（以案定网格 1280 面元为准）。
- `verdict.json` / `tiers.json`：机器可读判据。
- `table_band_case.csv` / `table_band_economy.csv` / `table_ladder.csv` / `table_low_frequency.csv` / `table_submergence.csv`：对照表。
- `CASE.md`：由 `make_doc.py` 从 `numbers.json` **程序化生成（零手抄）**的完整案例文档。
- `fig1_band.png` / `fig2_error.png` / `fig3_ladder.png` / `fig4_submergence.png`：4 张结果图。
- `run.log`：运行日志。
- `smoke_economy_320/`：economy mesh 320 面元的独立工件。
- `capytaine_crosscheck.json`：独立第三方软件 capytaine 3.1.dev0 交叉核对，最差 |capytaine−ref| = 3.96%（ka=2.0，2 细分）→ **PASS**。

实现与入口（已入仓）：
- `benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py` — 预声明判据 + 分级运行 + 工件输出（entry）。
- `benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py` — 由 numbers.json 零手抄渲染 CASE.md + 4 图。
- `src/tensorbem/benchmarks/case_pool_registry.py` + `scripts/case_pool_run.py` — 算例池锚点注册表与分发 CLI（平台侧调用面）。
- `scripts/case_havelock_capytaine_crosscheck.py` — 独立软件交叉核对。
- `tests/test_casepool_submerged_sphere_havelock.py`（10/10 PASS）、`tests/test_case_havelock_submerged_sphere.py`（7/7 PASS）、`tests/test_case_pool_registry.py`（HAVELOCK 项 PASS）— CI 守卫。
- 复用库：`src/tensorbem/benchmarks/case_havelock_submerged_sphere.py`、`submerged_sphere_havelock.py`、`submerged_sphere_havelock_integrity.py`、`analytic.submerged_sphere_wave_multipole`、`free_surface_solver3d`。

参考解自证（全部 PASS）：
- multipole truncation 20→40 terms: worst |dcoef| < 1e-12
- Haskind reciprocity: residual 8.80e-16
- Lamb (1932) A'→0.5 at f/a=20: 0.500024
- Lamb (1932) C_M→1.5 at f/a=20: 1.500024
- B33'→0 as ka→0: B33'(ka=1e-3) = 4.84e-09

复现命令：
```
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py --stage all --outdir docs/case_bem_submerged_sphere_havelock
PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py --numbers docs/case_bem_submerged_sphere_havelock/numbers.json --outdir docs/case_bem_submerged_sphere_havelock
```

判级自评（诚实，非自声明生效）：L2 具备（非占位实现 + 入口 + 真实运行工件）；L3 具备（注册表/CLI/测试的真实调用面）；L4 具备（ref vs got 逐项 + 误差 + repro_cmd + CASE.md + 4 图全部落地）。
