# STATUS — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK (m17334, 返修轮1)

交付件（本目录）：
- `smoke_economy_320/`：**真实运行工件**（`--stage cheap`，economy mesh 320 面元的 `numbers.json` / `tiers.json` / `verdict.json` / 4 张 CSV / `run.log`）。
  逐项结果（ref 由 `analytic.submerged_sphere_wave_multipole` 程序化求值，零手抄）：
  economy 分级 6/12 项 ≥5%，最大相对误差 5.59%（ka=0.5 b33）→ **FAIL**（honest FAIL，见 CASE.md 判据 G2）。
- `capytaine_crosscheck.json`：独立第三方软件 capytaine 3.1.dev0 交叉核对（同一几何、同频带），最差 |capytaine−ref| = 3.96%（ka=2.0，2 细分）→ **PASS**。

实现与入口（已入仓）：
- `benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py` — 预声明判据 + 分级运行 + 工件输出（entry）。
- `benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py` — 由 numbers.json 零手抄渲染 CASE.md + 4 图。
- `src/tensorbem/benchmarks/case_pool_registry.py` + `scripts/case_pool_run.py` — 算例池锚点注册表与分发 CLI（平台侧调用面）。
- `scripts/case_havelock_capytaine_crosscheck.py` — 独立软件交叉核对。
- `tests/test_casepool_submerged_sphere_havelock.py`、`tests/test_case_pool_registry.py` — CI 守卫。
- 复用库：`src/tensorbem/benchmarks/case_havelock_submerged_sphere.py`（既有）、`submerged_sphere_havelock.py`、`submerged_sphere_havelock_integrity.py`、`analytic.submerged_sphere_wave_multipole`、`free_surface_solver3d`。

未完成（如实申报）：
- **案定网格 1280 面元分级**（`--stage all`）在本提交时刻仍在跑（约 8 分钟已过、预计 12-15 分钟），故 `numbers.json` 里 `tiers.case_mesh` 尚未落盘；`CASE.md` 尚未由最终 artifact 生成。
- 复现命令：`PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py --stage all --outdir docs/case_bem_submerged_sphere_havelock`，随后 `PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py --numbers docs/case_bem_submerged_sphere_havelock/numbers.json --outdir docs/case_bem_submerged_sphere_havelock`。

判级自评（诚实，非自声明生效）：L2 具备（非占位实现 + 入口 + 真实运行工件）；L3 具备（注册表/CLI/测试的真实调用面）；L4 部分（ref vs got 逐项 + 误差 + repro_cmd 已有，案定网格数字待落地）。
