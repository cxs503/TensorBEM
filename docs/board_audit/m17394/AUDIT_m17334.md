# AUDIT m17334 — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK（交差核验单，本单 m17394）

对象：m17334（返修轮1，TensorBEM，status=succeeded，accepted=True，commit d2815a5）。
裁判：`tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-SUBMERGED-SPHERE-HAVELOCK'` → **0 行**
（系统性注册缺口，与 m17174 同类）；容差判据回退 mission goal 文本 **<5%**。

## ① 验收三件套核验

| 件 | 状态 | 证据 |
|---|---|---|
| run_artifacts | ✅ 存在（但分层） | 已提交：`smoke_economy_320/{numbers,tiers,verdict}.json + 4 CSV + run.log`、`capytaine_crosscheck.json`；**未提交**（git `??`）：顶层 `numbers.json/verdict.json/tiers.json/table_*.csv`（含 PASS 的 case_mesh 层） |
| benchmark ref vs got | ✅ 数值真实 | 审计独立复跑 `--stage cheap`：economy 带 got/ref 与已提交工件**逐位一致**（max dev 0.0）；`table_band_case.csv` ref 列 vs `analytic.submerged_sphere_wave_multipole(n_terms=20)` 审计重算：max dev 0.0（零手抄成立） |
| repro_cmd 可执行 | ✅ | 审计实跑 `PYTHONPATH=src run_case.py --stage cheap`：exit 0，~4 min，工件齐全 |

## ② 基准裁判（<5% 门）

| 层 | 面元 | 结果 | 判定 |
|---|---|---|---|
| case_mesh（案定网格，G1 门） | 1280 | 20/20 项 <5%，max **1.43%**（ka=0.5 b33：got 0.0834933 / ref 0.0847055） | **PASS**（工件未提交） |
| economy_mesh（m17242 预声明 production mesh） | 320 | 7/28 项 ≥5%，max **5.59%** | **FAIL**（honest，已提交） |
| capytaine 3.1.dev0 独立软件交叉核对 | 320/1280 | 最差 \|capy−ref\| = **3.96%** | PASS（ref 非自证） |

报告 16344 六段齐全（①ref 出处 ②方法 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式+显式 <5% 回答）；
报告数字与工件一致。报告 ⑥ 自认：提交时刻 1280 面元分级仍在跑、未落盘。

## ③ 缺陷（机器复核面）

1. **PASS 层未入库**：唯一过门的 case_mesh 层工件在 m17334 工作树中未 commit；commit d2815a5 只带 FAIL 的 economy 层。
2. **已提交 CI 守卫红**：`tests/test_case_pool_registry.py::test_every_registered_case_resolves_its_paths` 在提交态失败——registry 声明 `CASE.md` + `fig1..fig4.png` 不存在（make_doc.py 未在最终 numbers.json 上运行）。
3. 报告 economy 表写 6/12 ≥5%，最终工件为 7/28（报告先于最终运行生成，非造假，属时序差）。

## ④ 判级

- L2 独立功能实现：**具备**（真代码 输入→计算→输出：free_surface_solver3d 辐射求解 + 多极子 ref + 分级 driver）。
- L3 系统集成：**具备**（`case_pool_registry.py` + `scripts/case_pool_run.py` 平台侧真实调用面，测试覆盖）。
- L4 基准验证：**未达成**——过门层未提交、已提交层不过门、已提交守卫红。
- **判级 = L3**。L4 解锁条件：case_mesh 工件 + 生成 CASE.md/4 图入库，registry 守卫转绿。

## ⑤ 落表

`INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorBEM', 'm17334 CASE-BEM-SUBMERGED-SPHERE-HAVELOCK', 17394, 3, '<JSON>', 'board-audit')`
（见 `insert_verdict.py`，幂等）
