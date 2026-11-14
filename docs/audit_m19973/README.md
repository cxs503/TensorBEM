# docs/audit_m19973 — 交差核验 m19973 `CASE-BEM-HELMHOLTZ-SPHERE` 的审计工件

审计报告正文：`../audit_m19973_case_bem_helmholtz_sphere.md`
verdict：`tensor_maturity_verdicts` id=810（repo=TensorBEM，scope=`m19973 CASE-BEM-HELMHOLTZ-SPHERE`，
mission_id=19997，level=**3**，verdict_by=`board-audit`）
平台报告：`tensor_reports` id=18232

| 文件 | 内容 |
|---|---|
| `indep_check.py` / `indep_check.json` | ref 腿四路线独立复算：A mpmath Riccati–Bessel 级数、B 直接角积分 ∫|f|²dΩ、C 光学定理、D 瑞利极限；附远场求积敏感性（64/96/128/192 点） |
| `repro/run_study.py` | m19973 原件逐字节副本（sha256 见 `AUDITED_CASE_SHA256.txt`），路径层级保持 `parents[3]=repo root` |
| `repro/results.json` / `repro.log` | 审计方重跑输出：14/14 行与原始载荷 **bit-identical**（max |Δσ| = 0.0） |
| `audited_case/` | 被审交付物原件副本：`results.json`（14 行）、`report.md`（六段）、4 张交付 png |
| `figs/` | 本审计报告图：fig1 σ vs ka（vs Mie ±6%）、fig2 误差预算+求积不变性、fig3 ref 三路线一致性+重跑 bit-identical |
| `make_audit_figures.py` | 三张图的生成脚本（数字全部读自 JSON） |
| `insert_verdict.py` / `update_verdict_paths.py` | verdict 落表 / 证据路径改指 `docs/`（因 `runs/`、`artifacts/` 被 `.gitignore` 排除） |
| `post_report_m19997.py` | 平台报告发布脚本（六段正文 + 三图内嵌，零手抄） |
| `AUDITED_CASE_SHA256.txt` | 被审 7 个文件的 sha256 清单 |
| `verdict_summary.json` | verdict 落表回执快照（id / gate_state） |

## 一键复现审计结论

```bash
# 14 例复算（≈3 min，单进程）
cd docs/audit_m19973/repro && OMP_NUM_THREADS=1 python3 run_study.py

# ref 腿四路线 + 求积敏感性（≈90 s）
cd <repo root> && OMP_NUM_THREADS=1 python3 docs/audit_m19973/indep_check.py

# 回归面（既有 5% 门）
cd <repo root> && OMP_NUM_THREADS=1 python3 -m pytest tests/test_acoustic3d.py -q -k "sphere or mie or neumann"

# 裁判表查询（应返回 0 行 —— 这就是 L4 拒绝的直接理由）
# SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-HELMHOLTZ-SPHERE';
```

## 结论摘要（详见正文）

- 三件套（run_artifacts / ref-vs-got / repro_cmd）**实存且可核**：ref 独立复算至 8.7e-15，got 14/14 bit-identical。
- 6% 判据 **13/14** 通过，唯一 FAIL = ka=0.25 / 320 面元 **6.119%**；成因经三方探针确认为几何离散
  （求积 64→192 点漂移 ≤5.3e-15；误差比 3.89× ≈ 内接面积亏损比 3.95×；14/14 系统性负偏差）。
- **L3 判定**：L2/L3 实质成立；**L4 拒绝**——F1 交付物 0 提交（目录 untracked，`branch=None`）、
  F2 `tensor_benchmark_cases` 该 anchor 0 行（裁判腿断）。
- disposition：**request changes**（提交目录 / 登记裁判行 / 把失败点写进 pytest 并收录 README / 修收敛措辞）。