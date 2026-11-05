# TensorBEM Ursell 发布数字复核与文档-代码自动判据（m2186）

## 结论摘要
结清 2026-10-24 扩编审计遗留的 Ursell honest FAIL。

- **实测**：128 面元附加质量误差 `8.513e-3`（原声明 `7.72e-3`），阻尼 `1.336e-2`（原声明 `1.30e-2`）。
- **归因**：参考解（A=2545.17, B=2718.15）与 MFS 行（2544.69, 1.889e-4）在 HEAD 上逐位复现文档；每面元高斯点数 12/16/24/32 与周向网格相位 0/半面元对结果无影响（相对差 <1e-15）。故偏差只在面元解、且不由任何暴露参数产生 ⇒ **文档溯源缺口，不是求解器回归**（收敛阶仍为一阶：1.67→0.851→0.429e-2，比值 1.97/1.98）。
- **第二处缺陷**：文档 MFS 行标注「64 面元」，实测数值对应 **128 面元**（真 64 面元 MFS = 2543.24/3.53e-3）。已改标注。
- **未修好精度**：真实精度比原声明差约 10%，本次只把声明改成真实值，honest FAIL 判定不覆盖。

## 交付物
| 文件 | 作用 |
|---|---|
| `src/tensorbem/doc_claims.py` | 发布数字 claim 注册表 + markdown 解析 + 比对；`--write` 重测生成工件 |
| `artifacts/ursell_doc_claims.json` | 23 个实测量的判据工件（含 numpy 版本、参数、MFS 网格） |
| `tests/test_doc_claims.py` | 7 个用例：解析范围隔离、注册覆盖率、文档-工件一致、现场重跑复现 |
| `docs/marine_bem_benchmarks.md` §4.6 | 表格与正文按 HEAD 更新 + 勘误说明 |
| `README.md` | 里程碑 4 头条数字更新（8.5e-3 / 5.3e-3） |
| `docs/figures/ursell_doc_claims_recheck.png` | 新旧误差收敛对比图 |

## 重测表（HEAD）
| 面元数 | A | A 误差 | B | B 误差 | 近/远场比 |
|---|---|---|---|---|---|
| 64 | 2587.76 | 1.674e-2 | 2787.24 | 2.542e-2 | 1.01054 |
| 128 | 2566.83 | 8.513e-3 | 2754.47 | 1.336e-2 | 1.00534 |
| 256 | 2556.09 | 4.293e-3 | 2736.75 | 6.841e-3 | 1.00269 |
| Ursell | 2545.17 | — | 2718.15 | — | — |
| MFS(128) | 2544.69 | 1.889e-4 | 2715.76 | 8.823e-4 | — |

## 验证
- `python -m tensorbem.doc_claims` → `checked 23 published numbers / all published numbers reproduce`（exit 0）
- `pytest tests/test_doc_claims.py` → 7 passed
- `pytest tests/test_ursell_benchmark.py tests/test_doc_claims.py` → 25 passed in 55.7s
- 全量套件：本机 load15≈114（nproc=128）时 5 分钟仅跑到 21%，受 shell 时限未能跑完，未取得全绿证据 ⇒ 未验证项如实标注。改动不含既有代码路径修改（仅新增模块 + 文档），回归风险限于新模块。

## 下次可接的线索
1. 3-D 椭球表 1280 面元行（`subdivisions=3`）仍未直接复现，可把 `doc_claims` 的注册表模式扩展到 `docs/marine_bem_benchmarks.md` §4.4 表（需一次较重算，宜在低载时做）。
2. 其余里程碑发布数字（球体 20–1280 面元、Hulme 半球、Wang/Linton 潜没球）同样只有散文声明，可按同一 claim 注册表模式批量纳管。
