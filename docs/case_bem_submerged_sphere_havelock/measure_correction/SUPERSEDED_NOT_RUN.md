# 本目录 `verdicts.json::not_run` 已过期（第 9 轮）

`verdicts.json` 的 `not_run` 写着「T3 案定网格（1280 面元）未测、逐 ka 行丢失」。
该说法来自一次**中断**的运行；同一研究随后已跑完，工件提交在未合并分支
`agent/m20156`（commit `6d372578`，总 2890 s，T3 单项 2281 s）。

第 9 轮已把该运行**逐字节**找回并独立重判：

- 工件：`docs/case_bem_submerged_sphere_havelock_round9/recovered/`（sha256 钉扎于 `PROVENANCE.json`）
- 判决与复算：`docs/case_bem_submerged_sphere_havelock_round9/adjudicate.json`、`AUDIT.md`
- 复现：`PYTHONPATH=src python3 scripts/havelock_round9_recover_case_mesh.py`

结论（案定网格 1280 面元，声明带 ka=0.1/1.0/3.0，12 项，5% 门限逐项重算）：
flat 1.4202% → 精确球面测度 0.4869%（uniform 标量对照 0.4869%，不可分辨）；
第 6 轮登记预测 0.19324% 被实测否证（实测/预测 2.52）。
本文件不改写 `verdicts.json`（历史工件原样保留），仅作索引。
