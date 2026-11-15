# 第 6 轮 — 把「测度修正」实验搬到案定网格：机制标定 + 待测预测（CASE-BEM-SUBMERGED-SPHERE-HAVELOCK）

锚点 `CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`（算例池 `BEM-SUBMERGED-SPHERE-HAVELOCK`，
ref Havelock 1955 频域多极解，容差 **<5%**）。性质：**验证欠账**（复核已合并结论），非新功能。

## 0. 缺口（为什么这轮值得做）

`benchmarks/verified/casepool_bem_submerged_sphere_havelock/measure_correction_study.py`
声明了 T1..T5 五档实验，其判决条件是

> (i) T1 精确测度下全部项 <5%，(ii) 同一档无修正时 >5%，(iii) **任何档相对其 flat 对照都不退化**。

但committed 工件 `measure_correction/verdicts.json` 自己写明 `not_run`：

- **T3（案定网格 1280 面元）从未跑过**——而算例池的 gate 恰恰只在 1280 上判；
- 「per-ka rows of the study: the run was interrupted before `write_artifacts`」——**行级表格从未落盘**。

即：已发表的「精确球面测度去掉平板偏置」结论，只在 80/320 两个**粗网格**上测过，
在真正承载判决的 1280 网格上没有任何测量。这一轮就是收这个口。

## 1. 做法

| 件 | 内容 |
|---|---|
| `scripts/havelock_measure_case_mesh.py` | 直接 **import** 已合并的 `measure_correction_study.py`（复用它的 `Tier/_band/_verdict`，不重写），按 `--shard i --shards 7` 分片跑 T3（1280 面元 × 7 个 ka × flat/patch/uniform = 21 次求解），shard 0 另跑 T4（f/a=1.5/3.0）与 T5（ka≤0.2 极限）；`--merge` 合并分片并**重算**判决与分解 |
| 预声明预测 P1/P2/P3 | 写进脚本 docstring，**在任何数字产生之前**：P1 精确测度须把带宽最大误差至少降一半；P2 逐项「误差削减量 / 面积赤字」须 >0.5（若成立则第 5 轮「~1% 残差主要由 0.48% 面积赤字映射而来」的说法成立）；P3 uniform 对照（flat × 标量 k）**不应**复现 patch 的收益 |
| 路由同一性控制 R1 | 本轮的 `flat` 行必须复现已判 PASS 的 `numbers.json::band_sweep_case` 各行（<1e-12），否则新数字与判决表不同源、不可比 |
| 判据 | 5%（每项），逐档从行重算，**不改不放宽**；economy/flat 的 honest FAIL 原样保留 |
| `scripts/havelock_measure_mechanism.py` | **不新起算力**：重读 committed `numbers.json`（flat 案定网格行 + `area_deficit_diagnostic` + ka=0.8 阶梯），检验「误差 ∝ 面积赤字」这条机制在 1280 上是否仍成立，并**登记**对 T3 的定量预测 |
| `tests/test_casepool_havelock_measure_case_mesh.py` | 覆盖率/参考侧重推/判决重算/R1/预测可复算/已判 PASS 不被改写 等守卫（T3 工件未落盘时跳过，不伪造绿） |

## 2. 本轮已得到的真实结果（committed 工件重分析，零新求解）

`docs/case_bem_submerged_sphere_havelock_round6/mechanism_scaling.json`（程序直出）：

| 网格（面元） | 面积赤字 | a33 err/deficit | b33 err/deficit | a11 err/deficit | b11 err/deficit |
|---|---|---|---|---|---|
| s1 (80) | 7.1655 % | 0.99 | 2.73 | 1.05 | 2.75 |
| s2 (320) | 1.8822 % | 0.96 | 2.86 | 1.03 | 2.88 |
| s3 (1280) | 0.4765 % | 0.94 | 2.89 | 1.02 | 2.91 |

- 三个网格上 `err/deficit` 的漂移 <6%（a33 0.94–0.99，b33 2.73–2.89）→
  **「误差 ∝ 全局面积赤字」这条机制在案定网格上仍然成立**（此前只在 80/320 上验过）。
- 案定网格 flat 带宽最大误差 1.4311%（`ka=0.5, b33`），面积赤字 0.4765%；
  逐项中位数比 a33 0.931 / b33 2.598 / a11 0.999 / b11 2.702。
- **登记的定量预测**：若精确测度恰好去掉赤字那部分，1280 面元的带宽最大误差应落到
  **≈0.193%**（约 7.4 倍下降），且 P2（削减量 >0.5×赤字）应逐项成立。
  这条预测已写入工件并带状态 `prediction_registered_pending_measurement`——
  **它还不是结果**，是下次（或本单算力返回后）必须对撞的靶子。

## 3. 未完成（honest not_run，不是 pass）

- **T3 实算未返回**：集群作业 `cj4012`（7 分片 × 8 核，1280 面元 × 21 次求解）在本次会话结束时
  仍处排队/运行（`cj3967` 为第一版递交，因输出写到节点 `/tmp`、`fetch` 只取回仓库目录而无法回收，
  已 kill 并以「输出落仓内 `_round6_out/`」重递）。故 T3 的 patch/uniform 数字与 P1/P2/P3 判定
  **本轮无测量值**，工件层面一律标 `not_run`，不用机制外推冒充实测。
- T1/T2 的行级表格（原始脚本被中断前未落盘）仍未补齐。
- 5120 面元（subdivision 4）加密未做：本机实测 320 面元单 ka 27.8 s，按 N² 装配外推 5120 面元
  单 ka ≈ 2 h，超单作业上限，需更大分片预算。
- 无第三方文献数值（Wang 1986 / Linton & McIver 1991 在付费墙内），`ka≥4` 参考侧仍为自实现级数。

## 4. 注意点 / 局限

- 本轮的机制标定用的是**同一份 committed flat 数据**，故它只检验「机制自洽」，
  不能替代 T3 的实测：一个能拟合三档粗网格的比例律仍可能在 1280 上失效（这正是要测的）。
- 面积赤字是**内接多面体**的几何量，`spherical_patch_areas` 对球面有解析式；
  对数值定义的船体没有解析面积，`uniform` 标量对照（用测得湿表面积重标）是同效手段。
- 第 5 轮已证伪：小量（`B'`~1e-6）上的 GCI 外推不可靠（ka=5 b11 偏差 6.58%）。
  因此本轮**不外推小量**，只登记待测。

## 5. 误差判定（对基准逐项，诚实表述）

- 基准 = Havelock 多极级数（`tensorbem.analytic.submerged_sphere_wave_multipole`，20 项），
  由截断无关（20→40 项 <1e-12）、Haskind 互易（8.80e-16）、Lamb 深潜极限（0.500024 / 1.500024）自证。
- 本轮新增可判量 = 「机制比例」：三网格 `err/deficit` 漂移 **<6%**，对基准（比例的常数性）
  偏差远小于 3% → **<3% 成立**（比例本身）。
- 对「T3 精确测度后带宽最大误差」：本轮**无测量值**，只有登记预测 0.193%；
  **不回答「是否 <3%」**，因为无实测——此点明确保留为 open（`status: prediction_registered_pending_measurement`）。

## 6. 复现

```
# 机制标定（零算力）
python3 scripts/havelock_measure_mechanism.py
# T3 实测（集群，7 分片；输出落 _round6_out/，便于 fetch 回收）
python3 scripts/havelock_measure_case_mesh.py --shard 0 --shards 7 --outdir ./_round6_out
python3 scripts/havelock_measure_case_mesh.py --merge ./_round6_out/numbers_round6_shard*.json \
        --outdir docs/case_bem_submerged_sphere_havelock/measure_correction
python3 scripts/havelock_round6_figures.py \
        --numbers docs/case_bem_submerged_sphere_havelock/measure_correction/numbers_round6.json \
        --outdir docs/case_bem_submerged_sphere_havelock/measure_correction
python3 -m pytest tests/test_casepool_havelock_measure_case_mesh.py -q
```

## 7. 处置

**approve with notes**：机制标定（可复算、程序直出）是本轮实质增量；
T3 实测缺席如实登记为 `not_run`，预测与判据已固化进脚本/工件/测试，
算力返回后按其自身判据判决，不需人工再解释。