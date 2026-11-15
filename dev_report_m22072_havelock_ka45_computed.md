# m22072 — CASE-BEM-SUBMERGED-SPHERE-HAVELOCK：ka=4,5 案定网格实算，关闭第 4 轮 major #2

仓：TensorBEM 分支 `agent/m22072` 锚点：`CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`
性质：验证欠账（对已合并算例的估计行做独立重算 + 对抗式复核），非新功能。

## 1. 缺口

`numbers.json` 的 case_mesh（1280 面元）`ka=4.0`、`5.0` 两行为 320 面元 GCI 外推
（`est = ref + (econ-ref)/r^p`，r=2，p=1.91），`condition_number` 为两位占位符 2.05，
却计入"全带 28 项 PASS"。m21647/m21679 递交的集群作业未返回，缺口一直开着。

## 2. 做法

1. `scripts/havelock_ka45_case_mesh_compute.py`：用与表同路径的
   `run_frequency_sweep` 在 1280 面元上算 `ka=3.0`（路由控制）+ `4.0`、`5.0`，
   记录完整精度条件数、几何前置与 git 溯源；集群 `cj3884`（n330，4 核，1288 s）。
2. 路由控制：`ka=3.0` 与已提交行最大相对差 **1.79e-15** → 新值与其余行同源。
3. `scripts/havelock_ka45_apply.py`：控制不通过或几何 FAIL 即拒绝写入；
   替换两行、保留 `superseded_gci_estimate` 留痕、加 `provenance` 标记、
   按**未改动的** 5% 判据重算 tiers/refinement/verdict、追加 `degenerate_items`。
4. `scripts/havelock_ka45_figures.py`：误差图（含被替换估计的空心标记）+ 阻尼量级图。
5. `tests/test_casepool_havelock_ka45_computed.py`：9 项溯源/证据绑定守卫。

## 3. 结果

| ka | 量 | 估计误差 | 实算误差 | \|估计−实算\|/实算 |
|---|---|---|---|---|
| 4.0 | a33 | 0.4459% | 0.4437% | 0.22% |
| 4.0 | b33 | 1.1435% | 1.1622% | 1.89% |
| 4.0 | a11 | 0.4769% | 0.4758% | 0.11% |
| 4.0 | b11 | 1.0399% | 1.0717% | 3.21% |
| 5.0 | a33 | 0.4515% | 0.4494% | 0.20% |
| 5.0 | b33 | 1.1880% | 1.1947% | 0.68% |
| 5.0 | a11 | 0.4796% | 0.4786% | 0.10% |
| 5.0 | b11 | 0.5288% | 0.5942% | 6.58% |

case_mesh **PASS** 28/28，最大 1.431%（`ka=0.5 b33`，未变）；economy_mesh **FAIL** 7/28
（最大 5.591%，honest FAIL 保留）。几何：1280 面元，面积赤字 0.4765%，h_mean 0.1507。

## 4. 误差判定（逐项 vs 基准）

关键量 = 28 项相对误差，全部 < 1.5% → **<3% 成立**（最差 1.431%）。
参考侧为自实现 Havelock 多极级数（无第三方表：ka=4,5 文献在付费墙内），
故"<3%"只相对于该级数；级数自身由截断无关 1e-12 / Haskind 8.8e-16 / Lamb 极限自证。

## 5. 结论 / 局限

- 第 4 轮 major #2 关闭：全带 28 项现在都是 1280 面元实算。
- **新发现（证伪）**：GCI 对小量阻尼的外推偏差达 6.58%，"<0.2%"预判不成立。
- 未做：5120 面元加密、`spherical_patch` 测度变体、`ka≥4` 的 Haskind 残差、
  第三方文献数值。均登记为 open，不作 pass。

## 6. 复现

```
# 计算（集群）：python3 scripts/havelock_ka45_case_mesh_compute.py --out docs/.../ka45_case_mesh_computed.json
PYTHONPATH=src python3 scripts/havelock_ka45_apply.py --computed docs/case_bem_submerged_sphere_havelock/ka45_case_mesh_computed.json
PYTHONPATH=src python3 scripts/havelock_ka45_figures.py --outdir docs/case_bem_submerged_sphere_havelock
PYTHONPATH=src python3 -m pytest tests/test_casepool_havelock_ka45_computed.py tests/test_case_havelock_full_band.py -q
```
