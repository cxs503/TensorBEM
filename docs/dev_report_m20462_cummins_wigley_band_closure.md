# CASE-BEM-CUMMINS-WIGLEY round 5 — 扩展频带面板扫频管线（续跑就绪）

锚点：`CASE-BEM-CUMMINS-WIGLEY`（算例池·Wigley 船 Cummins 记忆核 6DOF；ref = Cummins 方程 + Kramers-Kronig 因果；容差 <10 %）。任务 m20462（generalist 席自驱工时），分支 `agent/m20462`。

## 1 案例介绍（物理背景/几何/参数）

Wigley III（L=3.0, B=0.3, T=0.1875, Cb=0.543），ρ=1025 kg/m³，g=9.80665 m/s²，无限水深，旋转中心 (0,0,−T/2)。Cummins 方程与 Hilbert 对：

```
(M+A∞) x'' + ∫₀ᵗ K(t−s) x'(s) ds + C x = F(t),   K(t) = (2/π)∫₀^∞ B(ω) cos(ωt) dω
B(ω) = −(2/π) PV∫₀^∞ s²(A(s)−A∞)/(s²−ω²) ds      A(ω) = A∞ + (2/π) PV∫₀^∞ B(s)/(s²−ω²) ds
```

五门不变：G1 交叉求解器 10 %、G2 K(0) 2 %、G3 Hilbert 对 10 %、G4 自由衰减 ζ vs 半功率 ζ 10 %、G5 往返 5 %。几何门（Cb/L-B/B-T）在任何数字之前先过：`Cb_abs_error = 0`，L/B=10，B/T=1.6。

## 2 怎么做的（方法/流程/关键设置/工具链）

**本锚点头号阻塞项**（round-4 `docs/dev_report_m20256_cummins_band_closure.md` §5 记录）：扩展频带 [0.10, 20] rad/s 的**真面板数据缺口**——round-2/3/4 的扫频作业（cj1658/cj1659/cj2597）都因 ssh 编排超时或预算截断未跑完，G3/G4/G5 在扩展带上从未重算。

本轮交付**续跑就绪的完整管线**（`scripts/run_cummins_wigley_closure_full.py`，新）：

1. **extended**：解 28 点 `EXTENDED_GRID` [0.10, 20] rad/s（round-4 交接预声明；round-2 `OMEGA_GRID` [0.30,12.5] 是其严格子集，因此窄/宽对照是**同一次求解**的子集，不引入新物理）。
2. **resonance**：从扩展扫频定位垂荡共振 ω_n（`heave_resonance_band`，随网格走：400 面元 7.379 rad/s）。
3. **cluster**：`closure_grid(ω_n)` = 扩展带 + 局部簇（`RESONANCE_STEP`=0.0125 rad/s，6 采样/半功率宽，G4 可判），解未缓存节点。
4. **gates**：全闭合网格上评 G1–G5，写 `results.json`（含窄带 A/B 对照与补充 KK 统计）。

**求解器已验证**：本机单点 `medium, ω=1.0`（29 s，残差 7.0e-5）与 round-2 G1 表逐项一致（surge A=2.5197/B=0.0023，sway A=136.72/B=0.0135，heave A=194.73/B=25.63，roll A=0.6525/B=5.85e-5，pitch A=52.27/B=0.0418，yaw A=91.74/B=1.38e-5）——求解器在 ω=1 上无回归。

## 3 计算结果（全部由脚本写出 JSON，零手抄）

**本轮无新面板数字**（诚实）：完整 69 点 medium 扫频已递交集群作业 `cj2737`（`--level medium --outdir artifacts/case_bem_cummins_wigley_ext`，4 核，timeout 7200），递交时排队位次 5、队列 25 作业在飞；本机粗扫（28 点 coarse）在 120 s 本地 guard 内未跑完（coarse 单点 ~7 s，28 点需 ~4 min，超本地 <10 min 短测上限）。**G1–G5 在扩展带上未重算，锚点状态不变（open）**。

## 4 结果判定（逐项 vs 基准，显式回答是否 <3 %）

* **求解器一致性**：单点 ω=1.0 与 round-2 G1 表逐项一致（误差 <3 % 量级，见 §3 数字）——**满足**，但仅单频点，非全带。
* **G3/G4/G5**：无新数字（扩展带未跑完），按 round-3/4 记 FAIL/不可判。**不声明通过**。
* **测试**：`tests/test_cummins_wigley_kk_causality.py` 全套 **16 passed**（含 round-4 新增的闭合网格/共振簇/双向 KK/G4 分辨率测试）。

## 5 注意点（坑 / 局限 / 适用范围）

* **`ref=main` 不能用于本算例**（round-4 已记录）：集群节点检出的 main 里没有 `tensorbem.benchmarks.cummins_wigley_kk_causality`（`ImportError`）。必须用 `stage_dir` 递交自己的工作树——本单 `run_compute` 已用 `stage_dir=/root/tp-agent/work/m20462`。
* **本地 guard**：本机只准 <8 核 <10 min 短测；28 点 coarse 扫频超限被收割（`TIMEOUT after 120s`）。重扫频必须走集群/计算门。
* **队列饱和**：递交时 25 作业在飞、排队位次 5；`cj2737` 需等待调度器派发，无资源自动排队。
* **G3 归一化**（round-4 已立项）：`max|A−A∞|` 对小变化量 DOF（纵荡/横荡/横摇/艏摇）把几 kg 绝对误差放大到数十个百分点；`B` 才是 Cummins 核的输入。建议条目 `CASE-BEM-CUMMINS-WIGLEY-G3-SCALE`。

## 6 复现

```
# 续跑（必须 stage_dir，不能用 ref=main）
PYTHONPATH=src python3 scripts/run_cummins_wigley_closure_full.py \
    --level medium --outdir artifacts/case_bem_cummins_wigley_ext
# 数据到手后：results.json 给出窄 vs 扩频带 G3/G5 对照、band_attributable、G4 是否可判
PYTHONPATH=src python3 -m pytest tests/test_cummins_wigley_kk_causality.py -q   # 16 passed
```

## 7 下次自找活的建议线索

1. **取回 `cj2737` 工件**：`run_compute fetch --job_id cj2737 --wait` 等终态后，`artifacts/case_bem_cummins_wigley_ext/results.json` 一次给出窄 vs 扩频带 G3/G5 对照、`band_attributable` 数字、G4 是否可判——这条能直接把锚点从 open 推到可判定/通过。
2. **G3 归一化立项**（`CASE-BEM-CUMMINS-WIGLEY-G3-SCALE`）：给出 A 向与 B 向两套归一化下的残差表，与 `B` 的物理误差（% of max B）对照。
3. **G3 合成臂约定冲突**：先用 `causality.kkr_added_mass` 在单一松弛对 `K=c e^{−at}`（已知残差 <1e-3）上标定常数因子，再上振荡对。
