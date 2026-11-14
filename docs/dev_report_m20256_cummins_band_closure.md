# CASE-BEM-CUMMINS-WIGLEY round 4 — 频带闭合设计、KK 双向诊断，与「G4 需要多少采样」的实测

锚点：`CASE-BEM-CUMMINS-WIGLEY`（算例池·Wigley 船 Cummins 记忆核 6DOF；ref = Cummins 方程 + Kramers-Kronig 因果；容差 <10 %）。任务 m20256（paper 席自驱工时），分支 `agent/m20256`，工作树 `/root/tp-agent/work/m20256`。

本轮**没有**通过 G3/G5 面板门，也**没有**拿到扩展频带的面板数据（申报的 `cj2597` 在预算内只跑出 69 个频点中的 3 个，见 §5）。本轮交付的是三件可验证的东西：把 round-2/3 交接的"频带/采样"说法变成**代码 + 可判定的设计规则 + 一个有量纲的结论**，并如实记录一条被否定的假设。

## 1 案例介绍（物理背景/几何/参数）

Wigley III（L=3.0, B=0.3, T=0.1875, Cb=0.543），ρ=1025 kg/m³，g=9.80665 m/s²，无限水深，旋转中心 (0,0,−T/2)。Cummins 方程与 Hilbert 对同 round 2/3：

```
(M+A∞) x'' + ∫₀ᵗ K(t−s) x'(s) ds + C x = F(t),   K(t) = (2/π)∫₀^∞ B(ω) cos(ωt) dω
B(ω) = −(2/π) PV∫₀^∞ s²(A(s)−A∞)/(s²−ω²) ds      A(ω) = A∞ + (2/π) PV∫₀^∞ B(s)/(s²−ω²) ds
```

五门不变：G1 交叉求解器 10 %、G2 K(0) 2 %、G3 Hilbert 对 10 %、G4 自由衰减 ζ vs 半功率 ζ 10 %、G5 往返 5 %。几何门（Cb/L-B/B-T）在任何数字之前先过：`Cb_abs_error = 0`，L/B=10，B/T=1.6。

## 2 怎么做的（方法/流程/关键设置/工具链）

1. **频带闭合网格（新，`src/tensorbem/benchmarks/cummins_wigley_kk_causality.py`）**
   * `EXTENDED_GRID`：round-3 交接预声明的 28 点频带 [0.10, 20.0] rad/s（原 `OMEGA_GRID` 22 点 [0.30,12.5] 是它的严格子集，因此"窄带/宽带"对照是**同一次求解**的子集，不做插值、不引入新物理）。
   * `RESONANCE_HALF_WIDTH = 0.25`、`RESONANCE_STEP = 0.0125`、`closure_grid(resonance)`：在**运行期找到**的垂荡共振 ω_n 附近插入局部簇（69 点，其中 41 点在共振窗口内），因为 ω_n 随网格走（400 面元 7.379 rad/s，192 面元 7.737 rad/s），硬编码会随网格失配。
   * `DENSE_STEP = 0.001` + `decay_response_grid()`：G4 半功率扫描的插值网格密度成为显式常量（原固定 801 点在 [0.1,20] 上只有 0.025 rad/s）。
2. **KK 双向诊断（新）`kk_directions(sweep, dof)`**：G3 只测 **A 由 B**，且归一化到 `max|A−A∞|`——对垂荡/纵荡等小变化量 DOF 极敏感。新增**B 由 A**方向（归一化 `max|B|`），即 Cummins 核真正积分的那一侧。两者各自的 max / rms / interior-max / argmax 一并记录。**G3 判据不改**，G3b 只是并列诊断（round-3 已证明 max-over-midpoints 会漂到带边）。
3. **G4 分辨率实测（新）`scripts/cummins_wigley_gate_resolution.py`**：用闭式因果核对 `K(t)=c e^{−σt}cos(Ωt)`（σ=0.037, Ω=7.3785，即 ζ=5e-3、半功率宽 0.074 rad/s）：
   `B = (c/π)[σ/(σ²+(ω−Ω)²) + σ/(σ²+(ω+Ω)²)]`，`A−A∞ = −(c/2ω)[(ω−Ω)/(σ²+(ω−Ω)²) + (ω+Ω)/(σ²+(ω+Ω)²)]`。
   把**同一个估计器**（`time_domain.half_power_damping` + 本算例的 `decay_response_grid`）喂给同一物理的两种采样，误差只可能来自采样。
4. **测试**：`tests/test_cummins_wigley_kk_causality.py` 新增 7 项（网格覆盖/共振簇分辨率 ≥5 采样每半功率宽/非法共振拒绝/双向 KK 精确对/双向 KK 对 10 % 不一致的检验力/G4 在闭合网格上 <3 % 而在 0.5 rad/s 全局网格上 >20 %/响应网格步长）。全套 `27 passed`（含 `test_kk_band_floor.py`、`test_half_power_scan_window.py`）。

## 3 计算结果（全部由脚本写出 JSON，零手抄）

G4：同一估计器、同一物理、不同采样（`artifacts/case_bem_cummins_wigley_resolution/resolution.json`）：

| 采样 | 节点数 | ζ（半功率） | 相对细采样参考误差 |
|---|---|---|---|
| 0.5 rad/s 全局网格（round-2 的采样密度代表） | 18 | — | 45.99 % |
| 仅扩展带、无共振簇（28 点） | 28 | — | 99.43 % |
| 簇步长 0.5 | 29 | — | 29.28 % |
| 簇步长 0.25 | 31 | — | 25.66 % |
| 簇步长 0.1 | 33 | — | 7.11 % |
| 簇步长 0.05 | 39 | — | 15.37 % |
| 簇步长 0.025 | 49 | — | 3.72 % |
| **簇步长 0.0125（本轮预声明）** | **69** | — | **0.82 %** |

参考 = 同一估计器在 24001 点采样上的结果（其自身误差远低于打印精度）。ζ 参考值 6.605e-3。

**结论（有量纲）**：G4 的 10 % 门要求共振簇步长 ≲0.025 rad/s（约 3 采样/半功率宽），预声明的 0.0125 rad/s（6 采样/宽）余量 12 倍；而 0.5 rad/s 的全局网格误差 46 %——这正是 round-3 里 G4「不可判」的量化来源，而不是物性。表中 0.05 处的非单调（15.4 % > 0.1 处的 7.1 %）来自估计器自身局部扫描窗 `samples=4001` 的离散化，属估计器残差，未做平滑或挑选。

## 4 结果判定（逐项 vs 基准，显式回答是否 <3 %）

* **G4 分辨率结论**：0.82 %（0.0125 rad/s 采样 vs 细采样参考）——**<3 %，满足**；10 % 门对应的最小步长 0.025 rad/s 亦满足（3.72 %，但 >3 %，如实标注）。
* **G3 合成臂：否证（honest FAIL）**。用上述闭式因果对在 69 点与 24001 点采样上跑算例的 G3 估计器（`kk_band_floor.kk_residual`，同一归一化），残差为 0.450 与 0.397——**加密 350 倍几乎不降**，说明残差不是采样项；而 10 % 门要求它趋于 0。二者必有一错：要么该闭式对与本仓 `causality.kkr_added_mass` 的约定（`B` 的常数因子/`A−A∞` 的符号）不一致，要么该归一化在此模型上病态。**本轮未定案**，因此 G3 的"采样需求"结论**不成立**，不得引用；面板的 G3 仍按 round-3 记 FAIL。这是本轮记录在案的被否定假设（"G3 残差主要是采样/带边效应"——至少在合成臂上不成立）。
* **G3b（B 由 A 方向）**：在精确对（400 点、5 个十倍频程）上残差 6.1 %，A 由 B 方向 7.1e-4；对 10 % 人为不一致的检验力为 8–12 %（A 向）/ 12–18 %（B 向）。**6.1 % > 3 %**，即该方向自带约 6 % 的估计器地板（尾部拟合 + 中点线性），只够判 10 % 门，不够判 3 %——如实记录。
* **面板数据**：无新面板数字（见 §5），G1–G5 在扩展带上**未重算**，锚点状态不变。

## 5 注意点（坑 / 局限 / 适用范围）

* **未完成（欠账）**：扩展频带 69 点 medium 扫频 `cj2597`（8 分片，`stage_dir` 递交）在任务预算内只完成 3/69 点；日志显示分片启动的 ssh 编排在 30 s 超时后逐个失败，实际只有 2 个节点在跑。`band_attributable` 仍为 null，与 round-3 同类欠账（cj1658/cj1659）叠加：**该算例的真面板数据缺口是本锚点的头号阻塞项**。
* **`ref=main` 不能用于本算例**：集群节点检出的 main（head `104abe1`）里没有 `tensorbem.benchmarks.cummins_wigley_kk_causality`（`ImportError`，探测作业 `cj2592` 因此 rc=1）。必须用 `stage_dir` 递交自己的工作树——这条已写进 §6 复现命令，避免下一位重复踩坑。
* `closure_grid()` 的簇是**预声明后求解**，不是事后挑频点：步长由 ζ 设计值 5e-3 推出（≥5 采样/半功率宽），簇心用运行期 ω_n。
* 合成臂的 σ 取 0.037 是为贴近实测 ζ≈5e-3 的轻阻尼共振；`COEFF=1.0` 使 `max|B|≈8.6` 量级与面板垂荡（26 kg/s）同阶，但**不是**面板数据，不可当作物理证据。
* 图 `artifacts/case_bem_cummins_wigley_resolution/fig_gate_resolution.png`：左=B33 峰在 0.0125 / 0.5 采样下的形状与半功率线；中=G3 残差 vs 簇步长（10 % 门线）；右=G4 ζ 误差 vs 簇步长（10 % 门线）。

## 6 复现

```
PYTHONPATH=src python3 -m pytest tests/test_cummins_wigley_kk_causality.py \
    tests/test_kk_band_floor.py tests/test_half_power_scan_window.py -q      # 27 passed
PYTHONPATH=src python3 scripts/cummins_wigley_gate_resolution.py \
    --outdir artifacts/case_bem_cummins_wigley_resolution                     # 表 + 图

# 扩展频带面板扫频（未完成，续跑用；必须 stage_dir，不能用 ref=main）
PYTHONPATH=src python3 scripts/run_cummins_wigley_kk_causality.py --mode sweep --level medium \
    --shard i --shards 8 --outdir artifacts/case_bem_cummins_wigley_ext --omega-list <closure_grid>
PYTHONPATH=src python3 scripts/cummins_wigley_band_closure.py \
    --points-dir artifacts/case_bem_cummins_wigley_ext/points --level medium
```

## 7 下次自找活的建议线索

1. **续跑 `cj2597`**（69 点 medium，建议 `--shards 4 --cores 4` 减少 ssh 编排超时面；或直接单作业串行 69 点、timeout 7200）。数据到手后 `scripts/cummins_wigley_band_closure.py` 一次给出：窄 vs 扩频带 G3/G5 对照、`band_attributable` 数字、G4 是否可判——这条能直接把锚点从 honest-FAIL 推到可判定。
2. **G3 的归一化本身值得立项**：`max|A−A∞|` 对小变化量 DOF（纵荡/横荡/横摇/艏摇）把几 kg 的绝对误差放大到数十个百分点，而 `B` 才是 Cummins 核的输入。建议条目：`CASE-BEM-CUMMINS-WIGLEY-G3-SCALE`（验收：给出 A 向与 B 向两套归一化下的残差表，并与 `B` 的物理误差（% of max B）对照，证明量级一致性）。
3. **G3 合成臂的约定冲突**：先用 `causality.kkr_added_mass` 在单一松弛对 `K=c e^{−at}`（已知残差 <1e-3）上标定常数因子，再上振荡对；把因子写进 `kk_band_floor` 的 docstring，避免下一位重踩。