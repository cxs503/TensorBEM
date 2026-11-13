## ① 案例介绍（被审对象 · 非新算例）

被审对象 = 任务 m17207（已 succeeded）交付的算例 **CASE-BEM-KP505-OPENWATER**（TensorBEM，算例池 case_code=BEM-KP505-OPENWATER，tol_policy `<5%`）。本轮 m17239 是**交差核验单**：不重做算例，只取证、判级、落表。

物理背景：KP505 是 KRISO 为 KCS 集装箱船设计的五叶定距桨，敞水（open-water）性能由无量纲系数 K_T、10K_Q、η_0 随进速比 J=V/(nD) 表征；该算例检验 TensorBEM 的涡格升力面核能否复现实验敞水曲线。

几何与参数（全部来自被审论文 Table 1, p.6，非本审计自选）：FPP，5 叶，右旋，D=0.25 m，毂径比 0.18，平均 P/D=0.950，Ae/Ao=0.800，剖面族 NACA66；未发表量（弦长/径向螺距分布/拱度/侧斜/纵倾/厚度律）由仓库既有重建（c(u)=A(u+0.12)^0.8(1-u)^0.5，常螺距，平剖面，t/c=0.06）代理，其敏感性单列。

计算配置（被审交付的 canonical_config）：n_radial=16、n_chord=2、cosine 径向分布、wake_model=geometric、wake_length=1.0、wake_steps=24、rev_per_second=10、viscous=True、core_ratio=0.05。

## ② 怎么核的（本轮方法 · 基准判据优先）

1. **看代码不看报告**：读 `kp505_openwater_case.py`（738 行）、`propeller.py` 的 +54 行、6 条契约测试；确认参考列是 **import 自 `kp505_openwater_bem.ANCHOR`**（由 PDF 表 2 逐行 verbatim 解析），驱动里没有任何手打锚值。
2. **基准裁判**：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-KP505-OPENWATER'` → **0 行**（机器裁判缺席，见 §⑤ 发现 F1）；因此改按算例池 `tensor_case_pool` 声明的 `<5%` 判据 + 仓库锚表逐项核。
3. **锚表独立复算**：用 pypdf 直接抽 PDF 第 10 页 Table 2，与被审转录逐位比对（`verify_numbers.py`，不 import 被审驱动）。
4. **got 数值独立复算**：从交付 CSV/JSON 读测量值，**自行**重算三项百分比、重数 5% 带内点数、重判 PASS/FAIL。
5. **工件与复现**：核对 `runs/m17207/` 四件套（json/csv/md/3×png）存在且非空；在 loongarch64 上重跑 `repro_cmd`，与交付工件逐位比对。
6. **回归面**：跑 6 条契约测试；另跑仓库既有的 `test_propeller_wageningen_crosscheck.py` 以核实"既有红测试"声明。

## ③ 计算结果（程序化注入，零手抄）

被审声明的汇总：{"n_points": 10, "n_admissible": 9, "n_pass_by_quantity": {"KT_pct": 3, "10KQ_pct": 3, "eta0_pct": 1}, "n_pass_all_three_5pct": 0, "n_pass_all_three_5pct_admissible": 0, "worst": {"KT_pct": {"J": 1.0, "error_pct": -240.7098986655049}, "10KQ_pct": {"J": 1.0, "error_pct": -115.86944612455544}, "eta0_pct": {"J": 1.0, "error_pct": 786.671768889141}}}

本轮独立复算：{"recomputed_pass_counts_10pts": {"KT": 3, "10KQ": 3, "eta0": 1}, "worst_abs_pct_all_10": {"KT_pct": 240.7098986655049, "10KQ_pct": 115.86944612455544, "eta0_pct": 786.671768889141}, "worst_abs_pct_admissible_only": {"KT_pct": 64.85233944301325, "10KQ_pct": 53.5176417018397, "eta0_pct": 24.38494550656685}, "mean_abs_pct_excluding_J1": {"KT_pct": 15.113297681678056, "10KQ_pct": 15.594636773307498, "eta0_pct": 10.684627916558693}, "anchor_vs_pdf_max_abs_diff": 0.0, "max_abs_agreement_diff_vs_reported": 1.4210854715202004e-14, "paper_error_column_max_dev_pp": 1.1782758620689564, "verdict_agrees": true}

| J | K_T got | K_T ref | Δ% | 10K_Q got | 10K_Q ref | Δ% | η_0 got | η_0 ref | Δ% | 可容许 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.5286 | 0.494 | +7.00 | 0.6529 | 0.683 | -4.41 | 0.1288 | 0.1151 | +11.93 | yes |
| 0.2 | 0.4734 | 0.447 | +5.91 | 0.6104 | 0.636 | -4.03 | 0.2469 | 0.2237 | +10.36 | yes |
| 0.3 | 0.4163 | 0.392 | +6.19 | 0.5599 | 0.585 | -4.29 | 0.3550 | 0.3199 | +10.95 | yes |
| 0.4 | 0.3570 | 0.343 | +4.10 | 0.5014 | 0.530 | -5.39 | 0.4533 | 0.4120 | +10.03 | yes |
| 0.5 | 0.2958 | 0.290 | +2.01 | 0.4349 | 0.471 | -7.66 | 0.5412 | 0.4900 | +10.46 | yes |
| 0.6 | 0.2326 | 0.241 | -3.51 | 0.3605 | 0.409 | -11.86 | 0.6160 | 0.5627 | +9.48 | yes |
| 0.7 | 0.1673 | 0.193 | -13.34 | 0.2781 | 0.342 | -18.70 | 0.6701 | 0.6287 | +6.59 | yes |
| 0.8 | 0.0999 | 0.141 | -29.13 | 0.1876 | 0.270 | -30.50 | 0.6781 | 0.6649 | +1.98 | yes |
| 0.9 | 0.0306 | 0.087 | -64.85 | 0.0892 | 0.192 | -53.52 | 0.4908 | 0.6491 | -24.38 | yes |
| 1.0 | -0.0408 | 0.029 | -240.71 | -0.0171 | 0.108 | -115.87 | 3.7893 | 0.4274 | +786.67 | **no** |

**复现**：`PYTHONPATH=src python -m tensorbem.benchmarks.kp505_openwater_case <outdir>` 在 loongarch64（同环境 python 3.11.6/numpy 2.4.6）35 s 跑完，`kp505_case_table.csv` 与交付件 **diff 为空**，JSON 的 canonical/ladders/probe/criteria/verdict 数值逐项 ≤ 1e-12 相同。契约测试 6/6 通过。

## ④ 结果图

图为本轮独立绘制（数据源=交付工件 CSV/JSON + PDF 重抽锚表）：面板 1–3 为 K_T / 10K_Q / η_0 随 J 的 got-vs-anchor 曲线；面板 4 为三项相对误差与 ±5% 验收带；第二张图给出可容许点上的逐点 |误差| 柱状对照。

## ⑤ 注意点（审计实测踩坑 + 局限）

- **F1 机器裁判缺席（阻塞级，平台侧）**：`tensor_benchmark_cases` 对本 anchor 0 行；`tensor_case_pool` 行 BEM-KP505-OPENWATER 仍为 `status=dispatched, mission_id=17207`（终态未回写）。判据口径只能取算例池 `<5%`，机器门无法自动复裁。
- **F2 交付未并入生态仓（中）**：`/root/tensor-ecosystem/TensorBEM` main 无该文件；代码只在 `refs/heads/agent/m17207`（commit a33c68a）。故 L3「系统集成」无法成立——生态仓内 grep 不到任何调用方，能力尚未落在产品侧。
- **F3 报告"最差误差"取的是数值崩点（轻，表述层面）**：-240.71% / -115.87% / +786.67% 全部来自 **J=1.0 这个被自己判为不可容许**的点（K_T<0、η=3.79 越动量上界）。可容许点上的真实模型缺口是 J=0.9 的 -64.85% / -53.52% / -24.38%，可容许点平均 |误差| 为 15.11 / 15.59 / 10.68%。结论（FAIL）不变，但"最差"一词容易被读成模型误差量级，宜改称"含数值崩点的最差值"。
- **F4 工件环境 commit 记的是父提交**：`environment.git_commit=96f19f7`（基线），而含代码的是 a33c68a；本轮已用 a33c68a 工作树复跑逐位一致，证明可复现，但工件内的 commit 指针本身指不到代码（溯源错位，轻）。
- **F5 参考列是二次转录**（被审方已诚实披露）：论文未引一次 KRISO/KCS workshop 数据；本轮额外发现该论文**自身误差列**与自身 K_T/10K_Q 列最大只差 1.18 pp（J=1.0 K_T 是outlier；符号约定为 |CFD−EXP|/EXP），说明转录忠实于 PDF，但不能替代一次实验报告。锚表 `benchmarks/anchors/kp505_openwater.json` 已自标 SECONDARY_REPRODUCTION。
- **F6 既有红测试确认为既存债（轻）**：`tests/test_propeller_wageningen_crosscheck.py` 6 例 FAIL 在基线 96f19f7 同样复现（同 6 例，BEM 侧 K_T=0），与本单改动无关。
- **本轮未做**：不重新设计桨叶重建、不做加拱度/几何反演、不动任何向锚表的标定；集群全量回归未重跑（沿用被审方 cj1456 声明，未复核）。

## ⑥ 误差判定（逐项，显式回答）

| 量 | 5% 带内点数（10 点） | 最差（可容许点） | 最差（含不可容许点） | 是否 <3% |
|---|---|---|---|---|
| K_T | 3/10 | -64.85%（J=0.9） | -240.71%（J=1.0） | **否**（1/10 进 3% 带） |
| 10K_Q | 3/10 | -53.52%（J=0.9） | -115.87%（J=1.0） | **否**（0/10） |
| η_0 | 1/10 | -24.38%（J=0.9） | +786.67%（J=1.0） | **否**（1/10） |

三项全达标 0/10；可容许点 0/9。**是否 <5%（本算例判据）？否。是否 <3%（生态严格线）？否。** 独立复算与被审声明逐点一致（最大偏差 1.4e-14 %，纯浮点），甲/乙双方对"FAIL"无争议——这是 honest FAIL：数字来自真实求解、有 GCI 阶梯与敏感性支撑，缺的是模型形式（平剖面/常螺距/无拱度），不是管线。

## ⑦ 判级与处置

- **判级：L2**（独立功能实现：真代码 输入→计算→输出；入口 `python -m tensorbem.benchmarks.kp505_openwater_case`；运行工件非空；6 条契约测试）。
- **不授 L3**：交付只在 `agent/m17207` 分支，生态仓 main 无文件、无任何调用方（gate 的 integration_callers grep 复核必失败）；且本单是在既有 `propeller.open_water_point` 之上加算例驱动器 + 一个 warning 护栏，未扩展求解器能力。
- **不授 L4**：L4 要求被验证能力通过基准——本算例 0/10 点三项全达标，最差可容许点误差 -64.85%，是最基本判据的 13 倍。
- **不授 L5**：无一次实验/独立实现对照，参考列是二次转录。
- **处置（request changes）**：(1) 合并 agent/m17207 或明确不合并，避免算例悬在分支上；(2) 平台侧登记 `tensor_benchmark_cases` 行（anchor_slug=CASE-BEM-KP505-OPENWATER, metric=K_T/10K_Q/η_0 vs Table 2 EXP, tolerance=<5%），并把 `tensor_case_pool` 行置终态，否则机器裁判永远缺席；(3) 追赶方向按被审方诊断：模型形式（拱度/径向螺距/非平剖面），不是加密网格（spanwise GCI 已达 13–14% 且观测阶 p<0）。
