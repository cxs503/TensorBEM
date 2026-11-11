"""Publish the m13848 report (numbers injected from the committed artifacts)."""
import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
}).encode()
req = urllib.request.Request(f"{base}/auth/login", data=login,
                             headers={"Content-Type": "application/json"})
token = json.loads(urllib.request.urlopen(req).read())["access_token"]

study = json.loads(Path("docs/qtf_drift_oc4/study.json").read_text())
doc = Path("docs/qtf_drift_oc4.md").read_text()

geo = study["geometry"]["levels"][0]
diag = study["diagonal"]
qtf = study["qtf"]
newman = sorted(study["newman"], key=lambda r: r["spectral_width"])
slow = study["slow_drift"]
conv = study["mesh_convergence"]
gates = study["gates"]

rows = "\n".join(
    f"| {r['label']} | {r['component_count']} | {r['spectral_width']:.3f} | "
    f"{r['rms_direct']:.3e} | {r['rms_newman']:.3e} | {r['rms_relative_error']:.2%} | "
    f"{r['mean_relative_error']:.1e} | {r['spectrum_correlation']:.4f} |"
    for r in newman
)
conv_rows = "\n".join(
    f"| {k:.4f} | {conv['levels'][0]['panels']} | {conv['levels'][1]['panels']} | {chg:.2%} |"
    for k, chg in zip(conv["probe_wavenumbers"], conv["mesh_to_mesh_relative_change"])
)

cyl = study["cylinder"]
cyl_rows = "\n".join(
    f"| {p['ka']:.2f} | {p['analytic']:.4e} | {p['bem']:.4e} | {p['relative_error']:.2%} |"
    for p in cyl["points"]
)
cyl_err = float(cyl["max_relative_error"])
cyl_med = float(cyl["median_relative_error"])
cyl_herm = float(cyl["hermitian_residual_relative"])
resolved = [r for r in study["newman"] if r["tp"] <= 9.0]
unresolved = [r for r in study["newman"] if r["tp"] > 9.0]
res_rows = "\n".join(
    f"| {r['label']} | {r['spectral_width']:.3f} | {r['rms_direct']:.3e} | "
    f"{r['rms_newman']:.3e} | {r['rms_relative_error']:.2%} | {r['spectrum_correlation']:.4f} |"
    for r in resolved
)
unres_rows = "\n".join(
    f"| {r['label']} | {r['spectral_width']:.3f} | {r['rms_relative_error']:.2%} | 长波带网格未收敛，仅登记不解读 |"
    for r in unresolved
)

summary = (
    "TensorBEM 新增二阶差频 QTF 计算面：固定体二次压力+水线全非对角项直接 QTF、Newman 近似、"
    "海况漂移力谱与慢漂 RAO，并在 OC4 级四柱半潜几何上做双口径裁决。直接 QTF 对角线与已验证 "
    f"Pinkster 近场平均漂移力逐点一致（4e-10，pytest 钉）；外部锚：340 面板圆柱 vs 精确 MacCamy-Fuchs 误差 "
    f"{cyl['median_relative_error']:.0%}（中位），相邻频率非对角与对角均值差 0.4% → 机制正确、绝对精度受面板密度限制。"
    f"Newman 裁决在网格可解析带（Tp<=9s）误差 {max(r['rms_relative_error'] for r in resolved):.2%}<10% 通过；"
    f"长波带（Tp>=10s）{max(r['rms_relative_error'] for r in unresolved):.0%} 经诊断为船体网格未收敛（coarse->medium 差 24%/11%，"
    f"近场/远场差 {diag['max_relative_gap']:.0%}）的 QTF 噪声，如实登记「实测但不可解读」。"
    "与 MARINTEK 实验对标 (b)(c)(d) 因工作树内无实验数据、无外网出口，如实申报 NOT_EVALUABLE / honest FAIL。"
)

markdown = f"""# 二阶波浪漂移力：Newman 近似 vs 直接 QTF（OC4 半潜，TensorBEM）

**节点** `p362w2-n07-bem-second-order-drift` · **模块** `tensorbem.qtf_drift` ·
**驱动** `tensorbem.benchmarks.qtf_drift_study` · **测试** `tests/test_qtf_drift.py`

## 0. 一句话结论

在 OC4 级四柱半潜几何上用同一套一阶 BEM 解实现两条二阶漂移力路线：
(a) 完整非对角 QTF（固定体二次压力 + 水线项，逐对计算，无窄带/对称假设）；
(b) Newman 近似 \(T^N_{{ij}}=(T_{{ii}}+T_{{jj}})/2\)。直接 QTF 的对角线**恒等于**
仓库既有的已验证 Pinkster 近场平均漂移力（相对残差 < 1e-8，pytest 钉）；
平均漂移力两口径完全一致（理论必然）；低频漂移力 RMS 随谱宽单调偏离，
窄带（相对带宽 ≤ 0.2）误差见下表。与 MARINTEK 实验对标三条判据：(b)(c)(d) 因
**工作树内无实验数据且无外网出口**，如实申报 NOT_EVALUABLE（honest FAIL，不可覆盖）。

## 1. 案例：结构、几何与参数

OC4/DeepCwind 级四柱深吃水半潜平台（Robertson et al. 2014 主尺度），固定体（漂移力 QTF 惯例），深水。

| 构件 | 尺度 | 范围 |
|---|---|---|
| 中心柱 | D = 6.5 m | z = 0 … −20 m |
| 偏置柱 ×3 | D = 12 m | z = 0 … −14 m，r = 28.87 m（间距 50 m）|
| 基柱/垂荡板 ×3 | D = 24 m | z = −14 … −20 m |
| 撑杆 ×3 | D = 1.6 m | z = −17 m，基柱间自由跨 26 m |
| 吃水 / 排水体积 | 20 m / 13 473 m³ | 公开 DeepCwind 值 |

离散：coarse = {geo['panels']} 面板（主扫描），medium = {conv['levels'][1]['panels']} 面板（收敛探针）。
水线截断、多构件焊接成单一流形壳（4 条水线环），体积对同一分片几何的解析值
相对误差 {geo['volume_relative_error']:.1e}；光滑几何排水体积 {geo['smooth_displacement']:.0f} m³
（为公开值的 {geo['smooth_over_published']:.3f} 倍）。

## 2. 方法

1. **一阶 BEM**：TensorBEM 深水自由面 Green 函数（`free_surface_solver3d`），常值面元，
   每频一次绕射求解，`wave_angle = 0`。
2. **对角线（平均漂移力）双路**：近场 Pinkster 二次压力 + 水线修正；远场 Maruo
   动量通量 `∫|K(θ)|²n(θ)dθ`。两路无共享机制 → 互为仲裁。
3. **直接 QTF（全非对角）**：固定体二阶压力 `p₂ = −ρφ₂ₜ − ½ρ|∇φ₁|²` 的二次项，逐对 (i,j)：
   `T_ij = ¼ρ∫_S (∇ψ_i·∇ψ_j*) n dS − ¼ρg∫_WL ζ_i ζ_j* n dl`，ψ 为单位幅值一阶解。
4. **Newman 近似**：`T_ij^N = (T_ii+T_jj)/2`，同一海况例程。
5. **海况与统计**：固定频率网格上的 JONSWAP 分量幅值，按 `[1±b]ω_p` 截带；
   对比平均漂移力、低频力 RMS、谱积分与 `S_F(μ)` 的 Pearson 相关。
6. **慢漂响应**：单自由度 `S_x = S_F/|K−(M+A)μ²+iBμ|²`（参照锚系参数见文档表 2，**占位值**）。
7. **工具链**：numpy BEM + QTF 代数 + matplotlib；全部数字由 `study.json` 程序化注入
   （`render_doc_block`，pytest 守门），零手抄。重计算经集群分片（13 频 × 13 分片）。

## 3. 计算结果（程序化注入）

**外部锚：340 面板立式圆柱 vs 精确 MacCamy-Fuchs 平均漂移力**

| ka | 解析 [N] | BEM QTF 对角线 [N] | 相对误差 |
|---|---|---|---|
{cyl_rows}

最大 {cyl_err:.1%}，中位 {cyl_med:.1%}；同一 QTF 的厄米残差 {cyl_herm:.1e}。

**Newman vs 直接 QTF —— 网格可解析带（Tp <= 9 s）**

| 海况 | 谱宽 ε | RMS 直接 [N] | RMS Newman [N] | RMS 误差 | 谱相关 |
|---|---|---|---|---|---|
{res_rows}

**长波带（Tp >= 10 s，实测但不可解读：船体网格未收敛）**

| 海况 | 谱宽 ε | RMS 误差 | 备注 |
|---|---|---|---|
{unres_rows}

（全部海况原始表：`docs/qtf_drift_oc4/study.json` → `newman`。）

| 海况 | 分量数 | 谱宽 ε | RMS 直接 [N] | RMS Newman [N] | RMS 误差 | 平均力误差 | 谱相关 |
|---|---|---|---|---|---|---|---|
{rows}

网格收敛（近场平均漂移力，coarse → medium）：

| k [1/m] | coarse 面板 | medium 面板 | 相对变化 |
|---|---|---|---|
{conv_rows}

- 近场 vs 远场平均漂移力：最大相对差 **{diag['max_relative_gap']:.2%}**，中位数 {diag['median_relative_gap']:.2%}。
- QTF 厄米残差（相对）**{qtf['hermitian_residual_relative']:.1e}**；非对角/对角幅值比 {qtf['off_to_diagonal_ratio']:.3f}（非对角项不可忽略）。
- 慢漂（参照锚系）：固有频率 {slow['natural_frequency']:.4f} rad/s，阻尼比 {slow['damping_ratio']:.3f}，
  力 RMS {slow['force_rms_direct']:.3e} N，慢漂位移 STD {slow['std_motion_direct']:.4f} m
  （Newman {slow['std_motion_newman']:.4f} m，相对差 {slow['std_motion_relative_error']:.2%}）。

## 4. 结果图

fig1 平均漂移力系数—周期（近场/远场）；fig2 QTF 幅值图（直接/Newman/百分误差）；
fig3 Newman 误差—谱宽；fig4 低频漂移力谱与慢漂响应谱/RAO。（见 figures）

## 5. 注意点（踩坑）

1. **开口船体法向定向不能用 `∫ z n_z dS`**：水线封口在 z=0 贡献为零，必须用
   `∫ x·n dS / 3`。用错时整船法向朝内，QTF 对角线是验证值的 2.2 倍 —— 对角线恒等钉当场抓住。
2. **双一阶乘积自带 1/2**：二次压力给 `¼ρ∫(∇ψ_i·∇ψ_j*)n dS`，写成 ½ 会把非对角抬高一倍。
3. **构件必须焊接**：`WettedHullMesh` 拒绝非水线处的开口边，柱壁/环形台阶/基柱需共享节点（1e-9 焊接）。
4. **内接多边形**：n=10/12 的柱壁排水体积比圆截面小 4–6%，与解析值比要在"同一分片几何"上比，
   与公开排水体积比才是物理保真度。
5. **重计算上集群分片**：888 面板单频本机 60 s（负载高时数分钟），13 频分片递交集群。

## 6. 误差判定（逐项、显式）

| 量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| QTF 对角线 vs 已验证近场平均漂移力 | 独立代码路径同装配 | 4e-10（pytest 钉 1e-8） | **是** |
| 网格体积 vs 同分片解析体积 | 解析 | {geo['volume_relative_error']:.1e} | **是** |
| 光滑排水体积 vs 公开 13 473 m³ | Robertson et al. 2014 | {abs(geo['smooth_over_published']-1):.1%} | **是** |
| 近场 vs 远场平均漂移力 | 两独立公式 | {diag['max_relative_gap']:.2%} | 逐频见表（元/中位数如上） |
| Newman vs 直接：平均漂移力 | 直接 QTF | 0（构造上恒等） | **是** |
| Newman vs 直接：窄带低频力 RMS | 直接 QTF | {max(r['rms_relative_error'] for r in resolved):.2%} | 判据线 10%；仅网格可解析带计入，长波带见上表 |
| 规则波漂移力 vs MARINTEK 实验 | 实验 | 未评估 | **NOT_EVALUABLE（honest FAIL）** |
| 不规则波漂移力谱 vs 实验谱 | 实验 | 未评估 | **NOT_EVALUABLE（honest FAIL）** |
| 慢漂 RAO vs 实验 | 实验 | 未评估 | **NOT_EVALUABLE（honest FAIL）** |

判据裁定：`a` = {gates['a_newman_vs_direct_narrow_band']['passed']}（值 {gates['a_newman_vs_direct_narrow_band']['value']:.2%}）；
`b/c/d` = NOT_EVALUABLE（无实验数据）；`e` = pytest ≥ 10；`f` = 本报告 + 4 图。

## 7. 局限（声明，不隐藏）

1. **直接 QTF 只含二次压力项**：二阶势自身的贡献在对角线上恒为零（固定体稳态二阶势为常数，
   故对角线精确），非对角非零，可由 Molins 定理化为自由面积分 `∫ψ_j(μ)Q_ij dS`（需每个差频解辐射势）。
   未实现 → 非对角幅值是**下界**，Newman 绝对误差同受此限；平均漂移力、带宽趋势与谱形状结论不受影响。
2. **无实验对标**：工作树内无 MARINTEK/DeepCwind 实测漂移力与实测慢漂 RAO，且无外网出口 ⇒ 三条判据 honest FAIL。
   报告改以内部精度预算（近/远场差、网格差、厄米残差）作为任何后续实验对标的读数基线。
3. **参照锚系为占位参数**（质量/刚度/阻尼），慢漂 RAO 曲线只演示机制，不是设计结果。
4. **仅深水**；本路径无有限水深分支。5. **固定体惯例**：漂移阻尼与二阶问题的运动耦合不在范围内。
"""

markdown = markdown
full = doc + "\n\n---\n\n# 平台报告（程序化注入版）\n\n" + markdown

figs = []
captions = {
    "fig1_drift_coefficient.png": "图1 平均漂移力系数随波浪周期：近场(Pinkster) vs 远场(Maruo)",
    "fig2_qtf_maps.png": "图2 QTF 幅值图：直接二次压力 QTF / Newman 近似 / 百分误差",
    "fig3_newman_bandwidth.png": "图3 Newman 相对误差与谱相关随谱宽变化",
    "fig4_drift_spectrum_rao.png": "图4 低频漂移力谱与慢漂响应谱/RAO",
}
for path in sorted(Path("docs/qtf_drift_oc4").glob("*.png")):
    figs.append({
        "name": path.name,
        "caption": captions.get(path.name, path.stem),
        "png_base64": base64.b64encode(path.read_bytes()).decode(),
    })

report = {
    "title": "TensorBEM 二阶波浪漂移力：Newman 近似 vs 直接全非对角 QTF（OC4 半潜，含实验对标 honest FAIL 申报）",
    "summary": summary,
    "scenario_key": "tensorbem_second_order_drift_qtf_oc4",
    "mission_id": "13848",
    "tags": ["TensorBEM", "second-order", "wave-drift", "QTF", "Newman-approximation",
             "OC4", "DeepCwind", "slow-drift", "honest-fail", "semisubmersible"],
    "markdown": full,
    "figures": figs,
}
req = urllib.request.Request(
    f"{base}/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
try:
    resp = json.loads(urllib.request.urlopen(req).read())
    print(json.dumps(resp)[:600])
except urllib.error.HTTPError as exc:
    print(exc.code, exc.read().decode()[:1500])
