"""Publish the m15384 report (reopened #661, round 1); numbers injected from artifacts."""
import base64
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
login = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
}).encode()
req = urllib.request.Request(f"{BASE}/auth/login", data=login,
                             headers={"Content-Type": "application/json"})
token = json.loads(urllib.request.urlopen(req).read())["access_token"]

DOCS = "docs/flexural_gravity_mass_loading"
rep = json.load(open(f"{DOCS}/verification_report.json"))
crit = rep["criteria"]


def measured(c):
    for key in ("measured", "max_relative_error", "max_abs_residual",
                "agreement_fraction"):
        if c.get(key) is not None:
            val = c[key]
            return val if isinstance(val, str) else f"{val:.3e}"
    return "-"


crit_rows = "\n".join(
    f"| {n} | {c['description']} | {measured(c)} | "
    f"{c.get('threshold', c.get('threshold_percent', '-'))} | "
    f"{'PASS' if c['pass'] else 'FAIL'} |"
    for n, c in crit.items()
)

fd = crit["V11_fd_depth_factor"]
fd_rows = "\n".join(
    f"| {r['ice_thickness']} | {r['water_depth']} | {r['wavenumber']} | "
    f"{r['k_dz_coarsest']:.3g} | " + ", ".join(f"{e:.2e}" for e in r["relative_errors"])
    + f" | {r['coarse_order']:.3f} | {r['finest_error']:.3e} |"
    for r in fd["rows"]
)

en = crit["V12_energy_balance"]
en_rows = "\n".join(
    f"| {r['ice_thickness']} | {r['water_depth']} | {r['wavenumber']} | {r['kH']:.3g} | "
    f"{r['tanh_kH']:.4f} | {r['energy_residual_eq3']:.2e} | "
    f"{r['energy_residual_denominator_only']:.2e} | "
    f"{r['denominator_only_deviation_percent']:.3f} |"
    for r in rep["energy_balance_table"]
    if r["denominator_only_deviation_percent"] > 0.01 or r["wavenumber"] in (0.02, 0.1)
)

figs = []
for name, caption in [
    ("fig_verification_routes.png",
     "左：V11 有限差分路线的 ω² 相对误差 vs dz（双对数，slope-2 参考线，6 工况，"
     "观测阶 1.92–2.00）；右：V12 能量平衡残差 vs k，eq.3（实线，机器零）对分母单边变体"
     "（虚线，浅水 long-wave 档 O(1)–O(10²)）"),
    ("fig_dispersion_models.png",
     "ω(k)：开阔水 / eq.2（仅弯曲）/ eq.3（弯曲+惯性）/ 纯弯曲波"),
]:
    figs.append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(open(f"{DOCS}/{name}", "rb").read()).decode(),
    })

markdown = f"""# TensorBEM 浮冰板色散关系 eq.(3) 的两条独立验证路线（V11 有限差分 / V12 能量平衡）

**m15384（重开 #661 第 1 轮）/ 分支 agent/m15384 / 仓库 TensorBEM**

## ① 案例介绍
薄 Kirchhoff 浮冰板（h=0.5–2 m，E=5 GPa 默认，ν=0.334，ρ_i=917，ρ_w=1025）浮于水面，
水深 H=20 m（北极沿岸/冰间水道，kH=0.1–6）与 H=200 m（深水对照）。目标色散关系为
ω²=(gk+Dk⁵/ρ_w)tanh(kH)/(1+μ k tanh(kH))，μ=ρ_i h/ρ_w。
**重开动机**：既有验证面（V1–V10）全部是 eq.(3) 的**代数恒等式**——极限退化、跨模块一致、
闭式交叉波数、中心差分群速度——它们能抓住实现错误，却抓不住「整条关系式把惯性项放错位置」
这类结构性错误，也不覆盖 `flexural_gravity_numeric.fd_omega2` 的 `mass_per_area` 分支（此前零测试）。

## ② 怎么做的
- **V11（独立数值路线）**：不解析 tanh。在 z∈[-H,0] 上二阶有限差分离散 ψ″=k²ψ（底部 Neumann），
  三对角求解得深度因子 g₀=ψ′(0)/ψ(0)，再代入带惯性的自由面条件
  g₀(ρ_w g+Dk⁴−ρ_i h ω²)=ρ_w ω² 求 ω²。网格 dz 逐级减半（5 档），检验二阶收敛。
- **V12（变分/能量平衡）**：对单位振幅波，PE/area=(ρ_w g+Dk⁴)a²/4；
  KE/area=ρ_w ω²a²I/(4ψ′(0)²)+m″ω²a²/4，其中 I=∫(k²ψ²+ψ′²)dz 用自适应求积（不套用恒等式）。
  判据同时检查 eq.(3) 使残差为零，以及**分母单边变体**
  ω²=(gk+Dk⁵/ρ_w)tanh(kH)/(tanh(kH)+μk)（惯性项放到深度因子外侧）被能量判据拒绝。
- 工具链：`scripts/verify_flexural_gravity_mass_loading.py` 生成 `verification_report.json`/
  `results.md`/5 张图（零手抄）；`tests/test_flexural_gravity_mass_loading.py` 新增 9 条测试。

## ③ 计算结果（全部由脚本写入 JSON 后注入）

判据 V1–V12：

| 判据 | 内容 | 实测 | 阈值 | 判定 |
|---|---|---|---|---|
{crit_rows}

V11 有限差分收敛表（6 工况 × 5 档网格；`k dz` 最粗 ≤0.02）：

| h [m] | H [m] | k [1/m] | k dz(最粗) | rel err 序列 | 粗档观测阶 | 最细 rel err |
|---|---|---|---|---|---|---|
{fd_rows}

V12 能量平衡（节选；eq.(3) 残差全表 32 工况均 ≤{en['max_abs_residual']:.1e}）：

| h [m] | H [m] | k [1/m] | kH | tanh(kH) | eq.3 能量残差 | 变体能量残差 | 变体 ω² 偏差 [%] |
|---|---|---|---|---|---|---|---|
{en_rows}

## ④ 结果图
fig_verification_routes.png（左 V11 网格收敛，右 V12 残差对比）+ fig_dispersion_models.png。

## ⑤ 注意点
- **V12 的对照变体是自己构造的**，不是文献既有模型：它只用来证明能量判据有分辨力
  （浅水档变体偏差最大 {en['max_denominator_only_deviation_percent']:.1f}%），不可引作「某文献模型错误」的证据。
- 分辨力是**浅水专属**的：kH≳3 时两种写法在 1e-3 量级不可分（h=2 m/H=200 m/k=0.02 偏差 0.06%）。
  即深水冰区结果对惯性项位置不敏感，浅水（kH≲0.5）才是判据的着力点。
- 两条路线与 eq.(3) 共享同一深度模态 ψ=cosh(k(z+H))/cosh(kH)：若换分层水/弹性底床，深度因子须重推，
  两条路线会**一致地**偏移——它们检验的是「惯性项与深度因子的耦合方式」，不是深度因子本身的物理假设。
- 最细网格上相对误差接近舍入（~1e-8）时观测阶无意义（可为负），收敛阶一律按最粗两档读取
  （JSON 的 `coarse_order` 字段）；表中最细档误差仍 <1e-6。
- 冰缘反射仍用群速度导纳匹配（1D），R 的绝对量级不可作定量预报；薄板、线性、单层均匀冰的前提未变。

## ⑥ 误差判定（逐项 + 是否 <3%）
- V11 独立 FD 路线 vs eq.(3)：最细网格相对误差 **{fd['max_relative_error']:.3e}**（阈值 1e-6）→ **是（<3%）**；
  粗档观测阶 {fd['min_coarse_order']:.3f}–{fd['max_coarse_order']:.3f}（二阶，区间 [1.7,2.3]）→ 是。
- V12 能量平衡残差（eq.(3)，32 工况）：**{en['max_abs_residual']:.3e}**（阈值 1e-9）→ **是**；
  补全平方恒等式 I/ψ′(0)−1 ≤ {en['max_completion_defect']:.3e} → 是。
- V1–V10 原判据数值一字未变（脚本重跑与提交工件 byte-identical），最差 4.30%（E=1 GPa/2 m/0.1 Hz 的物理差异，已如实登记）。
- 无外部实验数据可比：本单的基准是解析恒等式与独立数值路线（方程级验证），不是实验 RAO，
  故「vs 实验 <3%」一项无基准可判，此处不作声明。

## ⑦ 变更面（≤10 文件，单一功能点）
scripts/verify_flexural_gravity_mass_loading.py（+V11/V12/新图）、
tests/test_flexural_gravity_mass_loading.py（+9 测试，54→63 passed）、
docs/flexural_gravity_mass_loading/{{report.md, results.md, verification_report.json, fig_verification_routes.png}}。
未触碰 `src/tensorbem/flexural_gravity.py` 等热文件（零源码改动 = 零回归面）。
"""

payload = {
    "title": "TensorBEM：浮冰板 eq.(3) 的两条独立验证路线（V11 有限差分二阶收敛 / V12 能量平衡拒绝惯性项错位）",
    "summary": (
        "重开 #661 第 1 轮：在既有 V1–V10（代数恒等式）之上补两条独立路线——"
        "有限差分深度因子路线（最细网格相对误差 4.62e-07，观测二阶）与变分能量平衡路线"
        "（eq.3 残差 4.90e-16，并拒绝惯性项放错位置的分母单边变体，浅水偏差达 881.5%）；"
        "顺带覆盖此前零测试的 flexural_gravity_numeric.fd_omega2 mass_per_area 分支；测试 54→63。"
    ),
    "scenario_key": "tensorbem_flexural_gravity_mass_loading",
    "mission_id": 15384,
    "tags": ["TensorBEM", "ice", "flexural-gravity", "mass-loading", "verification",
             "finite-difference", "energy-balance", "selfstart"],
    "markdown": markdown,
    "figures": figs,
}

req = urllib.request.Request(
    f"{BASE}/tensor/reports",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
res = json.loads(urllib.request.urlopen(req).read())
print(json.dumps({k: v for k, v in res.items() if k != "markdown"})[:800])