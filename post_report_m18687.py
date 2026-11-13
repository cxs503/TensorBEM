"""Publish the m18687 report (reopened #737, round 1); numbers injected from artifacts."""
import base64
import json
import os
import urllib.error
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
v13 = crit["V13_bem_depth_factor"]
v11 = crit["V11_fd_depth_factor"]
v12 = crit["V12_energy_balance"]
sweep = rep["bem_omega_sweep"]


def measured(c):
    for key in ("measured", "max_relative_error", "max_abs_residual", "agreement_fraction"):
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

v13_rows = "\n".join(
    f"| {r['ice_thickness']} | {r['water_depth']} | {r['wavenumber']} | {r['kH']:.4g} | "
    f"{r['depth_factor_exact']:.6f} | {r['depth_factor_bem'][-1]:.6f} | "
    + ", ".join(f"{e:.2e}" for e in r["omega2_relative_errors"])
    + f" | {r['order_first_last']:.2f} | {r['finest_error']:.3e} | "
    f"{r['cross_route_rel_diff']:.1e} |"
    for r in v13["rows"]
)

sweep_header = "| k [1/m] | " + " | ".join(f"{r['wavenumber']:g}" for r in sweep) + " |"
sweep_row = lambda key, fmt: "| " + key + " | " + " | ".join(fmt(r) for r in sweep) + " |"
sweep_table = "\n".join([
    sweep_header,
    "|---|" + "---|" * len(sweep),
    sweep_row("ω_eq3 [rad/s]", lambda r: f"{r['omega_eq3']:.6f}"),
    sweep_row("ω_BEM [rad/s]", lambda r: f"{r['omega_bem']:.6f}"),
    sweep_row("rel", lambda r: f"{r['relative_error']:.2e}"),
])

figs = []
for name, caption in [
    ("fig_verification_routes.png",
     "2×2：左上 V11 有限差分路线 ω² 相对误差 vs dz（slope-2 参考线，6 工况）；"
     "右上 V12 能量平衡残差 vs k（eq.3 实线=机器零 / 分母单边变体虚线，浅水 O(1)–O(10²)）；"
     "左下 V13 边界积分路线 ω² 相对误差 vs 每波长面板数（8/16/32/64 ppw，误差单调下降，灰点线=2e-3 判据）；"
     "右下 V13 的 ω(k) 逐点对比：eq.3 解析曲线（实线）对条带 BEM Λ 代入 Λ 表述的点（h=2 m, H=20 m，9 波数，最大相对差 2.24e-03）"),
    ("fig_dispersion_models.png",
     "ω(k)：开阔水 / eq.2（仅弯曲刚度）/ eq.3（弯曲+冰质量惯性）/ 纯弯曲波；右图 ω₃/ω₂−1 [%] 并标出交叉波数 k_c"),
]:
    figs.append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(open(f"{DOCS}/{name}", "rb").read()).decode(),
    })

markdown = f"""# TensorBEM 浮冰板 eq.(3) 的第三条独立验证路线：周期条带边界积分（V13）

**m18687（重开 #737 第 1 轮）/ 分支 agent/m18687 / 仓库 TensorBEM**

## ① 案例介绍
薄 Kirchhoff 浮冰板浮于水面，冰面条件 ρ_w(φ_t+gη)+ρ_i hη_tt+Dη_xxxx=0、φ_z=η_t。
目标色散关系（eq. 3）：

ω² = (g k + D k⁵/ρ_w)·tanh(kH) / (1 + μ k tanh(kH)),  D = E h³/[12(1−ν²)],  μ = ρ_i h/ρ_w。

| 参数 | 取值 |
|---|---|
| 冰厚 h | 0.5 / 1.0 / 2.0 m（V13 六个工况与 V11 同组） |
| 杨氏模量 E | 5 GPa（默认；敏感性见主报告 results.md） |
| 泊松比 ν | 0.334 |
| ρ_w / ρ_i | 1025 / 917 kg/m³ |
| 水深 H | 20 m（kH=0.1–12，北极沿岸/冰间水道）与 200 m（深水对照） |
| 波数 k | 0.005–0.8 1/m（ω 0.07–702 rad/s，跨 4 个数量级） |

**为什么还要第三条路线**：既有 V11（有限差分深度因子）与 V12（变分能量平衡）都把 tanh(kH)
当成要被检验的解析对象，二者与 eq. 3 共享同一族深度模态 ψ=cosh(k(z+H))/cosh(kH)。
V13 换掉的是**求解手段**：用边界积分方程直接解 Laplace 方程拿深度因子，完全不做 tanh、也不做网格差分。

**重开说明（honest）**：PR#737 所述 m15384 交付（report/results/verification_report + 脚本 + 测试）
已随 #668 进入 main，并由 m15430 扩到 E∈[1,30] GPa。故本轮不重复落旧文件，而在其上做增量。

## ② 怎么做的
1. **代数关键步**：tanh(kH) 只以组合 Λ≡(∂φ/∂z)/φ = k tanh(kH) 出现在 eq. 3 中。代入 tanh=Λ/k 得

   ω² = (g + D k⁴/ρ_w)·Λ / (1 + μ Λ)，

   于是「深度因子从哪来」与「冰惯性/弯曲项怎么进」解耦——BEM 只需给出 Λ。
2. **Λ 的来源**：`tensorbem.ice_mass_loading_dispersion` 的周期条带混合边界积分方程
   （自由面给定 cos(kx) 迹、底/侧壁自由、周期联结），常单元 + 12 阶解析自项积分，LAPACK 直接解。
   取两条独立路线：驱动模态的 L2 瑞利商 `strip_bem_depth_factor().depth_factor_rq` 与
   组装后的 Dirichlet-to-Neumann 映射 `dtn_depth_factor()`。
3. **判据 V13**：(a) Λ 取精确值 k tanh(kH) 时 Λ 表述与 eq. 3 必须是恒等式（≤1e-12）；
   (b) 64 面板/波长下 |ω²_BEM/ω²_eq3 − 1| < 2e-3（6 工况）；(c) 误差随 ppw=8→16→32→64 单调下降，
   整体斜率 ∈[1.0,2.2]；(d) 两条 BEM 路线相对差 <1e-12；(e) ω(k) 全曲线（9 波数）相对差 <3e-3。
4. **工具链**：`scripts/verify_flexural_gravity_mass_loading.py` 新增 V13 判据 + `bem_route_table` +
   `bem_omega_sweep` + 第四幅子图，全部数字写入 `verification_report.json` / `results.md`（零手抄）；
   `tests/test_flexural_gravity_mass_loading.py` 新增 11 条测试（65→76 passed）。
   **零源码改动**：`src/tensorbem/*` 一行未动（只复用既有已验证模块），故零回归面。

## ③ 计算结果（程序化注入，取自 verification_report.json）

判据 V1–V13：

| 判据 | 内容 | 实测 | 阈值 | 判定 |
|---|---|---|---|---|
{crit_rows}

V13 面板收敛 + 双路线互校（六工况，Λ 精确值 vs BEM 值）：

| h [m] | H [m] | k [1/m] | kH | Λ 精确 | Λ_BEM(64 ppw) | ω² rel err (8/16/32/64 ppw) | 整体斜率 | 最细 rel err | driven−DtN 差 |
|---|---|---|---|---|---|---|---|---|---|
{v13_rows}

V13 的 ω(k) 逐点比对（h=2 m, H=20 m, 64 ppw）：

{sweep_table}

## ④ 结果图
`fig_verification_routes.png`（2×2：V11 网格收敛 / V12 能量残差判别 / V13 面板收敛 / V13 的 ω(k) 逐点对比）
与 `fig_dispersion_models.png`（ω(k) 四模型 + ω₃/ω₂−1 相对差曲线，标 k_c）。均为 matplotlib 程序生成、
提交在 docs/ 内，可 rerun 复现。

## ⑤ 注意点（踩过的坑 / 局限）
- **Λ 必须是 (∂φ/∂z)/φ 而不是 φ 或 k**：首版把 `depth_factor_rq` 误当 tanh(kH) 直接代分母，
  误差立即 ~99%（0.0097 vs 0.964）；正确做法是 tanh=Λ/k 代回，这条换算已在测试里显式钉住
  （`test_v13_bem_depth_factor_reproduces_eq3` 除了比 ω² 还比 Λ 本身，防「倒数/量纲混用」）。
- **误差下限 ~1e-3 是常单元通量近似的离散误差，不是模型误差**：32 ppw 最差 5.19e-3，64 ppw 最差
  1.43e-3，继续加密只到 ~4e-4 量级且成本按 ppw³ 增长（稠密解）。因此 V13 只能把 eq. 3 钉到 1e-3 档，
  **达不到 V11（4.6e-7）/V12（4.9e-16）的机器精度**，如实登记，不宣称等价强度。
- **独立性边界**：V13 独立于 tanh 与差分，但不独立于「水深方向单一模态」这一物理前提——BEM 求的
  正是该模态的 Λ；若换成密度分层或弹性底床，三条路线会一并偏移。
- 浅水长波端误差最大（k=0.005, kH=0.1 时 2.24e-3）：此处 ω 对 Λ 最敏感（弯曲项弱、重力项主导），
  也是真实冰区最关心的一档；提升需高阶面板或奇性提取，本轮未做。
- V11/V12 的既有坑（最细网格舍入导致观测阶无意义、V12 对照变体是自造物、冰缘仍是 1D 导纳匹配）
  在 `report.md` §5 已登记，本轮未改变。

## ⑥ 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | 基准 | 实测相对误差 | < 3% ? |
|---|---|---|---|
| V13 ω²（64 ppw，6 工况最差） | eq. 3 解析 | {v13['max_relative_error']:.2e} | **是** |
| V13 ω²（32 ppw，6 工况最差） | eq. 3 解析 | {max(r['omega2_relative_errors'][2] for r in v13['rows']):.2e} | **是** |
| V13 Λ 表述恒等式（Λ 取精确值） | eq. 3 解析 | {v13['max_identity_error']:.2e} | **是** |
| V13 driven 瑞利商 vs 组装 DtN | 同一 Λ | {v13['max_cross_route_rel_diff']:.2e} | **是** |
| V13 ω(k) 全曲线（9 波数，最差 k=0.005） | eq. 3 解析 | {v13['max_sweep_relative_error']:.2e} | **是** |
| V11 独立 FD 路线 ω²（最细网格） | eq. 3 解析 | {v11['max_relative_error']:.2e} | **是** |
| V12 能量平衡残差（32 工况） | 能量守恒 | {v12['max_abs_residual']:.2e} | **是** |
| V13 发散斜率（8→64 ppw） | 理论 1–2 阶 | {v13['min_order_first_last']:.2f}–{v13['max_order_first_last']:.2f} | 是（在区间内） |

**结论**：V13 的全部关键量相对解析 eq. 3 均远小于 3%（最差 2.24e-3），且误差可解释为常单元离散误差、
随加密单调下降。**无外部实验数据可比**（本单是方程级验证：解析恒等式 + 独立数值路线），
故「vs 实验/文献 <3%」一项无基准可判，此处不作粉饰性声明。

## ⑦ 变更面（≤10 文件，单一功能点）
`scripts/verify_flexural_gravity_mass_loading.py`（+V13）、`tests/test_flexural_gravity_mass_loading.py`
（+11 测试，65→76 passed）、`docs/flexural_gravity_mass_loading/{{report.md, results.md,
verification_report.json, fig_verification_routes.png}}`、`post_report_m18687.py`。
**未触碰 src/tensorbem 任何模块**——与其它开 PR 无热文件重叠。
"""

payload = {
    "title": "TensorBEM：浮冰板 eq.(3) 的第三条独立验证——周期条带边界积分（V13，Λ=(∂φ/∂z)/φ）",
    "summary": (
        "重开 #737 第 1 轮：在 V11 有限差分 / V12 能量平衡之外补第三条独立路线 V13——"
        "用周期条带混合边界积分方程直接求 Λ=(∂φ/∂z)/φ 并代入 eq.3 的 Λ 表述"
        "ω²=(g+Dk⁴/ρ_w)Λ/(1+μΛ)；64 面板/波长下 6 工况最差 1.43e-03，误差随加密单调下降"
        "（斜率 1.35–1.55），driven 瑞利商与组装 DtN 两条 BEM 路线差 8.9e-16，ω(k) 9 波数全曲线最差 2.24e-03；"
        "零源码改动，测试 65→76 passed。"
    ),
    "scenario_key": "tensorbem_flexural_gravity_mass_loading",
    "mission_id": 18687,
    "tags": ["TensorBEM", "ice", "flexural-gravity", "mass-loading", "verification",
             "boundary-element", "strip-BEM", "selfstart"],
    "markdown": markdown,
    "figures": figs,
}

req = urllib.request.Request(
    f"{BASE}/tensor/reports",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
try:
    res = json.loads(urllib.request.urlopen(req).read())
    print(json.dumps(res)[:400])
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code)
    print(exc.read().decode()[:1500])
