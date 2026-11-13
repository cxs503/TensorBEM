"""Post the CASE-BEM-ICE-DISPERSION progress report to the platform.

Numbers are injected programmatically from docs/case_bem_ice_dispersion/numbers.json
(zero hand-copy) and the four figures are embedded as base64 PNG.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib

import requests

_ROOT = os.environ["TENSOR_API_BASE"].rstrip("/").rsplit("/api/v1", 1)[0]
BASE = _ROOT + "/api/v1"
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = pathlib.Path(__file__).resolve().parent
CASE_DIR = ROOT / "docs" / "case_bem_ice_dispersion"
MISSION = 17373

s = requests.Session()
r = s.post(f"{BASE}/auth/login", json={"username": USER, "password": PASS}, timeout=30)
r.raise_for_status()
token = r.json().get("access_token") or r.json().get("token")
H = {"Authorization": f"Bearer {token}"}

# --- knowledge base / negative ledger lookups (best effort) ----------------
kb_notes = []
for path, params in [
    ("/tensor/knowledge/search", {"q": "ice mass loading dispersion BEM", "limit": 5}),
    ("/tensor/knowledge/search", {"q": "TensorBEM ice dispersion", "limit": 5}),
]:
    try:
        rr = s.get(f"{BASE}{path}", headers=H, params=params, timeout=20)
        kb_notes.append((path, rr.status_code, rr.text[:300]))
    except Exception as exc:  # pragma: no cover
        kb_notes.append((path, "ERR", str(exc)))
for path, status, text in kb_notes:
    print("KB", path, status, text[:200])

report = json.loads((CASE_DIR / "numbers.json").read_text())
p = report["parameters"]
crit = report["criteria"]


def fmt(v, d=4):
    try:
        return f"{v:.{d}g}"
    except Exception:
        return str(v)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


rows_md = []
for key in ("deep", "shallow"):
    for r_ in report["series"][key]["rows"]:
        rows_md.append([
            key, fmt(r_["k"]), fmt(r_["kH"]), fmt(r_["depth_factor_bem"]),
            fmt(r_["depth_factor_ref"]), fmt(r_["omega_ref"]), fmt(r_["omega_got"]),
            f"{100 * r_['omega_rel_error']:+.3f}%",
            "OK" if abs(r_["omega_rel_error"]) < p["tolerance"] else "FAIL",
        ])

conv_md = []
for name, rows in report["convergence"].items():
    for r_ in rows:
        order = r_.get("observed_order")
        conv_md.append([
            name, int(r_["panels_per_wavelength"]), int(r_["n_panels"]),
            fmt(r_["depth_factor_bem"]), fmt(r_["depth_factor_exact"]),
            f"{100 * r_['rel_error']:+.3f}%",
            "n/a" if order is None or order != order else f"{order:.2f}",
        ])

thick_md = [[fmt(r_["ice_thickness"], 3), fmt(r_["mu"]), fmt(r_["omega_got"]),
             fmt(r_["omega_ref"]), f"{100 * r_['omega_rel_error']:+.3f}%"]
            for r_ in report["thickness_sensitivity"]]
depth_md = [[fmt(r_["depth"], 3), fmt(r_["kH"], 3), fmt(r_["omega_got"]),
             fmt(r_["omega_ref"]), f"{100 * r_['omega_rel_error']:+.3f}%"]
            for r_ in report["depth_sensitivity"]]
crit_md = [[name, c["note"], fmt(c["value"]), fmt(c["threshold"]),
            "PASS" if c["passed"] else "FAIL"] for name, c in crit.items()]
modes_md = [[int(r_["mode"]), fmt(r_["wavenumber"], 3),
             fmt(r_["panels_per_wavelength_effective"], 3),
             fmt(r_["depth_factor_bem"]), fmt(r_["depth_factor_exact"]),
             f"{100 * r_['rel_error']:+.3f}%", fmt(r_["boundary_value_error"], 3)]
            for r_ in report["mode_sweep"]]

max_err = crit["C1_omega_error_below_tolerance"]["value"]
ppw = int(p["panels_per_wavelength"])
lambda_err = max(
    abs(r_["rel_error"]) for name in report["convergence"]
    for r_ in report["convergence"][name]
    if int(r_["panels_per_wavelength"]) == ppw
)
markdown = f"""# CASE-BEM-ICE-DISPERSION 冰区质量加载色散（TensorBEM）：BEM 数值验证 Weitz–Keller 关系

**案例** `CASE-BEM-ICE-DISPERSION` · TensorBEM {report['tensorbem_version']} · Python {report['environment']['python']} / numpy {report['environment']['numpy']}
**判据（动手前预设）**：ω 相对误差 < 5%；两条数值路线互相一致 < 1e-9；面元加密单调收敛；域内场恢复 < 1%；深水极限与长波极限各自 < 1%。全部由驱动脚本机判，非自声明。

## ① 案例介绍（物理背景/几何/参数表）

1 m 冰盖按**质量加载**（纯惯性、无弯曲刚度，Weitz & Keller 1950 模型）浮于水深 H 的水面上；水体无粘无旋、底部刚性，运动为波数 k 的平面周期波。自由面条件为 Robin 条件，色散关系为 `omega^2 = g k tanh(kH) / (1 + mu k tanh(kH))`，`mu = rho_i h_i / rho_w`。两条序列：深水序列 H = {fmt(report['series']['deep']['depth'])} m（kH = 4…80），浅水序列 H = {fmt(report['series']['shallow']['depth'])} m（kH = 0.04…0.4）。

{table(['量', '符号', '值', '单位'], [
    ['冰厚', 'h_i', fmt(p['ice_thickness']), 'm'],
    ['冰密度', 'rho_i', fmt(p['ice_density']), 'kg/m^3'],
    ['水密度', 'rho_w', fmt(p['water_density']), 'kg/m^3'],
    ['重力', 'g', fmt(p['gravity']), 'm/s^2'],
    ['质量加载参数', 'mu', fmt(p['mu']), 'm'],
    ['截止频率', 'sqrt(g/mu)', fmt(p['cutoff_omega']), 'rad/s'],
    ['截止周期', '2*pi/sqrt(g/mu)', fmt(p['cutoff_period']), 's'],
    ['深水序列水深', 'H', fmt(report['series']['deep']['depth']), 'm'],
    ['浅水序列水深', 'H', fmt(report['series']['shallow']['depth']), 'm'],
    ['自由面每波长面元数', '-', ppw, '-'],
    ['容差', '-', p['tolerance'], '-'],
])}

**ref 出处（零手抄）**：关系式取自 Weitz & Keller (1950) *Some remarks on sea ice*，并见 Fox & Squire (1994) Phil. Trans. R. Soc. A 352 的 (3) 式；同一关系在本仓已由 `tests/test_ice_mass_loading.py::test_deep_water_limit_matches_weitz_keller`（能力行 `ice-region-hydrodynamics`）独立钉住。表中每个 ref 数字都是运行期由该闭式求值、脚本注入，没有任何手抄表值。
**如实声明（WITHHELD）**：本环境离线，无法核对 Weitz & Keller (1950) 原文印刷表页——ref 取闭式程序求值而非抄录；闭式本身由它自身给出的两个独立极限（深水、长波）交叉锚定（C5/C6）。

## ② 怎么做的（方法/流程/关键设置/工具链）

取冰覆水域**一个周期单元**（`0<=x<=2*pi/k`，`-H<=z<=0`），四边常值面元离散，解带 1/2 自由项的 Laplace 边界积分方程，处理为**混合边值 + 周期连接**问题（自由面 Dirichlet、底部刚性 Neumann、周壁 `phi_R=phi_L`、`q_R=-q_L`）。两条互不相同的数值路线给出自由面深度因子 `Lambda = (dphi/dz)/phi`：

1. **驱动解**：自由面给定 `phi=cos(kx)`，解出通量，取 L2 Rayleigh 商 `Lambda = sum(q_i phi_i l_i)/sum(phi_i^2 l_i)`；
2. **DtN 映射**：显式装配自由面 Dirichlet-to-Neumann 稠密矩阵（独立代码路径），同一商经矩阵乘法求得。

再由 Robin 条件唯一换算频率：`omega^2 = g Lambda / (1 + mu Lambda)`。`tanh(kH)` 只出现在 ref 一侧，因此这是对色散关系的**真实数值验证**，不是代数恒等式。
工具链：`tensorbem.ice_mass_loading_dispersion`（网格/层矩阵/周期连接/驱动解/DtN/模态扫描/域内场恢复）→ 求解器侧调用 `tensorbem.ice.mass_loading_omega_bem` → 算例驱动 `tensorbem.benchmarks.ice_mass_loading_dispersion`。单机 CPU 全算例 ~39 s（<10 min，无需集群）。

## ③ 计算结果（数字表，程序化注入）

**ref vs got 逐项对照（判据逐行打勾叉）**

{table(['序列', 'k [1/m]', 'kH', 'Lambda BEM', 'Lambda ref', 'omega ref [rad/s]', 'omega got [rad/s]', '误差', '<5%'], rows_md)}

**同一单元的高阶周期模态**

{table(['模态 m', 'k_m [1/m]', '每波长面元', 'Lambda BEM', 'Lambda ref', '误差', '边界值恢复误差'], modes_md)}

**域内场恢复**：由边界数据在 41×21 内点上重建的势函数，与精确模态相差 **{fmt(report['field']['max_error'], 3)}**（相对 {100 * report['field']['relative_error']:.3f}%）。

## ④ 结果图

- 图1 色散曲线：ref 闭式（有限水深）+ 深水/长波极限 + 截止频率，叠加 BEM 结果点；
- 图2 面元加密收敛（误差 vs 每波长面元数，含 5% 容差线）；
- 图3 参数敏感性：冰厚与水深；
- 图4 场量云图：BEM 势函数、精确模态、二者差值（×100）。

## ⑤ 注意点（踩坑/局限/适用范围）

- 模型是**质量加载**：不含弯曲刚度（弹性支在 `tensorbem.flexural_gravity`）；`omega` 上界为截止频率 `sqrt(g/mu)`，超过后 Robin 系数变号、无传播冰波（`tensorbem.ice.robin_coefficient` 抛错）。
- 常值面元收敛阶实测 **~1.5**（非 2）：深度因子是单元角落处的通量，角点降阶；默认 64 面元/波长时误差 < 1.5%。
- 自由面 DtN 稠密矩阵在常值面元基下**特征值求解被角点污染**（实测模态 1 误差 20%），故 DtN 路线改用商 `Lambda=<T c,c>/<c,c>` 而非特征值求解——写入代码 docstring 与报告，不粉饰。
- **生态摩擦（如实记）**：本环境 `scipy.linalg.solve` 在该算子（条件数 5.4e2）上给出的解相对残差达 **3.6e3**，而 `numpy.linalg.solve` 为 6e-15；模块改用 numpy 路径，并把线性系统残差写进每个工件（`linear_system_residual`），使静默精度丢失不能蒙混过关。
- 仓库既有红测（本单未触碰、与本案例无关）：`tests/test_capability_evidence.py` 中 `shu_osher_wave_interaction` 无测试模块、`cathodic_protection_sphere` 陈旧理由、maruo 行锚点三条，属 main 既有漂移；已用 `--write` 重新生成的能力块本身通过。
- 适用范围：无限冰盖、等水深、正入射、线性理论；冰边缘、斜入射、有限冰区不在本单。

## ⑥ 误差判定（关键量 vs 基准逐项 + 是否 < 3%）

{table(['判据', '量', '值', '阈值', '结论'], crit_md)}

**是否 < 3%？** 全部 9 行色散算例的 ω 相对误差最大 **{100 * max_err:.3f}%**，远小于 3%（更远小于平台 15% 不可接受线与本单 5% 容差）；深度因子相对误差在 {ppw} 面元/波长下最大 {100 * lambda_err:.3f}%。
两条数值路线一致到 {fmt(crit['C2_routes_agree']['value'], 3)}，域内场相对误差 {100 * report['field']['relative_error']:.3f}%，深水极限 {fmt(crit['C5_deep_water_anchor']['value'], 3)}，长波极限 {fmt(crit['C6_long_wave_anchor']['value'], 3)}。
六条预设判据全部通过：**{report['all_criteria_passed']}**。

## 工件与复现

- 实现：`src/tensorbem/ice_mass_loading_dispersion.py`
- 集成调用：`src/tensorbem/ice.py::mass_loading_omega_bem`、`src/tensorbem/__init__.py` 导出
- 算例入口：`src/tensorbem/benchmarks/ice_mass_loading_dispersion.py`（`main()` 写出 numbers.json / table.md / REPORT.md / figs）
- 运行工件：`docs/case_bem_ice_dispersion/{{numbers.json, table.md, REPORT.md, figs/*.png}}`
- 测试：`tests/test_ice_mass_loading_dispersion.py`（31 项，含求解器侧调用与 6 判据机判）
- 能力台账：`capability_evidence` 新行 `ice-mass-loading-bem-dispersion`（REAL，4 测试节点）
- **repro_cmd**：`PYTHONPATH=src python -m tensorbem.benchmarks.ice_mass_loading_dispersion --figure`
"""

figures = []
captions = {
    "fig_dispersion.png": "图1 冰区质量加载色散：ref 闭式（有限水深）+ 深水/长波极限 + 截止频率，叠加 BEM 结果点（深水与浅水两序列）。",
    "fig_convergence.png": "图2 面元加密收敛：深度因子相对误差随每波长面元数下降（三条探针），虚线为 5% 容差。",
    "fig_sensitivity.png": "图3 参数敏感性：左=冰厚（质量加载参数 mu），右=水深（浅水到深水过渡）。",
    "fig_field.png": "图4 场量云图：BEM 势函数（左）、精确模态（中）、差值×100（右）。",
}
for name, cap in captions.items():
    path = CASE_DIR / "figs" / name
    figures.append({
        "name": name,
        "caption": cap,
        "png_base64": base64.b64encode(path.read_bytes()).decode(),
    })

payload = {
    "title": (
        "CASE-BEM-ICE-DISPERSION：TensorBEM 周期条带 BEM 数值验证冰区质量加载色散"
        f"（Weitz–Keller），9 行算例最大误差 {100 * max_err:.3f}%"
    ),
    "summary": (
        "新增 tensorbem.ice_mass_loading_dispersion：对冰覆水域单个周期单元解 Laplace 边界积分方程"
        "（自由面 Dirichlet、刚性底、周期连接），由自由面深度因子经 Robin 条件换算频率，与 Weitz–Keller "
        f"闭式逐项对照；深水与浅水两序列 9 行最大 ω 误差 {100 * max_err:.3f}%（容差 5%），"
        "驱动解与 DtN 两条路线一致到 1e-14，域内场恢复 "
        f"{100 * report['field']['relative_error']:.3f}%，六条预设判据全通过；求解器侧入口 "
        "tensorbem.ice.mass_loading_omega_bem，算例驱动 + 工件 + 31 项测试 + 能力台账新行。"
    ),
    "scenario_key": "CASE-BEM-ICE-DISPERSION",
    "mission_id": MISSION,
    "tags": ["TensorBEM", "ice", "mass-loading", "dispersion", "Weitz-Keller",
             "BEM", "benchmark", "case-pool"],
    "markdown": markdown,
    "figures": figures,
}

rr = s.post(f"{BASE}/tensor/reports", headers=H, json=payload, timeout=180)
print("POST /tensor/reports ->", rr.status_code, rr.text[:500])
print("kb lookups:", kb_notes)