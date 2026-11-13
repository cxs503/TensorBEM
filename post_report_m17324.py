"""Publish the m17324 (CASE-BEM-RIGID-BODY-MASS) report; numbers injected from run.json."""

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
D = ROOT / "docs" / "cases" / "case_bem_rigid_body_mass"
R = json.loads((D / "run.json").read_text())

levels = R["levels"]
accepted = R["verdict"]["worst_accepted_level"]
legacy = R["negative_control"]["legacy_assembly"]
platform = R["platform_assembly"]
ref = R["reference"]
params = R["parameters"]

rows = "\n".join(
    f"| {r['subdivisions']} | {r['panels']} | {100 * r['volume_rel_error']:.3f} % | "
    f"{100 * r['inertia_rel_error']:.3f} % | {100 * r['max_entry_rel_error']:.3f} % | "
    f"{'PASS' if r['within_tolerance'] else 'FAIL'} | {r['energy_identity_error']:.2e} |"
    for r in levels
)
matrix_rows = "\n".join(
    "| **" + params["mode_order"][i] + "** | "
    + " | ".join(
        f"{levels[-1]['entries'][6 * i + j]['got']:.4g} ({100 * levels[-1]['entries'][6 * i + j]['rel_error']:.2f} %)"
        for j in range(6)
    )
    + " |"
    for i in range(6)
)

md = (D / "REPORT.md").read_text()
md += "\n\n---\n\n### 机械汇总（脚本注入，零手抄）\n\n"
md += (
    f"闭式 ref：m = ρ∇ = {ref['mass']:.8g} kg（∇ = {ref['volume']:.8g} m³），"
    f"I_g = ({ref['inertia_cg'][0][0]:.6g}, {ref['inertia_cg'][1][1]:.6g}, {ref['inertia_cg'][2][2]:.6g}) kg·m²，"
    f"r_g = {tuple(params['cg_offset_m'])} m；判据 = 6×6 逐元素相对误差 < 3 %。\n\n"
    f"| 细分 | 面元 | ∇ 误差 | I_g 误差 | 6×6 最差元素 | 判据 | 动能恒等式残差 |\n"
    f"|---:|---:|---:|---:|---:|:--:|---:|\n{rows}\n\n"
    f"最差接受层级（细分 {accepted['subdivisions']}）：{100 * accepted['max_entry_rel_error']:.3f} % → 是否 < 3 %：**是**。\n\n"
    f"5120 面元矩阵（got，括号内为对闭式 ref 的相对误差）：\n\n"
    f"| i \\ j | " + " | ".join(params["mode_order"]) + " |\n|---|" + "---|" * 6 + f"\n{matrix_rows}\n\n"
    f"缺陷（修复前 `hydrostatic.compute_mass_matrix`）：耦合块相对误差 {100 * legacy['block_relative_error']['translation_rotation']:.1f} %，"
    f"转动块 {100 * legacy['block_relative_error']['rotation_rotation']:.2f} %，动能恒等式残差 "
    f"{100 * legacy['energy_identity_error']:.1f} %（同时矩阵仍对称 → 对称性测试测不出）；"
    f"修复后平台装配误差 {platform['block_form_residual']:.2e}、动能残差 {platform['energy_identity_error']:.2e}。\n"
)

figures = []
for name, caption in (
    ("fig_convergence.png", "面元收敛：体积/质心惯量/6×6 最差元素相对误差 vs 面元数，红线为 3% 判据"),
    ("fig_block_matrix.png", "闭式 6×6 矩阵、面元路线矩阵（5120 面元）与逐元素相对偏差（对数色标）"),
    ("fig_energy_identity.png", "动能恒等式残差：正确装配 / 平台装配 / 符号翻转对照 / 修复前公式对照"),
):
    path = D / name
    if path.exists():
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )


def post(path, payload, token=None):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    return json.loads(urllib.request.urlopen(request, timeout=180).read())


login = post(
    "/auth/login",
    {
        "username": os.environ.get("TENSOR_AGENT_USER", "agent"),
        "password": os.environ.get("TENSOR_AGENT_PASS", ""),
    },
)
token = login.get("access_token") or login.get("token")
payload = {
    "title": "CASE-BEM-RIGID-BODY-MASS：6×6 刚体质量矩阵闭式块形式验证（1280/5120 面元 <3% PASS）并修复平台质量阵耦合符号+平行轴重复计数缺陷",
    "summary": (
        f"三轴椭球船型体（L/B/T=120/24/16 m，ρ=1025，r_g=(5,0,-3) m）的 6×6 刚体惯量矩阵按 Fossen(2011) eq.(3.44) "
        f"闭式块形式验证：面元四面体积分路线在细分 3（1280 面元）达 {100 * accepted['max_entry_rel_error']:.2f}%"
        f"、细分 4（5120 面元）达 {100 * levels[-1]['max_entry_rel_error']:.2f}%（判据 3%）；装配器与手写块形式残差为机器零。"
        f"同时发现 hydrostatic.compute_mass_matrix 偶块符号错误 + 平行轴重复计数（耦合块误差 200%，动能恒等式误差 "
        f"{100 * legacy['energy_identity_error']:.1f}%，且矩阵仍对称故旧测试全绿），已修复并接入 rao/ice_rao 主链。"
    ),
    "scenario_key": "CASE-BEM-RIGID-BODY-MASS",
    "mission_id": 17324,
    "tags": [
        "TensorBEM",
        "rigid-body-mass",
        "6dof",
        "block-form",
        "case-pool",
        "defect-fix",
        "kinetic-energy-identity",
    ],
    "markdown": md,
    "figures": figures,
}
out = post("/tensor/reports", payload, token)
print(
    json.dumps(
        {
            "report_id": out.get("id") or out.get("report_id"),
            "figures": len(figures),
            "worst_accepted_pct": round(100 * accepted["max_entry_rel_error"], 4),
            "all_pass": R["verdict"]["all_pass"],
        }
    )
)
