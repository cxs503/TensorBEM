"""Publish the m17246 report; every number read back from numbers.json (zero hand-copy)."""
import base64, json, os, pathlib, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
D = ROOT / "docs/case_bem_neumann_kelvin"
R = json.loads((D / "numbers.json").read_text())
a = R["route_a"]
rows = "\n".join(
    f"| {r['Fr']:.2f} | {r['U']:.4f} | {r['ref_michell_canonical']:.4f} | {r['ref_Cw']:.6e} | "
    f"{r['got_farfield_panel_finest']:.4f} | {r['got_Cw_panel_finest']:.6e} | "
    f"{100*r['err_panel_finest']:+.3f} % | {'PASS' if r['passed_finest'] else 'FAIL'} |" for r in a["rows"])
b = R["route_b"]
brows = "\n".join(f"| {r['Fr']:.2f} | {r['rw_far']:.1f} | {r['ref']:.1f} | {100*r['rel_error']:+.0f} % |"
                  for r in b["rows"])
md = (D / "REPORT.md").read_text()
md += f"\n\n---\n\n### 机械汇总（脚本注入）\n\n| Fr | U [m/s] | ref Rw [N] | ref Cw | got Rw [N] | got Cw | 误差 | 判据 |\n|---:|---:|---:|---:|---:|---:|---:|:--:|\n{rows}\n\n"
md += f"Route B（面元 NK，honest FAIL）：\n\n| Fr | got Rw [N] | ref Rw [N] | 误差 |\n|---:|---:|---:|---:|\n{brows}\n\n"
md += (f"主路线 max|误差| = {100*a['max_abs_error']:.3f} %（判据 5 %）→ all_passed = {a['all_passed']}；"
       f"Route B max|误差| = {100*b['max_abs_error']:.0f} % → any_passed = {b['any_passed']}；"
       f"核自由面残差 = {R['kernel']['free_surface_condition']['max_relative_residual']:.2e}。\n")

figs = []
for name, cap in (("fig_cw_vs_froude.png", "Cw(Fr)：NK 稳态远场（got）vs Michell 薄船理论（ref），点线为相对误差，虚线 5% 判据线"),
                  ("fig_route_a_convergence.png", "Route A 中心面面板加密的误差收敛（全部 <5%）"),
                  ("fig_kelvin_wake.png", "NK 薄船自由面波高场（Fr=0.30），虚线为 Kelvin cusp ±19.47°")):
    p = D / name
    if p.exists():
        figs.append({"name": name, "caption": cap, "png_base64": base64.b64encode(p.read_bytes()).decode()})

def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

d = post("/auth/login", {"username": os.environ.get("TENSOR_AGENT_USER", "agent"),
                         "password": os.environ.get("TENSOR_AGENT_PASS", "")})
token = d.get("access_token") or d.get("token")
payload = {
    "title": "CASE-BEM-NEUMANN-KELVIN：稳态 NK 兴波阻力 vs Michell Cw(Fr)（主路线 <5% PASS，面元路线 honest FAIL）",
    "summary": (f"稳态 Kelvin 远场算子（包含 1/U^2 的精确前因子 κ/2πg 缺陷修复）在 Fr=0.20…0.40 五点对标 "
                f"Michell 薄船理论，最大相对误差 {100*a['max_abs_error']:.2f}%（判据 5%）；真实船体面元 NK "
                f"单层法失败（{100*b['max_abs_error']:.0f}%），根因为 halfplane 辐射项 Heaviside 不连续使物面求积不收敛。"),
    "scenario_key": "CASE-BEM-NEUMANN-KELVIN",
    "mission_id": 17246,
    "tags": ["TensorBEM", "Neumann-Kelvin", "Michell", "wave-resistance", "case-pool", "honest-fail"],
    "markdown": md,
    "figures": figs,
}
out = post("/tensor/reports", payload, token)
print(json.dumps({"report_id": out.get("id") or out.get("report_id"), "figures": len(figs),
                  "n_rows": len(a["rows"]), "all_passed": a["all_passed"]})[:400])
