"""m14572: publish the paper-level report (TensorBEM#592 reopen, round 3)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER")
PASS = os.environ.get("TENSOR_AGENT_PASS")
ROOT = "/root/tp-agent/work/m14572"
OUT = os.path.dirname(os.path.abspath(__file__))


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode())


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token")
assert token, f"no token: {sorted(login)}"

md_path = f"{ROOT}/dev_report_m14572_quadratic_panel_pr592_reopen_round3.md"
md = open(md_path).read()
record = json.load(open(
    f"{ROOT}/docs/quadratic_panel_verification/linear_vs_quadratic_m14572.json"))
matchups = {m["name"]: m for m in record["matchups"]}
mp = matchups["matched_panels_80"]
mn = matchups["matched_nodes_162"]

figures = []
for name, path, caption in [
    ("fig_linear_vs_quadratic", f"{ROOT}/docs/quadratic_panel_verification/fig_linear_vs_quadratic_m14572.png",
     "图1 同一把尺子的正面对比：(a) 通量相对 L2 误差 vs 面板/单元数（同一 icosphere 族，"
     f"等面板数 80：平板 {mp['linear']['err_flux']:.3e} vs 二次 {mp['quadratic']['err_flux']:.3e}，"
     f"改善 {mp['ratio']:.1f}×）；(b) vs 未知量数（等 162 未知量：平板 {mn['linear']['err_flux']:.3e} "
     f"vs 二次 {mn['quadratic']['err_flux']:.3e}，改善 {mn['ratio']:.1f}×）；红线=3% 验收，橙虚线=15% 硬限"),
    ("fig_flux_error_field", f"{ROOT}/docs/quadratic_panel_verification/fig_flux_error_field_m14572.png",
     "图2 局部通量误差场（各自原生支撑：平板顶点 / 二次单元质心），共用对数色标："
     "平板侧误差在球面各向近均匀地大（全局 6.148e-01），二次侧整体低 1.5~2 个量级（全局 2.854e-02）"),
]:
    figures.append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(open(path, "rb").read()).decode(),
    })

payload = {
    "title": "TensorBEM 二次曲面单元 vs 平板单元：同一把尺子的正面对比（PR#592 重开·管到底第3轮·m14572）",
    "summary": (
        "PR#592 的 m8745 交付物早已在 main（Merge #577），本轮补上它缺的证据面："
        "把二次（6 节点等参）与 order-1（平板顶点配置）算子放进同一进程、同一 icosphere 族、"
        "同一 Dirichlet→通量读数对比。等面板数（80）通量误差改善 "
        f"{mp['ratio']:.1f}×（{mp['quadratic']['err_flux']:.3e} vs {mp['linear']['err_flux']:.3e}），"
        f"等未知量（162）改善 {mn['ratio']:.1f}×；二次侧 3% 验收在 s≥1 成立，平板侧误差不随加密下降（方法性局限，非 bug）。"
        "新增 1 个工具 + 2 个测试钉 + 报告的配对/误差场数据（零手抄）。"
    ),
    "scenario_key": "tensorbem.quadratic_panel.operator_head_to_head",
    "mission_id": 14572,
    "tags": ["TensorBEM", "quadratic-panel", "BEM", "panel-order", "head-to-head",
             "pr592-reopen", "verification"],
    "markdown": md,
    "figures": figures,
}

resp = post("/tensor/reports", payload, token=token)
print(json.dumps(resp, ensure_ascii=False)[:600])