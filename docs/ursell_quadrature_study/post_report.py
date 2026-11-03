"""Publish this mission's progress report to the platform.

Reads the credentials from the agent environment file at run time; nothing
secret is stored in the repository. The markdown body is the verification
report in this directory's parent, plus the generated tables.

Run: python docs/ursell_quadrature_study/post_report.py <mission_id>
"""

import base64
import json
import pathlib
import sys
import urllib.request

env = {}
for line in pathlib.Path("/root/tp-agent/agent.env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")

base = env["TENSOR_API_BASE"].rstrip("/")
worktree = pathlib.Path(__file__).resolve().parents[2]
mission_id = int(sys.argv[1]) if len(sys.argv) > 1 else 2223


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, json.loads(response.read().decode())


_, login = post(
    "/auth/login",
    {"username": env["TENSOR_AGENT_USER"], "password": env["TENSOR_AGENT_PASS"]},
)
token = login.get("access_token") or login.get("token") or login.get("data", {}).get(
    "access_token"
)
assert token, login

markdown = (worktree / "docs/ursell_verification_debt_2026-11-02.md").read_text(
    encoding="utf-8"
)
markdown += (
    "\n\n---\n\n## Appendix: generated tables "
    "(docs/ursell_quadrature_study/summary.md)\n\n"
)
markdown += (worktree / "docs/ursell_quadrature_study/summary.md").read_text(
    encoding="utf-8"
)

png = base64.b64encode(
    (worktree / "docs/figures/ursell_quadrature_study.png").read_bytes()
).decode()

payload = {
    "title": "TensorBEM 验证欠账清算：Ursell 128 面元 honest-FAIL 的根因与更正（quadrature vs discretisation）",
    "summary": (
        "m238 审计留下的 honest FAIL（Ursell 128 面元附加质量误差 8.51e-3 vs 文档 7.7e-3，偏 10.6%）已定位并结清："
        "不是文档生成环境/BLAS 差异，而是 267d5a3 更换了源势求解方式（解析 log 原函数 + 仅对光滑自由面余项求积）。"
        "以 878834a 存档树与 HEAD 同参数对跑 + 面元节点扫描证明旧数字是求积未收敛值"
        "（12→24→48 节点 A 从 2562.87 漂到 2565.81，仍差收敛值 4e-4 相对），"
        "新实现自 6 节点起对节点数完全不敏感（矩阵差 ~3e-17）。"
        "README 与 docs/marine_bem_benchmarks.md §4.6 已改为可复现的一阶收敛数字"
        "（128 面元 A 8.5e-3、B 1.3e-2、近/远场 5.3e-3），全部数字由 raw JSON 经脚本生成，"
        "并新增 5 条公式级回归测试（Ursell 相关 23 项测试全绿）。"
    ),
    "scenario_key": "tensorbem_verification",
    "mission_id": mission_id,
    "tags": ["TensorBEM", "verification", "Ursell", "quadrature", "honest-fail-closed"],
    "markdown": markdown,
    "figures": [
        {
            "name": "ursell_quadrature_study.png",
            "caption": (
                "Left: added mass at 128 panels versus Gauss-Legendre nodes per panel "
                "(current solver flat from 6 nodes; legacy 878834a climbing towards it). "
                "Right: remaining error versus panel count, first order (slope -1 line)."
            ),
            "png_base64": png,
        }
    ],
}
status, body = post("/tensor/reports", payload, token)
print("status", status)
print(json.dumps(body, ensure_ascii=False)[:800])