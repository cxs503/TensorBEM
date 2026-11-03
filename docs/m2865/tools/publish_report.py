"""Publish the m2865 progress report to the platform (report discipline).

Markdown is taken from the committed document; figures are embedded as base64
PNGs produced by the benchmark module.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = Path(__file__).resolve().parents[3]
MISSION_ID = 2865


def _post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def _figure(path: Path, caption: str) -> dict:
    data = base64.b64encode(path.read_bytes()).decode()
    assert len(data) < 200_000, f"{path} is too large for the report endpoint"
    return {"name": path.name, "caption": caption, "png_base64": data}


def main() -> None:
    token = _post(
        "/auth/login", {"username": USER, "password": PASS}
    )
    token = token.get("token") or token.get("access_token")
    markdown = (ROOT / "docs" / "ursell_frequency_envelope.md").read_text()
    payload = json.loads((ROOT / "artifacts" / "ursell_envelope.json").read_text())
    summary = payload["summary"]
    title = (
        "Ursell 辐射求解器频率—间隙精度包络：结清「只测单点」欠账（"
        f"{summary['grid_points']} 点，pass {summary['verdict_counts']['pass']} / "
        f"marginal {summary['verdict_counts']['marginal']} / "
        f"fail {summary['verdict_counts']['fail']}）"
    )
    body = {
        "title": title,
        "summary": (
            "把 docs/ursell_error_budget.md §5 的未验证项（ka>=1 与更浅间隙）扩成 "
            "ka∈[0.3,2.0]×a/f∈[0.5,0.85] 的 15 点网格：参考解截断、MFS 独立方法与 "
            "N=64/128/256 加密阶三重保护，最差误差 "
            f"{max(summary['worst_point']['added_mass_relative_error'], summary['worst_point']['damping_relative_error']):.3e}。"
        ),
        "scenario_key": "tensorbem-ursell-envelope",
        "mission_id": MISSION_ID,
        "tags": ["TensorBEM", "verification", "Ursell", "radiation", "envelope"],
        "markdown": markdown,
        "figures": [
            _figure(
                ROOT / "docs" / "figures" / "ursell_envelope.png",
                "误差随 ka 的包络（左/中）与系数本身对照（右）",
            )
        ],
    }
    response = _post("/tensor/reports", body, token)
    print(json.dumps(response)[:400])


if __name__ == "__main__":
    main()