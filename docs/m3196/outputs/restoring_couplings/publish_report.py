"""Login to the Tensor API and publish the m3196 progress report."""

from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/root/tp-agent/work/m3196")
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{API}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read() or b"{}")


def main() -> None:
    if not USER or not PASSWORD:
        raise SystemExit("credentials not in environment")
    token = post("/auth/login", {"username": USER, "password": PASSWORD})["access_token"]

    image = mpimg.imread(ROOT / "outputs/restoring_couplings/fig_restoring_couplings.png")
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.imshow(image)
    ax.axis("off")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=72, bbox_inches="tight",
                pil_kwargs={"optimize": True})
    png = buffer.getvalue()
    print("figure bytes", len(png))

    markdown = (ROOT / "docs/hydrostatic_restoring_couplings.md").read_text()
    payload = {
        "title": "TensorBEM 静水恢复矩阵完整化：水线面偏心引起的 heave–roll / heave–pitch / roll–pitch 耦合",
        "summary": (
            "compute_hydrostatic_restoring_matrix 原先只填 heave/roll/pitch 三个对角元并断言所有"
            "非对角项为零；该断言只在水线面形心落在原点时成立。本单补齐 C34/C35/C45 与平行轴项"
            "（C44/C55 旧值相对误差 70.9%/83.5%），新增独立能量 Hessian 基准（潜体多边形裁剪+散度定理，"
            "只用阿基米德浮力与重量），全矩阵相对偏差 1.3e-8，并借它抓出第一版 C34 的符号错误。"
            "对称船体数值不变：132 条静水/RAO 回归全绿 + 9 条新钉桩。"
        ),
        "scenario_key": "tensorbem-hydrostatic-restoring-couplings",
        "mission_id": 3196,
        "tags": ["TensorBEM", "hydrostatics", "restoring-matrix",
                 "verification-debt", "energy-hessian"],
        "markdown": markdown,
        "figures": [{
            "name": "fig_restoring_couplings.png",
            "caption": ("左：|C35|/C33 与纵摇响应放大因子随 LCF/L 线性增长；"
                        "右：单位纵摇力矩引起的垂荡响应（旧实现恒为 0）。"),
            "png_base64": base64.b64encode(png).decode(),
        }],
    }
    body = json.dumps(payload)
    print("report json bytes", len(body), "markdown chars", len(markdown))
    result = post("/tensor/reports", payload, token)
    print("POST result:", json.dumps(result)[:400])


if __name__ == "__main__":
    main()