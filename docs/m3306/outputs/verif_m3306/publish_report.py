"""Publish the m3306 progress report to the platform, with size-capped figures."""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path("/root/tp-agent/work/m3306")
HERE = ROOT / "outputs" / "verif_m3306"
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")

FIGURES = [
    ("fig_registry_3d_pins.png", "图 1 注册表 33 条三维自由液面文档数字的复算偏差（对数轴，按基准族着色，黑线为各条容差），无一条越线"),
    ("fig_hulme_orders_chief.png", "图 2 左：Hulme 半球五个量的观测阶（文档值 vs 复算值，2 阶参考线）；右：Ka=2.58 不规则频率处未增广 BIE 的 B33' 为负（违背耗散性），5 点内域 CHIEF 将其恢复为 +0.0687，条件数 145.7→4.24"),
    ("fig_3d_convergence.png", "图 3 半球（Ka=1）与潜没球体（f/a=1.5, Ka=0.8）各量相对误差随面元数变化，双对数斜率约 −1（误差 ~ N⁻¹，即 h² 二阶）"),
]


def encode(path: Path, budget: int = 52000) -> str:
    """Downscale until the PNG fits the API payload budget, then base64 it."""
    image = Image.open(path).convert("RGB")
    width = image.width
    while True:
        scaled = image.resize((width, int(image.height * width / image.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        scaled.save(buffer, format="PNG", optimize=True)
        blob = buffer.getvalue()
        if len(blob) <= budget or width <= 640:
            break
        width = int(width * 0.85)
    print(f"{path.name}: {len(blob)/1024:.0f} KB at width {width}")
    return base64.b64encode(blob).decode()


def login() -> str:
    payload = json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode()
    for path in ("/tensor/login", "/auth/login", "/login"):
        request = urllib.request.Request(
            BASE + path, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:
            print(path, "HTTP", error.code)
            continue
        for key in ("access_token", "token", "jwt"):
            if key in body:
                return body[key]
        print(path, "no token field:", list(body)[:8])
    raise SystemExit("login failed")


def main() -> None:
    markdown = (ROOT / "dev_report_m3306_three_dimensional_registry_pins.md").read_text()
    report = {
        "title": "TensorBEM 三维自由液面文档数字纳入 declared-number 注册表（33 条全 PASS + 一处收敛阶口径更正）",
        "summary": (
            "把 docs/marine_bem_benchmarks.md 表 10/11/12（Hulme 漂浮半球、Ka=2.58 免盖板 CHIEF、"
            "Wang/Linton 潜没球体）的 33 个文档数字从「仅由各基准测试模块断言」升级为注册表可执行复算项；"
            "standard 层 37 条全 PASS（新项最坏相对偏差 1.85%）。发现并更正表 10 中 B11' 观测阶的口径混用"
            "（名义 √N 比 1.88 vs 实际 h_mean 比 1.94），误差值本身不变。未改任何求解器代码。"
        ),
        "scenario_key": "tensorbem_verification_debt",
        "mission_id": 3306,
        "tags": ["TensorBEM", "verification", "declared-numbers", "3D-free-surface", "Hulme", "CHIEF"],
        "markdown": markdown,
        "figures": [
            {"name": name, "caption": caption, "png_base64": encode(HERE / name)}
            for name, caption in FIGURES
        ],
    }
    blob = json.dumps(report).encode()
    print(f"payload {len(blob)/1024:.0f} KB")
    request = urllib.request.Request(
        BASE + "/tensor/reports",
        data=blob,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + login(),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            print(response.status, response.read().decode()[:600])
    except urllib.error.HTTPError as error:
        print(error.code, error.read().decode()[:800])


if __name__ == "__main__":
    main()