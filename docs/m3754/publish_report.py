"""Publish the PR #128 healing-merge report to the platform report API.

Figures are re-encoded to stay well under the 250 KB payload ceiling that
silently fails on this endpoint.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def b64_png(path: Path, target_bytes: int = 60000) -> str:
    raw = path.read_bytes()
    if len(raw) <= target_bytes:
        return base64.b64encode(raw).decode()
    from PIL import Image

    img = Image.open(path).convert("RGB")
    for scale in (0.8, 0.65, 0.5, 0.4, 0.3):
        buf = io.BytesIO()
        small = img.resize((int(img.width * scale), int(img.height * scale)))
        small.save(buf, format="PNG", optimize=True)
        if buf.tell() <= target_bytes:
            return base64.b64encode(buf.getvalue()).decode()
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    token = None
    for key in ("access_token", "token"):
        try:
            res = post("/auth/login", {"username": USER, "password": PASS})
            token = res.get(key)
        except Exception as exc:  # noqa: BLE001
            print("login failed:", exc)
            break
        if token:
            break
    if not token:
        print("NO TOKEN")
        return 2

    md = (ROOT / "docs" / "m3754" / "merge_128_report.md").read_text()
    figs = []
    for name, caption in (
        ("fig_registry_union.png",
         "图 1  声明数字注册表的三方并集：main 84 行 + PR #128 63 行 = 合并后 147 行、147 个唯一 key，"
         "两套 3-D 自由液面命名（hulme_hemisphere/hulme_chief/submerged_sphere 与 "
         "hulme3d/hulme3d_chief/wang_linton3d）并存，无一侧被丢弃。"),
        ("fig_test_parity.png",
         "图 2  核心套件（8 个受影响测试文件，同机同参）main 基线 vs 合并树：passed 53 → 54，"
         "failed 0 → 0，合并树不低于基线。"),
    ):
        p = ROOT / "docs" / "m3754" / "figures" / name
        figs.append({"name": name, "caption": caption, "png_base64": b64_png(p)})

    payload = {
        "title": "TensorBEM PR #128 治愈合并：三维自由液面声明数字注册表的三方语义并集",
        "summary": (
            "agent/m3181（PR #128）对 main 变脏：两侧在同一文件同一区域各实现一份 3-D 自由液面"
            "声明数字注册表。三方合并后逐文件解冲：保留 main 的 84 行字面量登记与 PR 的 63 行表驱动"
            "登记（147 行 / 147 唯一 key），统一共享模块别名，并将 PR 的 _chief_rows() 改名"
            "_free_surface_chief_rows() 以消除 git 不报的同名函数静默覆盖；文档自动合并经逐行核验"
            "（PR §4.12 的 94 行全部存活，main 0 行丢失）。核心套件 main 53 passed / 合并 54 passed，"
            "failed 均为 0，无回归。"
        ),
        "scenario_key": "tensorbem.merge_healing",
        "mission_id": 3754,
        "tags": ["TensorBEM", "merge-healing", "PR#128", "declared-numbers", "3D-free-surface", "CI-parity"],
        "markdown": md,
        "figures": figs,
    }
    try:
        res = post("/tensor/reports", payload, token)
        print("POSTED", json.dumps(res)[:400])
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        print("HTTP", exc.code, exc.read().decode()[:400])
        return 1
    print("payload KB", len(json.dumps(payload)) // 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())