"""Publish the anchor-surface report to the platform (``POST /tensor/reports``).

The markdown is the generated document, not a copy: the same file a reader
opens in the repository is what the platform stores, and the figures are the
PNGs ``write_figures`` produced. Run::

    python docs/radiation_diffraction_anchors_3d/post_report.py
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parent / "radiation_diffraction_anchors_3d.md"
RESULT = HERE / "results.json"

CAPTIONS = {
    "fig_radiff_added_mass_anchors.png": (
        "Left: unbounded A33 vs panel count - sphere converges to 2/3 rho pi a^3, "
        "closed hemisphere sits above the thin-disk limit 8/3 rho a^3 (piston flow on "
        "its flat lid). Right: floating hemisphere A33(ka) climbing to rho pi a^3/3."
    ),
    "fig_radiff_froude_krylov.png": (
        "Froude-Krylov F1/F3/F5 relative error against the closed forms vs panel count, "
        "with a slope -2 reference: second-order convergence, 0.048 % worst at 2080 panels."
    ),
    "fig_radiff_rao_limits.png": (
        "Left: barge heave RAO with the long-wave band [0.95,1.05] and the high-frequency "
        "bound 0.2. Right: zero-frequency control - A33 plateaus, B33 decays with omega."
    ),
    "fig_radiff_rao_convergence.png": (
        "Heave RAO and A33 versus mesh refinement on the 10x5 -> 20x10 -> 40x20 barge "
        "ladder; observed order 2.0-2.7 feeds criterion C5."
    ),
}


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")

    login = json.dumps(
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]}
    ).encode()
    request = urllib.request.Request(
        f"{base}/auth/login", data=login, headers={"Content-Type": "application/json"}
    )
    token = json.loads(urllib.request.urlopen(request).read())["access_token"]

    figures = []
    for name, caption in CAPTIONS.items():
        path = HERE / name
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )

    payload = json.loads(RESULT.read_text())
    passed = [item["id"] for item in payload["criteria"] if item["pass"]]
    failed = [item["id"] for item in payload["criteria"] if not item["pass"]]
    summary = (
        "TensorBEM 3-D radiation/diffraction anchor surface: %d/%d criteria pass; "
        "honest FAIL on %s (the campaign's 2/3 rho pi a^3 value belongs to the sphere: "
        "a closed hemisphere drives a piston flow on its flat lid and sits above the "
        "thin-disk limit). New theorem-exact anchor verified: floating hemisphere "
        "A33(ka->inf) = rho pi a^3/3 = A33(sphere)/2."
        % (len(passed), len(payload["criteria"]), ", ".join(failed) or "none")
    )

    body = {
        "title": "TensorBEM 3-D 辐射/绕射验证面：半球/球附加质量锚 + Froude-Krylov 闭式 + 极限 RAO + 收敛律 + 负控",
        "summary": summary,
        "scenario_key": "p167w2-n01-bem-radiation",
        "mission_id": 9696,
        "tags": [
            "TensorBEM",
            "radiation-diffraction",
            "added-mass",
            "froude-krylov",
            "RAO",
            "verification-anchor",
            "honest-fail",
        ],
        "markdown": DOC.read_text(),
        "figures": figures,
    }
    request = urllib.request.Request(
        f"{base}/tensor/reports",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:  # surface the API's field-level detail
        print("HTTP", error.code, error.read().decode()[:2000])
        return 1
    print(response.status, response.read()[:400].decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
