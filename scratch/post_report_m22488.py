"""Post the m22488 report (capytaine refinement-ladder audit) to the platform."""
import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

DOCS = pathlib.Path("docs/case_bem_submerged_sphere_havelock")


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def get(path, token):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


auth = post("/auth/login", {"username": USER, "password": PASS})
token = auth.get("access_token") or auth.get("token")
print("token ok" if token else auth)

md = (DOCS / "capytaine_resolution_note.md").read_text()
fig = DOCS / "fig14_capytaine_refinement.png"
report = {
    "title": "Havelock 潜球算例：第三方 capytaine 分辨率阶梯复核——交叉核对能证到多准",
    "summary": (
        "对 CASE-BEM-SUBMERGED-SPHERE-HAVELOCK 的第三方交叉核对做对抗式复核：把 capytaine 的"
        " 320/1280 两档单网格比较扩展为 80/320/1280/5120 四档阶梯。结论：既有 crosscheck 的"
        " FAIL(6.76%) 不是对多极参考解的反证——capytaine 偏差随加密单调衰减(3.9%→2.6%→1.4~1.9%，"
        "27/28 项衰减，无平台无变号)；但预声明的 1% 认证判据 C1 未达成(honest FAIL)，该路线只能"
        "把参考解认证到约 2%。近场奇异项假设(low_freq vs low_freq_with_rankine_part)被证伪。"
        "TensorBEM 1280 面元对 capytaine 5120 面元最大 3.27%，0/28 超 5% 池判据。"),
    "scenario_key": "bem_submerged_sphere_havelock_capytaine_resolution",
    "mission_id": 22488,
    "tags": ["TensorBEM", "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK", "capytaine",
             "cross-validation", "verification-debt", "honest-FAIL"],
    "markdown": md,
    "figures": [{
        "name": fig.name,
        "caption": (
            "capytaine 四档网格(80/320/1280/5120)上对多极参考解的偏差衰减：加质量(左)与阻尼(右)。"
            "红=5% 池判据，绿点线=预声明 1% 判据。偏差单调下降但 5120 档仍在 1.4~1.9%，"
            "故 1% 认证未达成。"),
        "png_base64": base64.b64encode(fig.read_bytes()).decode(),
    }],
}
res = post("/tensor/reports", report, token)
print(json.dumps(res)[:400])