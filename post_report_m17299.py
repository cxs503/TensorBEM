import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token")
assert token, login

body = open("dev_report_m17299_fmm_cost_audit_round1.md", encoding="utf-8").read()
fig1 = base64.b64encode(open("docs/fmm_cost_audit/fig_fmm_cost.png", "rb").read()).decode()
fig2 = base64.b64encode(open("docs/fmm_cost_audit/fig_harmonics_ladder.png", "rb").read()).decode()

payload = {
    "title": "m17299 TensorBEM FMM 代价审计重开第 1 轮：谐波核 O(p²) + 叶容量 regime map",
    "summary": (
        "fmm 谐波核由 O(p³) 递推改为 O(p²)（数值 1.8e-16 等价，核 2.3-3.6x，matvec 1.2-1.5x@p>=4）；"
        "新增叶容量 regime map：N=5120、max_leaf≈N/40、p=2 达 0.58x 稠密墙钟、误差 5.8e-4；"
        "台账判据③（渐近斜率 0.9-1.2）实测 1.5，如实 FAIL。"
    ),
    "scenario_key": "tensorbem_fmm_cost_audit",
    "mission_id": 17299,
    "tags": ["TensorBEM", "FMM", "cost_audit", "benchmark", "m17299"],
    "markdown": body,
    "figures": [
        {"name": "fig_fmm_cost.png",
         "caption": "FMM vs 稠密：代价阶梯、精度-阶曲线、叶容量 regime map（虚线=1）、误差-代价散点（红线=1e-3）",
         "png_base64": fig1},
        {"name": "fig_harmonics_ladder.png",
         "caption": "实球谐核：逐 (l,m) 递推 O(p³) 与每阶一条梯子 O(p²) 的代价对比（数值一致到 1.9e-16）",
         "png_base64": fig2},
    ],
}
print(json.dumps(post("/tensor/reports", payload, token=token))[:500])