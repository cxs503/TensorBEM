"""Audit probe for m17235 (CASE-BEM-HULME-HEMISPHERE accountability review).

Read-only: fetches mission record, benchmark case rows and report metadata
from the platform API so the verdict can cite machine-readable evidence
instead of prose from the mission's own report.
"""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def call(path, token=None, method="GET", body=None):
    url = f"{BASE}/{path}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


def login():
    for path in ("tensor/login", "auth/login", "login"):
        res = call(path, method="POST", body={"username": USER, "password": PASS})
        if isinstance(res, dict) and not res.get("_error"):
            for key in ("access_token", "token", "accessToken"):
                if res.get(key):
                    return res[key], path
            if isinstance(res.get("data"), dict):
                d = res["data"]
                for key in ("access_token", "token", "accessToken"):
                    if d.get(key):
                        return d[key], path
        print(f"login via {path}: {str(res)[:160]}")
    raise SystemExit("login failed")


if __name__ == "__main__":
    tok, lpath = login()
    print("login ok via", lpath)
    probes = [
        "tensor/benchmark_cases?anchor_slug=CASE-BEM-HULME-HEMISPHERE",
        "tensor/missions/17181",
        "tensor/reports?mission_id=17181",
        "tensor/anchors/CASE-BEM-HULME-HEMISPHERE",
    ]
    out = {}
    for p in probes:
        res = call(p, tok)
        out[p] = res
        print("===", p)
        print(json.dumps(res, ensure_ascii=False)[:1200])
    with open("/tmp/m17235_probe.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)