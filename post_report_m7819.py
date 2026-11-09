"""Publish the m7819 interior-panel suppression report to the platform API."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER")
PASS = os.environ.get("TENSOR_AGENT_PASS")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode())


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token")
if not token:
    raise SystemExit(f"no token in login response keys: {sorted(login)}")
payload = json.loads(open("/tmp/report.json", encoding="utf-8").read())
try:
    result = post("/tensor/reports", payload, token)
except urllib.error.HTTPError as exc:  # pragma: no cover
    print("HTTP", exc.code, exc.read().decode()[:500])
    raise
print(json.dumps({k: result[k] for k in list(result)[:6]}, default=str)[:600])