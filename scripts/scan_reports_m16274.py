"""Read-only scan of the platform report feed (ad-hoc, m16274)."""

import os

import requests

base = os.environ["TENSOR_API_BASE"]
s = requests.Session()
r = s.post(
    f"{base}/auth/login",
    json={
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    },
    timeout=60,
)
r.raise_for_status()
tok = r.json().get("access_token") or r.json().get("token")
h = {"Authorization": f"Bearer {tok}"}
resp = s.get(f"{base}/tensor/reports", params={"limit": 20}, headers=h, timeout=120)
resp.raise_for_status()
d = resp.json()
items = d.get("items") if isinstance(d, dict) else d
open("outputs/tok_m16274.txt", "w").write(tok)
for it in items:
    print(it.get("id"), it.get("scenario_key"), "|", (it.get("title") or "")[:80])