"""Fetch one platform report body as markdown (ad-hoc, m16274)."""

import os
import sys

import requests

base = os.environ["TENSOR_API_BASE"]
rid = sys.argv[1]
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
resp = s.get(f"{base}/tensor/reports/{rid}", headers=h, timeout=120)
resp.raise_for_status()
d = resp.json()
print("TITLE:", d.get("title"))
print("SCENARIO:", d.get("scenario_key"))
print((d.get("markdown") or d.get("summary") or ""))