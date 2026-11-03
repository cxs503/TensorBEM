"""Read-only helper: list recent platform reports (titles + mission ids)."""

import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def get(path, token):
    req = urllib.request.Request(BASE + path)
    req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


if __name__ == "__main__":
    tok = post("/auth/login", {"username": USER, "password": PASS})
    token = tok.get("token") or tok.get("access_token")
    with open("/tmp/tok.txt", "w") as fh:
        fh.write(token)
    data = get("/tensor/reports?limit=20", token)
    items = data if isinstance(data, list) else (data.get("items") or data.get("reports") or [])
    for r in items[:20]:
        print(r.get("mission_id"), "|", (r.get("title") or "")[:95])