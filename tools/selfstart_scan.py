"""Read-only scan helper for selfstart missions: lists recent tensor reports (id/title/date)."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def call(path, data=None, token=None):
    req = urllib.request.Request(BASE + path, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    body = json.dumps(data).encode() if data else None
    with urllib.request.urlopen(req, body) as r:
        return json.loads(r.read().decode())


tok = call("/auth/login", {"username": USER, "password": PASS})
tok = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
reports = call("/tensor/reports?limit=20", token=tok)
items = reports if isinstance(reports, list) else (reports.get("items") or reports.get("reports") or reports.get("data") or [])
for r in items[:25]:
    print(r.get("id"), "|", str(r.get("title"))[:90], "|", r.get("created_at", ""), "|", r.get("mission_id"))