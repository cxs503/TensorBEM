"""Publish the m6271 progress report (credentials read from the environment only)."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def _call(path, payload=None, token=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


auth = _call("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                             "password": os.environ["TENSOR_AGENT_PASS"]})
token = auth.get("access_token") or auth.get("token") or auth.get("data", {}).get("access_token")
print("login ok:", bool(token))

with open("/tmp/report.json", encoding="utf-8") as fh:
    report = json.load(fh)
out = _call("/tensor/reports", report, token)
print("report response:", json.dumps(out)[:500])

rows = _call("/tensor/reports?mission_id=6271&limit=5", token=token)
items = rows.get("items") or rows.get("reports") or rows
if isinstance(items, list):
    for it in items[:3]:
        print("  id=%s title=%s" % (it.get("id"), str(it.get("title"))[:60]))
