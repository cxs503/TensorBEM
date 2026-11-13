#!/usr/bin/env python3
"""POST the audit report to the platform API (credentials from env, never inline)."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
LOGIN = os.environ.get("TENSOR_LOGIN_PATH", "/auth/login")


def post(url, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode()[:800]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:800]


st, body = post(BASE + LOGIN, {"username": USER, "password": PASS})
tok = json.loads(body).get("access_token") if st == 200 else None
if not tok:
    try:
        tok = json.loads(body).get("token")
    except Exception:
        pass
print("login", st, "token?", bool(tok))
if not tok:
    raise SystemExit("no token: " + body[:300])

rep = json.load(open("/root/tp-agent/work/m17113/docs/audit_restat_20261113/report_m17113.json"))
st2, body2 = post(BASE + "/tensor/reports", rep, tok)
print("report POST", st2, body2)