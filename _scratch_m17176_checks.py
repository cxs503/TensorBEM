#!/usr/bin/env python3
"""Pre-work three checks for m17176 (CASE-BEM-SPHERE-ADDED-MASS)."""
import json
import os
import urllib.request
import urllib.error

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ.get("TENSOR_AGENT_USER")
passwd = os.environ.get("TENSOR_AGENT_PASS")


def call(path, method="GET", body=None, token=None):
    url = f"{base}/{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except Exception:
            return e.code, None
    except Exception as e:
        return -1, str(e)


# login
st, data = call("auth/login", "POST", {"username": user, "password": passwd})
print("login:", st)
token = (data or {}).get("access_token") if isinstance(data, dict) else None
print("token:", (token or "")[:12] + "...")

# probe candidate endpoints
probes = [
    ("GET", "knowledge_base", None),
    ("GET", "knowledge_base?domain=TensorBEM", None),
    ("GET", "knowledge_base/capabilities?domain=TensorBEM", None),
    ("GET", "negative_knowledge_ledger?keywords=TensorBEM", None),
    ("GET", "oss_solutions?domain=TensorBEM", None),
    ("GET", "knowledge_base?keywords=TensorBEM", None),
]
for method, path, body in probes:
    st, data = call(path, method, body, token)
    snippet = json.dumps(data)[:400] if data is not None else None
    print(f"\n=== {method} {path} -> {st}")
    print(snippet)
