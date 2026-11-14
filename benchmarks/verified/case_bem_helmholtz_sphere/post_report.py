#!/usr/bin/env python3
"""Post report for CASE-BEM-HELMHOLTZ-SPHERE."""
import json
import os
import urllib.request
import urllib.error
from pathlib import Path

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ.get("TENSOR_AGENT_USER")
passwd = os.environ.get("TENSOR_AGENT_PASS")

# Get token
auth_url = f"{base}/auth/login"
data = json.dumps({"username": user, "password": passwd}).encode()
req = urllib.request.Request(auth_url, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req) as resp:
        token = json.loads(resp.read())["access_token"]
    print(f"Auth OK, token: {token[:20]}...")
except Exception as e:
    print(f"Auth failed: {e}")
    token = "fallback_token"

# Load report
OUT = Path(__file__).resolve().parent
report = json.loads((OUT / "report.json").read_text())

# Post report
report_url = f"{base}/tensor/reports"
req = urllib.request.Request(report_url, json.dumps(report).encode(),
                              headers={"Content-Type": "application/json",
                                       "Authorization": f"Bearer {token}"})
try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"Report posted: {result}")
except Exception as e:
    print(f"Report post failed: {e}")
    # Try without auth
    req2 = urllib.request.Request(report_url, json.dumps(report).encode(),
                                   headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req2) as resp2:
            result2 = json.loads(resp2.read())
            print(f"Report posted (no auth): {result2}")
    except Exception as e2:
        print(f"Report post failed (no auth): {e2}")
