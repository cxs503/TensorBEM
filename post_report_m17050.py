#!/usr/bin/env python3
"""Post the FEAT-OSS-INVENTORY-CAD report to the platform."""
import json
import os
import urllib.request
import urllib.error

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
        print(f"Token obtained: {token[:20]}...")
except Exception as e:
    print(f"Auth failed: {e}")
    token = "fallback_token"

# Read the dev report
report_md = open("/root/tp-agent/work/m17050/docs/dev_report_m17050_oss_inventory_cad.md", encoding="utf-8").read()

report = {
    "title": "OSS CAD/Geometry Kernel Inventory (FEAT-OSS-INVENTORY-CAD)",
    "summary": "Surveyed 25 open-source CAD/geometry kernel projects across 3 tiers; identified 8 GAP-CAD-* anchors with executed probes over TensorBEM src/tests; all 11 integrity tests pass.",
    "scenario_key": "FEAT-OSS-INVENTORY-CAD",
    "mission_id": 17050,
    "tags": ["TensorBEM", "CAD", "geometry-kernel", "inventory", "FEAT-OSS-INVENTORY-CAD"],
    "markdown": report_md,
}

# Post report
report_url = f"{base}/tensor/reports"
req = urllib.request.Request(
    report_url,
    json.dumps(report).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
)
try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"Report posted: {result}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP Error {e.code}: {body}")
except Exception as e:
    print(f"Report failed: {e}")
