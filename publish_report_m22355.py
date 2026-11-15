#!/usr/bin/env python3
"""Publish report to TensorPlatform."""
import json
import os
import urllib.request
import urllib.error

API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

def get_token():
    data = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(f"{API_BASE}/auth/login", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["access_token"]

def publish_report(token):
    with open("/tmp/report_m22355.json") as f:
        report = json.load(f)
    data = json.dumps(report).encode()
    req = urllib.request.Request(f"{API_BASE}/tensor/reports", data=data,
                                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
        print(f"Report published: {json.dumps(result, indent=2)}")
        return result

if __name__ == "__main__":
    token = get_token()
    print(f"Token obtained: {token[:20]}...")
    publish_report(token)
