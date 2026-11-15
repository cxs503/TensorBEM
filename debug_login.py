#!/usr/bin/env python3
"""Debug login and publish report."""
import json
import os
import urllib.request

API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")

def debug_login():
    data = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(f"{API_BASE}/auth/login", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        print("Login response:", json.dumps(result, indent=2))
        return result

if __name__ == "__main__":
    debug_login()
