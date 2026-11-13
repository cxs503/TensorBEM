#!/usr/bin/env python3
"""Tiny tensor API helper: python3 api.py <METHOD> <path> [json_body_file]"""
import json, os, sys, requests

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]

def token():
    r = requests.post(f"{BASE}/tensor/auth/login",
                      json={"username": USER, "password": PASS}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

TOK = token()
H = {"Authorization": f"Bearer {TOK}"}

method = sys.argv[1].upper()
path = sys.argv[2]
url = BASE + path
body = None
if len(sys.argv) > 3:
    body = json.load(open(sys.argv[3]))
r = requests.request(method, url, headers=H, json=body, timeout=120)
try:
    out = r.json()
except Exception:
    out = r.text[:2000]
print(json.dumps(out, ensure_ascii=False, indent=1)[:20000])
sys.exit(0 if r.status_code < 400 else 1)
