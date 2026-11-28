#!/usr/bin/env python3
import json
import requests

url = "http://127.0.0.1:8010/api/v1/tensor/reports"
login_url = "http://127.0.0.1:8010/api/v1/auth/login"

# Login
resp = requests.post(login_url, json={"username": "eco_admin", "password": "EcoTest!2026pass"})
token = resp.json()["access_token"]

# POST report
with open("report_final.json", "r") as f:
    report = json.load(f)

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(url, json=report, headers=headers)
print(f"Report Status: {resp.status_code}")
print(resp.text)
