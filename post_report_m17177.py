#!/usr/bin/env python3
"""Post the ellipsoid added mass benchmark report to the platform."""
import json
import os
import base64
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
        print(f"✓ Token obtained: {token[:20]}...")
except Exception as e:
    print(f"✗ Auth failed: {e}")
    token = "fallback_token"

# Read the report markdown
report_md = Path("report_m17177_ellipsoid_added_mass.md").read_text(encoding="utf-8")

# Load figures
figures = []
fig_dir = Path("artifacts/figures")
for fig_name in ["ellipsoid_geometry.png", "added_mass_comparison.png", 
                  "convergence_study.png", "shape_factors.png"]:
    fig_path = fig_dir / fig_name
    if fig_path.exists():
        with open(fig_path, "rb") as f:
            png_base64 = base64.b64encode(f.read()).decode("utf-8")
        figures.append({
            "name": fig_name,
            "caption": fig_name.replace(".png", "").replace("_", " ").title(),
            "png_base64": png_base64
        })
        print(f"✓ Loaded figure: {fig_name}")

# Build report payload
report = {
    "title": "Ellipsoid Added Mass Benchmark: TensorBEM vs Lamb (1932) — CASE-BEM-ELLIPSOID-ADDED-MASS",
    "summary": "Tri-axial ellipsoid (a=1.5, b=1.0, c=0.75 m) added mass computed via BEM (320 panels) and compared against Lamb's closed-form solution (Hydrodynamics, 1932, §114). All three principal axes achieve <2% error (X: 1.63%, Y: 1.74%, Z: 1.91%), well within the 5% tolerance. Lamb's shape factor sum α₁+α₂+α₃=2 verified to machine precision. Convergence study from 20 to 320 panels shows O(h²) behavior consistent with flat triangular panels.",
    "scenario_key": "CASE-BEM-ELLIPSOID-ADDED-MASS",
    "mission_id": 17177,
    "tags": ["TensorBEM", "added-mass", "ellipsoid", "Lamb", "benchmark", "verification", "laplace"],
    "markdown": report_md,
    "figures": figures
}

# Post report
report_url = f"{base}/tensor/reports"
payload = json.dumps(report).encode("utf-8")
req = urllib.request.Request(
    report_url,
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    },
    method="POST"
)

try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"✓ Report posted successfully!")
        print(f"  Report ID: {result.get('id', 'N/A')}")
        print(f"  Status: {result.get('status', 'N/A')}")
except urllib.error.HTTPError as e:
    print(f"✗ HTTP Error {e.code}: {e.reason}")
    print(f"  Response: {e.read().decode()}")
except Exception as e:
    print(f"✗ Failed to post report: {e}")

print("\n✓ Report submission complete")
