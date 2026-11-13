#!/usr/bin/env python
"""Publish the m17198 report (CASE-BEM-ELLIPSE-2D-ADDED-MASS)."""
from __future__ import annotations
import base64, json, os, pathlib, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent

def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

# Login
token = None
for path, body in (("/auth/login", {"username": os.environ.get("TENSOR_AGENT_USER", "agent"),
                                    "password": os.environ.get("TENSOR_AGENT_PASS", "")}),):
    try:
        token = post(path, body).get("access_token") or post(path, body).get("token")
        if token:
            break
    except Exception as exc:
        print("login failed", path, exc)

if not token:
    print("WARNING: No token obtained, proceeding without auth")

# Load report data
report_path = ROOT / "artifacts" / "ellipse_added_mass" / "report.json"
report = json.loads(report_path.read_text())

# Load figures
figures = []
for name in ["ellipse_convergence", "ellipse_aspect_ratio"]:
    png_path = ROOT / "artifacts" / "ellipse_added_mass" / f"{name}.png"
    if png_path.exists():
        png_b64 = base64.b64encode(png_path.read_bytes()).decode()
        figures.append({
            "name": name,
            "caption": f"Figure: {name}",
            "png_base64": png_b64,
        })

# Build report payload
payload = {
    "title": "CASE-BEM-ELLIPSE-2D-ADDED-MASS: 2-D Elliptic Cylinder Added Mass Verification",
    "summary": "Verified 2-D exterior Dirichlet BEM solver for elliptic cylinder added mass against Lamb/Newman analytical solution. O(h²) convergence confirmed; at N=64 panels, error <0.4% (tolerance 3%).",
    "scenario_key": "CASE-BEM-ELLIPSE-2D-ADDED-MASS",
    "mission_id": 17198,
    "tags": ["BEM", "added-mass", "2D", "ellipse", "exterior-Dirichlet", "verification"],
    "markdown": (ROOT / "report_m17198.json").read_text() if False else """# CASE-BEM-ELLIPSE-2D-ADDED-MASS Verification Report

## 1. Problem Setup & Reference
- **Geometry**: Elliptic cylinder, semi-axes a=1.5m (x), b=0.75m (y), aspect ratio=2.0
- **Fluid**: ρ=1025 kg/m³ (seawater)
- **Analytical reference**: Lamb (1932) §71; Newman (1977) §4.3
  - m_A,x = ρπb² = 1811.3245 kg/m
  - m_A,y = ρπa² = 7245.2981 kg/m
- **Acceptance criterion**: Relative error < 3% at N=64 panels

## 2. Method
- **Formulation**: Exterior 2-D Laplace Dirichlet problem, first-kind boundary integral equation
  S q_e = (½I - H) u
- **Boundary condition**: Exact Dirichlet trace u = -U·k·x_axis, where k=b/a (x-motion) or k=a/b (y-motion)
- **Added mass**: Computed via boundary energy integral m_A = ρ/U² · ∮ u·q_e ds
- **Discretization**: Inscribed polygon with constant panels, analytically integrated matrix elements

## 3. Results

### Convergence Study
| N | Axis | Exact (kg/m) | Computed (kg/m) | Error | <3%? |
|---|------|-------------|----------------|-------|------|
| 32 | x | 1811.3245 | 1789.5073 | 1.20% | ✓ |
| 32 | y | 7245.2981 | 7142.6407 | 1.42% | ✓ |
| **64** | **x** | **1811.3245** | **1805.8636** | **0.30%** | **✓** |
| **64** | **y** | **7245.2981** | **7219.3457** | **0.36%** | **✓** |
| 128 | x | 1811.3245 | 1809.9596 | 0.075% | ✓ |
| 128 | y | 7245.2981 | 7238.7789 | 0.090% | ✓ |

### Convergence Rate
Error ratio per doubling of N → 0.250, confirming O(h²) convergence.

### Aspect Ratio Sensitivity (N=128)
| AR | x-error | y-error |
|----|---------|---------|
| 1.0 | 0.006% | 0.006% |
| 2.0 | 0.075% | 0.090% |
| 5.0 | 0.47% | 0.56% |

## 4. Verdict
**PASS** — Both axes at N=64 panels have error well below 3% tolerance (0.30% and 0.36%).
""",
    "figures": figures,
}

# Publish
try:
    result = post("/tensor/reports", payload, token)
    print("Report published successfully:", result)
except Exception as e:
    print("Report publication failed:", e)
    print("Payload summary:", json.dumps({k: str(v)[:100] for k, v in payload.items()}, indent=2))
