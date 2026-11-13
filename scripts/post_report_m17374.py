#!/usr/bin/env python3
"""Post the m17374 report to the platform."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]

# Login
login_url = "http://127.0.0.1:8010/api/v1/auth/login"
resp = requests.post(login_url, json={"username": "eco_admin", "password": "EcoTest!2026pass"})
token = resp.json().get("token") or resp.json().get("access_token")
print(f"Login OK, token: {token[:20]}...")

# Load artifacts
fig_path = REPO / "runs" / "m17374_irregular_frequency_bessel" / "report_figure.png"
with open(fig_path, "rb") as f:
    fig_base64 = base64.b64encode(f.read()).decode()

# Build report
report = {
    "title": "CASE-BEM-IRREG-FREQ-CYL: Irregular-Frequency Bessel-Zero Prediction (m17374)",
    "summary": (
        "TensorBEM irregular-frequency predictor validated against analytical Bessel-zero references "
        "for a surface-piercing circular cylinder. All configurations pass the 2% tolerance. "
        "Error decreases from 0.32% (224 panels) to 0.02% (2176 panels) with mesh refinement. "
        "Module exported in tensorbem.__init__ for library use. Integration caller demonstrates "
        "CHIEF-point sanitisation in the solver pipeline."
    ),
    "scenario_key": "CASE-BEM-IRREG-FREQ-CYL",
    "mission_id": 17374,
    "tags": ["TensorBEM", "irregular_frequency", "bessel_zero", "verification", "m17374"],
    "markdown": """# CASE-BEM-IRREG-FREQ-CYL: Irregular-Frequency Bessel-Zero Prediction

## 1. Case Introduction

**Physical background**: For a surface-piercing hull, the exterior Neumann boundary-integral equation
loses uniqueness at discrete wavenumbers that coincide with the eigenvalues of the interior Dirichlet
problem. These irregular frequencies are predicted from the waterplane Bessel zeros:

    ν_{m,n} = j_{m,n} / R_wp

where j_{m,n} are zeros of Bessel functions J_m and R_wp is the equivalent waterplane radius.

**Geometry**: Wall-sided surface-piercing circular cylinder, R = 1 m, T = 0.5 m (T/R = 0.5).

**Reference**: Abramowitz & Stegun, Handbook of Mathematical Functions, Table 9.5:
- j₀,₁ = 2.404825557695773
- j₁,₁ = 3.831705970207512
- j₂,₁ = 5.135622301840683

**Tolerance**: < 2%

| Quantity | Value |
|---|---|
| Radius R | 1.0 m |
| Draft T | 0.5 m |
| Draft ratio T/R | 0.5 |
| Gravity | 9.80665 m/s² |
| Mesh range | 224–2176 panels |

## 2. Method

The TensorBEM `predict_irregular_frequencies` function:
1. Computes the equivalent waterplane radius from the mesh waterline polygon
2. Scales the Bessel zeros by R_wp to get predicted wavenumbers
3. Converts to frequencies via ω = √(gν)

Three verification routes (from m16777):
- **Route A**: Rigid-lid interior eigenvalue (what the formula solves exactly)
- **Route B**: Free-surface-consistent interior eigenvalue (exact reference for cylinder)
- **Route C**: Empirical BEM sweep (where the code actually breaks down)

## 3. Results

### 3.1 Predicted vs Reference Wavenumbers

| Mesh | Panels | R_wp (m) | Max ν error (%) | Pass (<2%) |
|---|---|---|---|---|
| Coarse | 224 | 0.9968 | 0.3222 | ✓ |
| Medium | 704 | 0.9992 | 0.0804 | ✓ |
| Fine | 2176 | 0.9998 | 0.0201 | ✓ |
| R=2, T=2 | 704 | 1.9984 | 0.0804 | ✓ |

### 3.2 Bessel-Zero Comparison (Fine Mesh, 2176 panels)

| Mode | Reference ν | Predicted ν | Error (%) |
|---|---|---|---|
| j₀,₁ (axisymmetric) | 2.4048 | 2.4053 | 0.0201 |
| j₁,₁ (dipole) | 3.8317 | 3.8325 | 0.0201 |
| j₂,₁ (quadrupole) | 5.1356 | 5.1367 | 0.0201 |

### 3.3 Convergence

Error decreases monotonically with mesh refinement:
- 224 panels → 0.32%
- 704 panels → 0.08%
- 2176 panels → 0.02%

## 4. Figures

![Bessel-zero comparison and convergence](figures/report_figure.png)

Left: Predicted vs reference wavenumbers (fine mesh, 2176 panels).
Right: Convergence of prediction error vs panel count.

## 5. Notes

- The waterplane radius error (polygon approximation of circle) dominates the prediction error
- Error scales as O((2π/n_θ)²) for the polygon area approximation
- The predictor is exact for a true circular waterplane (machine precision)
- CHIEF-point sanitisation is demonstrated in the integration caller

## 6. Error Judgement

| Key quantity | Reference | Measured | Error | < 2%? |
|---|---|---|---|---|
| ν (j₀,₁), coarse mesh | 2.4048 | 2.4126 | 0.32% | ✓ |
| ν (j₀,₁), fine mesh | 2.4048 | 2.4053 | 0.02% | ✓ |
| R_wp, fine mesh | 1.0000 | 0.9998 | 0.02% | ✓ |
| All modes, all configs | — | — | max 0.32% | ✓ |

**Verdict: PASS — all predicted frequencies within 2% of analytical Bessel-zero reference.**
""",
    "figures": [
        {
            "name": "bessel_zero_comparison_and_convergence",
            "caption": "Left: Predicted vs reference wavenumbers (fine mesh). Right: Convergence of prediction error vs panel count.",
            "png_base64": fig_base64,
        }
    ],
}

# Post report
report_url = "http://127.0.0.1:8010/api/v1/tensor/reports"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
resp = requests.post(report_url, json=report, headers=headers)
print(f"Report post status: {resp.status_code}")
print(f"Response: {resp.text[:500]}")
