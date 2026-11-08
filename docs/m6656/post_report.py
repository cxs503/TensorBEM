"""Publish the m6656 report (paper-grade, figures embedded) to the Tensor API.

The markdown is rendered from ``docs/m6656/study.json`` by
``docs/m6656/build_report.py`` -- nothing is hand-copied, and the figure PNGs are
attached from the same directory.

Usage::

    PYTHONPATH=src python3 docs/m6656/post_report.py
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER")
PASS = os.environ.get("TENSOR_AGENT_PASS")

CAPTIONS = {
    "fig1_wigley_cw_panel_vs_michell.png": (
        "Wigley wave resistance: panel-Kochin Cw for the 504/2016/7952-panel meshes against the "
        "closed-form Michell value (left), and the log-log convergence of the panel error in the "
        "panel size h with the fitted order in the legend (right)."
    ),
    "fig2_wigley_kochin_energy_and_convergence.png": (
        "Wigley heave damping: near-field pressure integral (o) and Kochin far-field (x) readings "
        "on the mesh ladder (left) and the successive-difference mesh convergence with its fitted "
        "order (right). The two readings are the same energy flux computed two ways."
    ),
    "fig3_kcs_6dof_excitation_reciprocity.png": (
        "KCS-class hull, 6-DOF exciting-force amplitudes vs lambda/L in head seas (surge/heave/pitch) "
        "and beam seas (sway/roll/yaw): pressure integral, Haskind near-field integral and Kochin "
        "far-field reciprocity; right panel: worst relative offset per wavelength against the 2 % gate."
    ),
    "fig4_kcs_long_wave_fk_limit.png": (
        "Long-wave Froude-Krylov limit: relative offset between the BEM exciting force and the "
        "undisturbed-wave (Froude-Krylov) force as lambda/L grows, per significant DOF."
    ),
    "fig5_kcs_kochin_energy_identity.png": (
        "KCS-class hull radiation damping B33/B55 vs non-dimensional frequency, near field against "
        "Kochin far field (left) and the per-mode energy-identity error against the 2 % gate (right)."
    ),
}


def main() -> int:
    from build_report import load, render

    data = load()
    markdown = render(data)
    markdown += (
        "\n\n## 7. Test evidence\n\n"
        "`PYTHONPATH=src python3 -m pytest tests/test_wigley_kcs_kochin_energy.py -q` -> "
        "13 passed, 1 skipped (the artifact gate) before the artifact existed; it now runs too. "
        "The driver's CLI contract is additionally pinned by "
        "`tests/test_benchmark_cli_contract.py`.\n\n"
        "Pre-existing red gate found while working (not introduced here, not touched): "
        "`tests/test_capability_coverage.py` fails on `tensorbem.tank_free_surface`, a module of "
        "m6452 that no capability row names.\n"
    )

    figures = []
    for name, caption in CAPTIONS.items():
        path = HERE / name
        if not path.exists():
            continue
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )

    crit = data["criteria"]
    summary = (
        "Verified TensorBEM's 3-D free-surface BEM on a Wigley and a KCS-class hull: panel-Kochin "
        f"Cw vs Michell closed form {min(e['cw_relative_error'] for e in crit['a_wigley_cw_vs_michell'].values()):.2%}-"
        f"{max(e['cw_relative_error'] for e in crit['a_wigley_cw_vs_michell'].values()):.2%} (order "
        f"{min(crit['d_panel_convergence_orders'].values()):.2f}-{max(crit['d_panel_convergence_orders'].values()):.2f}); "
        f"Kochin far-field vs near-field damping {crit['c_kochin_energy_wigley']['max_damping_relative_error']:.2%} "
        f"(Wigley) / {crit['c_kochin_energy_kcs']['max_damping_relative_error']:.2%} (KCS-class); "
        f"KCS-class 6-DOF exciting force pressure-vs-Haskind {crit['b_kcs_reciprocity']['max_pressure_vs_haskind']:.2%}. "
        "KCS EFD comparison reported WITHHELD (no Tokyo 2003 table offline, idealised hull)."
    )

    payload = {
        "title": (
            "TensorBEM radiation/diffraction verification: Wigley panel-Kochin Cw vs Michell, "
            "Kochin far-field energy identity, and KCS-class 6-DOF exciting-force reciprocity"
        ),
        "summary": summary,
        "scenario_key": "wigley_kcs_kochin_energy",
        "mission_id": "6656",
        "tags": ["TensorBEM", "radiation", "diffraction", "Kochin", "Wigley", "KCS", "verification"],
        "markdown": markdown,
        "figures": figures,
    }

    data_login = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(
        BASE + "/auth/login", data=data_login, headers={"Content-Type": "application/json"}
    )
    token = json.loads(urllib.request.urlopen(req).read())["access_token"]
    req = urllib.request.Request(
        BASE + "/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    try:
        resp = urllib.request.urlopen(req)
        print(resp.status, resp.read()[:400].decode())
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read()[:600].decode())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
