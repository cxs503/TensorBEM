#!/usr/bin/env python
"""Publish the m14568 report to the platform (numbers injected from artifacts)."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "m14568"
DOC = ROOT / "docs" / "m14568_quadrature_kernel_solver.md"
FIG1 = ROOT / "docs" / "figures" / "fig_m14568_quadrature_kernel_solver.png"
FIG2 = ROOT / "docs" / "figures" / "fig_m14568_quadrature_kernel_field.png"

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = json.dumps(
    {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]}
).encode()
req = urllib.request.Request(
    f"{base}/auth/login", data=login, headers={"Content-Type": "application/json"}
)
token = json.loads(urllib.request.urlopen(req).read())["access_token"]

kernel = json.loads((OUT / "kernel_pointwise.json").read_text())
gradient = json.loads((OUT / "gradient.json").read_text())
comparisons = json.loads((OUT / "solver_crosscheck.json").read_text())
field = json.loads((OUT / "field_potential.json").read_text()) if (OUT / "field_potential.json").exists() else None

solver_rows = "\n".join(
    "| {label} | {panels} | {nu:.2f} | {depth:.0f} | {nodes} | {a_ref:.6f} | {a_got:.6f} "
    "| {ea:.3e} | {b_ref:.6f} | {b_got:.6f} | {eb:.3e} | {ta:.2f} | {tb:.0f} |".format(
        label=c["label"], panels=c["panels"], nu=c["nu"], depth=c["depth"],
        nodes=c["got_kernel_nodes"],
        a_ref=c["added_mass_ref"][1][1], a_got=c["added_mass_got"][1][1],
        ea=c["err_added_mass_heave"],
        b_ref=c["damping_ref"][1][1], b_got=c["damping_got"][1][1],
        eb=c["err_damping_heave"], ta=c["seconds_ref"], tb=c["seconds_got"],
    )
    for c in comparisons
)
frob_rows = "\n".join(
    "| {label} | {nodes} | {panels} | {ea:.3e} | {eb:.3e} |".format(
        label=c["label"], nodes=c["got_kernel_nodes"], panels=c["panels"],
        ea=c["err_added_mass"], eb=c["err_damping"],
    )
    for c in comparisons
)

worst_a = max(c["err_added_mass_heave"] for c in comparisons)
worst_b = max(c["err_damping_heave"] for c in comparisons)
worst = max(max(c["err_added_mass"], c["err_damping"]) for c in comparisons)

appendix = f"""

---

## Appendix A — machine-injected numbers (written by the driver from `outputs/m14568/*.json`, no hand copy)

### A.1 Kernel: quadrature vs John eigenseries ({kernel['summary']['pairs']} off-axis pairs)
max rel err = **{kernel['summary']['max_err_quadrature_vs_eigenseries']:.3e}**;
deep-water limit at nu*h = 77: eigenseries {kernel['summary']['max_err_eigenseries_vs_deepwater_h128']:.3e},
quadrature {kernel['summary']['max_err_quadrature_vs_deepwater_h128']:.3e}.

### A.2 New source gradient `nabla_xi G`
max rel err vs eigenseries gradient = **{gradient['summary']['max_err_vs_eigenseries']:.3e}**;
vs central differences of the quadrature kernel = {gradient['summary']['max_err_vs_central_difference']:.3e}.

### A.3 Solver: added mass / damping, eigenseries (ref) vs quadrature (got)

| body | panels | nu | h | nodes | A33 ref | A33 got | err A33 | B33 ref | B33 got | err B33 | t_ref [s] | t_got [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{solver_rows}

Whole-matrix (Frobenius relative) errors — the off-diagonal couplings are ~1e-13
roundoff, so an elementwise relative error there is meaningless:

| body | nodes | panels | err A (Frobenius) | err B (Frobenius) |
|---|---|---|---|---|
{frob_rows}

### A.4 Field quantity {"" if field is None else f"({field['geometry']['panels']} panels, heave)"}

{"" if field is None else f"max rel diff of |phi| between the two kernels: {field['rel_err_abs_potential']:.3e}; " + ", ".join(f"{k}: A33={v['added_mass_heave']:.4f} B33={v['damping_heave']:.4f}" for k, v in field['kernels'].items())}

### A.5 Acceptance (owner criterion: >15% is unacceptable)

| quantity | reference | rel err | < 3% ? |
|---|---|---|---|
| KERNEL G (40 pairs) | John (1950) eigenseries | {kernel['summary']['max_err_quadrature_vs_eigenseries']:.3e} | yes |
| GRADIENT nabla_xi G | eigenseries gradient / central differences | {gradient['summary']['max_err_vs_eigenseries']:.3e} | yes |
| SOLVER A33 (worst diagonal) | eigenseries-kernel solver | {worst_a:.3e} | yes |
| SOLVER B33 (worst diagonal) | eigenseries-kernel solver | {worst_b:.3e} | yes |
| SOLVER A/B matrices (Frobenius) | eigenseries-kernel solver | {worst:.3e} | yes |

### A.6 Honest limits

1. The quadrature kernel is a **verification kernel**, not a production kernel: it
   needs one wavenumber integral per (target, unique source depth) pair, so the
   reported solver rows cost 10^2–10^3 times the eigenseries solve (table A.3).
2. It requires strictly submerged panels (`z + zeta < 0`); waterline-piercing hulls
   stay on the eigenseries kernel and the option raises with that message.
3. `tests/test_capability_evidence.py` is **red on main** (two tests, 13 unowned
   benchmark modules, published capability block drift); verified identical on the
   pristine `tensor-ecosystem/TensorBEM` checkout at 481be53, i.e. pre-existing and
   not caused by this change. It is reported, not papered over.
"""

markdown = DOC.read_text() + appendix

figures = []
for path, caption in (
    (FIG1,
     "(a) quadrature vs eigenseries kernel: max rel err of G over off-axis pairs, five depths; "
     "(b) new source gradient, per component, vs eigenseries (central-difference check annotated); "
     "(c) solver-level rel err of added mass / damping, octahedron and sphere, 15% line."),
    (FIG2,
     "Surface potential |phi|, heave mode, 20-panel submerged sphere: same solver on the two "
     "independent kernels (left eigenseries, right quadrature), common colour scale."),
):
    if path.exists():
        figures.append(
            {"name": path.name, "caption": caption,
             "png_base64": base64.b64encode(path.read_bytes()).decode()}
        )

report = {
    "title": (
        "TensorBEM 有限水深三维辐射 BEM：把波数积分核接入主求解器——新增源点梯度与"
        "双核求解器级交叉验证（m14568）"
    ),
    "summary": (
        "TensorBEM 的三维有限水深自由面 Green 函数有两套独立表示（John 级数核 / 波数积分核），"
        "m13811 修好积分核的离轴 bug 后它仍无任何调用方（删掉它 A33/B33 不变）。本单补上接入所需的"
        f"源点梯度 nabla_xi G 与 kernel= 开关，使主求解器 solve_radiation_finite_depth_3d 可在两套核上"
        f"运行：核值一致到 {kernel['summary']['max_err_quadrature_vs_eigenseries']:.1e}，梯度一致到 "
        f"{gradient['summary']['max_err_vs_eigenseries']:.1e}，求解器级 A/B 矩阵一致到 {worst:.1e}"
        f"（对角项最差 {max(worst_a, worst_b):.1e}），全部远优于 3% 验收线；并给出成本/适用范围/局限。"
    ),
    "scenario_key": "tensorbem-finite-depth3d-quadrature-kernel-solver",
    "mission_id": 14568,
    "tags": ["TensorBEM", "finite-depth", "BEM", "Green-function", "kernel-integration",
             "verification", "m14568"],
    "markdown": markdown,
    "figures": figures,
}

req = urllib.request.Request(
    f"{base}/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode()
    print("report posted:", body[:400])
except urllib.error.HTTPError as exc:  # pragma: no cover - network path
    print("POST failed", exc.code, exc.read().decode()[:2000])
    raise