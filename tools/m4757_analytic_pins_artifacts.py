"""Reproduce the m4757 analytic-pin numbers and figures (zero hand-copied values).

Runs the same computations as ``tests/test_core_api_analytic_pins.py`` and emits

* ``outputs/m4757/analytic_pins.json``  -- machine-readable numbers,
* ``outputs/m4757/untested_public_api.json`` -- the function-level coverage scan,
* ``outputs/m4757/analytic_pins.png``   -- convergence / decay figures,
* ``outputs/m4757/report_tables.md``    -- markdown tables injected into the report.

Usage::

    PYTHONPATH=src python3 tools/m4757_analytic_pins_artifacts.py
"""

from __future__ import annotations

import ast
import glob
import json
import math
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem import finite_depth as fd  # noqa: E402
from tensorbem import finite_depth3d as fd3  # noqa: E402
from tensorbem import free_surface_solver3d as fs3  # noqa: E402
from tensorbem import geometry3d as g3  # noqa: E402
from tensorbem.conditioning import condition_number_trend  # noqa: E402
from tensorbem.geometry import PanelMesh  # noqa: E402
from tensorbem.helmholtz import assemble_helmholtz_double_layer  # noqa: E402

OUT = ROOT / "outputs" / "m4757"
GRAVITY = 9.80665
GAUSS_X, GAUSS_W = np.polynomial.legendre.leggauss(200)


def quadrature(f, a: float, b: float) -> float:
    grid = 0.5 * (b - a) * GAUSS_X + 0.5 * (a + b)
    return float(0.5 * (b - a) * np.sum(GAUSS_W * f(grid)))


def wetted(hull):
    v = hull.vertices
    v0, v1, v2 = v[hull.faces[:, 0]], v[hull.faces[:, 1]], v[hull.faces[:, 2]]
    volume = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)
    area = float(0.5 * np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)))
    return volume, area


def sphere_hull_rows() -> list[dict]:
    rows = []
    for radius, z0 in ((1.0, -0.5), (2.0, -0.75)):
        height = radius - z0
        v_ref = math.pi * height**2 * (3.0 * radius - height) / 3.0
        a_ref = 2.0 * math.pi * radius * height
        r_wl = math.sqrt(radius**2 - z0**2)
        for sub in (2, 3, 4, 5):
            hull = g3.floating_sphere_hull_mesh(
                sub, radius=radius, center=(0.0, 0.0, z0)
            )
            volume, area = wetted(hull)
            wl = hull.vertices[np.abs(hull.vertices[:, 2]) <= 1.0e-12]
            radii = np.hypot(wl[:, 0], wl[:, 1])
            rows.append(
                {
                    "radius": radius,
                    "centre_z": z0,
                    "subdivisions": sub,
                    "faces": int(hull.faces.shape[0]),
                    "volume": volume,
                    "volume_ref": v_ref,
                    "volume_rel_error": volume / v_ref - 1.0,
                    "area": area,
                    "area_ref": a_ref,
                    "area_rel_error": area / a_ref - 1.0,
                    "waterline_radius": float(np.max(radii)),
                    "waterline_radius_ref": r_wl,
                    "waterline_sagitta": r_wl - float(np.min(radii)),
                }
            )
    return rows


def airy_rows() -> dict:
    nu, amplitude = 0.7, 1.7
    omega = math.sqrt(GRAVITY * nu)
    z = np.array([0.0, -0.4, -1.2, -2.5])
    pts = np.column_stack(
        (np.linspace(-2, 2, 4), np.linspace(-1.5, 1.5, 4), z)
    )
    phi, grad = fs3.incident_airy_wave_3d(pts, nu, amplitude=amplitude)
    analytic = (GRAVITY * amplitude / omega) * np.exp(nu * z)
    return {
        "nu": nu,
        "amplitude": amplitude,
        "free_surface_residual": float(np.max(np.abs(grad[:, 2] - nu * phi))),
        "magnitude_rel_error": float(np.max(np.abs(np.abs(phi) - analytic) / analytic)),
        "depth_decay_table": [
            {
                "z": float(zi),
                "abs_phi": float(a),
                "analytic": float(b),
                "rel_error": float(abs(a - b) / b),
            }
            for zi, a, b in zip(z, np.abs(phi), analytic)
        ],
    }


def mode_norm_rows() -> list[dict]:
    nu, depth = 1.2, 7.0
    mu = fd.evanescent_wavenumbers(nu, depth, 4)
    norms = fd.mode_norms(nu, depth, mu)
    rows = []
    for i, (m, norm) in enumerate(zip(mu, norms)):
        q = quadrature(lambda t, m=m: np.cos(m * (t + depth)) ** 2, -depth, 0.0)
        cross = max(
            abs(
                quadrature(
                    lambda t, m=m, o=o: np.cos(m * (t + depth)) * np.cos(o * (t + depth)),
                    -depth,
                    0.0,
                )
            )
            for o in mu
            if o != m
        )
        rows.append(
            {
                "mode": i,
                "mu": float(m),
                "norm_closed_form": float(norm),
                "norm_quadrature": q,
                "norm_abs_diff": abs(q - float(norm)),
                "max_cross_orthogonality": cross,
                "dispersion_residual": abs(float(m) * math.tan(float(m) * depth) + nu) / nu,
            }
        )
    return rows


def dispersion_rows() -> list[dict]:
    nu, depth = 0.8, 6.0
    bracketed = fd.evanescent_wavenumbers(nu, depth, 6)
    polished = fd3.polish_evanescent_roots(bracketed, nu, depth, steps=3)
    return [
        {
            "mode": i,
            "mu_bracketed": float(b),
            "mu_polished": float(p),
            "residual_before": abs(float(b) * math.tan(float(b) * depth) + nu) / nu,
            "residual_after": abs(float(p) * math.tan(float(p) * depth) + nu) / nu,
        }
        for i, (b, p) in enumerate(zip(bracketed, polished))
    ]


def deep_water_rows() -> list[dict]:
    nu = 0.5
    source = np.array([[0.0, 0.0, -1.0]])
    theta = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)
    targets = np.column_stack(
        (1.5 * np.cos(theta), 1.5 * np.sin(theta), -0.5 * np.ones_like(theta))
    )
    return [
        {"depth": d, "error": fd3.deep_water_limit_error(targets, source, nu, d)}
        for d in (1.0, 5.0, 10.0, 20.0, 40.0, 80.0)
    ]


def jump_rows() -> list[dict]:
    n = 128
    theta = 2.0 * math.pi * np.arange(n) / n
    mesh = PanelMesh(np.column_stack((np.cos(theta), np.sin(theta))))
    sigma = np.ones(n)
    rows = []
    for k in (0.2, 0.5, 1.0, 2.0):
        for offset in (1.0e-2, 2.0e-2, 5.0e-2):
            centre, normal = mesh.centers[0], mesh.normals[0]
            targets = np.stack((centre + offset * normal, centre - offset * normal))
            values = assemble_helmholtz_double_layer(
                mesh, k, targets, panel_nodes=16
            ) @ sigma
            jump = values[0] - values[1]
            rows.append(
                {
                    "wavenumber": k,
                    "offset": offset,
                    "jump_real": float(jump.real),
                    "jump_imag": float(jump.imag),
                    "jump_rel_error": abs(abs(jump) - 1.0),
                }
            )
    return rows


def conditioning_rows() -> list[dict]:
    trend = condition_number_trend(g3.icosphere, [1, 2], method="direct")
    return [
        {
            "subdivisions": sub,
            "panels": r.mesh_panels,
            "single_layer_condition": r.single_layer_condition,
            "double_layer_condition": r.double_layer_condition,
            "method": r.estimation_method,
        }
        for sub, r in zip((1, 2), trend)
    ]


def api_scan() -> dict:
    """Function-level scan: public functions no test file mentions."""
    funcs: dict[str, list[str]] = {}
    for path in glob.glob(str(ROOT / "src" / "tensorbem" / "**" / "*.py"), recursive=True):
        if os.path.basename(path) == "__init__.py":
            continue
        module = os.path.relpath(path, ROOT / "src").replace("/", ".")[:-3]
        tree = ast.parse(open(path).read())
        names = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        ]
        if names:
            funcs[module] = names

    referenced = set()
    for path in glob.glob(str(ROOT / "tests" / "**" / "*.py"), recursive=True):
        for node in ast.walk(ast.parse(open(path).read())):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    referenced.add(alias.name)

    pinned = [
        "tensorbem.geometry3d.floating_sphere_hull_mesh",
        "tensorbem.free_surface_solver3d.incident_airy_wave_3d",
        "tensorbem.free_surface_solver3d.rigid_body_normals_3d",
        "tensorbem.finite_depth.mode_norms",
        "tensorbem.finite_depth3d.vertical_mode",
        "tensorbem.finite_depth3d.polish_evanescent_roots",
        "tensorbem.finite_depth3d.deep_water_limit_error",
        "tensorbem.helmholtz.assemble_helmholtz_double_layer",
        "tensorbem.helmholtz.default_chief_points",
        "tensorbem.conditioning.condition_number_trend",
    ]
    unreferenced = [
        f"{module}.{name}"
        for module, names in sorted(funcs.items())
        for name in names
        if name not in referenced
    ]
    return {
        "public_functions": sum(len(v) for v in funcs.values()),
        "modules": len(funcs),
        "unreferenced_total": len(unreferenced),
        "unreferenced_maxline": sorted(
            [u for u in unreferenced if ".benchmarks." not in u]
        ),
        "unreferenced_benchmarks": len([u for u in unreferenced if ".benchmarks." in u]),
        "pinned_in_this_mission": pinned,
        "mainline_unreferenced_after": len(
            [u for u in unreferenced if ".benchmarks." not in u]
        ),
    }


def figures(sphere_rows, deep_rows, airy) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax = axes[0]
    for radius, z0, style in ((1.0, -0.5, "-o"), (2.0, -0.75, "-s")):
        rows = [r for r in sphere_rows if r["radius"] == radius]
        subs = [r["subdivisions"] for r in rows]
        ax.semilogy(
            subs,
            [abs(r["volume_rel_error"]) for r in rows],
            style,
            label=f"volume  R={radius}, z0={z0}",
        )
        ax.semilogy(
            subs,
            [abs(r["area_rel_error"]) for r in rows],
            style,
            alpha=0.55,
            label=f"wetted area  R={radius}, z0={z0}",
        )
    ax.set_xlabel("icosphere subdivision level")
    ax.set_ylabel("relative error vs spherical-cap closed form")
    ax.set_title("geometry3d.floating_sphere_hull_mesh")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    depths = [r["depth"] for r in deep_rows]
    errors = [r["error"] for r in deep_rows]
    ax.semilogy(depths, errors, "-o", label="finite-depth vs deep-water kernel")
    ax.set_xlabel("depth h (m),  nu = 0.5")
    ax.set_ylabel("max relative deviation")
    ax.set_title("finite_depth3d.deep_water_limit_error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    zs = [row["z"] for row in airy["depth_decay_table"]]
    ax.semilogy(zs, [row["abs_phi"] for row in airy["depth_decay_table"]], "o", label="|phi_I| (code)")
    ax.semilogy(
        zs,
        [row["analytic"] for row in airy["depth_decay_table"]],
        "-",
        label=r"$(gA/\omega)\,e^{\nu z}$",
    )
    ax.set_xlabel("z (m)")
    ax.set_ylabel("|phi| (m^2/s)")
    ax.set_title("incident_airy_wave_3d deep-water decay")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    target = OUT / "analytic_pins.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return str(target)


def markdown(table_rows: dict) -> str:
    airy = table_rows["airy"]
    out = []
    out.append("### A. 球冠几何（`geometry3d.floating_sphere_hull_mesh` vs 闭式解）\n")
    out.append("| R | z0 | 细分 | 面数 | V | V_ref | V 相对误差 | A | A_ref | A 相对误差 | 水线弦高 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in table_rows["sphere"]:
        out.append(
            "| {radius:g} | {centre_z:g} | {subdivisions} | {faces} | {volume:.6f} | {volume_ref:.6f} | "
            "{volume_rel_error:+.3e} | {area:.6f} | {area_ref:.6f} | {area_rel_error:+.3e} | {waterline_sagitta:.2e} |".format(**r)
        )
    out.append("\n### B. 有限水深本征模正交归一（`finite_depth.mode_norms` vs 200 点 Gauss 积分）\n")
    out.append("| 模 | mu | 闭式范数 | 数值积分 | 绝对差 | 最大交叉内积 | 色散残差 |")
    out.append("|---|---|---|---|---|---|---|")
    for r in table_rows["modes"]:
        out.append(
            "| {mode} | {mu:.9f} | {norm_closed_form:.12f} | {norm_quadrature:.12f} | {norm_abs_diff:.2e} | "
            "{max_cross_orthogonality:.2e} | {dispersion_residual:.2e} |".format(**r)
        )
    out.append("\n### C. 消逝根 Newton 抛光（`finite_depth3d.polish_evanescent_roots`）\n")
    out.append("| 模 | mu（二分） | mu（抛光） | 残差前 | 残差后 |")
    out.append("|---|---|---|---|---|")
    for r in table_rows["dispersion"]:
        out.append(
            "| {mode} | {mu_bracketed:.12f} | {mu_polished:.12f} | {residual_before:.2e} | {residual_after:.2e} |".format(**r)
        )
    out.append("\n### D. 深水极限诊断（`finite_depth3d.deep_water_limit_error`，nu = 0.5）\n")
    out.append("| 水深 h | 相对偏差 |")
    out.append("|---|---|")
    for r in table_rows["deep"]:
        out.append("| {depth:g} | {error:.3e} |".format(**r))
    out.append("\n### E. Helmholtz 双层位跳变（`helmholtz.assemble_helmholtz_double_layer`，128 面元，sigma=1）\n")
    out.append("| k | 探针偏移 | Re(u_out-u_in) | Im | |\\|jump\\|-1| |")
    out.append("|---|---|---|---|---|")
    for r in table_rows["jump"]:
        out.append(
            "| {wavenumber:g} | {offset:g} | {jump_real:+.6f} | {jump_imag:+.2e} | {jump_rel_error:.2e} |".format(**r)
        )
    out.append("\n### F. 入射 Airy 波（`free_surface_solver3d.incident_airy_wave_3d`，nu=0.7, A=1.7）\n")
    out.append("| z | \\|phi_I\\| | 解析值 | 相对误差 |")
    out.append("|---|---|---|---|")
    for r in airy["depth_decay_table"]:
        out.append("| {z:g} | {abs_phi:.9e} | {analytic:.9e} | {rel_error:.2e} |".format(**r))
    out.append(
        "\n自由面条件残差 max|dphi/dz - nu*phi| = {:.3e}\n".format(airy["free_surface_residual"])
    )
    out.append("\n### G. 条件数趋势（`conditioning.condition_number_trend`）\n")
    out.append("| 细分 | 面数 | kappa(S) | kappa(1/2 I - H) | 方法 |")
    out.append("|---|---|---|---|---|")
    for r in table_rows["conditioning"]:
        out.append(
            "| {subdivisions} | {panels} | {single_layer_condition:.5g} | {double_layer_condition:.5g} | {method} |".format(**r)
        )
    return "\n".join(out) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "sphere": sphere_hull_rows(),
        "modes": mode_norm_rows(),
        "dispersion": dispersion_rows(),
        "deep": deep_water_rows(),
        "jump": jump_rows(),
        "conditioning": conditioning_rows(),
    }
    airy = airy_rows()
    scan = api_scan()
    payload = {"tables": tables, "airy": airy, "api_scan": scan}
    (OUT / "analytic_pins.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "untested_public_api.json").write_text(json.dumps(scan, indent=2) + "\n")

    figure_path = None
    try:
        figure_path = figures(tables["sphere"], tables["deep"], airy)
    except Exception as exc:  # pragma: no cover - figure is optional
        print(f"figure skipped: {exc}")
    payload["figure"] = figure_path
    (OUT / "analytic_pins.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "report_tables.md").write_text(markdown({**tables, "airy": airy}))

    worst = {
        "sphere_volume_max_rel": max(abs(r["volume_rel_error"]) for r in tables["sphere"]),
        "sphere_area_max_rel": max(abs(r["area_rel_error"]) for r in tables["sphere"]),
        "mode_norm_max_abs_diff": max(r["norm_abs_diff"] for r in tables["modes"]),
        "cross_orthogonality_max": max(r["max_cross_orthogonality"] for r in tables["modes"]),
        "dispersion_residual_max_after": max(r["residual_after"] for r in tables["dispersion"]),
        "deep_water_error_at_h20": [r for r in tables["deep"] if r["depth"] == 20.0][0]["error"],
        "deep_water_error_at_h1": [r for r in tables["deep"] if r["depth"] == 1.0][0]["error"],
        "jump_max_rel_error": max(r["jump_rel_error"] for r in tables["jump"]),
        "airy_magnitude_max_rel": airy["magnitude_rel_error"],
        "mainline_unreferenced_before": scan["mainline_unreferenced_after"] + len(
            scan["pinned_in_this_mission"]
        ),
        "mainline_unreferenced_after": scan["mainline_unreferenced_after"],
    }
    print(json.dumps(worst, indent=2))
    print(f"artifacts -> {OUT}")


if __name__ == "__main__":
    main()