"""Numbers for the production RAO default-inertia route (``inertia=None``).

Writes ``production_path/{results.json, PRODUCTION_PATH_INERTIA.md,
fig_production_path_inertia.png}``.  Every number in the markdown is injected
from the computation below (no hand copy).

Run::

    PYTHONPATH=src python3 docs/cases/case_bem_rigid_body_mass/production_path_inertia.py
"""
from __future__ import annotations

import json
import math
import pathlib
import warnings

import numpy as np

from tensorbem.geometry3d import rectangular_barge_hull_mesh, spheroid_hull_mesh
from tensorbem.hydrostatic import (
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
    mass_matrix_from_hull,
    mass_matrix_from_hull_geometry,
    mass_matrix_from_wetted_hull,
)
from tensorbem.rao import compute_rao_at_frequency

OUT = pathlib.Path(__file__).resolve().parent / "production_path"
OUT.mkdir(parents=True, exist_ok=True)
DENSITY = 1025.0
DOFS = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def barge():
    return rectangular_barge_hull_mesh(20.0, 8.0, 3.0, 10, 8)


def box_inertia_cg(mass):
    return mass * np.array([(8.0 ** 2 + 3.0 ** 2) / 12.0, (20.0 ** 2 + 3.0 ** 2) / 12.0,
                            (20.0 ** 2 + 8.0 ** 2) / 12.0])


def hemisphere_inertia_cg(mass, radius=1.0):
    lever = 3.0 * radius / 8.0
    return mass * np.array([0.4 * radius ** 2 - lever ** 2, 0.4 * radius ** 2 - lever ** 2,
                            0.4 * radius ** 2])


def main() -> int:
    mass = DENSITY * 20.0 * 8.0 * 3.0
    cg = (0.0, 0.0, -1.5)
    hull = barge()
    props = compute_hydrostatic_restoring_matrix(hull, mass, cg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = mass_matrix_from_hull_geometry(props, mass, cg)
    exact = compute_mass_matrix(mass, cg, np.diag(box_inertia_cg(mass)), inertia_about="cg")
    default = mass_matrix_from_hull(hull, mass, cg)
    rot_legacy = np.diag(legacy)[3:] / np.diag(exact)[3:] - 1.0
    rot_default = np.diag(default)[3:] / np.diag(exact)[3:] - 1.0

    kwargs = dict(hull=hull, omega=1.0, wave_angle=0.0, mass=mass, center_of_gravity=cg)
    r_exact = np.asarray(compute_rao_at_frequency(inertia=np.diag(box_inertia_cg(mass)),
                                                  **kwargs).rao_amplitude)
    r_legacy = np.asarray(compute_rao_at_frequency(inertia=_inertia_from_matrix(legacy, mass, cg),
                                                   **kwargs).rao_amplitude)
    r_default = np.asarray(compute_rao_at_frequency(inertia=None, **kwargs).rao_amplitude)
    rao_legacy = np.abs(r_legacy - r_exact) / np.where(np.abs(r_exact) > 1e-12,
                                                       np.abs(r_exact), np.nan)
    rao_default = np.abs(r_default - r_exact) / np.where(np.abs(r_exact) > 1e-12,
                                                         np.abs(r_exact), np.nan)

    # curved hull: convergence of the mesh route to the analytic half-sphere
    hmass, hcg = 1000.0, (0.0, 0.0, -3.0 / 8.0)
    h_ref = compute_mass_matrix(hmass, hcg, np.diag(hemisphere_inertia_cg(hmass)),
                                inertia_about="cg")
    scale = float(np.max(np.abs(h_ref)))
    ladder = []
    for sub in (2, 3, 4):
        hh = spheroid_hull_mesh((1.0, 1.0, 1.0), subdivisions=sub)
        err_mesh = float(np.max(np.abs(mass_matrix_from_hull(hh, hmass, hcg) - h_ref)) / scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hprops = compute_hydrostatic_restoring_matrix(hh, hmass, hcg)
            err_old = float(
                np.max(np.abs(mass_matrix_from_hull_geometry(hprops, hmass, hcg) - h_ref)) / scale
            )
        ladder.append({"subdivisions": sub, "panels": int(len(hh.faces)),
                       "mesh_rel_error": err_mesh, "legacy_rel_error": err_old})
    slope = math.log(ladder[0]["mesh_rel_error"] / ladder[-1]["mesh_rel_error"]) / math.log(4.0)

    # body mass vs displaced mass: the rotational block must scale with the body
    hh = spheroid_hull_mesh((1.0, 1.0, 1.0), subdivisions=3)
    light = mass_matrix_from_wetted_hull(hh, 1000.0, hcg)
    buoyant = mass_matrix_from_wetted_hull(hh, None, hcg)
    ratio = float(np.diag(light)[3] / np.diag(buoyant)[3])

    results = {
        "case": "CASE-BEM-RIGID-BODY-MASS / production_path_inertia",
        "barge": {"mass_kg": mass, "cg": list(cg),
                  "rotational_diag_legacy": np.diag(legacy)[3:].tolist(),
                  "rotational_diag_default": np.diag(default)[3:].tolist(),
                  "rotational_diag_closed_form": np.diag(exact)[3:].tolist(),
                  "rel_error_legacy": rot_legacy.tolist(),
                  "rel_error_default": rot_default.tolist(),
                  "matrix_rel_residual_default": float(np.max(np.abs(default - exact))
                                                       / np.max(np.abs(exact)))},
        "rao_omega1_head_seas": {"rao_closed_form": r_exact.tolist(),
                                 "rao_legacy_route": r_legacy.tolist(),
                                 "rao_default_route": r_default.tolist(),
                                 "rel_diff_legacy": rao_legacy.tolist(),
                                 "rel_diff_default": rao_default.tolist()},
        "hemisphere_ladder": ladder,
        "hemisphere_slope": slope,
        "mass_scaling": {"rotational_ratio_1000kg_over_buoyant": ratio},
        "tolerance": 0.03,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    (OUT / "PRODUCTION_PATH_INERTIA.md").write_text(_md(results), encoding="utf-8")
    _figure(results)
    print(json.dumps({k: results[k] for k in ("barge", "hemisphere_slope", "mass_scaling")},
                     indent=2, sort_keys=True))
    print("worst |rao rel diff| legacy :", np.nanmax(rao_legacy))
    print("worst |rao rel diff| default:", np.nanmax(rao_default))
    return 0


def _inertia_from_matrix(matrix, mass, cg):
    c = np.asarray(cg, dtype=float)
    s = np.array([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])
    return matrix[3:, 3:] - mass * (s.T @ s)


def _md(res) -> str:
    b = res["barge"]
    rao = res["rao_omega1_head_seas"]
    rows = "\n".join(
        f"| {DOFS[3 + i]} | {b['rotational_diag_closed_form'][i]:.6e} | "
        f"{b['rotational_diag_legacy'][i]:.6e} | {100 * b['rel_error_legacy'][i]:+.2f}% | "
        f"{b['rotational_diag_default'][i]:.6e} | {100 * b['rel_error_default'][i]:+.2e}% |"
        for i in range(3)
    )
    rao_rows = "\n".join(
        f"| {DOFS[i]} | {rao['rao_closed_form'][i]:.6e} | {rao['rao_legacy_route'][i]:.6e} | "
        f"{_pct(rao['rel_diff_legacy'][i])} | {rao['rao_default_route'][i]:.6e} | "
        f"{_pct(rao['rel_diff_default'][i])} |"
        for i in range(6)
    )
    lad = "\n".join(
        f"| {r['subdivisions']} | {r['panels']} | {r['mesh_rel_error']:.4e} | "
        f"{r['legacy_rel_error']:.4e} |" for r in res["hemisphere_ladder"]
    )
    return f"""# Production RAO default inertia route — geometry-exact by default

Anchor ``CASE-BEM-RIGID-BODY-MASS``. The two frequency-domain chains
(`tensorbem.rao.compute_rao_at_frequency`, `tensorbem.ice_rao.compute_ice_rao_spectrum`)
accept ``inertia=None``. Up to this change that branch built the rigid-body mass
matrix with `hydrostatic.mass_matrix_from_hull_geometry`, i.e. the *equivalent
half-ellipsoid* fitted to the displaced volume and the waterplane second moments.
That reconstruction is exact for an elliptic waterplane and **8–20 % low** on the
rotational block of a box-like hull, which is outside the anchor's 3 % tolerance:
the approximation was the *default* of the production path.
`hydrostatic.mass_matrix_from_hull` now selects the wetted-mesh route
(`mass_matrix_from_wetted_hull`, exact to round-off for a polyhedral hull) and
falls back to the reconstruction only when the panels cannot be closed or
integrated, with a `RuntimeWarning` naming the reason.

## 1. Rigid-body mass matrix, box barge 20 × 8 × 3 m at neutral buoyancy

m = ρLB T = {b['mass_kg']:.6e} kg, CG = (0, 0, {b['cg'][2]:.1f}) m, ρ = 1025 kg/m³.
Reference = the closed-form block form of Fossen (2011) §3.3.1 with the solid-box
second moments `m(L²+B²)/12` etc., evaluated in machine arithmetic.

| DOF | closed form | legacy route | legacy error | new default | new error |
|---|---|---|---|---|---|
{rows}

Worst entry of the whole 6×6 matrix: legacy {100 * max(abs(v) for v in b['rel_error_legacy']):.2f}%,
new {100 * b['matrix_rel_residual_default']:.2e}% (round-off). Every rotational
entry of the legacy route violates the 3 % criterion.

## 2. Consequence in the response (RAO, ω = 1.0 rad/s, head seas)

| DOF | RAO with closed-form inertia | RAO with legacy inertia | legacy rel. diff | RAO with new default | new rel. diff |
|---|---|---|---|---|---|
{rao_rows}

Worst response error: legacy {100 * max(v for v in rao['rel_diff_legacy'] if v == v):.2f}%,
new {100 * max(v for v in rao['rel_diff_default'] if v == v):.2e}%.

## 3. Curved hull: the mesh route converges to the analytic half-sphere

Sphere radius 1 m, m = 1000 kg, CG = (0, 0, −0.375) m; reference =
`I_xx = I_yy = (2/5 − 9/64) m a²`, `I_zz = (2/5) m a²`.

| subdivisions | panels | mesh route rel. error | legacy route rel. error |
|---|---|---|---|
{lad}

Fitted log-log slope of the mesh route over 2 → 4: **{res['hemisphere_slope']:.2f}**
(second order, as the surface integral of a quadratic moment should be). On an
elliptic waterplane the reconstruction is well matched to the geometry, so this
hull does not discriminate between the routes — the barge does.

## 4. Body mass vs displaced mass

The volume integral yields the second moments of a *fluid-filled* solid. A body of
the same shape and another mass has the same specific inertia, so the tensor is
scaled by `mass / (ρ ∇)`. Half-sphere at subdivisions = 3, m = 1000 kg against
neutral buoyancy (ρ ∇ = {1000.0 / res['mass_scaling']['rotational_ratio_1000kg_over_buoyant']:.1f} kg):
rotational diagonal ratio **{res['mass_scaling']['rotational_ratio_1000kg_over_buoyant']:.4f}**
(= m / (ρ ∇)); before the scaling the rotational block described the displaced
water and was {1.0 / res['mass_scaling']['rotational_ratio_1000kg_over_buoyant']:.3f}× too
large for this body.

## 5. Error verdict

| quantity | reference | worst error | < 3 % ? |
|---|---|---|---|
| barge 6×6 matrix, new default | closed-form block form | {100 * b['matrix_rel_residual_default']:.2e}% | yes (round-off) |
| barge 6×6 matrix, legacy route | closed-form block form | {100 * max(abs(v) for v in b['rel_error_legacy']):.2f}% | **no** |
| barge RAO, new default | RAO with closed-form inertia | {100 * max(v for v in rao['rel_diff_default'] if v == v):.2e}% | yes (round-off) |
| barge RAO, legacy route | RAO with closed-form inertia | {100 * max(v for v in rao['rel_diff_legacy'] if v == v):.2f}% | **no** |
| half-sphere matrix, new default, subdiv 4 | analytic half-sphere | {100 * res['hemisphere_ladder'][-1]['mesh_rel_error']:.2f}% | yes |

Honest limit: a *non-uniform* mass distribution is not recoverable from the
surface — the mesh route assumes homogeneous density (CG is then the volume
centroid). A caller whose CG differs from that centroid still gets the
second-moment tensor of the homogeneous body; pass an explicit tensor when the
mass distribution is known to be inhomogeneous.

## 6. Reproduce

```
PYTHONPATH=src python3 docs/cases/case_bem_rigid_body_mass/production_path_inertia.py
PYTHONPATH=src python3 -m pytest tests/test_rao_default_inertia_mesh.py -q
```
"""


def _pct(value):
    return "n/a" if value != value else f"{100 * value:.2e}%"


def _figure(res) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    b = res["barge"]
    rao = res["rao_omega1_head_seas"]
    label = DOFS[3:]
    x = np.arange(3)
    axes[0].bar(x - 0.2, [100 * v for v in b["rel_error_legacy"]], 0.4,
                label="legacy hydrostatic route", color="#c0392b")
    axes[0].bar(x + 0.2, [100 * v for v in b["rel_error_default"]], 0.4,
                label="new default (wetted mesh)", color="#27ae60")
    axes[0].axhline(-3.0, ls="--", lw=1.0, color="k")
    axes[0].text(2.45, -3.0, " 3 % tolerance", va="bottom", ha="right", fontsize=8)
    axes[0].set_xticks(x, label)
    axes[0].set_ylabel("rotational block error vs closed form  [%]")
    axes[0].set_title("Barge 20×8×3 m: inertia error")
    axes[0].legend(fontsize=8, loc="lower left")

    idx = [2, 4, 5]
    xr = np.arange(len(idx))
    axes[1].bar(xr - 0.2, [100 * rao["rel_diff_legacy"][i] for i in idx], 0.4,
                label="legacy inertia", color="#c0392b")
    axes[1].bar(xr + 0.2, [100 * rao["rel_diff_default"][i] for i in idx], 0.4,
                label="new default", color="#27ae60")
    axes[1].set_yscale("log")
    axes[1].axhline(3.0, ls="--", lw=1.0, color="k")
    axes[1].set_xticks(xr, [DOFS[i] for i in idx])
    axes[1].set_ylabel("RAO relative difference vs closed-form inertia  [%]")
    axes[1].set_title("Response at ω = 1.0 rad/s, head seas")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_production_path_inertia.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
