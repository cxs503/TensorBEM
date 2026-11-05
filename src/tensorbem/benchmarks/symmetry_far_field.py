"""Far-field post-processing of the symmetry-reduced 3-D BEM: verification driver.

Closes the limitation that ``tensorbem.symmetry3d`` used to document
("far-field post-processing is not folded here; those helpers expect a
full-surface potential").  Generates every number quoted in
``docs/m3224/symmetry_far_field.md`` and its three figures.

Run::

    PYTHONPATH=src python -m tensorbem.benchmarks.symmetry_far_field \
        --output docs/m3224 --subdivisions 2

Two bodies are used, both *exactly* symmetric about ``y = 0`` with the seam on
mesh edges:

* floating hemisphere (surface-piercing, ``WettedHullMesh``, 4*4**s panels);
* submerged sphere (closed ``TriangleMesh``), built by gluing the hemisphere to
  its ``z``-reflection so that the meridian seam is shared bit-for-bit -- the
  stock :func:`tensorbem.geometry3d.submerged_sphere_mesh` is a geodesic
  icosphere whose panels straddle ``y = 0`` and therefore cannot be split.

Three independent criteria are reported per case:

1. **Folding algebra** -- the folded Kochin function of *symmetrised* full-surface
   data must reproduce the full-surface Kochin function to round-off.  This
   isolates ``K_full(theta) = K_half(theta) + s*K_half(-theta)`` from any solve.
2. **End-to-end** -- folded far-field values from the half solve vs the full
   values from the full solve; the residual is the half/full solve difference
   amplified through the Kochin integral, not an algebra error.
3. **Analytic reference** -- the folded Haskind loads of the submerged sphere vs
   the exact Wang/Linton multipole solution
   (:func:`tensorbem.analytic.submerged_sphere_wave_multipole`).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.free_surface_solver3d import (
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    kochin_function_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import TriangleMesh, hemisphere_hull_mesh
from tensorbem.symmetry3d import (
    MODE_PARITY,
    far_field_damping_matrix_3d_symmetric,
    folded_kochin_function_3d,
    haskind_excitation_force_3d_symmetric,
    solve_radiation_3d_symmetric,
    split_patch,
)

MODE_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
DENSITY = 1025.0
GRAVITY = 9.80665
AZIMUTHS = np.linspace(0.0, 2.0 * math.pi, 73, endpoint=False)[:-1]


# ----------------------------------------------------------------- geometry
def symmetric_submerged_sphere_mesh(
    subdivisions: int, radius: float = 1.0, submergence: float = 1.5
) -> TriangleMesh:
    """Closed sphere, exactly symmetric about ``y = 0``, submerged in ``z <= 0``.

    The lower half is the octahedron-subdivision hemisphere of
    :func:`tensorbem.geometry3d.hemisphere_hull_mesh`; the upper half is its
    ``z``-reflection with reversed vertex order (so normals stay outward).  The
    ``z = 0`` seam vertices are identical in both halves and are merged, which
    makes the far-field reflection ``M: y -> -y`` a panel-to-panel map.
    """
    base = hemisphere_hull_mesh(subdivisions, radius=radius)
    lower_v = np.asarray(base.vertices, dtype=float)
    lower_f = np.asarray(base.faces, dtype=np.int64)
    upper_v = lower_v * np.array([1.0, 1.0, -1.0])
    upper_f = lower_f[:, [0, 2, 1]] + lower_v.shape[0]
    verts = np.vstack((lower_v, upper_v))
    verts[np.abs(verts) < 1.0e-15] = 0.0
    faces = np.vstack((lower_f, upper_f))
    uniq, inverse = np.unique(verts, axis=0, return_inverse=True)
    faces = np.asarray(inverse, dtype=np.int64).reshape(-1)[faces]
    offset = np.array([0.0, 0.0, -float(submergence)])
    return TriangleMesh(uniq + offset, faces)


# ------------------------------------------------------------------ metrics
def _nearest(reference: np.ndarray, points: np.ndarray) -> np.ndarray:
    out = np.empty(points.shape[0], dtype=np.int64)
    for idx, point in enumerate(points):
        out[idx] = int(np.argmin(np.linalg.norm(reference - point, axis=1)))
    return out


def _relative(numer: np.ndarray, scale: float) -> float:
    return float(np.max(np.abs(numer)) / scale) if scale > 0.0 else float(np.max(np.abs(numer)))


def analyse_case(mesh, nu: float, center, label: str) -> dict[str, Any]:
    """Compare folded far-field values against the full-surface reference."""
    parities = np.array([MODE_PARITY[k] for k in range(6)], dtype=float)
    start = time.perf_counter()
    full = solve_radiation_3d(mesh, nu, center=center, density=DENSITY, gravity=GRAVITY)
    t_full = time.perf_counter() - start
    upper, lower = split_patch(mesh)
    start = time.perf_counter()
    half = solve_radiation_3d_symmetric(
        upper, nu, center=center, density=DENSITY, gravity=GRAVITY
    )
    t_half = time.perf_counter() - start

    idx_upper = _nearest(mesh.centroids, upper.centroids)
    idx_lower = _nearest(mesh.centroids, lower.centroids)
    idx_mirror = _nearest(
        mesh.centroids, mesh.centroids * np.array([1.0, -1.0, 1.0])
    )  # full panel -> its y = 0 image, panel for panel

    b_full = far_field_damping_matrix_3d(mesh, full, density=DENSITY, gravity=GRAVITY)
    b_half = far_field_damping_matrix_3d_symmetric(
        upper, half, density=DENSITY, gravity=GRAVITY
    )
    h_full = haskind_excitation_force_3d(mesh, full, density=DENSITY, gravity=GRAVITY)
    h_half = haskind_excitation_force_3d_symmetric(
        upper, half, density=DENSITY, gravity=GRAVITY
    )

    per_mode: list[dict[str, Any]] = []
    identity_abs: list[float] = []
    end_to_end_abs: list[float] = []
    kochin_scales: list[float] = []
    for i, mode in enumerate(full.modes):
        s = parities[int(mode)]
        k_full = kochin_function_3d(
            mesh, full.potentials[:, i], full.normal_velocities[:, i], nu, AZIMUTHS
        )
        k_folded = folded_kochin_function_3d(
            upper,
            half.potentials[:, i],
            half.normal_velocities[:, i],
            nu,
            AZIMUTHS,
            parity=s,
        )
        # symmetrised full-surface data: exact parity by construction
        phi_sym = 0.5 * (full.potentials[:, i] + s * full.potentials[idx_mirror, i])
        vn_sym = 0.5 * (full.normal_velocities[:, i] + s * full.normal_velocities[idx_mirror, i])
        k_sym_full = kochin_function_3d(mesh, phi_sym, vn_sym, nu, AZIMUTHS)
        k_sym_folded = folded_kochin_function_3d(
            upper, phi_sym[idx_upper], vn_sym[idx_upper], nu, AZIMUTHS, parity=s
        )
        scale = float(np.abs(k_full).max())
        identity = _relative(k_sym_full - k_sym_folded, scale)
        end_to_end = _relative(k_full - k_folded, scale)
        identity_abs.append(float(np.abs(k_sym_full - k_sym_folded).max()))
        end_to_end_abs.append(float(np.abs(k_full - k_folded).max()))
        kochin_scales.append(scale)
        per_mode.append(
            {
                "mode": int(mode),
                "name": MODE_NAMES[int(mode)],
                "parity": float(s),
                "kochin_abs_max": scale,
                "kochin_identity_rel": identity,
                "kochin_end_to_end_rel": end_to_end,
                "phi_parity_residual": _relative(
                    full.potentials[idx_lower, i] - s * full.potentials[idx_upper, i],
                    float(np.abs(full.potentials[:, i]).max()),
                ),
            }
        )

    # a field-level number: the largest absolute Kochin error anywhere, divided by
    # the largest |K| of *any* mode, so that a near-zero mode cannot dominate
    kochin_scale = float(np.max(kochin_scales))
    identity_err = float(np.max(identity_abs) / kochin_scale)
    end_to_end_err = float(np.max(end_to_end_abs) / kochin_scale)

    b_scale = float(np.abs(b_full).max())
    h_scale = float(np.abs(h_full["kochin_haskind"]).max())
    cross = parities[full.modes][:, None] != parities[full.modes][None, :]
    cross_parity = float(np.abs(b_half[cross]).max() / max(b_scale, 1e-300))

    return {
        "label": label,
        "wavenumber": nu,
        "panels_full": int(mesh.panel_count),
        "panels_half": int(upper.panel_count),
        "timing_seconds": {"full_solve": t_full, "half_solve": t_half},
        "kochin_identity_rel": identity_err,
        "kochin_end_to_end_rel": end_to_end_err,
        "far_field_damping_end_to_end_rel": _relative(b_full - b_half, b_scale),
        "haskind_boundary_end_to_end_rel": _relative(
            h_full["boundary_haskind"] - h_half["boundary_haskind"], h_scale
        ),
        "haskind_kochin_end_to_end_rel": _relative(
            h_full["kochin_haskind"] - h_half["kochin_haskind"], h_scale
        ),
        "near_far_haskind_full_rel": _relative(
            h_full["boundary_haskind"] - h_full["kochin_haskind"], h_scale
        ),
        "near_far_haskind_half_rel": _relative(
            h_half["boundary_haskind"] - h_half["kochin_haskind"], h_scale
        ),
        "cross_parity_far_field_rel": cross_parity,
        "radiation_damping_scale": float(np.abs(full.damping).max()),
        "far_field_damping_scale": b_scale,
        "per_mode": per_mode,
        "far_field_damping_full": b_full.tolist(),
        "far_field_damping_half": b_half.tolist(),
    }


def analytic_sphere_case(
    subdivisions: int,
    ka_values: Sequence[float],
    submergence: float = 1.5,
    radius: float = 1.0,
) -> dict[str, Any]:
    """Folded Haskind loads of the submerged sphere vs the exact multipole."""
    mesh = symmetric_submerged_sphere_mesh(
        subdivisions, radius=radius, submergence=submergence
    )
    upper, _ = split_patch(mesh)
    center = (0.0, 0.0, -submergence)
    rows: list[dict[str, Any]] = []
    for ka in ka_values:
        nu = float(ka) / radius
        ref = submerged_sphere_wave_multipole(
            ka,
            radius=radius,
            submergence=submergence,
            density=DENSITY,
            gravity=GRAVITY,
        )
        half = solve_radiation_3d_symmetric(
            upper, nu, center=center, density=DENSITY, gravity=GRAVITY
        )
        hask = haskind_excitation_force_3d_symmetric(
            upper, half, density=DENSITY, gravity=GRAVITY
        )
        b_far = far_field_damping_matrix_3d_symmetric(
            upper, half, density=DENSITY, gravity=GRAVITY
        )
        modes = [int(m) for m in half.modes]
        i_x, i_z = modes.index(0), modes.index(2)
        fx = complex(hask["kochin_haskind"][i_x])
        fz = complex(hask["kochin_haskind"][i_z])
        fx_b = complex(hask["boundary_haskind"][i_x])
        fz_b = complex(hask["boundary_haskind"][i_z])
        ex, ez = complex(ref.excitation_force_x), complex(ref.excitation_force_z)
        rows.append(
            {
                "ka": float(ka),
                "excitation_x": [fx.real, fx.imag],
                "excitation_x_boundary": [fx_b.real, fx_b.imag],
                "excitation_z": [fz.real, fz.imag],
                "excitation_z_boundary": [fz_b.real, fz_b.imag],
                "analytic_x": [ex.real, ex.imag],
                "analytic_z": [ez.real, ez.imag],
                "excitation_z_error_rel": float(abs(fz - ez) / abs(ez)),
                "excitation_x_error_rel": float(abs(fx - ex) / abs(ex)),
                "heave_added_mass": float(half.added_mass[i_z, i_z]),
                "heave_added_mass_analytic": float(ref.heave_added_mass),
                "heave_damping_near": float(half.damping[i_z, i_z]),
                "heave_damping_far": float(b_far[i_z, i_z]),
                "heave_damping_analytic": float(ref.heave_damping),
                "heave_added_mass_error_rel": float(
                    abs(half.added_mass[i_z, i_z] - ref.heave_added_mass)
                    / abs(ref.heave_added_mass)
                ),
                "heave_damping_error_rel": float(
                    abs(b_far[i_z, i_z] - ref.heave_damping) / abs(ref.heave_damping)
                ),
            }
        )
    return {
        "subdivisions": int(subdivisions),
        "panels_full": int(mesh.panel_count),
        "panels_half": int(upper.panel_count),
        "radius": radius,
        "submergence": submergence,
        "rows": rows,
    }


# ------------------------------------------------------------------ figures
def make_figures(data: dict[str, Any], outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1) Kochin functions: half-solve folded vs full-solve reference
    first = sorted(data["cases"])[0]
    hemi = data["cases"][first]
    mesh = hemisphere_hull_mesh(data["subdivisions"][0])
    full = solve_radiation_3d(mesh, 1.0, density=DENSITY, gravity=GRAVITY)
    upper, _ = split_patch(mesh)
    half = solve_radiation_3d_symmetric(upper, 1.0, density=DENSITY, gravity=GRAVITY)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, mode in zip(axes, (2, 1, 5)):
        i = int(np.nonzero(full.modes == mode)[0][0])
        k_full = kochin_function_3d(
            mesh, full.potentials[:, i], full.normal_velocities[:, i], 1.0, AZIMUTHS
        )
        k_fold = folded_kochin_function_3d(
            upper,
            half.potentials[:, i],
            half.normal_velocities[:, i],
            1.0,
            AZIMUTHS,
            parity=MODE_PARITY[mode],
        )
        ax.plot(np.degrees(AZIMUTHS), np.abs(k_full), "-", lw=3.0, label="full surface")
        ax.plot(
            np.degrees(AZIMUTHS),
            np.abs(k_fold),
            "--",
            lw=1.2,
            label="folded half (2 solves of N/2)",
        )
        ax.set_title(f"{MODE_NAMES[mode]} ($s={MODE_PARITY[mode]:+.0f}$)")
        ax.set_xlabel(r"azimuth $\theta$ [deg]")
        ax.set_ylabel(r"$|K(\theta)|$ [m$^2$]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Floating hemisphere, $\\nu a = 1$: partner analysis vs full surface "
        f"(max rel. diff {hemi['kochin_end_to_end_rel']:.1e})"
    )
    fig.tight_layout()
    path = figdir / "fig_kochin_overlap.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    # 2) error budget, log scale
    labels = []
    identity = []
    end2end = []
    for key, case in data["cases"].items():
        labels.append(key.replace("_", " "))
        identity.append(case["kochin_identity_rel"])
        end2end.append(
            max(
                case["kochin_end_to_end_rel"],
                case["far_field_damping_end_to_end_rel"],
                case["haskind_kochin_end_to_end_rel"],
                case["haskind_boundary_end_to_end_rel"],
            )
        )
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, identity, width=0.4, label="folding algebra (round-off)")
    ax.bar(x + 0.2, end2end, width=0.4, label="half solve vs full solve")
    ax.axhline(2.2e-16, color="k", ls=":", lw=1.0, label="double precision")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("max relative difference")
    ax.set_title("Far-field agreement: algebra vs solve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = figdir / "fig_relative_errors.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    # 3) folded Haskind loads vs exact multipole
    sphere = data["analytic_sphere"][str(data["subdivisions"][0])]
    ka = np.array([row["ka"] for row in sphere["rows"]])
    fx = np.array([complex(*row["excitation_x"]) for row in sphere["rows"]])
    fz = np.array([complex(*row["excitation_z"]) for row in sphere["rows"]])
    ax_mag = np.array([complex(*row["analytic_x"]) for row in sphere["rows"]])
    az_mag = np.array([complex(*row["analytic_z"]) for row in sphere["rows"]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(ka, np.abs(az_mag), "-", color="k", label=r"exact multipole $|F_z|$")
    axes[0].plot(ka, np.abs(fz), "o", mfc="none", label=r"folded BEM $|F_z|$")
    axes[0].plot(ka, np.abs(ax_mag), "--", color="C2", label=r"exact multipole $|F_x|$")
    axes[0].plot(ka, np.abs(fx), "s", mfc="none", label=r"folded BEM $|F_x|$")
    axes[0].set_xlabel(r"$ka$")
    axes[0].set_ylabel("exciting force [N]")
    axes[0].set_title("Submerged sphere, half-domain Haskind loads")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    err_x = np.array([row["excitation_x_error_rel"] for row in sphere["rows"]])
    err_z = np.array([row["excitation_z_error_rel"] for row in sphere["rows"]])
    axes[1].semilogy(ka, err_x, "s--", label=r"$F_x$")
    axes[1].semilogy(ka, err_z, "o-", label=r"$F_z$")
    axes[1].axhline(0.03, color="r", ls=":", label="3 % gate")
    axes[1].set_xlabel(r"$ka$")
    axes[1].set_ylabel("relative error vs multipole")
    axes[1].set_title("Against the exact Wang/Linton solution")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = figdir / "fig_haskind_vs_multipole.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/m3224")
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[2])
    parser.add_argument("--ka", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.2])
    args = parser.parse_args(argv)

    outdir = Path(args.output)
    (outdir / "outputs").mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "density": DENSITY,
        "gravity": GRAVITY,
        "subdivisions": list(args.subdivisions),
        "cases": {},
    }

    data["analytic_sphere"] = {}
    for subdivisions in args.subdivisions:
        hemisphere = hemisphere_hull_mesh(subdivisions)
        for nu in (1.0, 0.6):
            key = f"hemisphere_s{subdivisions}_ka{nu:.1f}"
            data["cases"][key] = analyse_case(
                hemisphere, nu, None, f"floating hemisphere s={subdivisions}, nu a = {nu}"
            )
        sphere = symmetric_submerged_sphere_mesh(subdivisions)
        data["cases"][f"sphere_s{subdivisions}_ka0.8"] = analyse_case(
            sphere,
            0.8,
            (0.0, 0.0, -1.5),
            f"submerged sphere (f/a = 1.5) s={subdivisions}, ka = 0.8",
        )
        data["analytic_sphere"][str(subdivisions)] = analytic_sphere_case(
            subdivisions, args.ka
        )
    data["figures"] = make_figures(data, outdir)

    path = outdir / "outputs" / "symmetry_far_field.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"wrote {path}")

    for key, case in data["cases"].items():
        print(
            f"{key}: identity={case['kochin_identity_rel']:.2e} "
            f"end_to_end(kochin)={case['kochin_end_to_end_rel']:.2e} "
            f"Bfar={case['far_field_damping_end_to_end_rel']:.2e} "
            f"haskind={case['haskind_kochin_end_to_end_rel']:.2e} "
            f"near/far(half)={case['near_far_haskind_half_rel']:.2e}"
        )
    for sub, block in data["analytic_sphere"].items():
        print(f"--- analytic sphere, subdivisions={sub} (N={block['panels_full']} panels)")
        for row in block["rows"]:
            print(
                f"  ka={row['ka']}: |Fz| err={row['excitation_z_error_rel']*100:.2f}% "
                f"|Fx| err={row['excitation_x_error_rel']*100:.2f}% "
                f"A33 err={row['heave_added_mass_error_rel']*100:.2f}% "
                f"B33far err={row['heave_damping_error_rel']*100:.2f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())