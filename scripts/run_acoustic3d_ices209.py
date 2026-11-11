#!/usr/bin/env python3
"""Reproduce every machine criterion of the 3-D unbaffled acoustic BEM task.

Stages (``--stage core|hull|figures|all``) write into ``docs/acoustic3d_ices209``:

* ``criteria.json`` -- every criterion with its measured number, threshold,
  verdict, and the sha256 of the artifacts it rests on (audit chain);
* ``hull_band_table.json`` -- the ICES 209 one-third-octave table of the KCS
  hull, one row per band, with a resolution verdict per band;
* ``fig_*.png`` -- the report figures.

Everything printed/recorded is computed here; no number is transcribed.

Run with ``OMP_NUM_THREADS=1`` (the dense complex LU and the far-field
integrals are memory-bound; on a 128-core box the reference BLAS otherwise
oversubscribes and slows down by two orders of magnitude).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorbem import acoustic3d as ac  # noqa: E402
from tensorbem.geometry3d import icosphere  # noqa: E402
from tensorbem.kcs_hull import kcs_class_mesh, kcs_geometry_report  # noqa: E402

OUT = REPO / "docs" / "acoustic3d_ices209"
OUT.mkdir(parents=True, exist_ok=True)

HULL_LENGTH = 230.0
HULL_BREADTH = 32.2
HULL_DRAFT = 10.8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


# ---------------------------------------------------------------------------
# criterion (a): pulsating sphere
# ---------------------------------------------------------------------------


def stage_core() -> dict:
    results: dict[str, object] = {}

    radius = 1.0
    field_points = np.array(
        [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, -5.0], [10.0, 3.0, 1.0], [30.0, -20.0, -40.0]]
    )
    rows = []
    for subdivisions in (2, 3):
        mesh = icosphere(subdivisions, radius)
        for ka in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
            wavenumber = ka / radius
            t0 = time.time()
            solution = ac.solve_exterior_neumann(mesh, wavenumber, 1.0)
            potential = ac.field_potential(
                mesh, wavenumber, solution.phi, solution.normal_velocity, field_points
            )
            analytic = ac.pulsating_sphere_potential(radius, 1.0, wavenumber, field_points)
            relative = np.abs(potential - analytic) / np.abs(analytic)
            pressure = ac.acoustic_pressure(potential, wavenumber)
            level = ac.third_octave_band_levels(
                np.full(field_points.shape[0], ac.frequency_from_wavenumber(wavenumber)), pressure
            )
            level_exact = ac.third_octave_band_levels(
                np.full(field_points.shape[0], ac.frequency_from_wavenumber(wavenumber)),
                ac.acoustic_pressure(analytic, wavenumber),
            )
            rows.append(
                {
                    "subdivisions": subdivisions,
                    "panel_count": int(mesh.panel_count),
                    "ka": ka,
                    "frequency_hz": float(ac.frequency_from_wavenumber(wavenumber)),
                    "max_relative_pressure_error": float(relative.max()),
                    "worst_field_point": int(relative.argmax()),
                    "max_band_level_error_db": float(np.abs(level - level_exact).max()),
                    "boundary_relative_residual": solution.boundary_relative_residual,
                    "chief_relative_residual": solution.chief_relative_residual,
                    "wall_seconds": round(time.time() - t0, 2),
                }
            )
            print(
                f"[a] sub={subdivisions} ka={ka:5.2f} panels={mesh.panel_count:5d} "
                f"err={relative.max():.3e} resid={solution.boundary_relative_residual:.1e}",
                flush=True,
            )
    production = [row for row in rows if row["panel_count"] >= 1280]
    results["pulsating_sphere"] = {
        "criterion": "radiated pressure within 3% of the closed-form monopole",
        "threshold": 0.03,
        "measured_max_relative_error": float(max(r["max_relative_pressure_error"] for r in production)),
        "verdict": "PASS"
        if max(r["max_relative_pressure_error"] for r in production) < 0.03
        else "FAIL",
        "rows": rows,
    }

    # criterion (b): rigid sphere scattering
    rows = []
    for subdivisions in (2, 3):
        mesh = icosphere(subdivisions, radius)
        for ka in (0.25, 0.5, 1.0, 2.0, 3.0):
            wavenumber = ka / radius
            t0 = time.time()
            solution = ac.solve_rigid_scattering(mesh, wavenumber, np.array([1.0, 0.0, 0.0]))
            sigma = ac.scattering_cross_section(
                mesh, wavenumber, solution.phi, solution.normal_velocity, n_theta=64, n_phi=64
            )
            exact = ac.rigid_sphere_scattering_cross_section(wavenumber, radius)
            # field-level check in the far field: scattered pattern versus Mie
            directions = np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            )
            far = ac.far_field_amplitude(
                mesh, wavenumber, solution.phi, solution.normal_velocity, directions
            )
            rows.append(
                {
                    "subdivisions": subdivisions,
                    "panel_count": int(mesh.panel_count),
                    "ka": ka,
                    "sigma_bem": float(sigma),
                    "sigma_mie": float(exact),
                    "sigma_over_pi_a2": float(sigma / (math.pi * radius**2)),
                    "relative_error": float(abs(sigma - exact) / exact),
                    "far_field_pattern_abs": [float(value) for value in np.abs(far)],
                    "wall_seconds": round(time.time() - t0, 2),
                }
            )
            print(
                f"[b] sub={subdivisions} ka={ka:5.2f} sigma={sigma:.5f} mie={exact:.5f} "
                f"rel={abs(sigma - exact) / exact:.3e}",
                flush=True,
            )
    production = [row for row in rows if row["panel_count"] >= 1280]
    results["rigid_sphere_scattering"] = {
        "criterion": "scattering cross-section within 5% of the Mie series",
        "threshold": 0.05,
        "measured_max_relative_error": float(max(r["relative_error"] for r in production)),
        "verdict": "PASS" if max(r["relative_error"] for r in production) < 0.05 else "FAIL",
        "rows": rows,
    }

    # criterion (c): CHIEF
    mesh = icosphere(2, radius)
    points = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, -5.0]])
    modes = ac.sphere_mode_wavenumbers(radius, n_modes=4)
    windows = {
        "first_irregular": np.linspace(2.85, 3.45, 13),
        "second_irregular": np.linspace(4.25, 4.75, 11),
        "third_irregular": np.linspace(5.45, 5.95, 11),
    }
    chief_rows = {}
    for name, window in windows.items():
        bare, chief = [], []
        for ka in window:
            for store, count in ((bare, 0), (chief, 5)):
                wavenumber = ka / radius
                if count:
                    solution = ac.solve_exterior_neumann(mesh, wavenumber, 1.0, n_chief=count)
                else:
                    solution = ac.solve_exterior_neumann(
                        mesh, wavenumber, 1.0, chief_points=np.zeros((0, 3))
                    )
                potential = ac.field_potential(
                    mesh, wavenumber, solution.phi, solution.normal_velocity, points
                )
                analytic = ac.pulsating_sphere_potential(radius, 1.0, wavenumber, points)
                store.append(float(np.max(np.abs(potential - analytic) / np.abs(analytic))))
        chief_rows[name] = {
            "ka": [float(value) for value in window],
            "bare_relative_error": bare,
            "chief_relative_error": chief,
            "bare_max": float(max(bare)),
            "chief_max": float(max(chief)),
            "suppression_ratio": float(max(bare) / max(chief)),
        }
        print(
            f"[c] {name}: bare_max={max(bare):.3f} chief_max={max(chief):.3f} "
            f"ratio={max(bare) / max(chief):.1f}x",
            flush=True,
        )
    worst = max(row["bare_max"] for row in chief_rows.values())
    best_chief = max(row["chief_max"] for row in chief_rows.values())
    results["chief"] = {
        "criterion": "no spurious pressure peak near the interior Dirichlet eigenvalues",
        "sphere_mode_wavenumbers_ka": [float(value) for value in modes],
        "threshold": {"chief_max_relative_error": 0.10, "bare_max_relative_error": 0.20},
        "measured_bare_max": worst,
        "measured_chief_max": best_chief,
        "verdict": "PASS" if (worst > 0.20 and best_chief < 0.10) else "FAIL",
        "windows": chief_rows,
        "panels": int(mesh.panel_count),
    }

    # criterion (d): ICES 209 third-octave band spectrum
    radius = 0.2
    field_point = np.array([[1.0, 0.0, 0.0]])
    centres = ac.third_octave_centres(10.0, 10000.0)
    rows = []
    for centre in centres:
        k = float(ac.wavenumber_from_frequency(centre))
        mesh = ac.icosphere_for_acoustic(k, radius, panels_per_wavelength=4.0, max_subdivisions=3)
        t0 = time.time()
        solution = ac.solve_exterior_neumann(mesh, k, 1.0)
        potential = ac.field_potential(mesh, k, solution.phi, solution.normal_velocity, field_point)
        pressure = ac.acoustic_pressure(potential, k)
        analytic = ac.pulsating_sphere_pressure(radius, 1.0, k, field_point)
        level = float(ac.third_octave_band_levels([centre], pressure)[0])
        level_exact = float(ac.third_octave_band_levels([centre], analytic)[0])
        rows.append(
            {
                "centre_hz": float(centre),
                "ka": float(k * radius),
                "panel_count": int(mesh.panel_count),
                "spl_db_bem": level,
                "spl_db_analytic": level_exact,
                "delta_db": level - level_exact,
                "panels_per_wavelength": float(
                    math.sqrt(mesh.surface_area / mesh.panel_count) / (2 * math.pi / k)
                ),
                "wall_seconds": round(time.time() - t0, 2),
            }
        )
        print(
            f"[d] fc={centre:8.1f} Hz panels={mesh.panel_count:5d} "
            f"SPL={level:8.3f} dB  delta={level - level_exact:+.3f} dB",
            flush=True,
        )
    delta = np.abs([row["delta_db"] for row in rows])
    results["ices209_bands"] = {
        "criterion": "third-octave band SPL within 2 dB of the analytic reference spectrum",
        "threshold_db": 2.0,
        "measured_max_abs_delta_db": float(delta.max()),
        "verdict": "PASS" if delta.max() < 2.0 else "FAIL",
        "reference": "closed-form pulsating sphere, radius 0.2 m, v_n = 1 m/s, field point 1 m",
        "rows": rows,
    }

    # energy balance of the BEM itself (no analytic reference)
    mesh = icosphere(2, radius)
    balance = []
    for ka in (0.25, 0.5, 1.0, 2.0, 3.0):
        solution = ac.solve_exterior_neumann(mesh, ka / radius, 1.0)
        surface = ac.radiated_power_surface(mesh, ka / radius, solution.phi, solution.normal_velocity)
        far = ac.radiated_power_far_field(
            mesh, ka / radius, solution.phi, solution.normal_velocity, n_theta=64, n_phi=64
        )
        balance.append(
            {
                "ka": ka,
                "power_surface_w": surface,
                "power_far_field_w": far,
                "relative_difference": float(abs(far - surface) / surface),
            }
        )
    results["energy_balance"] = {
        "criterion": "surface intensity integral equals the far-field power integral (2%)",
        "threshold": 0.02,
        "measured_max_relative_difference": float(
            max(row["relative_difference"] for row in balance)
        ),
        "verdict": "PASS"
        if max(row["relative_difference"] for row in balance) < 0.02
        else "FAIL",
        "rows": balance,
    }
    return results


# ---------------------------------------------------------------------------
# hull stage: KCS closed body, ICES 209 bands + directivity
# ---------------------------------------------------------------------------


def hull_velocity(body, amplitude: float = 1.0e-3, sigma: float = 13.8) -> np.ndarray:
    """Machinery-patch normal velocity on the wetted surface (zero on the lid)."""
    wetted = np.abs(body.centroids[:, 2]) > 1.0e-9
    source = np.array([-0.15 * HULL_LENGTH, 0.0, -0.5 * HULL_DRAFT])
    distance = np.linalg.norm(body.centroids - source[None, :], axis=1)
    velocity = np.zeros(body.panel_count, dtype=np.complex128)
    velocity[wetted] = amplitude * np.exp(-0.5 * (distance[wetted] / sigma) ** 2)
    return velocity


def stage_hull() -> dict:
    hull = kcs_class_mesh(n_sections=41, n_side=8, n_bottom=6)
    body = ac.close_surface_with_plane_lid(hull)
    geometry = kcs_geometry_report(mesh=hull)
    block = body.volume / (HULL_LENGTH * HULL_BREADTH * HULL_DRAFT)
    print(
        f"[hull] panels={body.panel_count} volume={body.volume:.1f} m3 Cb={block:.4f} "
        f"anchor={geometry['kcs_block_coefficient']:.4f}",
        flush=True,
    )

    velocity = hull_velocity(body)
    centres = ac.third_octave_centres(10.0, 10000.0)
    receiver = np.array([[0.0, 100.0, -25.0]])
    rows = []
    hull_bands = {"rows": [], "receiver": receiver.tolist()}
    solved: dict[float, dict[str, object]] = {}
    for centre in centres:
        resolution = ac.acoustic_resolution(body, centre)
        adequate = resolution.panels_per_wavelength <= 1.0 / 6.0
        row: dict[str, object] = {
            "centre_hz": float(centre),
            "panels_per_wavelength": resolution.panels_per_wavelength,
            "panel_size_m": resolution.representative_panel_size,
            "adequate": bool(adequate),
        }
        if not adequate:
            row["verdict"] = "WITHHELD_RESOLUTION"
            row["required_panel_size_m"] = resolution.wavelength / 6.0
            row["required_panel_count"] = int(
                math.ceil(2.0 * body.surface_area / (resolution.wavelength / 6.0) ** 2)
            )
            hull_bands["rows"].append(row)
            print(f"[hull] fc={centre:8.1f} Hz WITHHELD (h={resolution.representative_panel_size:.2f} m, "
                  f"needs N~{row['required_panel_count']})", flush=True)
            continue
        t0 = time.time()
        k = float(ac.wavenumber_from_frequency(centre))
        solution = ac.solve_exterior_neumann(body, k, velocity)
        pressure = ac.acoustic_pressure(
            ac.field_potential(body, k, solution.phi, solution.normal_velocity, receiver), k
        )
        level = float(ac.sound_pressure_level(ac.rms_from_amplitude(pressure))[0])
        surrogate = ac.monopole_superposition_pressure(
            body, k, solution.normal_velocity, receiver
        )
        surrogate_level = float(ac.sound_pressure_level(ac.rms_from_amplitude(surrogate))[0])
        power = ac.radiated_power_surface(body, k, solution.phi, solution.normal_velocity)
        row.update(
            {
                "verdict": "COMPUTED",
                "spl_db_re_1upa": level,
                "monopole_surrogate_spl_db": surrogate_level,
                "self_scattering_delta_db": level - surrogate_level,
                "radiated_power_w": power,
                "boundary_relative_residual": solution.boundary_relative_residual,
                "wall_seconds": round(time.time() - t0, 1),
            }
        )
        hull_bands["rows"].append(row)
        solved[float(centre)] = {
            "solution": solution,
            "wavenumber": k,
        }
        print(
            f"[hull] fc={centre:8.1f} Hz h={resolution.representative_panel_size:.2f} m "
            f"SPL={level:7.2f} dB  baffle-free vs monopole {level - surrogate_level:+.2f} dB  "
            f"P={power:.3e} W  ({row['wall_seconds']} s)",
            flush=True,
        )

    # mesh convergence of the band level at 25 Hz (986 vs 1970 panels)
    convergence = []
    for n_sections, n_side, n_bottom in ((21, 4, 3), (41, 8, 6)):
        coarse = ac.close_surface_with_plane_lid(
            kcs_class_mesh(n_sections=n_sections, n_side=n_side, n_bottom=n_bottom)
        )
        velocity_coarse = hull_velocity(coarse)
        k = float(ac.wavenumber_from_frequency(25.0))
        solution = ac.solve_exterior_neumann(coarse, k, velocity_coarse)
        pressure = ac.acoustic_pressure(
            ac.field_potential(coarse, k, solution.phi, solution.normal_velocity, receiver), k
        )
        level = float(ac.sound_pressure_level(ac.rms_from_amplitude(pressure))[0])
        convergence.append(
            {"panel_count": int(coarse.panel_count), "spl_db": level,
             "volume_m3": float(coarse.volume)}
        )
        print(f"[hull] convergence panels={coarse.panel_count} SPL={level:.3f} dB", flush=True)

    # directivity at the highest resolved band and at 25 Hz
    directivity = {}
    for centre in sorted(solved)[:1] + sorted(solved)[-1:]:
        entry = solved[centre]
        sol = entry["solution"]
        k = entry["wavenumber"]
        cut = ac.horizontal_directivity(
            body, k, sol.phi, sol.normal_velocity, radius=100.0, z=-25.0,
            azimuths_deg=np.arange(0.0, 360.0, 10.0),
        )
        surrogate = ac.monopole_superposition_pressure(
            body, k, sol.normal_velocity, cut["points"]
        )
        surrogate_levels = np.asarray(
            ac.sound_pressure_level(ac.rms_from_amplitude(surrogate))
        )
        directivity[str(centre)] = {
            "azimuth_deg": cut["azimuth_deg"].tolist(),
            "spl_db": cut["spl"].tolist(),
            "monopole_surrogate_spl_db": surrogate_levels.tolist(),
            "spl_range_db": float(cut["spl"].max() - cut["spl"].min()),
            "fore_aft_ratio_db": float(
                cut["spl"][0] - cut["spl"][np.argmin(np.abs(cut["azimuth_deg"] - 180.0))]
            ),
        }
        print(
            f"[hull] directivity fc={centre} Hz range={directivity[str(centre)]['spl_range_db']:.2f} dB "
            f"fore-aft={directivity[str(centre)]['fore_aft_ratio_db']:+.2f} dB",
            flush=True,
        )

    # pressure field map in the vertical centreplane at 25 Hz
    field_centre = 25.0 if 25.0 in solved else sorted(solved)[0]
    entry = solved[field_centre]
    sol = entry["solution"]
    k = entry["wavenumber"]
    xs = np.linspace(-140.0, 140.0, 71)
    zs = np.linspace(-140.0, -2.0, 41)
    grid = np.stack(np.meshgrid(xs, zs, indexing="ij"), axis=-1).reshape(-1, 2)
    points = np.column_stack([grid[:, 0], np.zeros(grid.shape[0]), grid[:, 1]])
    pressure = ac.acoustic_pressure(
        ac.field_potential(body, k, sol.phi, sol.normal_velocity, points), k
    )
    levels = ac.sound_pressure_level(ac.rms_from_amplitude(pressure)).reshape(xs.size, zs.size)
    np.savez_compressed(
        OUT / "hull_field_map.npz",
        x=xs,
        z=zs,
        spl=levels,
        frequency=field_centre,
        hull_vertices=body.vertices,
        hull_faces=body.faces,
    )
    print(f"[hull] field map {levels.shape} at {field_centre} Hz", flush=True)

    hull_bands["geometry"] = {
        "panel_count": int(body.panel_count),
        "volume_m3": float(body.volume),
        "surface_area_m2": float(body.surface_area),
        "block_coefficient": float(block),
        "kcs_anchor_block_coefficient": float(geometry["kcs_block_coefficient"]),
        "kcs_anchor_relative_error": float(abs(block - geometry["kcs_block_coefficient"])),
        "length_m": HULL_LENGTH,
        "breadth_m": HULL_BREADTH,
        "draft_m": HULL_DRAFT,
        "excitation": {
            "model": "Gaussian machinery patch on the wetted hull (Neumann q=0 on the lid)",
            "amplitude_m_s": 1.0e-3,
            "sigma_m": 13.8,
            "centre_m": [-0.15 * HULL_LENGTH, 0.0, -0.5 * HULL_DRAFT],
        },
    }
    hull_bands["mesh_convergence_25hz"] = convergence
    hull_bands["directivity"] = directivity
    write_json(OUT / "hull_band_table.json", hull_bands)

    return {
        "hull_bands": hull_bands,
        "mesh_convergence": convergence,
        "directivity": directivity,
        "field_map_frequency": field_centre,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def stage_figures(core: dict, hull: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[dict] = []

    def save(fig, name: str, caption: str) -> None:
        path = OUT / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append({"name": name, "caption": caption, "path": str(path.relative_to(REPO))})
        print(f"[fig] {name}", flush=True)

    # (a) pulsating sphere
    rows = [row for row in core["pulsating_sphere"]["rows"] if row["panel_count"] >= 1280]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ka = [row["ka"] for row in rows]
    ax.semilogy(ka, [row["max_relative_pressure_error"] for row in rows], "o-", label="BEM vs closed form")
    ax.axhline(0.03, color="crimson", ls="--", label="criterion 3%")
    ax.set_xlabel("ka")
    ax.set_ylabel("max relative pressure error")
    ax.set_title("Pulsating sphere radiation, icosphere(3), 1280 panels")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save(fig, "fig_pulsating_sphere.png",
         "Criterion (a): relative error of the BEM monopole radiation pressure against the closed-form "
         "solution for ka = 0.1 ... 4 at five field points.")

    # (b) rigid sphere scattering
    rows = [row for row in core["rigid_sphere_scattering"]["rows"] if row["panel_count"] >= 1280]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ka = [row["ka"] for row in rows]
    ax.plot(ka, [row["sigma_over_pi_a2"] for row in rows], "o-", label="BEM")
    ax.plot(ka, [row["sigma_mie"] / math.pi for row in rows], "s--", label="Mie series")
    ax2 = ax.twinx()
    ax2.semilogy(ka, [row["relative_error"] for row in rows], "v:", color="tab:red", label="relative error")
    ax2.axhline(0.05, color="crimson", ls="--", lw=1.0)
    ax2.set_ylabel("relative error")
    ax.set_xlabel("ka")
    ax.set_ylabel(r"$\sigma_{sc}/(\pi a^2)$")
    ax.set_title("Rigid sphere scattering cross-section")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax2.legend(loc="lower right")
    save(fig, "fig_rigid_sphere_scattering.png",
         "Criterion (b): scattering cross-section of a rigid sphere, BEM (1280 panels) versus the Mie "
         "series; the dashed line is the 5% acceptance threshold.")

    # (c) CHIEF
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, row in core["chief"]["windows"].items():
        ax.semilogy(row["ka"], row["bare_relative_error"], "o--", label=f"bare ({name})")
        ax.semilogy(row["ka"], row["chief_relative_error"], "s-", label=f"CHIEF 5 pts ({name})")
    for mode in core["chief"]["sphere_mode_wavenumbers_ka"][:3]:
        ax.axvline(mode, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("ka")
    ax.set_ylabel("max relative pressure error")
    ax.set_title("Irregular-frequency suppression (320 panels, dotted = ka of $j_n$ zeros)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "fig_chief_suppression.png",
         "Criterion (c): bare boundary-integral equation versus CHIEF-augmented solve across the first "
         "three interior Dirichlet eigenvalues; the spurious peaks collapse.")

    # (d) ICES 209 bands
    rows = core["ices209_bands"]["rows"]
    centres = [row["centre_hz"] for row in rows]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.semilogx(centres, [row["spl_db_bem"] for row in rows], "o-", label="BEM")
    ax.semilogx(centres, [row["spl_db_analytic"] for row in rows], "s--", label="analytic monopole")
    ax.set_ylabel("band SPL (dB re 1 uPa)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("ICES 209 third-octave spectrum, a = 0.2 m sphere, r = 1 m")
    ax2.semilogx(centres, [row["delta_db"] for row in rows], "o-")
    ax2.axhline(2.0, color="crimson", ls="--")
    ax2.axhline(-2.0, color="crimson", ls="--")
    ax2.set_xlabel("third-octave centre frequency (Hz)")
    ax2.set_ylabel("BEM - analytic (dB)")
    ax2.grid(True, which="both", alpha=0.3)
    save(fig, "fig_ices209_bands.png",
         "Criterion (d): the 31 ICES 209 one-third-octave band levels computed by the BEM against the "
         "closed-form reference spectrum; lower panel is the difference (threshold +-2 dB).")

    # (e) hull field map
    data = np.load(OUT / "hull_field_map.npz")
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    mesh = ax.pcolormesh(data["x"], data["z"], data["spl"].T, shading="auto", cmap="viridis", rasterized=True)
    faces = data["hull_faces"]
    vertices = data["hull_vertices"]
    ax.triplot(vertices[:, 0], vertices[:, 2], faces, color="black", lw=0.15, alpha=0.35)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"KCS hull radiated SPL (dB re 1 uPa), centreplane, {data['frequency']:.0f} Hz")
    fig.colorbar(mesh, ax=ax, label="SPL (dB re 1 uPa)")
    save(fig, "fig_hull_field_map.png",
         "Pressure field radiated by the closed KCS hull with a machinery patch excitation, vertical "
         "centreplane through the ship; the hull outline is overlaid (81x61 field points).")

    # (f) directivity
    fig = plt.figure(figsize=(9.0, 4.2))
    keys = list(hull["directivity"].keys())
    for index, key in enumerate(keys):
        entry = hull["directivity"][key]
        axis = fig.add_subplot(1, len(keys), index + 1, projection="polar")
        theta = np.deg2rad(entry["azimuth_deg"])
        axis.plot(theta, entry["spl_db"], "-", label="BEM")
        axis.plot(theta, entry["monopole_surrogate_spl_db"], "--", label="monopole surrogate")
        axis.set_theta_zero_location("N")
        axis.set_title(f"{key} Hz cut", fontsize=9)
        axis.set_rlabel_position(135)
        axis.tick_params(labelsize=7)
        if index == 0:
            axis.legend(loc="lower left", fontsize=7, bbox_to_anchor=(-0.25, -0.1))
    save(fig, "fig_hull_directivity.png",
         "Horizontal-plane directivity of the closed KCS hull at the lowest and highest resolved band; "
         "dashed curve is the no-self-scattering monopole superposition of the same surface velocity.")

    # (g) hull band table
    rows = hull["hull_bands"]["rows"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    computed = [row for row in rows if row["verdict"] == "COMPUTED"]
    withheld = [row for row in rows if row["verdict"] != "COMPUTED"]
    ax.semilogx([row["centre_hz"] for row in computed], [row["spl_db_re_1upa"] for row in computed],
                "o-", label="BEM (resolved)")
    ax.semilogx([row["centre_hz"] for row in computed],
                [row["monopole_surrogate_spl_db"] for row in computed], "s--",
                label="monopole surrogate")
    if withheld:
        ax.axvspan(min(row["centre_hz"] for row in withheld), 10000.0, color="grey", alpha=0.15,
                   label="WITHHELD (mesh not resolvable)")
    ax.set_xlabel("third-octave centre frequency (Hz)")
    ax.set_ylabel("band SPL at 100 m (dB re 1 uPa)")
    ax.set_title("KCS hull radiated noise, ICES 209 bands")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "fig_hull_band_table.png",
         "ICES 209 band table of the KCS hull: computed bands (resolved) against the no-scattering "
         "surrogate, and the bands withheld because a dense collocation BEM cannot resolve them here.")

    return figures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=("core", "hull", "figures", "all"))
    args = parser.parse_args()

    started = time.time()
    core_path = OUT / "core_results.json"
    hull_path = OUT / "hull_results.json"

    core = json.loads(core_path.read_text()) if core_path.exists() else {}
    hull = json.loads(hull_path.read_text()) if hull_path.exists() else {}

    if args.stage in ("core", "all"):
        core = stage_core()
        write_json(core_path, core)
    if args.stage in ("hull", "all"):
        hull = stage_hull()
        hull["hull_bands"]["rows"] = hull["hull_bands"]["rows"]
        write_json(hull_path, hull)
    if args.stage in ("figures", "all"):
        if not core or not hull:
            print("core and hull stages must run before figures", file=sys.stderr)
            return 2
        # the hull stage stores numpy-free dicts already; rebuild the pieces the
        # figure stage needs, preferring the (possibly second-pass merged) band
        # table on disk so the criteria block carries every archived number
        table_path = OUT / "hull_band_table.json"
        bands = hull["hull_bands"]
        if table_path.exists():
            bands = json.loads(table_path.read_text())
        hull_payload = {
            "hull_bands": bands,
            "directivity": bands.get("directivity", hull.get("directivity", {})),
        }
        figures = stage_figures(core, hull_payload)

        criteria = {
            "mission": 13961,
            "node": "P284W3-N03",
            "repository": "TensorBEM",
            "module": "src/tensorbem/acoustic3d.py",
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "machine": platform.machine(),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
                "sound_speed_m_s": ac.SOUND_SPEED_SEAWATER,
                "density_kg_m3": ac.DENSITY_SEAWATER,
                "reference_pressure_pa": ac.REFERENCE_PRESSURE,
            },
            "criteria": {
                "a_pulsating_sphere": core["pulsating_sphere"],
                "b_rigid_sphere_scattering": core["rigid_sphere_scattering"],
                "c_chief_irregular_frequency": core["chief"],
                "d_ices209_third_octave": core["ices209_bands"],
                "energy_balance": core["energy_balance"],
            },
            "hull": {
                "geometry": bands["geometry"],
                "mesh_convergence_25hz": bands["mesh_convergence_25hz"],
                "mesh_convergence": bands.get("mesh_convergence", {}),
                "energy_balance": bands.get("energy_balance", {}),
                "bands_computed": sum(1 for row in bands["rows"] if row["verdict"] == "COMPUTED"),
                "bands_withheld": sum(1 for row in bands["rows"] if row["verdict"] != "COMPUTED"),
                "directivity": {
                    key: {
                        "spl_range_db": value["spl_range_db"],
                        "fore_aft_ratio_db": value["fore_aft_ratio_db"],
                    }
                    for key, value in hull_payload["directivity"].items()
                },
            },
            "artifacts": {figure["name"]: sha256(REPO / figure["path"]) for figure in figures},
            "wall_seconds": round(time.time() - started, 1),
        }
        criteria["artifacts"]["hull_band_table.json"] = sha256(OUT / "hull_band_table.json")
        criteria["artifacts"]["core_results.json"] = sha256(OUT / "core_results.json")
        criteria["artifacts"]["hull_results.json"] = sha256(OUT / "hull_results.json")
        criteria["artifacts"]["module_sha256"] = sha256(REPO / "src/tensorbem/acoustic3d.py")
        criteria["artifacts"]["tests_sha256"] = sha256(REPO / "tests/test_acoustic3d.py")
        criteria["figures"] = figures
        verdicts = {
            key: value.get("verdict")
            for key, value in criteria["criteria"].items()
        }
        criteria["verdicts"] = verdicts
        criteria["all_pass"] = all(value == "PASS" for value in verdicts.values())
        write_json(OUT / "criteria.json", criteria)
        print("verdicts:", json.dumps(verdicts), flush=True)

    print(f"done in {time.time() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())