#!/usr/bin/env python3
"""Second-pass checks on the KCS acoustic body, merged into the archived JSON.

The main driver (:mod:`scripts.run_acoustic3d_ices209`) establishes the four
task criteria and the band table.  This pass answers the two questions a
reviewer asks about the *ship* numbers, which the sphere cannot answer:

1. **Is the ship band level converged?**  986 panels versus 1970 panels at 25 Hz
   and 40 Hz (the two frequencies where the self-scattering delta changes
   sign/magnitude).
2. **Does the BEM solution close its own energy budget on a real hull?**
   Surface intensity integral versus far-field power integral at the same
   frequency -- a conservation check that needs no analytic reference.

Results are merged into ``docs/acoustic3d_ices209/hull_band_table.json`` and
``criteria.json`` (rewritten, so the audit chain stays consistent: re-run
``scripts/run_acoustic3d_ices209.py --stage figures`` afterwards).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorbem import acoustic3d as ac  # noqa: E402
from tensorbem.kcs_hull import kcs_class_mesh  # noqa: E402

OUT = REPO / "docs" / "acoustic3d_ices209"
RECEIVER = np.array([[0.0, 100.0, -25.0]])
APERTURES = ((21, 4, 3), (41, 8, 6))
FREQUENCIES = (25.0, 40.0)


def hull_velocity(body, amplitude: float = 1.0e-3, sigma: float = 13.8) -> np.ndarray:
    wetted = np.abs(body.centroids[:, 2]) > 1.0e-9
    source = np.array([-0.15 * 230.0, 0.0, -5.4])
    distance = np.linalg.norm(body.centroids - source[None, :], axis=1)
    velocity = np.zeros(body.panel_count, dtype=np.complex128)
    velocity[wetted] = amplitude * np.exp(-0.5 * (distance[wetted] / sigma) ** 2)
    return velocity


def main() -> int:
    started = time.time()
    convergence: dict[str, list[dict[str, float]]] = {}
    balance: dict[str, dict[str, float]] = {}
    for aperture in APERTURES:
        body = ac.close_surface_with_plane_lid(
            kcs_class_mesh(n_sections=aperture[0], n_side=aperture[1], n_bottom=aperture[2])
        )
        velocity = hull_velocity(body)
        for frequency in FREQUENCIES:
            k = float(ac.wavenumber_from_frequency(frequency))
            resolution = ac.acoustic_resolution(body, frequency)
            solution = ac.solve_exterior_neumann(body, k, velocity)
            pressure = ac.acoustic_pressure(
                ac.field_potential(body, k, solution.phi, solution.normal_velocity, RECEIVER), k
            )
            level = float(ac.sound_pressure_level(ac.rms_from_amplitude(pressure))[0])
            surface = ac.radiated_power_surface(body, k, solution.phi, solution.normal_velocity)
            far = ac.radiated_power_far_field(
                body, k, solution.phi, solution.normal_velocity, n_theta=64, n_phi=64
            )
            convergence.setdefault(f"{frequency:g}", []).append(
                {
                    "aperture": list(aperture),
                    "panel_count": int(body.panel_count),
                    "panels_per_wavelength": resolution.panels_per_wavelength,
                    "spl_db": level,
                    "volume_m3": float(body.volume),
                }
            )
            balance.setdefault(f"{frequency:g}", {})[str(body.panel_count)] = {
                "power_surface_w": surface,
                "power_far_field_w": far,
                "relative_difference": float(abs(far - surface) / surface),
            }
            print(
                f"[hullcheck] panels={body.panel_count} f={frequency:g} Hz "
                f"h/lambda={resolution.panels_per_wavelength:.3f} SPL={level:.3f} dB "
                f"P_surf={surface:.4e} P_far={far:.4e} "
                f"diff={abs(far - surface) / surface:.2e}",
                flush=True,
            )

    table_path = OUT / "hull_band_table.json"
    table = json.loads(table_path.read_text())
    table["mesh_convergence"] = convergence
    table["energy_balance"] = balance
    table_path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"[hullcheck] merged in {time.time() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())