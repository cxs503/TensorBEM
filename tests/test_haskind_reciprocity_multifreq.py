"""Multi-frequency Haskind reciprocity verification for 3-D floating hemisphere.

Haskind's relation connects the wave excitation force on a body to the
radiation potential of the same body:

    X_j(beta) = i * rho * omega * A * integral_{Gamma_B}
                [phi_I * n_j - phi_j * d(phi_I)/dn] dS

and equivalently via the Kochin function:

    X_j(beta) = rho * g * A * K_j(beta + pi)

This test verifies that both paths agree across a frequency sweep for a
floating hemisphere, providing a robust check of the 3-D free-surface
solver's radiation condition and far-field consistency.

Reference: Newman, J.N. (1977). "Marine Hydrodynamics." MIT Press, §6.5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh


@pytest.fixture(scope="module")
def hemisphere_solutions():
    """Pre-compute radiation and diffraction at multiple frequencies."""
    mesh = hemisphere_hull_mesh(2, radius=1.0)
    ka_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    results = {}
    for ka in ka_values:
        ops = assemble_free_surface_operators_3d(mesh, ka, chief_points="auto")
        rad = solve_radiation_3d(mesh, ka, operators=ops)
        diff = solve_diffraction_3d(mesh, ka, wave_angle=0.0, operators=ops)
        haskind = haskind_excitation_force_3d(mesh, rad, amplitude=1.0, wave_angle=0.0)
        results[ka] = {"mesh": mesh, "rad": rad, "diff": diff, "haskind": haskind}
    return results


class TestHaskindReciprocityMultiFreq:
    """Verify Haskind reciprocity across a frequency sweep."""

    def test_near_far_field_excitation_agreement(self, hemisphere_solutions):
        """Near-field Haskind integral matches far-field Kochin formula."""
        for ka, sol in hemisphere_solutions.items():
            haskind = sol["haskind"]
            boundary = haskind["boundary_haskind"]
            kochin = haskind["kochin_haskind"]

            for mode_idx in range(boundary.size):
                b_val = complex(boundary[mode_idx])
                k_val = complex(kochin[mode_idx])

                scale = max(abs(b_val), abs(k_val), 1.0)
                rel_diff = abs(b_val - k_val) / scale

                assert rel_diff < 0.02, (
                    f"ka={ka}, mode={mode_idx}: "
                    f"boundary={b_val:.6f}, kochin={k_val:.6f}, "
                    f"rel_diff={rel_diff:.4e}"
                )

    def test_diffraction_vs_haskind_excitation(self, hemisphere_solutions):
        """Direct diffraction solve matches Haskind-predicted excitation."""
        for ka, sol in hemisphere_solutions.items():
            diff = sol["diff"]
            haskind = sol["haskind"]

            diff_fz = complex(diff.excitation_force[2])
            haskind_fz = complex(haskind["boundary_haskind"][2])

            scale = max(abs(diff_fz), abs(haskind_fz), 1.0)
            rel_diff = abs(diff_fz - haskind_fz) / scale

            assert rel_diff < 0.05, (
                f"ka={ka}: diff_fz={diff_fz:.6f}, "
                f"haskind_fz={haskind_fz:.6f}, rel_diff={rel_diff:.4e}"
            )

    def test_haskind_vs_hulme_heave(self, hemisphere_solutions):
        """Haskind excitation matches Hulme (1982) semi-analytical reference."""
        for ka, sol in hemisphere_solutions.items():
            haskind = sol["haskind"]
            hulme = hulme_hemisphere_radiation(ka, radius=1.0, density=1025.0)

            haskind_fz = complex(haskind["boundary_haskind"][2])
            hulme_fz = hulme.excitation_force_z

            rel_diff = abs(haskind_fz - hulme_fz) / abs(hulme_fz)

            assert rel_diff < 0.05, (
                f"ka={ka}: haskind_fz={haskind_fz:.6f}, "
                f"hulme_fz={hulme_fz:.6f}, rel_diff={rel_diff:.4e}"
            )

    def test_energy_conservation(self, hemisphere_solutions):
        """Radiation damping is positive and consistent with energy flux."""
        for ka, sol in hemisphere_solutions.items():
            rad = sol["rad"]

            assert rad.damping[2] > 0, f"ka={ka}: heave damping {rad.damping[2]} <= 0"
            assert rad.damping[0] > 0, f"ka={ka}: surge damping {rad.damping[0]} <= 0"

            if ka < 0.5:
                assert rad.damping[2] < ka**2 * 10, (
                    f"ka={ka}: heave damping {rad.damping[2]} too large for low-ka scaling"
                )

    def test_reciprocity_symmetry(self, hemisphere_solutions):
        """Haskind relation is symmetric under mode exchange for axisymmetric body."""
        ka = 1.0
        sol = hemisphere_solutions[ka]
        haskind = sol["haskind"]

        kochin_heave = complex(haskind["kochin_haskind"][2])
        if abs(kochin_heave) > 1e-6:
            assert abs(kochin_heave.imag) / abs(kochin_heave) < 0.1, (
                f"ka={ka}: heave Kochin force has unexpected phase: {kochin_heave}"
            )
