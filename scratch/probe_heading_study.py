"""Probe the new oblique-heading study before wiring thresholds."""

import math

import numpy as np

from tensorbem.benchmarks import case_cylinder_stream as case


def main() -> None:
    probes = case.probe_points()
    # reference consistency: heading 0 reproduces the axis-aligned closed form
    a0 = case.heading_velocity(probes, 0.0)
    ref = case.exact_velocity(probes)
    print("heading0 vs exact", np.max(np.abs(a0 - ref)))
    # the heading reference must be the rotation of the axis-aligned one
    h = math.radians(30.0)
    c, s = math.cos(h), math.sin(h)
    back = np.column_stack((c * probes[:, 0] + s * probes[:, 1],
                            -s * probes[:, 0] + c * probes[:, 1]))
    flow = case.exact_velocity(back)
    rotated = np.column_stack((c * flow[:, 0] - s * flow[:, 1],
                               s * flow[:, 0] + c * flow[:, 1]))
    print("rotation identity", np.max(np.abs(case.heading_velocity(probes, h) - rotated)))
    # harmonicity / irrotationality of the heading reference, finite differences
    eps = 1e-6
    p = probes[:8]
    lap = np.zeros(len(p))
    curl = np.zeros(len(p))
    for i, d in enumerate(((eps, 0.0), (-eps, 0.0), (0.0, eps), (0.0, -eps))):
        pass
    for base, idx in ((0, 0), (1, 1)):
        off = np.zeros(2)
        off[base] = eps
        plus = case.heading_perturbation_potential(p + off, h)
        minus = case.heading_perturbation_potential(p - off, h)
        lap += (plus - 2 * case.heading_perturbation_potential(p, h) + minus) / eps**2
    vplus = case.heading_velocity(p + np.array([0.0, eps]), h)
    vminus = case.heading_velocity(p - np.array([0.0, eps]), h)
    uprime = case.heading_velocity(p + np.array([eps, 0.0]), h)
    uminus = case.heading_velocity(p - np.array([eps, 0.0]), h)
    curl = ((vplus[:, 0] - vminus[:, 0]) - (uprime[:, 1] - uminus[:, 1])) / (2 * eps)
    print("heading laplacian", np.max(np.abs(lap)))
    print("heading curl", np.max(np.abs(curl)))
    # neumann data must equal grad(phi') . n on the rotated mesh
    mesh = case.rotated_polygon(128, h)
    data = case.heading_neumann_data(mesh.centers, h)
    # finite difference of phi' along the outward normal
    eps = 1e-6
    fd = (case.heading_perturbation_potential(mesh.centers + eps * mesh.normals, h)
          - case.heading_perturbation_potential(mesh.centers - eps * mesh.normals, h)) / (2 * eps)
    print("neumann data vs fd", np.max(np.abs(data - fd)) / np.max(np.abs(data)))
    # lattice rotation maps the mesh onto itself
    lat = case.rotated_polygon(128, math.radians(45.0))
    base = case.regular_polygon(128, 1.0)
    d = np.array([np.min(np.linalg.norm(base.vertices - v, axis=1)) for v in lat.vertices])
    print("lattice vertex set distance", np.max(d))

    for panels in (32, 64, 128):
        study = case.heading_study(panels)
        print("N=", panels, "headings", study["headings_deg"])
        for row in study["levels"]:
            print("  %.6f deg D=%.3e Nsurf=%.3e normal=%.3e cond=%.2f/%.2f" % (
                row["heading_deg"], row["dirichlet_velocity_relative_l2"],
                row["neumann_surface_velocity_relative_l2"],
                row["normal_velocity_residual_over_speed"],
                row["dirichlet_condition_number"], row["neumann_condition_number"],
            ))
        print("  penalty D=%.3f N=%.3f  lattice cov=%.3e  lattice errs=%.3e/%.3e" % (
            study["dirichlet_penalty_max"], study["neumann_penalty_max"],
            study["lattice"]["velocity_field_max_difference_over_speed"],
            study["lattice"]["dirichlet_velocity_relative_l2"],
            study["lattice"]["neumann_surface_velocity_relative_l2"],
        ))


if __name__ == "__main__":
    main()