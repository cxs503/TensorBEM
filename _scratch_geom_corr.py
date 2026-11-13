"""Scratch: does a geometry-exact (curved) panel description fix the coarse-mesh
radiation-coefficient bias of the flat-panel Kelvin BEM on a submerged sphere?

Variants (all use the *same* flat-triangle quadrature; only the panel
description fed to the operators/integration changes):

  V0  baseline               : flat centroid / flat normal / flat area
  V1  exact surface measure  : flat centroid / flat normal / spherical patch area
  V2  + exact normals        : flat centroid / radial normal / spherical patch area
  V3  + exact collocation    : radially projected centroid / radial normal / patch area
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.free_surface_solver3d import _build_target_operators_3d
from tensorbem.geometry3d import submerged_sphere_mesh

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665


def spherical_triangle_area(v0, v1, v2, radius):
    """Exact area of the spherical triangle through three surface points."""
    def unit(v):
        c = v - np.array([0.0, 0.0, 0.0])
        return c / np.linalg.norm(c, axis=1, keepdims=True)

    a, b, c = unit(v0), unit(v1), unit(v2)
    num = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    den = 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a)
    return 2.0 * np.arctan2(num, den) * radius ** 2


class Panels:
    """Duck-typed stand-in for SurfaceMesh3D with overridable geometry."""

    def __init__(self, mesh, areas, normals, centroids):
        self.v0, self.v1, self.v2 = mesh.v0, mesh.v1, mesh.v2
        self.areas, self.normals, self.centroids = areas, normals, centroids
        self.panel_count = mesh.panel_count
        self.faces = mesh.faces
        self.vertices = mesh.vertices


def variants(mesh, center):
    flat_area, flat_n, flat_c = mesh.areas, mesh.normals, mesh.centroids
    radius = RADIUS
    # spherical patch area for each facet (vertices of the icosphere already lie on the sphere)
    sph_area = spherical_triangle_area(mesh.v0 - center, mesh.v1 - center, mesh.v2 - center, radius)
    radial = (flat_c - center) / np.linalg.norm(flat_c - center, axis=1, keepdims=True)
    proj_c = center + radius * radial
    out = {
        "V0": Panels(mesh, flat_area, flat_n, flat_c),
        "V1": Panels(mesh, sph_area, flat_n, flat_c),
        "V2": Panels(mesh, sph_area, radial, flat_c),
        "V3": Panels(mesh, sph_area, radial, proj_c),
    }
    return out, sph_area


def radiation(pan, nu, center):
    n6 = np.hstack((pan.normals, np.cross(pan.centroids - center, pan.normals)))
    n_modes = n6[:, [0, 2]]  # surge(0), heave(2)
    q_ext = (-n_modes).astype(np.complex128)
    s_body, h_body = _build_target_operators_3d(pan.centroids, pan, nu, self_interaction=True, rankine_order=5)
    system = 0.5 * np.eye(pan.panel_count, dtype=np.complex128) - h_body
    potentials = np.linalg.solve(system, s_body @ q_ext)
    weighted = pan.areas[:, None] * n_modes
    imp = -DENSITY * (weighted.T @ potentials)
    omega = np.sqrt(GRAVITY * nu)
    added = imp.real
    damping = omega * imp.imag
    return added[0, 0], damping[0, 0], added[1, 1], damping[1, 1]


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="2")
    ap.add_argument("--variants", default="V0,V1,V2,V3")
    ap.add_argument("--band", default="0.1,0.5,1.0,2.0,3.0")
    ap.add_argument("--out", default="/tmp/geom_corr.json")
    a = ap.parse_args()
    subs = [int(x) for x in a.subs.split(",")]
    keep = a.variants.split(",")
    band = tuple(float(x) for x in a.band.split(","))
    results = {}
    center = np.array([0.0, 0.0, -SUBMERGENCE])
    for sub in subs:
        mesh = submerged_sphere_mesh(sub, radius=RADIUS, submergence=SUBMERGENCE)
        pans, sph_area = variants(mesh, center)
        print(f"--- subdivision {sub}: {mesh.panel_count} panels; "
              f"flat area {mesh.surface_area:.4f} vs spherical {sph_area.sum():.4f} "
              f"(deficit {1 - mesh.surface_area / sph_area.sum():.5f})")
        pans = {k: v for k, v in pans.items() if k in keep}
        hdr = "ka    " + "  ".join(f"{k}: a11   b11   a33   b33" for k in pans)
        print(hdr)
        for ka in band:
            nu = ka / RADIUS
            ref = submerged_sphere_wave_multipole(nu, radius=RADIUS, submergence=SUBMERGENCE,
                                                 density=DENSITY, gravity=GRAVITY)
            vol = ref.displaced_volume
            omega = ref.omega
            row = f"{ka:4.1f} "
            for key, pan in pans.items():
                a11, b11, a33, b33 = radiation(pan, nu, center)
                a11 /= DENSITY * vol
                b11 /= DENSITY * omega * vol
                a33 /= DENSITY * vol
                b33 /= DENSITY * omega * vol
                e = [abs(a11 - ref.surge_added_mass_dimless) / ref.surge_added_mass_dimless,
                     abs(b11 - ref.surge_damping_dimless) / ref.surge_damping_dimless,
                     abs(a33 - ref.heave_added_mass_dimless) / ref.heave_added_mass_dimless,
                     abs(b33 - ref.heave_damping_dimless) / ref.heave_damping_dimless]
                row += "  " + " ".join(f"{100 * v:5.2f}" for v in e)
            print(row)
            sys.stdout.flush()
            results.setdefault(str(sub), {}).setdefault(f"{ka:g}", {})[key] = e
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()