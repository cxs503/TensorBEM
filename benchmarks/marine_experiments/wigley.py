"""Report 0909 hull family and the four free-motion mass properties."""

import numpy as np
from tensorbem.geometry3d import WettedHullMesh
from tensorbem.wigley3_hull import wigley3_half_breadth, wigley3_wetted_hull_mesh

L, T = 3.0, 0.1875
# Report 0909 p.2: beam, displaced mass, centre of gravity above calm water.
HULLS = {
    "I": (0.30, 94.6, -0.0175),
    "II": (0.60, 189.2, 0.0),
    "III": (0.30, 78.0, -0.0175),
    "IV": (0.60, 156.0, 0.0),
}


def fullness(hull, x, z):
    if hull in ("I", "II"):
        return (
            HULLS[hull][0]
            / 2
            * (z / T) ** 2
            * (1 - (z / T) ** 8)
            * (1 - (2 * x / L) ** 2) ** 4
        )
    return 0.0


def sections(hull, x, z):
    return wigley3_half_breadth(x, z, B=HULLS[hull][0]) + fullness(hull, x, z)


def mesh_for(hull, nx, nz):
    mesh = wigley3_wetted_hull_mesh(nx=nx, nz=nz, B=HULLS[hull][0])
    if hull in ("III", "IV"):
        return mesh
    v = mesh.vertices.copy()
    v[:, 1] += np.sign(v[:, 1]) * fullness(hull, v[:, 0], v[:, 2])
    return WettedHullMesh(v, mesh.faces)
