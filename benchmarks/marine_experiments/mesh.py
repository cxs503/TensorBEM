"""Weld a keel-to-waterline half-hull patch and its port/starboard mirror."""

import numpy as np
from tensorbem.geometry3d import WettedHullMesh


def mesh_from_patch(patch):
    # Each half covers forebody and afterbody. Weld the actual seam, keel and
    # stem nodes rather than adding fictitious closure panels.
    vertices = []
    index = {}
    faces = []

    def vertex(v):
        key = tuple(np.round(v, 10))
        if key not in index:
            index[key] = len(vertices)
            vertices.append(key)
        return index[key]

    for sign in (1, -1):
        nodes = np.array([[vertex(v * [1, sign, 1]) for v in row] for row in patch])
        for i, j in np.ndindex(nodes.shape[0] - 1, nodes.shape[1] - 1):
            a, b, c, d = (
                nodes[i, j],
                nodes[i, j + 1],
                nodes[i + 1, j + 1],
                nodes[i + 1, j],
            )
            for tri in ((a, b, c), (a, c, d)):
                if len(set(tri)) == 3 and any(vertices[v][1] != 0 for v in tri):
                    faces.append(tri if sign > 0 else tri[::-1])
    used, compact = np.unique(faces, return_inverse=True)
    return WettedHullMesh(np.array(vertices)[used], compact.reshape(-1, 3))
