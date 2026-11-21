"""Import NMRI's KRISO KVLCC2 mesh and body plan from the Gothenburg workshop archives."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import argparse
import re
from zipfile import ZipFile
import numpy as np
from tensorbem.geometry3d import WettedHullMesh

ROOT = Path(__file__).resolve().parent
L = 4.0
SCALE = 80.0
GRIDS = (
    (9, 6, 16),
    (14, 10, 24),
)  # panels on each half-length, girth divisions, section panels
GRID_SHA = "5d2b8023ec0efdb0e8c3c979a0e0d68ead56cd8c77e31a4da3500a06800b2d2a"
OFFSET_SHA = "693dfc1b07fe4759e2fa98ff8b6ac66c47ac579d14afae3f253a157d58a6e01f"


def checked(path, digest):
    b = path.read_bytes()
    if sha256(b).hexdigest() != digest:
        raise ValueError("unexpected NMRI archive")
    return ZipFile(BytesIO(b))


def read_patches(path):
    patches = []
    with checked(path, GRID_SHA) as z:
        for name in ("kvlcc_bow1.dat", "kvlcc2_stn1.dat"):
            lines = z.read(name).decode().splitlines()
            ni, nj = map(int, re.findall(r"\d+", lines[1])[:2])
            patches.append(np.loadtxt(lines[2:]).reshape(nj, ni, 3))
    if np.max(abs(patches[0][:, -1] - patches[1][:, 0])) > 5e-6:
        raise ValueError("incompatible midship seam")
    return patches


def read_offsets(path):
    with checked(path, OFFSET_SHA) as z:
        lines = z.read("kvlcc2.fix").decode().splitlines()
    data = np.fromstring(" ".join(lines[2:]), sep=" ")
    rows = []
    j = 0
    for i in range(int(lines[1])):
        x, count = data[j : j + 2]
        n = int(count)
        j += 2
        half = data[j : j + n]
        height = data[j + n : j + 2 * n]
        j += 2 * n
        rows.append((x, half, height))
    if j != len(data):
        raise ValueError("unconsumed body plan")
    return rows


def wet_contour(half, height, panels):
    draft = 20.8
    width = np.interp(draft, height, half)
    if width <= 0:
        return None
    mask = height < draft
    y = np.r_[half[mask], width]
    h = np.r_[height[mask], draft]
    # Follow the wetted component connected to the waterline; isolated shaft
    # sections below it do not meet the surface-piercing sectional BVP.
    start = np.flatnonzero(y <= 0)[-1] if np.any(y <= 0) else 0
    y = y[start:]
    h = h[start:]
    if y[0] != 0:
        y = np.r_[0, y]
        h = np.r_[h[0], h]
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(y), np.diff(h)))]
    t = np.linspace(0, arc[-1], panels // 2 + 1)
    return (
        np.column_stack((np.interp(t, arc, y), (np.interp(t, arc, h) - draft))) / SCALE
    )


def extract(grid, offsets, output):
    patches = read_patches(grid)
    stations = read_offsets(offsets)
    packed = {}
    for n, g, p in GRIDS:
        indices = np.round(np.linspace(0, patches[0].shape[1] - 1, n + 1)).astype(int)
        rows = np.round(np.linspace(0, patches[0].shape[0] - 1, g + 1)).astype(int)
        bow, stern = [patch[np.ix_(rows, indices)] for patch in patches]
        half = np.concatenate((bow, stern[:, 1:]), axis=1) * L
        half[:, :, 0] *= -1  # x forward, about amidships
        packed[f"patch{n}"] = half
        profiles = []
        xs = []
        for x, y, z in stations:
            line = wet_contour(y, z, p)
            if line is not None:
                xs.append((x - 160) / SCALE)
                profiles.append(line)
        packed[f"xs{n}"] = xs
        packed[f"sections{n}"] = profiles
    np.savez_compressed(output, **packed)


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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("grid", type=Path)
    p.add_argument("offsets", type=Path)
    a = p.parse_args()
    extract(a.grid, a.offsets, ROOT / "geometry.npz")
