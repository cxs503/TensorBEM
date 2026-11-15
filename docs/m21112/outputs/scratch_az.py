import sys, time
import numpy as np
sys.path.insert(0, "/root/tp-agent/work/m21112/src")
from tensorbem.geometry import regular_polygon
from tensorbem.solver import solve_laplace_exterior_neumann, evaluate_exterior_field
from tensorbem.benchmarks.case_cylinder_stream import exact_velocity

radius = speed = 1.0
FACTORS = (2.0, 1.0, 0.5, 0.25, 0.1)
OFFSETS = (0.0, 0.125, 0.25, 0.375, 0.5)


def min_dist_to_boundary(points, vertices):
    ends = np.roll(vertices, -1, axis=0)
    out = np.full(len(points), np.inf)
    for a, b in zip(vertices, ends):
        ab = b - a
        ap = points - a
        s = np.clip(ap @ ab / (ab @ ab), 0.0, 1.0)
        proj = a + s[:, None] * ab
        out = np.minimum(out, np.linalg.norm(points - proj, axis=1))
    return out


t0 = time.time()
for panels in (32, 64, 128):
    mesh = regular_polygon(panels, radius)
    c, v, L = mesh.centers, mesh.vectors, mesh.lengths
    hmax = L.max()
    t = v / L[:, None]
    n = np.column_stack((t[:, 1], -t[:, 0]))
    nxt = np.roll(n, -1, axis=0)
    bis = n + nxt
    bis /= np.linalg.norm(bis, axis=1)[:, None]
    r = np.hypot(c[:, 0], c[:, 1])
    qe = radius**2 * (c[:, 0] / r) / r**2
    res = solve_laplace_exterior_neumann(mesh, qe)
    resl = solve_laplace_exterior_neumann(mesh, qe, linear_density=True)

    stride = max(1, panels // 6)
    sel = np.arange(0, panels, stride)
    c, v, L, t, n, bis, qe = c[sel], v[sel], L[sel], t[sel], n[sel], bis[sel], qe[sel]
    blocks = []
    tags = []
    for factor in FACTORS:
        delta = factor * hmax
        for off in OFFSETS:
            if off >= 0.5:
                base = c + 0.5 * hmax * t
                d = bis
            else:
                base = c + off * hmax * t
                d = n
            blocks.append(base + delta * d)
            tags.append((factor, off))
    pts = np.vstack(blocks)
    exact = exact_velocity(pts, radius, speed)
    sp = np.linalg.norm(exact, axis=1)
    f = evaluate_exterior_field(mesh, res, pts)
    fl = evaluate_exterior_field(mesh, resl, pts, linear_density=True)
    e = np.linalg.norm(f.gradient + np.array([speed, 0.0]) - exact, axis=1) / sp
    el = np.linalg.norm(fl.gradient + np.array([speed, 0.0]) - exact, axis=1) / sp
    clear = min_dist_to_boundary(pts, mesh.vertices) / hmax
    e = e.reshape(len(FACTORS), len(OFFSETS), len(sel))
    el = el.reshape(len(FACTORS), len(OFFSETS), len(sel))
    clear = clear.reshape(len(FACTORS), len(OFFSETS), len(sel))
    print("=" * 100)
    print("panels", panels, "h_max", round(hmax, 6))
    for i, factor in enumerate(FACTORS):
        print(" delta/h=%.2f" % factor)
        for j, off in enumerate(OFFSETS):
            print("   off=%.3f  e=%.4e  e_lin=%.4e  minclear/h=%.4f" %
                  (off, e[i, j].max(), el[i, j].max(), clear[i, j].min()))
print("wall", round(time.time() - t0, 1))