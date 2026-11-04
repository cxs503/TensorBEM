"""One-shot helper: insert floating_sphere_hull_mesh into geometry3d.py (idempotent)."""
from pathlib import Path

path = Path("/root/tp-agent/work/m3727/src/tensorbem/geometry3d.py")
text = path.read_text()

marker = "def _oriented_triangle("
assert marker in text, "insertion marker not found"
if "def floating_sphere_hull_mesh(" in text:
    print("already present, skipping")
    raise SystemExit(0)

snippet = '''def floating_sphere_hull_mesh(
    subdivisions: int,
    radius: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, -0.5),
) -> WettedHullMesh:
    """Return the wetted cap of a floating sphere truncated at ``z = 0``.

    The body is the sphere of ``radius`` centred at ``center = (x0, y0, z0)``
    with ``-radius < z0 < 0`` (centre below the calm waterplane, sphere
    piercing it). The returned :class:`WettedHullMesh` contains the part of
    the sphere with ``z <= 0``; its open boundary is the circular waterline of
    radius ``sqrt(radius**2 - z0**2)`` centred at ``(x0, y0)``.

    The geodesic sphere of :func:`icosphere` is clipped against the plane
    ``z = 0`` triangle by triangle (Sutherland-Hodgman). Intersection points
    on a shared edge are welded by exact coordinates, so the waterline
    boundary closes. Clipping preserves the counter-clockwise ordering of the
    source panels, hence panel normals stay outward (into the fluid).
    """
    if isinstance(subdivisions, (bool, np.bool_)) or not isinstance(
        subdivisions, (int, np.integer)
    ):
        raise ValueError("subdivisions must be a non-negative integer")
    if subdivisions < 0:
        raise ValueError("subdivisions must be a non-negative integer")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number")
    c = np.asarray(center, dtype=float)
    if c.shape != (3,) or not np.isfinite(c).all():
        raise ValueError("center must be a finite 3-tuple")
    if not (-radius < c[2] < 0.0):
        raise ValueError(
            "center z must lie strictly between -radius and 0 "
            "(the sphere must pierce the calm waterplane)"
        )

    base = icosphere(subdivisions, radius=radius)
    verts = base.vertices + c
    faces = base.faces

    new_verts: list[tuple[float, float, float]] = []
    vertex_map: dict[int, int] = {}
    point_map: dict[tuple[float, float], int] = {}

    def weld_below(index: int) -> int:
        if index not in vertex_map:
            z = float(verts[index, 2])
            vertex_map[index] = len(new_verts)
            new_verts.append(
                (
                    float(verts[index, 0]),
                    float(verts[index, 1]),
                    0.0 if abs(z) < 1.0e-15 else z,
                )
            )
        return vertex_map[index]

    def weld_on_waterline(point: np.ndarray) -> int:
        key = (float(point[0]), float(point[1]))
        if key not in point_map:
            point_map[key] = len(new_verts)
            new_verts.append((key[0], key[1], 0.0))
        return point_map[key]

    new_faces: list[tuple[int, int, int]] = []
    for a, b, cc in faces:
        z = verts[[a, b, cc], 2]
        if np.all(z > 0.0):
            continue  # panel entirely above the waterplane
        if np.all(z <= 0.0):
            new_faces.append((weld_below(a), weld_below(b), weld_below(cc)))
            continue
        # Clip the triangle against z <= 0 (Sutherland-Hodgman). Each output
        # point is tagged with its provenance: an original vertex index or a
        # cut point on the waterplane.
        order = (a, b, cc)
        polygon: list[tuple[str, object]] = []
        for i in range(3):
            current = verts[order[i]]
            following = verts[order[(i + 1) % 3]]
            current_in = current[2] <= 0.0
            following_in = following[2] <= 0.0
            if current_in:
                polygon.append(("orig", order[i]))
            if current_in != following_in:
                t = -current[2] / (following[2] - current[2])
                if 0.0 < t < 1.0:
                    polygon.append(("cut", current + t * (following - current)))
        if len(polygon) < 3:
            continue
        coords = np.asarray(
            [p if kind == "cut" else verts[i] for kind, p in polygon], dtype=float
        )
        area = 0.0
        for k in range(1, len(coords) - 1):
            tri = np.array([coords[0], coords[k], coords[k + 1]])
            area += 0.5 * float(
                np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            )
        if area <= 1.0e-14 * radius * radius:
            continue  # sliver on the waterplane: no fluid-side area
        indices = [
            weld_below(i) if kind == "orig" else weld_on_waterline(p)
            for kind, p in polygon
        ]
        for k in range(1, len(coords) - 1):
            new_faces.append((indices[0], indices[k], indices[k + 1]))

    return WettedHullMesh(
        np.asarray(new_verts, dtype=float), np.asarray(new_faces, dtype=np.int64)
    )


'''

text = text.replace(marker, snippet + marker, 1)
path.write_text(text)
print("inserted")
