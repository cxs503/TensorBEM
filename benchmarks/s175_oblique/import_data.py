"""Read the published S-175 MGF stations and ITTC markers replotted by Rosa (2019)."""

from hashlib import sha256
from pathlib import Path
import argparse
import numpy as np
from benchmarks.marine_experiments.io import write

ROOT = Path(__file__).resolve().parent
MGF = "f804f537c4369b1011fc508a01e00583b5a912e08e8e95071a8bb7cf57883b9b"
PDF = "ce710f9b7b9b1ca66a3d69923a949f4d4e4118e8cfd8036671bc4f341b16cb2b"
GRIDS = ((6, 1), (8, 2))
SCALE = 70.0


def geometry(path):
    if sha256(path.read_bytes()).hexdigest() != MGF:
        raise ValueError("Unexpected S-175 MGF")
    lines = path.read_text().splitlines()[5:]
    i = 0
    stations = []
    while i < len(lines):
        number, x, count = int(lines[i]), -float(lines[i + 1]), int(lines[i + 2])
        i += 3
        if number != 24 - len(stations):
            raise ValueError("unexpected station sequence")
        yz = np.loadtxt(lines[i : i + count])[::-1]
        i += count
        # MGF x is aft, X in the solver is forward. Heights are above baseline.
        y, z = yz.T
        wet = z < 9.5
        y = np.r_[y[wet], np.interp(9.5, z, y)]
        z = np.r_[z[wet], 9.5] - 9.5
        arc = np.r_[0, np.cumsum(np.hypot(np.diff(y), np.diff(z)))]
        stations.append((x, y, z, arc))
    packed = {}
    for contours, subdivisions in GRIDS:
        strips = []
        for x, y, z, arc in stations:
            s = np.linspace(0, arc[-1], contours + 1)
            strips.append(
                np.column_stack(
                    (np.full(len(s), x), np.interp(s, arc, y), np.interp(s, arc, z))
                )
            )
        originals = np.stack(strips, axis=1)
        strips = [originals[:, 0]]
        for j in range(1, len(stations)):
            if subdivisions == 2 and abs(stations[j][0] - stations[j - 1][0]) > 7:
                strips.append(0.5 * (originals[:, j - 1] + originals[:, j]))
            strips.append(originals[:, j])
        # Planar physical closures at the first and last supplied stations.
        bow, stern = strips[0].copy(), strips[-1].copy()
        bow[:, 1] = 0
        stern[:, 1] = 0
        packed[f"patch{contours}"] = np.stack([bow, *strips, stern], axis=1) / SCALE
    np.savez_compressed(ROOT / "geometry.npz", **packed)


def measurements(path):
    import fitz

    if sha256(path.read_bytes()).hexdigest() != PDF:
        raise ValueError("Unexpected Rosa paper")
    with fitz.open(path) as doc:
        draw = doc[4].get_drawings()
        words = doc[4].get_text("words")
    frames = [
        d["rect"]
        for d in draw
        if d["type"] == "f"
        and 105.4 < d["rect"].width < 105.5
        and 83.2 < d["rect"].height < 83.4
    ]
    if len(frames) != 9:
        raise ValueError("Expected nine panels")
    # Limits from printed axis ticks. Match colour/shape to the actual legend in each panel.
    scales = [
        ("heave", 180, 1, 7, 2),
        ("heave", 90, 0, 18, 1.4),
        ("heave", 60, 0, 6, 1),
        ("roll", 150, 0, 7, 3.5),
        ("roll", 90, 0, 18, 7),
        ("roll", 30, 0.5, 4.5, 3),
        ("pitch", 180, 1, 7, 1.4),
        ("pitch", 90, 1, 4, 0.5),
        ("pitch", 30, 0, 12, 1.2),
    ]
    result = []
    for rect, (mode, angle, x0, x1, ymax) in zip(frames, scales, strict=True):
        markers = []
        for d in draw:
            box = d["rect"]
            cx = (box.x0 + box.x1) / 2
            cy = (box.y0 + box.y1) / 2
            if not (
                rect.x0 - 1 < cx < rect.x1 + 1
                and rect.y0 - 1 < cy < rect.y1 + 1
                and 2.4 < box.width < 2.9
                and len(d["items"]) == 4
            ):
                continue
            markers.append((cx, cy, d["color"], d["type"]))
        labels = [
            w
            for w in words
            if w[4] in ("SRI", "IHI", "SHI") and rect.contains(fitz.Point(w[0], w[1]))
        ]
        legend = {}
        for w in labels:
            candidate = min(
                (m for m in markers if w[0] - 14 < m[0] < w[0] - 1),
                key=lambda m: abs(m[1] - (w[1] + w[3]) / 2),
            )
            if abs(candidate[1] - (w[1] + w[3]) / 2) > 1:
                raise ValueError("unmatched legend")
            legend[candidate[2:]] = (w[4], candidate[:2])
        for x, y, col, kind in markers:
            lab, loc = legend[(col, kind)]
            if (x, y) == loc:
                continue
            result.append(
                dict(
                    quantity=mode,
                    heading_deg=angle,
                    laboratory=lab,
                    omega_sqrt_L_g=round(
                        x0 + (x - rect.x0) / rect.width * (x1 - x0), 3
                    ),
                    observed=round((rect.y1 - y) / rect.height * ymax, 3),
                )
            )
    write(ROOT / "observations.csv", list(result[0]), result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mgf", type=Path)
    p.add_argument("paper", type=Path)
    a = p.parse_args()
    geometry(a.mgf)
    measurements(a.paper)
