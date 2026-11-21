"""Recover experimental vector marker centres from two fixed publication copies (PyMuPDF)."""

from hashlib import sha256
from pathlib import Path
import argparse
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "kvlcc2": ("a4b815f14fb71736300d5efd0c2b92c10399212808485560509108317d29ce3e", 3),
    "floe": ("8ba4135d949c4ee1aeebc19d9a3bdef21fdf33f2e893d6f3e733208bd38f622b", 16),
}


def kvlcc2(drawings):
    # Tick extrema (0..2.5 on x) and vector plot frames, page 469 figs 5-6.
    # Select Lee filled blue circles in Fig 5 and Lee filled blue squares in Fig 6.
    panels = [
        ("heave", 68.608, 209.719, 277.015, 400.111, 2),
        ("pitch", 70.365, 435.589, 279.323, 626.582, 2),
        ("resistance", 339.439, 61.081, 548.473, 252.147, 10),
    ]
    rows = []
    for name, x0, y0, x1, y1, height in panels:
        for d in drawings:
            r = d["rect"]
            x = (r.x0 + r.x1) / 2
            y = (r.y0 + r.y1) / 2
            if not (
                d["type"] == "f"
                and d["fill"]
                and d["fill"][2] > 0.99
                and d["fill"][0] < 0.2
            ):
                continue
            if not (x0 < x < x1 and y0 + 45 < y <= y1 + 1):
                continue  # excludes all legend markers
            if name == "resistance":
                if len(d["items"]) != 1 or d["items"][0][0] != "re":
                    continue
            elif len(d["items"]) != 11 or not 5.4 < r.width < 5.8:
                continue
            rows.append(
                [
                    name,
                    round(2.5 * (x - x0) / (x1 - x0), 2),
                    round(height * (y1 - y) / (y1 - y0), 3),
                ]
            )
    if [sum(r[0] == n for r in rows) for n, *_ in panels] != [9, 8, 8]:
        raise ValueError("Unexpected marker count")
    return ["quantity", "lambda_L", "observed"], rows


def floe_markers(drawings, tops):
    # Fig 14: published axes span ka = 0.04..0.16 and energy fraction -0.2..1.2.
    # Select one unique stroked path per plotted shape, excluding whiskers.
    frames = [
        (d["rect"].x0, d["rect"].y0, d["rect"].y1)
        for d in drawings
        if d["color"]
        and 0.14 < d["color"][0] < 0.15
        and 100 < d["rect"].height < 102
        and d["rect"].width == 0
    ]
    rows = []
    for material, top in zip(("PP", "PVC"), tops):
        for period, left in zip((0.8, 0.9, 1.0), (96, 241, 386)):
            x0, y0, y1 = next(
                f for f in frames if abs(f[0] - left) < 0.3 and abs(f[1] - top) < 0.45
            )
            x1 = next(f[0] for f in frames if 128 < f[0] - x0 < 130 and f[1] == y0)
            for d in drawings:
                r = d["rect"]
                x = (r.x0 + r.x1) / 2
                y = (r.y0 + r.y1) / 2
                if not (x0 < x < x1 and y0 < y < y1 and d["color"]):
                    continue
                col = d["color"]
                items = d["items"]
                if len(items) == 14 and col[2] > 0.99 and col[0] < 0.01:
                    kind = "barriers"
                elif len(items) == 1 and items[0][0] == "re" and 0.49 < col[1] < 0.51:
                    kind = "free"
                elif (
                    len(items) == 1
                    and items[0][0] == "l"
                    and col[:3] == (1.0, 0.0, 1.0)
                    and 4.9 < r.width < 5.1
                    and r.height == 0
                ):
                    kind = "moored"
                else:
                    continue
                rows.append(
                    [
                        material,
                        period,
                        kind,
                        round(0.04 + 0.12 * (x - x0) / (x1 - x0), 4),
                        round(-0.2 + 1.4 * (y1 - y) / (y1 - y0), 4),
                    ]
                )
    if len(rows) != 90:
        raise ValueError(f"Expected 90 measured markers, got {len(rows)}")
    return sorted(rows)


def floe(pdf, transmission_page):
    reflected = floe_markers(pdf[transmission_page - 1].get_drawings(), (121.2, 236.0))
    transmitted = floe_markers(pdf[transmission_page].get_drawings(), (150.0, 265.0))
    rows = []
    for r, t in zip(reflected, transmitted, strict=True):
        if r[:3] != t[:3] or abs(r[3] - t[3]) > 0.001:
            raise ValueError("Unmatched reflection and transmission markers")
        rows.append([*t, r[3], r[4]])
    return [
        "material",
        "period",
        "configuration",
        "ka",
        "observed_energy_T",
        "ka_R",
        "observed_energy_R",
    ], rows


def extract(name, path):
    import fitz

    checksum, page = SOURCES[name]
    if sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError("Unexpected publication PDF")
    with fitz.open(path) as pdf:
        fields, rows = (
            floe(pdf, page) if name == "floe" else kvlcc2(pdf[page].get_drawings())
        )
    dest = (
        ROOT
        / {"floe": "floe_transmission", "kvlcc2": "kvlcc2_head_seas"}[name]
        / "observations.csv"
    )
    with dest.open("w", newline="") as stream:
        w = csv.writer(stream, lineterminator="\n")
        w.writerow(fields)
        w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=SOURCES)
    p.add_argument("pdf", type=Path)
    a = p.parse_args()
    extract(a.case, a.pdf)
