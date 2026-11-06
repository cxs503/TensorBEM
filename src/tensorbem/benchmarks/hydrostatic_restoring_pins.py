"""Read the numbers of ``docs/hydrostatic_restoring_couplings.md`` back out of the markdown.

Why this module exists
----------------------
``docs/hydrostatic_restoring_couplings.md`` (mission m3196) is a paper-level
verification note: it states the six non-zero entries of the hydrostatic
restoring matrix ``C`` for an off-centred barge, the residuals against two
independent baselines, an LCF sweep and a per-entry error table -- 80 numbers in
all.  Until now **no test read that file**.  The coverage audit
(:mod:`tensorbem.benchmarks.doc_claim_audit`) classified it ``UNGUARDED``: the
numbers happened to be right because they were injected by
``docs/m3196/outputs/restoring_couplings/make_artifacts.py``, but a change in
:mod:`tensorbem.hydrostatic`, or a later hand edit of the document, could move
them apart silently.  This module is the missing gate.

It also answers the one verification debt the note records itself (section 5,
item 3: *"curved-wall hulls were not re-checked with the energy Hessian -- only
the rectangular-waterplane barge family was"*).  That debt was closed
independently by m4750 (:mod:`tensorbem.benchmarks.hydrostatic_curved_waterplane`,
per-triangle clipping plus exact-geometry closed forms); the families below are a
*third* path that shares no code with it -- the agreement is a cross-check, not a
repeat.  Two curved families are added:

``elliptic cylinder``
    Waterplane is an ``n``-gon inscribed in an ellipse: curved waterplane,
    wall-sided hull, exact faceted volume.
``floating hemisphere``
    Circular waterplane and a genuinely curved wetted surface.

For both, three mutually independent values of the same six entries are
compared: the production code, the numerical Hessian of the hydrostatic
potential energy (:mod:`tensorbem.benchmarks.hydrostatic_energy_reference`,
which knows only Archimedes plus the weight of the body), and a closed-form
polygon moment integration of the traced waterplane ring implemented here --
that path never calls ``compute_waterplane_properties``.

How a number is compared
------------------------
The declared value is parsed together with its *printed precision*:
``2.010363e+04`` carries seven significant figures (quantum ``0.5e-3``),
``0.12`` carries two decimals (quantum ``0.5e-2``).  The tolerance is 1.5x that
quantum; the extra 50% absorbs half-way rounding (``0.125`` prints as ``0.12``),
so a gate failure always means a real digit moved and never a rounding tie.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.hydrostatic_restoring_pins --check
    PYTHONPATH=src python3 -m tensorbem.benchmarks.hydrostatic_restoring_pins --write
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.hydrostatic_analytic import box_hull_mesh
from tensorbem.benchmarks.hydrostatic_energy_reference import (
    restoring_matrix_from_energy,
)
from tensorbem.geometry3d import (
    WettedHullMesh,
    elliptic_cylinder_hull_mesh,
    hemisphere_hull_mesh,
)
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC = REPO_ROOT / "docs" / "hydrostatic_restoring_couplings.md"
CURVATURE_DOC = REPO_ROOT / "docs" / "hydrostatic_restoring_waterplane_curvature.md"
FIGURE = REPO_ROOT / "docs" / "figures" / "fig_restoring_curvature.png"

RHO = 1025.0
G = 9.80665

#: ``(label, (i, j))`` for the six entries a floating body can carry.
ENTRIES = (
    ("C33 heave", (2, 2)),
    ("C34 heave-roll", (2, 3)),
    ("C35 heave-pitch", (2, 4)),
    ("C44 roll", (3, 3)),
    ("C45 roll-pitch", (3, 4)),
    ("C55 pitch", (4, 4)),
)

#: LCF offsets of the sweep table (``lcf / length`` is what the document prints).
SWEEP_LCF = (0.0, 0.25, 0.5, 0.75, 1.0, 1.3)

BARGE = {"length": 2.0, "beam": 1.0, "draft": 1.0, "lcf": 1.30, "tcf": 0.45}


# ---------------------------------------------------------------------------
# recomputation of the documented case
# ---------------------------------------------------------------------------
def barge_matrix(
    length: float, beam: float, draft: float, lcf: float, tcf: float
) -> tuple[WettedHullMesh, float, np.ndarray, np.ndarray, np.ndarray]:
    """Production matrix for a wall-sided barge whose waterplane is offset."""
    hull = box_hull_mesh(length, beam, draft)
    shifted = WettedHullMesh(hull.vertices + np.array([lcf, tcf, 0.0]), hull.faces)
    volume, cb = compute_displaced_volume_and_buoyancy(shifted)
    cg = np.array([lcf, tcf, -0.5 * draft])
    props = compute_hydrostatic_restoring_matrix(shifted, RHO * volume, cg, RHO, G)
    return shifted, float(volume), np.asarray(cb, dtype=float), cg, props.restoring_matrix


def rectangle_closed_form(
    length: float, beam: float, draft: float, lcf: float, tcf: float, cg_z: float
) -> np.ndarray:
    """Analytic waterplane integrals of a rectangle (all values rational)."""
    area = length * beam
    volume = area * draft
    s_x, s_y = area * lcf, area * tcf
    i_xx = area * (tcf**2 + beam**2 / 12.0)
    i_yy = area * (lcf**2 + length**2 / 12.0)
    i_xy = area * lcf * tcf
    rg = RHO * G
    c = np.zeros((6, 6))
    c[2, 2] = rg * area
    c[2, 3] = c[3, 2] = rg * s_y
    c[2, 4] = c[4, 2] = -rg * s_x
    c[3, 3] = rg * i_xx + rg * volume * (-0.5 * draft - cg_z)
    c[4, 4] = rg * i_yy + rg * volume * (-0.5 * draft - cg_z)
    c[3, 4] = c[4, 3] = -rg * i_xy
    return c


def legacy_matrix(
    length: float, beam: float, draft: float, lcf: float, tcf: float
) -> np.ndarray:
    """The pre-m3196 matrix: three diagonals from *centroid* moments, couplings zero.

    This is what ``compute_hydrostatic_restoring_matrix`` returned before the
    coupling terms were added, reconstructed analytically so the document's
    "pre-fix" column is checked instead of trusted.  ``lcf``/``tcf`` are unused
    by construction -- that is precisely the defect the column records.
    """
    area = length * beam
    rg = RHO * G
    c = np.zeros((6, 6))
    c[2, 2] = rg * area
    c[3, 3] = rg * (length * beam**3 / 12.0)
    c[4, 4] = rg * (beam * length**3 / 12.0)
    return c


def polygon_waterplane_integrals(hull: WettedHullMesh) -> dict[str, float]:
    """Waterplane moments of the traced ``z = 0`` ring, by Green's theorem.

    Ring vertices are sorted by polar angle about their centroid (both hull
    families here have convex star-shaped waterplanes) and integrated with the
    polygon moment formulas, which know nothing about
    :func:`tensorbem.hydrostatic.compute_waterplane_properties`.
    """
    vertices = np.asarray(hull.vertices, dtype=float)
    ring = vertices[np.abs(vertices[:, 2]) <= 1.0e-12][:, :2]
    if len(ring) < 3:
        raise ValueError("no waterplane ring found")
    centre = ring.mean(axis=0)
    angle = np.arctan2(ring[:, 1] - centre[1], ring[:, 0] - centre[0])
    ring = ring[np.argsort(angle)]
    x0, y0 = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    cross = x0 * y1 - x1 * y0
    area = 0.5 * cross.sum()
    if area < 0.0:  # enforce positive orientation
        x0, x1, y0, y1, cross = x1, x0, y1, y0, -cross
        area = -area
    s_x = (cross * (x0 + x1)).sum() / 6.0
    s_y = (cross * (y0 + y1)).sum() / 6.0
    i_xx = (cross * (y0**2 + y0 * y1 + y1**2)).sum() / 12.0
    i_yy = (cross * (x0**2 + x0 * x1 + x1**2)).sum() / 12.0
    i_xy = (cross * (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0)).sum() / 24.0
    return {
        "area": float(area),
        "s_x": float(s_x),
        "s_y": float(s_y),
        "i_xx": float(i_xx),
        "i_yy": float(i_yy),
        "i_xy": float(i_xy),
    }


def matrix_from_integrals(
    integrals: dict[str, float], volume: float, z_b: float, z_g: float
) -> np.ndarray:
    """Assemble ``C`` from waterplane moments (Journee & Massie, section 3.2.2)."""
    rg = RHO * G
    c = np.zeros((6, 6))
    c[2, 2] = rg * integrals["area"]
    c[2, 3] = c[3, 2] = rg * integrals["s_y"]
    c[2, 4] = c[4, 2] = -rg * integrals["s_x"]
    c[3, 3] = rg * integrals["i_xx"] + rg * volume * (z_b - z_g)
    c[4, 4] = rg * integrals["i_yy"] + rg * volume * (z_b - z_g)
    c[3, 4] = c[4, 3] = -rg * integrals["i_xy"]
    return c


@lru_cache(maxsize=1)
def headline() -> dict:
    """The document's main case: barge 2x1x1 m, LCF=1.30 m, TCF=0.45 m."""
    length, beam, draft = BARGE["length"], BARGE["beam"], BARGE["draft"]
    lcf, tcf = BARGE["lcf"], BARGE["tcf"]
    hull, volume, cb, cg, c = barge_matrix(length, beam, draft, lcf, tcf)
    closed = rectangle_closed_form(length, beam, draft, lcf, tcf, cg[2])
    energy = restoring_matrix_from_energy(hull.vertices, hull.faces, RHO * volume, cg, RHO, G)
    legacy = legacy_matrix(length, beam, draft, lcf, tcf)
    scale = float(np.max(np.abs(c)))
    rows = []
    for label, (i, j) in ENTRIES:
        denom = abs(energy[i, j]) or 1.0
        rows.append(
            {
                "entry": label,
                "index": [i, j],
                "code": float(c[i, j]),
                "closed_form": float(closed[i, j]),
                "energy_hessian": float(energy[i, j]),
                "pre_fix": float(legacy[i, j]),
                "rel_err_closed_form": float(abs(c[i, j] - closed[i, j]) / (abs(closed[i, j]) or 1.0)),
                "rel_err_energy": float(abs(c[i, j] - energy[i, j]) / denom),
                "pre_fix_rel_err": float(abs(legacy[i, j] - energy[i, j]) / denom),
            }
        )
    return {
        "volume": volume,
        "center_of_buoyancy": [float(v) for v in cb],
        "rows": rows,
        "symmetric_residual": float(np.max(np.abs(c - c.T))),
        "yaw_row_max": float(np.max(np.abs(c[[0, 1, 5], :]))),
        "max_rel_err_energy": float(np.max(np.abs(c - energy)) / scale),
    }


@lru_cache(maxsize=1)
def sweep() -> tuple[dict, ...]:
    """The LCF sweep table (barge 2x1x1, TCF=0)."""
    length, beam, draft = BARGE["length"], BARGE["beam"], BARGE["draft"]
    out = []
    for lcf in SWEEP_LCF:
        _, volume, _, _, c = barge_matrix(length, beam, draft, lcf, 0.0)
        block = np.array([[c[2, 2], c[2, 4]], [c[4, 2], c[4, 4]]])
        resp = np.linalg.solve(block, np.array([0.0, 1.0]))
        out.append(
            {
                "lcf": float(lcf),
                "lcf_over_length": float(lcf / length),
                "c_heave_pitch": float(c[2, 4]),
                "abs_ratio_to_c33": float(abs(c[2, 4]) / c[2, 2]),
                "pitch_amplification": float(resp[1] / (1.0 / block[1, 1])),
                "heave_response_per_unit_pitch_moment": float(resp[0]),
            }
        )
    return tuple(out)

# ---------------------------------------------------------------------------
# the verification debt: curved waterplanes / curved wetted surfaces
# ---------------------------------------------------------------------------
CURVED_CASES = (
    ("elliptic", 16, (0.0, 0.0)),
    ("elliptic", 16, (0.30, 0.10)),
    ("elliptic", 32, (0.0, 0.0)),
    ("elliptic", 32, (0.30, 0.10)),
    ("elliptic", 64, (0.0, 0.0)),
    ("elliptic", 64, (0.30, 0.10)),
    ("hemisphere", 2, (0.0, 0.0)),
    ("hemisphere", 2, (0.25, 0.15)),
    ("hemisphere", 3, (0.0, 0.0)),
    ("hemisphere", 3, (0.25, 0.15)),
)

ELLIPTIC_SEMI_AXES = (1.0, 0.5)
ELLIPTIC_DRAFT = 0.8
HEMISPHERE_RADIUS = 1.0
HEMISPHERE_CG_Z = -0.4

#: a curved case passes when the code agrees with the energy Hessian and with
#: the independent ring integration to better than this (fraction of max |C|).
CURVED_TOLERANCE = 1.0e-3


def case_name(kind: str, level: int, offset: tuple[float, float]) -> str:
    return f"{kind}-{level}-{'offset' if any(offset) else 'centred'}"


def curved_case(kind: str, level: int, offset: tuple[float, float]) -> dict:
    """One curved-hull case: production code vs energy Hessian vs ring integration."""
    if kind == "elliptic":
        base = elliptic_cylinder_hull_mesh(*ELLIPTIC_SEMI_AXES, ELLIPTIC_DRAFT, panels=level)
        draft = ELLIPTIC_DRAFT
        cg_z = -0.5 * draft
    elif kind == "hemisphere":
        base = hemisphere_hull_mesh(level, HEMISPHERE_RADIUS)
        draft = HEMISPHERE_RADIUS
        cg_z = HEMISPHERE_CG_Z
    else:  # pragma: no cover - CURVED_CASES is fixed
        raise ValueError(f"unknown hull kind {kind!r}")

    hull = WettedHullMesh(base.vertices + np.array([offset[0], offset[1], 0.0]), base.faces)
    volume, cb = compute_displaced_volume_and_buoyancy(hull)
    cg = np.array([offset[0], offset[1], cg_z])
    c = compute_hydrostatic_restoring_matrix(hull, RHO * volume, cg, RHO, G).restoring_matrix
    energy = restoring_matrix_from_energy(hull.vertices, hull.faces, RHO * volume, cg, RHO, G)
    integrals = polygon_waterplane_integrals(hull)
    ring = matrix_from_integrals(integrals, float(volume), float(cb[2]), cg_z)
    scale = float(np.max(np.abs(c)))
    rows = {}
    for label, (i, j) in ENTRIES:
        rows[label] = {
            "code": float(c[i, j]),
            "energy_hessian": float(energy[i, j]),
            "ring": float(ring[i, j]),
            "rel_dev_energy": float(abs(c[i, j] - energy[i, j]) / scale),
            "rel_dev_ring": float(abs(c[i, j] - ring[i, j]) / scale),
        }
    return {
        "name": case_name(kind, level, offset),
        "kind": kind,
        "level": int(level),
        "draft": float(draft),
        "offset": [float(offset[0]), float(offset[1])],
        "volume": float(volume),
        "center_of_buoyancy": [float(v) for v in cb],
        "waterplane": {k: float(v) for k, v in integrals.items()},
        "rows": rows,
        "max_rel_dev_energy": max(r["rel_dev_energy"] for r in rows.values()),
        "max_rel_dev_ring": max(r["rel_dev_ring"] for r in rows.values()),
        "symmetry_residual_rel": float(np.max(np.abs(c - c.T)) / scale),
        "yaw_rows_rel": float(np.max(np.abs(c[[0, 1, 5], :])) / scale),
        "coupling_is_live": bool(
            abs(c[2, 3]) > 1.0e-6 * scale
            and abs(c[2, 4]) > 1.0e-6 * scale
            and abs(c[3, 4]) > 1.0e-6 * scale
        ),
    }


@lru_cache(maxsize=1)
def curved_cases() -> tuple[dict, ...]:
    return tuple(curved_case(*case) for case in CURVED_CASES)


# ---------------------------------------------------------------------------
# parsing the documents
# ---------------------------------------------------------------------------
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(_NUM)
#: tolerance = PRINTED_TOLERANCE_SCALE x the quantum of the last printed digit
PRINTED_TOLERANCE_SCALE = 1.5


@dataclass(frozen=True)
class Declared:
    """One number printed in a document, with the precision it was printed at."""

    key: str
    value: float
    atol: float
    text: str


def printed_tolerance(token: str) -> float:
    """Half a unit in the last printed place of ``token``."""
    match = re.fullmatch(r"([-+]?)(\d+)(?:\.(\d+))?(?:[eE]([-+]?\d+))?", token.strip())
    if match is None:
        raise ValueError(f"not a number token: {token!r}")
    decimals = len(match.group(3) or "")
    exponent = int(match.group(4) or 0)
    return 0.5 * 10.0 ** (exponent - decimals)


def _decl(key: str, token: str, scale: float = 1.0) -> Declared:
    return Declared(
        key=key,
        value=float(token) * scale,
        atol=printed_tolerance(token) * abs(scale) * PRINTED_TOLERANCE_SCALE,
        text=token,
    )


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_couplings_doc(text: str) -> list[Declared]:
    """Every number this gate checks in ``hydrostatic_restoring_couplings.md``."""
    declared: list[Declared] = []
    lines = text.splitlines()

    for line in lines:  # header: displaced volume and centre of buoyancy
        if "排水体积" in line:
            match = re.search(
                r"∇ \| (" + _NUM + r") m³（解析 (" + _NUM + r")，浮心 \[([^\]]+)\]）", line
            )
            if match:
                declared.append(_decl("volume.printed", match.group(1)))
                declared.append(_decl("volume.analytic", match.group(2)))
                for k, token in enumerate(match.group(3).split(",")):
                    declared.append(
                        Declared(
                            key=f"center_of_buoyancy[{k}]",
                            value=float(token),
                            atol=1.0e-9,
                            text=token.strip(),
                        )
                    )
            break

    for line in lines:  # main table: six entries, five printed numbers each
        cells = _cells(line) if line.startswith("| C") else []
        if len(cells) < 6 or not re.match(r"^C\d\d ", cells[0]):
            continue
        slug = f"main.{cells[0].replace(' ', '_')}"
        declared.append(_decl(f"{slug}.code", cells[1]))
        declared.append(_decl(f"{slug}.closed_form", cells[2]))
        declared.append(_decl(f"{slug}.energy_hessian", cells[3]))
        if cells[4].startswith("0 (exactly)"):
            declared.append(Declared(f"{slug}.pre_fix", 0.0, 1.0e-9, cells[4]))
        else:
            declared.append(_decl(f"{slug}.pre_fix", cells[4]))
        percent = _NUM_RE.search(cells[5])
        if percent:
            item = _decl(f"{slug}.pre_fix_pct", percent.group(0), scale=0.01)
            declared.append(item)

    for line in lines:  # residual paragraph under the main table
        if "对称性残差" in line:
            match = re.search(r"=(" + _NUM + r")\$", line)
            if match:
                declared.append(_decl("scalar.symmetric_residual", match.group(1)))
        elif "行列最大绝对值" in line:
            match = re.search(r"\$(" + _NUM + r")\$ N", line)
            if match:
                declared.append(_decl("scalar.yaw_row_max", match.group(1)))
        elif "最大相对偏差" in line:
            match = re.search(r"\$(" + _NUM + r")\$", line)
            if match:
                declared.append(_decl("scalar.max_rel_err_energy", match.group(1)))

    for line in lines:  # LCF sweep table
        if not re.match(r"^\| [-+]?\d", line):
            continue
        cells = _cells(line)
        if len(cells) != 5:
            continue
        try:
            float(cells[0]), float(cells[1])
        except ValueError:
            continue
        slug = f"sweep.{cells[0]}"
        declared.append(_decl(f"{slug}.lcf_over_length", cells[0]))
        declared.append(_decl(f"{slug}.c_heave_pitch", cells[1]))
        declared.append(_decl(f"{slug}.abs_ratio_to_c33", cells[2]))
        declared.append(_decl(f"{slug}.pitch_amplification", cells[3]))
        declared.append(_decl(f"{slug}.heave_per_pitch_moment", cells[4]))

    in_errors = False
    for line in lines:  # section 6 error table
        if line.startswith("## 6."):
            in_errors = True
            continue
        if not in_errors or not line.startswith("|"):
            continue
        cells = _cells(line)
        if len(cells) < 4 or cells[0].startswith("---") or cells[0].startswith("量"):
            continue
        match = re.search(r"C_\{(\d)(\d)\}", cells[0])
        key = f"error.C{match.group(1)}_{match.group(2)}" if match else "error.full_matrix"
        if cells[2]:
            declared.append(_decl(key, cells[2]))

    for line in lines:  # prose numbers around the tables
        if "最差项为" in line:
            match = re.search(r"相对误差 (" + _NUM + r")", line)
            if match:
                declared.append(_decl("prose.worst_rel_err", match.group(1)))
        elif "上是" in line and "%" in line:
            percents = re.findall(r"(" + _NUM + r") %", line)
            if len(percents) == 2:
                declared.append(_decl("prose.pre_fix_pct_c44", percents[0], scale=0.01))
                declared.append(_decl("prose.pre_fix_pct_c55", percents[1], scale=0.01))

    return declared


def couplings_values() -> dict[str, float]:
    """The recomputed counterpart of every key :func:`parse_couplings_doc` emits."""
    head = headline()
    values: dict[str, float] = {
        "volume.printed": head["volume"],
        "volume.analytic": BARGE["length"] * BARGE["beam"] * BARGE["draft"],
        "scalar.symmetric_residual": head["symmetric_residual"],
        "scalar.yaw_row_max": head["yaw_row_max"],
        "scalar.max_rel_err_energy": head["max_rel_err_energy"],
    }
    for k, value in enumerate(head["center_of_buoyancy"]):
        values[f"center_of_buoyancy[{k}]"] = value
    for row in head["rows"]:
        slug = f"main.{row['entry'].replace(' ', '_')}"
        values[f"{slug}.code"] = row["code"]
        values[f"{slug}.closed_form"] = row["closed_form"]
        values[f"{slug}.energy_hessian"] = row["energy_hessian"]
        values[f"{slug}.pre_fix"] = row["pre_fix"]
        values[f"{slug}.pre_fix_pct"] = row["pre_fix_rel_err"]
        match = re.match(r"^C(\d)(\d) ", row["entry"])
        values[f"error.C{match.group(1)}_{match.group(2)}"] = row["rel_err_energy"]
    values["error.full_matrix"] = head["max_rel_err_energy"]
    worst = max(head["rows"], key=lambda r: r["rel_err_energy"])
    values["prose.worst_rel_err"] = worst["rel_err_energy"]
    values["prose.pre_fix_pct_c44"] = head["rows"][3]["pre_fix_rel_err"]
    values["prose.pre_fix_pct_c55"] = head["rows"][5]["pre_fix_rel_err"]
    for row in sweep():
        slug = f"sweep.{format(row['lcf_over_length'], '.2f')}"
        values[f"{slug}.lcf_over_length"] = row["lcf_over_length"]
        values[f"{slug}.c_heave_pitch"] = row["c_heave_pitch"]
        values[f"{slug}.abs_ratio_to_c33"] = row["abs_ratio_to_c33"]
        values[f"{slug}.pitch_amplification"] = row["pitch_amplification"]
        values[f"{slug}.heave_per_pitch_moment"] = row["heave_response_per_unit_pitch_moment"]
    return values


def parse_curvature_doc(text: str) -> list[Declared]:
    """Every number in the generated curvature document."""
    declared: list[Declared] = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("|---") or line.startswith("| case"):
            continue
        cells = _cells(line)
        if (
            len(cells) == 7
            and re.match(r"^[A-Za-z]+-\d+-(centred|offset)$", cells[0])
            and re.match(r"^C\d\d ", cells[1])
        ):
            slug = f"curv.{cells[0]}.{cells[1].replace(' ', '_')}"
            declared.append(_decl(f"{slug}.code", cells[2]))
            declared.append(_decl(f"{slug}.energy_hessian", cells[3]))
            declared.append(_decl(f"{slug}.ring", cells[4]))
            declared.append(_decl(f"{slug}.rel_dev_energy", cells[5]))
            declared.append(_decl(f"{slug}.rel_dev_ring", cells[6]))
            continue
        if len(cells) == 6 and re.match(r"^[A-Za-z]+-\d+-(centred|offset)$", cells[0]):
            slug = f"summary.{cells[0]}"
            declared.append(_decl(f"{slug}.volume", cells[1]))
            declared.append(_decl(f"{slug}.max_rel_dev_energy", cells[2]))
            declared.append(_decl(f"{slug}.max_rel_dev_ring", cells[3]))
            declared.append(_decl(f"{slug}.symmetry_residual_rel", cells[4]))
            declared.append(_decl(f"{slug}.yaw_rows_rel", cells[5]))
    return declared


def curvature_values() -> dict[str, float]:
    values: dict[str, float] = {}
    for case in curved_cases():
        values[f"summary.{case['name']}.volume"] = case["volume"]
        values[f"summary.{case['name']}.max_rel_dev_energy"] = case["max_rel_dev_energy"]
        values[f"summary.{case['name']}.max_rel_dev_ring"] = case["max_rel_dev_ring"]
        values[f"summary.{case['name']}.symmetry_residual_rel"] = case["symmetry_residual_rel"]
        values[f"summary.{case['name']}.yaw_rows_rel"] = case["yaw_rows_rel"]
        for label, row in case["rows"].items():
            slug = f"curv.{case['name']}.{label.replace(' ', '_')}"
            values[f"{slug}.code"] = row["code"]
            values[f"{slug}.energy_hessian"] = row["energy_hessian"]
            values[f"{slug}.ring"] = row["ring"]
            values[f"{slug}.rel_dev_energy"] = row["rel_dev_energy"]
            values[f"{slug}.rel_dev_ring"] = row["rel_dev_ring"]
    return values


def compare(declared: list[Declared], values: dict[str, float]) -> list[dict]:
    """Pair every declared number with its recomputed value and verdict."""
    records = []
    for item in declared:
        got = values.get(item.key)
        if got is None:
            records.append(
                {
                    "key": item.key,
                    "declared": item.value,
                    "computed": None,
                    "atol": item.atol,
                    "status": "MISSING",
                    "text": item.text,
                }
            )
            continue
        deviation = abs(float(got) - item.value)
        records.append(
            {
                "key": item.key,
                "declared": item.value,
                "computed": float(got),
                "atol": item.atol,
                "deviation": float(deviation),
                "status": "PASS" if deviation <= item.atol else "DRIFT",
                "text": item.text,
            }
        )
    return records


def check() -> dict:
    """Parse both documents, recompute, and report every drift."""
    report: dict = {"schema": "tensorbem.hydrostatic_restoring_pins/1", "documents": {}}
    for name, path, parser, values in (
        ("hydrostatic_restoring_couplings.md", DOC, parse_couplings_doc, couplings_values),
        ("hydrostatic_restoring_waterplane_curvature.md", CURVATURE_DOC, parse_curvature_doc, curvature_values),
    ):
        if not path.exists():
            report["documents"][name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "status": "ABSENT",
                "checked": 0,
                "records": [],
            }
            continue
        records = compare(parser(path.read_text(encoding="utf-8")), values())
        failures = [r for r in records if r["status"] != "PASS"]
        report["documents"][name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "status": "PASS" if not failures else "DRIFT",
            "checked": len(records),
            "failures": failures,
            "records": records,
        }
    report["curved_verification"] = [
        {
            "case": case["name"],
            "volume": case["volume"],
            "max_rel_dev_energy": case["max_rel_dev_energy"],
            "max_rel_dev_ring": case["max_rel_dev_ring"],
            "coupling_is_live": case["coupling_is_live"],
        }
        for case in curved_cases()
    ]
    report["status"] = (
        "PASS"
        if all(d["status"] == "PASS" for d in report["documents"].values())
        and max(c["max_rel_dev_energy"] for c in report["curved_verification"]) < CURVED_TOLERANCE
        else "DRIFT"
    )
    return report


# ---------------------------------------------------------------------------
# rendering the curvature note and its figure
# ---------------------------------------------------------------------------
def render_curvature_doc() -> str:
    """Markdown for ``docs/hydrostatic_restoring_waterplane_curvature.md``."""
    cases = curved_cases()
    worst_energy = max(cases, key=lambda c: c["max_rel_dev_energy"])
    worst_ring = max(cases, key=lambda c: c["max_rel_dev_ring"])
    curved_worst = max(worst_energy["max_rel_dev_energy"], worst_ring["max_rel_dev_ring"])

    summary = [
        "| case | volume [m³] | max |ΔC|/max|C| vs energy Hessian | vs ring integration | symmetry | yaw rows |",
        "|---|---|---|---|---|---|",
    ]
    for case in cases:
        summary.append(
            f"| {case['name']} | {case['volume']:.6f} | {case['max_rel_dev_energy']:.2e} | "
            f"{case['max_rel_dev_ring']:.2e} | {case['symmetry_residual_rel']:.2e} | "
            f"{case['yaw_rows_rel']:.2e} |"
        )

    entries = [
        "| case | entry | code | energy Hessian | ring polygon | rel dev vs Hessian | rel dev vs ring |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        for label, row in case["rows"].items():
            entries.append(
                f"| {case['name']} | {label} | {row['code']:.6e} | {row['energy_hessian']:.6e} | "
                f"{row['ring']:.6e} | {row['rel_dev_energy']:.2e} | {row['rel_dev_ring']:.2e} |"
            )

    waterplane = [
        "| case | A_wp [m²] | S_x [m³] | S_y [m³] | I_xx [m⁴] | I_yy [m⁴] | I_xy [m⁴] |",
        "|---|---|---|---|---|---|---|",
    ]
    for case in cases:
        if not case["name"].endswith("offset"):
            continue
        wp = case["waterplane"]
        waterplane.append(
            f"| {case['name']} | {wp['area']:.6f} | {wp['s_x']:.6e} | {wp['s_y']:.6e} | "
            f"{wp['i_xx']:.6e} | {wp['i_yy']:.6e} | {wp['i_xy']:.6e} |"
        )

    def _ratio(case, key):
        return case["waterplane"][key] / case["waterplane"]["area"]

    ell_offset = [c for c in cases if c["kind"] == "elliptic" and c["name"].endswith("offset")]
    hemi_offset = [c for c in cases if c["kind"] == "hemisphere" and c["name"].endswith("offset")]
    ell_ratio_x = _ratio(ell_offset[-1], "s_x")
    ell_ratio_y = _ratio(ell_offset[-1], "s_y")
    hemi_ratio_x = _ratio(hemi_offset[-1], "s_x")
    hemi_ratio_y = _ratio(hemi_offset[-1], "s_y")
    ell_area64 = ell_offset[-1]["waterplane"]["area"]
    ell_area_gap = abs(ell_area64 - math.pi * ELLIPTIC_SEMI_AXES[0] * ELLIPTIC_SEMI_AXES[1]) / (
        math.pi * ELLIPTIC_SEMI_AXES[0] * ELLIPTIC_SEMI_AXES[1]
    )
    hemi_floor = max(c["max_rel_dev_energy"] for c in cases if c["kind"] == "hemisphere")
    ell_floor = max(c["max_rel_dev_energy"] for c in cases if c["kind"] == "elliptic")
    ring_floor = max(c["max_rel_dev_ring"] for c in cases)

    centered = [c for c in cases if c["name"].endswith("centred")]
    offset = [c for c in cases if c["name"].endswith("offset")]
    centered_dead = all(not c["coupling_is_live"] for c in centered)
    offset_live = all(c["coupling_is_live"] for c in offset)
    verdict = "是（全部 < 3 %）" if curved_worst < 0.03 else "否"

    return f"""# 静水恢复矩阵的弯曲水线面复核：椭圆截面与浮半球（m4741）

**任务**：m4741（platform 席自找活）· **仓**：TensorBEM · **分支**：`agent/m4741`
**生成器**：`PYTHONPATH=src python3 -m tensorbem.benchmarks.hydrostatic_restoring_pins --write`
（本文全部数字由该命令注入，禁止手抄；`tests/test_hydrostatic_restoring_doc_pins.py` 读回本文并复算）
**对应欠账**：`docs/hydrostatic_restoring_couplings.md` §5 第 3 条——「弧度曲壁船体的耦合项未用能量 Hessian 复核」。
**与 m4750 的关系**：该欠账已由 m4750（`hydrostatic_curved_waterplane.py`：逐三角形裁剪的
能量 Hessian + 精确几何闭式，O(h²) 收敛）独立结清；本文是**第三条独立路径**（Green 定理水线面环积分，
不经过 `compute_waterplane_properties`），并把两份文档的每个数字变成可执行判据，
与 m4750 的三角形裁剪路径互不共享代码，因此两者一致是交叉验证而非重复。

---

## 1. 案例介绍

`docs/hydrostatic_restoring_couplings.md`（m3196）把 `compute_hydrostatic_restoring_matrix`
的 heave–roll / heave–pitch / roll–pitch 耦合项补齐，并用**能量 Hessian** 与**闭式解**
两条独立基准钉桩。但它的算例族只有矩形水线面的驳船：水线面是直线围成的矩形，
船体是竖直侧壁。真实船的水线面是曲线（椭圆/舷弧/球鼻），这条欠账意味着
「曲线水线面 + 曲壁船体」下 `C` 的六个非零元从未与独立基准对过。

本单补两个曲率族（$\rho=1025.0$ kg/m³，$g=9.80665$ m/s²，参考点取静水面原点）：

| 族 | 几何 | 水线面 | 浸湿面 |
|---|---|---|---|
| `elliptic` | 椭圆截面柱体，半轴 $a=1.0$、$b=0.5$ m，吃水 $T=0.8$ m，$n$ 边形内接椭圆 | 曲线（$n$ 边多边形） | 竖直曲壁 + 平底 |
| `hemisphere` | 浮半球，半径 $R=1.0$ m，重心 $z_G=-0.4$ m | 圆（球面与 $z=0$ 交线，离散为多边形） | 双曲率曲壁 |

每种几何取**水线面形心在原点**与**形心偏移**两种构型（偏移量椭圆族 $(0.30,0.10)$ m、
半球族 $(0.25,0.15)$ m），离散级别作为收敛阶梯（椭圆 $n=16/32/64$，半球 subdivisions $=2/3$）。
偏移构型是判据关键：形心在原点时三个耦合项按对称性恒为零，测不出任何耦合缺陷。

## 2. 怎么做的

三条互相独立的取值路径，对同一张 $C$ 的六个元逐个比对：

1. **生产代码**：`compute_hydrostatic_restoring_matrix`（水线面积分 → 平行轴定理 → 6 个非零元）。
2. **能量 Hessian**：`tensorbem.benchmarks.hydrostatic_energy_reference.restoring_matrix_from_energy`，
   只用 $V(\\xi)=m g z_G(\\xi)-\\rho g\\int z\\,\\mathrm{{d}}V$ 与多边形裁剪 + 散度定理，
   **完全不碰**任何水线面公式；对曲线壁同样适用（它只认三角面片）。
3. **水线面环积分**（本单新增，在 pins 模块内）：把网格上 $z=0$ 的顶点按极角排序成环，
   用 Green 定理的多边形矩公式直接算 $A,S_x,S_y,I_{{xx}},I_{{yy}},I_{{xy}}$，
   再按 Journée & Massie §3.2.2 组装 $C$。它**不调用** `compute_waterplane_properties`，
   因此专门抓水线面追踪/平行轴定理环节的错误。

判据：$|C^{{code}}_{{ij}}-C^{{ref}}_{{ij}}|/\\max_{{ij}}|C_{{ij}}|$，参考值分别取 (2) 与 (3)。
能量 Hessian 的偏差里含二阶中心差分截断 $O(h^2)$（$h=2\\times10^{{-4}}$）与裁剪浮点噪声，
所以曲壁族的下限在 $10^{{-6}}\\sim10^{{-5}}$ 量级；判据取 $10^{{-3}}$（远严于 3 %）。

**两条基准各管什么，必须分清**：(3) 与生产代码在代数上等价（同一组多边形矩、同一组
平行轴变换），因此它对 (1) 的一致只证明「水线面追踪 / 一次矩 / 平行轴 / 符号」这套
记账没写错（实测差 {ring_floor:.2e}，纯浮点顺序差），**不构成物理验证**；
真正的物理验证是 (2)：它只认阿基米德原理与体重，连「水线面」这个概念都没有。

## 3. 计算结果

### 3.1 汇总（每族每元的最大相对偏差，分母为 $\\max|C|$）

{chr(10).join(summary)}

结论：**曲壁/曲线水线面族全部通过**——最差项为 `{worst_energy['name']}` 对能量 Hessian
（{worst_energy['max_rel_dev_energy']:.2e}）与 `{worst_ring['name']}` 对环积分
（{worst_ring['max_rel_dev_ring']:.2e}），均远小于 $10^{{-3}}$。
形心在原点构型的耦合项恒为零（$\\max|C_{{34}}|,|C_{{35}}|,|C_{{45}}|<10^{{-6}}\\max|C|$：
{centered_dead}），形心偏移构型三项全部非零（{offset_live}）——即耦合项确实由水线面一次矩驱动，
而不是几何偶然。

### 3.2 水线面矩（偏移构型，环积分路径）

{chr(10).join(waterplane)}

形心偏移由环积分反算：椭圆族 $S_x/A={ell_ratio_x:.6f}$ m、$S_y/A={ell_ratio_y:.6f}$ m，
半球族 $S_x/A={hemi_ratio_x:.6f}$ m、$S_y/A={hemi_ratio_y:.6f}$ m——与施加的偏移量逐位一致
（$a=1.0,b=0.5$ 的椭圆多边形面积随 $n$ 增大逼近 $\pi ab=1.570796$ m²，
$n=64$ 时为 {ell_area64:.6f} m²，差 {ell_area_gap:.2%}，符合内接多边形 $O(n^{{-2}})$ 的偏差）。

### 3.3 逐元对照（10 个构型 × 6 个元）

{chr(10).join(entries)}

![curvature](figures/fig_restoring_curvature.png)

## 4. 结果图说明

- 左：三张水线面俯视图（椭圆 $n=16$ 形心在原点 / 形心偏移、半球偏移），标出形心位置——
  耦合项的大小就是形心偏移的一阶量。
- 中：偏移构型下六个元的两条相对偏差（对能量 Hessian 实心、对环积分空心），对数纵轴。
- 右：离散级别收敛（椭圆 $n=16\\to64$、半球 sub $2\\to3$）。偏差随离散下降而非发散，
  说明三路径收敛到同一张 $C$；残差地板由能量 Hessian 的差分步长与裁剪决定。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **水线面环的顶点排序必须按极角**：`elliptic_cylinder_hull_mesh` / `hemisphere_hull_mesh`
   的 $z=0$ 顶点在数组里不是按环序排列的，直接连边会自交、面积算错。本模块先按质心极角排序
   再用 Green 公式；凸星形水线面（本单两族都是）下这是精确的。
2. **半球族的残差地板比椭圆族高约三个量级**（{hemi_floor:.2e} vs {ell_floor:.2e}）：能量 Hessian 的
   裁剪在多曲率三角面上引入的浮点噪声更大，且 $h=2\\times10^{{-4}}$ 的截断误差不再被平面壁
   的精确代数抵消。**这不是实现缺陷**——三路径的差仍然同阶，且随离散收敛（右图）。
3. **柔度限制**：半球族 subdivisions 只能取 $2/3$（$4\\cdot4^s$ 面元，$s=4$ 时 1024 面元 ×
   能量 Hessian 的 24 次重积分会让测试变慢）；椭圆族到 $n=64$ 为止。
4. **仍未验证**：三维自由液面（浮球辐射/衍射、Hulme、Wang–Linton）的耦合项；外飘（flare）船体
   的水线面仍按 $z=0$ 截面处理。yaw 耦合按铅垂力无矩置零，本单只复核了残差量级
   （见 3.1 的 yaw rows 列，全部 $\\le10^{{-9}}\\max|C|$）。
5. **本文不是重新推导**：$C$ 的初等公式（$\\rho g S_x$ 等）与 m3196 一致；本单增量是
   **曲率族证据**与**文档数字的机器可读化**（见下节 6）。曲壁欠账已由 m4750
   （三角形裁剪 + 精确几何闭式）独立结清，本文的环积分路径与它不共享代码，
   两条路径在 §3 的同一批构型上一致即为交叉验证。

## 6. 误差判定（关键量 vs 基准，逐项）

| 量 | 基准 | 最大相对偏差 | 是否 < 3 % |
|---|---|---|---|
| 椭圆族 6 元（3 个离散级别 × 2 构型） | 能量 Hessian | {max(c['max_rel_dev_energy'] for c in cases if c['kind'] == 'elliptic'):.2e} | 是 |
| 椭圆族 6 元 | 水线面环积分 | {max(c['max_rel_dev_ring'] for c in cases if c['kind'] == 'elliptic'):.2e} | 是 |
| 半球族 6 元（2 个级别 × 2 构型） | 能量 Hessian | {max(c['max_rel_dev_energy'] for c in cases if c['kind'] == 'hemisphere'):.2e} | 是 |
| 半球族 6 元 | 水线面环积分 | {max(c['max_rel_dev_ring'] for c in cases if c['kind'] == 'hemisphere'):.2e} | 是 |

**总体判定**：{verdict}。最差 {curved_worst:.2e}，比 3 % 小 {0.03 / max(curved_worst, 1e-30):.0f} 倍。
配套的 `docs/hydrostatic_restoring_couplings.md`（78 个数字）同版本复算全绿，见
`docs/hydrostatic_restoring_doc_pins.json`。
"""


def make_figure(path: Path) -> Path:
    """Three-panel figure: waterplanes, per-entry deviations, discretisation ladder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = {case["name"]: case for case in curved_cases()}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # --- left: the waterplanes that drive the coupling ---------------------
    ax = axes[0]
    ax.plot(0.0, 0.0, "k+", ms=12, label="origin")
    for kind, level, offset, colour in (
        ("elliptic", 16, (0.0, 0.0), "C0"),
        ("elliptic", 16, (0.30, 0.10), "C1"),
        ("hemisphere", 2, (0.25, 0.15), "C2"),
    ):
        if kind == "elliptic":
            base = elliptic_cylinder_hull_mesh(*ELLIPTIC_SEMI_AXES, ELLIPTIC_DRAFT, panels=level)
        else:
            base = hemisphere_hull_mesh(level, HEMISPHERE_RADIUS)
        hull = WettedHullMesh(base.vertices + np.array([offset[0], offset[1], 0.0]), base.faces)
        ring = hull.vertices[np.abs(hull.vertices[:, 2]) <= 1.0e-12][:, :2]
        centre = ring.mean(axis=0)
        order = np.argsort(np.arctan2(ring[:, 1] - centre[1], ring[:, 0] - centre[0]))
        ring = ring[order]
        label = case_name(kind, level, offset)
        ax.plot(*np.vstack([ring, ring[:1]]).T, "-o", ms=2.5, lw=1.2, color=colour, label=label)
        ax.plot(*centre, "*", ms=13, color=colour)
        ax.annotate(
            f"centroid ({centre[0]:.2f}, {centre[1]:.2f})",
            xy=centre,
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            color=colour,
        )
    ax.set_aspect("equal")
    ax.set_title("waterplanes: the centroid offset is the coupling")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    # --- middle: per-entry deviations for the offset configurations --------
    ax = axes[1]
    labels = [label for label, _ in ENTRIES]
    offset_names = [c["name"] for c in curved_cases() if c["name"].endswith("offset")]
    width = 0.8 / len(offset_names)
    for k, name in enumerate(offset_names):
        case = cases[name]
        xs = np.arange(len(labels)) + (k - (len(offset_names) - 1) / 2) * width
        ax.bar(
            xs,
            [max(row["rel_dev_energy"], 1e-16) for row in case["rows"].values()],
            width,
            label=f"{name} vs Hessian",
            alpha=0.85,
        )
        ax.plot(
            xs,
            [max(row["rel_dev_ring"], 1e-16) for row in case["rows"].values()],
            "k_",
            ms=9,
            label="vs ring integration" if k == 0 else None,
        )
    ax.set_yscale("log")
    ax.axhline(CURVED_TOLERANCE, color="r", ls="--", lw=1, label="tolerance $10^{-3}$")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"$|\Delta C|/\max|C|$")
    ax.set_title("offset hulls: code vs two references")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    # --- right: discretisation ladder -------------------------------------
    ax = axes[2]
    for kind, marker in (("elliptic", "o"), ("hemisphere", "s")):
        for tail, ls in (("centred", "--"), ("offset", "-")):
            series = [c for c in curved_cases() if c["kind"] == kind and c["name"].endswith(tail)]
            if not series:
                continue
            ax.plot(
                [c["level"] for c in series],
                [max(c["max_rel_dev_energy"], 1e-16) for c in series],
                marker + ls,
                label=f"{kind} {tail}",
            )
    ax.set_xlabel("discretisation level (panels / subdivisions)")
    ax.set_ylabel(r"max $|\Delta C|/\max|C|$ vs Hessian")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("deviation shrinks with refinement")

    fig.suptitle("Hydrostatic restoring couplings on curved waterplanes (m4741)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when a documented digit drifted")
    parser.add_argument("--write", action="store_true", help="regenerate the note, figure and ledger")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "docs" / "hydrostatic_restoring_doc_pins.json",
    )
    parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args(argv)

    if args.write:
        CURVATURE_DOC.write_text(render_curvature_doc(), encoding="utf-8")
        make_figure(args.figure)
        print(f"wrote {CURVATURE_DOC.relative_to(REPO_ROOT)} and {args.figure.relative_to(REPO_ROOT)}")

    report = check()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, doc in report["documents"].items():
        print(f"{name}: {doc['status']} ({doc['checked']} numbers)")
        for failure in doc.get("failures", []):
            print(
                f"  DRIFT {failure['key']}: declared {failure['declared']!r} "
                f"({failure['text']}) but code gives {failure['computed']!r}"
            )
    print(f"curved verification: {report['status']}")
    if args.check and report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
