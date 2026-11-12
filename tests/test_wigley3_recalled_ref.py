"""Adjudication of the recalled Wigley III WAMIT tables must stay true.

The artefact is only worth its numbers if three things keep holding: the two
mesh builders still produce the arrays the frozen capytaine reference was taken
on, the TensorBEM numbers in the artefact still come out of the solver, and the
published document still equals the rendered one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks import wigley3_recalled_ref as study
from tensorbem.benchmarks.wigley3_recalled_ref import (
    DENSITY,
    MESH_LEVELS,
    OMEGA_STATED,
    build_mesh,
    recalled,
    render_document,
    tensorbem_cell,
)
from tensorbem.wigley3_hull import wigley3_displaced_volume

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / study.OUT_PATH
DOCUMENT = ROOT / study.DOC_PATH
REFERENCE = ROOT / study.REF_PATH


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads(REFERENCE.read_text())


def _signature(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertices, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(faces, dtype="<i8").tobytes())
    return h.hexdigest()


def test_frozen_reference_meshes_still_match_the_builders(reference: dict) -> None:
    # ``wigley3_hull`` is deliberately excluded: m16935 retired the geometry the
    # capytaine reference was frozen on (fullness parameter moved from zeta to
    # xi), so the frozen signature is the record of the *retired* hull.  The
    # fixed builder is graded by tests/test_wigley3_hull_alpha_axis.py, which
    # pins it pointwise against ``wigley3_freq``.
    for builder, rows in reference["builders"].items():
        if builder == "wigley3_hull":
            continue
        for row in rows:
            level = int(row["level"].split("=")[1].split(",")[0])
            mesh = build_mesh(builder, level)
            assert study.level_label(builder, level) == row["level"]
            signature = _signature(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64))
            assert signature == row["mesh_signature"], (
                f"{builder} {row['level']} mesh changed since the capytaine reference was frozen"
            )


def test_tensorbem_values_reproduce(payload: dict) -> None:
    """Live solver must still reproduce the artefact for the *unchanged* builder.

    The ``wigley3_hull`` cells in the frozen artefact are the pre-m16935
    record: that builder's geometry was corrected afterwards (Cwp 2/3 -> 0.8145),
    so its numbers are history, not a reproducibility target.
    """
    volume = wigley3_displaced_volume()
    for cell in payload["tensorbem"]:
        if cell["level"] != study.level_label(cell["builder"], MESH_LEVELS[0]):
            continue
        if cell["builder"] == "wigley3_hull":
            continue
        added, damping = tensorbem_cell(cell["builder"], MESH_LEVELS[0], OMEGA_STATED)
        for dof in study.DOFS:
            assert added[dof] == pytest.approx(cell["added_mass"][dof], rel=5e-3)
            assert damping[dof] == pytest.approx(cell["damping"][dof], rel=5e-3)
        assert cell["added_mass"]["Heave"] / (DENSITY * volume) == pytest.approx(
            added["Heave"] / (DENSITY * volume), rel=5e-3
        )


def test_every_recalled_entry_is_adjudicated(payload: dict) -> None:
    entries = recalled()
    adjudicated = payload["adjudication"]
    assert len(adjudicated) == len(entries) >= 14
    for original, result in zip(entries, adjudicated):
        assert result["constant"] == original["constant"]
        assert result["recalled"] == pytest.approx(original["recalled"])
        assert result["verdict"] in ("REPRODUCED", "FREQUENCY_MISMATCH", "REFUTED")
        # the recalled table is separated from both codes by a factor, not a band
        assert result["worst_abs_deviation"] > 0.10 if result["verdict"] == "REFUTED" else True
    refuted = [e for e in adjudicated if e["verdict"] == "REFUTED"]
    assert len(refuted) >= 14 - 1


def test_refuted_entries_name_the_factor_and_the_codes(payload: dict) -> None:
    for entry in payload["adjudication"]:
        if entry["verdict"] != "REFUTED":
            continue
        assert len(entry["computed"]) >= 2
        assert any(name.startswith("tensorbem/") for name in entry["computed"])
        assert any(name.startswith("capytaine/") for name in entry["computed"])
        assert abs(entry["factor"] - 1.0) > 0.10, entry["quantity"] + entry["dof"]


def test_geometry_gate_records_the_retired_defect_and_the_fix(payload: dict) -> None:
    """The frozen artefact documents the defect; the live code must be fixed."""
    geometry = payload["geometry"]
    # 1. the frozen record still carries the two-builder split it adjudicated
    assert geometry["cwp_wigley3_freq"] == pytest.approx(0.815, abs=5e-3)
    assert geometry["cwp_wigley3_hull"] == pytest.approx(2.0 / 3.0, abs=5e-3)
    assert geometry["max_abs_y_difference"] > 1e-2
    # 2. the shipped builders now describe one surface (m16935 fix)
    from tensorbem import wigley3_freq as wf
    from tensorbem import wigley3_hull as wh

    x = np.linspace(-0.5 * wh.WIGLEY3_L, 0.5 * wh.WIGLEY3_L, 401)
    z = np.zeros_like(x)
    live = np.asarray(wh.wigley3_half_breadth(x, z), dtype=float)
    other = wf.wigley3_halfbreadth(x, z, wh.WIGLEY3_L, wh.WIGLEY3_B, wh.WIGLEY3_T)
    assert np.max(np.abs(live - other)) < 1e-15 * wh.WIGLEY3_B
    assert 2.0 * np.trapezoid(live, x) / (wh.WIGLEY3_L * wh.WIGLEY3_B) == pytest.approx(0.8145, abs=1e-4)


def test_document_is_the_rendered_artefact(payload: dict) -> None:
    text = DOCUMENT.read_text()
    assert text == render_document(payload)


def test_figure_exists(payload: dict) -> None:
    figure = ROOT / study.FIG_PATH
    assert figure.exists() and figure.stat().st_size > 10_000
