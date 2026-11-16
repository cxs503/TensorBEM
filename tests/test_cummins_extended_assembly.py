"""Refusals of the extended-band assembler.

``sweep_extended.npz`` is judged by gates C8 / C3a_extended / C9, so the assembler
that writes it is part of the evidence chain: an incomplete sweep, a sweep from a
different frequency grid, or a sweep whose solver residual was defaulted to zero
must fail loudly instead of producing a certifiable artifact.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

WORKTREE = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = WORKTREE / "scripts"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def assembler():
    return _load("assemble_cummins_extended_json", "assemble_cummins_extended_json.py")


@pytest.fixture(scope="module")
def declared():
    return _load("case_cummins_wigley_extended_band", "case_cummins_wigley_extended_band.py")


def _record(assembler, index: int, **overrides) -> dict:
    omega = float(assembler.grid()[index])
    record = {
        "index": int(index),
        "omega": omega,
        "added_mass": np.zeros((6, 6)).tolist(),
        "damping": np.zeros((6, 6)).tolist(),
        "solver_relative_residual": 1e-10,
        "panel_count": 3840,
        "max_panel_scale": 0.0232,
        "n_sections": 60,
        "n_levels": 16,
        "wall_s": 1.0,
    }
    record.update(overrides)
    return record


def _line(record: dict) -> str:
    return f"NODE {record['index']:02d} {float(record['omega']):.6f} " + json.dumps(
        record, sort_keys=True
    )


def test_job_grid_matches_declared_grid(assembler, declared):
    """The parallel job wrapper and the pre-declared shard script share one grid."""
    assert np.array_equal(assembler.grid(), declared.grid())
    assert assembler.EXTENDED_MESH == declared.EXTENDED_MESH
    assert assembler.EXTENDED_BAND == declared.EXTENDED_BAND
    assert assembler.EXTENDED_N_OMEGA == declared.EXTENDED_N_OMEGA


def test_declared_grid_has_declared_band_ends(assembler):
    nodes = assembler.grid()
    assert nodes.size == 55
    assert nodes[0] == pytest.approx(0.2, rel=0, abs=0)
    assert nodes[-1] == pytest.approx(20.0, rel=0, abs=0)


def test_parse_lines_round_trips_all_nodes(assembler):
    lines = [_line(_record(assembler, k)) for k in range(55)]
    records = assembler.parse_lines(lines)
    assert sorted(records) == list(range(55))
    sweep = assembler.assemble(records)
    assert sweep.panel_count == 3840
    assert sweep.n_sections == 60 and sweep.n_levels == 16
    assert sweep.omega.size == 55
    assert sweep.added_mass.shape == (55, 6, 6)


def test_parse_lines_ignores_non_node_output(assembler):
    lines = [
        "extended sweep 60x16: 55 nodes [0, 55) of 55, 8 workers",
        _line(_record(assembler, 0)),
        "done 1 nodes in 12.0 s wall",
    ]
    assert sorted(assembler.parse_lines(lines)) == [0]


def test_parse_lines_rejects_duplicate_node(assembler):
    lines = [_line(_record(assembler, 3)), _line(_record(assembler, 3))]
    with pytest.raises(SystemExit):
        assembler.parse_lines(lines)


def test_parse_lines_rejects_foreign_frequency_grid(assembler):
    omega = float(assembler.grid()[7])
    lines = [_line(_record(assembler, 7, omega=omega * 1.001))]
    with pytest.raises(SystemExit):
        assembler.parse_lines(lines)


def test_parse_lines_rejects_missing_residual(assembler):
    record = _record(assembler, 0)
    record.pop("solver_relative_residual")
    with pytest.raises(SystemExit):
        assembler.parse_lines([_line(record)])


def test_parse_lines_rejects_out_of_grid_index(assembler):
    record = _record(assembler, 0)
    record["index"] = 55
    with pytest.raises(SystemExit):
        assembler.parse_lines([f"NODE 55 {20.0:.6f} " + json.dumps(record)])


def test_assemble_refuses_incomplete_sweep(assembler):
    records = {k: _record(assembler, k) for k in range(54)}
    with pytest.raises(SystemExit):
        assembler.assemble(records)


def test_assemble_refuses_mixed_mesh(assembler):
    records = {k: _record(assembler, k) for k in range(55)}
    records[10] = _record(assembler, 10, panel_count=48)
    with pytest.raises(SystemExit):
        assembler.assemble(records)


def test_assemble_refuses_undeclared_mesh(assembler):
    records = {
        k: _record(assembler, k, n_sections=44, n_levels=14, panel_count=616)
        for k in range(55)
    }
    with pytest.raises(SystemExit):
        assembler.assemble(records)


def test_assemble_takes_max_panel_scale(assembler):
    records = {k: _record(assembler, k) for k in range(55)}
    records[54] = _record(assembler, 54, max_panel_scale=0.0245)
    sweep = assembler.assemble(records)
    assert sweep.max_panel_scale == pytest.approx(0.0245)