"""Gates for the two-route finite-depth kernel map (m14854).

The published artefacts live in ``docs/finite_depth3d_route_map/`` and the prose
page ``docs/finite_depth3d_route_map.md`` embeds the generated block.  These
tests do three things:

1. re-derive the block from the committed JSON and refuse a hand edit;
2. refuse a stale numeric claim (the map's worst cell, the deep-water ladder and
   the rule ladder are pinned to the values the artefacts publish);
3. recompute a reduced grid *live* from the two kernels, so a regression in
   either route shows up even if nobody regenerates the artefacts.

Runtime is kept to a few seconds: the live check uses two depths, not the six of
the published map, and the quadrature route is called with the pinned
``nodes = 32``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem.benchmarks import finite_depth3d_route_map as route_map

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACT_DIR = REPO / "docs" / "finite_depth3d_route_map"
JSON_PATH = ARTIFACT_DIR / "route_map.json"
BLOCK_PATH = ARTIFACT_DIR / "route_map.md"
PAGE_PATH = REPO / "docs" / "finite_depth3d_route_map.md"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(JSON_PATH.read_text())


def test_committed_block_matches_the_json(report: dict) -> None:
    """The markdown block is generated from the JSON, never typed."""
    assert BLOCK_PATH.read_text() == route_map.render_markdown(report)


def test_published_page_embeds_the_block(report: dict) -> None:
    """The prose page must carry the current generated block verbatim."""
    block = route_map.render_markdown(report)
    page = PAGE_PATH.read_text()
    assert route_map.BEGIN_MARKER_FRAGMENT in page
    assert block.strip() in page


def test_map_worst_cell_is_below_the_pinned_floor(report: dict) -> None:
    """Two independent representations of the same kernel must agree tightly."""
    assert report["grid"]["nodes"] == route_map.MAP_NODES
    assert report["grid"]["cells"] == (
        len(report["grid"]["depth_parameters_k0h"])
        * len(report["grid"]["elevation_fractions"])
        * len(report["grid"]["ranges_k"])
    )
    assert report["worst_relative_difference"] < 1e-9
    for row in report["by_depth"].values():
        assert row["max_relative_difference"] < 1e-9
        assert row["min_abs_g"] > 0.0


def test_every_cell_agrees_and_no_cell_is_null(report: dict) -> None:
    """No cell may be silently empty, coincident, or dominated by one route."""
    for cell in report["cells"]:
        assert cell["relative_difference"] < 1e-9, cell
        assert cell["abs_g_quadrature"] > 0.0
        # the two routes must also agree in phase, not only in magnitude
        assert cell["g_eigenseries_real"] == pytest.approx(
            cell["g_quadrature_real"], abs=1e-9 * max(1.0, cell["abs_g_quadrature"])
        )
        assert cell["g_eigenseries_imag"] == pytest.approx(
            cell["g_quadrature_imag"], abs=1e-9 * max(1.0, cell["abs_g_quadrature"])
        )


def test_rule_ladder_identifies_the_discrepant_route(report: dict) -> None:
    """The default-rule residual is the *quadrature* route's panel rule.

    Refining the panel rule by a factor four must cut the near-surface residual
    by at least two orders of magnitude.  If the eigenseries were the culprit the
    residual would be flat in ``nodes``.
    """
    ladder = {
        (row["k0h"], row["elevation_fraction"], row["range_k"]): row[
            "relative_difference_by_nodes"
        ]
        for row in report["rule_ladder"]
    }
    for key in ((0.5, -0.02, 0.5), (0.5, -0.02, 2.0), (1.0, -0.02, 0.5)):
        values = ladder[key]
        default = values[str(route_map.LADDER_NODES[0])]     # nodes = 16
        refined = values[str(route_map.MAP_NODES)]           # nodes = 32
        assert default > 1e-9
        assert default / refined > 50.0


def test_deep_water_limit_decays_monotonically(report: dict) -> None:
    """At fixed absolute elevations the eigenseries approaches the Kelvin kernel."""
    raw = report["deep_water_worst_by_depth"]
    depths = sorted(float(key) for key in raw)
    by_depth = {f"{d:g}": raw[min(raw, key=lambda k: abs(float(k) - d))] for d in depths}
    values = [by_depth[f"{d:g}"] for d in depths]
    assert depths == sorted(depths)
    assert values == sorted(values, reverse=True)
    assert by_depth["77"] < 1e-8
    assert by_depth["30"] < 1e-5
    # honest FAIL guard: shallow water must NOT be claimed as a deep-water limit
    assert by_depth[min(by_depth, key=float)] > 1e-3


def test_live_reduced_grid_reproduces_the_published_agreement() -> None:
    """Recompute two depths from the kernels, independent of the artefacts."""
    for k0h in (0.5, 10.0):
        cell = route_map._cell(k0h, -0.02, 0.5)
        assert cell["relative_difference"] < 1e-9, cell
    deep = route_map._deep_limit(77.0, -0.2, -1.0, 0.5)
    assert deep["relative_difference"] < 1e-8, deep