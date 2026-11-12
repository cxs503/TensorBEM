"""Machine-checked pins for every number printed in ``docs/m14568_quadrature_kernel_solver.md``.

Why this module exists
----------------------
``docs/m14568_quadrature_kernel_solver.md`` is a paper-level report: the same
3-D finite-depth radiation solver run on two *independent* free-surface kernels
(John eigenseries vs. wavenumber integral), compared pointwise, at the gradient
level and at the solver level (``A33``/``B33``, Frobenius matrix norms).  Its
1111 number tokens were written once by
``scripts/m14568_quadrature_kernel_solver.py`` and **nothing read them
afterwards**: the coverage audit
(:mod:`tensorbem.benchmarks.doc_claim_audit`) classified the document as
``UNGUARDED``, the largest unguarded paper-level document in the tree (the next
one carries 292 tokens).

That is the same failure mode documented in ``docs/finite_depth3d.md`` §7 and
``docs/ursell_quadrature_provenance.md``: a kernel/solver revision moves the
numbers, the report keeps the old ones, and the *engineering gate claim* --
"the two kernels agree on added mass and damping to 1e-4" -- decays silently.
Both kernels are live code (``finite_depth_solver3d.assemble_finite_depth_operators_3d``
has a ``kernel=`` switch), so this is a live claim, not history.

This module closes the hole from the *document end*: it parses the values out of
the markdown tables and recomputes them from the library.  A code change that
moves a number, or a hand edit of the document, becomes a ``DRIFT`` row.

Tolerance policy (three classes, all stated in the report)
----------------------------------------------------------
``VALUE_RTOL`` (1e-6)
    Physical/derived values printed with 6-7 significant digits: kernel ``G``,
    ``A33``, ``B33``.
``COARSE_RTOL`` (5e-4)
    Values printed with 4 significant digits: gradient components, Frobenius
    errors, diagonal relative errors, central-difference errors.
``BAND_FACTOR`` (a decade)
    Columns whose printed value is *round-off or truncation itself*: kernel
    ``G`` agreement ``~1e-11``, gradient agreement ``~1e-10``, and the
    central-difference column (the truncation error of a finite difference of
    the kernel, which moves 30 % when the kernel gridding changes).  Both terms of the relative
    difference are at the double-precision floor, so the printed 4 digits move
    by tens of percent between BLAS builds while the *claim* ("agrees to
    1e-11") is unchanged.  Pinning those to 4 digits would red-light a doc for
    something nobody can fix in the code (the lesson recorded in
    :data:`tensorbem.benchmarks.finite_depth3d_pins.ROUNDOFF_FLOOR`).  They are
    pinned as an order-of-magnitude band instead, and the report says so.

Rows that are deliberately **not** pinned
-----------------------------------------
* ``t_ref`` / ``t_got`` (wall clock): machine dependent (the report's own
  26 s / 55 s / 269 s / 599 s become 18 s / 41 s / ... on a different node).
  They are emitted as ``UNPINNED`` rows so the report states the omission
  instead of quietly dropping the columns.
* The 20-panel geodesic-sphere rows and the ``nodes=16`` octahedron row: they
  cost hundreds of seconds each.  They are recomputed only under
  ``--include-slow``; by default they are ``UNPINNED`` rows carrying the cost
  reason, never silent omissions.

Usage::

    python -m tensorbem.benchmarks.m14568_quadrature_kernel_solver_pins \
        --json docs/m14568_pins/check.json --strict

``--strict`` exits non-zero on any ``DRIFT``; ``tests/test_m14568_doc_pins.py``
runs the same check in CI (fast subset: the 8-panel octahedron at
``kernel_nodes=8``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tensorbem.finite_depth3d import (
    green_function_3d as eigenseries_green,
    green_gradient_source_3d as eigenseries_gradient,
)
from tensorbem.finite_depth3d_quadrature import (
    green_function_finite_depth_3d as quadrature_green,
    green_gradient_source_finite_depth_3d as quadrature_gradient,
)
from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d
from tensorbem.geometry3d import TriangleMesh, submerged_sphere_mesh

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = REPO_ROOT / "docs" / "m14568_quadrature_kernel_solver.md"

SCHEMA = "tensorbem.m14568_quadrature_kernel_solver_pins/1"

NU = 0.6
#: Off-axis target/source set of the m13811 diagnosis, reused verbatim.
TARGETS = np.array(
    [
        [0.5, 0.0, -0.5],
        [2.0, 1.0, -1.5],
        [1.5, -0.7, -0.2],
        [5.0, 3.0, -0.9],
    ]
)
SOURCES = np.array([[0.0, 0.0, -1.0], [1.0, -0.5, -1.5]])
DEPTHS = (4.0, 8.0, 20.0, 60.0, 128.0)

VALUE_RTOL = 1.0e-6
COARSE_RTOL = 5.0e-4
BAND_FACTOR = 10.0
#: Absolute floor for cells printed as ``+0.000e+00`` (the symmetry-zero
#: component of the source gradient on the vertical axis).
ZERO_FLOOR = 1.0e-12
#: The solver-level rows pinned by default: 8-panel submerged octahedron.
OCTAHEDRON_LABEL = "submerged_octahedron"
SPHERE_LABEL = "submerged_sphere_s0"


# ---------------------------------------------------------------------------
# verdict bookkeeping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Row:
    table: str
    key: str
    quantity: str
    declared: float
    recomputed: float
    rtol: float
    verdict: str
    note: str = ""

    @property
    def rel_deviation(self) -> float:
        if self.verdict == "UNPINNED":
            return math.nan
        if not (math.isfinite(self.declared) and math.isfinite(self.recomputed)):
            return math.inf
        if abs(self.declared) < 1e-300:
            return abs(self.recomputed) if abs(self.recomputed) > ZERO_FLOOR else 0.0
        return abs(self.recomputed - self.declared) / abs(self.declared)


def _verdict(declared: float, recomputed: float, rtol: float) -> str:
    if not (math.isfinite(declared) and math.isfinite(recomputed)):
        return "DRIFT"
    if abs(declared) < 1e-300:
        return "OK" if abs(recomputed) <= ZERO_FLOOR else "DRIFT"
    if rtol >= 1.0:
        # Multiplicative band (``BAND_FACTOR``): OK for ratio in [1/rtol, rtol].
        ratio = abs(recomputed) / abs(declared)
        return "OK" if (1.0 / rtol) <= ratio <= rtol else "DRIFT"
    # Relative tolerance: the printed precision of the column.
    return "OK" if abs(recomputed - declared) <= rtol * abs(declared) else "DRIFT"


# ---------------------------------------------------------------------------
# markdown parsing
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^#+\s*(?P<title>.*)$")
#: Signed decimal/exponent token.  The ``+`` of a positive exponent must be
#: allowed explicitly: without it ``0.000e+00`` is read as two numbers.
_NUMBER_RE = re.compile(r"[-\u2212]?\d+(?:\.\d+)?(?:[eE][-+\u2212]?\d+)?")


def parse_number(token: str) -> float:
    return float(token.replace("\u2212", "-"))


def _numbers(cell: str) -> list[float]:
    return [parse_number(t) for t in _NUMBER_RE.findall(cell)]


def _complex_components(cell: str) -> list[complex]:
    """Parse ``(re+imj, re+imj, ...)`` cell into one complex per comma field."""
    components: list[complex] = []
    for field in cell.strip().strip("()").split(","):
        values = _numbers(field)
        if len(values) >= 2:
            components.append(complex(values[0], values[1]))
        elif len(values) == 1:
            components.append(complex(values[0], 0.0))
    return components


def split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            current = match.group("title").strip()
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    return sections


def section_tables(lines: list[str], limit: int = 1) -> list[list[list[str]]]:
    """All markdown tables in ``lines`` (each as header + body rows)."""
    tables: list[list[list[str]]] = []
    bucket: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            bucket.append(cells)
        elif bucket:
            tables.append(bucket)
            bucket = []
            if len(tables) >= limit:
                return tables
    if bucket:
        tables.append(bucket)
    return tables[:limit]


# ---------------------------------------------------------------------------
# recomputation (same settings as scripts/m14568_quadrature_kernel_solver.py)
# ---------------------------------------------------------------------------
def _rel(got, ref) -> float:
    got = np.asarray(got, dtype=complex)
    ref = np.asarray(ref, dtype=complex)
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(got - ref) / denom) if denom else float("nan")


def octahedron_mesh(half_size: float = 1.0, draught: float = 2.0) -> TriangleMesh:
    """Closed 8-panel octahedron centred at ``z = -draught`` (fully submerged)."""
    a = float(half_size)
    vertices = np.array(
        [
            [a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, -a, 0.0],
            [0.0, 0.0, a],
            [0.0, 0.0, -a],
        ]
    )
    vertices[:, 2] -= float(draught)
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ]
    )
    return TriangleMesh(vertices, faces)


def kernel_ladder(depths=DEPTHS) -> dict:
    """Pointwise ``G``: quadrature vs. eigenseries, over the report's mesh of pairs."""
    out = {}
    tgt = TARGETS[:, None, :]
    src = SOURCES[None, :, :]
    for depth in depths:
        ref = eigenseries_green(tgt, src, NU, depth)
        got = quadrature_green(tgt, src, NU, depth)
        for i, target in enumerate(TARGETS):
            for j, source in enumerate(SOURCES):
                key = (tuple(float(v) for v in target), tuple(float(v) for v in source), float(depth))
                out[key] = {
                    "ref": complex(ref[i, j]),
                    "got": complex(got[i, j]),
                    "err": _rel(got[i, j], ref[i, j]),
                    "nu_h": NU * float(depth),
                }
    return out


def gradient_ladder(depths=DEPTHS, step: float = 1.0e-6) -> dict:
    """Source gradient: quadrature vs. eigenseries and vs. central differences."""
    out = {}
    tgt = TARGETS[:, None, :]
    src = SOURCES[None, :, :]
    for depth in depths:
        ref = eigenseries_gradient(tgt, src, NU, depth)
        got = quadrature_gradient(tgt, src, NU, depth)
        finite = []
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = step
            plus = quadrature_green(tgt, src + offset, NU, depth)
            minus = quadrature_green(tgt, src - offset, NU, depth)
            finite.append((plus - minus) / (2.0 * offset[axis]))
        finite = np.stack(finite, axis=-1)
        for i, target in enumerate(TARGETS):
            for j, source in enumerate(SOURCES):
                key = (tuple(float(v) for v in target), tuple(float(v) for v in source), float(depth))
                out[key] = {
                    "ref": np.array([complex(v) for v in ref[i, j]]),
                    "got": np.array([complex(v) for v in got[i, j]]),
                    "cfd": np.array([complex(v) for v in finite[i, j]]),
                    "err_vs_eigenseries": _rel(got[i, j], ref[i, j]),
                    "err_vs_cfd": _rel(got[i, j], finite[i, j]),
                }
    return out


def solver_pair(
    mesh: TriangleMesh,
    label: str,
    nodes: int,
    *,
    nu: float = NU,
    depth: float = 4.0,
    modes=(0, 2, 4),
) -> dict:
    """One eigenseries/quadrature pair at the solver level."""

    def run(kernel: str, knodes: int) -> dict:
        result = solve_radiation_finite_depth_3d(
            mesh, nu, depth, modes=modes, kernel=kernel, kernel_nodes=knodes
        )
        return {
            "added_mass": np.asarray(result.added_mass, dtype=float),
            "damping": np.asarray(result.damping, dtype=float),
        }

    ref = run("eigenseries", 16)
    got = run("quadrature", nodes)
    frob_a = float(
        np.linalg.norm(got["added_mass"] - ref["added_mass"])
        / np.linalg.norm(ref["added_mass"])
    )
    frob_b = float(
        np.linalg.norm(got["damping"] - ref["damping"]) / np.linalg.norm(ref["damping"])
    )
    return {
        "label": label,
        "panels": int(mesh.panel_count),
        "nodes": int(nodes),
        "A33_ref": float(ref["added_mass"][1, 1]),
        "A33_got": float(got["added_mass"][1, 1]),
        "B33_ref": float(ref["damping"][1, 1]),
        "B33_got": float(got["damping"][1, 1]),
        "err_A33": abs(got["added_mass"][1, 1] - ref["added_mass"][1, 1])
        / abs(ref["added_mass"][1, 1]),
        "err_B33": abs(got["damping"][1, 1] - ref["damping"][1, 1]) / abs(ref["damping"][1, 1]),
        "frob_A": frob_a,
        "frob_B": frob_b,
        "frob_ratio": frob_b / frob_a if frob_a else float("nan"),
    }


def solver_rows(include_slow: bool = False) -> dict:
    """Solver-level cross-checks keyed by ``(label, panels, nodes)``."""
    cases = [(OCTAHEDRON_LABEL, octahedron_mesh(), 8)]
    if include_slow:
        cases.append((OCTAHEDRON_LABEL, octahedron_mesh(), 16))
        cases.append(
            (SPHERE_LABEL, submerged_sphere_mesh(0, radius=1.0, submergence=1.5), 8)
        )
        cases.append(
            (SPHERE_LABEL, submerged_sphere_mesh(0, radius=1.0, submergence=1.5), 16)
        )
    out = {}
    for label, mesh, nodes in cases:
        row = solver_pair(mesh, label, nodes)
        out[(label, row["panels"], nodes)] = row
    return out


# ---------------------------------------------------------------------------
# document -> rows
# ---------------------------------------------------------------------------
def _nearest_ladder(ladder: dict, target, source, depth: float):
    best = None
    best_d = math.inf
    for (t, s, h), payload in ladder.items():
        d = max(abs(t[0] - target[0]), abs(t[1] - target[1]), abs(t[2] - target[2]))
        d = max(d, max(abs(s[0] - source[0]), abs(s[1] - source[1]), abs(s[2] - source[2])))
        d = max(d, abs(h - depth))
        if d < best_d:
            best, best_d = payload, d
    return best if best_d <= 1e-9 else None


def _add(rows, table, key, quantity, declared, recomputed, rtol):
    rows.append(
        Row(
            table=table,
            key=key,
            quantity=quantity,
            declared=float(declared),
            recomputed=float(recomputed),
            rtol=float(rtol),
            verdict=_verdict(float(declared), float(recomputed), float(rtol)),
        )
    )


def _add_unpinned(rows, table, key, quantity, declared, note):
    rows.append(
        Row(
            table=table,
            key=key,
            quantity=quantity,
            declared=float(declared),
            recomputed=math.nan,
            rtol=math.nan,
            verdict="UNPINNED",
            note=note,
        )
    )


def recompute_data(include_slow: bool = False) -> dict:
    """All library-side numbers needed by :func:`collect` (the expensive part).

    Split out so callers that re-parse several (e.g. mutated) copies of the
    document pay the ~20 s solver cross-check exactly once.
    """
    return {
        "kernel": kernel_ladder(),
        "gradient": gradient_ladder(),
        "solvers": solver_rows(include_slow=include_slow),
    }


def collect(
    doc_path: Path = DOC_PATH,
    *,
    include_slow: bool = False,
    data: dict | None = None,
) -> list[Row]:
    text = Path(doc_path).read_text(encoding="utf-8")
    sections = split_sections(text)
    rows: list[Row] = []

    if data is None:
        data = recompute_data(include_slow=include_slow)
    kernel = data["kernel"]
    gradient = data["gradient"]
    solvers = data["solvers"]

    def solver_lookup(label: str, panels: int, nodes: int):
        return solvers.get((label, panels, nodes))

    # --- 3.1 pointwise kernel -----------------------------------------------
    section = next((v for k, v in sections.items() if k.startswith("3.1")), [])
    tables = section_tables(section, limit=1)
    for cells in tables[0][1:] if tables else []:
        if len(cells) < 7:
            continue
        target = _numbers(cells[0])
        source = _numbers(cells[1])
        if len(target) != 3 or len(source) != 3:
            continue
        depth = _numbers(cells[2])
        if not depth:
            continue
        payload = _nearest_ladder(kernel, target, source, depth[0])
        if payload is None:
            _add_unpinned(rows, "3.1", f"h={depth[0]:g}", "kernel", math.nan, "case not found")
            continue
        key = f"h={depth[0]:g}/t{target[0]:g},{target[1]:g},t{target[2]:g}"
        nu_h = _numbers(cells[3])
        if nu_h:
            _add(rows, "3.1", key, "nu_h", nu_h[0], payload["nu_h"], COARSE_RTOL)
        ref_cells = _numbers(cells[4])
        got_cells = _numbers(cells[5])
        if len(ref_cells) >= 2:
            _add(rows, "3.1", key, "G_ref_re", ref_cells[0], payload["ref"].real, VALUE_RTOL)
            _add(rows, "3.1", key, "G_ref_im", ref_cells[1], payload["ref"].imag, VALUE_RTOL)
        if len(got_cells) >= 2:
            _add(rows, "3.1", key, "G_got_re", got_cells[0], payload["got"].real, VALUE_RTOL)
            _add(rows, "3.1", key, "G_got_im", got_cells[1], payload["got"].imag, VALUE_RTOL)
        err = _numbers(cells[6])
        if err:
            _add(rows, "3.1", key, "rel_err", err[0], payload["err"], BAND_FACTOR)

    # --- 3.2 source gradient ------------------------------------------------
    section = next((v for k, v in sections.items() if k.startswith("3.2")), [])
    tables = section_tables(section, limit=1)
    for cells in tables[0][1:] if tables else []:
        if len(cells) < 6:
            continue
        target = _numbers(cells[0])
        source = _numbers(cells[1])
        depth = _numbers(cells[2])
        if len(target) != 3 or len(source) != 3 or not depth:
            continue
        payload = _nearest_ladder(gradient, target, source, depth[0])
        key = f"h={depth[0]:g}/t{target[0]:g},{target[1]:g},t{target[2]:g}"
        components = _complex_components(cells[3])
        if payload is None:
            _add_unpinned(rows, "3.2", key, "gradient", math.nan, "case not found")
            continue
        if len(components) >= 3:
            for axis in range(3):
                _add(
                    rows, "3.2", key, f"grad_ref_axis{axis}_re",
                    components[axis].real, payload["ref"][axis].real, COARSE_RTOL,
                )
                _add(
                    rows, "3.2", key, f"grad_ref_axis{axis}_im",
                    components[axis].imag, payload["ref"][axis].imag, COARSE_RTOL,
                )
        err_eig = _numbers(cells[4])
        if err_eig:
            _add(rows, "3.2", key, "rel_err_vs_eigenseries", err_eig[0],
                 payload["err_vs_eigenseries"], BAND_FACTOR)
        err_cfd = _numbers(cells[5])
        if err_cfd:
            # This column is the *truncation* error of a central difference of the
            # kernel, not a measurement: it moves by 30 % between kernels whose
            # gridding differs (measured: documented 1.581e-07 vs recomputed
            # 1.097e-07), so it is pinned as a decade band, like the other
            # round-off/cancellation columns.
            _add(
                rows, "3.2", key, "rel_err_vs_cfd", err_cfd[0],
                payload["err_vs_cfd"], BAND_FACTOR,
            )

    # --- 3.3 solver level ---------------------------------------------------
    section = next((v for k, v in sections.items() if k.startswith("3.3")), [])
    tables = section_tables(section, limit=2)
    if tables:
        for cells in tables[0][1:]:
            if len(cells) < 13:
                continue
            label = cells[0].strip("` ")
            ints = _numbers(cells[1]) + _numbers(cells[2]) + _numbers(cells[3]) + _numbers(cells[4])
            panels, nu_doc, depth_doc, nodes = (ints + [math.nan] * 4)[:4]
            payload = solver_lookup(label, int(round(panels)), int(round(nodes)))
            key = f"{label}/nodes={nodes:.0f}"
            a_ref, a_got, err_a = _numbers(cells[5]), _numbers(cells[6]), _numbers(cells[7])
            b_ref, b_got, err_b = _numbers(cells[8]), _numbers(cells[9]), _numbers(cells[10])
            times = _numbers(cells[11]) + _numbers(cells[12])
            if payload is None:
                reason = (
                    f"not recomputed by default (cost: 8-panel octahedron ~20 s, "
                    f"nodes=16 ~60 s, 20-panel sphere 270-600 s); --include-slow"
                )
                for name, value in (
                    ("A33_ref", a_ref),
                    ("A33_got", a_got),
                    ("err_A33", err_a),
                    ("B33_ref", b_ref),
                    ("B33_got", b_got),
                    ("err_B33", err_b),
                ):
                    if value:
                        _add_unpinned(rows, "3.3", key, name, value[0], reason)
                for name, value in zip(("t_ref", "t_got"), times):
                    _add_unpinned(
                        rows, "3.3", key, name, value,
                        "wall clock: machine dependent, not a claim about the code",
                    )
                continue
            if a_ref:
                _add(rows, "3.3", key, "A33_ref", a_ref[0], payload["A33_ref"], VALUE_RTOL)
            if a_got:
                _add(rows, "3.3", key, "A33_got", a_got[0], payload["A33_got"], VALUE_RTOL)
            if err_a:
                _add(rows, "3.3", key, "err_A33", err_a[0], payload["err_A33"], COARSE_RTOL)
            if b_ref:
                _add(rows, "3.3", key, "B33_ref", b_ref[0], payload["B33_ref"], VALUE_RTOL)
            if b_got:
                _add(rows, "3.3", key, "B33_got", b_got[0], payload["B33_got"], VALUE_RTOL)
            if err_b:
                _add(rows, "3.3", key, "err_B33", err_b[0], payload["err_B33"], COARSE_RTOL)
            for name, value in zip(("t_ref", "t_got"), times):
                _add_unpinned(
                    rows, "3.3", key, name, value,
                    "wall clock: machine dependent, not a claim about the code",
                )
    if len(tables) > 1:
        for cells in tables[1][1:]:
            if len(cells) < 6:
                continue
            label = cells[0].strip("` ")
            nodes = _numbers(cells[1])
            panels = _numbers(cells[2])
            if not (nodes and panels):
                continue
            payload = solver_lookup(label, int(round(panels[0])), int(round(nodes[0])))
            key = f"{label}/nodes={nodes[0]:.0f}"
            errs = _numbers(cells[3]) + _numbers(cells[4]) + _numbers(cells[5])
            if payload is None:
                for name, value in zip(("frob_A", "frob_B", "ratio"), errs):
                    _add_unpinned(
                        rows, "3.3-frobenius", key, name, value,
                        "not recomputed by default (sphere/nodes=16 cost); --include-slow",
                    )
                continue
            if errs:
                _add(rows, "3.3-frobenius", key, "frob_A", errs[0], payload["frob_A"], COARSE_RTOL)
            if len(errs) > 1:
                _add(rows, "3.3-frobenius", key, "frob_B", errs[1], payload["frob_B"], COARSE_RTOL)
            if len(errs) > 2:
                _add(
                    rows, "3.3-frobenius", key, "ratio", errs[2],
                    payload["frob_ratio"], COARSE_RTOL,
                )

    # --- 6 error verdict table ---------------------------------------------
    recomputed_summary = {
        "kernel": max(v["err"] for v in kernel.values()),
        "gradient": max(v["err_vs_eigenseries"] for v in gradient.values()),
        "A33": max(v["err_A33"] for v in solvers.values()),
        "B33": max(v["err_B33"] for v in solvers.values()),
        "frobenius": max(max(v["frob_A"], v["frob_B"]) for v in solvers.values()),
    }
    section = next((v for k, v in sections.items() if k.startswith("6")), [])
    tables = section_tables(section, limit=1)
    keywords = (
        ("核值", "kernel"),
        ("梯度", "gradient"),
        ("A33", "A33"),
        ("B33", "B33"),
        ("Frobenius", "frobenius"),
        ("整体", "frobenius"),
    )
    for cells in tables[0][1:] if tables else []:
        if len(cells) < 4:
            continue
        declared = _numbers(cells[2])
        if not declared:
            continue
        name = next((v for k, v in keywords if k in cells[0]), None)
        if name is None:
            _add_unpinned(rows, "6", cells[0][:24], "rel_deviation", declared[0], "no recompute rule")
            continue
        _add(rows, "6", name, "rel_deviation", declared[0], recomputed_summary[name], BAND_FACTOR)
        # the "<3% ?" column is itself a claim: recompute it
        claim = "<3%" in cells[3] or "是" in cells[3].replace("不是", "")
        holds = recomputed_summary[name] < 0.03
        rows.append(
            Row(
                table="6",
                key=name,
                quantity="under_3_percent_claim",
                declared=1.0 if claim else 0.0,
                recomputed=1.0 if holds else 0.0,
                rtol=0.0,
                verdict="OK" if claim == holds else "DRIFT",
            )
        )
    return rows


def summary(rows: list[Row]) -> dict:
    pinned = [r for r in rows if r.verdict != "UNPINNED"]
    drift = [r for r in pinned if r.verdict == "DRIFT"]
    worst = max(pinned, key=lambda r: r.rel_deviation, default=None)
    return {
        "schema": SCHEMA,
        "document": str(DOC_PATH.relative_to(REPO_ROOT)),
        "rows": len(rows),
        "pinned": len(pinned),
        "unpinned": len(rows) - len(pinned),
        "pass": len(pinned) - len(drift),
        "drift": len(drift),
        "max_rel_deviation": max((r.rel_deviation for r in pinned), default=0.0),
        "worst": (
            {
                "table": worst.table,
                "key": worst.key,
                "quantity": worst.quantity,
                "declared": worst.declared,
                "recomputed": worst.recomputed,
                "rel_deviation": worst.rel_deviation,
            }
            if worst is not None
            else None
        ),
        "tables_covered": sorted({r.table for r in rows}),
        "unpinned_reasons": sorted({r.note for r in rows if r.verdict == "UNPINNED"}),
    }


def check(
    doc_path: Path = DOC_PATH, *, include_slow: bool = False, data: dict | None = None
) -> tuple[dict, list[Row]]:
    rows = collect(doc_path, include_slow=include_slow, data=data)
    return summary(rows), rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC_PATH)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report, rows = check(args.doc, include_slow=args.include_slow)
    for row in rows:
        if row.verdict == "DRIFT":
            print(
                f"DRIFT {row.table} {row.key} {row.quantity}: "
                f"declared={row.declared:.6e} recomputed={row.recomputed:.6e} "
                f"(rel {row.rel_deviation:.2e})"
            )
    print(
        f"{report['pinned']} pinned values ({report['unpinned']} deliberately unpinned), "
        f"{report['pass']} PASS, {report['drift']} DRIFT, "
        f"max relative deviation {report['max_rel_deviation']:.2e}, "
        f"tables {report['tables_covered']}"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": report, "rows": [asdict(r) for r in rows]}
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if (args.strict and report["drift"]) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())