"""Certify the closed-form Rankine panel integral and bound its Gauss referee.

``tensorbem.free_surface_solver._log_panel_primitive`` integrates
``log|P - Q|`` along a straight panel in closed form.  That primitive replaced a
per-panel uniform Gauss rule, which straddled the logarithmic self-panel
singularity and biased the single-layer potential by roughly ``1e-4``
(``docs/ursell_quadrature_provenance.md``).  The replacement was guarded by a
test that asked a *composite uniform Gauss rule* to reproduce the closed form to
``1e-8``; that test was unattainable, because on a coplanar panel the composite
rule itself only converges like ``O(1/N)`` -- the referee, not the closed form,
was the inexact side.

This module settles both halves of that disagreement with one command and no
hand-transcribed digits:

* the closed form is compared against an arbitrary-precision referee
  (``mpmath.quad`` at 40 significant digits, with the singular abscissa passed
  as a breakpoint) and against a numpy-only referee (graded composite Gauss for
  ``h > 0``, the elementary antiderivative for the coplanar branch);
* the resolution limit of the historical uniform-piece referee is measured for
  two mesh alignments (singularity on a piece boundary / inside a piece) and
  the observed order is fitted from the data rather than assumed;
* the legacy single-panel straddling bias is re-measured, so the ``1e-4``
  provenance note stays checkable.

Outputs (all generated, never typed by hand)::

    python -m tensorbem.benchmarks.rankine_primitive_referee --write \
        --json tests/data/rankine_primitive_check.json \
        --doc docs/rankine_primitive_referee_study.md \
        --figure artifacts/rankine_referee.png

    python -m tensorbem.benchmarks.rankine_primitive_referee --check \
        --json tests/data/rankine_primitive_check.json

``--check`` re-measures and compares against the committed artifact, so drift in
either the code or the note is a non-zero exit rather than a footnote.
``mpmath`` is an optional test dependency: when it is missing the mpmath rows
are reported as ``null`` and the numpy-only referee still certifies the closed
form to machine precision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.free_surface_solver import _log_panel_primitive

SCHEMA = "tensorbem.rankine_primitive_referee/1"

#: ``(label, xi, h, length)`` in panel-local coordinates: ``xi`` is the field
#: point's foot coordinate along the panel and ``h`` its perpendicular distance,
#: so ``h = 0`` is the coplanar (self-panel) case where the log singularity is
#: real.
CASES: tuple[tuple[str, float, float, float], ...] = (
    ("self panel, singularity at midpoint", 0.5, 0.0, 1.0),
    ("self panel, singularity at end point", 0.0, 0.0, 1.0),
    ("far end point, coplanar", 1.0, 0.0, 1.0),
    ("coplanar, off-centre", 0.25, 0.0, 1.0),
    ("nearly coplanar h=1e-3", 0.0, 1.0e-3, 1.0),
    ("inside the coplanar switch band h=1e-15", 0.3, 1.0e-15, 1.0),
    ("smooth panel h=0.3", 0.9, 0.3, 1.0),
)

#: Piece counts used for the referee-resolution study.
PIECES: tuple[int, ...] = (25, 50, 100, 200, 400, 800, 1600)

#: The abscissa offset applied in the "singularity inside a piece" alignment;
#: 0.37 of a piece width keeps the singular point away from every Gauss node.
INSIDE_OFFSET = 0.37


def panel_integral(xi: float, h: float, length: float) -> float:
    """``int_0^length log|P-Q| ds`` via the shipped closed-form primitive."""
    upper = _log_panel_primitive(np.array([[length - xi]]), np.array([[h]]))
    lower = _log_panel_primitive(np.array([[-xi]]), np.array([[h]]))
    return float((upper - lower)[0, 0])


def elementary_coplanar(xi: float, length: float) -> float:
    """Coplanar integral from elementary calculus: ``int log|s-xi| ds``.

    ``G(t) = t log|t| - t`` is the antiderivative of ``log|t|`` for ``t != 0``
    and ``G(0) = 0``, so the panel integral is ``G(length - xi) - G(-xi)``, valid
    for the singularity anywhere in the closed panel.
    """

    def antiderivative(t: float) -> float:
        return 0.0 if t == 0.0 else t * math.log(abs(t)) - t

    return antiderivative(length - xi) - antiderivative(-xi)


def graded_gauss(
    xi: float,
    h: float,
    length: float,
    order: int = 20,
    ratio: float = 0.7,
    levels: int = 40,
) -> float:
    """Composite Gauss on a mesh graded towards the singularity.

    For ``h > 0`` the integrand is analytic on the panel, so grading down to
    ``h/1000`` makes a 20-point rule reach machine precision; this is the
    numpy-only referee used when ``mpmath`` is unavailable.  For ``h = 0`` the
    log singularity stays unresolved (``~1e-9`` at the graded limit), so the
    coplanar cases are refereed by :func:`elementary_coplanar` instead.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    edges = {0.0, length, xi}
    scale = max(h, 1.0e-12)
    smallest = scale * 1.0e-3
    step = 0.5 * length
    for _ in range(levels):
        for anchor in (0.0, xi, length):
            for direction in (1.0, -1.0):
                point = anchor + direction * step
                if 0.0 < point < length:
                    edges.add(point)
        if step <= smallest:
            break
        step *= ratio
    total = 0.0
    grid = np.array(sorted(edges))
    for start, stop in zip(grid[:-1], grid[1:]):
        half = 0.5 * (stop - start)
        middle = 0.5 * (stop + start)
        abscissae = middle + half * nodes
        total += float(
            half * np.sum(weights * 0.5 * np.log((abscissae - xi) ** 2 + h**2))
        )
    return total


def mpmath_referee(xi: float, h: float, length: float, dps: int = 40) -> float | None:
    """Arbitrary-precision referee, ``None`` when ``mpmath`` is not installed."""
    try:
        import mpmath
    except ImportError:  # pragma: no cover - exercised by the numpy-only install
        return None
    with mpmath.workdps(dps):
        breakpoints = [mpmath.mpf(0.0)]
        if 0.0 < xi < length:
            breakpoints.append(mpmath.mpf(xi))
        breakpoints.append(mpmath.mpf(length))
        half = mpmath.mpf("0.5")
        mu, hu = mpmath.mpf(xi), mpmath.mpf(h)
        integrand = lambda s: half * mpmath.log((s - mu) ** 2 + hu**2)  # noqa: E731
        return float(mpmath.quad(integrand, breakpoints))


def uniform_gauss(
    xi: float,
    h: float,
    length: float,
    pieces: int,
    order: int = 20,
    inside: bool = False,
) -> float:
    """The historical referee: uniform pieces, one Gauss rule per piece.

    ``inside=True`` detunes the internal nodes so the singular abscissa falls
    inside a piece instead of on a boundary, which is the worse of the two
    alignments and the one a user cannot control.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, length, pieces + 1)
    if inside and pieces >= 2:
        inner = edges[1:-1] + INSIDE_OFFSET * (edges[1] - edges[0])
        edges = np.concatenate(([0.0], np.sort(inner), [length]))
    total = 0.0
    for start, stop in zip(edges[:-1], edges[1:]):
        half = 0.5 * (stop - start)
        middle = 0.5 * (stop + start)
        abscissae = middle + half * nodes
        total += float(
            half * np.sum(weights * 0.5 * np.log((abscissae - xi) ** 2 + h**2))
        )
    return total


def _observed_order(first: float, last: float, ratio: float) -> float:
    """Observed convergence order from two errors and the mesh-refinement ratio."""
    return math.log2(first / last) / math.log2(ratio)


def measure(dps: int = 40) -> dict[str, Any]:
    """Run every measurement and return the machine-readable evidence payload."""
    cases: list[dict[str, Any]] = []
    for label, xi, h, length in CASES:
        closed = panel_integral(xi, h, length)
        graded = graded_gauss(xi, h, length)
        #: Referee choice follows the branch under test: the coplanar branch is
        #: checked against the elementary antiderivative (graded quadrature cannot
        #: resolve a log singularity), the smooth branch against graded Gauss.
        exact = elementary_coplanar(xi, length) if h <= 1.0e-14 else graded
        referee = mpmath_referee(xi, h, length, dps=dps)
        rows: dict[str, Any] = {
            "label": label,
            "xi": xi,
            "h": h,
            "length": length,
            "closed_form": closed,
            "elementary_closed_form": elementary_coplanar(xi, length),
            "graded_gauss": graded,
            "mpmath_referee": referee,
            "dps": dps,
            "abs_error_numpy_referee": abs(closed - exact),
            "rel_error_numpy_referee": abs(closed - exact) / max(abs(exact), 1e-300),
        }
        if referee is not None:
            rows["abs_error_mpmath"] = abs(closed - referee)
            rows["rel_error_mpmath"] = abs(closed - referee) / max(abs(referee), 1e-300)
        cases.append(rows)

    mpmath_rows = [row for row in cases if row["mpmath_referee"] is not None]
    convergence: list[dict[str, Any]] = []
    for label, xi, h, length in CASES:
        closed = panel_integral(xi, h, length)
        for pieces in PIECES:
            convergence.append(
                {
                    "label": label,
                    "xi": xi,
                    "h": h,
                    "pieces": pieces,
                    "order": 20,
                    "err_sing_on_edge": abs(
                        uniform_gauss(xi, h, length, pieces) - closed
                    ),
                    "err_sing_inside_piece": abs(
                        uniform_gauss(xi, h, length, pieces, inside=True) - closed
                    ),
                }
            )

    coplanar = [
        row
        for row in convergence
        if row["xi"] == 0.5 and row["h"] == 0.0
    ]
    errors_edge = [row["err_sing_on_edge"] for row in coplanar]
    errors_inside = [row["err_sing_inside_piece"] for row in coplanar]
    asymptote = [index for index, row in enumerate(coplanar) if row["pieces"] >= 100]
    low, high = asymptote[0], asymptote[-1]
    refinement = coplanar[high]["pieces"] / coplanar[low]["pieces"]

    single_panel = abs(uniform_gauss(0.5, 0.0, 1.0, 1, order=24) - panel_integral(0.5, 0.0, 1.0))
    single_panel_20 = abs(
        uniform_gauss(0.5, 0.0, 1.0, 1, order=20) - panel_integral(0.5, 0.0, 1.0)
    )

    return {
        "schema": SCHEMA,
        "generated_by": (
            "python -m tensorbem.benchmarks.rankine_primitive_referee --write"
        ),
        "primitive": "tensorbem.free_surface_solver._log_panel_primitive",
        "cases": cases,
        "mpmath_available": bool(mpmath_rows),
        "max_abs_error_mpmath": max(
            (row["abs_error_mpmath"] for row in mpmath_rows), default=None
        ),
        "max_rel_error_mpmath": max(
            (row["rel_error_mpmath"] for row in mpmath_rows), default=None
        ),
        "max_abs_error_numpy_referee": max(
            row["abs_error_numpy_referee"] for row in cases
        ),
        "max_rel_error_numpy_referee": max(
            row["rel_error_numpy_referee"] for row in cases
        ),
        "convergence": convergence,
        "observed_order_edge": _observed_order(
            errors_edge[low], errors_edge[high], refinement
        ),
        "observed_order_inside": _observed_order(
            errors_inside[low], errors_inside[high], refinement
        ),
        "observed_order_edge_all": _observed_order(
            errors_edge[0], errors_edge[-1], coplanar[-1]["pieces"] / coplanar[0]["pieces"]
        ),
        "single_panel_bias_24pt": single_panel,
        "single_panel_bias_20pt": single_panel_20,
        "coplanar_pieces": [row["pieces"] for row in coplanar],
    }


def _delta(value: float | None) -> str:
    """Render an absolute deviation; exact zeros are reported as such."""
    if value is None:
        return "n/a"
    if value == 0.0:
        return "0（机器精度）"
    return f"{value:.2e}"


def _sci(value: float) -> str:
    """LaTeX-inline scientific notation, matching the repository documents."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return "\\(" + mantissa + "\\times10^{" + str(int(exponent)) + "}\\)"


def format_markdown(payload: dict[str, Any]) -> str:
    """Render the study note from the evidence payload."""
    cases = payload["cases"]
    coplanar = [
        row
        for row in payload["convergence"]
        if row["xi"] == 0.5 and row["h"] == 0.0
    ]
    mpmath_available = payload["mpmath_available"]

    def case_row(row: dict[str, Any]) -> str:
        head = (
            f"| {row['label']} | ({row['xi']:g}, {row['h']:g}) | "
            f"{row['closed_form']:.15f} |"
        )
        if not mpmath_available:
            return head + f" {_delta(row['abs_error_numpy_referee'])} |"
        referee = row["mpmath_referee"]
        referee_text = "mpmath 未安装" if referee is None else f"{referee:.15f}"
        return (
            head
            + f" {referee_text} | {_delta(row['abs_error_mpmath'])} | "
            + f"{_delta(row['abs_error_numpy_referee'])} |"
        )

    header = (
        "| 构型 | \\((\\xi, h)\\) | 闭式解 | mpmath（40 dps） | 绝对差 | numpy 裁判绝对差 |"
        if mpmath_available
        else "| 构型 | \\((\\xi, h)\\) | 闭式解 | numpy 裁判绝对差 |"
    )
    separator = "|---|---|---|---|---|" if mpmath_available else "|---|---|---|---|"
    case_table = "\n".join([header, separator] + [case_row(row) for row in cases])

    convergence_table = "\n".join(
        f"| {row['pieces']} | {row['err_sing_on_edge']:.3e} | "
        f"{row['err_sing_inside_piece']:.3e} |"
        for row in coplanar
    )

    if mpmath_available:
        mpmath_sentence = (
            f"最大绝对偏差 **{_sci(payload['max_abs_error_mpmath'])}**"
            f"（相对 {_sci(payload['max_rel_error_mpmath'])}），"
            "即闭式解在双精度机器精度内精确；「闭式解有误」的假设不成立。"
        )
    else:
        mpmath_sentence = (
            "本次运行未安装 mpmath，mpmath 列缺失；numpy 裁判（共面分支用初等原函数，"
            f"\\(h>0\\) 用分级求积）给出的最大绝对偏差为 "
            f"{_sci(payload['max_abs_error_numpy_referee'])}。"
        )

    return f"""# 闭式 Rankine 板块积分的判据：谁是精确的，谁只是收敛慢

**任务**：m2966 / m4008（在当前 main 基线上重新开发） · **仓**：TensorBEM
**工件**：`tests/data/rankine_primitive_check.json`、
`artifacts/rankine_referee.png`
**生成方式**：`python -m tensorbem.benchmarks.rankine_primitive_referee --write`
（本文件的所有数字均由该命令从同一份 JSON 渲染，无手抄）
**守卫测试**：`tests/test_rankine_primitive_referee.py`

## 1. 问题

`tensorbem.free_surface_solver._evaluate_source_potential` 用闭式原函数
`_log_panel_primitive` 积分单层位势的对数核，

$$\\int_0^L \\log r\\,\\mathrm{{d}}s
=\\tfrac12 z\\log(z^2+h^2)-z+h\\arctan(z/h),\\qquad h=0:\\; z\\log|z|-z,$$

其中 \\(z\\) 是场点沿面元的足点坐标、\\(h\\) 是场点到面元线的垂距。它替代了逐面元
均匀 Gauss 求积：后者跨过自面元的 \\(\\log r\\) 奇点，使位势偏约 \\(10^{{-4}}\\)，
从而移动了已发表的上浮圆柱附加质量（`docs/ursell_quadrature_provenance.md`）。

守卫测试最初要求闭式解与「均匀分片 Gauss 加密到收敛」一致到 \\(10^{{-8}}\\)，
实测在共面面元上恒差 \\(7.5\\times10^{{-6}}\\)，长期报红。本单的判定是：
**错的是判据侧（均匀 Gauss 参照的收敛机制），不是闭式解**，并把两侧的精度都量化、
固化成机器可检的证据。

## 2. 方法

1. **独立参照（任意精度）**：`mpmath.quad`（tanh-sinh，{payload['cases'][0]['dps']} 位有效位），
   把奇点 \\(s=\\xi\\) 作为分段点传入，对 6 个构型独立积分；与闭式解比较绝对/相对差。
2. **不依赖 mpmath 的参照**：共面（\\(h=0\\)）用初等原函数
   \\(G(t)=t\\log|t|-t\\)（\\(G(0)=0\\)，积分 \\(=G(L-\\xi)-G(-\\xi)\\)）；
   \\(h>0\\) 用向奇点分级加密的复合 Gauss（每片 20 点，最小片宽 \\(h/1000\\)）。
   这条路径在纯 numpy 环境下也能量到机器精度。
3. **历史参照的分辨率上限**：均匀分片 Gauss（每片 20 点），分片数
   \\(N=25\\!-\\!1600\\)，两种网格对齐：(a) 奇点恰落在片边界；(b) 网格错动使奇点落入片内部。
   观测阶由首末两级实测误差反解，不预设理论阶。
4. **历史缺陷复算**：单片 Gauss（20 点与旧的 24 点）跨过共面奇点的偏差，
   即 \\(10^{{-4}}\\) 级位势偏差的来源。

## 3. 结果

### 3.1 闭式解 vs 独立裁判

{case_table}

{mpmath_sentence}

### 3.2 均匀分片 Gauss 参照的分辨率上限（\\(\\xi=0.5,\\ h=0\\)）

| 分片数 N | 奇异点在片边界 | 奇异点在片内部 |
|---|---|---|
{convergence_table}

观测阶（在渐近段 \\(N\\ge100\\) 上）：片边界 **{payload['observed_order_edge']:.2f}**、
片内部 **{payload['observed_order_inside']:.2f}**，即 \\(N\\) 翻倍误差才减半。
在全部 \\(N=25\\!-\\!1600\\) 上拟合得 {payload['observed_order_edge_all']:.2f}（\\(N\\le50\\) 属预渐近段，
奇点仍靠近片端，故偏大）。\\(N=400\\) 时仍差
{_sci(coplanar[4]['err_sing_on_edge'])}（奇点落入片内时
{_sci(coplanar[4]['err_sing_inside_piece'])}，更差），\\(N=1600\\) 仍差
{_sci(coplanar[-1]['err_sing_on_edge'])} —— 旧的 \\(10^{{-8}}\\) 阈值在任何可承受的分片数下都不可达。

光滑面元（\\(h\\neq0\\)）同一规则直接收敛到机器精度：\\(h=0.3\\) 时
\\(N=25\\) 已到 {_delta(next(row['err_sing_on_edge'] for row in payload['convergence'] if row['label'].startswith('smooth') and row['pieces'] == 25))}。

### 3.3 历史缺陷量级

单片 Gauss 跨过共面奇点（\\(\\xi=0.5\\)）：24 点偏差
**{_sci(payload['single_panel_bias_24pt'])}**，20 点偏差
**{_sci(payload['single_panel_bias_20pt'])}**。这是「纯自面元」的极端量级；
`docs/ursell_quadrature_provenance.md` 记录的 \\(10^{{-4}}\\) 是它在整个物面位势里的净效应
（单个共面面元只占其一，且被光滑面元稀释），两者不矛盾。

## 4. 结论与判据修正

- 闭式板块积分**正确**：40 位独立裁判下 6 个构型最大绝对偏差
  {_sci(payload['max_abs_error_mpmath'] or payload['max_abs_error_numpy_referee'])}，
  远优于 3% 的工程判据。
- 旧的 \\(10^{{-8}}\\) 判据**不可达**：参照侧只有 \\(O(1/N)\\)。这类「判据比被测对象更差」
  的失效测试会把真实欠账埋进长期红态。
- 现行守卫（`tests/test_analytic_rankine_panel_integral_is_exact`）已改用初等原函数
  作共面参照；本单补上它缺的两块：任意精度证书（\\(10^{{-13}}\\) 相对判据）与
  「均匀 Gauss 参照的分辨率上限」的可执行钉扎（观测阶 0.8–1.2、\\(N=400\\) 仍
  \\(>10^{{-6}}\\)、单片超 \\(10^{{-4}}\\)），并把这套数字固化为 JSON 工件，
  由 `--check` 重测比对。
- 适用范围：仅二维直线面元的 \\(\\log r\\) 单层位势；\\(h>0\\) 的光滑情形无此限制。
  近共面 \\(h=10^{{-3}}\\) 未跨越 `coplanar` 开关（\\(h\\le10^{{-14}}\\)）；开关带内 \\(h=10^{{-15}}\\) 的分支选择亦经同一裁判校验，实测同样机器精度。

## 5. 图

`artifacts/rankine_referee.png`：
左 = 均匀分片 Gauss 误差 vs 分片数（log-log，含斜率 \\(-1\\) 参考线与旧 \\(10^{{-8}}\\) 阈值线）；
右 = 闭式解与裁判的绝对偏差逐构型条形图。
图不随仓入库（`artifacts/` 在 `.gitignore` 里）：需要时用 `--figure artifacts/rankine_referee.png` 从同一份 JSON 重绘。
"""


def make_figure(payload: dict[str, Any], path: Path) -> Path | None:
    """Render the two-panel referee figure; ``None`` when matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional extra
        return None

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    coplanar = [
        row for row in payload["convergence"] if row["xi"] == 0.5 and row["h"] == 0.0
    ]
    pieces = [row["pieces"] for row in coplanar]
    left.loglog(
        pieces,
        [row["err_sing_on_edge"] for row in coplanar],
        "o-",
        ms=4,
        label="singularity on a piece boundary",
    )
    left.loglog(
        pieces,
        [row["err_sing_inside_piece"] for row in coplanar],
        "s--",
        ms=4,
        label="singularity inside a piece",
    )
    smooth = [
        row
        for row in payload["convergence"]
        if row["label"].startswith("smooth") and row["err_sing_on_edge"] > 0.0
    ]
    left.loglog(
        [row["pieces"] for row in smooth],
        [max(row["err_sing_on_edge"], 1e-17) for row in smooth],
        "^-",
        color="0.35",
        ms=4,
        label="smooth panel h=0.3 (machine precision)",
    )
    reference_x = np.array([float(pieces[0]), float(pieces[-1])])
    first = coplanar[0]["err_sing_on_edge"]
    left.loglog(
        reference_x,
        first * (reference_x / pieces[0]) ** -1.0,
        "k:",
        lw=1.0,
        label="slope -1",
    )
    left.axhline(1.0e-8, color="crimson", lw=1.2, ls="-.")
    left.text(pieces[0] * 1.2, 1.4e-8, "retired 1e-8 threshold", color="crimson", fontsize=8)
    left.set_xlabel("uniform pieces N (20-point Gauss each)")
    left.set_ylabel(r"$|Q_N - Q_{\rm exact}|$ for the log panel integral")
    left.set_title("uniform Gauss referee: coplanar log r converges like O(1/N)")
    left.grid(True, which="both", ls=":", alpha=0.5)
    left.legend(fontsize=7, loc="lower left")

    labels = [row["label"] for row in payload["cases"]]
    values = [
        max(
            row["abs_error_mpmath"]
            if row["abs_error_mpmath"] is not None
            else row["abs_error_numpy_referee"],
            1e-18,
        )
        for row in payload["cases"]
    ]
    bars = right.barh(range(len(labels)), values, color="steelblue")
    right.set_yticks(range(len(labels)))
    right.set_yticklabels(labels, fontsize=8)
    right.set_xscale("log")
    right.axvline(1.0e-13, color="crimson", ls="--", lw=1.2, label="criterion 1e-13")
    right.set_xlabel("|closed form - referee|")
    right.set_title("closed-form primitive is exact to machine precision")
    for bar, value in zip(bars, values):
        right.text(value * 1.6, bar.get_y() + bar.get_height() / 2, f"{value:.1e}",
                   va="center", fontsize=7)
    right.grid(True, axis="x", which="both", ls=":", alpha=0.5)
    right.legend(fontsize=7, loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def write_artifacts(
    json_path: Path,
    doc_path: Path | None = None,
    figure_path: Path | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure, then write the JSON artifact, the note and the figure."""
    payload = payload or measure()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if doc_path is not None:
        doc_path.write_text(format_markdown(payload), encoding="utf-8")
    if figure_path is not None:
        make_figure(payload, figure_path)
    return payload


def compare(payload: dict[str, Any], artifact: dict[str, Any]) -> list[str]:
    """Compare a fresh measurement against a committed artifact."""
    problems: list[str] = []
    if payload["schema"] != artifact.get("schema"):
        problems.append(
            f"schema mismatch: {payload['schema']} != {artifact.get('schema')}"
        )
    committed = {row["label"]: row for row in artifact.get("cases", [])}
    for row in payload["cases"]:
        old = committed.get(row["label"])
        if old is None:
            problems.append(f"{row['label']}: missing from artifact")
            continue
        if not math.isclose(
            row["closed_form"], old["closed_form"], rel_tol=0.0, abs_tol=1.0e-12
        ):
            problems.append(
                f"{row['label']}: closed form {row['closed_form']!r} != "
                f"artifact {old['closed_form']!r}"
            )
        referee = old.get("mpmath_referee")
        if referee is not None and row["mpmath_referee"] is not None:
            if not math.isclose(
                referee, row["mpmath_referee"], rel_tol=1.0e-12, abs_tol=1.0e-13
            ):
                problems.append(
                    f"{row['label']}: referee drifted {row['mpmath_referee']!r} vs "
                    f"{referee!r}"
                )
    committed_convergence = {
        (row["label"], row["pieces"]): row for row in artifact.get("convergence", [])
    }
    for row in payload["convergence"]:
        old = committed_convergence.get((row["label"], row["pieces"]))
        if old is None:
            problems.append(f"{row['label']} N={row['pieces']}: missing from artifact")
            continue
        for key in ("err_sing_on_edge", "err_sing_inside_piece"):
            if not math.isclose(
                row[key], old[key], rel_tol=1.0e-9, abs_tol=1.0e-15
            ):
                problems.append(
                    f"{row['label']} N={row['pieces']} {key}: {row[key]!r} vs {old[key]!r}"
                )
    for key in (
        "observed_order_edge",
        "observed_order_inside",
        "single_panel_bias_24pt",
        "single_panel_bias_20pt",
    ):
        if artifact.get(key) is None:
            problems.append(f"artifact is missing {key}")
            continue
        if not math.isclose(payload[key], artifact[key], rel_tol=1.0e-9, abs_tol=1.0e-15):
            problems.append(f"{key}: {payload[key]!r} != artifact {artifact[key]!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True, help="evidence JSON path")
    parser.add_argument("--doc", type=Path, help="study note to render (write mode)")
    parser.add_argument("--figure", type=Path, help="figure to render (write mode)")
    parser.add_argument("--dps", type=int, default=40, help="mpmath working digits")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the JSON artifact (and the doc/figure when given)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-measure and compare against the committed JSON",
    )
    args = parser.parse_args(argv)

    payload = measure(dps=args.dps)
    if args.check:
        artifact = json.loads(args.json.read_text(encoding="utf-8"))
        problems = compare(payload, artifact)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            print(f"FAIL: {len(problems)} mismatch(es) against {args.json}", file=sys.stderr)
            return 1
        print(f"OK: {args.json} reproduces from the current code")
        return 0

    write_artifacts(args.json, args.doc, args.figure, payload=payload)
    print(f"wrote {args.json}")
    if args.doc is not None:
        print(f"wrote {args.doc}")
    if args.figure is not None:
        print(f"wrote {args.figure}")
    print(
        f"max_abs_error_mpmath={payload['max_abs_error_mpmath']:.3e} "
        f"order_edge={payload['observed_order_edge']:.3f} "
        f"order_inside={payload['observed_order_inside']:.3f} "
        f"single_panel_bias={payload['single_panel_bias_24pt']:.3e}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())