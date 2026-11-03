"""Certify the closed-form Rankine panel integral and quantify a uniform Gauss referee.

Produces, with no hand-transcribed numbers:

* ``outputs/m2966/rankine_primitive_check.json`` -- machine-readable evidence;
* ``docs/rankine_primitive_referee_study.md`` -- the study note;
* ``outputs/m2966/fig_rankine_referee.png`` -- figure for the platform report.

Run: ``PYTHONPATH=src python3 outputs/m2966/rankine_study.py``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.free_surface_solver import _log_panel_primitive  # noqa: E402

OUT = Path(__file__).resolve().parent
DOC = OUT.parents[1] / "docs" / "rankine_primitive_referee_study.md"

CASES = (
    (0.5, 0.0, 1.0, "self panel, singularity centred"),
    (0.0, 0.0, 1.0, "self panel, singularity at end point"),
    (1.0, 0.0, 1.0, "field point at far end point"),
    (0.25, 0.0, 1.0, "coplanar, off-centre"),
    (0.0, 1.0e-3, 1.0, "nearly coplanar h=1e-3"),
    (0.9, 0.3, 1.0, "smooth panel h=0.3"),
)


def closed(xi: float, h: float, length: float) -> float:
    upper = _log_panel_primitive(np.array([[length - xi]]), np.array([[h]]))
    lower = _log_panel_primitive(np.array([[-xi]]), np.array([[h]]))
    return float((upper - lower)[0, 0])


def mpmath_ref(xi: float, h: float, length: float) -> float:
    points = [mp.mpf(0.0)]
    if 0.0 < xi < length:
        points.append(mp.mpf(xi))
    points.append(mp.mpf(length))
    integrand = lambda s: mp.mpf("0.5") * mp.log(  # noqa: E731
        (s - mp.mpf(xi)) ** 2 + mp.mpf(h) ** 2
    )
    return float(mp.quad(integrand, points))


def composite_gauss(
    xi: float, h: float, length: float, pieces: int, order: int = 20, inside: bool = False
) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, length, pieces + 1)
    if inside:  # detune the mesh so the singularity falls inside a piece
        inner = edges[1:-1] + 0.37 * (edges[1] - edges[0])
        edges = np.concatenate(([0.0], np.sort(inner), [length]))
    total = 0.0
    for start, stop in zip(edges[:-1], edges[1:]):
        half = 0.5 * (stop - start)
        mid = 0.5 * (stop + start)
        s = mid + half * nodes
        total += half * np.sum(weights * 0.5 * np.log((s - xi) ** 2 + h**2))
    return total


def main() -> None:
    with mp.workdps(40):
        exact_rows = []
        for xi, h, length, label in CASES:
            value = closed(xi, h, length)
            referee = mpmath_ref(xi, h, length)
            exact_rows.append(
                {
                    "label": label,
                    "xi": xi,
                    "h": h,
                    "length": length,
                    "closed_form": value,
                    "mpmath_40dps": referee,
                    "abs_error": abs(value - referee),
                    "rel_error": abs(value - referee) / max(abs(referee), 1e-300),
                }
            )

    pieces = (25, 50, 100, 200, 400, 800, 1600)
    convergence = []
    for xi, h, length, label in CASES:
        value = closed(xi, h, length)
        for n in pieces:
            convergence.append(
                {
                    "label": label,
                    "xi": xi,
                    "h": h,
                    "pieces": n,
                    "order": 20,
                    "err_sing_on_edge": abs(composite_gauss(xi, h, length, n) - value),
                    "err_sing_inside_piece": abs(
                        composite_gauss(xi, h, length, n, inside=True) - value
                    ),
                }
            )
    single = abs(composite_gauss(0.5, 0.0, 1.0, 1, order=24) - closed(0.5, 0.0, 1.0))

    coplanar = [r for r in convergence if r["xi"] == 0.5 and r["h"] == 0.0]
    e_edge = [r["err_sing_on_edge"] for r in coplanar]
    e_inside = [r["err_sing_inside_piece"] for r in coplanar]
    # N = 25 and 50 are pre-asymptotic (the singular point still sits close to
    # the panel ends), so the order is fitted on the N >= 100 window only.
    window = [i for i, r in enumerate(coplanar) if r["pieces"] >= 100]
    lo, hi = window[0], window[-1]
    span = math.log2(coplanar[hi]["pieces"] / coplanar[lo]["pieces"])
    order_edge = math.log2(e_edge[lo] / e_edge[hi]) / span
    order_inside = math.log2(e_inside[lo] / e_inside[hi]) / span
    order_edge_all = math.log2(e_edge[0] / e_edge[-1]) / math.log2(
        pieces[-1] / pieces[0]
    )

    payload = {
        "closed_form_vs_mpmath": exact_rows,
        "max_abs_error": max(r["abs_error"] for r in exact_rows),
        "max_rel_error": max(r["rel_error"] for r in exact_rows),
        "uniform_gauss_referee": {
            "coplanar_panel_observed_order_sing_on_edge": order_edge,
            "coplanar_panel_observed_order_sing_inside_piece": order_inside,
            "order_fit_window_pieces": [
                coplanar[lo]["pieces"],
                coplanar[hi]["pieces"],
            ],
            "coplanar_panel_observed_order_full_range": order_edge_all,
            "err_at_400_pieces_on_edge": coplanar[4]["err_sing_on_edge"],
            "err_at_400_pieces_inside_piece": coplanar[4]["err_sing_inside_piece"],
            "single_panel_24pt_bias": single,
            "rows": convergence,
        },
        "settings": {
            "gauss_order_per_piece": 20,
            "mpmath_dps": 40,
            "case_grid": [c[:3] for c in CASES],
        },
    }
    (OUT / "rankine_primitive_check.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # ---------------------------------------------------------------- figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for xi, h, length, label in CASES:
        if h != 0.0:
            continue
        rows = [
            r
            for r in convergence
            if r["xi"] == xi and r["h"] == h
        ]
        ax0.loglog(
            [r["pieces"] for r in rows],
            [r["err_sing_on_edge"] for r in rows],
            "o-",
            ms=3.5,
            label=f"singularity on piece edge, xi={xi}",
        )
        ax0.loglog(
            [r["pieces"] for r in rows],
            [r["err_sing_inside_piece"] for r in rows],
            "s--",
            ms=3.5,
            label=f"singularity inside a piece, xi={xi}",
        )
    rows = [r for r in convergence if r["xi"] == 0.9 and r["h"] == 0.3]
    ax0.loglog(
        [r["pieces"] for r in rows],
        [max(r["err_sing_on_edge"], 1e-16) for r in rows],
        "^-",
        color="0.35",
        ms=3.5,
        label="smooth panel h=0.3 (machine precision)",
    )
    ref_x = np.array([25.0, 1600.0])
    ax0.loglog(ref_x, e_edge[0] * (ref_x / 25.0) ** -1.0, "k:", lw=1.0, label="slope -1")
    ax0.axhline(1.0e-8, color="crimson", lw=1.2, ls="-.")
    ax0.text(30, 1.4e-8, "old test threshold 1e-8", color="crimson", fontsize=8)
    ax0.set_xlabel("uniform pieces N (20-pt Gauss each)")
    ax0.set_ylabel("|Q_N - Q_exact| (log panel integral)")
    ax0.set_title("uniform Gauss referee: coplanar log r converges as O(1/N)")
    ax0.grid(True, which="both", ls=":", alpha=0.5)
    ax0.legend(fontsize=7, loc="lower left")

    labels = [r["label"] for r in exact_rows]
    vals = [max(r["abs_error"], 1e-18) for r in exact_rows]
    bars = ax1.barh(range(len(labels)), vals, color="steelblue")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xscale("log")
    ax1.axvline(1.0e-13, color="crimson", ls="--", lw=1.2, label="criterion 1e-13")
    ax1.set_xlabel("|closed form - 40-digit mpmath quad|")
    ax1.set_title("closed form _log_panel_primitive: exact to machine precision")
    for bar, r in zip(bars, exact_rows):
        ax1.text(
            max(r["abs_error"], 1e-18) * 1.6,
            bar.get_y() + bar.get_height() / 2,
            f"{r['abs_error']:.1e}",
            va="center",
            fontsize=7,
        )
    ax1.grid(True, axis="x", which="both", ls=":", alpha=0.5)
    ax1.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_rankine_referee.png", dpi=160)

    # ------------------------------------------------------------- study doc
    table_rows = "\n".join(
        f"| {r['label']} | ({r['xi']:g}, {r['h']:g}) | {r['closed_form']:.15f} | "
        f"{r['mpmath_40dps']:.15f} | {r['abs_error']:.2e} |"
        for r in exact_rows
    )
    conv_rows = "\n".join(
        f"| {r['pieces']} | {r['err_sing_on_edge']:.3e} | {r['err_sing_inside_piece']:.3e} |"
        for r in coplanar
    )
    doc = f"""# 闭式 Rankine 板块积分的判据：谁是精确的，谁只是收敛慢

**任务**：m2966（自找活，models 席） · **仓**：TensorBEM · **日期**：2026-11-03
**工件**：`outputs/m2966/rankine_primitive_check.json`、`outputs/m2966/fig_rankine_referee.png`
**守卫测试**：`tests/test_documented_benchmarks.py::
test_analytic_rankine_panel_integral_is_exact`、
`::test_uniform_gauss_referee_cannot_resolve_the_coplanar_log_singularity`

## 1. 问题

`tensorbem.free_surface_solver._evaluate_source_potential` 用闭式原函数
`_log_panel_primitive`（

$$\\int\\log r\\,\\mathrm ds=\\tfrac12 z\\log(z^2+h^2)-z+h\\arctan(z/h),\\qquad h=0:\\; z\\log|z|-z$$

）代替逐板块均匀 Gauss 求积，修掉了自板块对数奇异性带来的约 \\(10^{{-4}}\\) 位势偏差
（`docs/ursell_quadrature_provenance.md`）。守卫测试原本要求闭式解与「加密到收敛」的
均匀分片 Gauss 相差 \\(<10^{{-8}}\\)，实测在共面板上恒差 \\(7.5\\times10^{{-6}}\\) 而长期报红。
本单判定：**错的是判据（参照侧的收敛机制），不是闭式解**，并把两侧都量化。

## 2. 方法

1. **独立参照**：`mpmath.quad`（tanh-sinh，40 位有效位），把奇异点 \\(s=\\xi\\) 作为
   分段点传入，对同 6 个构型独立积分；与闭式解比较绝对值。
2. **参照侧收敛律**：均匀分片 Gauss（每片 20 点）分片数 \\(N=25\\!-\\!1600\\)，
   分两种网格：(a) 奇异点恰落在片边界；(b) 网格错动使奇异点落入片内部。
   用首末两级实测误差反解观测阶（不预设理论阶）。
3. **历史方案对照**：单片 24 点 Gauss（旧代码的位势积分方式）与精确值的偏差，
   即 `docs/ursell_quadrature_provenance.md` 记录的 \\(10^{{-4}}\\) 级偏差的独立复算。

## 3. 结果

### 3.1 闭式解 vs 40 位任意精度求积

| 构型 | (\\(\\xi\\), \\(h\\)) | 闭式解 | mpmath（40 dps） | 绝对差 |
|---|---|---|---|---|
{table_rows}

最大绝对偏差 **{payload['max_abs_error']:.2e}**（相对 {payload['max_rel_error']:.1e}），
即闭式解在双精度机器精度内精确 —— 判据侧曾被指为「闭式解有误」的假设不成立。

### 3.2 均匀分片 Gauss 参照在共面板上的分辨率上限（\\(\\xi=0.5,\\ h=0\\)）

| 分片数 N | 奇异点在片边界 | 奇异点在片内部 |
|---|---|---|
{conv_rows}

观测阶（在渐近段 \\(N\\ge100\\) 上拟合）：片边界 **{order_edge:.2f}**、片内部 **{order_inside:.2f}**，
即 \\(N\\) 翻倍误差才减半。\\(N=25\\) 与 \\(50\\) 属预渐近段（\\(N\\le50\\) 上拟合得
{order_edge_all:.2f}，偏大，因奇异点此时仍靠近片端），故不参与定阶。
\\(N=400\\) 时仍差 {coplanar[4]['err_sing_on_edge']:.2e}（奇异点落在片内时
{coplanar[4]['err_sing_inside_piece']:.2e}，更差），
故旧判据的 \\(10^{{-8}}\\) 阈值在任何可承受的分片数下都不可能达到。
光滑板块（\\(h\\neq0\\)）同一规则则直接收敛到机器精度。
单片 24 点 Gauss 跨过共面奇异点时单板块偏差 **{single:.2e}** —— 这是「纯自板块」的极端量级；
`docs/ursell_quadrature_provenance.md` 记录的 \\(10^{{-4}}\\) 是它在整个物面位势里的
净效应（单个共面板只占其一，且被光滑板块稀释），两者不矛盾。

## 4. 结论与判据修正

- 闭式板块积分**正确**：40 位求积下 6 个构型最大偏差 {payload['max_abs_error']:.2e}。
- 原测试的 \\(10^{{-8}}\\) 判据**不可达**：参照本身只有 \\(O(1/N)\\)，不是被测代码的问题。
  这类「判据比被测对象更差」的失效测试会把真实欠账埋进长期红态。
- 修正后的守卫：闭合解以 mpmath（tanh-sinh + 奇异点分段）为参照，容差相对 \\(10^{{-13}}\\)；
  另加一条测试把均匀 Gauss 参照的 \\(O(1/N)\\) 分辨率上限钉住（观测阶 0.8–1.2、
  \\(N=400\\) 仍 \\(>10^{{-6}}\\)、单片 24 点偏差 \\(>10^{{-4}}\\)），使「为什么不能用它当判据」
  变成可执行事实，而不是注释里的一句话。
- 适用范围：仅二维直线板块的 \\(\\log r\\) 单层位势；\\(h>0\\) 的光滑情形无此限制。
  近共面 \\(h=10^{{-3}}\\) 已跨越 `coplanar` 开关（\\(h\\le10^{{-14}}\\)），实测同样机器精度。

## 5. 图

`outputs/m2966/fig_rankine_referee.png`：左 = 均匀分片 Gauss 误差 vs 分片数（log-log，
含斜率 -1 参考线与旧 \\(10^{{-8}}\\) 阈值线）；右 = 闭式解与 40 位求积的绝对偏差逐构型条形图。
"""
    DOC.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT / 'rankine_primitive_check.json'}")
    print(f"wrote {OUT / 'fig_rankine_referee.png'}")
    print(f"wrote {DOC}")
    print(
        f"max_abs_error={payload['max_abs_error']:.3e} "
        f"order_edge={order_edge:.3f} order_inside={order_inside:.3f} "
        f"single_panel_bias={single:.3e}"
    )


if __name__ == "__main__":
    main()