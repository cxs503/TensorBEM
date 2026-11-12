"""m16156 -- evidence generator for the ``FMMOperator`` far/near diagnostic.

Closes the open contradiction left by m16000: on the geometry whose second-call
error was 2.46 (246 %), ``far_fraction`` was reported as 0.0.  Root cause: the
probe operator had never been ``setup``, and the pre-fix API answered from an
empty octree (``matvec`` -> zeros, ``far_fraction`` -> 0.0) instead of raising.

This script measures, on the identical geometry:

* the pre-fix fail-silent answers (an operator that is not set up, class
  ``_UnsetProbe`` keeps the old behaviour so the A/B has one varying input),
* the post-fix answers / exceptions,
* the pair-weighted and source-weighted far fractions,
* the invariant "far_fraction == 0  =>  matvec == dense direct sum",
* the multipole share over icosphere refinements (the cost-audit surface).

Every number quoted in the report is written to ``evidence.json``; the figure is
built from that file, nothing is typed by hand.

Run::

    PYTHONPATH=src python scripts/fmm_far_fraction_diagnostic.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from tensorbem import geometry3d
from tensorbem.fmm import FMMOperator, direct_laplace_matvec

OUT = pathlib.Path("docs/fmm_far_fraction_diagnostic")


class _UnsetProbe(FMMOperator):
    """Pre-m16156 fail-silent semantics (no setup guard, empty tree answers)."""

    def _require_setup(self, caller: str) -> None:  # noqa: D102
        return None

    def far_fraction(self, targets):  # noqa: D102
        if not self.leaves:
            return 0.0
        return super().far_fraction(targets)

    def matvec(self, targets, sources, charges):  # noqa: D102
        if not self.leaves:
            return np.zeros(len(np.asarray(targets, dtype=float)))
        return super().matvec(targets, sources, charges)


def _separated(seed: int = 42):
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(200, 3)) * 0.5
    targets = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    charges = rng.normal(size=200)
    return sources, targets, charges


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def unset_probe() -> dict:
    sources, targets, charges = _separated()
    pre = {"far_fraction": float(_UnsetProbe(p=4, max_leaf=25).far_fraction(targets)),
           "matvec_norm": float(np.linalg.norm(_UnsetProbe(p=4, max_leaf=25).matvec(
               targets, sources, charges)))}

    def _post(call):
        try:
            call()
        except RuntimeError as exc:
            return type(exc).__name__ + ": " + str(exc)[:120]
        return "no error"

    op = FMMOperator(p=4, max_leaf=25)
    post = {
        "far_fraction": _post(lambda: op.far_fraction(targets)),
        "matvec": _post(lambda: op.matvec(targets, sources, charges)),
    }

    op.setup(sources)
    after = {
        "far_fraction": float(op.far_fraction(targets)),
        "far_work_fraction": float(op.far_work_fraction(targets)),
        "matvec_rel_err_vs_dense": _rel(op.matvec(targets, sources, charges),
                                        direct_laplace_matvec(targets, sources, charges)),
        "leaves": len(op.leaves),
        "leaf_sizes": [int(len(i)) for i in op._leaf_indices],
    }
    return {"pre_fix": pre, "post_fix_refuses": post, "after_setup": after}


def zero_far_invariant() -> dict:
    rng = np.random.default_rng(7)
    sources = rng.normal(size=(40, 3)) * 0.3
    charges = rng.normal(size=40)
    targets = rng.normal(size=(11, 3))
    op = FMMOperator(p=4, max_leaf=len(sources))
    op.setup(sources)
    phi = op.matvec(targets, sources, charges)
    direct = direct_laplace_matvec(targets, sources, charges)
    return {"far_fraction": float(op.far_fraction(targets)),
            "rel_err_vs_dense": _rel(phi, direct),
            "bitwise_equal": bool(np.array_equal(phi, direct))}


def multipole_share_ladder(levels=(1, 2, 3, 4)) -> list:
    rows = []
    for level in levels:
        mesh = geometry3d.icosphere(subdivisions=level, radius=1.0)
        centroids = np.asarray(mesh.centroids, dtype=float)
        charges = np.asarray(mesh.areas, dtype=float)
        op = FMMOperator(p=4, max_leaf=32)
        op.setup(centroids, charges)
        rows.append({
            "level": level,
            "panels": int(len(centroids)),
            "leaves": len(op.leaves),
            "far_fraction": float(op.far_fraction(centroids)),
            "far_work_fraction": float(op.far_work_fraction(centroids)),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {
        "mission": "m16156",
        "what": "FMMOperator far/near diagnostic: fail-silent un-set-up operator "
                "(m16000 limitation 2) and far_fraction accounting definition",
        "unset_probe": unset_probe(),
        "zero_far_invariant": zero_far_invariant(),
        "multipole_share": multipole_share_ladder(),
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8))
    up = evidence["unset_probe"]
    ax0.bar(["pre-fix\n(no setup)", "post-fix\nafter setup"],
            [up["pre_fix"]["far_fraction"], up["after_setup"]["far_fraction"]],
            color=["#c44", "#2a7"], width=0.5)
    ax0.axhline(float(4 / 23), ls="--", c="k", lw=1)
    ax0.text(1.35, float(4 / 23) + 0.01, "4/23 pair fraction", fontsize=8)
    ax0.set_ylabel("far_fraction on the m16000 geometry")
    ax0.set_title("fail-silent 0.0 vs measured far share")
    ax0.set_ylim(0, 0.3)

    rows = evidence["multipole_share"]
    n = [r["panels"] for r in rows]
    ax1.loglog(n, [r["far_fraction"] for r in rows], "o-", label="pair fraction")
    ax1.loglog(n, [r["far_work_fraction"] for r in rows], "s--", label="source-weighted work")
    ax1.set_xlabel("panels N (icosphere)")
    ax1.set_ylabel("multipole-evaluated share")
    ax1.set_title("far share vs N (p=4, max_leaf=32)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_far_fraction.png", dpi=150)
    (OUT / "tables.md").write_text(_render_tables(evidence))


def _render_tables(evidence: dict) -> str:
    """Render every number in the report -- nothing is transcribed by hand."""
    up = evidence["unset_probe"]
    zf = evidence["zero_far_invariant"]
    lines = [
        "<!-- generated by scripts/fmm_far_fraction_diagnostic.py -->",
        "",
        "**表 1 · 未 `setup` 算子的失败模式（同一几何：200 源簇 + 3 远靶，p=4,"
        " max_leaf=25）**",
        "",
        "| 调用 | 修复前（fail-silent） | 修复后 |",
        "|---|---|---|",
        f"| `far_fraction(targets)` | {up['pre_fix']['far_fraction']:.1f} | "
        f"{up['post_fix_refuses']['far_fraction'].split(':')[0]} |",
        f"| `matvec(...)` 范数 | {up['pre_fix']['matvec_norm']:.1f}（全零静默返回） | "
        f"{up['post_fix_refuses']['matvec'].split(':')[0]} |",
        "",
        "**表 2 · `setup` 之后的真实远近场份额（同几何）**",
        "",
        "| 量 | 值 |",
        "|---|---|",
        f"| 叶数 / 叶容量 | {up['after_setup']['leaves']} / 叶内源数 {min(up['after_setup']['leaf_sizes'])}–"
        f"{max(up['after_setup']['leaf_sizes'])} |",
        f"| `far_fraction`（配对口径） | {up['after_setup']['far_fraction']:.6f}（= 4/23） |",
        f"| `far_work_fraction`（源数加权，代价口径） | {up['after_setup']['far_work_fraction']:.6f} |",
        f"| `matvec` 相对稠密直接求和误差 | {up['after_setup']['matvec_rel_err_vs_dense']:.3e} |",
        "",
        "**表 3 · 不变量：far_fraction = 0 ⇒ matvec 即稠密直接求和**"
        "（40 源单叶、11 靶、p=4）",
        "",
        "| 量 | 值 |",
        "|---|---|",
        f"| `far_fraction` | {zf['far_fraction']:.1f} |",
        f"| 相对稠密误差 | {zf['rel_err_vs_dense']:.3e} |",
        f"| 逐位相等 | {zf['bitwise_equal']}（仅求和顺序差） |",
        "",
        "**表 4 · 多极份额随面元数的变化（icosphere, p=4, max_leaf=32）**",
        "",
        "| level | panels | leaves | far_fraction | far_work_fraction |",
        "|---|---|---|---|---|",
    ]
    for row in evidence["multipole_share"]:
        lines.append(
            f"| {row['level']} | {row['panels']} | {row['leaves']} | "
            f"{row['far_fraction']:.6f} | {row['far_work_fraction']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()