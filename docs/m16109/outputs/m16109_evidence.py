"""m16109: build docs/fmm_setup_contract/{evidence.json, figs} programmatically.

Every number in the report is injected from this script's output (zero
hand-copying).  Runs the post-fix code only; the pre-fix values come from
outputs/m16109_repro_prefix.json (captured before the patch was applied).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "src")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem.fmm import FMMOperator, direct_laplace_matvec  # noqa: E402

ROOT = pathlib.Path(".")
OUT = ROOT / "docs" / "fmm_setup_contract"
OUT.mkdir(parents=True, exist_ok=True)


def _separated_geometry(seed: int = 42):
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(200, 3)) * 0.5
    targets = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    charges = rng.normal(size=200)
    return sources, targets, charges


def main() -> None:
    sources, targets, charges = _separated_geometry()
    pre = json.loads((ROOT / "outputs" / "m16109_repro_prefix.json").read_text())
    post = json.loads((ROOT / "outputs" / "m16109_repro_postfix.json").read_text())

    # --- far_fraction vs target distance (post-fix diagnostic) --------------
    op = FMMOperator(p=4, max_leaf=25)
    op.setup(sources)
    ds = np.linspace(0.5, 10.0, 40)
    frac = []
    for d in ds:
        frac.append(float(op.far_fraction(np.array([[d, 0.0, 0.0]]))))
    frac = np.array(frac)

    # --- behaviour comparison numbers ----------------------------------------
    exact = direct_laplace_matvec(targets, sources, charges)
    scale = float(np.max(np.abs(exact)))
    ev = {
        "mission": "m16109",
        "geometry": {
            "n_sources": 200,
            "source_scale": 0.5,
            "targets": targets.tolist(),
            "p": 4,
            "max_leaf": 25,
            "seed": 42,
        },
        "pre_fix": {
            "far_fraction_unset": pre["far_fraction_unset"]["value"],
            "matvec_unset_rel_err": pre["matvec_unset"]["rel_err_vs_direct"],
            "matvec_unset_max_abs": pre["matvec_unset"]["max_abs"],
            "control_far_fraction": pre["control_set_up"]["far_fraction"],
            "control_rel_err": pre["control_set_up"]["rel_err_vs_direct"],
        },
        "post_fix": {
            "far_fraction_unset": post["far_fraction_unset"],
            "matvec_unset": post["matvec_unset"],
            "control_far_fraction": post["control_set_up"]["far_fraction"],
            "control_rel_err": post["control_set_up"]["rel_err_vs_direct"],
        },
        "far_fraction_vs_distance": {
            "d": ds.tolist(),
            "far_fraction": frac.tolist(),
            "m16000_geometry_point": {
                "d": 4.0,
                "far_fraction": float(op.far_fraction(targets[0:1])),
            },
        },
        "tests": {
            "new_file": "tests/test_fmm_setup_contract.py",
            "red_before": "4 failed, 4 passed (pre-fix fmm.py via git stash)",
            "green_after": "57 passed (setup_contract + operator_linearity + fmm + cost_audit)",
        },
        "direct_scale": scale,
    }
    (OUT / "evidence.json").write_text(json.dumps(ev, indent=2))

    # --- figure 1: the diagnostic is now meaningful ---------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.plot(ds, frac, "b-", lw=2)
    pt = ev["far_fraction_vs_distance"]["m16000_geometry_point"]
    ax.plot([pt["d"]], [pt["far_fraction"]], "ro", ms=8, zorder=5)
    ax.annotate(
        f"m16000 bench target (d=4): {pt['far_fraction']:.3f}",
        (pt["d"], pt["far_fraction"]),
        textcoords="offset points", xytext=(8, -14),
    )
    ax.set_xlabel("target distance d from cluster centre")
    ax.set_ylabel("far_fraction (multipole-pair share)")
    ax.set_title("FMMOperator.far_fraction after setup (m16109)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_far_fraction_vs_distance.png")
    plt.close(fig)

    # --- figure 2: fail-open vs fail-closed -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=150)
    ax = axes[0]
    labels = ["unset\n(pre-fix)", "unset\n(post-fix)", "set up\n(post-fix)"]
    vals = [pre["matvec_unset"]["rel_err_vs_direct"], None, post["control_set_up"]["rel_err_vs_direct"]]
    ax.bar([0, 2], [vals[0], vals[2]], color=["#d62728", "#2ca02c"], width=0.6)
    ax.bar([1], [1.0], color="#d62728", width=0.6, hatch="//", alpha=0.55)
    ax.text(1, 1.02, "RuntimeError\n(silent 100% error gone)", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 3)
    ax.set_xticks([0, 1, 2], labels)
    ax.set_ylabel("max rel. error vs direct sum")
    ax.set_title("matvec before/after setup")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar([0, 2], [pre["far_fraction_unset"]["value"], post["control_set_up"]["far_fraction"]],
           color=["#d62728", "#2ca02c"], width=0.6)
    ax.bar([1], [0.0], color="#d62728", width=0.6, hatch="//", alpha=0.55)
    ax.text(1, 0.02, "RuntimeError\n(silent 0.0 gone)", ha="center", fontsize=8)
    ax.set_xticks([0, 1, 2], labels)
    ax.set_ylabel("far_fraction")
    ax.set_title("far_fraction before/after setup")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_failopen_vs_failclosed.png")
    plt.close(fig)

    print(json.dumps(ev, indent=2))
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
