"""Figures + numeric tables for the context-craft protocol (docs/context_craft_*).

    python -m tools.context_craft.make_figures --results docs/context_craft_ab/ab_results.json \
        --out-dir docs/context_craft_ab

Produces
    fig1_ab_bars.png      recall / gate / decision-share per arm at each budget
    fig2_budget_curves.png recall vs token budget, one curve per arm
    fig3_memory_trace.png three-layer memory under a synthetic 60-mission workload
    simulation.json       numbers behind fig3 + the tables quoted in the report
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..context_craft import memory as cc_mem  # noqa: E402
from ..context_craft.model import FactItem, MemoryEntry  # noqa: E402

ARM_LABEL = {
    "head_dump": "own-doc dump (baseline)",
    "bm25_raw": "raw lexical search",
    "brief_global": "brief, no locality",
    "brief_local": "brief + locality (protocol)",
}
ARM_COLOR = {
    "head_dump": "#9aa5b1",
    "bm25_raw": "#7c9", "brief_global": "#e0a458", "brief_local": "#3b7dd8",
}


def fig_bars(res: dict, out: str) -> None:
    budgets = list(res["per_budget"])
    arms = list(res["arms"])
    metrics = [("recall", "held-out decision-fact recall"),
               ("tol_recall", "acceptance-criterion recall"),
               ("gate", "gate hit-rate (>=1 criterion)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    width = 0.8 / len(arms)
    for ax, (metric, title) in zip(axes, metrics):
        for k, arm in enumerate(arms):
            xs, ys, es = [], [], []
            for j, b in enumerate(budgets):
                s = res["per_budget"][b]["summary"][arm][metric]
                xs.append(j + k * width - 0.4 + width / 2)
                ys.append(s["mean"])
                es.append(s["ci95"])
            ax.bar(xs, ys, width=width * 0.94, yerr=es, capsize=2.5,
                   label=ARM_LABEL.get(arm, arm), color=ARM_COLOR.get(arm, None),
                   error_kw={"lw": 0.8})
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f"{b} tok" for b in budgets])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, lw=0.6)
    axes[0].set_ylabel("mean over 40 held-out tasks (95% CI)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Task brief package vs bare-context baselines (offline A/B, TensorBEM corpus)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_curves(res: dict, out: str) -> None:
    budgets = [int(b) for b in res["per_budget"]]
    arms = list(res["arms"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm in arms:
        ys = [res["per_budget"][str(b)]["summary"][arm]["recall"]["mean"] for b in budgets]
        ax1.plot(budgets, ys, marker="o", label=ARM_LABEL.get(arm, arm),
                 color=ARM_COLOR.get(arm, None))
        ys2 = [res["per_budget"][str(b)]["summary"][arm]["decision_share"]["mean"] for b in budgets]
        ax2.plot(budgets, ys2, marker="s", label=ARM_LABEL.get(arm, arm),
                 color=ARM_COLOR.get(arm, None))
    ax1.set_xlabel("token budget"); ax1.set_ylabel("held-out decision-fact recall")
    ax1.set_title("recall vs budget", fontsize=10); ax1.grid(alpha=0.3, lw=0.6)
    ax1.set_ylim(0, 1.0)
    ax2.set_xlabel("token budget"); ax2.set_ylabel("decision-bearing line share")
    ax2.set_title("information density of the window", fontsize=10)
    ax2.grid(alpha=0.3, lw=0.6); ax2.set_ylim(0, 1.0)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


# ------------------------------------------------------- memory simulation
VOCAB = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
    "sigma tau upsilon phi chi psi omega bracket kernel mesh panel gauge vector scalar tensor "
    "solver route ledger criterion archive digest cadence budget threshold drift anchor"
).split()


def simulate_memory(missions: int = 60, seed: int = 7) -> dict:
    """Synthetic seat workload: facts arrive, some are reused, some rot."""
    rnd = random.Random(seed)
    kinds = ["TOLERANCE", "NEGATIVE", "TRAP", "VERDICT", "PIN", "CONTEXT", "CONTEXT", "CONTEXT"]
    entries: list[MemoryEntry] = []

    def new_entry(mission: int, idx: int) -> MemoryEntry:
        kind = rnd.choice(kinds)
        layer = "L2" if rnd.random() < 0.45 else "L3"
        # every note gets its own vocabulary so the near-duplicate filter does
        # not eat the synthetic workload (that would hide the decay behaviour)
        words = " ".join(rnd.choice(VOCAB) for _ in range(14))
        text = (f"synthetic {kind.lower()} note #{idx} from mission {mission}: {words}")
        return MemoryEntry(
            item=FactItem.build(kind, text, f"sim/{layer}.md", idx),
            layer=layer,
            use_count=0,
            last_used_mission=mission,
        )

    trace = []
    last_compress = 0
    total_forgotten = 0
    idx = 0
    for mission in range(1, missions + 1):
        for _ in range(rnd.randint(4, 9)):          # new notes per mission
            idx += 1
            entries.append(new_entry(mission, idx))
        for e in rnd.sample(entries, min(len(entries), 5)):   # reuse
            cc_mem.touch(entries, e.item.item_id, mission)
        out_dir = os.path.join("/tmp", "cc_mem_sim")
        res = cc_mem.compress(entries, mission, out_dir, force=False,
                              last_compress_mission=last_compress)
        if res.triggered:
            last_compress = mission
            entries = [e for lst in res.kept.values() for e in lst]
            total_forgotten += len(res.forgotten)
            trace.append({
                "mission": mission,
                "kept_tokens": res.kept_tokens,
                "forgotten": len(res.forgotten),
                "overflow": res.overflow_tokens,
                "duplicates": res.dropped_duplicates,
            })
    return {
        "missions": missions,
        "compressions": len(trace),
        "total_forgotten": total_forgotten,
        "final_entries": len(entries),
        "trace": trace,
        "budgets": {c.name: c.budget_tokens for c in cc_mem.DEFAULT_LAYERS},
    }


def fig_memory(sim: dict, out: str) -> None:
    trace = sim["trace"]
    ms = [t["mission"] for t in trace]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for layer, color in (("L1", "#3b7dd8"), ("L2", "#e0a458"), ("L3", "#2a9d8f")):
        ys = [t["kept_tokens"].get(layer, 0) for t in trace]
        ax1.step(ms, ys, where="post", label=f"{layer} kept tokens", color=color, lw=1.6)
        ax1.axhline(sim["budgets"][layer], ls=":", lw=0.9, color=color, alpha=0.8)
    ax1.set_xlabel("mission index"); ax1.set_ylabel("tokens held in layer")
    ax1.set_title("three-layer memory stays inside its budgets", fontsize=10)
    ax1.grid(alpha=0.3, lw=0.6); ax1.legend(fontsize=8)
    ax2.bar(ms, [t["forgotten"] for t in trace], color="#b5495b", width=1.6)
    ax2.set_xlabel("mission index (compression events only)")
    ax2.set_ylabel("entries forgotten (archived)")
    ax2.set_title(f"forget events: {sim['total_forgotten']} total, every entry archived+logged",
                  fontsize=10)
    ax2.grid(axis="y", alpha=0.3, lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="docs/context_craft_ab/ab_results.json")
    ap.add_argument("--out-dir", default="docs/context_craft_ab")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(a.results, "r", encoding="utf-8") as fh:
        res = json.load(fh)
    fig_bars(res, os.path.join(a.out_dir, "fig1_ab_bars.png"))
    fig_curves(res, os.path.join(a.out_dir, "fig2_budget_curves.png"))
    sim = simulate_memory()
    fig_memory(sim, os.path.join(a.out_dir, "fig3_memory_trace.png"))
    with open(os.path.join(a.out_dir, "simulation.json"), "w", encoding="utf-8") as fh:
        json.dump(sim, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({"figures": 3, "out_dir": a.out_dir,
                      "compressions": sim["compressions"],
                      "forgotten": sim["total_forgotten"],
                      "final_entries": sim["final_entries"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())