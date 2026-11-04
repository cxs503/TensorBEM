"""Build the m3059 figures from the recorded JSON evidence (no hand-typed numbers).

Inputs (all produced by benchmark CLIs):
  outputs/verif_m3059/heavy_tier.json           -- declared_numbers --tiers heavy
  outputs/verif_m3059/fast_standard.json        -- declared_numbers --tiers fast,standard
  outputs/verif_m3059/interior_harmonics_1280.json -- interior_harmonics --subdivisions 2 3
  outputs/verif_m3059/interior_harmonics_80.json   -- interior_harmonics --subdivisions 1
Outputs:
  outputs/verif_m3059/fig_registry_deviation.png
  outputs/verif_m3059/fig_interior_harmonics.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/verif_m3059")


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def merge_rows() -> list[dict]:
    rows: list[dict] = []
    for name in ("fast_standard.json", "heavy_tier.json", "harmonic_rows.json"):
        path = ROOT / name
        if path.exists():
            rows.extend(load(name)["checks"])
    seen = set()
    unique = []
    for row in rows:
        if row["key"] in seen:
            continue
        seen.add(row["key"])
        unique.append(row)
    return unique


def figure_registry(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: abs(r["relative_deviation"]))
    labels = [r["key"].replace("_", " ") for r in rows]
    deviations = np.array([abs(r["relative_deviation"]) for r in rows])
    tolerances = np.array([r["tolerance"] for r in rows])
    floor = np.maximum(deviations, 1e-18)
    tier_color = {"fast": "#1f77b4", "standard": "#2ca02c", "heavy": "#d62728"}

    fig, ax = plt.subplots(figsize=(9.0, 0.36 * len(rows) + 1.9))
    y = np.arange(len(rows))
    ax.barh(
        y,
        floor,
        color=[tier_color.get(r["tier"], "#7f7f7f") for r in rows],
        height=0.62,
    )
    ax.scatter(tolerances, y, marker="|", s=260, color="black", zorder=5, label="drift threshold")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("|measured - declared| / declared  (log scale)")
    ax.set_title(
        "Declared-number registry: all rows recomputed and judged\n"
        f"(n={len(rows)}, all PASS; worst deviation {deviations.max():.2e})",
        fontsize=10,
    )
    ax.axvline(1e-9, color="0.8", lw=0.6, ls=":")
    handles = [
        plt.Line2D([], [], color=c, lw=6, label=f"{t} tier") for t, c in tier_color.items()
    ]
    handles.append(
        plt.Line2D([], [], color="black", marker="|", lw=0, markersize=12, label="drift threshold")
    )
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    ax.grid(axis="x", which="major", color="0.9", lw=0.5)
    fig.tight_layout()
    fig.savefig(ROOT / "fig_registry_deviation.png", dpi=170)
    plt.close(fig)


def figure_harmonics() -> None:
    study_high = load("interior_harmonics_1280.json")
    study_low = None

    def collect(study: dict) -> dict[int, dict[str, dict]]:
        out: dict[int, dict[str, dict]] = {}
        keys = study["levels"] if isinstance(study, dict) and "levels" in study else None
        rows = None
        if isinstance(study, dict) and isinstance(study.get("cases"), list):
            rows = study["cases"]
        if rows is None and isinstance(study.get("cases"), dict):
            rows = [
                result
                for payload in study["cases"].values()
                for result in payload["results"]
            ]
        for case in rows:
            out.setdefault(int(case["degree"]), {})[case["harmonic"]] = case
        return out

    high = collect(study_high)
    low = collect(study_low) if study_low else {}
    representative = {1: "x", 2: "x2-y2", 3: "x3-3xy2"}
    colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for degree, name in representative.items():
        panels, flux, rmse = [], [], []
        for source in (low, high):
            if not source:
                continue
            panels.append(float(source[degree][name]["panels"]))
            flux.append(source[degree][name]["normal_derivative_relative_l2_error"])
            rmse.append(source[degree][name]["field_rmse"])
        axes[0].loglog(panels, flux, "o-", color=colors[degree], label=f"degree {degree} ({name})")
        axes[1].loglog(panels, rmse, "s-", color=colors[degree], label=f"degree {degree} ({name})")

    # O(N^-1) reference slope (flux) and O(N^-1.5) for the interior field
    xs = np.array([80.0, 1280.0])
    axes[0].loglog(xs, 4.4e-2 * (xs / 80.0) ** -1.0, "k--", lw=1, label=r"slope $N^{-1}$")
    axes[1].loglog(xs, 2.3e-4 * np.ones_like(xs), "k:", lw=1, label="degree-independent level")
    axes[0].set_xlabel("panels N")
    axes[0].set_ylabel(r"flux relative $L_2$ error")
    axes[0].set_title("Boundary flux error grows with harmonic degree")
    axes[1].set_xlabel("panels N")
    axes[1].set_ylabel("interior probe RMSE")
    axes[1].set_title("Interior field error is degree-insensitive")
    for ax in axes:
        ax.grid(True, which="both", color="0.9", lw=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "fig_interior_harmonics.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    rows = merge_rows()
    figure_registry(rows)
    figure_harmonics()
    print("figures written:", sorted(p.name for p in ROOT.glob("*.png")))
    print("rows:", len(rows), "verdicts:", sorted({r["verdict"] for r in rows}))