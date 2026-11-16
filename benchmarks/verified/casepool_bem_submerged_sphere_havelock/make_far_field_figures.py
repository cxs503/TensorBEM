#!/usr/bin/env python3
"""Figures for the high-ka near-field / far-field damping study (Havelock sphere).

Reads ``docs/case_bem_submerged_sphere_havelock/far_field_high_ka.json`` and writes

* ``fig15_b33_near_far_ref.png``   damping vs ka: reference, near field, far field
* ``fig16_near_vs_far_agreement.png`` two numerical routes vs each other (no reference)
* ``fig17_kochin_azimuth_convergence.png``  Kochin azimuth resolution study

Run::

    PYTHONPATH=src python3 \
        benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_far_field_figures.py \
        --artifact docs/case_bem_submerged_sphere_havelock/far_field_high_ka.json \
        --outdir docs/case_bem_submerged_sphere_havelock
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CASE_SUB = 3
ECONOMY_SUB = 2
COLORS = {CASE_SUB: "#1f4e79", ECONOMY_SUB: "#c55a11"}


def _rows(payload: dict, subdivision: int) -> list[dict]:
    rows = [r for r in payload["rows"] if r["subdivision"] == subdivision]
    return sorted(rows, key=lambda r: r["ka"])


def _series(rows: list[dict], path: tuple[str, ...]) -> list[float]:
    out = []
    for row in rows:
        node = row
        for key in path:
            node = node[key]
        out.append(float(node))
    return out


def fig_band(payload: dict, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for subdivision, style in ((CASE_SUB, "-o"), (ECONOMY_SUB, "--s")):
        rows = _rows(payload, subdivision)
        ka = [r["ka"] for r in rows]
        ax.semilogy(
            ka,
            _series(rows, ("damping_near", "value", "b33")),
            style,
            color=COLORS[subdivision],
            markersize=5,
            label=f"near field (pressure), {rows[0]['n_panels']} panels",
        )
        ax.semilogy(
            ka,
            _series(rows, ("damping_far", "value", "b33")),
            style,
            color=COLORS[subdivision],
            markerfacecolor="none",
            markersize=7,
            label=f"far field (Kochin), {rows[0]['n_panels']} panels",
        )
    rows = _rows(payload, CASE_SUB)
    ax.semilogy(
        [r["ka"] for r in rows],
        _series(rows, ("damping_near", "ref", "b33")),
        "k-",
        linewidth=1.0,
        label="multipole reference (Havelock/Thorne/Wang)",
    )
    ax.set_xlabel("ka")
    ax.set_ylabel("B33 / (rho omega nabla)")
    ax.set_title("Heave damping: two numerical routes vs the analytic reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    out = outdir / "fig15_b33_near_far_ref.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_agreement(payload: dict, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for subdivision in (CASE_SUB, ECONOMY_SUB):
        rows = _rows(payload, subdivision)
        ka = [r["ka"] for r in rows]
        for key, marker in (("b33", "o"), ("b11", "s")):
            ax.loglog(
                ka,
                [max(r["near_vs_far"][key], 1e-18) for r in rows],
                "-" + marker,
                color=COLORS[subdivision],
                markerfacecolor="none" if key == "b11" else COLORS[subdivision],
                label=f"{key}, {rows[0]['n_panels']} panels",
            )
    ax.axhline(1e-2, color="crimson", linestyle=":", linewidth=1.2,
               label="1 % cross-route tolerance")
    ax.set_xlabel("ka")
    ax.set_ylabel("|B_near - B_far| / B_far")
    ax.set_title("Pressure route vs energy route (no reference solution involved)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    out = outdir / "fig16_near_vs_far_agreement.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_azimuth(payload: dict, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rows = _rows(payload, CASE_SUB)
    ka = [r["ka"] for r in rows]
    az = sorted(int(k) for k in rows[0]["damping_far"]["by_azimuth"])
    for n_prev, n_here in zip(az[:-1], az[1:]):
        values = []
        for row in rows:
            coarse = row["damping_far"]["by_azimuth"][str(n_prev)]["b33"]
            fine = row["damping_far"]["by_azimuth"][str(n_here)]["b33"]
            values.append(abs(coarse - fine) / abs(fine))
        ax.loglog(ka, [max(v, 1e-16) for v in values], "-o", markersize=4,
                  label=f"n_azimuth {n_prev} -> {n_here}")
    ax.axhline(5e-3, color="crimson", linestyle=":", linewidth=1.2,
               label="5e-3 convergence tolerance")
    ax.set_xlabel("ka")
    ax.set_ylabel("relative change of B33_far")
    ax.set_title("Kochin azimuth resolution study (1280 panels)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    out = outdir / "fig17_kochin_azimuth_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.artifact).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for maker in (fig_band, fig_agreement, fig_azimuth):
        print(maker(payload, outdir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())