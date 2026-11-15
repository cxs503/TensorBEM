"""Generate the KK-gate calibration artifacts for CASE-BEM-CUMMINS-WIGLEY.

Writes, under ``benchmarks/verified/casepool_bem_cummins_wigley/``:

* ``kk_gate_calibration.json``  -- the certificate plus the gate record;
* ``kk_gate_calibration.md``    -- the module's own render of it;
* ``kk_gate_scale_audit.json``  -- normalisation audit of the *committed*
  ``case_result.json`` (read-only input): the pre-declared KK bar is stated as a
  fraction of ``max|A|``, which contains the arbitrary ``A_inf`` offset, so the
  same bar is a different fraction of the added-mass amplitude for each DOF.

and, with matplotlib available, ``docs/figures/kk_gate_calibration.png``.

The committed case artifact is never modified: the audit reads it and writes new
files, so the published verdict and its ``CASE.md`` render stay in step (the
invariant locked by ``test_published_case_md_is_the_render_of_the_published_json``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks import kk_gate_calibration as kc  # noqa: E402

CASE_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"


def scale_audit(case_result: dict) -> dict:
    """Re-read the committed closure rows in a normalisation-free unit."""
    audit: dict = {
        "source": "benchmarks/verified/casepool_bem_cummins_wigley/case_result.json",
        "question": (
            "the pre-declared KK criterion is |A - A_KK| / max|A| <= 10 %, and "
            "max|A| contains the arbitrary instantaneous offset A_inf; the same "
            "bar therefore corresponds to a different fraction of the added-mass "
            "*amplitude* max|A - A_inf| for every DOF -- by how much, per gate"
        ),
        "gates": {},
    }
    for gate_id in ("C3", "C3a", "C6b"):
        gate = case_result.get("gates", {}).get(gate_id)
        if not gate:
            continue
        rows = []
        for row in gate["values"]["per_dof"]:
            mass = np.asarray(row["reference_added_mass"], dtype=float)
            offset = float(row["added_mass_inf_closure"])
            scale = float(row["residual_scale"])
            amplitude = float(np.max(np.abs(mass - offset)))
            dilution = amplitude / scale if scale > 0 else float("nan")
            reported = float(row["max_relative_residual"])
            rows.append(
                {
                    "dof": row["dof"],
                    "dof_name": row["dof_name"],
                    "residual_scale_max_abs_A": scale,
                    "added_mass_amplitude": amplitude,
                    "dilution_amplitude_over_scale": dilution,
                    "residual_reported": reported,
                    "residual_amplitude_normalised": (
                        reported / dilution if dilution > 0 else float("nan")
                    ),
                    "bar_in_amplitude_units": (
                        kc.KK_BAR / dilution if dilution > 0 else float("nan")
                    ),
                    "above_bar_in_amplitude_units": bool(
                        dilution > 0
                        and reported / dilution > kc.KK_BAR
                    ),
                }
            )
        worst = max(
            (r for r in rows if np.isfinite(r["residual_amplitude_normalised"])),
            key=lambda r: r["residual_amplitude_normalised"],
            default=None,
        )
        audit["gates"][gate_id] = {
            "passed_as_declared": bool(gate.get("passed")),
            "status_as_declared": gate.get("status"),
            "rows": rows,
            "worst_dof_amplitude_normalised": (
                worst["dof_name"] if worst else None
            ),
            "worst_residual_amplitude_normalised": (
                worst["residual_amplitude_normalised"] if worst else None
            ),
            "dofs_above_bar_in_amplitude_units": [
                r["dof_name"] for r in rows if r["above_bar_in_amplitude_units"]
            ],
        }
    return audit


def _figure(payload: dict, audit: dict, path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    channels = payload["certificate"]["channels"]
    for name, style in (
        ("damping_step", dict(marker="o", label="B * (1+delta) above 1 rad/s")),
        ("damping_bump", dict(marker="s", label="B * (1+delta*w), local bump")),
        ("added_mass_bump", dict(marker="^", label="A + delta*scale*w")),
        ("added_mass_bump_zero_mean", dict(marker="v", label="A + delta*scale*w, zero mean")),
    ):
        curve = channels[name]
        x = [p["delta"] for p in curve["points"]]
        y = [p["reported_residual"] for p in curve["points"]]
        axes[0].plot(x, y, linewidth=1.4, **style)
        ya = [
            p["amplitude_normalised_residual"] for p in curve["points"]
        ]
        axes[1].plot(x, ya, linewidth=1.4, **style)
    for ax, title in (
        (axes[0], "reported residual |dA|/max|A| (case statistic)"),
        (axes[1], "amplitude-normalised |dA|/max|A-A_inf|"),
    ):
        ax.axhline(kc.KK_BAR, color="k", linestyle="--", linewidth=1.0, label="10 % bar")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("injected error delta")
        ax.set_ylabel("residual")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[2].axhline(kc.KK_BAR, color="k", linestyle="--", linewidth=1.0, label="10 % bar")
    names, reported, amplitude = [], [], []
    for gate_id in ("C3", "C3a", "C6b"):
        rows = audit["gates"].get(gate_id, {}).get("rows", [])
        if not rows:
            continue
        names.append(gate_id)
        reported.append(max(r["residual_reported"] for r in rows))
        amplitude.append(max(r["residual_amplitude_normalised"] for r in rows))
    index = np.arange(len(names))
    axes[2].bar(index - 0.2, reported, width=0.4, label="|dA|/max|A| (declared)")
    axes[2].bar(index + 0.2, amplitude, width=0.4, label="|dA|/max|A-A_inf|")
    axes[2].set_xticks(index)
    axes[2].set_xticklabels(names)
    axes[2].set_yscale("log")
    axes[2].set_title("committed case, worst DOF under two normalisations", fontsize=10)
    axes[2].grid(True, which="both", axis="y", alpha=0.3)
    axes[2].legend(fontsize=7)
    fig.suptitle(
        "Kramers-Kronig gate calibration: injected error -> reported residual",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CASE_DIR)
    parser.add_argument(
        "--figure",
        type=Path,
        default=ROOT / "docs" / "figures" / "kk_gate_calibration.png",
    )
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    certificate = kc.certificate()
    gate = kc.calibration_gate(certificate)
    payload = {"certificate": certificate, "gate": gate}
    json_path = out / "kk_gate_calibration.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path = out / "kk_gate_calibration.md"
    markdown_path.write_text(kc.render_markdown(certificate, gate), encoding="utf-8")

    case_result = json.loads((out / "case_result.json").read_text())
    audit = scale_audit(case_result)
    audit_path = out / "kk_gate_scale_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    figure = _figure(payload, audit, args.figure)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(markdown_path),
                "audit": str(audit_path),
                "figure": figure,
                "gate_passed": gate["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())