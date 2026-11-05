"""Generate the artifacts for the prose-claim ledger (docs/m3556/).

Everything printed here is computed: the JSON payload, the markdown table and the
figure all read the committed artifacts or re-run the 80-panel layer probe, so no
number in the mission report is hand-copied from a terminal.

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python docs/m3556/build_artifacts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorbem.benchmarks import prose_claims as ledger  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = ledger.check()
    coverage = ledger.coverage()
    (HERE / "prose_claims.json").write_text(
        json.dumps({"check": report, "coverage": coverage}, indent=2, ensure_ascii=False)
    )
    table = ledger.render_section(report)

    rows = ledger.representation_ladder()
    panels = np.array([row["panels"] for row in rows], dtype=float)
    represented = np.array([row["representation_relative_l2_error"] for row in rows])
    recovered = np.array([row["recovered_relative_l2_error"] for row in rows])
    low, high = ledger.representation_band()
    layer = ledger.near_body_layer()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    quoted = np.isin(panels, ledger.BAND_PANELS)
    ax.axhspan(low / 100.0, high / 100.0, color="tab:red", alpha=0.12,
               label=f"quoted band {low:.1f}%–{high:.1f}%")
    ax.loglog(panels, represented, "o-", color="tab:red", label="representation formula")
    ax.loglog(panels[quoted], represented[quoted], "o", ms=9, mfc="none", mec="tab:red",
              label="levels quoted in prose")
    ax.loglog(panels, recovered, "s-", color="tab:blue", label="recovered path")
    reference = recovered[0] * (panels / panels[0]) ** -1.0
    ax.loglog(panels, reference, "k--", lw=1.0, label="first-order reference")
    ax.set_xlabel("panels")
    ax.set_ylabel(r"relative $L_2$ error of surface velocity")
    ax.set_title("Representation path stalls, recovered path converges")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    labels = ["layer probe\n(1e-3 inside)", "body centre", "exact"]
    values = [layer["layered_velocity"], layer["centre_velocity"], layer["exact_velocity"]]
    colors = ["tab:orange", "tab:green", "0.6"]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}",
                ha="center", fontsize=9)
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.set_ylim(0.0, 1.2)
    ax.set_ylabel("velocity, x-component (exact = 1)")
    ax.set_title(
        "Near-body probes, live 80-panel solve: "
        f"{layer['layered_percent_error']:.1f}% and {layer['centre_percent_error']:.1f}%"
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("m3556 · prose claims in docs/three_dimensional_bem.md, re-measured", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "prose_claims.png", dpi=140)
    plt.close(fig)

    readme = f"""# m3556 — conclusion-class prose numbers become an executable contract

The four result tables of `docs/three_dimensional_bem.md` are pinned by
`three_dimensional_doc_pins` (§11). Everything a *reader quotes*, however, sits
outside those tables: the abstract, §5.5, §6.4, §7 and §9 restate the same three
headline facts in {len(ledger.PROSE_CLAIMS)} independently worded sentences. This mission binds
them to measurements with `tensorbem.benchmarks.prose_claims`.

Generated output — do not edit by hand, re-run `build_artifacts.py`:

* `prose_claims.json` — full `--check` verdict plus the coverage delta;
* `figures/prose_claims.png` — left: both surface-velocity paths with the quoted
  band shaded and the quoted levels ringed; right: the two near-body probes from
  a live 80-panel solve;
* the table below, rendered from the measurement side.

## Verdict

| quantity | value |
|---|---:|
| registered restatements | {report['claims']} |
| distinct printed numbers bound | {report['numeric_tokens_bound']} |
| qualitative claims checked | {report['qualitative_claims']} |
| drift | {report['drift']} |
| overall | **{report['verdict']}** |

Coverage of the document's numeric tokens: tables protect
{coverage['protected_in_tables']} of {coverage['numeric_tokens_total']}; this ledger adds
{coverage['bound_by_prose_claims']} distinct quoted numbers
({', '.join(coverage['bound_tokens'])}), i.e. unguarded drops from
{coverage['unguarded_before']} to {coverage['unguarded_after']}
(guarded fraction {coverage['guarded_fraction']:.3f}). It is **not** a claim that the
document is fully verified: derived tokens and formula coefficients remain
unguarded, and the honest statement stays "these quoted numbers are now counted".

## Claim ledger

{table}

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \\
  python -m tensorbem.benchmarks.prose_claims --check       # DRIFT -> exit 1
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \\
  python -m tensorbem.benchmarks.prose_claims --coverage
PYTHONPATH=src python -m pytest tests/test_prose_claims.py -q
```
"""
    (HERE / "README.md").write_text(readme)
    print(json.dumps({"verdict": report["verdict"], "coverage": coverage}, ensure_ascii=False))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())