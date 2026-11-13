"""Figures for the high-``ka`` band extension of CASE-BEM-HULME-DIFF-VERIF.

Two panels, both driven only by the JSON artefacts (no hand-typed numbers):

* ``force``  -- |F1| and |F3| against the Hulme reference, the TensorBEM BIE and
  the live capytaine route over the whole band ``ka in [0.2, 5]``;
* ``error``  -- the two error metrics side by side, which is what shows the
  cancellation regime where the base case's relative metric stops being
  informative.

Usage::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_diff_verif/make_band_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
BAND = HERE / "band_extension.json"
BASE = HERE / "results.json"
SCALE = 1025.0 * 9.80665  # rho g a^2, a = 1


def _cx(row: dict[str, float]) -> complex:
    return complex(row["re"], row["im"])


def _series() -> dict[str, list[float]]:
    """Merge the delivered low band with the extension into one ka axis."""
    band = json.loads(BAND.read_text())
    data: dict[str, list[float]] = {k: [] for k in
                                    ("ka", "ref_surge", "ref_heave",
                                     "bem_surge", "bem_heave",
                                     "cpt_surge", "cpt_heave",
                                     "rel_surge", "rel_heave",
                                     "abs_surge", "abs_heave")}

    base = json.loads(BASE.read_text())
    rows: list[tuple[float, dict, dict]] = []
    for cell in base["ladder"]:
        if int(cell["subdivisions"]) != 4:  # the case mesh, 1024 panels
            continue
        rows.append((float(cell["ka"]), cell["surge"], cell["heave"]))
    for ka, surge, heave in rows:
        ka = float(ka)
        data["ka"].append(ka)
        data["ref_surge"].append(abs(_cx(surge["reference"])))
        data["ref_heave"].append(abs(_cx(heave["reference"])))
        data["bem_surge"].append(abs(_cx(surge["bem"])))
        data["bem_heave"].append(abs(_cx(heave["bem"])))
        data["cpt_surge"].append(float("nan"))
        data["cpt_heave"].append(float("nan"))
        data["rel_surge"].append(surge["complex_relative"])
        data["rel_heave"].append(heave["complex_relative"])
        data["abs_surge"].append(abs(_cx(surge["bem"]) - _cx(surge["reference"])) / SCALE)
        data["abs_heave"].append(abs(_cx(heave["bem"]) - _cx(heave["reference"])) / SCALE)

    for key, cell in sorted(band["per_ka"].items(), key=lambda kv: float(kv[0])):
        ka = float(key)
        data["ka"].append(ka)
        data["ref_surge"].append(abs(_cx(cell["reference"]["surge"])))
        data["ref_heave"].append(abs(_cx(cell["reference"]["heave"])))
        case = cell["bem"]["4"]
        data["bem_surge"].append(abs(_cx(case["surge"])))
        data["bem_heave"].append(abs(_cx(case["heave"])))
        indep = cell.get("independent_finest")
        data["cpt_surge"].append(abs(_cx(indep["surge"]["value"])) if indep else float("nan"))
        data["cpt_heave"].append(abs(_cx(indep["heave"]["value"])) if indep else float("nan"))
        data["rel_surge"].append(cell["metrics"]["4"]["surge"]["relative"])
        data["rel_heave"].append(cell["metrics"]["4"]["heave"]["relative"])
        data["abs_surge"].append(cell["metrics"]["4"]["surge"]["absolute_normalised"])
        data["abs_heave"].append(cell["metrics"]["4"]["heave"]["absolute_normalised"])
    return data


def main() -> int:
    d = _series()
    ka = np.asarray(d["ka"])

    fig, (ax_f, ax_e) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    ax_f.plot(ka, np.asarray(d["ref_surge"]) / SCALE, "k-", lw=1.6, label="ref |F1| surge")
    ax_f.plot(ka, np.asarray(d["ref_heave"]) / SCALE, "k--", lw=1.6, label="ref |F3| heave")
    ax_f.plot(ka, np.asarray(d["bem_surge"]) / SCALE, "o", ms=4.5, mfc="none", label="BEM 1024 surge")
    ax_f.plot(ka, np.asarray(d["bem_heave"]) / SCALE, "s", ms=4.5, mfc="none", label="BEM 1024 heave")
    cpt_s = np.asarray(d["cpt_surge"]) / SCALE
    cpt_h = np.asarray(d["cpt_heave"]) / SCALE
    ax_f.plot(ka, cpt_s, "x", ms=6, color="tab:red", label="capytaine surge")
    ax_f.plot(ka, cpt_h, "+", ms=7, color="tab:green", label="capytaine heave")
    ax_f.axvline(2.0, color="0.6", lw=1.0, ls=":")
    ax_f.text(2.05, 0.02, "base case band ->", color="0.4", fontsize=8, rotation=90)
    ax_f.set_xlabel("ka (a = 1 m)")
    ax_f.set_ylabel("|F| / (rho g A a$^2$)")
    ax_f.set_title("Hemisphere exciting force, three routes")
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=7.5, ncol=2)

    ax_e.semilogy(ka, np.maximum(d["rel_surge"], 1e-16), "o-", ms=4, color="tab:blue",
                  label="rel. err surge (base metric)")
    ax_e.semilogy(ka, np.maximum(d["rel_heave"], 1e-16), "s-", ms=4, color="tab:cyan",
                  label="rel. err heave (base metric)")
    ax_e.semilogy(ka, np.maximum(d["abs_surge"], 1e-16), "o--", ms=4, color="tab:orange",
                  label="|dF|/(rho g A a$^2$) surge")
    ax_e.semilogy(ka, np.maximum(d["abs_heave"], 1e-16), "s--", ms=4, color="tab:red",
                  label="|dF|/(rho g A a$^2$) heave")
    ax_e.axhline(0.05, color="k", lw=1.0, ls=":", label="ledger tolerance 5 %")
    ax_e.set_xlabel("ka")
    ax_e.set_ylabel("error (dimensionless)")
    ax_e.set_title("Why the relative metric inflates near heave cancellation")
    ax_e.grid(alpha=0.3, which="both")
    ax_e.legend(fontsize=7.5)

    fig.tight_layout()
    out_force = HERE / "band_extension_force.png"
    out_err = HERE / "band_extension_error.png"
    fig.savefig(out_force, dpi=150)
    # second file: the error panel on its own, for the report figure slot
    fig2, ax2 = plt.subplots(figsize=(6.6, 4.6))
    for series, style, colour, label in (
        (d["rel_surge"], "o-", "tab:blue", "rel. err surge"),
        (d["rel_heave"], "s-", "tab:cyan", "rel. err heave"),
        (d["abs_surge"], "o--", "tab:orange", "|dF|/scale surge"),
        (d["abs_heave"], "s--", "tab:red", "|dF|/scale heave"),
    ):
        ax2.semilogy(ka, np.maximum(series, 1e-16), style, ms=4, color=colour, label=label)
    ax2.axhline(0.05, color="k", lw=1.0, ls=":")
    ax2.set_xlabel("ka")
    ax2.set_ylabel("error")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_err, dpi=150)
    print(f"wrote {out_force} and {out_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())