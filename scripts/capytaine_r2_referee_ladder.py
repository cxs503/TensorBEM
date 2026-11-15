"""Is the *referee* converged at the density the round-2 anchor used?

The anchor's internal diagonal band (5 %) still fails at the short-wave end
(nu = 2 m^-1) after both codes were put on comparable meshes: TensorBEM 1024 /
1026 panels against capytaine 784 / 1120.  That deviation has so far been read
as "TensorBEM discretisation error", which assumes the referee is closer to the
continuum than the defendant.  This script tests that assumption on capytaine's
own mesh ladder, **without touching TensorBEM**: it refines the referee to 8800
panels at nu = 2 and predeclares (here, before any of the new numbers is looked
at) that the referee counts as converged only if the two finest meshes differ by
at most ``CONVERGENCE_TOL`` (1 %) on the key diagonal entries.

Two predeclared criteria share the 1 % threshold:

* **C1 (asymptotic)** -- the two finest referee meshes must differ by at most
  1 %: is the referee converged *at all* on this ladder?
* **C2 (anchor density)** -- the referee's value at the anchor's own mesh must
  differ from its finest value by at most 1 %: can the anchor's referee number
  be read as the continuum limit?

C2 is the decision-relevant one: the anchor compares TensorBEM's finest mesh
against the referee value *at the anchor's mesh*, so a referee still moving by
several percent between those two meshes cannot adjudicate a 5 % band.  Nothing here claims that TensorBEM is right; it only measures how
much of the 6.5 % the referee itself moves.

Only this script and ``capytaine_crosscheck_r2_ref.py`` import ``capytaine``
(optional dependency; see ``test_capytaine_does_not_leak_into_src_or_tests``).

    python scripts/capytaine_r2_referee_ladder.py            # produce artifact
    python scripts/capytaine_r2_referee_ladder.py --check     # fail if stale
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tensorbem.benchmarks import capytaine_crosscheck_r2 as r2  # noqa: E402

DENSITY = r2.DENSITY
GRAVITY = r2.GRAVITY
DOF_ORDER = r2.DOF_ORDER

#: The one frequency at which the anchor's internal band fails.
NU = 2.0
#: capytaine ``mesh_sphere`` resolutions: (n_theta, n_phi).  The first row is the
#: anchor's own referee mesh (784 / 1120 wetted faces), so the ladder starts
#: exactly where the committed comparison does; the rest are new.
REF_LEVELS: dict[str, tuple[tuple[int, int], ...]] = {
    "hemisphere_r1": ((28, 56), (40, 80), (56, 112), (80, 160)),
    "sphere_r1_z-0.5": ((28, 56), (40, 80), (56, 112), (80, 160)),
}

#: Predeclared convergence threshold (both criteria, see module docstring).
CONVERGENCE_TOL = 0.01

#: Diagonal entries scored: the two that carry the anchor's failure (heave
#: damping on the hemisphere, surge/sway added mass on the sphere) plus their
#: symmetric partners.
KEY_ENTRIES = {
    "hemisphere_r1": (
        ("added_mass", 2, 2),
        ("damping", 2, 2),
        ("added_mass", 0, 0),
        ("damping", 0, 0),
    ),
    "sphere_r1_z-0.5": (
        ("added_mass", 0, 0),
        ("added_mass", 1, 1),
        ("added_mass", 2, 2),
        ("damping", 2, 2),
    ),
}

OUT = r2.ARTIFACT_ROOT / "referee_ladder.json"
DOC = Path("docs/capytaine_r2_referee_ladder.md")


def _referee_side() -> dict[str, Any]:
    """Live capytaine ladder.  Imported lazily: the module must stay importable
    on nodes without capytaine."""
    import capytaine as cpt

    import capytaine_crosscheck_r2_ref as ref

    solver = cpt.BEMSolver()
    cases: dict[str, Any] = {}
    for case, levels in REF_LEVELS.items():
        centre = tuple(float(v) for v in r2.CASES[case]["centre"])
        rows = []
        for level in levels:
            mesh = ref._mesh(cpt, case, level)
            body = ref._body(cpt, mesh, centre)
            columns = ref._columns(cpt, solver, body, DOF_ORDER, (NU,))
            key = "{:.3f}".format(NU)
            rows.append(
                {
                    "level": list(level),
                    "n_faces": int(mesh.nb_faces),
                    "matrices": {
                        "added_mass": columns[key]["added_mass"],
                        "damping": columns[key]["damping"],
                    },
                }
            )
        cases[case] = {"rotation_center": list(centre), "levels": rows}
    return {
        "cases": cases,
        "provenance": {
            "tool": "capytaine",
            "version": getattr(cpt, "__version__", "unknown"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "density": DENSITY,
            "gravity": GRAVITY,
            "water_depth": "infinite",
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
        },
    }


def _anchor_entries() -> dict[str, Any]:
    """The committed TensorBEM value of every key entry at its finest mesh."""
    study = json.loads((r2.ARTIFACT_ROOT / r2.RESULT_FILE).read_text())
    out: dict[str, Any] = {}
    for case, spec in KEY_ENTRIES.items():
        block = study["comparison"]["cases"][case]
        chosen = {}
        for quantity, i, j in spec:
            match = [
                e
                for e in block["entries"]
                if e["quantity"] == quantity and e["row"] == DOF_ORDER[i]
                and e["column"] == DOF_ORDER[j] and e["nu"] == NU
            ]
            if len(match) != 1:
                raise RuntimeError(f"{case} {quantity} A[{i},{j}] not in comparison")
            chosen["{}|{}|{}".format(quantity, i, j)] = match[0]
        out[case] = {
            "tensorbem_faces": block["tensorbem_faces"],
            "capytaine_faces": block["capytaine_faces"],
            "entries": chosen,
        }
    return out


def _scores(referee: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    worst_change = 0.0
    worst_label = None
    for case, spec in KEY_ENTRIES.items():
        levels = referee["cases"][case]["levels"]
        rows = []
        for quantity, i, j in spec:
            values = [float(np.asarray(row["matrices"][quantity])[i, j]) for row in levels]
            faces = [row["n_faces"] for row in levels]
            last_pair = abs(values[-1] - values[-2]) / abs(values[-1])
            from_anchor = abs(values[-1] - values[0]) / abs(values[-1])
            entry = anchor[case]["entries"]["{}|{}|{}".format(quantity, i, j)]
            rows.append(
                {
                    "quantity": quantity,
                    "row": DOF_ORDER[i],
                    "column": DOF_ORDER[j],
                    "n_faces": faces,
                    "values": values,
                    "change_anchor_mesh_to_finest": from_anchor,
                    "change_finest_pair": last_pair,
                    "referee_asymptotically_converged": bool(
                        last_pair <= CONVERGENCE_TOL
                    ),
                    "referee_converged_at_anchor_mesh": bool(
                        from_anchor <= CONVERGENCE_TOL
                    ),
                    "anchor_tensorbem": entry["tensorbem"],
                    "anchor_capytaine": entry["capytaine"],
                    "anchor_deviation": entry["relative"],
                    "anchor_band": entry["band"],
                    "deviation_vs_finest_referee": (
                        entry["tensorbem"] - values[-1]
                    )
                    / abs(values[-1]),
                    "within_band_vs_finest_referee": bool(
                        abs(entry["tensorbem"] - values[-1]) / abs(values[-1])
                        <= entry["band"]
                    ),
                }
            )
            if last_pair > worst_change:
                worst_change, worst_label = last_pair, "{} A[{},{}] {}".format(
                    case, i, j, quantity
                )
        cases[case] = rows
    first = cases[list(cases)[0]][0]["n_faces"]
    asymptotic = all(
        row["referee_asymptotically_converged"] for rows in cases.values() for row in rows
    )
    at_anchor = all(
        row["referee_converged_at_anchor_mesh"] for rows in cases.values() for row in rows
    )
    worst_from_anchor = max(
        (row["change_anchor_mesh_to_finest"] for rows in cases.values() for row in rows)
    )
    rows = [row for rows in cases.values() for row in rows]
    resolved = [
        "{} A[{},{}] {}: {:+.2f}% -> {:+.2f}%".format(
            case,
            DOF_ORDER.index(row["row"]),
            DOF_ORDER.index(row["column"]),
            row["quantity"],
            100.0 * row["anchor_deviation"],
            100.0 * row["deviation_vs_finest_referee"],
        )
        for case, rows_ in cases.items()
        for row in rows_
        if abs(row["anchor_deviation"]) > row["anchor_band"]
        and row["within_band_vs_finest_referee"]
    ]
    return {
        "anchor_band_failures": [
            "{} A[{},{}] {}: {:+.2f}%".format(
                case,
                DOF_ORDER.index(row["row"]),
                DOF_ORDER.index(row["column"]),
                row["quantity"],
                100.0 * row["anchor_deviation"],
            )
            for case, rows_ in cases.items()
            for row in rows_
            if abs(row["anchor_deviation"]) > row["anchor_band"]
        ],
        "anchor_band_failures_resolved_against_finest_referee": resolved,
        "criteria": {
            "C1_asymptotic": {
                "definition": "|f_n - f_(n-1)| / |f_n| on the two finest referee meshes",
                "faces": [first[-2], first[-1]],
                "tolerance": CONVERGENCE_TOL,
                "satisfied": bool(asymptotic),
            },
            "C2_anchor_mesh": {
                "definition": "|f_n - f_anchor| / |f_n|, anchor mesh vs finest referee mesh",
                "faces": [first[0], first[-1]],
                "tolerance": CONVERGENCE_TOL,
                "satisfied": bool(at_anchor),
            },
        },
        "cases": cases,
        "worst_finest_pair_change": worst_change,
        "worst_entry": worst_label,
        "worst_anchor_mesh_to_finest_change": worst_from_anchor,
        "verdict": (
            "CONVERGED"
            if (asymptotic and at_anchor)
            else "NOT_CONVERGED"
        ),
    }


def build() -> dict[str, Any]:
    referee = _referee_side()
    anchor = _anchor_entries()
    return {
        "anchor": "XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE",
        "nu": NU,
        "purpose": (
            "measure how much the referee (capytaine) itself moves at the "
            "anchor's wave frequency, so that the residual 5-7 % diagonal "
            "deviation is not attributed to TensorBEM by default"
        ),
        "provenance": referee["provenance"],
        "levels": {case: levels for case, levels in REF_LEVELS.items()},
        "referee": referee["cases"],
        "anchor_entries": anchor,
        "scoring": _scores(referee, anchor),
    }


def render_document(study: dict[str, Any]) -> str:
    scoring = study["scoring"]
    out: list[str] = []
    add = out.append
    add("# capytaine 自身网格收敛（nu = {:g} m^-1，裁判侧独立测量）\n".format(study["nu"]))
    add("锚点 `{}` 的内部对角带（{:.0%}）在短波端（nu = {:g} m^-1）未达标。".format(
        study["anchor"], r2.TOL_DIAGONAL, study["nu"]))
    add("此前把该偏差默认读作 TensorBEM 的离散误差，前提是裁判（capytaine）比被告更接近连续极限。")
    add("本文件在同一频率上单独加密裁判网格，**不重解 TensorBEM**，测量裁判自身在锚点密度上的移动量。\n")
    add("## 预登记判据\n")
    add("- 指标：关键对角项在裁判两层最细网格之间的相对变化 `|f_n - f_(n-1)| / |f_n|`；")
    add("- 阈值：`{:.0f}%`（`CONVERGENCE_TOL`，在看数之前登记）；".format(100.0 * CONVERGENCE_TOL))
    add("- C1（渐近）：最细两层之间的变化 ≤ 阈值 —— 裁判在这条阶梯上**是否**已经收敛；")
    add("- C2（锚点密度）：锚点自己的网格与最细网格之间的变化 ≤ 阈值 —— 锚点用的那个裁判数能否当连续极限读。")
    add("- 任一项超过阈值即判裁判**未收敛**，此时锚点的 5 % 带不能被当作 TensorBEM 的对错判决。\n")
    add("## 裁判网格阶梯（capytaine {}）\n".format(study["provenance"].get("version")))
    for case, rows in scoring["cases"].items():
        anchor = study["anchor_entries"][case]
        add("### {}\n".format(case))
        add("| 项 | " + " | ".join("{} 面板".format(n) for n in rows[0]["n_faces"]) + " | C1 最细两层 | C2 锚点→最细 | 判定 |")
        add("|---|" + "---:|" * len(rows[0]["n_faces"]) + "---:|---:|---|")
        for row in rows:
            add(
                "| {} A[{},{}] | ".format(row["quantity"], row["row"], row["column"])
                + " | ".join("{:.4g}".format(v) for v in row["values"])
                + " | {:.2f}% | {:.2f}% | {} |".format(
                    100.0 * row["change_finest_pair"],
                    100.0 * row["change_anchor_mesh_to_finest"],
                    "收敛"
                    if row["referee_asymptotically_converged"]
                    and row["referee_converged_at_anchor_mesh"]
                    else "未收敛",
                )
            )
        add("")
        add(
            "锚点处 TensorBEM {} 面板的对应项与裁判 {} 面板的偏差见下表。\n".format(
                anchor["tensorbem_faces"], anchor["capytaine_faces"]
            )
        )
        add("| 项 | TensorBEM | 裁判(锚点密度) | 锚点偏差 | 登记带 | 裁判自移动(锚点→最细) | 对最细裁判的偏差 | 入带 |")
        add("|---|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            add(
                "| {} A[{},{}] | {:.4g} | {:.4g} | {:+.2f}% | {:.0f}% | {:.2f}% | {:+.2f}% | {} |".format(
                    row["quantity"],
                    row["row"],
                    row["column"],
                    row["anchor_tensorbem"],
                    row["anchor_capytaine"],
                    100.0 * row["anchor_deviation"],
                    100.0 * row["anchor_band"],
                    100.0 * row["change_anchor_mesh_to_finest"],
                    100.0 * row["deviation_vs_finest_referee"],
                    "是" if row["within_band_vs_finest_referee"] else "否",
                )
            )
        add("")
    add("## 判定\n")
    c1, c2 = scoring["criteria"]["C1_asymptotic"], scoring["criteria"]["C2_anchor_mesh"]
    add(
        "- **C1**（各算例最细两层）：最大变化 {:.2f}%（{}），阈值 {:.0f}% → {}。".format(
            100.0 * scoring["worst_finest_pair_change"],
            scoring["worst_entry"],
            100.0 * CONVERGENCE_TOL,
            "满足" if c1["satisfied"] else "**不满足**",
        )
    )
    add(
        "- **C2**（锚点自身网格 vs 各算例最细网格）：最大变化 {:.2f}%，阈值 {:.0f}% → {}。".format(
            100.0 * scoring["worst_anchor_mesh_to_finest_change"],
            100.0 * CONVERGENCE_TOL,
            "满足" if c2["satisfied"] else "**不满足**",
        )
    )
    add("")
    add(
        "- 锚点密度处超带的 {} 项中，换成最细裁判后有 {} 项回到带内：{}。".format(
            len(scoring["anchor_band_failures"]),
            len(scoring["anchor_band_failures_resolved_against_finest_referee"]),
            "；".join(scoring["anchor_band_failures_resolved_against_finest_referee"])
            or "无",
        )
    )
    add(
        "- 总判定：**{}** —— {}。".format(
            scoring["verdict"],
            "裁判在锚点密度上可当连续极限，锚点偏差可归给 TensorBEM"
            if scoring["verdict"] == "CONVERGED"
            else "裁判在锚点密度上仍在移动，锚点的 5% 对角带对这一密度不构成 TensorBEM 的对错判决",
        )
    )
    add("")
    add("## 注意点\n")
    add("1. 本文件只测裁判侧的移动量，**不**证明 TensorBEM 正确；两侧同时加密才是判决实验。")
    add("2. 阶梯只有 {} 层、单一频率（nu = {:g} m^-1），不外推到其它频率。".format(
        len(next(iter(scoring["cases"].values()))[0]["n_faces"]), study["nu"]))
    add("3. capytaine 的网格族是 (n_theta, n_phi) 经纬网格，与 TensorBEM 的 icosphere 不同族；"
        "两条阶梯的收敛速率不可直接比较，只能各自读趋势。")
    add("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    path = ROOT / OUT
    if args.check:
        if not path.exists():
            print("MISSING: {}".format(path))
            return 1
        study = json.loads(path.read_text())
        fresh = _scores({"cases": study["referee"]}, study["anchor_entries"])
        stale = json.dumps(fresh, sort_keys=True) != json.dumps(
            study["scoring"], sort_keys=True
        )
        print("STALE" if stale else "OK")
        return 1 if stale else 0

    study = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
    doc = ROOT / DOC
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render_document(study))
    print("wrote {}".format(path.relative_to(ROOT)))
    print("wrote {}".format(DOC))
    print(
        "referee verdict {} (worst finest-pair change {:.2%} on {})".format(
            study["scoring"]["verdict"],
            study["scoring"]["worst_finest_pair_change"],
            study["scoring"]["worst_entry"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())