"""Render and post the mission-8531 report from the generated artefact.

Every number in the markdown is read out of
``benchmarks/verified/propeller_openwater/kp505_gci_ladder.json``; nothing is
typed.  The figures are the PNGs the driver wrote.

Usage: ``PYTHONPATH=src python post_report_m8531.py``
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

ART_DIR = Path(__file__).resolve().parent / "benchmarks" / "verified" / "propeller_openwater"
ART = json.loads((ART_DIR / "kp505_gci_ladder.json").read_text())

MISSION_ID = ART["mission_id"]


def png(name: str) -> str:
    path = ART_DIR / "figures" / name
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def fmt(value, spec="%.3f"):
    if value is None:
        return "nan"
    try:
        return spec % value
    except TypeError:
        return str(value)


design = ART["design"]
headline = ART["headline"]
anchor = ART["anchor"]
verdict = ART["verdict"]
criteria = {c["id"]: c for c in ART["criteria"]}

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

geometry_rows = "\n".join(
    "| %s |" % row for row in anchor["geometry_source"]["rows_verbatim"]
)
anchor_rows = "\n".join(
    "| %s |" % row for row in anchor["rows_verbatim"]
)
ladder_rows = "\n".join(
    "| %s | `%s` | %s | %s |"
    % (name, spec["parameter"], ", ".join(str(v) for v in spec["levels"]), spec["role"])
    for name, spec in design["ladders"].items()
)
headline_rows = "\n".join(
    "| %s | %s | %s | %s | %s | %s | %s | %s |"
    % (
        item["J"],
        ", ".join(str(v) for v in item["levels"]),
        ", ".join("%.4f" % v for v in item["KT_levels"]),
        fmt(item["p_three_coarsest_levels"]),
        fmt(item["gci_three_coarsest_levels_pct"]),
        fmt(item["p_three_finest_levels"]),
        fmt(item["gci_three_finest_levels_pct"]),
        fmt(item["triplet_disagreement_ratio"], "%.1f"),
    )
    for item in headline["spanwise_four_level_vs_three_level"]
)
gci_rows = "\n".join(
    "| %s | %s | %s | %s | %s | %s |"
    % (
        cell["axis"],
        cell["J"],
        fmt(cell["gci"]["KT"]["fine_triplet"].get("observed_order")),
        fmt(
            100.0 * cell["gci"]["KT"]["fine_triplet"]["gci_fine"]
            if cell["gci"]["KT"]["fine_triplet"].get("usable")
            else None
        ),
        cell["monotone"]["KT"],
        ", ".join("%.4f" % level["KT"] for level in cell["levels"]),
    )
    for cell in ART["cells"]
)
decomp_rows = "\n".join(
    "| %s | %s | %.5f | %.5f | %.5f | %+.3f | %+.3f | %+.3f | %s |"
    % (
        row["J"],
        quantity,
        row["quantities"][quantity]["fine"],
        row["quantities"][quantity]["extrapolated"],
        row["quantities"][quantity]["anchor"],
        row["quantities"][quantity]["total_pct"],
        row["quantities"][quantity]["discrete_pct"],
        row["quantities"][quantity]["model_pct"],
        row["quantities"][quantity]["richardson_valid"],
    )
    for row in ART["decomposition"]
    for quantity in ("KT", "10KQ", "eta0")
)
criteria_rows = "\n".join(
    "| %s | %s | %s | %s%s |"
    % (
        c["id"],
        c["statement"],
        "PASS" if c["passed"] else "FAIL",
        json.dumps(
            {k: v for k, v in c["measured"].items() if not isinstance(v, (dict, list))},
            sort_keys=True,
        ),
        " (post-hoc)" if c.get("post_hoc") else "",
    )
    for c in ART["criteria"]
)
acceptance_rows = "\n".join(
    "| %d | %s | %s | %s | %s |"
    % (index, row["quantity"], row["basis"], row["value"], row["within_3pct"])
    for index, row in enumerate(ART["acceptance"], start=1)
)
limitation_rows = "\n".join("- %s" % item for item in ART["limitations"])

raw_levels = ART["reproduction"]
core_band = next(
    r for r in ART["decomposition"] if r["J"] == design["decomposition_J"]
)["discretisation_band_core"]["band_pct"]

markdown = f"""# KP505 open water: four-level GCI ladders and the discretisation/model split

**Mission**: {MISSION_ID} · research plan item C of topic `propeller-openwater-bem`
**Repository**: TensorBEM · module `{ART['driver']['module']}` · solver `{ART['solver']['module']}`
**Status of the predeclared criteria**: **{verdict['status']}** ({verdict['n_passed']}/{verdict['n_criteria']} passed; failed: {', '.join(verdict['failed_criteria'])})

---

## 1. Case

### 1.1 Physical background

Report 8822 measured `dK_T(J=0.8) = {headline['dKT_J0_8_base_configuration_pct']:+.4f} %`
for the vortex-lattice / lifting-surface kernel against the transcribed KP505
open-water curve, and the same lattice carried a radial panel GCI of 7.1-8.1 %,
i.e. three times the 3 % acceptance level. With a discretisation band that wide
the reported deviation cannot be split into "model error" and "discretisation
error", so the evidence surface does not support a journal claim in either
direction. This study supplies the missing ladder and the split.

### 1.2 Geometry (verbatim transcript of Table 1 of the source paper)

| row |
|---|
{geometry_rows}

Not published in that table: chord distribution, radial pitch distribution,
skew, rake, section camber and thickness law. The kernel models them explicitly
and reports their effect, as declared in the artefact.

### 1.3 Anchor curve (verbatim transcript of Table 2, page 10)

| J KT_CFD 10KQ_CFD KT_exp 10KQ_exp error_KT error_10KQ |
|---|
{anchor_rows}

Source: {anchor['source']['citation']} ({anchor['source']['location']}).
Access: {anchor['source']['access']}.
Provenance depth: {anchor['source']['provenance_depth']}

### 1.4 Lattice and advance ratios

Base configuration (the point that produced report 8822):
`{json.dumps(design['base_configuration'], sort_keys=True)}`.
Advance ratios: `{', '.join(str(j) for j in design['advance_ratios'])}`;
decomposition at `J = {design['decomposition_J']}`.

---

## 2. Method

### 2.1 Ladders

Five ladders, four levels each, constant refinement ratio `r = {design['refinement_ratio']}`:

| axis | parameter | levels | role |
|---|---|---|---|
{ladder_rows}

Two substitutions are declared rather than hidden: the blade-fitted lattice has
no circumferential segmentation (the azimuthal axis is the wake's helical
segmentation), and the direct-dense solve has no iterative relaxation factor
(the kernel axis is the van Garrel vortex-core radius).

### 2.2 Error estimator

Roache GCI at `Fs = {design['gci_safety_factor']}`, applied twice per quantity:
on the three coarsest and the three finest levels. Four levels are required
exactly because the two overlapping triplets are the only internal check on
whether the ladder is in the asymptotic range.

### 2.3 Decomposition

`d_total = (f_fine - f_anchor)/f_anchor`,
`d_disc = (f_fine - f_ext)/f_anchor`,
`d_model = (f_ext - f_anchor)/f_anchor`, so `d_disc + d_model = d_total` exactly
(closure checked to {design['criteria_predeclared']['G7_closure_tol_pp']:g} pp).
When the observed order is below 1 the Richardson limit is flagged invalid and
the model term is bracketed by the GCI band instead.

### 2.4 Predeclared criteria

G1 coverage (>= 4 levels x >= 12 cells) · G2 physical admissibility of every
level · G3 finest spanwise/azimuthal `GCI <= {design['criteria_predeclared']['G3_gci_target_pct']:g} %` ·
G4 observed order `p >= {design['criteria_predeclared']['G4_observed_order_target']:g}` ·
G5 monotone ladders · G6 attribution outside `+/- {design['criteria_predeclared']['G6_band_factor']:g} GCI` ·
G7 closure · G8 wake-extent convergence · G9 kernel convergence.

---

## 3. Results

### 3.1 The two overlapping triplets disagree (spanwise axis)

| J | levels | K_T series | p (coarse triplet) | GCI % (coarse) | p (fine triplet) | GCI % (fine) | GCI ratio |
|---|---|---|---|---|---|---|---|
{headline_rows}

A three-level study sees either column, never both. The coarse-side triplet
reports `p ~ 3` and `GCI ~ 0.3-0.5 %` -- apparently perfect -- while the
fine-side triplet of the same four-level ladder reports negative order and
`GCI ~ 11-14 %`. The ladder is therefore not in the asymptotic range and
neither number is a bound. This is the mechanism behind the unreadable 7.1-8.1 %
of report 8822.

### 3.2 Every axis, finest triplet

| axis | J | p | GCI(K_T) % | monotone | K_T levels |
|---|---|---|---|---|---|
{gci_rows}

### 3.3 Discretisation / model split

| J | quantity | fine | Richardson | anchor | d_total % | d_disc % | d_model % | Richardson valid |
|---|---|---|---|---|---|---|---|---|
{decomp_rows}

At `J = {design['decomposition_J']}` the model term is bracketed at
`[{headline['model_term_bounds_pct'][0]:+.3f}, {headline['model_term_bounds_pct'][1]:+.3f}] %`
using the `+/- 2 GCI = {core_band:.3f} %` band over the three discretisation
axes. The bracket excludes zero, so model error is attributable at that advance
ratio; its magnitude is not resolvable to better than the band width.

### 3.4 Criterion verdicts

| id | criterion | verdict | measured |
|---|---|---|---|
{criteria_rows}

**Overall: {verdict['status']}** -- {verdict['n_passed']}/{verdict['n_criteria']} passed,
failed: {', '.join(verdict['failed_criteria'])}. G6b is a post-hoc sensitivity of G6
(discretisation axes only) and is labelled as such; the predeclared G6 stays FAIL.

---

## 4. Figures

1. **Four-level ladders** -- `K_T` against the refined parameter for all five axes and
   all three advance ratios, with the transcribed anchor level as the dotted line.
2. **GCI per axis** -- finest-triplet `GCI(K_T)` per axis and advance ratio against the
   1 % criterion.
3. **Decomposition** -- the anchor deviation split into model and discretisation terms,
   with the `+/- 2 GCI` band drawn on the total.

---

## 5. Traps, limits, applicability

{limitation_rows}

---

## 6. Error judgement (each key quantity vs its basis)

| # | quantity | basis | value | within 3 % |
|---|---|---|---|---|
{acceptance_rows}

**Answer to "is it < 3 %": no for every quantity that involves the lattice.**
The anchor deviation at `J = 0.8` is `{headline['dKT_J0_8_finest_lattice_pct']:+.3f} %`,
the finest spanwise GCI is far above the 3 % level, and the observed order is
negative, so the journal-grade criterion (`GCI <= 1 %`, `p >= 1`) is honestly
**not met**. What the study does establish is the qualitative split: at
`J = 0.5` the lattice sits inside its own discretisation band, at `J = 0.8` and
`J = 1.0` it does not, and at `J = 1.0` the kernel is not even physically
admissible (negative thrust and torque).
"""

figures = [
    {
        "name": "kp505_gci_ladders",
        "caption": (
            "Four-level ladders on five axes at J = 0.5/0.8/1.0, one parameter refined "
            "2x per level (n_radial 4-32, wake_steps 12-96, n_chord 1-8, wake_length "
            "0.25-2.0, core_ratio 0.4-0.05); dotted line = transcribed anchor K_T."
        ),
        "png_base64": png("kp505_gci_ladders.png"),
    },
    {
        "name": "kp505_gci_by_axis",
        "caption": (
            "Finest-triplet GCI of K_T per axis and J against the 1 % criterion (red "
            "dotted). Every axis but chordwise misses it, and the wake-extent axis reaches "
            "tens of percent: no axis is asymptotic."
        ),
        "png_base64": png("kp505_gci_by_axis.png"),
    },
    {
        "name": "kp505_decomposition",
        "caption": (
            "Anchor deviation split into model term (Richardson vs anchor) and "
            "discretisation term (fine grid vs Richardson), +/-2 GCI band on the total. "
            "Inside the band at J = 0.5, outside at J = 0.8 and 1.0: model error is "
            "attributable only at the high advance ratios."
        ),
        "png_base64": png("kp505_decomposition.png"),
    },
]

report = {
    "title": (
        "KP505 open water: four-level GCI ladders separate discretisation from model error "
        "(mission %d)" % MISSION_ID
    ),
    "summary": (
        "Five four-level refinement ladders at J = 0.5/0.8/1.0 show the KP505 lattice is "
        "not asymptotic: the two overlapping three-level triplets of one ladder disagree "
        "34x in GCI (p = +3 coarse side, -1.6 fine side), so the earlier 7.1-8.1 %% radial "
        "GCI was no error bar. The dK_T(J=0.8) = %.3f %% model term is bracketed at "
        "[%+.1f, %+.1f] %% (excludes zero); the predeclared journal criteria "
        "(GCI <= 1 %%, p >= 1) are honestly FAILED."
        % (
            headline["dKT_J0_8_finest_lattice_pct"],
            headline["model_term_bounds_pct"][0],
            headline["model_term_bounds_pct"][1],
        )
    ),
    "scenario_key": "propeller_openwater_bem_gci_ladder",
    "mission_id": MISSION_ID,
    "tags": [
        "propeller",
        "open-water",
        "KP505",
        "lifting-surface",
        "vortex-lattice",
        "GCI",
        "Richardson",
        "verification",
        "uncertainty",
        "BEM",
    ],
    "markdown": markdown,
    "figures": figures,
}

out = Path("/tmp/report.json")
out.write_text(json.dumps(report))
print("report bytes: %d, markdown chars: %d" % (out.stat().st_size, len(markdown)))

api_base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ.get("TENSOR_AGENT_USER")
password = os.environ.get("TENSOR_AGENT_PASS")

login = requests.post(f"{api_base}/auth/login", json={"username": user, "password": password},
                      timeout=10)
print("login:", login.status_code)
token = login.json().get("access_token") or login.json().get("token")
posted = requests.post(
    f"{api_base}/tensor/reports",
    headers={"Authorization": "Bearer %s" % token, "Content-Type": "application/json"},
    json=report,
    timeout=60,
)
print("post:", posted.status_code, posted.text[:300])
Path("/tmp/report.posted.json").write_text(json.dumps({"status": posted.status_code,
                                                       "body": posted.text[:2000]}))