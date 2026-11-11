#!/usr/bin/env python3
"""Render the archived numbers of the acoustic BEM node into markdown and a report payload.

Every table here is generated from ``docs/acoustic3d_ices209/criteria.json`` and
``docs/acoustic3d_ices209/hull_band_table.json`` -- no number is retyped, which
is what the repository's *zero transcription* rule requires.  Outputs:

* ``docs/acoustic3d_ices209/results_tables.md`` -- the numeric sections quoted by
  ``docs/acoustic3d_helmholtz.md``;
* ``docs/acoustic3d_ices209/report_payload.json`` -- the platform report body
  (title / summary / markdown / figures as base64 PNG).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "acoustic3d_ices209"


def md_table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    if value != 0.0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
        return f"{value:.{digits-1}e}"
    return f"{value:.{digits}f}"


def main() -> int:
    criteria = json.loads((OUT / "criteria.json").read_text())
    hull = json.loads((OUT / "hull_band_table.json").read_text())
    crit = criteria["criteria"]
    blocks: list[str] = []

    # verdict summary --------------------------------------------------------
    rows = []
    for key, label in (
        ("a_pulsating_sphere", "a. pulsating sphere: |p_BEM - p_exact| / |p_exact| < 3 %"),
        ("b_rigid_sphere_scattering", "b. rigid sphere: |sigma_BEM - sigma_Mie| / sigma_Mie < 5 %"),
        ("c_chief_irregular_frequency", "c. CHIEF: no spurious peak near interior eigenvalues"),
        ("d_ices209_third_octave", "d. ICES 209 bands: |dSPL| < 2 dB vs reference spectrum"),
        ("energy_balance", "extra: surface power = far-field power (2 %)"),
    ):
        entry = crit[key]
        if key == "a_pulsating_sphere":
            measured = f"{entry['measured_max_relative_error']*100:.3f} %"
            limit = "< 3 %"
        elif key == "b_rigid_sphere_scattering":
            measured = f"{entry['measured_max_relative_error']*100:.3f} %"
            limit = "< 5 %"
        elif key == "c_chief_irregular_frequency":
            measured = (
                f"bare {entry['measured_bare_max']*100:.1f} % -> CHIEF "
                f"{entry['measured_chief_max']*100:.1f} %"
            )
            limit = "no peak"
        elif key == "d_ices209_third_octave":
            measured = f"{entry['measured_max_abs_delta_db']:.3f} dB"
            limit = "< 2 dB"
        else:
            measured = f"{entry['measured_max_relative_difference']*100:.3f} %"
            limit = "< 2 %"
        rows.append([label, measured, limit, entry["verdict"]])
    blocks.append("## Verdict summary\n\n" + md_table(["criterion", "measured", "limit", "verdict"], rows))

    # (a) ------------------------------------------------------------------
    rows = []
    for row in crit["a_pulsating_sphere"]["rows"]:
        rows.append(
            [
                str(row["panel_count"]),
                fmt(row["ka"], 2),
                fmt(row["frequency_hz"], 2),
                f"{row['max_relative_pressure_error']*100:.3f} %",
                fmt(row["boundary_relative_residual"], 2),
                fmt(row["chief_relative_residual"], 2),
            ]
        )
    blocks.append(
        "## (a) Pulsating sphere radiation, a = 1 m, v_n = 1 m/s\n\n"
        + md_table(
            ["panels", "ka", "f (Hz)", "max rel. pressure error", "boundary residual", "CHIEF residual"],
            rows,
        )
    )

    # (b) ------------------------------------------------------------------
    rows = []
    for row in crit["b_rigid_sphere_scattering"]["rows"]:
        rows.append(
            [
                str(row["panel_count"]),
                fmt(row["ka"], 2),
                fmt(row["sigma_bem"], 5),
                fmt(row["sigma_mie"], 5),
                f"{row['relative_error']*100:.3f} %",
                fmt(row["sigma_over_pi_a2"], 4),
            ]
        )
    blocks.append(
        "## (b) Rigid sphere scattering cross-section, a = 1 m\n\n"
        + md_table(["panels", "ka", "sigma BEM (m^2)", "sigma Mie (m^2)", "rel. error", "sigma/(pi a^2)"], rows)
    )

    # (c) ------------------------------------------------------------------
    rows = []
    for name, row in crit["c_chief_irregular_frequency"]["windows"].items():
        peaks = max(zip(row["bare_relative_error"], row["chief_relative_error"]), key=lambda p: p[0])
        rows.append(
            [
                name,
                fmt(row["ka"][0], 2),
                fmt(row["ka"][-1], 2),
                f"{row['bare_max']*100:.2f} %",
                f"{row['chief_max']*100:.2f} %",
                f"{row['suppression_ratio']:.1f}x",
                f"{peaks[0]*100:.1f}% -> {peaks[1]*100:.1f}%",
            ]
        )
    blocks.append(
        "## (c) CHIEF irregular-frequency suppression (320-panel sphere)\n\n"
        f"interior Dirichlet eigenvalues of the unit sphere (ka): "
        + ", ".join(f"{value:.4f}" for value in crit["c_chief_irregular_frequency"]["sphere_mode_wavenumbers_ka"])
        + "\n\n"
        + md_table(
            ["window", "ka from", "ka to", "bare max", "CHIEF max", "suppression", "worst point"],
            rows,
        )
    )

    # (d) ------------------------------------------------------------------
    rows = []
    for row in crit["d_ices209_third_octave"]["rows"]:
        rows.append(
            [
                fmt(row["centre_hz"], 1),
                fmt(row["ka"], 3),
                str(row["panel_count"]),
                f"{row['spl_db_bem']:.3f}",
                f"{row['spl_db_analytic']:.3f}",
                f"{row['delta_db']:+.3f}",
            ]
        )
    blocks.append(
        "## (d) ICES 209 one-third-octave spectrum\n\n"
        f"reference: {crit['d_ices209_third_octave']['reference']}\n\n"
        + md_table(
            ["fc (Hz)", "ka", "panels", "SPL BEM (dB)", "SPL analytic (dB)", "delta (dB)"], rows
        )
    )

    # energy balance -------------------------------------------------------
    rows = [
        [
            fmt(row["ka"], 2),
            fmt(row["power_surface_w"], 4),
            fmt(row["power_far_field_w"], 4),
            f"{row['relative_difference']*100:.3f} %",
        ]
        for row in crit["energy_balance"]["rows"]
    ]
    blocks.append(
        "## BEM energy balance (no analytic reference)\n\n"
        + md_table(["ka", "P from surface intensity (W)", "P from far field (W)", "difference"], rows)
    )

    # hull -----------------------------------------------------------------
    geometry = hull["geometry"]
    blocks.append(
        "## Hull case: KCS closed body\n\n"
        + md_table(
            ["quantity", "value"],
            [
                ["panels", str(geometry["panel_count"])],
                ["displacement volume (m^3)", f"{geometry['volume_m3']:.1f}"],
                ["wetted + lid area (m^2)", f"{geometry['surface_area_m2']:.1f}"],
                ["block coefficient", f"{geometry['block_coefficient']:.4f}"],
                ["KCS anchor Cb", f"{geometry['kcs_anchor_block_coefficient']:.4f}"],
                ["L / B / T (m)", f"{geometry['length_m']:.0f} / {geometry['breadth_m']:.1f} / {geometry['draft_m']:.1f}"],
                ["excitation", "Gaussian machinery patch, 1 mm/s peak, sigma = 13.8 m, x = -34.5 m"],
                ["receiver", "(0, 100, -25) m"],
            ],
        )
    )

    rows = []
    for row in hull["rows"]:
        if row["verdict"] == "COMPUTED":
            rows.append(
                [
                    fmt(row["centre_hz"], 1),
                    f"{row['spl_db_re_1upa']:.2f}",
                    f"{row['monopole_surrogate_spl_db']:.2f}",
                    f"{row['self_scattering_delta_db']:+.2f}",
                    fmt(row["radiated_power_w"], 3),
                    f"{row['panels_per_wavelength']:.3f}",
                ]
            )
        else:
            rows.append(
                [
                    fmt(row["centre_hz"], 1),
                    "WITHHELD",
                    "-",
                    "-",
                    "-",
                    f"{row['panels_per_wavelength']:.3f} (needs ~{row['required_panel_count']} panels)",
                ]
            )
    blocks.append(
        "## Hull band table (ICES 209)\n\n"
        + md_table(
            ["fc (Hz)", "SPL BEM (dB re 1 uPa)", "monopole surrogate (dB)", "self-scattering (dB)",
             "radiated power (W)", "h/lambda"],
            rows,
        )
    )

    rows = [
        [str(row["panel_count"]), f"{row['volume_m3']:.1f}", f"{row['spl_db']:.3f}"]
        for row in hull["mesh_convergence_25hz"]
    ]
    blocks.append(
        "## Mesh convergence at 25 Hz\n\n" + md_table(["panels", "volume (m^3)", "SPL at receiver (dB)"], rows)
    )

    if hull.get("mesh_convergence"):
        rows = []
        for frequency, entries in hull["mesh_convergence"].items():
            for entry in entries:
                rows.append(
                    [
                        f"{float(frequency):g}",
                        str(entry["panel_count"]),
                        f"{entry['panels_per_wavelength']:.3f}",
                        f"{entry['spl_db']:.3f}",
                    ]
                )
        spread = {}
        for frequency, entries in hull["mesh_convergence"].items():
            levels = [entry["spl_db"] for entry in entries]
            spread[frequency] = max(levels) - min(levels)
        blocks.append(
            "## Hull mesh convergence (second pass)\n\n"
            + md_table(["fc (Hz)", "panels", "h/lambda", "SPL (dB)"], rows)
            + "\n\nmax-min per frequency: "
            + ", ".join(f"{key} Hz: {value:.3f} dB" for key, value in spread.items())
        )

    if hull.get("energy_balance"):
        rows = []
        for frequency, entries in hull["energy_balance"].items():
            for panels, entry in entries.items():
                rows.append(
                    [
                        f"{float(frequency):g}",
                        panels,
                        fmt(entry["power_surface_w"], 4),
                        fmt(entry["power_far_field_w"], 4),
                        f"{entry['relative_difference']*100:.3f} %",
                    ]
                )
        blocks.append(
            "## Hull energy balance\n\n"
            + md_table(["fc (Hz)", "panels", "P surface (W)", "P far field (W)", "difference"], rows)
        )

    rows = []
    for key, entry in hull["directivity"].items():
        rows.append(
            [
                key,
                f"{min(entry['spl_db']):.2f}",
                f"{max(entry['spl_db']):.2f}",
                f"{entry['spl_range_db']:.2f}",
                f"{entry['fore_aft_ratio_db']:+.2f}",
            ]
        )
    blocks.append(
        "## Horizontal directivity (r = 100 m, z = -25 m)\n\n"
        + md_table(["fc (Hz)", "min SPL", "max SPL", "range (dB)", "fore-aft (dB)"], rows)
    )

    artifacts = criteria["artifacts"]
    rows = [[name, f"`{digest[:16]}...`"] for name, digest in artifacts.items()]
    blocks.append("## Audit chain (sha256 of every artifact)\n\n" + md_table(["artifact", "sha256"], rows))

    # ---- pinned headline numbers, injected into the narrative doc ---------
    computed = [row for row in hull["rows"] if row["verdict"] == "COMPUTED"]
    pins = {
        "a_max_relative_error_pct": round(crit["a_pulsating_sphere"]["measured_max_relative_error"] * 100, 3),
        "b_max_relative_error_pct": round(crit["b_rigid_sphere_scattering"]["measured_max_relative_error"] * 100, 3),
        "c_bare_max_pct": round(crit["c_chief_irregular_frequency"]["measured_bare_max"] * 100, 2),
        "c_chief_max_pct": round(crit["c_chief_irregular_frequency"]["measured_chief_max"] * 100, 2),
        "d_max_abs_delta_db": round(crit["d_ices209_third_octave"]["measured_max_abs_delta_db"], 3),
        "energy_balance_max_pct": round(crit["energy_balance"]["measured_max_relative_difference"] * 100, 3),
        "hull_panels": int(hull["geometry"]["panel_count"]),
        "hull_block_coefficient": round(hull["geometry"]["block_coefficient"], 4),
        "hull_max_resolved_frequency_hz": max(row["centre_hz"] for row in computed),
        "hull_max_self_scattering_delta_db": round(
            max(row["self_scattering_delta_db"] for row in computed), 2
        ),
        "hull_bands_computed": len(computed),
        "hull_bands_withheld": sum(1 for row in hull["rows"] if row["verdict"] != "COMPUTED"),
    }
    (OUT / "pins.json").write_text(json.dumps(pins, indent=2) + "\n")

    doc = REPO / "docs" / "acoustic3d_helmholtz.md"
    text = doc.read_text()
    begin = "<!-- pins:begin -->"
    end = "<!-- pins:end -->"
    if begin in text and end in text:
        block = begin + "\n```json\n" + json.dumps(pins, indent=2) + "\n```\n" + end
        text = text[: text.index(begin)] + block + text[text.index(end) + len(end):]
        doc.write_text(text)
        print("pins injected into docs/acoustic3d_helmholtz.md")

    body = "\n\n".join(blocks) + "\n"
    (OUT / "results_tables.md").write_text(body)
    print("wrote results_tables.md", len(body), "chars")

    # ---- platform report payload -----------------------------------------
    figures = []
    for figure in criteria["figures"]:
        path = REPO / figure["path"]
        figures.append(
            {
                "name": figure["name"],
                "caption": figure["caption"],
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    headline = (
        f"三维无障板外域 Helmholtz 声学 BEM（TensorBEM `tensorbem.acoustic3d`）：复波数层势 + 精确立体角双层位势 + "
        f"CHIEF 消不规则频率 + 辐射/散射分解 + ICES 209 三分之一倍频带出口。判据：脉动球辐射 {crit['a_pulsating_sphere']['measured_max_relative_error']*100:.2f}% (<3%)、"
        f"刚性球散射截面 vs Mie {crit['b_rigid_sphere_scattering']['measured_max_relative_error']*100:.2f}% (<5%)、"
        f"CHIEF 把 ka≈3.2 的虚假峰值 {crit['c_chief_irregular_frequency']['measured_bare_max']*100:.0f}%→{crit['c_chief_irregular_frequency']['measured_chief_max']*100:.1f}%、"
        f"31 个三分之一倍频带 vs 解析参考 {crit['d_ices209_third_octave']['measured_max_abs_delta_db']:.2f} dB (<2 dB)、"
        f"能量守恒 {crit['energy_balance']['measured_max_relative_difference']*100:.2f}%。KCS 闭合船体 1970 面板算到 "
        f"{max(row['centre_hz'] for row in hull['rows'] if row['verdict']=='COMPUTED'):.0f} Hz，自散射相对无散射代理最多改变 "
        f"{max(row['self_scattering_delta_db'] for row in hull['rows'] if row['verdict']=='COMPUTED'):+.1f} dB。"
    )
    payload = {
        "title": "三维无障板外域 Helmholtz 声学 BEM：真实曲面船体的辐射-散射与 ICES 209 三分之一倍频带出口 (P284W3-N03)",
        "summary": headline,
        "scenario_key": "ship-acoustics",
        "mission_id": 13961,
        "tags": ["TensorBEM", "acoustics", "Helmholtz", "BEM", "CHIEF", "ICES209", "P284W3-N03"],
        "markdown": body,
        "figures": figures,
    }
    (OUT / "report_payload.json").write_text(json.dumps(payload))
    print("wrote report_payload.json with", len(figures), "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())