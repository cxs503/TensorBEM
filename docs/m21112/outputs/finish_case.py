import json, sys, time
from pathlib import Path
sys.path.insert(0, "src")
from tensorbem.benchmarks import case_cylinder_stream as case

outdir = Path("benchmarks/verified/case_bem_laplace_cylinder")
art = json.loads((outdir / "results.json").read_text())
started = time.time()
near = art["near_wall"]
near["azimuth"] = case.attach_azimuth_study(near["window"]["levels"], ladder=near["ladder"])
art["criteria"] = case.criteria(art)
failed = [item["name"] for item in art["criteria"] if not item["ok"]]
art["acceptance"] = {
    "passed": not failed,
    "failed_checks": failed,
    "tolerance": case.TOLERANCE,
    "order_min": case.ORDER_MIN,
    "n_checks": len(art["criteria"]),
}
art["overall"] = "PASS" if not failed else "FAIL"
art["wall_seconds"] = float(art["wall_seconds"]) + round(time.time() - started, 1)
(outdir / "results.json").write_text(json.dumps(art, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
rendered = case.render_markdown(art) + "\n"
(outdir / "report.md").write_text(rendered, encoding="utf-8")
Path("docs/CASE-BEM-LAPLACE-CYLINDER.md").write_text(rendered, encoding="utf-8")
case.write_case_card(art, outdir / "case.json")
figures = outdir / "figures"
case.plot_near_wall(art, figures / "near_wall.png")
az = art["near_wall"]["azimuth"]
print(json.dumps({
    "verdict": art["overall"],
    "n_checks": art["acceptance"]["n_checks"],
    "failed": failed,
    "window_worst": az["window"]["worst_over_levels"],
    "exclusion_penalty": az["exclusion"]["min_azimuth_penalty"],
    "exclusion_worst": az["exclusion"]["worst_over_levels"],
    "linear_ratio": az["linear_at_window"]["min_ratio"],
    "windows": az["windows_including_azimuth"],
    "frame_diff": az["midpoint_cross_check"]["max_relative_difference"],
    "doc_bytes": len(rendered),
}, indent=1))
