"""Staged runner: the case exceeds one local shell slice, so run it in two."""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from tensorbem.benchmarks import case_cylinder_stream as case

CARRY = Path("outputs/artifact_stage.pkl")
OUTDIR = Path("benchmarks/verified/case_bem_laplace_cylinder")
DOC = Path("docs/CASE-BEM-LAPLACE-CYLINDER.md")

stage = sys.argv[1]
if stage == "1":
    started = time.time()
    real_attach, real_criteria = case.attach_azimuth_study, case.criteria
    case.attach_azimuth_study = lambda *args, **kwargs: {}
    case.criteria = lambda artifact: []
    artifact = case.run_case()
    case.attach_azimuth_study, case.criteria = real_attach, real_criteria
    CARRY.write_bytes(pickle.dumps(artifact))
    print(json.dumps({"stage": 1, "wall_seconds": round(time.time() - started, 1)}))
else:
    started = time.time()
    artifact = pickle.loads(CARRY.read_bytes())
    near = artifact["near_wall"]
    near["azimuth"] = case.attach_azimuth_study(
        near["window"]["levels"], ladder=near["ladder"]
    )
    artifact["criteria"] = case.criteria(artifact)
    failed = [item["name"] for item in artifact["criteria"] if not item["ok"]]
    artifact["acceptance"] = {
        "passed": not failed,
        "failed_checks": failed,
        "tolerance": case.TOLERANCE,
        "order_min": case.ORDER_MIN,
        "n_checks": len(artifact["criteria"]),
    }
    artifact["overall"] = "PASS" if not failed else "FAIL"
    artifact["wall_seconds"] = float(artifact["wall_seconds"]) + (
        time.time() - started
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rendered = case.render_markdown(artifact) + "\n"
    (OUTDIR / "report.md").write_text(rendered, encoding="utf-8")
    case.write_case_card(artifact, OUTDIR / "case.json")
    figures = OUTDIR / "figures"
    case.plot_field(artifact, figures / "field.png")
    case.plot_convergence(artifact, figures / "convergence.png")
    case.plot_near_wall(artifact, figures / "near_wall.png")
    case.plot_multipole(artifact, figures / "multipole.png")
    case.plot_circulation(artifact, figures / "circulation.png")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": 2,
                "wall_seconds": round(time.time() - started, 1),
                "verdict": artifact["overall"],
                "failed_checks": failed,
                "n_checks": len(artifact["criteria"]),
            },
            indent=1,
        )
    )
