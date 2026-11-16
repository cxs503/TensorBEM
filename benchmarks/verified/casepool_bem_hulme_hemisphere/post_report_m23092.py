#!/usr/bin/env python3
"""Publish the lid-referee adjudication report to the platform API."""
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path("/root/tp-agent/work/m23092/benchmarks/verified/casepool_bem_hulme_hemisphere")
API = os.environ["TENSOR_API_BASE"].rstrip("/")

doc = (HERE / "referee_lid_adjudication.md").read_text()
rad = json.loads((HERE / "referee_lid_adjudication.json").read_text())
exc = json.loads((HERE / "referee_lid_excitation.json").read_text())

unresolved = [r for r in rad["rows"] if r["role"] == "unresolved_in_xval"]
cross_rad = max(max(r["cross_code_tensorbem_vs_lid"]["per_quantity"].values()) for r in unresolved)
cross_exc = max(max(r["cross_code_vs_lid"].values()) for r in exc["rows"] if "cross_code_vs_lid" in r)
lid_rad = max(r["lid_referee_4096"]["worst_relerr"] for r in rad["rows"])
lid_exc = exc["gates"]["E2_lid_referee_excitation_accuracy"]["measured"]["worst"]
open_258 = next(r for r in rad["rows"] if abs(r["ka"] - 2.58) < 1e-9)["open_referee_4096"]["worst_relerr"]
open_fz = next(r for r in exc["rows"] if abs(r["ka"] - 2.58) < 1e-9)["open_error"]["Fz"]

summary = (
    "算例池 BEM-HULME-HEMISPHERE 的验证欠账：radiation_capytaine_band_xval 把 "
    f"ka=2.58/3.0/4.0 三格记为 referee_unresolved（裁判 capytaine 自身有内部 Dirichlet "
    f"不规则频率，自漂移 37%）。用 lid 闭合后的裁判重新裁定：三格两码互差辐射 {100*cross_rad:.2f}%、"
    f"激励 {100*cross_exc:.2f}%，全部落在 3% 以内，裁定从「无解」变成 agree；"
    f"开口裁判在 ka=2.58 的误差为辐射 {100*open_258:.1f}% / 激励 |Fz| {100*open_fz:.1f}%，"
    "同一枚盖一并治好。lid 裁判精度 vs Hulme 最坏 "
    f"{100*lid_rad:.2f}%（辐射 4096）/ {100*lid_exc:.2f}%（激励 4096）。"
    "诚实记账：G2（裁判 1024→4096 自漂移）在 ka=4 达 5.50%，判 FAIL，故裁判只用 4096 档。"
)

figures = []
for name, caption in (
    ("fig_referee_lid_error_vs_ka.png",
     "左：辐射四量最坏误差 vs ka（开口裁判 / lid 裁判 / TensorBEM），右：激励两量同一比较；灰带为原未裁定区，虚线 5% 容差"),
    ("fig_referee_lid_b33_pole.png",
     "B33 误差随 ka 扫频：开口裁判 1024/4096 两档的极点（2.54 与 2.58 之间变号）与加盖后的平坦曲线"),
    ("fig_referee_lid_crosscode.png",
     "三格上 TensorBEM 与 lid 裁判的六量互差柱状图（辐射四量 + 激励两量），虚线 5% 容差"),
):
    path = HERE / name
    figures.append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(path.read_bytes()).decode(),
    })

report = {
    "title": "CASE-BEM-HULME-HEMISPHERE：用 lid 闭合裁判裁定三格 referee_unresolved",
    "summary": summary,
    "scenario_key": "ship_seakeeping_bem_3d",
    "mission_id": 23092,
    "tags": ["TensorBEM", "BEM", "case-pool", "verification-debt", "irregular-frequency",
             "Hulme", "capytaine", "referee", "CASE-BEM-HULME-HEMISPHERE"],
    "markdown": doc,
    "figures": figures,
}
Path("/tmp/report.json").write_text(json.dumps(report))
print("report bytes:", Path("/tmp/report.json").stat().st_size)

login = subprocess.run(
    ["curl", "-s", "-X", "POST", f"{API}/auth/login",
     "-H", "Content-Type: application/json",
     "-d", json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]})],
    capture_output=True, text=True, check=True)
token = json.loads(login.stdout)["access_token"]

post = subprocess.run(
    ["curl", "-s", "-X", "POST", f"{API}/tensor/reports",
     "-H", f"Authorization: Bearer {token}",
     "-H", "Content-Type: application/json",
     "-d", "@/tmp/report.json"],
    capture_output=True, text=True, check=True)
print(post.stdout[:600])