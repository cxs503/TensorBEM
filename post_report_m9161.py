"""m9161: publish the paper-level progress report for node p106w2-n04 (TensorBEM multihull).

Reads the generated record + report and posts markdown + base64 figures.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER")
PASS = os.environ.get("TENSOR_AGENT_PASS")
ROOT = Path("/root/tp-agent/work/m9161")
OUT = ROOT / "benchmarks" / "multibody_frequency_dependence_3d"
MISSION_ID = 9161


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=240) as response:
        return json.loads(response.read().decode())


def figure(name: str, path: Path, caption: str) -> dict:
    return {
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(path.read_bytes()).decode(),
    }


def main() -> None:
    criteria = json.loads((OUT / "results.json").read_text())["criteria"]
    md = (OUT / "report.md").read_text()

    key = {
        k: (round(v["value"], 4) if isinstance(v.get("value"), (int, float)) else v.get("value"))
        for k, v in criteria.items()
        if isinstance(v, dict) and k in {
            "zero_frequency_damping", "zero_frequency_added_mass",
            "zero_frequency_added_mass_fine", "zero_frequency_added_mass_extrapolated",
            "infinite_frequency_added_mass", "infinite_frequency_added_mass_catamaran",
            "infinite_frequency_added_mass_catamaran_smooth",
            "infinite_frequency_from_kramers_kronig",
            "infinite_frequency_from_kramers_kronig_dense",
            "energy_non_negative_damping", "energy_far_field_agreement",
            "distance_decay_far_field", "distance_decay_far_field_kd_gated",
        }
    }

    front = "\n\n".join(
        [
            "## 判据数字（程序化注入，来自 results.json）",
            "```json\n" + json.dumps(key, indent=1, ensure_ascii=False) + "\n```",
            f"`all_pass_original` = {criteria['all_pass_original']}, "
            f"`all_pass` = {criteria['all_pass']}；"
            f"原始判据失败：{criteria.get('gated_original') and [k for k in criteria['gated_original'] if not criteria[k]['pass']] or []}；"
            f"新增判据失败：{[k for k in criteria.get('gated_added', []) if not criteria[k]['pass']]}。",
        ]
    )

    figures = []
    for name, fname, caption in [
        ("fig_resonance", "resonance_band.png",
         "图1 细扫（Δω/ωn=0.025，nx=12）双体间隙 0.5B 的附加质量/阻尼：垂荡在 ω/ωn≈1.1-1.3 出现窄带共振"
         "（阻尼峰 + 附加质量过零变负），金色区为判据检出的频带；这是首轮 1/ω² 外推得到非物理值的根因"),
        ("fig_distance", "distance_decay.png",
         "图2 (a) 干扰因子 σ_heave 随 d/L（点线 d/L=2）；(b) 同一批点的 |σ_A-1| 随无量纲间距 kd："
         "3D 辐射波按 (kr)^-1/2 衰减，因此远场退化必须用 kd 而非 d/L 设门"),
        ("fig_asymptotes", "asymptotes.png",
         "图3 已解析频段内单/双/三体附加质量（横荡/垂荡/纵摇）；未解析行已剔除"),
        ("fig_kk", "kramers_kronig.png",
         "图4 单体 B33/ω² 积分核随 ω：Kramers-Kronig 带内矩决定 A(0)-A(∞)，梯子上限由网格决定"),
        ("fig_interference", "interference_frequency.png",
         "图5 双体干扰因子 σ_A(ω) 三个自由度"),
    ]:
        path = OUT / fname
        if path.exists():
            figures.append(figure(name, path, caption))

    payload = {
        "title": "TensorBEM 多体水动力系数频率依赖性：双体窄带共振识别 + 无限/零频渐近判据重测（p106w2-n04）",
        "summary": (
            "Wigley 双/三体（间距 0.5B，ω/ωn=0.1-2.5）附加质量与阻尼频率扫描；"
            "细扫识别出双体窄带共振（阻尼峰 + 附加质量过零），并用 CHIEF 移除/网格收敛/能量恒等式/间隙扫描"
            "四条独立证据排除 irregular-frequency 与近奇异积分伪迹；零频极限改用匹配网格+Richardson，"
            "无限频改用带内 Kramers-Kronig（梯子延伸至 ω/ωn=2.0 已解析），远场退化判据改以 kd 设门。"
            "原始判据逐条保留（honest FAIL 不覆盖），新增判据分列。"
        ),
        "scenario_key": "tensorbem_multihull_frequency_dependence",
        "mission_id": MISSION_ID,
        "tags": ["TensorBEM", "多体水动力", "附加质量", "辐射阻尼", "Kramers-Kronig",
                 "窄带共振", "p106w2-n04", "catamaran", "trimaran"],
        "markdown": front + "\n\n---\n\n" + md,
        "figures": figures,
    }
    login = post("/auth/login", {"username": USER, "password": PASS})
    token = login.get("access_token") or login.get("token")
    assert token, f"no token: {sorted(login)}"
    result = post("/tensor/reports", payload, token)
    print(json.dumps(result)[:400])


if __name__ == "__main__":
    main()