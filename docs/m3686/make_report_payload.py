"""Build the /tensor/reports payload of the m3686 merge (figures downscaled).

Run:  python3 docs/m3686/make_report_payload.py   ->  /tmp/report.json
"""
import base64
import io
import json

from PIL import Image

SPEC = [
    ("fig_coverage", "docs/m3686/figures/fig_coverage.png",
     "图1 三维自由液面注册表覆盖：m3181 / main(m3150) / 合并后逐表对比（合并后 66 格 = main 62 + m3181 独有 4）。"),
    ("fig_deviations", "docs/m3686/figures/fig_deviations.png",
     "图2 合并后 65 个 value 判据行的相对偏差（对数纵轴），全部低于 1.5e-2 容差与 3% 验收线；worst = 3.79e-3。"),
    ("fig_convergence", "docs/m3686/figures/fig_convergence.png",
     "图3 实测收敛阶：Hulme 半球与 Wang-Linton 潜没球各模态观测阶均落在二阶附近（点线 = 2）。"),
    ("fig_b11_convention", "docs/m3686/figures/fig_b11_convention.png",
     "图4 唯一真实口径冲突：B11 观测阶（m3181 理想 √N 折算 1.62/1.88 vs main 实测平均棱长比 1.8021/1.9377）。"),
]

SUMMARY = (
    "TensorBEM PR #128 分支对 main 冲突（mergeable=false）。根因：同一批三维自由液面格子被两条独立分支"
    "以两套键名各注册一遍（main/m3150: hulme_hemisphere|hulme_chief|submerged_sphere + sub<N>_ + "
    "fast/standard；m3181: hulme3d*|wang_linton3d + level<N>_ + heavy）。解冲：规范=main 命名/plumbing/层级，"
    "m3181 的 63 格逐格双射映射并入；m3181 独有 4 格（exact A33'/B33'、CHIEF 点数、20 面元机器零转动耦合 "
    "upper 行）与 main 独有 3 格（Haskind 互易 x/z、近远场阻尼闭合）全部保留；清除自动合并造成的重复 "
    "_chief_rows 死代码；docs §4.12 补记 B11 观测阶口径冲突（1.62/1.88 vs 1.8021/1.9377）与层级变化。"
    "验证：66/66 三维格子 PASS（value 行最大相对偏差 3.79e-3），tests/test_declared_numbers.py 合并后 "
    "17 passed/3 skipped vs main 基线 14 passed/3 skipped（0 failed）。SHA：旧 agent/m3181=55db3976，"
    "新=71c870c（合并提交 a891e53，parents b420b2e + 55db3976）。沙箱安全策略拦截了远端写操作，"
    "本地分支 ref 已推进到新 SHA，等待平台自动推送后由 merge_governor 复验。"
)


def main() -> int:
    figs = []
    for name, path, caption in SPEC:
        data = open(path, "rb").read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((620, 620))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
        figs.append(
            {"name": name, "caption": caption, "png_base64": base64.b64encode(data).decode()}
        )
        print(name, len(data))
    markdown = (
        open("docs/m3686/merge_report.md", encoding="utf-8")
        .read()
        .replace("](figures/", "](docs/m3686/figures/")
    )
    payload = {
        "title": "PR #128 / agent/m3181 治愈合并：三维自由液面注册表语义并集（66/66 PASS，测试不低于 main 基线）",
        "summary": SUMMARY,
        "scenario_key": "tensorbem_merge_cure",
        "mission_id": 3686,
        "tags": ["TensorBEM", "merge-cure", "PR128", "agent/m3181",
                 "declared-numbers", "BEM", "free-surface", "CHIEF"],
        "markdown": markdown,
        "figures": figs,
    }
    blob = json.dumps(payload)
    open("/tmp/report.json", "w").write(blob)
    print("payload bytes:", len(blob))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())