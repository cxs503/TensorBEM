from pathlib import Path

path = Path("src/tensorbem/benchmarks/case_cylinder_stream.py")
text = path.read_text()

# ------------------------------------------------------------------- figure
old = '''    near = artifact["near_wall"]
    fig, axes = plt.subplots(figsize=(6.4, 4.4))'''
assert text.count(old) == 1
new = '''    near = artifact["near_wall"]
    fig, (axes, azimuth_axes) = plt.subplots(1, 2, figsize=(11.6, 4.4))'''
text = text.replace(old, new, 1)

old = '''    axes.set_title("exterior field: near-wall validity window")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)'''
assert text.count(old) == 1
new = '''    axes.set_title("midpoint rays: validity window")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=7)

    azimuth = near["azimuth"]
    for level in azimuth["levels"]:
        clearance = np.array(
            [row["clearance_over_h_max"] for row in level["sweep"]]
        )
        penalty = np.array([row["azimuth_penalty"] for row in level["sweep"]])
        penalty_linear = np.array(
            [row["azimuth_penalty_linear"] for row in level["sweep"]]
        )
        azimuth_axes.semilogx(clearance, penalty, "o-", ms=4.0, label=f"N={level['panels']}")
        azimuth_axes.semilogx(
            clearance,
            penalty_linear,
            "s--",
            ms=3.5,
            lw=1.0,
            alpha=0.85,
            color=azimuth_axes.lines[-1].get_color(),
            label=f"N={level['panels']} linear",
        )
    azimuth_axes.axvline(
        near["window"]["clearance_over_h_max"],
        color="tab:green",
        ls="--",
        lw=1.0,
    )
    azimuth_axes.axvline(
        near["exclusion"]["clearance_over_h_max"],
        color="tab:red",
        ls=":",
        lw=1.0,
    )
    azimuth_axes.set_xlabel(r"$\\delta/h_{max}$")
    azimuth_axes.set_ylabel("worst azimuth error / midpoint error")
    azimuth_axes.set_title("across one panel: azimuth penalty")
    azimuth_axes.grid(True, which="both", alpha=0.3)
    azimuth_axes.legend(fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)'''
text = text.replace(old, new, 1)

# ----------------------------------------------------------- figure caption
old = '''        "`figures/near_wall.png`：近壁误差随 $\\\\delta/h_{max}$ 的变化（三层网格），"
        "竖线为适用窗与排除区边界。",'''
assert text.count(old) == 1, text.count(old)
new = '''        "`figures/near_wall.png`：左=近壁误差随 $\\\\delta/h_{max}$ 的变化（三层网格，实线常值基、"
        "虚线一阶矩基），竖线为适用窗与排除区边界；右=面内方位扫描的「最差方位/中点」误差比，"
        "比值 1 表示该层该间隙下中点就是最差方位。",'''
text = text.replace(old, new, 1)
path.write_text(text)
print("figure + caption patched")
