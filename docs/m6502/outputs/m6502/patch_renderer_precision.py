"""Fourth patch: precision of the error attribution sentence + Chinese labels.

Mission m6502.
"""
from pathlib import Path

p = Path("/root/tp-agent/work/m6502/src/tensorbem/benchmarks/tank_free_surface_verification.py")
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep(
    '''    attr_rows = "\\n".join(
        f"| {a['variant']} | {a['roll']:.4e} | {a['ratio']:.4f} | {a['rel_error']:.2e} |"
        for a in attr
    )''',
    '''    zh = {
        "centroidal_plane + frozen reference (this check)": "形心平面 + 冻结基线（本报告采用）",
        "origin-based plane + frozen reference (m6452 symptom)": "原点平面 + 冻结基线（m6452 症状）",
        "centroidal plane, no frozen reference": "形心平面，不扣冻结基线",
    }
    attr_rows = "\\n".join(
        f"| {zh.get(a['variant'], a['variant'])} | {a['roll']:.4e} | {a['ratio']:.4f} "
        f"| {a['rel_error']:.2e} |"
        for a in attr
    )''',
)

rep(
    '''最大相对误差 {energy_worst:.2e}（ε = {c8['fd_steps'][-1]:.0e}；ε 从 {c8['fd_steps'][0]:.0e} 减到
{c8['fd_steps'][-1]:.0e} 时误差按 O(ε²) 下降，来自 `sin φ ≈ φ` 这一阶截断，是能量路线自身的
线性化误差而非实现误差）。求积网格 {coarse_n} → {grid_n} 的曲率变化 {energy_grid:.2e}
（中点法对 `i_xx` 的二阶收敛）；`∫h dA = A·H` 的相对偏差 {energy_vol:.1e}（机器精度级）；
两种算差方式（先算总能量再相减 / 在求和内直接算差）一致到 1e-12 量级。''',
    '''最大相对误差 {energy_worst:.2e}。误差有两个来源，都可分辨：ε = {c8['fd_steps'][0]:.0e} 时由
`sin φ ≈ φ` 的一阶截断主导（理论值 `ε²/6 = {c8['fd_steps'][0] ** 2 / 6:.1e}`，实测
{c8['cases']['rectangle']['rows'][0]['relerr']['roll']:.1e}，同一量级）；ε → {c8['fd_steps'][-1]:.0e} 时截断项降到
{c8['fd_steps'][-1] ** 2 / 6:.1e}，误差落到网格求积底噪（{grid_n} 网格上 `i_xx` 的求积误差
{c2['Lshape'][-1]['i_xx_relerr']:.1e}，见 §3.2）。两者都不是实现误差。求积网格 {coarse_n} → {grid_n} 的曲率变化
{energy_grid:.2e}（中点法对 `i_xx` 的二阶收敛）；`∫h dA = A·H` 的相对偏差 {energy_vol:.1e}（机器精度级）；
两种算差方式（先算总能量再相减 / 在求和内直接算差）的相对一致度
{max(cse['max_cancellation_relerr'] for cse in c8['cases'].values()):.1e}，即没有抵消放大。''',
)

p.write_text(s)
print("patched precision paragraph")