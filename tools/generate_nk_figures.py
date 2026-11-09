"""Generate figures for the Neumann-Kelvin verification report."""

import math
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.neumann_kelvin import (
    KELVIN_ANGLE_DEG,
    compute_kelvin_wave_pattern,
    froude_number,
    green_function_nk,
    michell_thin_ship_limit,
    speed_for_froude,
    verify_kelvin_angle,
)
from tensorbem.wigley import michell_wave_resistance

RHO = 1025.0
G = 9.80665
L, B, T = 2.0, 0.2, 0.125


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_kelvin_wave_pattern_figure() -> tuple[str, str]:
    """Generate Kelvin wave pattern figure."""
    pattern = compute_kelvin_wave_pattern(1.0, nx=300, ny=300,
                                          x_range=(-2.0, 12.0),
                                          y_range=(-10.0, 10.0))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot wave elevation
    x = pattern.x[:, 0] * G / 1.0**2  # Convert to physical units
    y = pattern.y[0, :] * G / 1.0**2
    eta = pattern.eta

    # Mask ahead of ship
    mask = x > 0
    X, Y = np.meshgrid(x, y, indexing="ij")
    eta_masked = np.where(mask, eta, np.nan)

    im = ax.pcolormesh(X, Y, eta_masked.T, cmap="RdBu_r", shading="auto",
                       vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, label="Wave elevation (normalized)")

    # Draw Kelvin wedge
    theta_k = math.asin(1.0 / 3.0)
    x_wedge = np.linspace(0, 12 * G, 100)
    y_wedge = x_wedge * math.tan(theta_k)
    ax.plot(x_wedge, y_wedge, "k--", linewidth=1.5, label=f"Kelvin wedge ({KELVIN_ANGLE_DEG:.1f}°)")
    ax.plot(x_wedge, -y_wedge, "k--", linewidth=1.5)

    # Ship position
    ax.plot(0, 0, "k^", markersize=12, label="Ship")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Kelvin Wave Pattern (Stationary Phase Approximation)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64, "Kelvin wave pattern with theoretical wedge angle"


def generate_cw_froude_figure() -> tuple[str, str]:
    """Generate Cw vs Froude number figure."""
    fr_values = np.linspace(0.15, 0.55, 50)
    cw_values = []

    for fn in fr_values:
        U = speed_for_froude(fn, L, G)
        Rw = michell_wave_resistance(L, B, T, U, RHO, G)
        S_ref = L * T
        Cw = Rw / (0.5 * RHO * U**2 * S_ref)
        cw_values.append(Cw)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fr_values, cw_values, "b-", linewidth=2, label="Michell thin-ship theory")

    # Mark specific Froude numbers
    fr_marks = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for fn in fr_marks:
        U = speed_for_froude(fn, L, G)
        Rw = michell_wave_resistance(L, B, T, U, RHO, G)
        Cw = Rw / (0.5 * RHO * U**2 * S_ref)
        ax.plot(fn, Cw, "ro", markersize=6)

    ax.set_xlabel("Froude Number (Fn = U/√(gL))")
    ax.set_ylabel("Wave Resistance Coefficient (Cw)")
    ax.set_title(f"Wigley III Hull: Cw vs Froude Number\n(L={L}m, B={B}m, T={T}m, Cb=4/9)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64, "Wigley III wave resistance coefficient vs Froude number"


def generate_green_function_figure() -> tuple[str, str]:
    """Generate Green function verification figure."""
    kappa = G / 1.0**2
    dz_values = np.linspace(0.05, 2.0, 50)

    g_values = []
    rankine_values = []

    for dz in dz_values:
        target = np.array([0.0, 0.0, -0.1])
        source = np.array([0.0, 0.0, -0.1 - dz])
        g = green_function_nk(target, source, kappa)
        rankine = 1.0 / (4 * math.pi * dz) + 1.0 / (4 * math.pi * (0.2 + dz))
        g_values.append(g)
        rankine_values.append(rankine)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(dz_values, g_values, "b-", linewidth=2, label="G = G_R + G_W")
    ax1.plot(dz_values, rankine_values, "r--", linewidth=1.5, label="G_R (Rankine only)")
    ax1.set_xlabel("Vertical separation dz (m)")
    ax1.set_ylabel("Green function value")
    ax1.set_title("Green Function vs Distance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ratio plot
    ratio = np.array(g_values) / np.array(rankine_values)
    ax2.plot(dz_values, ratio, "g-", linewidth=2)
    ax2.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Vertical separation dz (m)")
    ax2.set_ylabel("G / G_R ratio")
    ax2.set_title("Wave Part Contribution\n(ratio > 1 means G_W adds constructively)")
    ax2.grid(True, alpha=0.3)

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64, "Green function decomposition: Rankine + wave parts"


def generate_all_figures(output_dir: str = "artifacts/neumann_kelvin_verification") -> list[dict]:
    """Generate all figures and return metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    figures = []

    # Figure 1: Kelvin wave pattern
    b64, caption = generate_kelvin_wave_pattern_figure()
    fig_path = out / "kelvin_wave_pattern.png"
    with open(fig_path, "wb") as f:
        f.write(base64.b64decode(b64))
    figures.append({"name": "kelvin_wave_pattern", "caption": caption, "png_base64": b64})

    # Figure 2: Cw vs Fr
    b64, caption = generate_cw_froude_figure()
    fig_path = out / "cw_vs_froude.png"
    with open(fig_path, "wb") as f:
        f.write(base64.b64decode(b64))
    figures.append({"name": "cw_vs_froude", "caption": caption, "png_base64": b64})

    # Figure 3: Green function
    b64, caption = generate_green_function_figure()
    fig_path = out / "green_function.png"
    with open(fig_path, "wb") as f:
        f.write(base64.b64decode(b64))
    figures.append({"name": "green_function", "caption": caption, "png_base64": b64})

    print(f"Generated {len(figures)} figures in {out}")
    return figures


if __name__ == "__main__":
    generate_all_figures()
