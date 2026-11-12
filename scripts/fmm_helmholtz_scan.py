"""m16451 A/B evidence: Helmholtz single-layer kernel before/after the fix.

Nothing is re-implemented by hand: the pre-fix operator is loaded straight from
git (``PRE_FIX_REV:src/tensorbem/fmm.py``) and the fixed one from the working
tree, then both are compared against the analytic kernel

    Phi(x_i) = sum_j q_j exp(i k r_ij) / (4 pi r_ij)

on the same source/target cloud.  Writes ``docs/fmm_helmholtz_kernel/evidence.json``
and the figure; every number in the report comes from that JSON.

Cost: O(N*M) with N=12 targets, M=40 sources -- deliberately tiny (the host was
at load15 ~ 109 when this task ran; no compute-farm work here).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

PRE_FIX_REV = "daef8c3"  # main before the m16451 fix merged
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "fmm_helmholtz_kernel"

BANDS = [0.02, 0.05, 0.099, 0.5, 1.0, 3.0, 5.0, 10.0, 12.0, 40.0]


def _load_pre_fix():
    """Import the pre-fix ``tensorbem/fmm.py`` as a standalone module."""
    src = subprocess.run(
        ["git", "show", f"{PRE_FIX_REV}:src/tensorbem/fmm.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    tmp = pathlib.Path(tempfile.mkdtemp()) / "prefix_fmm.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("prefix_fmm", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prefix_fmm"] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


def _analytic(targets, sources, charges, k):
    diff = targets[:, None, :] - sources[None, :, :]
    r = np.maximum(np.linalg.norm(diff, axis=2), 1e-15)
    return np.sum(charges[None, :] * np.exp(1j * k * r) / (4.0 * np.pi * r), axis=1)


def _rel(num, ref):
    return float(np.max(np.abs(np.asarray(num) - ref)) / np.max(np.abs(ref)))


def main() -> None:
    from tensorbem.fmm import fmm_helmholtz_matvec, source_extent

    prefix = _load_pre_fix()

    rng = np.random.default_rng(0)
    sources = rng.normal(size=(40, 3)) * 0.1
    charges = rng.normal(size=40)
    targets = rng.normal(size=(12, 3)) * 1.5
    extent = source_extent(sources)

    rows = []
    for ka in BANDS:
        k = ka / extent
        ref = _analytic(targets, sources, charges, k)
        imag_fraction = float(np.max(np.abs(ref.imag)) / np.max(np.abs(ref)))

        old, old_mode = prefix.fmm_helmholtz_matvec(targets, sources, charges, k=k)
        new, new_mode = fmm_helmholtz_matvec(targets, sources, charges, k=k)

        rows.append({
            "ka": ka,
            "k": k,
            "mode_before": old_mode,
            "mode_after": new_mode,
            "relerr_before": _rel(old, ref),
            "relerr_after": _rel(new, ref),
            "dtype_before": str(np.asarray(old).dtype),
            "dtype_after": str(np.asarray(new).dtype),
            "imag_fraction_of_exact": imag_fraction,
        })

    # low-ka Laplace asymptotic: still reachable, but only by explicit opt-in
    optin = []
    for ka in (0.02, 0.05, 0.099):
        k = ka / extent
        ref = _analytic(targets, sources, charges, k)
        pot, mode = fmm_helmholtz_matvec(targets, sources, charges, k=k,
                                         low_ka_asymptotic=True)
        optin.append({"ka": ka, "mode": mode, "relerr": _rel(pot, ref),
                      "imag_zero": bool(np.all(pot.imag == 0.0))})

    # complex charges: the pre-fix path cast them to float64 (imag discarded)
    complex_charges = charges + 1j * rng.normal(size=40)
    ref_c = _analytic(targets, sources, complex_charges, 3.0)
    old_c, old_mode_c = prefix.fmm_helmholtz_matvec(targets, sources, complex_charges, k=3.0)
    new_c, new_mode_c = fmm_helmholtz_matvec(targets, sources, complex_charges, k=3.0)
    complex_case = {
        "mode_before": old_mode_c,
        "mode_after": new_mode_c,
        "dtype_before": str(np.asarray(old_c).dtype),
        "dtype_after": str(np.asarray(new_c).dtype),
        "relerr_before": _rel(old_c, ref_c),
        "relerr_after": _rel(new_c, ref_c),
    }

    # PDE residual (nabla^2 + k^2) Phi = 0 away from the sources, fixed operator
    rng2 = np.random.default_rng(7)
    src2 = rng2.normal(size=(60, 3))
    src2 /= np.linalg.norm(src2, axis=1)[:, None]
    src2 *= (0.5 * rng2.random(60) ** (1.0 / 3.0))[:, None]
    q2 = rng2.normal(size=60)
    k2 = 3.0
    x0 = np.array([2.0, 1.5, -1.7])
    residuals = []
    for h in (0.04, 0.02, 0.01, 0.005):
        pts = [x0]
        for axis in range(3):
            for sign in (1.0, -1.0):
                e = np.zeros(3)
                e[axis] = sign * h
                pts.append(x0 + e)
        phi, _ = fmm_helmholtz_matvec(np.array(pts), src2, q2, k=k2)
        res = k2 * k2 * phi[0]
        for axis in range(3):
            res += (phi[1 + 2 * axis] - 2.0 * phi[0] + phi[2 + 2 * axis]) / h ** 2
        residuals.append({"h": h, "rel_residual": float(abs(res) / (k2 * k2 * abs(phi[0])))})

    evidence = {
        "task": "m16451",
        "repo": "TensorBEM",
        "pre_fix_rev": PRE_FIX_REV,
        "geometry": {"n_sources": int(len(sources)), "n_targets": int(len(targets)),
                     "source_extent": extent},
        "bands": rows,
        "low_ka_optin": optin,
        "complex_charges": complex_case,
        "pde_residual": residuals,
        "summary": {
            "max_relerr_before": max(r["relerr_before"] for r in rows),
            "min_relerr_before": min(r["relerr_before"] for r in rows),
            "max_relerr_after": max(r["relerr_after"] for r in rows),
            "n_bands_with_err_above_3pct_before": sum(
                1 for r in rows if r["relerr_before"] > 0.03),
            "n_bands_with_err_above_3pct_after": sum(
                1 for r in rows if r["relerr_after"] > 0.03),
            "residual_order": (
                residuals[0]["rel_residual"] / residuals[-1]["rel_residual"]
                if residuals[-1]["rel_residual"] else None
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence["summary"], indent=2))

    _figure(evidence)


def _figure(evidence: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = evidence["bands"]
    ka = [b["ka"] for b in bands]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    ax.loglog(ka, [max(b["relerr_before"], 1e-18) for b in bands], "o-",
              label="before m16451 (real part only / Laplace kernel)")
    ax.loglog(ka, [max(b["relerr_after"], 1e-18) for b in bands], "s-",
              label="after m16451 (exact complex kernel)")
    ax.axhline(0.03, color="k", ls="--", lw=1.0, label="3 % acceptance")
    ax.axvline(0.1, color="grey", ls=":", lw=1.0, label="ka = 0.1 (old Laplace branch)")
    ax.axvline(10.0, color="red", ls=":", lw=1.0, label="ka_max = 10 (fail-closed tag)")
    ax.set_xlabel("ka = k * source extent")
    ax.set_ylabel("max relative error vs analytic kernel")
    ax.set_title("Helmholtz single-layer operator: before vs after")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="center right")

    ax = axes[1]
    ax.semilogx(ka, [b["imag_fraction_of_exact"] for b in bands], "d-", color="C2")
    ax.set_xlabel("ka = k * source extent")
    ax.set_ylabel("max |Im Phi| / max |Phi|  (exact)")
    ax.set_title("Quadrature that the pre-fix operator discarded")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_error_vs_ka.png", dpi=150)
    print("wrote", OUT / "fig_error_vs_ka.png")


if __name__ == "__main__":
    main()