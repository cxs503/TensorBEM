"""CASE-BEM-RIGID-BODY-MASS, equation-level binding of ``M_RB`` (spectrum route).

Why this module
---------------
The anchor is verified at the *matrix* level: closed block form (route R),
signed-tetrahedron quadrature (route M), capytaine assembly (route X) and the
momentum definitions (route P) all agree, and the seakeeping caller replay
(``casepool_bem_rigid_body_mass.rao_caller_section``) pins the *call site*.
Both are fidelity checks: they compare the shipped matrix with another
transcription of the same convention, and the replay proves only that the
solver consumes what was injected.

Nothing in that chain ties ``M_RB`` to a **physical observable** of the
assembled equation. This module closes that gap with the one quantity nobody
has to be told the convention to derive: the **natural frequencies** of the
floating body. Three independent statements are checked.

(a) Decoupled modes against the textbook value.
    A wall-sided box with the CG on the vertical axis has ``S_x = S_y = 0``,
    so heave does not couple: the assembled equation must return
    ``omega_heave**2 = rho g A_wp / m`` exactly (no added mass in vacuum).

(b) Referencing covariance of the spectrum.
    The frequencies of a rigid body do not depend on where the origin of the
    generalised coordinates sits, while ``M_RB``, the added mass and the
    restoring matrix all do. Translating the body horizontally therefore must
    leave the spectrum invariant. The matrix-level covariance identity
    (``A' = M^-T A M^-1``) is available from ``six_dof_matrix_algebra``; here
    the same identity is imposed on ``M_RB`` *and* ``C`` and the check is made
    on the eigenvalues, i.e. on the quantity the seakeeping chain delivers.

(c) Cross-formulation agreement with the textbook CG form.
    Textbook rigid-body dynamics writes the generalised inertia about the CG,
    ``diag(m I3, I_g)``, in the *rotational* coordinates. The shipped
    assembly writes it about the panel-moment origin, with the
    ``-m S(r_g)`` coupling blocks and the parallel-axis term. Pushing the
    shipped matrices through the twist map ``P`` must reproduce the CG form
    and must leave the spectrum unchanged; a wrong coupling sign, a wrong
    parallel-axis placement or a transposed block all break this, and none of
    them is visible to the entry-wise block-form comparison.

Controls (the checks must have teeth)
-------------------------------------
* the historical ``inertia=None`` fallback of ``hydrostatic.compute_mass_matrix``
  (``I_cg = m (0.4 |r_g|)**2``) must move the spectrum by far more than the
  anchor tolerance;
* flipping the sign of the ``-m S(r_g)`` coupling block must break the
  referencing covariance (b) and the cross-formulation check (c), while
  leaving the *decoupled* heave frequency (a) untouched -- i.e. the probe is
  sensitive exactly where the matrix-level checks are blind.

Honest scope
------------
* No hydrodynamics: the reference values are the *vacuum* (zero added mass)
  spectrum. That is deliberate -- added mass would import the BEM solve's own
  discretisation error into the reference and hide convention errors behind
  it -- and it is stated as such: this module does not verify ``A``.
* Wall-sided box only: a prismatic hull with a rectangular waterplane is the
  form whose hydrostatics is closed-form, so any residual above round-off is
  a bug rather than faceting.
* ``scipy.linalg.eigh`` reads one triangle of ``M``, so an *antisymmetric*
  corruption of the ``-m S(r_g)`` block is invisible to this route by
  construction; the momentum route (``rigid_body_mass.mass_matrix_from_momentum``)
  is the one that covers it, and the control below uses the symmetric form of
  the wrong convention on purpose.
* The comparison with the classical metacentric roll period
  ``2 pi r / sqrt(g GM)`` is **reported, not pinned**: see ``NOT_PINNED`` in
  ``study()``. The shipped ``C[3,3]`` follows Journee & Massie's
  waterline-origin form ``rho g [I_xx + nabla (z_B - z_G)]``, which is
  algebraically different from ``rho g nabla GM``; the two agree only in the
  coupling-free limit. Resolving that difference needs a static righting
  moment computed to second order, which is outside this case's scope and is
  registered as an open question instead of being asserted.

Run::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.case_rigid_body_mass_spectrum_binding \
        --outdir docs/case_bem_rigid_body_mass
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from tensorbem.geometry3d import box_hull_mesh
from tensorbem.hydrostatic import (
    compute_hydrostatic_restoring_matrix,
    compute_mass_matrix,
)

__all__ = ["TOLERANCE", "barge", "assemble", "spectrum", "twist_map", "study", "main"]

TOLERANCE = 0.03          # anchor tolerance, entry-wise equivalent for spectra (< 3 %)
COVARIANCE_ROUNDOFF = 1.0e-9
LEGACY_MATERIAL = 0.10    # a defect control must move the spectrum by > 10 %
MODES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


# --------------------------------------------------------------------------- #
# case definition
# --------------------------------------------------------------------------- #
def barge(
    length: float = 20.0,
    beam: float = 8.0,
    draft: float = 3.0,
    *,
    density: float = 1025.0,
    gravity: float = 9.81,
    kg: float = 2.6,
) -> dict:
    """Wall-sided box barge (keel at ``z = -T``, waterline at ``z = 0``)."""
    mass = density * length * beam * draft
    inertia = np.diag([
        mass / 12.0 * (beam ** 2 + draft ** 2),   # roll  (about x)
        mass / 12.0 * (length ** 2 + draft ** 2), # pitch (about y)
        mass / 12.0 * (length ** 2 + beam ** 2),  # yaw   (about z)
    ])
    nabla = length * beam * draft
    bm = (length * beam ** 3 / 12.0) / nabla
    return {
        "length": length,
        "beam": beam,
        "draft": draft,
        "density": density,
        "gravity": gravity,
        "kg": kg,
        "mass": mass,
        "inertia": inertia,
        "center_of_gravity": np.array([0.0, 0.0, -kg]),
        "displacement": nabla,
        "waterplane_area": length * beam,
        "km": draft / 2.0 + bm,
        "gm": draft / 2.0 + bm - kg,
    }


def _mesh(cfg: dict, offset) -> object:
    base = box_hull_mesh(cfg["length"], cfg["beam"], cfg["draft"], n_length=4, n_beam=3, n_draft=3)
    vertices = np.asarray(base.vertices, float) + np.asarray(offset, float)
    return type(base)(vertices=vertices, faces=np.asarray(base.faces, int))


def assemble(cfg: dict, *, offset=(0.0, 0.0, 0.0), legacy: bool = False, flip_coupling: bool = False):
    """Return ``(M_RB, C)`` of the assembled equation in the given frame."""
    shift = np.asarray(offset, float)
    cg = cfg["center_of_gravity"] + shift
    inertia = None if legacy else cfg["inertia"]
    mass_matrix = np.asarray(
        compute_mass_matrix(cfg["mass"], cg, inertia), float
    )
    if flip_coupling:
        # the *symmetric* form of the wrong convention (r_g -> -r_g): negating only
        # one block would produce an antisymmetric matrix that ``scipy.linalg.eigh``
        # cannot see, because it reads one triangle only.
        corrupted = mass_matrix.copy()
        corrupted[np.ix_((0, 1, 2), (3, 4, 5))] *= -1.0
        corrupted[np.ix_((3, 4, 5), (0, 1, 2))] *= -1.0
        mass_matrix = corrupted
    hull = _mesh(cfg, shift)
    restoring = np.asarray(
        compute_hydrostatic_restoring_matrix(
            hull, cfg["mass"], cg, density=cfg["density"], gravity=cfg["gravity"]
        ).restoring_matrix,
        float,
    )
    return mass_matrix, restoring


# --------------------------------------------------------------------------- #
# spectrum
# --------------------------------------------------------------------------- #
def spectrum(mass_matrix, restoring_matrix):
    """Non-zero circular frequencies of ``(M, C)``, ascending, with mode labels.

    The label is taken from the ``(M, v)``-normalised eigenvector's largest
    mass-weighted component, which is stable because the generalised
    eigenproblem normalises w.r.t. ``M``.
    """
    m = np.asarray(mass_matrix, float)
    c = np.asarray(restoring_matrix, float)
    values, vectors = sla.eigh(c, m)
    keep = values > 1e-9 * max(1.0, float(np.max(np.abs(values))))
    order = np.argsort(values[keep])
    values = values[keep][order]
    vectors = vectors[:, keep][:, order]
    weights = np.sqrt(np.abs(np.diag(m)))
    labels = []
    for column in vectors.T:
        dominant = int(np.argmax(np.abs(column) * weights))
        labels.append(MODES[dominant])
    return np.sqrt(values), tuple(labels), vectors


def spectrum_change(reference, other) -> float:
    """Worst relative change of the sorted (canonical) non-zero spectra.

    Eigenvalues are compared as the sorted multiset: the physical claim is that
    the *set* of natural frequencies is frame independent. Pairing by mode
    identity is deliberately not used -- a change of reference point legally
    mixes the translational coordinate into the rotational modes, so an
    eigenvector-overlap pairing would flag the (legitimate) mixing instead of
    the (illegitimate) frequency drift.
    """
    reference = np.asarray(reference, float)
    other = np.asarray(other, float)
    if reference.shape != other.shape:
        return float("inf")
    return float(np.max(np.abs(other - reference) / reference))


def twist_map(center_of_gravity) -> np.ndarray:
    """``u_cg = P u_O``: the twist with translation taken at the origin."""
    rg = np.asarray(center_of_gravity, float)
    skew = np.array([[0.0, -rg[2], rg[1]], [rg[2], 0.0, -rg[0]], [-rg[1], rg[0], 0.0]])
    p = np.eye(6)
    p[:3, 3:] = -skew
    return p


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / scale)


# --------------------------------------------------------------------------- #
# study
# --------------------------------------------------------------------------- #
def study(*, offsets=((0.0, 0.0, 0.0), (5.0, -3.0, 0.0), (-7.2, 4.4, 0.0))) -> dict:
    cfg = barge()
    m0, c0 = assemble(cfg)
    omega0, labels0, vectors0 = spectrum(m0, c0)

    analytic_heave = float(np.sqrt(cfg["density"] * cfg["gravity"] * cfg["waterplane_area"] / cfg["mass"]))
    heave_ok = "heave" in labels0
    heave_error = (
        abs(float(omega0[labels0.index("heave")]) - analytic_heave) / analytic_heave if heave_ok else None
    )

    # (b) referencing covariance of the spectrum
    covariance = []
    for offset in offsets:
        m1, c1 = assemble(cfg, offset=offset)
        omega1, labels1, vectors1 = spectrum(m1, c1)
        worst = spectrum_change(omega0, omega1)
        covariance.append({
            "offset": list(map(float, offset)),
            "spectrum": [float(v) for v in omega1],
            "labels": list(labels1),
            "worst_rel_change": float(worst),
        })

    # (c) cross-formulation: shipped origin form vs the textbook CG form
    p = twist_map(cfg["center_of_gravity"])
    pinv = np.linalg.inv(p)
    m_cg = pinv.T @ m0 @ pinv
    c_cg = pinv.T @ c0 @ pinv
    textbook_cg = np.zeros((6, 6))
    textbook_cg[:3, :3] = cfg["mass"] * np.eye(3)
    textbook_cg[3:, 3:] = cfg["inertia"]
    omega_cg, labels_cg, vectors_cg = spectrum(m_cg, c_cg)
    cross = spectrum_change(omega0, omega_cg)
    inertia_form_error = _rel(m_cg, textbook_cg)

    # controls
    legacy = []
    for offset in offsets:
        m_l, c_l = assemble(cfg, offset=offset, legacy=True)
        omega_l, labels_l, vectors_l = spectrum(m_l, c_l)
        worst = spectrum_change(omega0, omega_l)
        legacy.append({"offset": list(map(float, offset)), "worst_rel_change": float(worst)})

    m_f, c_f = assemble(cfg, offset=(5.0, -3.0, 0.0), flip_coupling=True)
    omega_f, labels_f, vectors_f = spectrum(m_f, c_f)
    flip_change = spectrum_change(omega0, omega_f)
    m_f0, c_f0 = assemble(cfg, flip_coupling=True)
    omega_f0, labels_f0, _ = spectrum(m_f0, c_f0)
    flip_heave = abs(dict(zip(labels_f0, map(float, omega_f0)))["heave"] - analytic_heave) / analytic_heave

    worst_covariance = max(row["worst_rel_change"] for row in covariance)
    worst_legacy = min(row["worst_rel_change"] for row in legacy)

    checks = {
        "heave_matches_closed_form": bool(heave_ok and heave_error is not None and heave_error < TOLERANCE),
        "spectrum_invariant_under_reference_shift": worst_covariance < COVARIANCE_ROUNDOFF,
        "cg_form_reproduces_shipped_spectrum": cross < COVARIANCE_ROUNDOFF,
        "cg_form_is_textbook_diagonal_inertia": inertia_form_error < COVARIANCE_ROUNDOFF,
        "legacy_inertia_fallback_rejected": worst_legacy > LEGACY_MATERIAL,
        "coupling_sign_flip_rejected": flip_change > LEGACY_MATERIAL,
        "coupling_sign_flip_invisible_to_heave": flip_heave < TOLERANCE,
    }
    return {
        "case": "CASE-BEM-RIGID-BODY-MASS/spectrum-binding",
        "parameters": {
            "length": cfg["length"], "beam": cfg["beam"], "draft": cfg["draft"],
            "kg": cfg["kg"], "density": cfg["density"], "gravity": cfg["gravity"],
            "mass": cfg["mass"], "km": cfg["km"], "gm": cfg["gm"],
            "inertia_diagonal": [float(v) for v in np.diag(cfg["inertia"])],
            "waterplane_area": cfg["waterplane_area"],
        },
        "reference": {
            "route": "vacuum rigid-body spectrum, no added mass",
            "heave_omega": analytic_heave,
            "formula": "omega^2 = rho g A_wp / m",
        },
        "base_spectrum": {
            "omega": [float(v) for v in omega0],
            "labels": list(labels0),
            "heave_rel_error": heave_error,
        },
        "reference_shift": covariance,
        "cross_formulation": {
            "worst_rel_spectrum_change": float(cross),
            "cg_form_vs_textbook_inertia_rel_error": float(inertia_form_error),
            "omega_cg_form": [float(v) for v in omega_cg],
        },
        "controls": {
            "legacy_inertia": legacy,
            "coupling_sign_flip_worst_rel_change": float(flip_change),
            "coupling_sign_flip_heave_rel_error": float(flip_heave),
        },
        "not_pinned": {
            "classic_metacentric_roll_period": (
                "the shipped C[3,3] uses the waterline-origin form rho g [I_xx + nabla (z_B - z_G)]; "
                "its difference from rho g nabla GM is reported as an open question, not asserted"
            ),
            "added_mass": "no hydrodynamic added mass is included; A_ij stays verified elsewhere",
        },
        "checks": checks,
        "tolerance": TOLERANCE,
        "roundoff_limit": COVARIANCE_ROUNDOFF,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
        },
    }


# --------------------------------------------------------------------------- #
# document
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}e}"


def write_doc(report: dict, path: Path) -> None:
    params = report["parameters"]
    base = report["base_spectrum"]
    cov = report["reference_shift"]
    cross = report["cross_formulation"]
    ctrl = report["controls"]
    lines = [
        "# CASE-BEM-RIGID-BODY-MASS — 装配方程的谱绑定（自然频率路线）",
        "",
        "本文件由 `src/tensorbem/benchmarks/case_rigid_body_mass_spectrum_binding.py` 从 "
        "`spectrum_binding.json` 程序注入，无手抄数字。",
        "",
        "## ① 为什么补这一层",
        "",
        "锚点的既有证据全部是**矩阵级**的（闭式块形式 R / 四面体积分 M / capytaine X / 动量定义 P）"
        "加上**调用点回放**：它们把出厂矩阵与同一约定的另一次转写相比，或证明求解器确实吃进了注入的矩阵。",
        "没有任何一条把 `M_RB` 绑到装配方程的**物理可观测量**上。本单用自然频率补这一层：频率不需要约定即可推出。",
        "",
        "## ② 参数",
        "",
        "| 参数 | 数值 |",
        "|---|---|",
        f"| L x B x T | {params['length']} x {params['beam']} x {params['draft']} m |",
        f"| rho / g | {params['density']} kg/m3 / {params['gravity']} m/s2 |",
        f"| 排水量 / 质量 | {params['mass']:.6e} kg |",
        f"| KG / KM / GM | {params['kg']} / {params['km']:.4f} / {params['gm']:+.4f} m |",
        f"| I_xx / I_yy / I_zz (关于质心) | {params['inertia_diagonal'][0]:.4e} / "
        f"{params['inertia_diagonal'][1]:.4e} / {params['inertia_diagonal'][2]:.4e} kg m2 |",
        f"| 水线面面积 A_wp | {params['waterplane_area']} m2 |",
        "",
        "## ③ 结果",
        "",
        "**(a) 解耦垂荡 vs 闭式**：`omega^2 = rho g A_wp / m`，判据 < 3%。",
        "",
        f"实测 omega_heave = **{base['omega'][base['labels'].index('heave')]:.6f} rad/s**，"
        f"闭式 {report['reference']['heave_omega']:.6f} rad/s，相对误差 **{_fmt(base['heave_rel_error'])}**。",
        "",
        "**(b) 参考点平移下谱不变**（物理不变量；判据 {:.0e}）".format(report["roundoff_limit"]),
        "",
        "| 平移 (m) | 谱 (rad/s) | 标签 | 最大相对变化 |",
        "|---|---|---|---|",
    ]
    for row in cov:
        lines.append(
            f"| {row['offset']} | {', '.join(f'{v:.6f}' for v in row['spectrum'])} | "
            f"{', '.join(row['labels'])} | {_fmt(row['worst_rel_change'])} |"
        )
    lines += [
        "",
        f"**(c) 交叉表述**：把出厂矩阵经 twist 映射压到质心系，得到 "
        f"`M_cg`；它与教材 `diag(m I3, I_g)` 的相对差 **{_fmt(cross['cg_form_vs_textbook_inertia_rel_error'])}**，"
        "其谱与原点系谱的相对差 **"
        + _fmt(cross["worst_rel_spectrum_change"])
        + "**（判据 {:.0e}）。".format(report["roundoff_limit"]),
        "",
        "**(d) 对照（证明判据有牙齿）**",
        "",
        "| 对照 | 结果 | 结论 |",
        "|---|---|---|",
        f"| 历史 `inertia=None` 回退 | 最小谱变化 {_fmt(min(r['worst_rel_change'] for r in ctrl['legacy_inertia']))} | 拒绝 |",
        f"| `-m S(r_g)` 耦合块符号取反（平移后） | 谱变化 {_fmt(ctrl['coupling_sign_flip_worst_rel_change'])} | 拒绝 |",
        f"| 同上，垂荡频率 | 相对误差 {_fmt(ctrl['coupling_sign_flip_heave_rel_error'])} | 看不见（这正是矩阵级判据的盲区） |",
        "",
        "## ④ 判据汇总",
        "",
        "| 判据 | 通过 |",
        "|---|---|",
    ]
    for name, ok in report["checks"].items():
        lines.append(f"| `{name}` | {'✅' if ok else '❌'} |")
    lines += [
        "",
        "## ⑤ 注意点 / 明确不声称",
        "",
        f"* **不含水动力附加质量**：参照是真空谱。加进 `A` 会把 BEM 自身的离散误差带进参照，"
        "从而掩盖约定错误；`A_ij` 的验证在别处（Hulme/椭球/协变恒等式）。",
        "* **不钉经典横摇周期** `2 pi r / sqrt(g GM)`：出厂的 `C[3,3]` 用 Journee & Massie 的水线原点形式 "
        "`rho g [I_xx + nabla (z_B - z_G)]`，与 `rho g nabla GM` 只在无耦合极限一致；"
        "两者之差需要二阶静水复原力矩独立仲裁，已登记为开放问题（`not_pinned`），本单不作断言。",
        f"* 判据容差：闭式比较用锚点 3%；协变/交叉表述用圆整极限 {report['roundoff_limit']:.0e}；"
        "对照必须 > 10% 才算被拒绝。",
        f"* 环境：Python {report['environment']['python']} · NumPy {report['environment']['numpy']} · "
        f"SciPy {report['environment']['scipy']}。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("docs/case_bem_rigid_body_mass"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    report = study()
    if args.write:
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "spectrum_binding.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        write_doc(report, args.outdir / "SPECTRUM_BINDING.md")
    print(json.dumps(report["checks"], indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())