"""Apply the finite-depth 3-D mode-truncation fixes (one-shot patch script)."""
from pathlib import Path

p = Path("src/tensorbem/finite_depth_solver3d.py")
src = p.read_text()

# --- 1. operators report whether the mode cap truncated the near field --------
old = '''    system_matrix: ComplexArray
    condition_number: float
    kernel: str = "eigenseries"
'''
new = '''    system_matrix: ComplexArray
    condition_number: float
    kernel: str = "eigenseries"
    #: Rows whose per-row mode requirement (``_modes_for`` at the nearest
    #: target/quadrature separation) exceeded ``mode_count``, i.e. rows the cap
    #: truncated below the requested ``exp(-_MODE_DECAY)`` tail.  Non-zero means
    #: the operator is *under-resolved* by construction and the result is a
    #: truncation study artefact rather than a converged solve.
    mode_cap_bound_rows: int = 0
    #: Largest ``requirement / mode_count`` ratio over the rows (1.0 = no cap).
    mode_shortfall: float = 1.0
'''
assert old in src
src = src.replace(old, new)

# --- 2. honour _BLOCK_ELEMENTS and the per-row mode counts -------------------
old = '''        row_modes = np.array(
            [min(mode_count, _modes_for(depth, r)) for r in row_min], dtype=np.int64
        )
        order_rows = np.argsort(-row_modes, kind="stable")
        step = max(1, int(group_rows))
        blocks = [
            order_rows[start : start + step] for start in range(0, m_cnt, step)
        ]
    else:
        blocks = [np.arange(m_cnt)]

    for rows in blocks:
        tgt = targets[rows][:, None, :]
        src = qpts[None, :, :]
'''
new = '''        row_modes = np.array(
            [min(mode_count, _modes_for(depth, r)) for r in row_min], dtype=np.int64
        )
        order_rows = np.argsort(-row_modes, kind="stable")
        step = max(1, int(group_rows))
        blocks = [
            order_rows[start : start + step] for start in range(0, m_cnt, step)
        ]
    else:
        row_modes = np.full(m_cnt, int(mode_count), dtype=np.int64)
        blocks = np.arange(m_cnt)[None, :].tolist()

    n_qp = int(qpts.shape[0])
    for rows in blocks:
        rows = np.asarray(rows, dtype=np.int64)
        # Every row of a block is evaluated with the *same* mode count (the
        # largest requirement inside it), so the block must be small enough that
        # ``rows * n_qp * modes`` stays inside the element budget: without this
        # the temporaries grow as ``group_rows * n_qp * mode_count`` and a
        # converged (large) mode count exhausts memory.
        block_modes = int(row_modes[rows].max())
        per_block = max(1, _BLOCK_ELEMENTS // max(1, n_qp * block_modes))
        for start in range(0, rows.size, per_block):
            _quadrature_remainder_block(
                rows[start : start + per_block], targets, qpts, qweights, qnormals,
                per_panel, single, double, nu, depth, block_modes, kernel,
                kernel_nodes,
            )
    return single, double


def _quadrature_remainder_block(
    rows: NDArray[np.int64],
    targets: FloatArray,
    qpts: FloatArray,
    qweights: FloatArray,
    qnormals: FloatArray,
    per_panel: int,
    single: ComplexArray,
    double: ComplexArray,
    nu: float,
    depth: float,
    block_modes: int,
    kernel: str,
    kernel_nodes: int,
) -> None:
    """Add one row block of the quadratured ``G_fd - 1/(4*pi*r)`` operators."""
    value_fn, grad_fn = kernel_functions(
        kernel, nu, depth, block_modes, nodes=kernel_nodes
    )
    tgt = targets[rows][:, None, :]
    src = qpts[None, :, :]
'''
assert old in src
src = src.replace(old, new)

# the body of the old loop now belongs to the helper: de-indent/rename taps
old = '''        diff = tgt - src
        r = np.maximum(np.linalg.norm(diff, axis=2), 1.0e-300)
        inv = 1.0 / (_FOUR_PI * r)

        value = value_fn(tgt, src)
        single[rows] += ((value - inv) * qweights[None, :]).reshape(
            rows.size, p_cnt, per_panel
        ).sum(axis=2)

        grad = grad_fn(tgt, src)
        dn_wave = np.einsum("mqk,qk->mq", grad, qnormals)
        dn_rankine = np.einsum("mqk,qk->mq", diff, qnormals) / (_FOUR_PI * r**3)
        double[rows] += ((dn_wave - dn_rankine) * qweights[None, :]).reshape(
            rows.size, p_cnt, per_panel
        ).sum(axis=2)
    return single, double
'''
new = '''    diff = tgt - src
    r = np.maximum(np.linalg.norm(diff, axis=2), 1.0e-300)
    inv = 1.0 / (_FOUR_PI * r)

    value = value_fn(tgt, src)
    single[rows] += ((value - inv) * qweights[None, :]).reshape(
        rows.size, -1, per_panel
    ).sum(axis=2)

    grad = grad_fn(tgt, src)
    dn_wave = np.einsum("mqk,qk->mq", grad, qnormals)
    dn_rankine = np.einsum("mqk,qk->mq", diff, qnormals) / (_FOUR_PI * r**3)
    double[rows] += ((dn_wave - dn_rankine) * qweights[None, :]).reshape(
        rows.size, -1, per_panel
    ).sum(axis=2)
'''
assert old in src
src = src.replace(old, new)

# --- 3. record/report the truncation shortfall at assembly time -------------
old = '''    system = 0.5 * np.eye(mesh.panel_count, dtype=np.complex128) - double
    return FiniteDepth3DOperators('''
new = '''    system = 0.5 * np.eye(mesh.panel_count, dtype=np.complex128) - double
    need = _row_mode_requirements(mesh, h)
    bound = int(np.count_nonzero(need > count))
    shortfall = float(need.max() / count) if count else 1.0
    if bound and warn_on_truncation:
        warnings.warn(
            f"mode_count={count} truncates {bound}/{need.size} rows below their "
            f"e^{-_MODE_DECAY:.0f} tail requirement (max requirement {need.max()}); "
            "the operator is under-resolved by construction",
            RuntimeWarning,
            stacklevel=2,
        )
    return FiniteDepth3DOperators('''
assert old in src
src = src.replace(old, new)

old = '''        system_matrix=system,
        condition_number=float(np.linalg.cond(system)),
        kernel=kernel,
    )'''
new = '''        system_matrix=system,
        condition_number=float(np.linalg.cond(system)),
        kernel=kernel,
        mode_cap_bound_rows=bound,
        mode_shortfall=shortfall,
    )'''
assert old in src
src = src.replace(old, new)

old = '''    kernel: str = "eigenseries",
    kernel_nodes: int = 16,
    panel_areas: ArrayLike | None = None,
) -> FiniteDepth3DOperators:'''
new = '''    kernel: str = "eigenseries",
    kernel_nodes: int = 16,
    panel_areas: ArrayLike | None = None,
    warn_on_truncation: bool = False,
) -> FiniteDepth3DOperators:'''
assert old in src
src = src.replace(old, new)

# helper: per-row requirement vector
old = '''def mode_count_for(mesh: SurfaceMesh3D, depth: float) -> int:'''
new = '''def _row_mode_requirements(mesh: SurfaceMesh3D, depth: float) -> NDArray[np.int64]:
    """Per-row evanescent mode requirement of the quadratured remainder.

    One entry per panel row: the mode count ``_modes_for`` asks for at that
    row's smallest target/quadrature *horizontal* separation.  The near field
    (small separation) needs thousands of modes on a 320-panel sphere, so a
    pinned ``mode_count`` below this vector silently truncates the operator.
    """
    qpts, _, _ = quadrature_points(mesh)
    horiz = np.hypot(
        mesh.centroids[:, None, 0] - qpts[None, :, 0],
        mesh.centroids[:, None, 1] - qpts[None, :, 1],
    )
    with np.errstate(invalid="ignore"):
        row_min = np.where(horiz > 0.0, horiz, np.inf).min(axis=1)
    return np.array([_modes_for(depth, r) for r in row_min], dtype=np.int64)


def mode_count_for(mesh: SurfaceMesh3D, depth: float) -> int:'''
assert old in src
src = src.replace(old, new)

# imports: warnings
old = "from __future__ import annotations"
new = "from __future__ import annotations\n\nimport warnings"
assert old in src
src = src.replace(old, new, 1)

p.write_text(src)
print("patched", p, len(src))