"""Second patch: make the two historical failure modes switchable, so C8 can attribute them.

Mission m6502. Run from the worktree root with PYTHONPATH=src.
"""
from pathlib import Path

p = Path("/root/tp-agent/work/m6502/src/tensorbem/benchmarks/tank_free_surface_verification.py")
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


# flags on the energy helper so the failure modes can be reproduced on purpose
rep(
    'def _liquid_energies(verts, n, depth, phi, theta, density, g, z_bottom=0.0) -> dict:',
    'def _liquid_energies(\n'
    '    verts, n, depth, phi, theta, density, g, z_bottom=0.0, centroidal=True\n'
    ') -> dict:',
)
rep(
    '''    u = y - float((w * y).sum() / area)
    v = x - float((w * x).sum() / area)''',
    '''    if centroidal:
        u = y - float((w * y).sum() / area)
        v = x - float((w * x).sum() / area)
    else:  # historical bug: plane written about the footprint origin
        u, v = y, x''',
)
rep(
    '''def _energy_hessian(verts, n, depth, eps, density, g) -> dict:
    """Central-difference Hessian of ``U_free - U_frozen`` in ``(phi, theta)``."""

    def delta(phi, theta):
        return _liquid_energies(verts, n, depth, phi, theta, density, g)["delta"]
''',
    '''def _energy_hessian(verts, n, depth, eps, density, g, centroidal=True, reference=True) -> dict:
    """Central-difference Hessian of ``U_free - U_frozen`` in ``(phi, theta)``.

    ``centroidal=False`` reproduces the historical bug (free-surface plane written
    about the footprint origin, so the tank volume is not preserved once it
    tilts); ``reference=False`` drops the frozen-liquid baseline, which leaves the
    rigid weight-at-height energy in the curvature.
    """

    def delta(phi, theta):
        key = "delta" if reference else "free"
        return _liquid_energies(
            verts, n, depth, phi, theta, density, g, centroidal=centroidal
        )[key]
''',
)
rep(
    '''    probe = _liquid_energies(verts, n, depth, eps, eps, density, g)''',
    '''    probe = _liquid_energies(verts, n, depth, eps, eps, density, g, centroidal=centroidal)''',
)

# attribution table inside C8
rep(
    '''    out["checks"]["C8_energy_hessian"] = c8
''',
    '''    # Attribution of the 4x symptom that made m6452 abandon this path: the two
    # variants below isolate which defect causes it (rectangle roll entry).
    attribution = []
    for name, centroidal, reference in (
        ("centroidal_plane + frozen reference (this check)", True, True),
        ("origin-based plane + frozen reference (m6452 symptom)", False, True),
        ("centroidal plane, no frozen reference", True, False),
    ):
        h = _energy_hessian(
            RECT, grids[-1], depth, 1.0e-3, rho_w, g,
            centroidal=centroidal, reference=reference,
        )
        target = -rho_w * g * rect.i_xx
        attribution.append(
            {
                "variant": name,
                "roll": h["roll"],
                "analytic": target,
                "ratio": h["roll"] / target,
                "rel_error": abs(h["roll"] - target) / abs(target),
            }
        )
    c8["attribution"] = attribution
    out["checks"]["C8_energy_hessian"] = c8
''',
)

p.write_text(s)
print("patched")