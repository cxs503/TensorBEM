"""Patch 3: robust reference scale instead of an 'undeclared_null' failure class.

A degenerate ``sqrt(|M_ii M_jj|)`` below the body-scale floor means the
*normaliser* is a floor, not that the entry is structurally zero: entries on a
body whose rotational block is genuinely small (off-centre cases) would be
flagged as an undeclared invariant.  Fall back to the physical scale instead.
"""
from __future__ import annotations

import pathlib

PATH = pathlib.Path("src/tensorbem/benchmarks/capytaine_crosscheck_r2.py")
text = PATH.read_text()

OLD = '''def _scale(mat: np.ndarray, i: int, j: int) -> float:
    scale = math.sqrt(abs(mat[i, i] * mat[j, j]))
    return scale if scale > 0.0 else float("nan")'''
NEW = '''def _scale(mat: np.ndarray, i: int, j: int) -> float:
    scale = math.sqrt(abs(mat[i, i] * mat[j, j]))
    return scale if scale > 0.0 else float("nan")


def reference_scale(mat: np.ndarray, i: int, j: int) -> tuple[float, bool]:
    """Normaliser of the null test, with a fall-back when it is itself a floor.

    ``sqrt(|M_ii M_jj|)`` is the natural reference for "is this entry
    negligible".  It collapses whenever a diagonal entry is itself a
    structural null -- on ``sphere_r1_z-0.5_c0`` the rotational block is real
    but four orders below ``rho L**5``, so the pair normaliser is a floor, and
    dividing by it turns a residual into an apparent O(1) deviation.  When it
    falls below ``NULL_FLOOR * physical_scale`` the body scale is used instead
    and the fall-back is reported (``degenerate=True``) so the choice is
    visible in the artifact rather than silent.
    """
    scale = _scale(mat, i, j)
    phys = physical_scale(i, j)
    if not math.isfinite(scale) or scale <= NULL_FLOOR * phys:
        return phys, True
    return scale, False'''
assert OLD in text
text = text.replace(OLD, NEW)

# compare(): use reference_scale and drop the undeclared_null class
OLD2 = '''                        scale = _scale(a_tb, i, j)
                        phys = physical_scale(i, j)'''
NEW2 = '''                        scale, degenerate = reference_scale(a_tb, i, j)
                        phys = physical_scale(i, j)'''
assert OLD2 in text
text = text.replace(OLD2, NEW2)

OLD3 = '''                            "scale": scale,
                            "physical_scale": phys,
                        }'''
NEW3 = '''                            "scale": scale,
                            "scale_degenerate": bool(degenerate),
                            "physical_scale": phys,
                        }'''
assert OLD3 in text
text = text.replace(OLD3, NEW3)

start = text.index("                        degenerate = (\n")
end = text.index("                        cap_null = abs(a_cap[i, j]) <= NULL_FLOOR * scale")
text = text[:start] + text[end:]

# drop the undeclared_null counters
OLD4 = '''            "n_undeclared_null": len(
                [e for e in entries if e["class"] == "undeclared_null"]
            ),
'''
assert OLD4 in text
text = text.replace(OLD4, "")

# document prose: no undeclared_null class any more
OLD5 = '''"**conflict**（一码显著、另一码为零 → 直接失败）、**undeclared_null**"
        "（未声明的恒零项，说明零自由度清单不完整 → 直接失败）。".format('''
NEW5 = '''"**conflict**（一码显著、另一码为零 → 直接失败）。若 "
        "`sqrt(|M_ii M_jj|)` 本身落在物理尺度地板上（转动块整体很小的偏心算例），"
        "改用 ρL^k 作归一化尺度并在 artifact 里标记 `scale_degenerate`。".format('''
assert OLD5 in text
text = text.replace(OLD5, NEW5)

PATH.write_text(text)
print("patched 3")