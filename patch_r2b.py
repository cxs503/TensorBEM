"""Patch 2: exercise the round-1 finest meshes and harden ``convergence``."""
from __future__ import annotations

import pathlib

PATH = pathlib.Path("src/tensorbem/benchmarks/capytaine_crosscheck_r2.py")
text = PATH.read_text()

OLD_LEVELS = '''#: Mesh levels actually exercised under this round's compute budget: 64/256
#: panels (hemisphere) and 270 (sphere).  The 1024/1026 levels used by round 1
#: were not re-run here -- a documented limitation, not an assumed equivalence.
HEMISPHERE_LEVELS = (2, 3)
SPHERE_LEVELS = (2,)'''
NEW_LEVELS = '''#: Mesh levels exercised by the comparison.  The first submission of this round
#: ran 64/256 (hemisphere) and 270 (sphere) only and then compared those solves
#: against capytaine's *finest* meshes (784 / 1120 panels): the reported gap
#: mixed code-to-code disagreement with a 3-4x panel-density mismatch.  The
#: finest levels of round 1 (1024 / 1026 panels) are therefore exercised here as
#: well, so the codes meet at comparable density and the ladder shows the
#: discretisation trend.  ``reference`` is the finest level (``run_case``).
HEMISPHERE_LEVELS = (2, 3, 4)
SPHERE_LEVELS = (2, 3)'''
assert OLD_LEVELS in text
text = text.replace(OLD_LEVELS, NEW_LEVELS)

OLD_CONV = '''                        scale = _scale(cap, i, j)
                        if abs(cap[i, j]) <= NULL_FLOOR * scale:
                            continue'''
NEW_CONV = '''                        scale = _scale(cap, i, j)
                        if not math.isfinite(scale) or scale <= NULL_FLOOR * physical_scale(i, j):
                            # the reference normaliser is itself a floor: the
                            # ratio would compare two zeros (see ``compare``)
                            continue
                        if abs(cap[i, j]) <= NULL_FLOOR * scale:
                            continue'''
assert OLD_CONV in text
text = text.replace(OLD_CONV, NEW_CONV)

PATH.write_text(text)
print("patched 2")