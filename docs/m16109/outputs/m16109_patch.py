"""Apply the m16109 fail-closed setup contract to FMMOperator (minimal edits)."""
from pathlib import Path

root = Path("/root/tp-agent/work/m16109")

# ---------------------------------------------------------------- fmm.py ----
fmm = root / "src" / "tensorbem" / "fmm.py"
text = fmm.read_text()

# 1. matvec: fail closed before the silent-zero early return
old_matvec = """        if charges.shape != (sources.shape[0],):
            raise ValueError("charges must have shape (M,)")
        n = len(targets)
        result = np.zeros(n, dtype=float)
        if n == 0 or not self.leaves:
            return result
"""
new_matvec = """        if charges.shape != (sources.shape[0],):
            raise ValueError("charges must have shape (M,)")
        self._require_setup("matvec")
        n = len(targets)
        result = np.zeros(n, dtype=float)
        if n == 0:
            return result
"""
assert text.count(old_matvec) == 1, "matvec anchor not found"
text = text.replace(old_matvec, new_matvec)

# 2. far_fraction: fail closed instead of a misleading 0.0
old_ff = """    def far_fraction(self, targets: FloatArray) -> float:
        \"\"\"Fraction of target-leaf pairs evaluated with the multipole expansion.\"\"\"
        targets = np.asarray(targets, dtype=float)
        if len(targets) == 0 or not self.leaves:
            return 0.0
"""
new_ff = """    def far_fraction(self, targets: FloatArray) -> float:
        \"\"\"Fraction of target-leaf pairs evaluated with the multipole expansion.

        Requires :meth:`setup`; an unset operator raises ``RuntimeError``
        instead of silently reporting 0.0 (m16109).
        \"\"\"
        targets = np.asarray(targets, dtype=float)
        self._require_setup("far_fraction")
        if len(targets) == 0:
            return 0.0
"""
assert text.count(old_ff) == 1, "far_fraction anchor not found"
text = text.replace(old_ff, new_ff)

# 3. the guard itself, right after setup()
old_guard_anchor = """    def _build_multipoles(
        self, points: FloatArray, charges: FloatArray
    ) -> FloatArray:
"""
new_guard = """    def _require_setup(self, method: str) -> None:
        \"\"\"Fail closed: public entry points need a built octree (m16109).

        Before the fix, ``matvec`` on a never-set-up operator returned an
        all-zero vector and ``far_fraction`` reported 0.0 -- both
        indistinguishable from legitimate values, so a forgotten ``setup``
        produced silent wrong answers (m16000 limitation #2).
        \"\"\"
        if self.tree is None or not self.leaves:
            raise RuntimeError(
                f"FMMOperator.{method}() called before setup(): "
                "call setup(points[, charges]) first"
            )

    def _build_multipoles(
        self, points: FloatArray, charges: FloatArray
    ) -> FloatArray:
"""
assert text.count(old_guard_anchor) == 1, "guard anchor not found"
text = text.replace(old_guard_anchor, new_guard)

fmm.write_text(text)
print("fmm.py patched")

# ------------------------------------------------- bench script (the misuse) --
bench = root / "scripts" / "fmm_linearity_bench.py"
btext = bench.read_text()
old_bench = '    return {"far_fraction": float(FMMOperator(p=4, max_leaf=25).far_fraction(targets)),'
new_bench = (
    "    ff_op = FMMOperator(p=4, max_leaf=25)\n"
    "    ff_op.setup(sources)  # m16109: far_fraction needs a built octree\n"
    '    return {"far_fraction": float(ff_op.far_fraction(targets)),'
)
assert btext.count(old_bench) == 1, "bench anchor not found"
btext = btext.replace(old_bench, new_bench)
bench.write_text(btext)
print("bench patched")
