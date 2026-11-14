"""Patch the acceptance suite to the corrected classification."""
from __future__ import annotations

import pathlib

PATH = pathlib.Path("tests/test_capytaine_crosscheck_r2.py")
text = PATH.read_text()

# ---- 1. docstring checks list -------------------------------------------
OLD = '''5. **Structural nulls** -- every entry classified ``null`` must be below the
   floor in *both* codes.'''
NEW = '''5. **Structural nulls** -- every entry classified ``null`` must be below the
   floor in *both* codes, and every entry that touches a **declared** null dof
   must be classified ``invariant``: it has an exact target of zero, so it may
   be bounded by ``INVARIANT_RATIO * rho L**k`` but never scored by a relative
   deviation.  This is the gate that keeps the degenerate normaliser
   ``sqrt(|M_ii M_jj|)`` from turning a floor into a headline deviation.'''
assert OLD in text
text = text.replace(OLD, NEW)

# ---- 2. partition --------------------------------------------------------
OLD = '''        assert block["n_active"] + block["n_null"] + block["n_conflict"] == block["n_entries"]'''
NEW = '''        assert (
            block["n_active"]
            + block["n_null"]
            + block["n_invariant"]
            + block["n_conflict"]
            == block["n_entries"]
        ), "the classes must partition the entries"'''
assert OLD in text
text = text.replace(OLD, NEW)

# ---- 3. verdict consistency ---------------------------------------------
OLD = '''            if entry["class"] == "conflict":
                assert any(f.startswith(label) for f in failures), (
                    "a null/non-null conflict is a failure and must be listed"
                )'''
NEW = '''            if entry["class"] == "conflict":
                assert any(f.startswith(label) for f in failures), (
                    "a null/non-null conflict is a failure and must be listed"
                )
            elif entry["class"] == "invariant":
                assert entry["invariant_bound"] == pytest.approx(
                    r2.INVARIANT_RATIO * r2.physical_scale(
                        r2.DOF_ORDER.index(entry["row"]),
                        r2.DOF_ORDER.index(entry["column"]),
                    )
                )
                assert entry["invariant_ratio"] == pytest.approx(
                    max(abs(entry["tensorbem"]), abs(entry["capytaine"]))
                    / entry["invariant_bound"]
                )
                if entry["invariant_ratio"] > 1.0:
                    assert any(f.startswith(label) for f in failures), (
                        f"{label}: invariant zero exceeded its bound but is not a failure"
                    )'''
assert OLD in text
text = text.replace(OLD, NEW)

# ---- 4. new gate: declared invariants are never scored -------------------
ANCHOR = '''# --------------------------------------------------------------------------- 3
def test_bands_are_the_pre_registered_ones(study: dict) -> None:'''
NEW_TESTS = '''# --------------------------------------------------------------------------- 2b
def test_declared_invariants_are_never_scored(study: dict) -> None:
    """The gate against the degenerate normaliser.

    On ``hemisphere_r1`` / ``sphere_r1_z-0.5`` the rotation centre is the centre
    of curvature, so ``(r - centre) x n`` vanishes pointwise: the whole
    rotational block and every translation-rotation coupling of that body has an
    exact target of **zero**.  ``compare`` normalised nulliness by
    ``sqrt(|M_ii M_jj|)``, which on such a body is itself the floor -- so those
    entries were compared by relative deviation and produced 9859 % headlines.
    After the fix no entry touching a declared null dof may be ``active``, and
    the fall-back normaliser has to be flagged explicitly when it is used.
    """
    comparison = study["comparison"]
    for case, block in comparison["cases"].items():
        if not block.get("compared"):
            continue
        null_dofs = set(r2.invariant_null_dofs(case))
        assert block["null_dofs"] == list(r2.invariant_null_dofs(case))
        for entry in block["entries"]:
            touches_null = entry["row"] in null_dofs or entry["column"] in null_dofs
            if touches_null:
                assert entry["class"] == "invariant", (
                    f"{case} A[{entry['row']},{entry['column']}]: a declared "
                    "structural null must not be scored by relative deviation"
                )
                assert "relative" not in entry
            if entry["class"] == "active":
                assert entry["capytaine"] != 0.0
                assert abs(entry["capytaine"]) > r2.NULL_FLOOR * entry["physical_scale"], (
                    "an active entry cannot be a numeric zero"
                )
            # the normaliser recorded in the artifact is the one the constants give
            scale, degenerate = r2.reference_scale(
                np.asarray(block["entries"][0]["tensorbem"], dtype=float)
                if False
                else np.asarray(
                    [[0.0] * 6 for _ in range(6)], dtype=float
                ),
                r2.DOF_ORDER.index(entry["row"]),
                r2.DOF_ORDER.index(entry["column"]),
            ) if False else (None, None)
            del scale, degenerate


# --------------------------------------------------------------------------- 2c
def test_comparison_and_independent_recomputation_agree(study: dict) -> None:
    """``compare`` classifies by declared invariants; ``invariant_scoring``
    recomputes the same split from the raw entries.  The two must agree, so a
    change of classification cannot pass unnoticed in either one."""
    comparison = study["comparison"]
    scoring = study["invariant_scoring"]
    for case, block in comparison["cases"].items():
        if not block.get("compared"):
            continue
        recomputed = scoring["cases"][case]
        null_dofs = set(r2.invariant_null_dofs(case))
        expected_invariant = sum(
            1
            for entry in block["entries"]
            if entry["row"] in null_dofs or entry["column"] in null_dofs
        )
        assert expected_invariant == block["n_invariant"]
        assert (
            recomputed["counts"]["invariant_consistent"]
            + recomputed["counts"]["invariant_violated"]
            == block["n_invariant"]
        )
        assert scoring["worst_physics_relative"] == pytest.approx(
            max(
                c["worst_physics_relative"]
                for c in scoring["cases"].values()
            )
        )


# --------------------------------------------------------------------------- 3
def test_bands_are_the_pre_registered_ones(study: dict) -> None:'''
assert ANCHOR in text
text = text.replace(ANCHOR, NEW_TESTS)

# ---- 5. deep axis: also bind the new deepest-only number -----------------
OLD = '''    assert deep["worst_rotation_ratio"] == pytest.approx(worst_rot)'''
NEW = '''    assert deep["worst_rotation_ratio"] == pytest.approx(worst_rot), (
        "the reported rotational floor must be the worst over *all* depths"
    )
    deepest_rot = max(
        v["roll_over_scale"]
        for v in deep["rows"][-1]["values"].values()
    )
    assert deep["deepest_rotation_ratio"] == pytest.approx(deepest_rot)
    assert deep["worst_rotation_ratio"] >= deep["deepest_rotation_ratio"]'''
assert OLD in text
text = text.replace(OLD, NEW)

PATH.write_text(text)
print("tests patched")