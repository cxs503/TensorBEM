"""Insert the C8/C9 attachment helpers into the Cummins/Wigley case module."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "tensorbem" / "benchmarks" / "case_bem_cummins_wigley.py"

HELPERS = '''
#: Committed artifacts of the *pre-declared* band extension (0.2 .. 20 rad/s on
#: :data:`EXTENDED_MESH`).  Gates ``C8``/``C9`` exist only when
#: ``sweep_extended.npz`` is present: the truncation budget of surge/sway/yaw is
#: unbounded on the 12 rad/s band, and only a wider band can move it, so an
#: absent artifact has to read as *not evaluated* instead of as a pass.
EXTENDED_SWEEP_NAME = "sweep_extended.npz"
EXTENDED_ANCHOR_NAME = "anchor_ladder_extended.json"


def extended_anchor_for_sweep(
    sweep: Sweep, ladder_path: str | Path | None = None
) -> dict[str, Any] | None:
    """``A_inf`` anchor of an extended-band mesh, from the committed ladder.

    The image solve that produces the anchor doubles the panel count (7680
    unknowns at ``60 x 16``), so the ladder is a committed artifact rather than
    something a replay recomputes.  The rung is selected by *mesh identity*
    (``n_sections``/``n_levels``), never by label: an anchor solved on another
    mesh is not an anchor for this sweep.

    Returns ``None`` when no rung matches, and the caller then falls back to the
    band-fitted offset of :func:`kk_gate` -- weaker evidence, and the gate says
    so through its ``anchor`` field instead of pretending the anchored form ran.
    """
    if ladder_path is None:
        return None
    candidate = Path(ladder_path)
    if not candidate.exists():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    for rung in payload.get("rungs", []):
        if (int(rung["n_sections"]), int(rung["n_levels"])) == (
            int(sweep.n_sections),
            int(sweep.n_levels),
        ):
            return rung
    return None


def attach_extended_band_gates(
    gates: dict[str, Any],
    committed_band: Sweep,
    directory: str | Path,
    *,
    extended_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Attach ``C8`` / ``C3a_extended`` / ``C9`` when the extended sweep exists.

    ``C8`` asks whether the wider band reproduces the committed one where they
    overlap (otherwise the two verdicts are not comparable), ``C3a_extended`` is
    the zero-free-parameter closure on the wider band, and ``C9`` asks whether
    the damping at the new band top has entered algebraic decay, i.e. whether the
    truncation budget that made surge/sway/yaw undecidable at 12 rad/s has become
    finite.

    Nothing is attached when the artifact is missing, and the return value is
    ``None``: the case verdict then keeps the 16-gate shape of the 12 rad/s band.
    """
    path = (
        Path(extended_path)
        if extended_path is not None
        else Path(directory) / EXTENDED_SWEEP_NAME
    )
    if not path.exists():
        return None
    extended = load_sweep(path)
    anchor = extended_anchor_for_sweep(extended, path.parent / EXTENDED_ANCHOR_NAME)
    anchored = kk_gate(
        extended,
        added_mass_inf=(
            anchor["added_mass_inf_diagonal"] if anchor is not None else None
        ),
        identifier="C3a_extended",
    )
    committed = gates.get("C3a")
    committed_rows = committed["values"]["per_dof"] if committed else None
    gates["C8"] = extension_consistency_gate(committed_band, extended)
    gates["C3a_extended"] = anchored
    gates["C9"] = truncation_decidability_gate(
        committed_band,
        extended,
        closure_rows=anchored["values"]["per_dof"],
        reference_rows=committed_rows,
    )
    gates["C9"]["values"]["anchor_source"] = (
        "image-solve ladder rung "
        f"{anchor['n_sections']}x{anchor['n_levels']}"
        if anchor is not None
        else "band-fitted offset (no ladder rung for this mesh)"
    )
    return {"extended_sweep": extended.summary(), "anchor": anchor}


'''

MARKER = "def gates_from_artifact("
SIG_OLD = """    include_decay_modes: bool = True,
    include_external_anchor: bool = True,
) -> dict[str, Any]:"""
SIG_NEW = """    include_decay_modes: bool = True,
    include_external_anchor: bool = True,
    extended_path: str | Path | None = None,
) -> dict[str, Any]:"""

ATTACH_OLD = """        try:
            gates["C10"] = xval.external_anchor_gate(sweep, xval.load_reference())
        except FileNotFoundError:
            gates["C10"] = None
    return {
        "case": CASE_SLUG,"""
ATTACH_NEW = """        try:
            gates["C10"] = xval.external_anchor_gate(sweep, xval.load_reference())
        except FileNotFoundError:
            gates["C10"] = None
    extended_info = attach_extended_band_gates(
        gates,
        sweep,
        Path(path).parent,
        extended_path=extended_path,
    )
    return {
        "case": CASE_SLUG,"""

RETURN_OLD = """        "sweep": sweep.summary(),
        "gates": gates,
        "verdict": case_verdict(gates),
    }"""
RETURN_NEW = """        "sweep": sweep.summary(),
        "extended": extended_info,
        "gates": gates,
        "verdict": case_verdict(gates),
    }"""

RUNCASE_OLD = """        if len(meshes) > 1:
            results["mesh_convergence"] = _mesh_convergence(sweeps[meshes[0]], production)

    results["verdict"] = case_verdict(results["gates"])"""
RUNCASE_NEW = """        if len(meshes) > 1:
            results["mesh_convergence"] = _mesh_convergence(sweeps[meshes[0]], production)
        extended_info = attach_extended_band_gates(
            results["gates"], production, out_path
        )
        if extended_info is not None:
            results["extended"] = extended_info

    results["verdict"] = case_verdict(results["gates"])"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    for name, old, new in (
        ("helpers", MARKER, HELPERS.lstrip("\n") + MARKER),
        ("signature", SIG_OLD, SIG_NEW),
        ("attach", ATTACH_OLD, ATTACH_NEW),
        ("return", RETURN_OLD, RETURN_NEW),
        ("run_case", RUNCASE_OLD, RUNCASE_NEW),
    ):
        if text.count(old) != 1:
            raise SystemExit(f"{name}: marker count = {text.count(old)}")
        text = text.replace(old, new)
    TARGET.write_text(text, encoding="utf-8")
    print("patched", TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())