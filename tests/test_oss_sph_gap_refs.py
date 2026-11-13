"""Tests for the OSS-SPH wave-3 gap-reference pack (anchor FEAT-OSS-INVENTORY-SPH).

Three layers, deliberately separated:

1. committed-artifact schema/count checks -- run everywhere, no clones needed;
2. verifier self-test on a synthetic fixture in tmp_path, including a *negative*
   case (tampered quote must be reported as drift) -- a verifier that only ever
   passes proves nothing;
3. reproduction against the real pinned clones, only when they are present
   (``SPH_OSS_CLONES`` env var or /tmp/sph); otherwise skipped with a reason, never
   reported as passed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "oss_sph_inventory"))

import gap_refs as gr  # noqa: E402

EVIDENCE = ROOT / gr.OUT_DIR / gr.EVIDENCE_NAME


@pytest.fixture(scope="module")
def evidence():
    assert EVIDENCE.is_file(), f"missing committed evidence: {EVIDENCE}"
    return json.loads(EVIDENCE.read_text())


def test_schema_and_claim_count(evidence):
    assert evidence["anchor"] == "FEAT-OSS-INVENTORY-SPH"
    assert evidence["channel"] == "CODE-LEVEL"
    claims = evidence["claims"]
    assert len(claims) >= 20, f"expected >=20 code-level claims, got {len(claims)}"
    assert len({c["repo"] for c in claims}) >= 3
    seen = set()
    for c in claims:
        assert re.fullmatch(r"[A-Z]{3}-\d{2}", c["id"]), c["id"]
        assert c["id"] not in seen, f"duplicate id {c['id']}"
        seen.add(c["id"])
        assert re.fullmatch(r"[0-9a-f]{40}", c["commit"]), c["id"]
        assert re.fullmatch(r"[0-9a-f]{64}", c["file_sha256"]), c["id"]
        assert c["line"] >= 1 and c["quote"].strip(), c["id"]
        assert c["statement"] and c["port_recipe"], c["id"]
        # every claim must point at a file the module pins, not an ad-hoc path
        assert any(c["path"] == s["path"] for s in gr.SPECS if s["id"] == c["id"])


def test_every_gap_anchor_has_evidence(evidence):
    """The four open capability gaps must each carry at least two claims."""
    open_gaps = [
        "GAP-SPH-MPI-DOMAIN",
        "GAP-SPH-ISPH-PROJECTION",
        "GAP-SPH-AKINCI-BOUNDARY",
        "GAP-SPH-SPS-LES",
    ]
    per_gap = evidence["claims_per_gap"]
    for gap in open_gaps:
        assert per_gap.get(gap, 0) >= 2, f"{gap} has {per_gap.get(gap, 0)} claims"
    # the recorded per-gap totals must actually equal the claim list
    recount = {}
    for c in evidence["claims"]:
        recount[c["gap"]] = recount.get(c["gap"], 0) + 1
    assert recount == per_gap


def test_uncovered_gap_is_declared_not_padded(evidence):
    """MULTIPHASE-SHARP has no wave-3 code evidence; it must be declared explicitly."""
    uncovered = evidence["gaps_not_covered"]
    assert "GAP-SPH-MULTIPHASE-SHARP" in uncovered
    assert any(c["gap"] == "GAP-SPH-MULTIPHASE-SHARP" for c in evidence["claims"]) is False


def _write_fixture(tmp: Path) -> Path:
    (tmp / "jax-sph").mkdir()
    f = tmp / "jax-sph" / "solver.py"
    f.write_text("import x\n\ndef delta_term():\n    return c_ref * delta * diff\n")
    return f


def test_verifier_accepts_true_record(tmp_path, monkeypatch):
    _write_fixture(tmp_path)
    spec = dict(id="TST-01", repo="jax-sph", gap="GAP-SPH-ISPH-PROJECTION",
                path="solver.py", anchor=r"c_ref \* delta \* diff",
                statement="fixture", port="fixture")
    monkeypatch.setitem(gr.REPOS, "jax-sph", dict(gr.REPOS["jax-sph"]))
    monkeypatch.setattr(gr, "SPECS", [spec])
    rec = gr.extract(tmp_path, spec)
    assert rec["line"] == 4 and rec["quote"] == "return c_ref * delta * diff"

    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"claims": [rec]}))
    n, problems = gr.verify(tmp_path, ev)
    assert (n, problems) == (1, [])


def test_verifier_rejects_tampered_quote(tmp_path, monkeypatch):
    """Anti-green-wash: a record whose quote was edited must fail verification."""
    _write_fixture(tmp_path)
    spec = dict(id="TST-02", repo="jax-sph", gap="GAP-SPH-ISPH-PROJECTION",
                path="solver.py", anchor=r"c_ref \* delta \* diff",
                statement="fixture", port="fixture")
    monkeypatch.setitem(gr.REPOS, "jax-sph", dict(gr.REPOS["jax-sph"]))
    monkeypatch.setattr(gr, "SPECS", [spec])
    rec = gr.extract(tmp_path, spec)
    rec["quote"] = "return c_ref * delta * diffusion"  # plausible, but invented
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"claims": [rec]}))
    _, problems = gr.verify(tmp_path, ev)
    assert len(problems) == 1 and "quote drifted" in problems[0]


def test_verifier_errors_on_drifted_anchor(tmp_path, monkeypatch):
    """A renamed upstream construct must raise, not silently drop the row."""
    _write_fixture(tmp_path)
    spec = dict(id="TST-03", repo="jax-sph", gap="GAP-SPH-ISPH-PROJECTION",
                path="solver.py", anchor=r"not_present_anywhere",
                statement="fixture", port="fixture")
    monkeypatch.setattr(gr, "SPECS", [spec])
    with pytest.raises(LookupError):
        gr.extract(tmp_path, spec)


def _clone_root() -> Path | None:
    import os

    root = Path(os.environ.get("SPH_OSS_CLONES", "/tmp/sph"))
    if not root.is_dir():
        return None
    if not all((root / d / ".git").exists() for d in gr.REPOS):
        return None
    return root


@pytest.mark.skipif(_clone_root() is None, reason="pinned clones not present (set SPH_OSS_CLONES)")
def test_verify_against_real_clones():
    root = _clone_root()
    n, problems = gr.verify(root, EVIDENCE)
    assert n >= 20 and not problems, problems