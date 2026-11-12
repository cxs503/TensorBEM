"""Invariants of the OSS BEM census (FEAT-OSS-INVENTORY-BEM).

These are *assertions on evidence*, not smoke tests: every repo that carries a
capability claim must point at committed evidence, every gap anchor must carry an
executed probe, and the human-readable census.md must be cryptographically tied
to the JSON it summarises (so the table cannot drift from the data).

Offline: the census itself needs network, this validation does not.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(HERE, "docs", "oss_inventory_bem")
REG = os.path.join(DOCS, "registry.json")

GAP_STATUS = {"verified_absent", "partially_present_review_required",
              "limitation_documented"}
TIERS = {"A", "B", "C"}

pytestmark = pytest.mark.skipif(not os.path.exists(REG),
                                reason="OSS inventory not built in this checkout")


@pytest.fixture(scope="module")
def reg():
    with open(REG) as f:
        return json.load(f)


def test_anchor_and_evidence_floor(reg):
    assert reg["schema_version"] == "1.0"
    assert reg["anchor"] == "FEAT-OSS-INVENTORY-BEM"
    census = reg["census"]
    # the ledger anchor demands >=20 graded repos and >=6 GAP-BEM-* proposals
    assert census["registered_and_graded"] >= 20, census
    assert len(reg["gap_anchors"]) >= 6
    assert census["candidates_scanned"] >= census["registered_and_graded"]
    assert census["tier_counts"]["A"] + census["tier_counts"]["B"] == \
        census["registered_and_graded"]


def test_every_repo_traceable(reg):
    for r in reg["repos"]:
        assert r["tier"] in {"A", "B"}, r["full_name"]
        assert r["html_url"].startswith("https://github.com/")
        assert r["pushed_at"], f"{r['full_name']}: no live pushed_at"
        assert "license" in r, f"{r['full_name']}: license field missing"
        assert r["license_note"], f"{r['full_name']}: license note required"
        assert r["domain"] and r["method_family"]
        assert r["evidence_confidence"] in {"readme", "description_only"}
        if r["evidence_confidence"] == "readme":
            # a readme claim is only allowed with a stored, fingerprinted README
            ev = r["readme_evidence"]
            assert ev["readme_available"] is True, r["full_name"]
            assert ev["sha256_16"] and os.path.exists(os.path.join(DOCS, ev["file"])), \
                r["full_name"]
        else:
            # unverified claims must be declared as such, never silently upgraded
            assert r["evidence_confidence"] == "description_only"


def test_tier_c_screened_with_reason(reg):
    for r in reg["screened_out"]:
        assert r["tier"] == "C"
        assert r["exclude_reason"], r["full_name"]


def test_gap_anchors_are_measured_and_quantified(reg):
    slugs = {a["slug"] for a in reg["gap_anchors"]}
    assert len(slugs) == len(reg["gap_anchors"]), "duplicate anchor slug"
    for a in reg["gap_anchors"]:
        assert a["slug"].startswith("GAP-BEM-")
        assert a["gap_status"] in GAP_STATUS
        # quantified acceptance: a number must be present, prose alone is not enough
        assert any(ch.isdigit() for ch in a["quantified_target"]), a["slug"]
        assert len(a["acceptance"]) >= 3, a["slug"]
        assert a["source_repos"], a["slug"]
        p = a["gap_probe"]
        assert p["scanned_files"] > 0
        assert p["total_hits"] == sum(v["count"] for v in p["hits"].values())
        assert set(p["hits"]) == set(a["probe_terms"])
        # absence claims must be consistent with the probe it ships
        if a["gap_status"] == "verified_absent":
            assert p["total_hits"] == 0, f"{a['slug']}: claims absent but probe hit"


def test_no_dangling_gap_references(reg):
    slugs = {a["slug"] for a in reg["gap_anchors"]}
    for r in reg["repos"]:
        for g in r["gaps_addressed"]:
            assert g in slugs, f"{r['full_name']} -> {g}"


def test_census_md_is_tied_to_registry(reg):
    digest = hashlib.sha256(open(REG, "rb").read()).hexdigest()[:16]
    md = open(os.path.join(DOCS, "census.md")).read()
    # the generated table must not be hand-edited after the fact
    assert digest in md, "census.md is stale against registry.json"
    for a in reg["gap_anchors"]:
        assert a["slug"] in md
    assert "禁止手改" in md


def test_provenance_present(reg):
    for key, rel in reg["provenance"].items():
        if rel.endswith("/"):
            assert os.path.isdir(os.path.join(HERE, rel)), key
        elif "*" in rel:
            import fnmatch
            base, pat = os.path.split(rel)
            d = os.path.join(HERE, base)
            assert os.path.isdir(d) and fnmatch.filter(os.listdir(d), pat), \
                f"{key}: no file matching {pat} in {d}"
        else:
            assert os.path.exists(os.path.join(HERE, rel)), f"{key}: {rel}"
    assert reg["validation"]["ok"] is True, reg["validation"]["problems"]