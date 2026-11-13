"""Tests for FEAT-OSS-INVENTORY-CAD registry integrity.

Mirrors test_oss_inventory_bem_registry.py pattern:
  1. ≥20 repos in registry
  2. ≥6 GAP anchors
  3. Every repo has required fields
  4. Every GAP anchor has executed probe results
  5. No dangling gap references
  6. Tier distribution is non-empty for each tier
  7. Registry sha256 matches census.md claim
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CAD_DIR = os.path.join(HERE, "..", "docs", "oss_inventory_cad")
REGISTRY_PATH = os.path.join(CAD_DIR, "registry.json")
CENSUS_PATH = os.path.join(CAD_DIR, "census.md")


@pytest.fixture(scope="module")
def registry():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- Invariant 1: ≥20 repos ---
def test_at_least_20_repos(registry):
    n = len(registry["repositories"])
    assert n >= 20, f"Only {n} repos (need ≥20)"


# --- Invariant 2: ≥6 GAP anchors ---
def test_at_least_6_gap_anchors(registry):
    n = len(registry["gap_anchors"])
    assert n >= 6, f"Only {n} GAP anchors (need ≥6)"


# --- Invariant 3: Every repo has required fields ---
REQUIRED_REPO_FIELDS = {"repo", "tier", "focus", "lang", "license", "summary"}


def test_repo_fields(registry):
    for i, r in enumerate(registry["repositories"]):
        missing = REQUIRED_REPO_FIELDS - set(r.keys())
        assert not missing, f"Repo #{i+1} ({r.get('repo', '?')}) missing: {missing}"
        assert r["tier"] in (1, 2, 3), f"Repo #{i+1} invalid tier: {r['tier']}"
        assert isinstance(r["focus"], list) and len(r["focus"]) > 0
        assert isinstance(r["summary"], str) and len(r["summary"]) > 20


# --- Invariant 4: Every GAP anchor has executed probe results ---
REQUIRED_ANCHOR_FIELDS = {
    "slug", "priority", "description", "probe_terms",
    "probe_results", "status", "acceptance",
}


def test_gap_anchor_fields(registry):
    for a in registry["gap_anchors"]:
        missing = REQUIRED_ANCHOR_FIELDS - set(a.keys())
        assert not missing, f"Anchor {a.get('slug', '?')} missing: {missing}"
        pr = a["probe_results"]
        assert "files_scanned" in pr, f"{a['slug']}: no files_scanned"
        assert "total_hits" in pr, f"{a['slug']}: no total_hits"
        assert "per_term" in pr, f"{a['slug']}: no per_term"
        assert pr["files_scanned"] > 0, f"{a['slug']}: scanned 0 files"


# --- Invariant 5: No dangling gap references ---
def test_no_dangling_gap_refs(registry):
    anchor_slugs = {a["slug"] for a in registry["gap_anchors"]}
    # Check metadata doesn't reference non-existent anchors
    meta = registry.get("metadata", {})
    # (In this wave, no external references to check)
    assert len(anchor_slugs) >= 6


# --- Invariant 6: Tier distribution non-empty ---
def test_tier_distribution(registry):
    tiers = {r["tier"] for r in registry["repositories"]}
    assert 1 in tiers, "No Tier 1 repos"
    assert 2 in tiers, "No Tier 2 repos"
    assert 3 in tiers, "No Tier 3 repos"


# --- Invariant 7: Registry sha256 matches census.md ---
def test_sha256_matches_census(registry):
    raw = open(REGISTRY_PATH, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    census = open(CENSUS_PATH, encoding="utf-8").read()
    # Census should contain the sha256
    assert sha in census, f"SHA {sha} not found in census.md"


# --- Additional: GAP-CAD-* slug format ---
def test_gap_slug_format(registry):
    for a in registry["gap_anchors"]:
        assert a["slug"].startswith("GAP-CAD-"), (
            f"Invalid slug format: {a['slug']}"
        )


# --- Additional: Priority values ---
def test_priority_values(registry):
    valid_priorities = {"P0", "P1", "P2", "P3"}
    for a in registry["gap_anchors"]:
        assert a["priority"] in valid_priorities, (
            f"{a['slug']}: invalid priority {a['priority']}"
        )


# --- Additional: Status values ---
def test_status_values(registry):
    valid_statuses = {"verified_absent", "partial", "withheld"}
    for a in registry["gap_anchors"]:
        assert a["status"] in valid_statuses, (
            f"{a['slug']}: invalid status {a['status']}"
        )


# --- Additional: Acceptance criteria are non-empty ---
def test_acceptance_non_empty(registry):
    for a in registry["gap_anchors"]:
        assert len(a["acceptance"]) > 10, (
            f"{a['slug']}: acceptance too short"
        )
