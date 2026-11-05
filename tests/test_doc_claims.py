"""Guards the numbers TensorBEM publishes for the Ursell benchmark.

The published table in ``docs/marine_bem_benchmarks.md`` §4.6 and the headline
values in ``README.md`` are claims about this code.  Before this module they
were only prose: an audit on 2026-10-24 re-measured the 128-panel added-mass
error as :math:`8.51\\times10^{-3}` while the documents stated
:math:`7.72\\times10^{-3}`, and nothing in the suite noticed.

The checks here are deliberately cheap except the last one:

* parsing tests, which make sure the comparison reads the right numbers from
  the right table (a whole-document scan happily mixes the Ursell table with
  the sphere and ellipsoid convergence tables);
* a coverage test, which fails when a document states a number that no claim
  in :data:`tensorbem.doc_claims.CLAIMS` guards;
* a document-versus-artifact test, which fails when a document and the
  measured artifact disagree at the precision the document prints;
* one live benchmark run at the coarsest tabulated mesh, which fails when the
  artifact itself has drifted from what the code computes.

To accept deliberately updated numbers: re-measure with
``python -m tensorbem.doc_claims --write`` and edit the documents to match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorbem import doc_claims
from tensorbem.benchmarks import ursell_cylinder

#: The artifact the document checks compare against.  On a working tree that
#: has run ``python -m tensorbem.doc_claims --write`` this is
#: ``artifacts/ursell_doc_claims.json``; on a fresh clone (``artifacts/`` is
#: gitignored) it is the tracked copy under ``tests/data/``.
ARTIFACT: Path = doc_claims.resolve_artifact()

_SNIPPET = """### 4.6 Ursell 潜没圆柱辐射（自由液面，深水）

| 面元数 | \\(A\\) | 相对误差 | \\(B\\) | 相对误差 | 近场/远场阻尼比 |
|---|---|---|---|---|---|
| 64 | 2587.76 | \\(1.67\\times10^{-2}\\) | 2787.24 | \\(2.54\\times10^{-2}\\) | 1.0106 |
| 参考解（Ursell） | 2545.17 | — | 2718.15 | — | — |
| 基本解方法（MFS，128 面元） | 2544.69 | \\(1.9\\times10^{-4}\\) | 2715.76 | \\(8.8\\times10^{-4}\\) | — |

### 4.7 外域刚体 Neumann 附加质量张量

| 面元数 \\(N\\) | \\(h_{\\max}\\) | \\(V/V_{\\text{exact}}\\) | \\(x\\) 相对误差 |
|---:|---:|---:|---:|
| 320 | 0.48738 | 0.966161 | \\(1.6313\\times10^{-2}\\) |
"""


def test_parses_panel_reference_and_mfs_rows_of_the_ursell_table() -> None:
    found = doc_claims.parse_ursell_table(_SNIPPET)

    assert found["panel_added_mass_64"] == pytest.approx(2587.76)
    assert found["panel_added_mass_relative_error_64"] == pytest.approx(1.67e-2)
    assert found["panel_damping_64"] == pytest.approx(2787.24)
    assert found["panel_damping_relative_error_64"] == pytest.approx(2.54e-2)
    # The last column is a ratio, compared as its deviation from one.
    assert found["near_far_field_relative_difference_64"] == pytest.approx(1.06e-2)

    assert found["reference_added_mass"] == pytest.approx(2545.17)
    assert found["reference_damping"] == pytest.approx(2718.15)
    assert found["mfs_added_mass"] == pytest.approx(2544.69)
    assert found["mfs_added_mass_relative_error"] == pytest.approx(1.9e-4)
    assert found["mfs_damping_relative_error"] == pytest.approx(8.8e-4)


_MULTI_MFS_SNIPPET = """### 4.6 Ursell 潜没圆柱辐射（自由液面，深水）

| 面元数 | \\(A\\) | 相对误差 | \\(B\\) | 相对误差 | 近场/远场阻尼比 |
|---|---|---|---|---|---|
| 64 | 2587.76 | \\(1.67\\times10^{-2}\\) | 2787.24 | \\(2.54\\times10^{-2}\\) | 1.0105 |
| 参考解（Ursell） | 2545.17 | — | 2718.15 | — | — |
| 基本解方法（MFS，64 面元、24 源） | 2543.24 | \\(7.57\\times10^{-4}\\) | 2708.57 | \\(3.53\\times10^{-3}\\) | — |
| 基本解方法（MFS，128 面元、24 源） | 2544.69 | \\(1.89\\times10^{-4}\\) | 2715.76 | \\(8.82\\times10^{-4}\\) | — |
| 基本解方法（MFS，256 面元、24 源） | 2545.05 | \\(4.72\\times10^{-5}\\) | 2717.55 | \\(2.21\\times10^{-4}\\) | — |

### 4.7 外域刚体 Neumann 附加质量张量
"""


def test_mfs_rows_are_claimed_per_mesh_not_by_row_order() -> None:
    """A second MFS row must not hijack the unqualified claim keys.

    The table carries one MFS row per panel count. Keying those rows by row
    order (the previous behaviour) makes the unqualified ``mfs_added_mass``
    whatever row was printed last -- i.e. the finest mesh -- while the
    measurement artifact describes :data:`doc_claims.MFS_PANELS`. The unqualified
    keys are now pinned to that mesh, and every row also gets its own keys.
    """
    found = doc_claims.parse_ursell_table(_MULTI_MFS_SNIPPET)

    assert found["mfs_added_mass"] == pytest.approx(2544.69)
    assert found["mfs_damping"] == pytest.approx(2715.76)
    for panels, (added_mass, damping, added_error, damping_error) in {
        64: (2543.24, 2708.57, 7.57e-4, 3.53e-3),
        128: (2544.69, 2715.76, 1.89e-4, 8.82e-4),
        256: (2545.05, 2717.55, 4.72e-5, 2.21e-4),
    }.items():
        assert found[f"mfs_added_mass_{panels}"] == pytest.approx(added_mass)
        assert found[f"mfs_damping_{panels}"] == pytest.approx(damping)
        assert found[f"mfs_added_mass_relative_error_{panels}"] == pytest.approx(added_error)
        assert found[f"mfs_damping_relative_error_{panels}"] == pytest.approx(damping_error)


def test_only_the_ursell_section_is_read() -> None:
    """Other convergence tables also start with a panel count; they must not leak."""
    found = doc_claims.parse_ursell_table(_SNIPPET)

    assert "panel_added_mass_320" not in found
    panel_rows = {
        int(key[len("panel_added_mass_"):])
        for key in found
        if key.startswith("panel_added_mass_") and key[len("panel_added_mass_"):].isdigit()
    }
    assert panel_rows == {64}


def test_prose_claims_are_read_from_readme_and_abstract() -> None:
    readme = r"the panel solution is within \\(7.7\times10^{-3}\\) in added mass at 128 panels"
    abstract = r"128 面元附加质量与阻尼误差分别为 \(7.7\times10^{-3}\) 与 \(1.3\times10^{-2}\)"

    assert doc_claims.parse_readme_claims(readme) == {
        "headline_added_mass_error_128": pytest.approx(7.7e-3)
    }
    assert doc_claims.parse_readme_claims(abstract) == {
        "headline_added_mass_error_128": pytest.approx(7.7e-3),
        "headline_damping_error_128": pytest.approx(1.3e-2),
    }


def test_artifact_is_present_and_self_describing() -> None:
    payload = json.loads(ARTIFACT.read_text())

    assert payload["benchmark"] == "ursell_submerged_circular_cylinder_deep_water"
    assert payload["parameters"]["submergence"] == 2.0
    assert payload["parameters"]["ka"] == 0.6
    for panels in doc_claims.TABLE_PANELS:
        assert payload["values"][f"panel_added_mass_{panels}"] > 0.0
    # Stored numbers must be finite and ordered like a converging family.
    errors = [
        payload["values"][f"panel_added_mass_relative_error_{panels}"]
        for panels in doc_claims.TABLE_PANELS
    ]
    assert errors[0] > errors[1] > errors[2]


def test_every_number_the_documents_state_is_guarded_by_a_claim() -> None:
    """No unguarded number: each parsed table/prose value must map to a Claim."""
    documented = doc_claims.parse_ursell_table(doc_claims.MARINE_DOC.read_text())
    documented.update(doc_claims.parse_readme_claims(doc_claims.MARINE_DOC.read_text()))
    documented.update(doc_claims.parse_readme_claims(doc_claims.README.read_text()))

    registered = {claim.key for claim in doc_claims.CLAIMS}
    unguarded = sorted(set(documented) - registered)
    assert not unguarded, f"documented but unguarded numbers: {unguarded}"
    # ...and every table panel count the report tabulates must be covered.
    for panels in doc_claims.TABLE_PANELS:
        assert f"panel_added_mass_{panels}" in documented


def test_documents_agree_with_the_measured_artifact() -> None:
    report = doc_claims.check_documents()

    assert report.missing == [], report.describe()
    assert report.failures == [], report.describe()
    assert report.checked >= 20


def test_live_benchmark_reproduces_the_artifact() -> None:
    """The artifact is only useful if the current code still produces it."""
    payload = json.loads(ARTIFACT.read_text())["values"]
    panels = min(doc_claims.TABLE_PANELS)

    result = ursell_cylinder.run(panel_count=panels)

    for key, live in (
        ("panel_added_mass", result["panel_added_mass"]),
        ("panel_damping", result["panel_damping"]),
        ("panel_added_mass_relative_error", result["panel_added_mass_relative_error"]),
        ("panel_damping_relative_error", result["panel_damping_relative_error"]),
        (
            "near_far_field_relative_difference",
            result["near_far_field_relative_difference"],
        ),
    ):
        stored = payload[f"{key}_{panels}"]
        assert live == pytest.approx(stored, rel=1.0e-9), key

# --------------------------------------------------------------------------
# artifact / checkout discovery (added 2026-11-03, m2918)
#
# Regression guard: ``REPO_ROOT`` used to be ``Path(__file__).parents[2]``, which
# is only the repository when the package is imported from ``src/``.  With a
# non-editable install the same expression resolved to ``/usr/lib/python3.11``
# and every document path became a FileNotFoundError, so the document-vs-code
# gate could not run at all outside an editable checkout.
# --------------------------------------------------------------------------
def test_repo_root_is_discovered_and_holds_the_documents() -> None:
    root = doc_claims.repo_root()

    assert root == Path(__file__).resolve().parents[1]
    assert doc_claims.REPO_ROOT == root
    for marker in doc_claims.ROOT_MARKERS:
        assert (root / marker).exists(), marker
    assert doc_claims.MARINE_DOC.exists()
    assert doc_claims.README.exists()


def test_repo_root_honours_the_environment_override(monkeypatch, tmp_path) -> None:
    for marker in doc_claims.ROOT_MARKERS:
        target = tmp_path / marker
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    monkeypatch.setenv(doc_claims.ROOT_ENV_VAR, str(tmp_path))

    assert doc_claims.repo_root() == tmp_path


def test_a_directory_without_the_markers_is_not_a_checkout(tmp_path) -> None:
    assert not doc_claims._looks_like_checkout(tmp_path)
    (tmp_path / "pyproject.toml").touch()
    # One marker is not enough: a bare pyproject.toml is any Python project.
    assert not doc_claims._looks_like_checkout(tmp_path)


def test_a_fresh_clone_falls_back_to_the_tracked_artifact(monkeypatch, tmp_path) -> None:
    pinned = tmp_path / "pinned.json"
    pinned.write_text(ARTIFACT.read_text())
    monkeypatch.setattr(doc_claims, "ARTIFACT", tmp_path / "generated" / "ursell.json")
    monkeypatch.setattr(doc_claims, "PINNED_ARTIFACT", pinned)

    # No generated artifact yet -> the tracked copy is used.
    assert doc_claims.resolve_artifact() == pinned

    # Once the benchmark has been re-measured, the fresh copy wins.
    generated = doc_claims.ARTIFACT
    generated.parent.mkdir(parents=True)
    generated.write_text(ARTIFACT.read_text())
    assert doc_claims.resolve_artifact() == generated

    # Only when neither exists is the caller told to measure.
    generated.unlink()
    pinned.unlink()
    with pytest.raises(FileNotFoundError):
        doc_claims.resolve_artifact()


def test_the_tracked_artifact_is_current_and_self_describing() -> None:
    payload = json.loads(doc_claims.PINNED_ARTIFACT.read_text())

    assert payload["benchmark"] == "ursell_submerged_circular_cylinder_deep_water"
    assert set(payload["values"]) == {
        claim.key for claim in doc_claims.CLAIMS
    }, "tracked artifact and claim registry disagree on the value set"


def test_near_far_ratio_tolerance_follows_the_two_printed_digits() -> None:
    """A correctly rounded 1.0053 must pass; a 5% error must not.

    The ratio is tabulated as ``1.0053``, so the deviation carries two
    significant digits.  The tolerance therefore has to be ~1% of the
    deviation, not the 0.5% used for the three-digit error columns.
    """
    key = "near_far_field_relative_difference_128"
    measured = {key: 5.34137e-3}

    def failures(stated: float) -> list[str]:
        report = doc_claims.compare({key: stated}, measured)
        return [f for f in report.failures if "near/far-field" in f]

    assert failures(5.3e-3) == []
    assert failures(5.6e-3), "a 5.6e-3 claim must be rejected"


def test_installed_copy_outside_a_checkout_names_the_override(tmp_path) -> None:
    """A wheel install has no documents next to ``__file__``.

    Before the discovery rewrite this case produced
    ``FileNotFoundError: /usr/lib/python3.11/docs/marine_bem_benchmarks.md``;
    now it must say what to set.
    """
    import os
    import shutil
    import subprocess
    import sys

    site = tmp_path / "site"
    shutil.copytree(doc_claims.REPO_ROOT / "src" / "tensorbem", site / "tensorbem")
    env = {k: v for k, v in os.environ.items() if k != doc_claims.ROOT_ENV_VAR}
    env["PYTHONPATH"] = str(site)

    failed = subprocess.run(
        [sys.executable, "-c", "import tensorbem.doc_claims"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert doc_claims.ROOT_ENV_VAR in failed.stderr

    env[doc_claims.ROOT_ENV_VAR] = str(doc_claims.REPO_ROOT)
    ok = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tensorbem.doc_claims as d; print(d.REPO_ROOT); print(d.MARINE_DOC.exists())",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.split() == [str(doc_claims.REPO_ROOT), "True"]
