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

ARTIFACT: Path = doc_claims.ARTIFACT

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