"""The published cp-linear artefacts must stay reproducible, and its figures linked.

``docs/cathodic_protection_linear`` publishes the results of
:mod:`tensorbem.cathodic_protection` (``results.json``,
``sphere-spreading-resistance.json``, ``hull_field.json``) together with three
figures and ``REPORT.md``.  All of it is produced by
``docs/cathodic_protection_linear/make_figures.py``; nothing stops the
publication from drifting away from the solver, and a stale artefact is
indistinguishable from a hand-edited one (the report quotes every number from
these files, so a stale file silently rewrites the published claims).

These tests close that gap without re-implementing the pipeline: they load the
committed generator, point its output at a temporary directory, run it, and
require the regenerated artefacts to match the committed ones.  They also use
the shared figure probe (:mod:`tests._figure_integrity`) so a blank render or a
figure that is no longer linked from the report cannot pass, and they resolve
every image link of the report against the filesystem -- the links are relative
to ``REPORT.md`` itself, which is what a reader (and every markdown renderer)
does.

Costs about one minute (the 1280-panel conducting-sphere level dominates), i.e.
the same order as the rest of ``tests/test_cathodic_protection.py``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import re

import pytest

from _figure_integrity import assert_reproducible_figures

DOCS_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "docs"
    / "cathodic_protection_linear"
)
REPORT_PATH = DOCS_DIR / "REPORT.md"
MAKE_FIGURES_PATH = DOCS_DIR / "make_figures.py"

NUMERIC_ARTEFACTS = (
    "results.json",
    "sphere-spreading-resistance.json",
    "hull_field.json",
)

EXPECTED_FIGURES = (
    "hull_potential_field.png",
    "sphere_convergence.png",
    "dnv_comparison.png",
)

# Free-text fields that are prose, not measurements; comparing them adds
# nothing (the test is about the numbers) but they are still required to exist.
PROSE_KEYS = {"note", "case", "study", "metric", "quantity"}

IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _load_generator():
    """Import ``make_figures.py`` from the docs directory (it is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "cp_linear_make_figures", MAKE_FIGURES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> pathlib.Path:
    """Run the documented generator once and return its output directory."""
    out = tmp_path_factory.mktemp("cp_linear_regenerated")
    generator = _load_generator()
    generator.OUT = out
    generator.FIG = out / "figures"
    generator.main()
    return out


def _compare(published, produced, path="", rtol=1e-12, problems=None):
    problems = [] if problems is None else problems
    if isinstance(published, dict):
        if not isinstance(produced, dict):
            problems.append(f"{path}: type {type(produced).__name__} != dict")
            return problems
        for key, value in published.items():
            if key not in produced:
                problems.append(f"{path}/{key}: missing in regenerated artefact")
                continue
            if key in PROSE_KEYS:
                continue
            _compare(value, produced[key], f"{path}/{key}", rtol, problems)
        for key in produced:
            if key not in published:
                problems.append(f"{path}/{key}: new field not in published artefact")
    elif isinstance(published, list):
        if not isinstance(produced, list) or len(produced) != len(published):
            problems.append(
                f"{path}: list length {len(published)} != "
                f"{len(produced) if isinstance(produced, list) else produced}"
            )
            return problems
        for index, (a, b) in enumerate(zip(published, produced)):
            _compare(a, b, f"{path}[{index}]", rtol, problems)
    elif isinstance(published, bool) or published is None or isinstance(published, str):
        if published != produced:
            problems.append(f"{path}: {published!r} != {produced!r}")
    elif isinstance(published, (int, float)):
        if isinstance(produced, bool) or not isinstance(produced, (int, float)):
            problems.append(f"{path}: {published!r} != {produced!r}")
            return problems
        if not math.isclose(
            float(published), float(produced), rel_tol=rtol, abs_tol=1e-15
        ):
            problems.append(f"{path}: published {published!r} regenerated {produced!r}")
    else:  # pragma: no cover - defensive
        problems.append(f"{path}: unhandled type {type(published).__name__}")
    return problems


@pytest.mark.parametrize("name", NUMERIC_ARTEFACTS)
def test_published_numeric_artefact_regenerates(regenerated, name):
    """The committed numbers must be what the committed code still produces."""
    published_path = DOCS_DIR / name
    produced_path = regenerated / name
    assert published_path.is_file(), published_path
    assert produced_path.is_file(), f"generator did not write {name}"

    published = json.loads(published_path.read_text(encoding="utf-8"))
    produced = json.loads(produced_path.read_text(encoding="utf-8"))
    problems = _compare(published, produced, path=name)
    assert not problems, "published artefact has drifted from the generator:\n" + "\n".join(
        problems[:20]
    )


def test_regenerated_figures_are_real_and_linked(regenerated):
    """Every figure must be regenerable, non-trivial, and still cited by the report."""
    report = REPORT_PATH.read_text(encoding="utf-8")
    figures = [regenerated / "figures" / name for name in EXPECTED_FIGURES]
    assert_reproducible_figures(figures, minimum=3, embedded_in=report)


def test_every_report_figure_link_resolves():
    """Image links are relative to the report file; a broken one is a dead figure."""
    text = REPORT_PATH.read_text(encoding="utf-8")
    links = [link.strip() for link in IMAGE_LINK.findall(text)]
    assert links, "report embeds no figures"
    missing = []
    for link in links:
        target = link.split("#")[0].split(" ")[0]
        assert not target.startswith(("http://", "https://")), (
            f"external figure link {target!r}: the report must ship its own figures"
        )
        path = (REPORT_PATH.parent / target).resolve()
        if not path.is_file() or path.stat().st_size <= 10_000:
            missing.append(f"{target} -> {path}")
    assert not missing, "report links figures that do not exist / are empty:\n" + "\n".join(
        missing
    )


def test_sphere_exact_anchor_algebra():
    """The anchor's 'exact' column is the closed form, independently of the solver."""
    sphere = json.loads(
        (DOCS_DIR / "sphere-spreading-resistance.json").read_text(encoding="utf-8")
    )
    radius = sphere["radius_m"]
    potential = sphere["potential_V"]
    sigma = sphere["conductivity_S_m"]
    assert sphere["exact_current_A"] == pytest.approx(
        4.0 * math.pi * sigma * radius * potential, rel=1e-12
    )
    assert sphere["exact_spreading_resistance_ohm"] == pytest.approx(
        1.0 / (4.0 * math.pi * sigma * radius), rel=1e-12
    )
    # The documented Ohm's-law defect (q reported without sigma) is exactly this
    # factor, and the report must keep stating it in the same place.
    assert sphere["unscaled_current_error_percent"] == pytest.approx(
        100.0 * (1.0 - 1.0 / sigma), rel=1e-12
    )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])