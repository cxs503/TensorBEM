"""Pinned mutants: the defects the suite is *required* to kill.

The mutation audit (``docs/mutation_audit_tensorbem.md``) finished with four
survivors that it classified as genuine coverage gaps and handed to a follow-up
mission.  Backfilling a test for each of them settles nothing by itself: a test
written today can be weakened or deleted tomorrow, and the gap reopens silently.
This module closes the loop by pinning the *exact* mutants and re-running them
against the named test on demand:

    PYTHONPATH=src python3 -m tensorbem.benchmarks.mutation_regression --out DIR

For every pin it

1. re-enumerates the module's AST mutants (so the pin is expressed as an
   operator plus the original source span, not as a line number that rots),
2. writes the mutant into a private copy of ``src/tensorbem``,
3. runs *only* the test that is supposed to notice, in that shadow tree, and
4. requires the result to be ``killed``.

A pin that can no longer be located, or a mutant that now survives, is reported
as a failure -- the suite would then be weaker than the audit claims.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from . import mutation_audit as _audit

#: ``occurrence`` disambiguates identical source spans inside one function.
@dataclass(frozen=True)
class Pin:
    """A mutant that must stay dead, and the test that kills it."""

    module: str
    function: str
    operator: str
    original: str
    occurrence: int
    test: str
    gap: str

    @property
    def label(self) -> str:
        return f"{self.module}:{self.function}:{self.operator}:{self.original}"


#: The four survivors of ``docs/mutation_audit_tensorbem.md`` §5.
PINS: tuple[Pin, ...] = (
    Pin(
        module="tensorbem.hydrostatic",
        function="compute_waterplane_properties",
        operator="swap_binop",
        original="sign * np.sum((x + x_next) * cross) / 6.0",
        occurrence=0,
        test=(
            "tests/test_mutation_gaps.py::"
            "test_eccentric_barge_waterplane_centroid_and_second_moments"
        ),
        gap="waterplane first moment -> centroid (off-centre waterplane)",
    ),
    Pin(
        module="tensorbem.hydrostatic",
        function="compute_waterplane_properties",
        operator="shift_constant",
        original="0.0",
        occurrence=0,
        test=(
            "tests/test_mutation_gaps.py::"
            "test_fully_submerged_hull_reports_an_exactly_empty_waterplane"
        ),
        gap="no-waterplane branch returns area 1.0 instead of 0.0",
    ),
    Pin(
        module="tensorbem.hydrostatic",
        function="compute_displaced_volume_and_buoyancy",
        operator="shift_constant",
        original="3",
        occurrence=0,
        test=(
            "tests/test_mutation_gaps.py::"
            "test_degenerate_displaced_volume_returns_a_three_vector"
        ),
        gap="zero-displacement exit returns a length-4 centre (first exit)",
    ),
    Pin(
        module="tensorbem.hydrostatic",
        function="compute_displaced_volume_and_buoyancy",
        operator="shift_constant",
        original="3",
        occurrence=1,
        test=(
            "tests/test_mutation_gaps.py::"
            "test_degenerate_displaced_volume_returns_a_three_vector"
        ),
        gap="zero-displacement exit returns a length-4 centre (second exit)",
    ),
    Pin(
        module="tensorbem.free_surface",
        function="_source_gradient_terms",
        operator="swap_binop",
        original="2.0 * (elementary_cos + frequency * cos_integral)",
        occurrence=0,
        test=(
            "tests/test_mutation_gaps.py::"
            "test_depth_derivative_matches_finite_differences_and_the_field_gradient"
        ),
        gap="depth derivative (dG/dA) had no consumer and no test",
    ),
)


def _module_path(module: str) -> Path:
    relative = module.split(".")
    return _audit.SRC_ROOT.joinpath(*relative).with_suffix(".py")


def locate(pin: Pin) -> _audit.Mutant:
    """Return the unique mutant ``pin`` describes, or raise with a diagnosis."""
    path = _module_path(pin.module)
    source = path.read_text()
    candidates = [
        mutant
        for mutant in _audit.enumerate_mutants(pin.module, source, path)
        if mutant.function == pin.function
        and mutant.operator == pin.operator
        and mutant.original == pin.original
    ]
    if not candidates:
        raise LookupError(
            f"pin is stale: {pin.label} inside {pin.function} no longer exists"
        )
    if pin.occurrence >= len(candidates):
        raise LookupError(
            f"pin is stale: {pin.label} has {len(candidates)} candidate(s), "
            f"occurrence {pin.occurrence} requested"
        )
    return candidates[pin.occurrence]


def verify(
    pins: tuple[Pin, ...] = PINS, timeout: float = 600.0
) -> dict[str, object]:
    """Apply each pin's mutant in a shadow tree and require the test to fail."""
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="tensorbem-mutation-pins-") as staging:
        shadow_src = _audit._shadow_package(Path(staging))
        _audit._check_isolation(shadow_src)
        for pin in pins:
            shadow_file = shadow_src.joinpath(*pin.module.split(".")).with_suffix(".py")
            pristine = shadow_file.read_text()
            try:
                mutant = locate(pin)
            except LookupError as error:
                rows.append(
                    {
                        "pin": asdict(pin),
                        "verdict": "stale",
                        "detail": str(error),
                    }
                )
                continue
            shadow_file.write_text(_audit.apply_mutant(pristine, mutant))
            try:
                returncode, output = _audit.run_tests(
                    shadow_src, (pin.test,), timeout
                )
            finally:
                shadow_file.write_text(pristine)
            verdict, failing = _audit._classify(returncode, output)
            rows.append(
                {
                    "pin": asdict(pin),
                    "mutant": mutant.key,
                    "mutated_source": mutant.summary,
                    "verdict": verdict,
                    "failing_test": failing,
                }
            )
    passed = all(row["verdict"] == "killed" for row in rows)
    return {
        "pins": len(rows),
        "killed": sum(1 for row in rows if row["verdict"] == "killed"),
        "all_killed": passed,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    payload = verify(timeout=args.timeout)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "mutation_pins.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    for row in payload["rows"]:
        pin = row["pin"]
        label = f"{pin['module']}:{pin['function']}:{pin['operator']}"
        print(f"{row['verdict']:>8}  {label}  <- {pin['test']}")
    print(
        f"{payload['killed']}/{payload['pins']} pinned mutants killed "
        f"(all_killed={payload['all_killed']})"
    )
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()