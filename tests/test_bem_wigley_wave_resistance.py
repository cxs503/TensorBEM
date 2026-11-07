"""Test for Wigley wave resistance verified benchmark.

This test ensures the Wigley wave resistance verification is reproducible
and all criteria are met.
"""

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks.bem_wigley_wave_resistance import (
    run_wigley_wave_resistance_verification,
)


class TestWigleyWaveResistanceVerification:
    """Test suite for Wigley wave resistance verification."""

    def test_all_criteria_passed(self):
        """Test that all verification criteria are met."""
        results = run_wigley_wave_resistance_verification()
        assert results["all_criteria_passed"], "Not all criteria passed"

    @pytest.mark.parametrize("froude", [0.2, 0.3, 0.4])
    def test_wave_resistance_error(self, froude):
        """Test wave resistance error < 5% at each Froude number."""
        results = run_wigley_wave_resistance_verification(
            froude_numbers=(froude,)
        )
        case = results["cases"][0]
        error = case["rel_error"]
        assert error < 0.05, f"Wave resistance error {error:.4%} >= 5% at Fr={froude}"

    @pytest.mark.parametrize("froude", [0.2, 0.3, 0.4])
    def test_gci_criterion(self, froude):
        """Test GCI < 3% at each Froude number."""
        results = run_wigley_wave_resistance_verification(
            froude_numbers=(froude,)
        )
        case = results["cases"][0]
        gci = case["gci"]["gci_fine"]
        assert gci < 0.03, f"GCI {gci:.4%} >= 3% at Fr={froude}"

    def test_geometry_properties(self):
        """Test that hull geometry properties are correct."""
        results = run_wigley_wave_resistance_verification()
        
        # Wigley III: Cb = 4/9, Cm = Cp = 2/3
        assert abs(results["reference"]["Cb"] - 4.0/9.0) < 1e-10
        assert abs(results["reference"]["Cm"] - 2.0/3.0) < 1e-10
        assert abs(results["reference"]["Cp"] - 2.0/3.0) < 1e-10
        
        # Volume = 4LBT/9
        L = results["geometry"]["L"]
        B = results["geometry"]["B"]
        T = results["geometry"]["T"]
        expected_volume = 4.0 * L * B * T / 9.0
        assert abs(results["reference"]["volume"] - expected_volume) < 1e-10

    def test_wave_resistance_positive(self):
        """Test that wave resistance is positive at all Froude numbers."""
        results = run_wigley_wave_resistance_verification()
        for case in results["cases"]:
            assert case["Rw_reference"] > 0
            assert case["Rw_production"] > 0

    def test_wave_resistance_increases_with_froude(self):
        """Test that wave resistance increases with Froude number."""
        results = run_wigley_wave_resistance_verification()
        Rw_values = [case["Rw_reference"] for case in results["cases"]]
        for i in range(1, len(Rw_values)):
            assert Rw_values[i] > Rw_values[i-1], \
                "Wave resistance should increase with Froude number"

    def test_artifacts_exist(self):
        """Test that verification artifacts can be written."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            
            # Run verification and write results
            results = run_wigley_wave_resistance_verification()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = output_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            
            assert results_path.exists()
            
            # Verify JSON is valid
            with open(results_path) as f:
                loaded = json.load(f)
            
            assert loaded["study"] == "bem_wigley_wave_resistance"
            assert loaded["all_criteria_passed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
