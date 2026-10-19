import pytest

from tensorbem.benchmarks.unit_disk import plot_field_comparison


def test_field_comparison_plot_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "field_comparison.png"

    plot_field_comparison(24, output, grid_size=31)

    assert output.is_file()
    assert output.stat().st_size > 0
