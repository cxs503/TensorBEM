"""Integrated diagnostics report combining all diagnostic modules.

This module provides a unified interface to run and report all diagnostics:
- Conditioning diagnostics (condition numbers)
- Quadrature diagnostics (near-singular integrals)
- Mesh quality diagnostics (panel quality metrics)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.conditioning import (
    ConditioningReport,
    estimate_operator_conditions,
)
from tensorbem.geometry3d import TriangleMesh
from tensorbem.mesh_quality import (
    MeshQualityReport,
    diagnose_mesh_quality,
)
from tensorbem.quadrature_diagnostics import (
    NearSingularReport,
    diagnose_near_singular_quadrature,
)


@dataclass
class DiagnosticsReport:
    """Complete diagnostic report for a BEM computation.
    
    Combines conditioning, quadrature, and mesh quality diagnostics
    into a single report object.
    
    Attributes
    ----------
    benchmark_name : str
        Name of the benchmark or computation
    mesh_description : str
        Description of the mesh (e.g., "sphere r=1.0, 128 panels")
    conditioning : ConditioningReport
        Operator conditioning diagnostics
    quadrature : NearSingularReport
        Near-singular quadrature diagnostics
    mesh_quality : MeshQualityReport
        Mesh quality diagnostics
    """
    benchmark_name: str
    mesh_description: str
    conditioning: ConditioningReport
    quadrature: NearSingularReport
    mesh_quality: MeshQualityReport
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "mesh_description": self.mesh_description,
            "conditioning": {
                "mesh_panels": self.conditioning.mesh_panels,
                "single_layer_condition": self.conditioning.single_layer_condition,
                "double_layer_condition": self.conditioning.double_layer_condition,
                "estimation_method": self.conditioning.estimation_method,
                "power_iterations": self.conditioning.power_iterations,
            },
            "quadrature": {
                "mesh_panels": self.quadrature.mesh_panels,
                "total_pairs": self.quadrature.total_pairs,
                "near_singular_pairs": self.quadrature.near_singular_pairs,
                "near_singular_fraction": self.quadrature.near_singular_fraction,
                "threshold_factor": self.quadrature.threshold_factor,
                "min_distance_ratio": self.quadrature.min_distance_ratio,
                "max_distance_ratio": self.quadrature.max_distance_ratio,
                "estimated_max_error": self.quadrature.estimated_max_error,
            },
            "mesh_quality": {
                "mesh_panels": self.mesh_quality.mesh_panels,
                "min_area": self.mesh_quality.min_area,
                "max_area": self.mesh_quality.max_area,
                "area_ratio": self.mesh_quality.area_ratio,
                "min_angle_deg": self.mesh_quality.min_angle_deg,
                "max_angle_deg": self.mesh_quality.max_angle_deg,
                "mean_min_angle_deg": self.mesh_quality.mean_min_angle_deg,
                "min_aspect_ratio": self.mesh_quality.min_aspect_ratio,
                "max_aspect_ratio": self.mesh_quality.max_aspect_ratio,
                "mean_aspect_ratio": self.mesh_quality.mean_aspect_ratio,
                "panels_below_30_deg": self.mesh_quality.panels_below_30_deg,
                "panels_above_5_aspect": self.mesh_quality.panels_above_5_aspect,
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        lines = [
            f"# Diagnostics Report: {self.benchmark_name}",
            "",
            f"**Mesh:** {self.mesh_description}",
            "",
            "## Operator Condition Numbers",
            "",
            f"- **Single-layer operator condition number:** {self.conditioning.single_layer_condition:.2f}",
            f"- **Double-layer operator condition number:** {self.conditioning.double_layer_condition:.2f}",
            f"- **Estimation method:** {self.conditioning.estimation_method}",
            f"- **Power iterations:** {self.conditioning.power_iterations}",
            "",
            "## Near-Singular Quadrature Analysis",
            "",
            f"- **Total integration pairs:** {self.quadrature.total_pairs}",
            f"- **Near-singular pairs:** {self.quadrature.near_singular_pairs} ({self.quadrature.near_singular_fraction*100:.2f}%)",
            f"- **Threshold factor:** {self.quadrature.threshold_factor}",
            f"- **Distance ratio range:** [{self.quadrature.min_distance_ratio:.3f}, {self.quadrature.max_distance_ratio:.3f}]",
            f"- **Estimated max error:** {self.quadrature.estimated_max_error:.3e}",
            "",
            "## Mesh Quality Metrics",
            "",
            f"- **Panel area range:** [{self.mesh_quality.min_area:.4f}, {self.mesh_quality.max_area:.4f}]",
            f"- **Area ratio:** {self.mesh_quality.area_ratio:.2f}",
            f"- **Angle range:** [{self.mesh_quality.min_angle_deg:.1f}°, {self.mesh_quality.max_angle_deg:.1f}°]",
            f"- **Mean minimum angle:** {self.mesh_quality.mean_min_angle_deg:.1f}°",
            f"- **Aspect ratio range:** [{self.mesh_quality.min_aspect_ratio:.2f}, {self.mesh_quality.max_aspect_ratio:.2f}]",
            f"- **Mean aspect ratio:** {self.mesh_quality.mean_aspect_ratio:.2f}",
            f"- **Panels with min angle < 30°:** {self.mesh_quality.panels_below_30_deg}",
            f"- **Panels with aspect ratio > 5:** {self.mesh_quality.panels_above_5_aspect}",
        ]
        return "\n".join(lines)
    
    def summary(self) -> str:
        """Generate human-readable summary of diagnostics."""
        lines = [
            f"Diagnostics Report: {self.benchmark_name}",
            f"Mesh: {self.mesh_description}",
            "",
            "Conditioning:",
            f"  Single-layer operator: {self.conditioning.single_layer_condition:.2f}",
            f"  Double-layer operator: {self.conditioning.double_layer_condition:.2f}",
            f"  Estimation method: {self.conditioning.estimation_method}",
            "",
            "Quadrature:",
            f"  Near-singular pairs: {self.quadrature.near_singular_pairs}/{self.quadrature.total_pairs}",
            f"  Near-singular fraction: {self.quadrature.near_singular_fraction:.4f}",
            f"  Estimated max error: {self.quadrature.estimated_max_error:.4e}",
            "",
            "Mesh Quality:",
            f"  Min angle: {self.mesh_quality.min_angle_deg:.2f}°",
            f"  Mean min angle: {self.mesh_quality.mean_min_angle_deg:.2f}°",
            f"  Max aspect ratio: {self.mesh_quality.max_aspect_ratio:.2f}",
            f"  Panels below 30°: {self.mesh_quality.panels_below_30_deg}",
            f"  Panels above 5:1 aspect: {self.mesh_quality.panels_above_5_aspect}",
        ]
        return "\n".join(lines)


def run_diagnostics(
    mesh: TriangleMesh,
    boundary_potential: np.ndarray | None = None,
    domain: str = "exterior",
    benchmark_name: str = "unnamed",
    mesh_description: str | None = None,
) -> DiagnosticsReport:
    """Run all diagnostics on a mesh and boundary value problem.
    
    Parameters
    ----------
    mesh : TriangleMesh
        The surface mesh
    boundary_potential : np.ndarray, optional
        Boundary potential values (shape: [n_panels]). If None, conditioning
        diagnostics will use a default uniform potential.
    domain : str
        Domain type: "exterior" or "interior"
    benchmark_name : str
        Name for the benchmark
    mesh_description : str, optional
        Description of the mesh. If None, auto-generated from mesh properties.
    
    Returns
    -------
    DiagnosticsReport
        Complete diagnostic report
    """
    # Generate mesh description if not provided
    if mesh_description is None:
        mesh_description = f"{mesh.panel_count} panels, {mesh.vertex_count} vertices"
    
    # If no boundary_potential provided, use uniform potential for conditioning
    if boundary_potential is None:
        boundary_potential = np.ones(mesh.panel_count)
    
    # Run conditioning diagnostics
    conditioning = estimate_operator_conditions(
        mesh, boundary_potential, domain=domain
    )
    
    # Run quadrature diagnostics
    quadrature = diagnose_near_singular_quadrature(mesh)
    
    # Run mesh quality diagnostics
    mesh_quality = diagnose_mesh_quality(mesh)
    
    return DiagnosticsReport(
        benchmark_name=benchmark_name,
        mesh_description=mesh_description,
        conditioning=conditioning,
        quadrature=quadrature,
        mesh_quality=mesh_quality,
    )


def save_diagnostics_report(
    report: DiagnosticsReport,
    output_dir: str | Path,
    prefix: str = "diagnostics",
) -> dict[str, Path]:
    """Save diagnostics report to JSON and Markdown files.
    
    Parameters
    ----------
    report : DiagnosticsReport
        The report to save
    output_dir : str or Path
        Output directory path
    prefix : str
        Filename prefix (default: "diagnostics")
    
    Returns
    -------
    dict[str, Path]
        Dictionary with keys "json" and "markdown" mapping to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f"{prefix}.json"
    md_path = output_dir / f"{prefix}.md"
    
    json_path.write_text(report.to_json())
    md_path.write_text(report.to_markdown())
    
    return {"json": json_path, "markdown": md_path}


def load_diagnostics_report(input_path: str | Path) -> dict[str, Any]:
    """Load diagnostics report from JSON file.
    
    Parameters
    ----------
    input_path : str or Path
        Input JSON file path
    
    Returns
    -------
    dict
        Report data as dictionary
    """
    input_path = Path(input_path)
    return json.loads(input_path.read_text())
