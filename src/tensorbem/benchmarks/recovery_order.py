"""Recovery-only order of the superconvergent tangential-gradient patch fit.

The documentation of :mod:`tensorbem.recovery` and of
:mod:`tensorbem.benchmarks.sphere_velocity` describes the least-squares patch
fit as *second-order accurate* in the panel size.  The convergence tests that
exist measure the **composite** pipeline (boundary integral solve + recovery),
so the order they report mixes the ``O(h**2)`` trace error of the
constant-panel solve with the order of the recovery itself.  This benchmark
isolates the recovery.

An analytic harmonic field ``u`` is sampled at the panel centroids of a
flat-faceted icosphere, the tangential gradient is recovered with
:func:`tensorbem.recovery.recover_tangent_gradient`, and the result is
compared with two analytic references:

``panel_target``
    ``P_p grad(u)(c)``, the ambient gradient projected onto the *panel plane*.
    This is what the patch fit is built to approximate, so it measures the
    operator's own error.

``smooth_target``
    The same projection onto the *smooth sphere* normal.  On a flat-faceted
    sphere the two references differ by ``O(h)``, so the pair separates an
    operator error from a faceting/definition error.

A third column tests *why* the measured order is what it is.  The fit is
linear in the data and reproduces affine fields exactly, so its error equals
the gradient the same fit returns for the curvature residual

    q_j = u(x_j) - u(x_i) - grad(u)(x_i) . (x_j - x_i) = O(h**2):

``predicted`` evaluates ``fit(q)`` independently of the library routine.  When
``prediction_residual`` is at round-off the error is caused by the third
moment of the patch offsets (patch asymmetry) alone -- not by quadrature,
faceting, or the choice of reference.

Fields (all harmonic, so each case is an admissible velocity potential):

    u1 = x * y                       (quadratic, harmonic)
    u2 = z**2 - (x**2 + y**2) / 2    (quadratic, axisymmetric, harmonic)
    u3 = x**3                        (cubic control)

Run: python -m tensorbem.benchmarks.recovery_order --subdivisions 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..geometry3d import TriangleMesh, icosphere
from ..recovery import panel_patches, recover_tangent_gradient

Field = tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]

#: name -> (value, ambient gradient)
FIELD_SPECS: dict[str, Field] = {
    "x_times_y": (
        lambda p: p[:, 0] * p[:, 1],
        lambda p: np.stack([p[:, 1], p[:, 0], np.zeros(p.shape[0])], axis=1),
    ),
    "axisym_quadratic": (
        lambda p: p[:, 2] ** 2 - 0.5 * (p[:, 0] ** 2 + p[:, 1] ** 2),
        lambda p: np.stack([-p[:, 0], -p[:, 1], 2.0 * p[:, 2]], axis=1),
    ),
    "cubic_x3": (
        lambda p: p[:, 0] ** 3,
        lambda p: np.stack(
            [3.0 * p[:, 0] ** 2, np.zeros(p.shape[0]), np.zeros(p.shape[0])], axis=1
        ),
    ),
}


def project(vectors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Remove the component of each row of ``vectors`` along ``normals``."""
    return vectors - np.sum(vectors * normals, axis=1)[:, None] * normals


def relative_error(error: np.ndarray, reference: np.ndarray) -> float:
    """L2 norm of ``error`` relative to the L2 norm of ``reference``."""
    return math.sqrt(float(np.sum(error**2)) / float(np.sum(reference**2)))


def targets(mesh: TriangleMesh, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Panel-plane and smooth-surface projections of the analytic gradient."""
    ambient = FIELD_SPECS[field][1](mesh.centroids)
    radial = mesh.centroids / np.linalg.norm(mesh.centroids, axis=1)[:, None]
    return project(ambient, mesh.normals), project(ambient, radial)


def predicted_fit_error(mesh: TriangleMesh, field: str) -> np.ndarray:
    """Gradient that the same least-squares fit returns for the residual ``q``."""
    value, grad = FIELD_SPECS[field]
    patches = panel_patches(mesh)
    width = max(len(patch) for patch in patches)
    centroids = mesh.centroids
    scale = mesh.h_mean
    exact = grad(centroids)

    index = np.zeros((mesh.panel_count, width), dtype=np.int64)
    mask = np.zeros((mesh.panel_count, width))
    for row, patch in enumerate(patches):
        index[row, : len(patch)] = patch
        mask[row, : len(patch)] = 1.0
    offsets = (centroids[index] - centroids[:, None, :]) / scale
    design = np.stack(
        [mask, offsets[..., 0] * mask, offsets[..., 1] * mask, offsets[..., 2] * mask],
        axis=2,
    )
    patch_values = value(centroids[index].reshape(-1, 3)).reshape(
        mesh.panel_count, width
    )
    residual = (
        patch_values
        - value(centroids)[:, None]
        - np.einsum("fpi,fi->fp", offsets * scale, exact)
    ) * mask
    # The constant degree of freedom couples to the gradient block, so the
    # whole normal-equation right-hand side enters.
    right = np.zeros((mesh.panel_count, 4))
    right[:, 0] = np.sum(residual, axis=1)
    right[:, 1:] = np.einsum("fpi,fp->fi", design[:, :, 1:], residual)
    gram = np.einsum("fpi,fpq->fiq", design, design)
    solution = np.linalg.solve(gram, right[:, :, None])[:, :, 0]
    return project(solution[:, 1:] / scale, mesh.normals)


def _orders(levels: Sequence[dict], key: str) -> list[dict]:
    out = []
    for lower, upper in zip(levels, levels[1:]):
        ratio = lower[key] / upper[key]
        out.append(
            {
                "from_panels": lower["panels"],
                "to_panels": upper["panels"],
                "error_ratio": ratio,
                "observed_order": math.log(ratio) / math.log(2.0),
            }
        )
    return out


def study(
    subdivisions: Sequence[int] = (1, 2, 3, 4),
    fields: Sequence[str] = tuple(FIELD_SPECS),
) -> dict:
    """Tabulate the recovery-only error over an icosphere refinement ladder."""
    rows = []
    for field in fields:
        value, _ = FIELD_SPECS[field]
        levels = []
        for sub in subdivisions:
            mesh = icosphere(int(sub), radius=1.0)
            panel_target, smooth_target = targets(mesh, field)
            recovered = recover_tangent_gradient(mesh, value(mesh.centroids))
            predicted = predicted_fit_error(mesh, field)
            measured_error = recovered - panel_target
            levels.append(
                {
                    "subdivisions": int(sub),
                    "panels": mesh.panel_count,
                    "h_mean": float(mesh.h_mean),
                    "panel_target": relative_error(measured_error, panel_target),
                    "smooth_target": relative_error(
                        recovered - smooth_target, smooth_target
                    ),
                    "target_definition_gap": relative_error(
                        panel_target - smooth_target, panel_target
                    ),
                    "predicted": relative_error(predicted, panel_target),
                    "prediction_residual": relative_error(
                        predicted - measured_error, measured_error
                    ),
                    "max_abs_error": float(np.max(np.abs(measured_error))),
                }
            )
        rows.append(
            {
                "field": field,
                "levels": levels,
                "orders": {
                    key: _orders(levels, key)
                    for key in (
                        "panel_target",
                        "smooth_target",
                        "target_definition_gap",
                        "predicted",
                    )
                },
            }
        )
    return {
        "study": "recovery_order",
        "quantity": "tangential gradient recovered by the least-squares patch fit",
        "mesh": "flat-faceted icosphere, radius 1",
        "panels": [row["panels"] for row in rows[0]["levels"]],
        "minimum_patch_size": 4,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Recovery-only convergence of the tangential-gradient patch fit."
    )
    parser.add_argument("--json", type=Path, default=None, help="write the table here")
    parser.add_argument(
        "--subdivisions", type=int, nargs="+", default=[1, 2, 3, 4]
    )
    args = parser.parse_args(argv)
    payload = study(tuple(args.subdivisions))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()