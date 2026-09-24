"""Run the independent shifted point-oracle certificate on a native checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from polish_recovery_refine import _load_state
from vmex.core.strong_force import certify_strong_force


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    state = _load_state(args.input_state, prefix="accepted")
    report = certify_strong_force(state)
    payload = {
        "schema": "vmex.polish-recovery-certificate/1",
        "method": "independent shifted point oracle",
        "source": {
            "native_state": str(args.input_state),
            "native_state_sha256": _sha256(args.input_state),
            "generator": str(Path(__file__).resolve().relative_to(Path.cwd())),
            "generator_sha256": _sha256(Path(__file__)),
        },
        "checks": {
            "force_rms_N_per_m3": float(report.absolute_l2),
            "epsilon_B": float(report.absolute_l2 / 5915447.712414409),
            "normalized_l2": float(report.normalized_l2),
            "near_axis_force_rms_N_per_m3": float(report.near_axis_l2),
            "bulk_force_rms_N_per_m3": float(report.bulk_l2),
            "edge_force_rms_N_per_m3": float(report.edge_l2),
            "radial_refinement_difference": float(
                report.radial_refinement_difference
            ),
            "angular_spectral_tail": float(report.angular_spectral_tail),
            "minimum_signed_jacobian": float(report.minimum_signed_jacobian),
            "nestedness_margin": float(report.nestedness_margin),
            "boundary_residual": float(report.boundary_residual),
            "lambda_gauge_residual": float(report.gauge_residual),
            "target_epsilon_B": 1.0e-5,
            "passes_force_target": float(report.absolute_l2 / 5915447.712414409)
            <= 1.0e-5,
        },
        "work": {"total_process_seconds": time.perf_counter() - started},
        "limitations": [
            "This certificate qualifies the axisymmetric fixed-profile native state only.",
            "It does not qualify 3-D, prescribed-current, LASYM, implicit derivatives, or the public polish driver.",
        ],
    }
    _write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
