#!/usr/bin/env python3
"""Sample total and vacuum fields with native MAGVAL and compare two grids."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


COMPONENTS = ("B_R", "B_phi", "B_Z")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_f64(array):
    return np.ascontiguousarray(np.asarray(array, dtype="<f8"))


def array_sha256(array):
    return hashlib.sha256(canonical_f64(array).view(np.uint8)).hexdigest()


def vector_delta_metrics(coarse, fine):
    coarse = canonical_f64(coarse)
    fine = canonical_f64(fine)
    if coarse.shape != fine.shape or coarse.ndim != 2 or coarse.shape[1] != 3:
        raise ValueError("sample arrays must have the same (point, 3) shape")
    magnitudes = np.linalg.norm(fine - coarse, axis=1)
    return {
        "rms_vector_delta_T": float(np.sqrt(np.mean(magnitudes**2))),
        "max_vector_delta_T": float(np.max(magnitudes)),
    }


def write_sample_archive(path, xyz, samples):
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            xyz=xyz,
            coarse_total=samples["coarse"]["total"],
            coarse_vacuum=samples["coarse"]["vacuum"],
            fine_total=samples["fine"]["total"],
            fine_vacuum=samples["fine"]["vacuum"],
        )


def read_case(snapshot_path, vacuum_path):
    with np.load(snapshot_path) as snapshot:
        total = canonical_f64(snapshot["B_cyl"])
        pressure = canonical_f64(snapshot["P_mu0_Pa"])
        snapshot_coordinates = {
            key: canonical_f64(snapshot[key]) for key in ("R", "phi", "Z")
        }
    with Dataset(vacuum_path) as vacuum_file:
        vacuum = canonical_f64(
            np.stack([vacuum_file[key][:] for key in ("Bvac_R", "Bvac_phi", "Bvac_Z")], axis=-1)
        )
        vacuum_coordinates = {
            key: canonical_f64(vacuum_file[key][:]) for key in ("R", "phi", "Z")
        }
    if total.shape != vacuum.shape or pressure.shape != total.shape[:-1]:
        raise ValueError("snapshot and vacuum field shapes differ")
    for key in snapshot_coordinates:
        if not np.array_equal(snapshot_coordinates[key], vacuum_coordinates[key]):
            raise ValueError(f"snapshot and vacuum {key} coordinates differ")
    if not np.isfinite(total).all() or not np.isfinite(vacuum).all():
        raise ValueError("field arrays contain nonfinite values")
    if not np.isfinite(pressure).all():
        raise ValueError("pressure array contains nonfinite values")
    return total, vacuum, pressure


def write_native_input(template_path, output_path, field, pressure):
    with Dataset(template_path) as source, Dataset(output_path, "w") as output:
        for key, dimension in source.dimensions.items():
            output.createDimension(key, len(dimension))
        for key in ("mtor", "rminb", "rmaxb", "zminb", "zmaxb", "R", "phi", "Z"):
            value = source[key]
            output.createVariable(key, value.dtype, value.dimensions)[:] = value[:]
        for index, key in enumerate(COMPONENTS):
            output.createVariable(key, "f8", ("phi", "Z", "R"))[:] = field[..., index]
        output.createVariable("P", "f8", ("phi", "Z", "R"))[:] = pressure
        for key in ("v_R", "v_phi", "v_Z"):
            output.createVariable(key, "f8", ("phi", "Z", "R"))[:] = 0.0


def sample_native(sampler, template, field, pressure, point_input, count, timeout):
    with tempfile.TemporaryDirectory(prefix="native-field-sample-") as directory:
        work = Path(directory)
        native_input = work / "field.nc"
        write_native_input(template, native_input, field, pressure)
        result = subprocess.run(
            [str(sampler), native_input.name],
            cwd=work,
            input=point_input,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        if result.returncode:
            raise RuntimeError(f"native sampler failed with status {result.returncode}: {result.stdout[-2000:]}")
        values = canonical_f64(np.loadtxt(work / "native-points.csv", delimiter=",", ndmin=2))
    if values.shape != (count, 3) or not np.isfinite(values).all():
        raise ValueError("native sampler returned an invalid point array")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler", required=True, type=Path, help="compiled sample-points executable")
    parser.add_argument("--targets", required=True, type=Path, help="NPZ containing Cartesian xyz in metres")
    parser.add_argument("--coarse-snapshot", required=True, type=Path)
    parser.add_argument("--coarse-vacuum", required=True, type=Path)
    parser.add_argument("--fine-snapshot", required=True, type=Path)
    parser.add_argument("--fine-vacuum", required=True, type=Path)
    parser.add_argument("--hint-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--samples-output",
        type=Path,
        help="optional NPZ containing targets and native coarse/fine samples",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per native sample")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if re.fullmatch(r"[0-9a-fA-F]{40}", args.hint_commit) is None:
        raise ValueError("hint commit must be exactly 40 hexadecimal characters")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.samples_output is not None and args.samples_output.exists():
        raise FileExistsError(f"refusing to overwrite sample output: {args.samples_output}")
    sampler = args.sampler.resolve()
    with np.load(args.targets) as target_file:
        xyz = canonical_f64(target_file["xyz"])
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 1 or not np.isfinite(xyz).all():
        raise ValueError("target xyz must be a finite (point, 3) array")
    cylindrical = canonical_f64(
        np.column_stack(
            (np.linalg.norm(xyz[:, :2], axis=1), np.arctan2(xyz[:, 1], xyz[:, 0]), xyz[:, 2])
        )
    )
    point_input = str(len(cylindrical)) + "\n" + "\n".join(
        " ".join(map(str, row)) for row in cylindrical
    )

    paths = {
        "coarse": (args.coarse_snapshot, args.coarse_vacuum),
        "fine": (args.fine_snapshot, args.fine_vacuum),
    }
    samples = {}
    source_hashes = {}
    for label, (snapshot_path, vacuum_path) in paths.items():
        total, vacuum, pressure = read_case(snapshot_path, vacuum_path)
        sampled_total = sample_native(
            sampler, vacuum_path, total, pressure, point_input, len(xyz), args.timeout
        )
        sampled_vacuum = sample_native(
            sampler,
            vacuum_path,
            vacuum,
            np.zeros_like(pressure),
            point_input,
            len(xyz),
            args.timeout,
        )
        sampled_response = canonical_f64(sampled_total - sampled_vacuum)
        samples[label] = {
            "total": sampled_total,
            "vacuum": sampled_vacuum,
            "response": sampled_response,
        }
        source_hashes[label] = {
            "snapshot_container_sha256": file_sha256(snapshot_path),
            "vacuum_container_sha256": file_sha256(vacuum_path),
            "total_raw_array_sha256": array_sha256(total),
            "vacuum_raw_array_sha256": array_sha256(vacuum),
            "total_target_array_sha256": array_sha256(sampled_total),
            "vacuum_target_array_sha256": array_sha256(sampled_vacuum),
            "response_target_array_sha256": array_sha256(sampled_response),
        }

    report = {
        "schema_version": 1,
        "scope": {
            "target_count": len(xyz),
            "coordinate_units": {
                "cartesian_targets": "m",
                "native_cylindrical_targets": ["m", "rad", "m"],
            },
            "field_components": list(COMPONENTS),
            "field_unit": "T",
            "array_hash_encoding": "C-contiguous little-endian float64 bytes",
        },
        "source": {
            "hint_commit": args.hint_commit.lower(),
            "postprocessor_source_sha256": file_sha256(Path(__file__)),
        },
        "shared_hashes": {
            "native_sampler_binary_sha256": file_sha256(sampler),
            "target_container_sha256": file_sha256(args.targets),
            "target_cartesian_array_sha256": array_sha256(xyz),
            "target_native_cylindrical_array_sha256": array_sha256(cylindrical),
        },
        "coarse_hashes": source_hashes["coarse"],
        "fine_hashes": source_hashes["fine"],
        "coarse_to_fine_metrics": {
            label: vector_delta_metrics(samples["coarse"][label], samples["fine"][label])
            for label in ("total", "vacuum", "response")
        },
        "response_definition": "At each target and on each grid, subtract the native vacuum-field vector from the native total-field vector, then form the fine-minus-coarse vector difference.",
        "limitations": [
            "This is a field-sampling diagnostic, not an equilibrium or convergence certificate.",
            "The caller must independently establish target location relative to current support.",
        ],
    }
    if args.samples_output is not None:
        write_sample_archive(args.samples_output, xyz, samples)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["coarse_to_fine_metrics"], indent=2))


if __name__ == "__main__":
    main()
