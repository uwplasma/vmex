"""Compare saved raw (pre-line-search) steps of two driver runs (R7 F2).

Both runs must start from the same input checkpoint.  Reports cosine, norm
ratio, relative difference of the raw directions, and the accepted fractions,
so accepted-increment differences are not misread as solver differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.first) as a, np.load(args.second) as b:
        if str(a["input_state_sha256"]) != str(b["input_state_sha256"]):
            raise ValueError("runs start from different checkpoints")
        p, q = a["raw_steps"][0], b["raw_steps"][0]
        fp, fq = float(a["accepted_fraction"][0]), float(b["accepted_fraction"][0])
        record = {
            "schema": "vmex-r7-raw-step-comparison/1",
            "input_state_sha256": str(a["input_state_sha256"]),
            "first": str(args.first),
            "second": str(args.second),
            "raw_norms": [float(np.linalg.norm(p)), float(np.linalg.norm(q))],
            "raw_norm_ratio_second_over_first": float(np.linalg.norm(q) / np.linalg.norm(p)),
            "raw_cosine": float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q))),
            "raw_relative_difference": float(np.linalg.norm(p - q) / max(np.linalg.norm(p), np.linalg.norm(q))),
            "accepted_fractions": [fp, fq],
            "accepted_increment_norm_ratio": float(np.linalg.norm(fq * q) / np.linalg.norm(fp * p)),
        }
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
