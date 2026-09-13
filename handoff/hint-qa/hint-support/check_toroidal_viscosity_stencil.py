"""Check HINT's actual v_phi second-derivative stencil against a Fourier mode.

Extracts offsets from the maintained Fortran expression, and also evaluates
the upstream erroneous expression. Pure Python; no equilibrium or MPI job.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/HINT/stepb_mod.f90"


def offsets(text):
    block = text.split("  SUBROUTINE cal_viscos\n", 1)[1].split("  END SUBROUTINE cal_viscos", 1)[0]
    fourth = block.split("#else", 1)[1]
    expression = re.search(r"d2vphi1dphi2\s*=(.*?)(?=\n\s*d2vphi1dz2)", fourth, re.S).group(1)
    terms = re.findall(r"ct2\((\d)\)\s*\*\s*vphi1\(i\s*,\s*j\s*,\s*k\s*([+-]\d)?\s*\)", expression)
    assert [int(index) for index, _ in terms] == [1, 2, 3, 4, 5]
    return [int(offset or 0) for _, offset in terms]


def errors(shift):
    results = []
    for n in (16, 32, 64, 128):
        h = math.pi / n  # nfp=2, sin(2 phi) spans one field period.
        differences = []
        for k in range(n):
            phi = k * h
            actual = sum(c * math.sin(2 * (phi + offset * h))
                         for c, offset in zip((-1, 16, -30, 16, -1), shift)) / (12 * h * h)
            differences.append(actual + 4 * math.sin(2 * phi))
        results.append({"ntor": n, "max_abs_error": max(map(abs, differences)),
                        "rms_error": math.sqrt(sum(x*x for x in differences) / n)})
    return results


def main():
    current = SOURCE.read_text()
    upstream = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "bf31fc39f7179bdd91d84319c51c68e2f6fff25f:src/HINT/stepb_mod.f90"], text=True)
    before, after = errors(offsets(upstream)), errors(offsets(current))
    orders = [math.log2(a["rms_error"] / b["rms_error"]) for a, b in zip(after, after[1:])]
    assert all(3.9 < order < 4.1 for order in orders)
    assert before[-1]["rms_error"] > before[0]["rms_error"]
    assert after[-1]["max_abs_error"] < 3e-7
    configs = {}
    for name in ("build-debug", "build-release"):
        config = json.loads((ROOT/name/"build_config.json").read_text())
        configs[name] = {"flags": config["flags"], "fourth_order_branch_active": "-DVISCOS2ND" not in config["flags"]}
    report = dict(source_sha256=hashlib.sha256(current.encode()).hexdigest(),
                  upstream_offsets=offsets(upstream), patched_offsets=offsets(current),
                  upstream=before, patched=after, patched_rms_orders=orders,
                  build_configurations=configs,
                  scope="Fourier stencil verification; equilibrium impact requires nu1 != 0 and separate validation.")
    (ROOT/"local-support/toroidal-viscosity-stencil.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
