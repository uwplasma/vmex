"""Bounded coil-adapter parity/timing check; no plasma equilibrium solve.

Select ESSOS with PYTHONPATH before launching this script. Run once per version
with the same --coils input, then compare the JSON timings and NPZ arrays.
"""

import argparse
import hashlib
import inspect
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from essos.coils import Coils, Curves

from vmex.core.coil_parameters import CoilParameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coils", type=Path, required=True, help="Maintained case's physical-array JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 200:
        parser.error("--repeats must be between 5 and 200")
    jax.config.update("jax_enable_x64", True)
    data = json.loads(args.coils.read_text())
    # The maintained case authenticates these input arrays as metres/amperes.
    coils = Coils(
        Curves(jnp.asarray(data["dofs_curves"]), 75, data["nfp"], data["stellsym"]),
        jnp.asarray(data["dofs_currents"]),
    )
    chart = CoilParameters.from_coils(coils, current_dofs=tuple(range(1, len(data["dofs_currents"]))))
    scales = jnp.asarray(chart.scales)
    x = jnp.asarray(np.random.default_rng(747).normal(size=chart.size))
    r, phi, z = jnp.linspace(0.85, 1.15, 16), jnp.linspace(0.0, 1.5, 16), jnp.linspace(-0.1, 0.1, 16)

    def field(u):
        return jnp.stack(chart(u * scales).b_cyl(r, phi, z))

    functions = {"field": field, "jacfwd": jax.jacfwd(field), "jacrev": jax.jacrev(field)}
    report = {
        "essos_source": inspect.getsourcefile(Coils),
        "essos_coils_sha256": hashlib.sha256(Path(inspect.getsourcefile(Coils)).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(inspect.getsourcefile(CoilParameters)).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.coils.read_bytes()).hexdigest(),
        "jax": jax.__version__,
        "python": platform.python_version(),
        "device": str(jax.devices()[0]),
        "platform": platform.platform(),
        "parameters": chart.size,
        "points": 16,
        "segments": 75,
        "seed": 747,
        "repeats": args.repeats,
        "equilibrium_solved": False,
    }
    arrays = {}
    for name, function in functions.items():
        start = time.perf_counter()
        lowered = jax.jit(function).lower(x)
        compiled = lowered.compile()
        report[name] = {"trace_compile_s": time.perf_counter() - start}
        report[name]["hlo_sha256"] = hashlib.sha256(lowered.compiler_ir("hlo").as_hlo_text().encode()).hexdigest()
        for _ in range(3):
            result = compiled(x).block_until_ready()
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = compiled(x).block_until_ready()
            samples.append(1000 * (time.perf_counter() - start))
        arrays[name] = np.asarray(result)
        assert np.isfinite(arrays[name]).all()
        report[name].update(median_ms=float(np.median(samples)), samples_ms=samples)
    np.testing.assert_allclose(arrays["jacfwd"], arrays["jacrev"], rtol=2e-11, atol=1e-12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **arrays)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "essos": report["essos_source"],
                **{name: report[name]["median_ms"] for name in functions},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
