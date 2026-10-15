"""DeltaRes reduction contracts, not an equilibrium or orbit implementation.

Labbate, Paul and Chambliss, arXiv:2609.01890v1, equations (2.8),
(3.5)-(3.10). Inputs must already use matched, complete magnetic wells.
The shear-cancelled spectrum is evaluated without a singular division.
Timings exclude field geometry, bounce integration and equilibrium adjoints.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


from vmex.core.deltares import stabilized_power


def benchmark(repeats=9):
    """Synchronized synthetic-reduction timings; input map remains dynamic."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError("Run the reference in x64")
    rng = np.random.default_rng(421)
    inputs = jnp.asarray(rng.normal(size=(8, 9, 1, 40)))
    def scalar(data):
        return jnp.sum(stabilized_power(data, 8))
    timings = {}
    for name, function in (("value", scalar), ("value_and_grad", jax.value_and_grad(scalar))):
        compiled = jax.jit(function)
        start = time.perf_counter()
        jax.block_until_ready(compiled(inputs))
        first = time.perf_counter() - start
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(compiled(inputs))
            samples.append(time.perf_counter() - start)
        timings[name] = {"first_call_s": first, "warm_median_s": float(np.median(samples))}
    return {"scope": "synthetic reduction only; no equilibrium, bounce or orbit solve",
            "python": platform.python_version(), "jax": jax.__version__,
            "devices": [str(d) for d in jax.devices()], "x64": True,
            "xla_flags": os.environ.get("XLA_FLAGS", ""), "numpy": np.__version__,
            "map_shape": list(inputs.shape), "q_max": 8, "repeats": repeats, "timings": timings}


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    result = benchmark()
    Path("deltares_reference_timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
