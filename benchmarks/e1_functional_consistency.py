"""Bounded E1 probe: constrained energy versus independently evaluated force work.

Fixed boundary, prescribed phipf/chipf and pressure (GAMMA=0), stellarator
symmetry. W = integral (B²/(2 mu0) - p) dV. This is a functional audit,
not a solver or a force-accuracy benchmark. The lambda term is essential:
https://princetonuniversity.github.io/STELLOPT/VMEC.html#theory

Run from the checkout with float64 and compilation caching disabled. The
analytic fixture is shared with the existing Solov'ev oracle tests. Output
is diagnostic; the shaped-tokamak and production-chart audits remain separate.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

import vmex
from vmex.core.profiles import MU0
from vmex.core.radial_basis import BSplineBasis, _span_quadrature
from vmex.core.strong_force import _basic_fields, _point_force, _position, _RZL, lift_high_order_state
from tests.test_strong_force_solovev import _analytic_state, _Solovev
from benchmarks._provenance import assert_repo_vmex, git_state


def coordinates(native):
    """All symmetric spline coefficients, with fixed R/Z boundary and no L00."""
    fields = ("R_cos", "Z_sin", "L_sin")
    indices = []
    for name in fields:
        for mode in range(len(native.m)):
            if name != "R_cos" and native.m[mode] == 0 and native.n[mode] == 0:
                continue
            for radial in range(native.radial_basis.size - (name != "L_sin")):
                indices.append((name, mode, radial))

    def state(vector):
        values = {name: getattr(native, name) for name in fields}
        for i, (name, mode, radial) in enumerate(indices):
            values[name] = values[name].at[mode, radial].add(vector[i])
        return replace(native, **values)

    return state, indices


def probe(native, order, angles):
    if native.nfp != 1 or np.any(native.n != 0):
        raise ValueError("this bounded E1 probe integrates axisymmetric nfp=1 states only")
    state, indices = coordinates(native)
    # Deliberately move off the analytic root: cancellation at a root is not
    # useful evidence of a variational identity. Keep profiles/boundary fixed.
    offset = 0.002 * jnp.sin(jnp.arange(len(indices)) + 1)
    current = state(offset)
    rho, weights = _span_quadrature(np.sqrt(native.radial_basis.breakpoints), order)
    theta = (np.arange(angles) + 0.37) * 2 * np.pi / angles
    points = jnp.asarray([(r, t, 0.0) for r in rho for t in theta])
    weights = jnp.asarray(np.repeat(weights, angles) * (2 * np.pi)**2 / angles)

    def density(vector, point):
        _, jacobian, _, _, field, pressure = _basic_fields(state(vector), point)
        return (jnp.vdot(field, field) / (2 * MU0) - pressure) * jnp.abs(jacobian)

    energy = jax.jit(lambda vector: jnp.dot(
        weights, jax.vmap(density, in_axes=(None, 0))(vector, points)))
    gradient = jax.jit(jax.grad(energy))(offset)
    direction = jnp.cos(jnp.arange(len(indices))) / np.sqrt(len(indices))
    _, jvp = jax.jvp(energy, (offset,), (direction,))
    _, pullback = jax.vjp(energy, offset)
    vjp = jnp.vdot(pullback(jnp.asarray(1.0))[0], direction)
    finite_difference = [float(jnp.abs(
        (energy(offset + step * direction) - energy(offset - step * direction)) / (2 * step)
        - jvp) / jnp.maximum(jnp.abs(jvp), 1)) for step in (1e-4, 1e-5, 1e-6)]

    def work(point):
        jacobian, _, _, force, _, helical, *_ = _point_force(current, point)
        position_map = jax.jacfwd(lambda vector: _position(state(vector), point))(offset)
        lambda_map = jax.jacfwd(lambda vector: _RZL(state(vector), point)[2])(offset)
        phip = native.radial_basis.evaluate(native.phipf, point[0]**2)
        geometric = -jnp.abs(jacobian) * (force @ position_map)
        angular = -jnp.sign(jacobian) * 2 * point[0] * phip * helical * lambda_map
        basis, _, _, _, _, _ = _basic_fields(current, point)
        lambda_theta = jax.grad(lambda x: _RZL(current, x)[2])(point)[1]
        displacement = position_map - jnp.outer(basis[:, 1], lambda_map) / (1 + lambda_theta)
        return geometric, angular, displacement, jacobian, 1 + lambda_theta

    geometric, angular, displacement, jacobian, label_derivative = jax.jit(jax.vmap(work))(points)
    virtual_work = weights @ (geometric + angular)
    scale = jnp.maximum(jnp.linalg.norm(gradient), 1.0)
    # Rank of the sampled fixed-label displacement map, not a claim about
    # the production chart's gauge quotient or the equilibrium Hessian.
    matrix = np.asarray(displacement * jnp.sqrt(weights * jnp.abs(jacobian))[:, None, None]).reshape(-1, len(indices))
    singular = np.linalg.svd(matrix, compute_uv=False)
    defect = np.asarray(gradient - virtual_work)
    channel_errors = {}
    for name in ("R_cos", "Z_sin", "L_sin"):
        mask = np.array([entry[0] == name for entry in indices])
        channel_errors[name] = float(np.linalg.norm(defect[mask]) / max(np.linalg.norm(np.asarray(gradient)[mask]), 1.0))
    return {
        "order": order, "angles": angles, "coordinates": len(indices),
        "points": len(points), "energy_J": float(energy(offset)),
        "ad_duality_relative": float(jnp.abs(jvp - vjp) / jnp.maximum(jnp.abs(jvp), 1)),
        "central_difference_relative_1e-4_1e-5_1e-6": finite_difference,
        "work_relative": float(jnp.linalg.norm(gradient - virtual_work) / scale),
        "channel_work_relative": channel_errors,
        "omitted_lambda_relative": float(jnp.linalg.norm(gradient - weights @ geometric) / scale),
        "minimum_oriented_jacobian": float(jnp.min(native.jacobian_sign * jacobian)),
        "minimum_label_derivative": float(jnp.min(label_derivative)),
        "sampled_displacement_rank": int(np.sum(singular > 1e-10 * singular[0])),
        "reference_matrix_bytes": matrix.nbytes,
        "sampled_displacement_singular_values": singular.tolist(),
        "gradient": np.asarray(gradient).tolist(),
        "virtual_work": np.asarray(virtual_work).tolist(),
    }


def shaped_state():
    """Seed from the current-driven deck, then freeze both flux derivatives.

    This audits the prescribed-iota functional at that geometry; varying the
    fixed-current closure would require different terms and is NOT tested.
    """
    from vmex.core import implicit
    from vmex.core.input import VmecInput

    inp = VmecInput.from_file(ROOT / "examples/data/input.shaped_tokamak_pressure_polished")
    inp = replace(inp.change_resolution(mpol=5, ntor=0, ntheta=16, nzeta=4),
                  ns_array=np.asarray([9]), ftol_array=np.asarray([1e-10]),
                  niter_array=np.asarray([1000]))
    config = implicit.make_config(inp, ftol=1e-10, max_iterations=1000)
    params = implicit.params_from_input(inp)
    state, _ = implicit.solve_implicit_with_aux(params, config)
    runtime = implicit.runtime_from_params(params, config)
    basis = BSplineBasis.clamped(np.linspace(0, 1, 3), degree=3)
    return lift_high_order_state(state, runtime, radial_basis=basis)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--angles", type=int, default=32)
    parser.add_argument("--case", choices=("solovev", "shaped-frozen-iota"), default="solovev")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error("set JAX_ENABLE_X64=1")
    if any(order < 2 or order > 24 for order in args.orders) or not 8 <= args.angles <= 128:
        parser.error("bounded probe requires orders 2..24 and angles 8..128")
    def expired(*_):
        raise TimeoutError("E1 600 s budget")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    started = time.perf_counter()
    provenance = git_state(ROOT)
    provenance.update(jax=jax.__version__, python=platform.python_version(),
                      numpy=np.__version__, platform=platform.system(),
                      device=str(jax.devices()[0]), precision="float64",
                      vmex_module=assert_repo_vmex(vmex.__file__, ROOT))
    native = (_analytic_state(_Solovev(mmax=2), degree=3, spans=2)
              if args.case == "solovev" else shaped_state())
    records = []
    for order in args.orders:
        record = probe(native, order, args.angles)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k not in
                          ("gradient", "virtual_work", "sampled_displacement_singular_values")}), flush=True)
        jax.clear_caches()
    output = {"_provenance": provenance, "case": args.case,
              "scope": "fixed-boundary prescribed profiles, GAMMA=0; not the production gauge quotient",
              "records": records, "seconds": time.perf_counter() - started,
              "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
              (1024**2 if platform.system() == "Darwin" else 1024)}
    output["identity_passed"] = all(
        record["ad_duality_relative"] < 1e-10
        and record["minimum_oriented_jacobian"] > 0
        and record["minimum_label_derivative"] > 0
        for record in records
    ) and records[-1]["work_relative"] < 1e-8
    output["E1_complete"] = False  # production chart/gauge quotient still requires its audit
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if not output["identity_passed"]:
        raise SystemExit("E1 identity unresolved: refine quadrature or diagnose closure/work terms")


if __name__ == "__main__":
    main()
