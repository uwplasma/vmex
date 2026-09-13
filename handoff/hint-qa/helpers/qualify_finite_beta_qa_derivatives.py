"""Bounded live-solve QA derivative diagnostic; no optimization qualification.

Run under run_bounded.py after device coordination. This deliberately avoids
cached Equilibrium snapshot fields: their parameter VJPs are unsupported.
The state tangent uses a single matrix-free solve, not full block assembly.
"""
import argparse
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import time

from qualify_finite_beta_qa_root import digest, source_manifest, write


def worker(a):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vmex
    import solvax
    from vmex.core import implicit as imp
    from vmex.core.input import VmecInput

    jax.config.update('jax_enable_x64', True)
    assert Path(vmex.__file__).resolve().parent.parent == a.vmex_root.resolve()
    if a.solvax_root:
        assert Path(solvax.__file__).resolve() == (a.solvax_root/'src/solvax/__init__.py').resolve()
    inp = VmecInput.from_file(a.input)
    assert inp.nfp == 2 and inp.ncurr == 1 and not inp.lfreeb
    assert inp.mpol == 7 and inp.ntor == 6
    assert float(inp.am[0]) > 1e5 and float(inp.curtor) < -9e4
    device = jax.devices()[0]
    # Match run()'s active-device convention: explicit CPU, ambient default GPU.
    cfg = imp.make_config(inp, hot_restart=False,
                          device=device if device.platform == 'cpu' else None)
    assert int(cfg.resolution.ns) == 31
    p0 = imp.params_from_input(inp, device=device)
    scale = abs(float(p0.rbc[inp.ntor, 0]))
    assert scale > 0
    direction = jax.tree.map(jnp.zeros_like, p0)
    direction = dataclasses.replace(direction,
        rbc=direction.rbc.at[inp.ntor, 1].set(scale))
    report = dict(device=str(device), jax_version=jax.__version__,
        direction=dict(family='rbc', m=1, n=0, meters_per_unit_parameter=scale),
        fixed_constraints=dict(ncurr=int(inp.ncurr), pressure_axis_Pa=float(inp.am[0]),
            curtor_A=float(inp.curtor), phiedge_Wb=float(inp.phiedge)),
        objective='mhd_energy(state,runtime)[0] (VMEX wb normalization)',
        primal_tol=float(cfg.primal_tol), adjoint_tol=float(cfg.adjoint_tol),
        phases=[], taylor=[], completed=False,
        scope='One fixed-boundary QA energy direction; no exterior-field, free-boundary, optimization or grid-convergence certificate.')

    def save():
        write(a.output/'report.json', report)

    def phase(name):
        report['active_phase'] = name
        report['phases'].append(dict(name=name, start_unix=time.time()))
        save()
        print(name, flush=True)

    def finish():
        report['phases'][-1]['elapsed_seconds'] = time.time()-report['phases'][-1]['start_unix']
        save()

    def dot(x, y):
        return sum(jnp.vdot(u, v).real for u, v in zip(jax.tree.leaves(x), jax.tree.leaves(y), strict=True))

    def norm(x):
        return float(jnp.sqrt(dot(x, x)))

    def certified(params):
        state, mask, status, fsq, ratio = imp._host_solve_and_mask_status(cfg, params)
        certificate = imp._LAST_PRIMAL_CERTIFICATE.get(cfg)
        evidence = dict(certificate[2]) if certificate is not None else {}
        matches = (certificate is not None and certificate[0] == imp._params_key(params)
                   and certificate[1] == imp._primal_state_key(state))
        evidence.update(status=int(status), host_fsq=float(fsq), host_fsq_ratio=float(ratio),
                        certificate_matches_returned_state=bool(matches))
        report['last_primal'] = evidence
        save()
        if int(status) != 0 or not matches or not evidence.get('derivative_certified'):
            raise RuntimeError('Live solve lacks an actual-state primal certificate')
        return jax.tree.map(jnp.asarray, state), jax.tree.map(jnp.asarray, mask), evidence

    def metric(state, params):
        return imp.mhd_energy(state, imp.runtime_from_params(params, cfg))[0]

    def objective(params):
        return metric(imp.solve_implicit(params, cfg), params)

    try:
        phase('base_primal')
        state, mask, evidence = certified(p0)
        report['base_primal'] = evidence
        finish()

        phase('live_reverse_energy_gradient')
        value, gradient = jax.value_and_grad(objective)(p0)
        value, slope = float(value), float(dot(gradient, direction))
        if not np.isfinite(value) or not np.isfinite(slope):
            raise FloatingPointError('Nonfinite ordinary value or reverse derivative')
        report.update(value=value, reverse_directional_derivative=slope)
        finish()

        phase('matrix_free_state_tangent_and_duality')
        frozen = jax.lax.stop_gradient(state)
        project = imp._dof_projector(cfg, mask)
        edge = imp._edge_mask(cfg)
        residual = imp.residual_fn(cfg, frozen, mask)
        z = project(state)
        _, action = jax.linearize(lambda zz: residual(zz, p0), z)
        rhs = jax.tree.map(jnp.negative, jax.jvp(lambda pp: residual(z, pp),
            (p0,), (direction,))[1])
        dz, linear = imp._adjoint_solve_gcrot(action, rhs, cfg, enforce=True)
        linear_error = norm(jax.tree.map(jnp.subtract, action(dz), rhs))
        linear_tol = 10 * cfg.adjoint_tol * norm(rhs)
        if not np.isfinite(linear_error) or linear_error > linear_tol:
            raise FloatingPointError('Tangent fails independently recomputed residual bound')
        dx = jax.jvp(lambda zz, pp: imp._assemble(zz,
            imp.runtime_from_params(pp, cfg), frozen, project, edge),
            (z, p0), (project(dz), direction))[1]
        gs, explicit = jax.grad(metric, argnums=(0, 1))(state, p0)
        lhs = float(dot(gs, dx))
        explicit_slope = float(dot(explicit, direction))
        rhs_dot = slope - explicit_slope
        tangent_slope = lhs + explicit_slope
        report['duality'] = dict(state_cotangent_dot_state_tangent=lhs,
            parameter_direction_dot_state_pullback=rhs_dot,
            explicit_parameter_derivative=explicit_slope,
            full_tangent_directional_derivative=tangent_slope,
            absolute_error=abs(lhs-rhs_dot),
            relative_error=abs(lhs-rhs_dot)/max(abs(lhs),abs(rhs_dot),1e-300),
            tangent_residual_norm=linear_error, tangent_residual_bound=linear_tol,
            note='Energy state cotangent versus boundary direction; explicit metric parameter dependence is subtracted from the ordinary live-solve VJP.')
        finish()

        for i in range(4):
            h = a.step / 2**i
            row = dict(h=h, boundary_displacement_m=h*scale)
            report['taylor'].append(row)
            for sign, label in ((1, 'plus'), (-1, 'minus')):
                phase(f'taylor_{i}_{label}_primal_and_value')
                params = jax.tree.map(lambda p, d: p+sign*h*d, p0, direction)
                _, _, trial_evidence = certified(params)
                trial_value = float(objective(params))
                if not np.isfinite(trial_value):
                    raise FloatingPointError('Nonfinite perturbed objective')
                row[label] = dict(value=trial_value, primal=trial_evidence,
                    first_order_remainder=abs(trial_value-value-sign*h*slope))
                finish()
            fd = (row['plus']['value']-row['minus']['value'])/(2*h)
            row.update(central_difference=fd, derivative_absolute_error=abs(fd-slope),
                derivative_relative_error=abs(fd-slope)/max(abs(fd),abs(slope),1e-300),
                max_first_order_remainder=max(row['plus']['first_order_remainder'],
                                              row['minus']['first_order_remainder']))
            if i:
                previous = report['taylor'][i-1]['max_first_order_remainder']
                current = row['max_first_order_remainder']
                row['observed_remainder_order'] = float(np.log2(previous/current)) if previous > 0 and current > 0 else None
            save()
        report['completed'] = True
        report['active_phase'] = None
        report['interpretation'] = 'Inspect error magnitudes and an O(h^2) remainder region; completion alone is not derivative qualification. Small steps can reach solver/roundoff noise.'
        save()
    except Exception as exc:
        report['exception'] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vmex-root', type=Path, required=True)
    p.add_argument('--solvax-root', type=Path)
    p.add_argument('--source-commit')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--step', type=float, default=1e-4)
    p.add_argument('--per-run-timeout', type=int, default=600)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    if not 0 < a.step <= 1e-2 or not 1 <= a.per_run_timeout <= 600:
        p.error('step must be positive and <=1e-2; timeout must be1..600 seconds')
    if a.worker:
        worker(a)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(input_sha256=digest(a.input), harness_sha256=digest(Path(__file__)),
        helper_sha256=digest(Path(__file__).with_name('qualify_finite_beta_qa_root.py')),
        vmex_source=source_manifest(a.vmex_root, 'vmex', a.source_commit),
        step=a.step, per_run_timeout=a.per_run_timeout)
    if a.solvax_root:
        report['solvax_source'] = source_manifest(a.solvax_root, 'src/solvax')
    write(a.output/'provenance.json', report)
    paths=[str(a.vmex_root.resolve())]
    if a.solvax_root:
        paths.append(str(a.solvax_root.resolve()/'src'))
    env=dict(os.environ, PYTHONPATH=os.pathsep.join(paths+[os.environ.get('PYTHONPATH','')]),
        JAX_ENABLE_X64='1', VMEX_COMPILATION_CACHE='disabled',
        JAX_COMPILATION_CACHE_DIR=str(a.output.resolve()/'jax-cache'),
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',
         '--vmex-root',str(a.vmex_root.resolve()),'--input',str(a.input.resolve()),
         '--output',str(a.output.resolve()),'--step',str(a.step)]
    if a.solvax_root:
        cmd += ['--solvax-root',str(a.solvax_root.resolve())]
    with (a.output/'output.log').open('w') as log:
        try:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.per_run_timeout)
            report['returncode']=result.returncode
        except subprocess.TimeoutExpired:
            report['timeout']=True
    write(a.output/'provenance.json',report)
    print(a.output/'report.json')
    if report.get('returncode') != 0:
        raise SystemExit(124 if report.get('timeout') else report['returncode'])


if __name__ == '__main__':
    main()
