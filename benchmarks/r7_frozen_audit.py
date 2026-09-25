"""Read-only R7 frozen-state audit for a local VMEX PR #448 checkout.

This diagnostic does not solve an equilibrium, alter a checkpoint, or certify a
production derivative. Run under the repository's run_checked.py watchdog.
It was syntax-checked, but not run against VMEX in the review environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.json', delete=False) as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        temporary = Path(stream.name)
    os.replace(temporary, path)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    repo = args.repo.resolve()
    state_path = args.state if args.state.is_absolute() else repo / args.state
    state_path = state_path.resolve()
    output = args.output.resolve()
    started = time.perf_counter()
    report = {
        'schema': 'vmex-r7-frozen-state-audit/1', 'complete': False,
        'production_qualified': False, 'phase': 'initialization',
        'repo': str(repo), 'state': str(state_path),
    }
    atomic_json(output, report)
    try:
        import numpy as np
        import scipy
        from scipy import sparse
        from scipy.sparse.linalg import splu
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp

        report['state_sha256'] = sha_file(state_path)
        if args.expected_sha256 and report['state_sha256'] != args.expected_sha256:
            raise ValueError('checkpoint hash mismatch; refusing a substituted state')
        report['repo_head'] = subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True
        ).strip()
        report['environment'] = {'jax': jax.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / 'benchmarks'))
        # Import the actual reviewed research functions; do not reproduce their physics.
        module = importlib.import_module('polish_recovery_sparse')
        report['generator_sha256'] = sha_file(Path(module.__file__))
        report['phase'] = 'load_validated_original_problem'
        atomic_json(output, report)
        base, accepted, plan, layout, gauge, scale, coordinates, order = module._load_original_problem(state_path)
        saved_scale = np.asarray(scale)
        recomputed = np.asarray(module.native_coordinate_scales(base, layout, plan))
        scale_delta = recomputed - saved_scale
        report['replay'] = {
            'saved_scale_equal_recomputed': bool(np.array_equal(saved_scale, recomputed)),
            'scale_unequal_entries': int(np.count_nonzero(saved_scale != recomputed)),
            'scale_max_absolute_difference': float(np.max(np.abs(scale_delta))),
            'scale_max_relative_difference': float(np.max(np.abs(scale_delta) / np.maximum(np.abs(saved_scale), np.finfo(float).tiny))),
            'loaded_scale_is_used_below': True,
        }
        def array_hash(value):
            a = np.ascontiguousarray(np.asarray(value))
            digest = hashlib.sha256()
            digest.update(str(a.shape).encode())
            digest.update(a.dtype.str.encode())
            digest.update(a.tobytes())
            return digest.hexdigest()
        report['snapshot'] = {
            'coordinates_sha256': array_hash(coordinates), 'scale_sha256': array_hash(saved_scale),
            'rho_sha256': array_hash(plan.rho), 'quadrature_sha256': array_hash(plan.quadrature_weights),
            'layout_sha256': array_hash(layout.active_indices), 'radial_order': int(order),
        }
        restored = module.apply_high_order_correction(base, layout.unpack(scale * coordinates))
        changed = module.apply_high_order_correction(base, layout.unpack(jnp.asarray(recomputed) * coordinates))
        report['replay']['field_errors'] = {}
        for name in ('R_cos','R_sin','Z_cos','Z_sin','L_cos','L_sin'):
            truth = np.asarray(getattr(accepted, name))
            retained = np.asarray(getattr(restored, name))
            other = np.asarray(getattr(changed, name))
            report['replay']['field_errors'][name] = {
                'saved_scale_max_difference_from_checkpoint': float(np.max(np.abs(retained-truth))),
                'recomputed_scale_max_difference_from_checkpoint': float(np.max(np.abs(other-truth))),
                'saved_scale_bitwise_equal': bool(np.array_equal(retained,truth)),
            }
        report['phase'] = 'construct_one_fixed_constraint_projector'
        atomic_json(output, report)
        C = module.native_tangential_gauge_matrix(base, layout, gauge, scale).tocsr()
        n, q = C.shape[1], C.shape[0]
        Kp = sparse.bmat([[sparse.eye(n,format='csc'),C.T],[C,None]],format='csc')
        factor = splu(Kp)
        def project(v):
            rhs = np.r_[np.asarray(v),np.zeros(q)]
            sol = factor.solve(rhs)
            rel = float(np.linalg.norm(Kp@sol-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny))
            if not np.isfinite(rel) or rel > 1e-10:
                raise ArithmeticError(f'projection true residual {rel} fails')
            return sol[:n], rel
        report['constraint'] = {'shape':list(C.shape),'nnz':int(C.nnz),'factorization_count':1,
            'data_sha256':array_hash(C.data),'indices_sha256':array_hash(C.indices),'indptr_sha256':array_hash(C.indptr)}
        force, constraint = module._linear_problem(base,plan,layout,gauge,scale,coordinates)
        report['phase'] = 'compare_identical_state_gradient_paths'
        atomic_json(output, report)
        def objective(c):
            r = force(c)
            return 0.5*jnp.vdot(r,r).real
        r, pb = jax.vjp(force, coordinates)
        gv = np.asarray(pb(r)[0])
        gg = np.asarray(jax.grad(objective)(coordinates))
        gj = np.asarray(jax.jit(jax.grad(objective))(coordinates))
        gradients = {'vjp_force':gv, 'grad_objective':gg, 'jit_grad_objective':gj}
        projected = {}
        for key,g in gradients.items():
            projected[key], rr = project(g)
            report.setdefault('gradient_paths',{})[key] = {
                'full_norm':float(np.linalg.norm(g)), 'projected_norm':float(np.linalg.norm(projected[key])),
                'projection_true_relative_residual':rr,'array_sha256':array_hash(g)}
        pairs = {}
        keys = list(gradients)
        for i, a in enumerate(keys):
            for b in keys[i+1:]:
                difference = gradients[a]-gradients[b]
                pd,_ = project(difference)
                pairs[f'{a}__{b}'] = {
                    'absolute_full_gradient_difference':float(np.linalg.norm(difference)),
                    'absolute_projected_gradient_difference':float(np.linalg.norm(pd)),
                    'relative_to_first_projected_gradient':float(np.linalg.norm(pd)/max(np.linalg.norm(projected[a]),np.finfo(float).tiny)),
                }
        report['gradient_path_comparisons'] = pairs
        report['residual_norm'] = float(np.linalg.norm(np.asarray(r)))
        report['original_gauge_norm'] = float(np.linalg.norm(np.asarray(constraint(coordinates))))
        report['snapshot']['residual_sha256'] = array_hash(r)
        report['snapshot']['full_gradient_sha256'] = array_hash(gv)
        report['complete'] = True
        report['phase'] = 'complete'
        report['limitations'] = [
            'No physical-force recertification, nonlinear solve, or derivative qualification is performed.',
            'Matching gradient paths does not prove negligible rounding error shared by those paths.',
            'A difference is diagnostic evidence, not authorization to choose whichever path passes a gate.',
        ]
    except BaseException as error:
        report['phase'] = 'failed'
        report['error_type'] = type(error).__name__
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        atomic_json(output, report)


if __name__ == '__main__':
    main()
