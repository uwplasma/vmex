"""Bounded, fresh-process qualification of the actual finite-beta QA input.

Run under run_bounded.py after coordinating a device slot. Each of one/two fresh
workers has a timeout and a private compilation cache. Native ns31/MPOL7/NTOR6
is retained; pressure, current, boundary and flux are never changed. This uses
private VMEX diagnostics and must be checked against the recorded source head.
No derivatives, optimized-design claim or free-boundary qualification.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(root, package, declared_commit=None):
    root = root.resolve()
    files = sorted((root / package).rglob('*.py'))
    if not files:
        raise ValueError(f'No Python package sources under {root / package}')
    # An archive can live inside another checkout: never discover that parent's HEAD.
    git = root / '.git'
    commit, diff_hash = declared_commit, None
    if git.exists():
        top = Path(subprocess.check_output(['git','-C',str(root),'rev-parse','--show-toplevel'],text=True).strip()).resolve()
        if top != root:
            raise ValueError('Git root differs from declared source root')
        actual = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        if declared_commit is not None and declared_commit != actual:
            raise ValueError('Declared source commit does not match checkout')
        commit = actual
        diff_hash = hashlib.sha256(subprocess.check_output(['git','-C',str(root),'diff','HEAD'])).hexdigest()
    elif package == 'vmex' and declared_commit is None:
        raise ValueError('VMEX archive requires explicit --source-commit')
    return dict(commit=commit, commit_is_declared_archive=not git.exists(),
                diff_sha256=diff_hash,
                source_sha256={str(p.relative_to(root)):digest(p) for p in files})


def write(path, value):
    def finite(v):
        if isinstance(v, float) and not math.isfinite(v): return None
        if isinstance(v, dict): return {k:finite(x) for k,x in v.items()}
        if isinstance(v, (list, tuple)): return [finite(x) for x in v]
        return v
    path.write_text(json.dumps(finite(value), indent=2, allow_nan=False) + '\n')


def worker(a):
    import jax
    import numpy as np
    import vmex
    import solvax
    from vmex.core import implicit as imp
    from vmex.core.input import VmecInput
    from vmex.core.fields import surface_currents
    from vmex.core.profiles import pressure
    from vmex.core.statephysics import _field_chain
    from scipy.io import netcdf_file

    jax.config.update('jax_enable_x64', True)
    assert Path(vmex.__file__).resolve().parent.parent == a.vmex_root.resolve()
    if a.solvax_root is not None:
        assert Path(solvax.__file__).resolve() == (a.solvax_root/'src/solvax/__init__.py').resolve()
    inp = VmecInput.from_file(a.input)
    assert int(inp.nfp) == 2 and int(inp.ncurr) == 1 and not bool(inp.lfreeb)
    assert float(inp.am[0]) > 1e5 and float(inp.curtor) < -9e4
    dev = jax.devices()[0]
    cfg = imp.make_config(inp, hot_restart=False, device=dev)
    params = imp.params_from_input(inp)
    start = time.perf_counter()
    state, mask, status, host_fsq, host_ratio = imp._host_solve_and_mask_status(cfg, params)
    elapsed = time.perf_counter() - start
    certificate = imp._LAST_PRIMAL_CERTIFICATE.get(cfg)
    evidence = dict(certificate[2]) if certificate is not None else {}
    host = imp._LAST_SOLVE.get(cfg)
    report = dict(device=str(dev), jax_version=jax.__version__, status=int(status),
                  solvax_import_path=str(Path(solvax.__file__).resolve()),
                  imported_solvax_source_sha256={str(p.relative_to(Path(solvax.__file__).resolve().parent)):digest(p)
                      for p in sorted(Path(solvax.__file__).resolve().parent.rglob('*.py'))},
                  elapsed_solve_and_certificate_seconds=elapsed,
                  host_fsq=float(host_fsq) if np.isfinite(host_fsq) else None,
                  host_fsq_ratio=float(host_ratio) if np.isfinite(host_ratio) else None,
                  primal=evidence,
                  input=dict(nfp=int(inp.nfp), ncurr=int(inp.ncurr),
                             mpol=int(inp.mpol), ntor=int(inp.ntor), ns=int(cfg.resolution.ns),
                             pressure_axis_Pa=float(inp.am[0]), curtor_A=float(inp.curtor),
                             phiedge_Wb=float(inp.phiedge)))
    if host is not None:
        report['host_iterations'] = int(host[1].iterations)
        report['host_converged'] = bool(host[1].converged)
    arrays = {f'state_{i}':np.asarray(v) for i,v in enumerate(jax.tree.leaves(state))}
    report['finite_returned_state'] = all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(a.output/'state.npz', **arrays)
    if int(status) == 1:
        report['error'] = str(imp._LAST_STATUS_ERROR.get(cfg))
        report['failed_state_may_be_callback_placeholder'] = True
    write(a.output / 'report.json', report)
    if not report['finite_returned_state']:
        write(a.output / 'report.json', report)
        return
    rt = imp.runtime_from_params(params, cfg)
    geometry, jacobian, metrics, fields, energies = _field_chain(state, rt)
    currents = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv,
                               trig=rt.trig, s=rt.setup.s_full, signgs=int(rt.setup.signgs))
    mu0 = 4e-7 * np.pi
    current_A = float(currents.ctor) / mu0
    sf = np.asarray(rt.setup.s_full)
    sh = np.r_[0., (sf[1:] + sf[:-1]) / 2]
    target_pressure = np.asarray(pressure(inp.pmass_type, inp.am, inp.am_aux_s,
        inp.am_aux_f, sh, pres_scale=inp.pres_scale, bloat=inp.bloat, spres_ped=inp.spres_ped))
    actual_pressure = np.asarray(fields.pressure) / mu0
    with netcdf_file(a.reference_wout, 'r', mmap=False) as f:
        reference_current = float(f.variables['ctor'].data)
        reference_beta = float(f.variables['betatotal'].data)
    report['constraints'] = dict(
        computed_current_A=current_A, input_current_A=float(inp.curtor),
        reference_wout_current_A=reference_current, reference_beta_fraction=reference_beta,
        current_minus_input_A=current_A-float(inp.curtor),
        current_minus_reference_A=current_A-reference_current,
        pressure_half_mesh_relative_max_error=float(np.max(abs(actual_pressure[1:]-target_pressure[1:])) / np.max(abs(target_pressure[1:]))),
        note='Current edge extrapolation need not equal input CURTOR exactly on this finite mesh; record differences without silently changing constraints.')
    if int(status) == 0 and evidence.get('derivative_certified'):
        start = time.perf_counter()
        repeated = imp._host_solve_and_mask_status(cfg, params)
        report['memo_repeat_seconds'] = time.perf_counter()-start
        report['memo_repeat_exact'] = all(np.array_equal(x,y) for x,y in zip(jax.tree.leaves(state),jax.tree.leaves(repeated[0])))
    arrays.update(pressure_half=actual_pressure, pressure_target_half=target_pressure, buco=np.asarray(currents.buco))
    np.savez_compressed(a.output/'state.npz', **arrays)
    report['finite_state_and_profiles'] = all(np.isfinite(v).all() for v in arrays.values())
    write(a.output/'report.json', report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vmex-root',type=Path,required=True)
    p.add_argument('--solvax-root',type=Path)
    p.add_argument('--source-commit')
    p.add_argument('--cold-runs',type=int,choices=(1,2),default=1)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reference-wout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--per-run-timeout',type=int,default=600)
    a=p.parse_args()
    if a.worker:
        worker(a)
        return
    if not 1 <= a.per_run_timeout <= 600:
        p.error('Per-run timeout must be 1..600 seconds')
    a.output.mkdir(parents=True,exist_ok=False)
    source=a.vmex_root.resolve()
    report=dict(input_sha256=digest(a.input),reference_wout_sha256=digest(a.reference_wout),
        vmex_source=source_manifest(source, 'vmex', a.source_commit),
        requested_cold_runs=a.cold_runs,
        harness_sha256=digest(Path(__file__)),runs=[],
        scope='Finite-beta QA fixed-boundary root/constraint/repeatability diagnostic; no AD, optimization, grid-convergence or HINT comparison certificate.')
    if a.solvax_root is not None:
        report['solvax_source'] = source_manifest(a.solvax_root, 'src/solvax')
    write(a.output/'qualification.json',report)
    for i in range(a.cold_runs):
        folder=(a.output/f'cold-{i+1}').resolve(); folder.mkdir()
        pythonpath=[str(source)]
        if a.solvax_root is not None: pythonpath.append(str(a.solvax_root.resolve()/'src'))
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(pythonpath+[os.environ.get('PYTHONPATH','')]),JAX_ENABLE_X64='1',
                 VMEX_COMPILATION_CACHE='disabled',JAX_COMPILATION_CACHE_DIR=str(folder/'jax-cache'),
                 XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--vmex-root',str(source),
             '--input',str(a.input.resolve()),'--reference-wout',str(a.reference_wout.resolve()),'--output',str(folder)]
        if a.solvax_root is not None: cmd += ['--solvax-root',str(a.solvax_root.resolve())]
        with (folder/'output.log').open('w') as log:
            try:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.per_run_timeout)
                row=dict(returncode=result.returncode)
            except subprocess.TimeoutExpired:
                row=dict(timeout=True)
        if (folder/'report.json').exists(): row['measurements']=json.loads((folder/'report.json').read_text())
        report['runs'].append(row); write(a.output/'qualification.json',report)
        if (row.get('returncode') != 0 or row.get('measurements',{}).get('status') != 0
                or not row.get('measurements',{}).get('primal',{}).get('derivative_certified')):
            report['stopped_after_uncertified_or_failed_run'] = True
            write(a.output/'qualification.json',report)
            break
    paths=[a.output/f'cold-{i+1}'/'state.npz' for i in range(2)]
    if all(x.exists() for x in paths):
        import numpy as np
        with np.load(paths[0]) as first,np.load(paths[1]) as second:
            keys=[k for k in first.files if k.startswith('state_')]
            d=np.concatenate([(first[k]-second[k]).ravel() for k in keys])
            v=np.concatenate([first[k].ravel() for k in keys])
            report['cold_repeat_relative_state_l2']=float(np.linalg.norm(d)/max(np.linalg.norm(v),1e-300))
            report['cold_repeat_max_abs_state_difference']=float(np.max(abs(d)))
        write(a.output/'qualification.json',report)
    print(a.output/'qualification.json')


if __name__=='__main__':
    main()
