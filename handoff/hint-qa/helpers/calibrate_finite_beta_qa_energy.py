"""Synthetic one-parameter QA energy calibration with an independent cold check.

Host safeguarded Newton consumes the ordinary live VMEX custom VJP. VMEX's
primal refinement and adjoint use SOLVAX; this is not a native SOLVAX nonlinear
wrapper (those request JVPs unavailable on VMEX's custom-VJP solve).
The target comes from a prior certified interior point, not a reactor design
objective. Run only after coordinating the CPU slot, under run_bounded.py.
"""
import argparse
import dataclasses
import json
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
    assert Path(solvax.__file__).resolve() == (a.solvax_root/'src/solvax/__init__.py').resolve()
    target_report=json.loads((a.target/'report.json').read_text())
    target_row=next(r for r in target_report['taylor'] if r['h'] == 5e-5)
    target=float(target_row['plus']['value'])
    assert target_row['plus']['primal']['derivative_certified']
    inp=VmecInput.from_file(a.input)
    assert inp.ncurr == 1 and not inp.lfreeb and inp.nfp == 2
    device=jax.devices()[0]
    cfg=imp.make_config(inp,hot_restart=False,
        device=device if device.platform == 'cpu' else None)
    assert (inp.mpol,inp.ntor,int(cfg.resolution.ns)) == (7,6,31)
    base=imp.params_from_input(inp,device=device)
    scale=abs(float(base.rbc[inp.ntor,0]))
    assert scale == target_report['direction']['meters_per_unit_parameter']
    def params(q):
        return dataclasses.replace(base,rbc=base.rbc.at[inp.ntor,1].add(q*scale))
    def objective(q):
        pp=params(q)
        return imp.mhd_energy(imp.solve_implicit(pp,cfg),imp.runtime_from_params(pp,cfg))[0]
    def certify(q):
        pp=params(q)
        state,_,status,_,_=imp._host_solve_and_mask_status(cfg,pp)
        cert=imp._LAST_PRIMAL_CERTIFICATE.get(cfg)
        matched=(cert is not None and cert[0] == imp._params_key(pp)
                 and cert[1] == imp._primal_state_key(state))
        if int(status) != 0 or not matched or not cert[2]['derivative_certified']:
            raise RuntimeError('Calibration trial lacks an actual-state primal certificate')
        return state,dict(cert[2])

    if a.worker == 'cold':
        opt=json.loads((a.output/'optimization.json').read_text())
        q=opt['accepted_q']
        state,cert=certify(q)
        value=float(objective(jnp.asarray(q)))
        with np.load(a.output/'accepted-state.npz') as previous:
            old=[previous[f'state_{i}'] for i in range(len(jax.tree.leaves(state)))]
            differences=np.concatenate([(np.asarray(v)-w).ravel() for v,w in zip(jax.tree.leaves(state),old,strict=True)])
            original=np.concatenate([v.ravel() for v in old])
        report=dict(q=q,value=value,target=target,residual=value-target,primal=cert,
            independent_fresh_process=True,
            relative_state_difference=float(np.linalg.norm(differences)/max(np.linalg.norm(original),1e-300)),
            value_difference_from_accepted=value-opt['accepted_value'],
            within_calibration_tolerance=abs(value-target)<=1e-9)
        write(a.output/'cold-final.json',report)
        if not report['within_calibration_tolerance']:
            raise RuntimeError('Independent cold final value misses calibration tolerance')
        return

    report=dict(label='Synthetic QA energy calibration; not a physical optimized reactor',
        target=target,known_target_q=5e-5,bounds=[-1e-4,1e-4],absolute_energy_tolerance=1e-9,
        fixed_constraints=dict(ncurr=int(inp.ncurr),curtor_A=float(inp.curtor),
            pressure_axis_Pa=float(inp.am[0]),phiedge_Wb=float(inp.phiedge)),
        nonlinear_method='Host safeguarded scalar Newton; ordinary VMEX custom VJP, with SOLVAX inside primal refinement and adjoint',
        iterations=[],trials=[],completed=False)
    def save(): write(a.output/'optimization.json',report)
    q=0.0
    vg=jax.value_and_grad(objective)
    try:
        for iteration in range(4):
            report['active_phase']=f'iteration_{iteration}_certified_value_and_gradient';save()
            print(report['active_phase'],flush=True)
            started=time.perf_counter()
            state,cert=certify(q)
            value,gradient=vg(jnp.asarray(q))
            value,gradient=float(value),float(gradient)
            residual=value-target
            report['iterations'].append(dict(q=q,value=value,residual=residual,gradient=gradient,
                primal=cert,elapsed_seconds=time.perf_counter()-started))
            save()
            if not np.isfinite(value) or not np.isfinite(gradient):
                raise FloatingPointError('Nonfinite live objective or derivative')
            if abs(residual)<=1e-9:
                report.update(completed=True,accepted_q=q,accepted_value=value,
                    recovered_q_error=q-5e-5,active_phase=None)
                np.savez_compressed(a.output/'accepted-state.npz',
                    **{f'state_{i}':np.asarray(v) for i,v in enumerate(jax.tree.leaves(state))},
                    rbc=np.asarray(params(q).rbc))
                save();return
            if iteration == 3:
                raise RuntimeError('Three Newton updates exhausted before calibration tolerance')
            if abs(gradient)<1e-12:
                raise RuntimeError('Derivative too small for safeguarded scalar Newton')
            step=float(np.clip(q-residual/gradient,-1e-4,1e-4))-q
            accepted=False
            for backtrack in range(3):
                candidate=q+step/(2**backtrack)
                report['active_phase']=f'iteration_{iteration}_trial_{backtrack}';save()
                print(report['active_phase'],flush=True)
                _,trial_cert=certify(candidate)
                trial_value=float(objective(jnp.asarray(candidate)))
                trial_residual=trial_value-target
                accepted=(np.isfinite(trial_value) and abs(trial_residual)<abs(residual))
                report['trials'].append(dict(q=candidate,value=trial_value,residual=trial_residual,
                    accepted=bool(accepted),primal=trial_cert));save()
                if accepted:
                    q=candidate;break
            if not accepted:
                raise RuntimeError('No decreasing certified trial within two backtracks')
    except Exception as exc:
        report['exception']=dict(type=type(exc).__name__,message=str(exc));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vmex-root',type=Path,required=True)
    p.add_argument('--solvax-root',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=['optimize','cold'])
    a=p.parse_args()
    if a.worker:
        worker(a);return
    a.output.mkdir(parents=True,exist_ok=False)
    source=source_manifest(a.vmex_root,'vmex')
    solvax_source=source_manifest(a.solvax_root,'src/solvax')
    target_source=json.loads((a.target/'provenance.json').read_text())
    assert source['source_sha256'] == target_source['vmex_source']['source_sha256']
    assert solvax_source['source_sha256'] == target_source['solvax_source']['source_sha256']
    assert digest(a.input) == target_source['input_sha256']
    report=dict(vmex_source=source,solvax_source=solvax_source,input_sha256=digest(a.input),
        harness_sha256=digest(Path(__file__)),helper_sha256=digest(Path(__file__).with_name('qualify_finite_beta_qa_root.py')),
        target_report_sha256=digest(a.target/'report.json'),
        target_provenance_sha256=digest(a.target/'provenance.json'),workers=[])
    write(a.output/'provenance.json',report)
    start=time.monotonic()
    for mode in ('optimize','cold'):
        timeout=min(180,240-(time.monotonic()-start))
        if timeout<=0: raise SystemExit('Total calibration budget exhausted')
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(a.vmex_root.resolve()),str(a.solvax_root.resolve()/'src')]),
            JAX_ENABLE_X64='1',VMEX_COMPILATION_CACHE='disabled',
            JAX_COMPILATION_CACHE_DIR=str(a.output.resolve()/f'jax-cache-{mode}'),
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',mode]
        for name in ('vmex_root','solvax_root','input','target','output'):
            cmd.extend(['--'+name.replace('_','-'),str(getattr(a,name).resolve())])
        with (a.output/f'{mode}.log').open('w') as log:
            try:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                row=dict(mode=mode,returncode=result.returncode)
            except subprocess.TimeoutExpired:
                row=dict(mode=mode,timeout=True)
        report['workers'].append(row);write(a.output/'provenance.json',report)
        if row.get('returncode') != 0: raise SystemExit(124 if row.get('timeout') else row['returncode'])
    print(a.output/'cold-final.json')


if __name__ == '__main__': main()
