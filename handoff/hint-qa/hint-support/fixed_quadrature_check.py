"""CPU/GPU parity and distinct fixed quadrature checks on identical QA targets."""
import argparse,json,time,hashlib,os
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from vmex.core.wout import read_wout
from vmex.core.virtual_casing import surface_field_data_from_wout
from vmex.core.extender import VmecExtender
from essos.coils import Coils
from essos.fields import BiotSavart
p=argparse.ArgumentParser();p.add_argument('--case',required=True,choices=['beta0p5','beta2p5']);p.add_argument('--platform',required=True,choices=['cpu','gpu']);p.add_argument('--quadrature',type=int,nargs='+',default=[512]);p.add_argument('--targets',type=int,default=16);p.add_argument('--source-root',type=Path);a=p.parse_args()
if jax.default_backend()!=a.platform:raise RuntimeError(f'Requested {a.platform}, got {jax.default_backend()}')
r=Path(__file__).resolve().parents[1];src=a.source_root or r/'qa-source';src=src/a.case/'inputs'
out=r/'runs/fixed-quadrature'/a.platform/a.case;out.mkdir(parents=True,exist_ok=True)
wpath=next(src.glob('wout*'));cpath=next(src.glob('*.json'));w=read_wout(wpath);bs=BiotSavart(Coils.from_json(str(cpath)));coil=jax.jit(jax.vmap(bs.B))
sd=surface_field_data_from_wout(w,nphi=128,ntheta=128,use_stellsym=False)
target=surface_field_data_from_wout(w,nphi=4,ntheta=12,use_stellsym=False)
g=np.moveaxis(np.asarray(target.gamma),0,-1).reshape(-1,3);n=np.moveaxis(np.asarray(target.normal),0,-1).reshape(-1,3);n=n/np.linalg.norm(n,axis=1)[:,None]
offsets=np.array([.05,.15,.3,.5])*float(w.Aminor_p);xyz=np.concatenate([g+d*n for d in offsets]);indices=np.unique(np.linspace(0,len(xyz)-1,min(a.targets,len(xyz)),dtype=int));xyz=xyz[indices];bc=np.asarray(coil(jnp.asarray(xyz)))
rows=[];previous=None
for q in a.quadrature:
 start=time.monotonic();ext=VmecExtender.from_surface_data(sd,external_field=coil,digits=6,levels=((q,q),),chunk_size=256,target_chunk_size=4)
 b=np.concatenate([np.asarray(ext.B(jnp.asarray(xyz[i:i+4]))) for i in range(0,len(xyz),4)])
 if not np.isfinite(b).all():raise RuntimeError('Nonfinite field')
 row={'q_full_torus':q,'seconds':time.monotonic()-start,'finite':True}
 if previous is not None:row.update(max_delta_B_T=float(np.linalg.norm(b-previous,axis=1).max()),rms_delta_B_T=float(np.sqrt(np.mean(np.sum((b-previous)**2,axis=1)))))
 np.savez_compressed(out/f'q{q}-targets{len(xyz)}.npz',xyz=xyz,indices=indices,B=b,B_coils=bc,offsets=offsets)
 rows.append(row);previous=b;print(json.dumps(row),flush=True)
 report={'platform':jax.default_backend(),'devices':[str(x) for x in jax.devices()],'jax':jax.__version__,'source_nphi':128,'source_ntheta':128,'targets':len(xyz),'input_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [wpath,cpath]},'interpretation':'Singleton fixed quadrature; no adaptive reuse. Numerical field evaluation, not HINT agreement.','levels':rows}
 (out/f'report-targets{len(xyz)}.json').write_text(json.dumps(report,indent=2)+'\n')
 jax.clear_caches()
