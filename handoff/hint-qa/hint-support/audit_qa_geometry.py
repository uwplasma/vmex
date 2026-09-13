"""Measure QA coil clearance and boundary normal mismatch without HINT evolution."""
import json,time,argparse
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from vmex.core.wout import read_wout
from vmex.core import virtual_casing as vc
from essos.coils import Coils
from essos.fields import BiotSavart
from virtual_casing_jax import VirtualCasingJAX
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--platform',required=True,choices=['cpu','gpu']);p.add_argument('--case',choices=['beta0p5','beta2p5'],required=True);p.add_argument('--levels',type=int,nargs='+',default=[64,128,256]);p.add_argument('--source-root',type=Path);opts=p.parse_args()
if jax.default_backend()!=opts.platform:raise RuntimeError('Wrong backend')
rows=[]
for case in [opts.case]:
 src=(opts.source_root or r/'qa-source')/case/'inputs'; out=r/'runs/qa-geometry'/opts.platform/case;out.mkdir(parents=True,exist_ok=True)
 w=read_wout(next(src.glob('wout*')));coils=Coils.from_json(str(next(src.glob('*.json')))); bs=BiotSavart(coils); coil=jax.jit(jax.vmap(bs.B))
 sd=vc.surface_field_data_from_wout(w,nphi=128,ntheta=256,use_stellsym=False)
 gamma=np.moveaxis(np.asarray(sd.gamma),0,-1).reshape(-1,3)
 # Replicate the WOUT field period for comparison with all physical coils.
 surfaces=[]
 for k in range(w.nfp):
  a=2*np.pi*k/w.nfp;c,s=np.cos(a),np.sin(a);surfaces.append(gamma@np.array([[c,s,0],[-s,c,0],[0,0,1]]))
 xyz=np.asarray(coils.gamma).reshape(-1,3);dist,idx=cKDTree(np.concatenate(surfaces)).query(xyz)
 gr=np.linalg.norm(gamma[:,:2],axis=1);pad=float(w.Aminor_p);zmax=float(np.abs(gamma[:,2]).max())+pad;box=[float(gr.min()-pad),float(gr.max()+pad),-zmax,zmax];rr=np.linalg.norm(xyz[:,:2],axis=1)
 inbox=(rr>=box[0])&(rr<=box[1])&(xyz[:,2]>=box[2])&(xyz[:,2]<=box[3])
 row={'case':case,'sampled_coil_to_lcfs_min_m':float(dist.min()),'clearance_note':'Discrete surface and coil samples; not a certified minimum or conductor-radius allowance.','coil_samples':len(xyz),'coil_samples_inside_old_rectangular_box':int(inbox.sum()),'old_box_m':box,'boundary_refinements':[]}
 print(json.dumps(row),flush=True)
 for n in opts.levels:
  start=time.monotonic();source=vc.surface_field_data_from_wout(w,nphi=64,ntheta=64,use_stellsym=False)
  sd=vc.surface_field_data_from_wout(w,nphi=16,ntheta=16,use_stellsym=False)
  g=np.moveaxis(np.asarray(sd.gamma),0,-1).reshape(-1,3);bc=np.concatenate([np.asarray(coil(jnp.asarray(g[i:i+256]))) for i in range(0,len(g),256)])
  v=VirtualCasingJAX();v.setup(6,int(w.nfp),False,64,64,source.gamma,64,64,16,16)
  bp=np.moveaxis(np.asarray(v.compute_internal_B(source.B_total,digits=6,quad_nt=2*n,quad_np=n,chunk_size=128,target_chunk_size=4,interp_block_size=16,remat=True)),0,-1).reshape(-1,3)
  normal=np.moveaxis(np.asarray(sd.normal),0,-1).reshape(-1,3);total=bc+bp;bn=np.sum(total*normal,axis=1)
  area=np.linalg.norm(np.asarray(sd.area_vector),axis=0).ravel();area/=area.sum()
  b2=np.sum(np.asarray(sd.B_total)**2,axis=0).ravel()
  diag={'quadrature_full_torus':[2*n,n],'source_grid_per_period':[64,64],'target_grid_per_period':[16,16],'rms_Bn_T':float(np.sqrt(np.sum(area*bn**2))),'max_abs_Bn_T':float(np.max(np.abs(bn))),'area_rms_relative_Bn':float(np.sqrt(np.sum(area*bn**2/b2))),'seconds':time.monotonic()-start}
  row['boundary_refinements'].append(diag);np.savez_compressed(out/f'boundary-q{n}.npz',xyz=g,B_plasma=bp,B_coils=bc,Bn=bn,area_weights=area)
  (out/'audit.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(diag),flush=True);jax.clear_caches()
 rows.append(row)
(out/'geometry-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
