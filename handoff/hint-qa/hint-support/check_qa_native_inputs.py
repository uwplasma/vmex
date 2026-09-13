"""Native spline transfer, initial pressure and cut-current diagnostics."""
from pathlib import Path
import json,subprocess,os
import numpy as np
from netCDF4 import Dataset
r=Path(__file__).resolve().parents[1];results=[]
for case in ['beta0p5','beta2p5']:
 d=r/'runs/qa-native-inputs'/case;old=r/'qa-source'/case;m=json.loads((old/'manifest.json').read_text())
 with Dataset(d/'vacuum.nc') as src,Dataset(d/'vacuum-as-field.nc','w') as f:
  for name,dim in src.dimensions.items():f.createDimension(name,len(dim))
  for name in ['mtor','rminb','rmaxb','zminb','zmaxb']:v=src[name];f.createVariable(name,v.dtype,v.dimensions)[:]=v[:]
  for name in ['R','phi','Z']:v=src[name];f.createVariable(name,v.dtype,v.dimensions)[:]=v[:]
  b=np.stack([src[name][:] for name in ['Bvac_R','Bvac_phi','Bvac_Z']],axis=-1)
  for name,k in [('B_R',0),('B_phi',1),('B_Z',2)]:f.createVariable(name,'f8',('phi','Z','R'))[:]=b[...,k]
  for name in ['v_R','v_phi','v_Z','P']:f.createVariable(name,'f8',('phi','Z','R'))[:]=0.
 target=np.load(r/'runs/qa-vacuum-grid/gpu'/case/'n64t32/target-check.npz');x=target['xyz'];phi=np.arctan2(x[:,1],x[:,0]);points=np.column_stack([np.linalg.norm(x[:,:2],axis=1),phi,x[:,2]])
 with (d/'points.input').open('w') as f:f.write(str(len(x))+'\n');np.savetxt(f,points)
 with (d/'points.input').open() as fi,(d/'native-points.log').open('w') as fo:subprocess.run([r/'build-debug/MAGVAL/magval_points.exe','vacuum-as-field.nc'],cwd=d,stdin=fi,stdout=fo,stderr=subprocess.STDOUT,check=True,timeout=30,env={**os.environ,'OMP_NUM_THREADS':'1'})
 q=np.loadtxt(d/'native-points.csv',delimiter=',');cart=np.stack([q[:,0]*np.cos(phi)-q[:,1]*np.sin(phi),q[:,0]*np.sin(phi)+q[:,1]*np.cos(phi),q[:,2]],axis=-1);err=np.linalg.norm(cart-target['B_direct'],axis=1)
 with Dataset(d/'flux.nc') as f:s=np.asarray(f['norm_s'][:])
 with Dataset(d/'vacuum.nc') as f:R=f['R'][:];Z=f['Z'][:]
 table=np.loadtxt(old/'current_mapping.csv',delimiter=',',skiprows=1);lam=np.interp(s,table[:,0],table[:,1]);lam=np.where(s<1,lam,0);integrals=np.sum(lam*b[...,1],axis=(1,2))*(R[1]-R[0])*(Z[1]-Z[0]);curr=m['source_current_A']*integrals/integrals[0]
 walls=[]
 for pad in [10,12,15]:
  with Dataset(d/f'limiter-{pad}.nc') as f:wall=np.asarray(f['limiter'][:]);walls.append({'buffer_cm':pad,'plasma_cells_excluded':int(np.sum((s<1)&(wall<.5)))})
 row={'case':case,'native_spline_rms_error_T':float(np.sqrt(np.mean(err**2))),'native_spline_max_error_T':float(err.max()),'native_spline_rms_relative':float(np.sqrt(np.mean(err**2/np.sum(target['B_direct']**2,axis=1)))),'initial_parallel_current_A_by_cut':curr.tolist(),'max_initial_parallel_current_relative_cut_variation':float(np.max(np.abs(curr/curr[0]-1))),'current_interpretation':'Only imposed lambda(s)B contribution on initial vacuum field; normalization on phi=0 is algebraic. Total relaxed current and closure still require validation.','walls':walls}
 results.append(row);np.savez_compressed(d/'native-target-check.npz',xyz=x,B_native=cart,B_direct=target['B_direct']);print(json.dumps(row),flush=True)
(r/'local-support/qa-native-input-validation.json').write_text(json.dumps(results,indent=2)+'\n')
