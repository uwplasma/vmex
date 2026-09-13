"""Native MAGVAL numerical-control differences at the fixed given-boundary targets."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--reference-steps',type=int,default=1000);p.add_argument('--candidate-steps',type=int,default=500);p.add_argument('--output',default='qa-cadence-comparison.json');p.add_argument('--reference-run');p.add_argument('--candidate-run');p.add_argument('--sample-tag');a=p.parse_args();r=a.root.resolve()
exe=r/'build-release/MAGVAL/magval_points.exe'
xyz=np.load(r/'runs/qa-native-inputs/exterior-targets.npz')['xyz'];phi=np.arctan2(xyz[:,1],xyz[:,0]);points=np.column_stack((np.linalg.norm(xyz[:,:2],axis=1),phi,xyz[:,2]))
stdin=str(len(points))+'\n'+'\n'.join(' '.join(map(str,row)) for row in points)
values=[]
for run_name,n in zip((a.reference_run,a.candidate_run),(a.reference_steps,a.candidate_steps)):
 d=r/'runs'/(run_name or f'qa-cadence-n{n}-001');w=d/(a.sample_tag or f'cadence-targets-{a.reference_steps}-{a.candidate_steps}');w.mkdir(exist_ok=False);f=np.load(d/'final-snapshot.npz');temp=w/'final-field.nc'
 with Dataset(d/'vacuum.nc') as src,Dataset(temp,'w') as dst:
  for key,dim in src.dimensions.items():dst.createDimension(key,len(dim))
  for key in ['mtor','rminb','rmaxb','zminb','zmaxb','R','phi','Z']:
   v=src[key];dst.createVariable(key,v.dtype,v.dimensions)[:]=v[:]
  for i,key in enumerate(['B_R','B_phi','B_Z']):dst.createVariable(key,'f8',('phi','Z','R'))[:]=f['B_cyl'][...,i]
  dst.createVariable('P','f8',('phi','Z','R'))[:]=f['P_mu0_Pa']
  for key in ['v_R','v_phi','v_Z']:dst.createVariable(key,'f8',('phi','Z','R'))[:]=0
 with (w/'native.log').open('w') as log:subprocess.run([str(exe),temp.name],cwd=w,input=stdin,text=True,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60,env={**os.environ,'OMP_NUM_THREADS':'1'})
 b=np.loadtxt(w/'native-points.csv',delimiter=',');assert b.shape==(len(xyz),3) and np.isfinite(b).all()
 values.append(b);temp.unlink()
delta=np.linalg.norm(values[1]-values[0],axis=1)
out=r/'local-support'/a.output;report=json.loads(out.read_text());report['given_boundary_targets']={'count':len(xyz),'rms_delta_B_T':float(np.sqrt(np.mean(delta**2))),'max_delta_B_T':float(delta.max()),'native_sampler_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'interpretation':'Same native spline and192fixed targets exterior to initialWOUT boundary, not certified exterior to relaxed current support.'};out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['given_boundary_targets'],indent=2))
