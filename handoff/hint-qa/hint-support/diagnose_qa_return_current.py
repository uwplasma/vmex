"""Nested-cut Ampere and pressure-distance shell checks; no solver execution."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt
from diagnose_qa_current_profile import derivative,MU0

p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',type=Path);a=p.parse_args()
if a.output.exists() or (a.plot and a.plot.exists()):raise FileExistsError('Use new output paths')
with np.load(a.run/'final-snapshot.npz') as f:R,Z,phi,B,P=[f[k] for k in ('R','Z','phi','B_cyl','P_mu0_Pa')]
with Dataset(a.run/'vacuum.nc') as f:
 for key,expected in (('R',R),('Z',Z),('phi',phi)):np.testing.assert_allclose(f[key][:],expected,rtol=1e-12,atol=1e-12)
 vacuum=np.stack([f[k][:] for k in ('Bvac_R','Bvac_phi','Bvac_Z')],axis=-1)
with Dataset(a.run/'limiter-12.nc') as f:wall=np.asarray(f['limiter'][:])>.5
assert B.shape==vacuum.shape==P.shape+(3,) and wall.shape==P.shape
dr,dz=R[1]-R[0],Z[1]-Z[0];response=B-vacuum
J=(derivative(response[...,0],dz,1)-derivative(response[...,2],dr,2))/MU0
assert np.isfinite(J).all()
interior=np.zeros(P.shape,bool);interior[:,2:-2,2:-2]=True
rectangles=[]
for inset in (2,4,6,8,12,16,20):
 if min(len(R),len(Z))-2*inset<3:continue
 left,right,bottom,top=inset,len(R)-1-inset,inset,len(Z)-1-inset
 r,z=R[left:right+1],Z[bottom:top+1]
 circulation=(np.trapezoid(response[:,top,left:right+1,0]-response[:,bottom,left:right+1,0],r,axis=1)
              -np.trapezoid(response[:,bottom:top+1,right,2]-response[:,bottom:top+1,left,2],z,axis=1))/MU0
 curl_integral=np.trapezoid(np.trapezoid(J[:,bottom:top+1,left:right+1],r,axis=2),z,axis=1)
 rectangles.append(dict(inset_grid_points=inset,R_bounds_m=[float(r[0]),float(r[-1])],Z_bounds_m=[float(z[0]),float(z[-1])],
  circulation_current_A_by_cut=circulation.tolist(),curl_trapezoidal_current_A_by_cut=curl_integral.tolist(),
  mean_circulation_current_A=float(circulation.mean()),mean_curl_current_A=float(curl_integral.mean()),
  rms_discrete_stokes_mismatch_A=float(np.sqrt(np.mean((circulation-curl_integral)**2)))))
shells={}
for threshold in (0.,.01):
 plasma=P>threshold*P.max();distance=np.empty(P.shape)
 for k in range(len(phi)):
  if not plasma[k].any():raise ValueError('No pressure support on one cut')
  distance[k]=distance_transform_edt(~plasma[k],sampling=(dz,dr))
 masks={'pressure_support':plasma}
 edges=(0.,.01,.02,.04,.08,.16,np.inf)
 for lo,hi in zip(edges[:-1],edges[1:]):masks[f'outside_pressure_{lo:g}_to_{hi:g}_m']=(~plasma)&(distance>lo)&(distance<=hi)
 rows={}
 for name,m in masks.items():
  m=m&interior
  signed=np.sum(np.where(m,J,0),axis=(1,2))*dr*dz
  absolute=np.sum(np.where(m,abs(J),0),axis=(1,2))*dr*dz
  rows[name]=dict(cells=int(m.sum()),current_A_by_cut=signed.tolist(),mean_current_A=float(signed.mean()),
   mean_absolute_current_A=float(absolute.mean()),outside_wall_mean_current_A=float(np.sum(np.where(m&~wall,J,0))*dr*dz/len(phi)))
 shells[f'pressure_threshold_fraction_{threshold:g}']=rows
report=dict(run=a.run.name,rectangles=rectangles,pressure_distance_shells=shells,convergence_certified=False,
 limitations=['Distances are per-cut Euclidean R/Z distances to pressure-selected grid cells, not 3D surface-normal distances.',
 'Nested rectangles change enclosed plasma/current as well as outer margin; all-domain current is not required to equal the plasma target.',
 'Circulation uses trapezoidal line quadrature; curl uses fourth-order derivatives and trapezoidal area quadrature. Their finite-grid mismatch is a discretization diagnostic, not an independent equilibrium test.',
 'Shell integrals retain rectangular point weights used in pilot summary, excluding two R/Z layers; they are not the nested rectangle trapezoidal integrals.'],
 source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.run/'final-snapshot.npz',a.run/'vacuum.nc',a.run/'limiter-12.nc',Path(__file__))})
with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
if a.plot:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,6));v=J[0,2:-2,2:-2]/1e6;scale=np.max(abs(v))
 im=ax.pcolormesh(R[2:-2],Z[2:-2],v,cmap='RdBu_r',vmin=-scale,vmax=scale,shading='auto')
 ax.contour(R,Z,P[0],levels=[.01*P.max()],colors='black',linewidths=1)
 ax.contour(R,Z,wall[0].astype(float),levels=[.5],colors='green',linewidths=1)
 ax.set(xlabel='R [m]',ylabel='Z [m]',title=a.run.name+'; phi = 0');ax.set_aspect('equal')
 fig.colorbar(im,ax=ax,label='Response J_phi [MA/m²]');fig.tight_layout();fig.savefig(a.plot,dpi=160);plt.close(fig)
print(json.dumps({'run':a.run.name,'rectangles':[(x['inset_grid_points'],x['mean_circulation_current_A'],x['rms_discrete_stokes_mismatch_A']) for x in rectangles],
 'shells':{k:v['mean_current_A'] for k,v in shells['pressure_threshold_fraction_0'].items()}},indent=2))
