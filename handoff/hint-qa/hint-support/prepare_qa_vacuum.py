"""Bounded direct coil-field grids plus common-target interpolation checks."""
import argparse,json,time
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from essos.coils import Coils
from essos.fields import BiotSavart
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--nr',type=int,default=64);p.add_argument('--ntor',type=int,default=32);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--platform',required=True);a=p.parse_args()
assert jax.default_backend()==a.platform
r=Path(__file__).resolve().parents[1];geometry=r/'runs/qa-native-inputs';box=json.loads((geometry/'wall-validation.json').read_text())['box_m'];d=r/'runs/qa-vacuum-grid'/a.platform/a.case/f'n{a.nr}t{a.ntor}';d.mkdir(parents=True,exist_ok=True)
c=Coils.from_json(str(next((a.source_root/a.case/'inputs').glob('*.json'))));coil=jax.jit(jax.vmap(BiotSavart(c).B))
R=np.linspace(box[0],box[1],a.nr);Z=np.linspace(box[2],box[3],a.nr);phi=np.arange(a.ntor)*np.pi/a.ntor
pp,zz,rr=np.meshgrid(phi,Z,R,indexing='ij');xyz=np.stack([rr*np.cos(pp),rr*np.sin(pp),zz],axis=-1);flat=xyz.reshape(-1,3);bc=np.empty_like(flat);start=time.monotonic()
for i in range(0,len(flat),512):bc[i:i+512]=np.asarray(coil(jnp.asarray(flat[i:i+512])))
bc=bc.reshape(xyz.shape);b=np.stack([bc[...,0]*np.cos(pp)+bc[...,1]*np.sin(pp),-bc[...,0]*np.sin(pp)+bc[...,1]*np.cos(pp),bc[...,2]],axis=-1)
assert np.isfinite(b).all()
with Dataset(d/'vacuum.nc','w') as f:
 for name,v in [('R',R),('Z',Z),('phi',phi)]:f.createDimension(name,len(v));f.createVariable(name,'f8',(name,))[:]=v
 for name,v in {'mtor':2,'rminb':R[0],'rmaxb':R[-1],'zminb':Z[0],'zmaxb':Z[-1]}.items():f.createVariable(name,'i4' if name=='mtor' else 'f8').assignValue(v)
 for k,name in enumerate(['Bvac_R','Bvac_phi','Bvac_Z']):f.createVariable(name,'f8',('phi','Z','R'))[:]=b[...,k]
x=np.load(geometry/'exterior-targets.npz')['xyz'];tphi=np.mod(np.arctan2(x[:,1],x[:,0]),np.pi);tr=np.linalg.norm(x[:,:2],axis=1);bexact=np.concatenate([np.asarray(coil(jnp.asarray(x[i:i+64]))) for i in range(0,len(x),64)])
# Same Cartesian points can be in any field period; interpolate cylindrical periodic components.
interp=RegularGridInterpolator((np.r_[phi,np.pi],Z,R),np.concatenate([b,b[:1]],axis=0));bgrid=interp(np.column_stack([tphi,x[:,2],tr]));ang=np.arctan2(x[:,1],x[:,0]);bgridxyz=np.stack([bgrid[:,0]*np.cos(ang)-bgrid[:,1]*np.sin(ang),bgrid[:,0]*np.sin(ang)+bgrid[:,1]*np.cos(ang),bgrid[:,2]],axis=-1)
err=np.linalg.norm(bgridxyz-bexact,axis=1);report={'case':a.case,'platform':a.platform,'grid':[a.nr,a.nr,a.ntor],'seconds':time.monotonic()-start,'targets':len(x),'interpolator':'Independent trilinear periodic cylindrical interpolation; native HINT spline has separate error.','rms_error_T':float(np.sqrt(np.mean(err**2))),'max_error_T':float(err.max()),'rms_relative_error':float(np.sqrt(np.mean(err**2/np.sum(bexact**2,axis=1))))}
np.savez_compressed(d/'target-check.npz',xyz=x,B_direct=bexact,B_interpolated=bgridxyz);(d/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
