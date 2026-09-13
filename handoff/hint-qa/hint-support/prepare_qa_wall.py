"""Build registered poloidal-buffer walls and conservative filament-clearance bounds."""
from pathlib import Path
import json
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
from scipy.spatial import cKDTree
from shapely.geometry import Polygon,Point
from shapely.geometry.polygon import orient
from matplotlib.path import Path as MPath
from vmex.core.wout import read_wout
from vmex.core.virtual_casing import surface_field_data_from_wout
from essos.coils import Coils
r=Path(__file__).resolve().parents[1];out=r/'runs/qa-native-inputs';out.mkdir(exist_ok=True)
src=r/'qa-source';w=read_wout(next((src/'beta0p5/inputs').glob('wout*')));nfp=int(w.nfp);nt=128;nu=256;theta=np.arange(nu)*2*np.pi/nu;phi=np.arange(nt)*2*np.pi/(nfp*nt)
phases=theta[None,:,None]*np.asarray(w.xm)[None,None,:]-phi[:,None,None]*np.asarray(w.xn)[None,None,:]
R=np.cos(phases)@np.asarray(w.rmnc[-1]);Z=np.sin(phases)@np.asarray(w.zmns[-1]);edge=np.stack([R,Z],axis=-1)
sd=surface_field_data_from_wout(w,nphi=4,ntheta=12,use_stellsym=False);g=np.moveaxis(np.asarray(sd.gamma),0,-1).reshape(-1,3);normal=np.moveaxis(np.asarray(sd.normal),0,-1).reshape(-1,3);targets=np.concatenate([g+d*float(w.Aminor_p)*normal for d in [.05,.15,.3,.5]])
coils=[]
for case in ['beta0p5','beta2p5']:
 c=Coils.from_json(str(next((src/case/'inputs').glob('*.json'))));c.n_segments=2048;gamma=np.asarray(c.gamma);coeff=np.asarray(c.curves.curves);order=(coeff.shape[-1]-1)//2
 speed=np.zeros(len(gamma))
 for m in range(1,order+1):speed+=2*np.pi*m*(np.linalg.norm(coeff[:,:,2*m-1],axis=1)+np.linalg.norm(coeff[:,:,2*m],axis=1))
 coils.append((case,gamma,float(speed.max()/(2*2048))))
reports=[]
for pad in [.10,.12,.15]:
 wall=[]
 for k in range(nt):
  poly=orient(Polygon(edge[k]).buffer(pad,quad_segs=32),sign=1)
  assert poly.is_valid and poly.geom_type=='Polygon'
  line=poly.exterior;start=line.project(Point(edge[k,0]+[pad,0]));ring=np.array([line.interpolate((start+u*line.length/nu)%line.length).coords[0] for u in range(nu)])
  assert Polygon(ring).covers(Polygon(edge[k]));wall.append(ring)
 wall=np.array(wall)
 def inside(xyz):
  ang=np.mod(np.arctan2(xyz[:,1],xyz[:,0]),2*np.pi/nfp);u=ang/(2*np.pi/nfp)*nt;k=np.floor(u).astype(int)%nt;a=u-np.floor(u);xy=np.column_stack([np.linalg.norm(xyz[:,:2],axis=1),xyz[:,2]])
  return np.array([MPath((1-f)*wall[i]+f*wall[(i+1)%nt]).contains_point(p) for i,f,p in zip(k,a,xy)])
 full=np.tile(wall,(nfp,1,1));phis=np.arange(nt*nfp)*2*np.pi/(nt*nfp);xyz=np.stack([full[:,:,0]*np.cos(phis[:,None]),full[:,:,0]*np.sin(phis[:,None]),full[:,:,1]],axis=-1)
 # Any point of native RZ-linear interpolated wall lies within this bound of a corner vertex.
 patch_bound=float(np.linalg.norm(np.roll(wall,-1,axis=1)-wall,axis=-1).max()+np.linalg.norm(np.roll(wall,-1,axis=0)-wall,axis=-1).max()+wall[:,:,0].max()*2*np.pi/(nt*nfp))
 tree=cKDTree(xyz.reshape(-1,3));row={'poloidal_buffer_m':pad,'surface_grid_per_period':[nt,nu],'all_192_exterior_targets_inside':bool(inside(targets).all()),'targets_outside':int((~inside(targets)).sum()),'wall_patch_to_vertex_bound_m':patch_bound,'coils':[]}
 for case,cg,gap in coils:
  nearest=float(tree.query(cg.reshape(-1,3))[0].min());lower=nearest-patch_bound-gap;starts_outside=bool((~inside(cg[:,0])).all())
  row['coils'].append({'case':case,'sample_vertex_distance_m':nearest,'continuous_curve_sampling_bound_m':gap,'conservative_curve_to_wall_lower_bound_m':lower,'all_coil_start_points_outside':starts_outside})
  assert lower>0 and starts_outside
 name=f'wall-{round(pad*100):02d}cm.dat'
 with (out/name).open('w') as f:
  f.write(f'{nfp}, {nu+1}, {nt},\n')
  for ring in wall:np.savetxt(f,np.vstack([ring,ring[0]]),fmt='%.16e',delimiter=', ')
 np.savez_compressed(out/name.replace('.dat','.npz'),RZ=wall,phi=phi)
 reports.append(row)
box=[float(R.min()-.19),float(R.max()+.19),float(Z.min()-.19),float(Z.max()+.19)]
(out/'wall-validation.json').write_text(json.dumps({'wall_definition':'Poloidal polygon buffer, registered/resampled and linearly interpolated between toroidal sections. Not constant 3D normal offset; no conductor thickness. Clearance bound includes wall interpolation and Fourier-curve sampling.','box_m':box,'results':reports},indent=2)+'\n');print(json.dumps(reports,indent=2))
for case in ['beta0p5','beta2p5']:
 d=out/case;d.mkdir(exist_ok=True);wf=next((src/case/'inputs').glob('wout*'))
 grid=f'mtor=2,nr=64,nz=64,ntor=32,rminb={box[0]},rmaxb={box[1]},zminb={box[2]},zmaxb={box[3]}'
 (d/'mkflx.input').write_text(f"&nlinp1 flx_form='netcdf',flx_file='flux.nc',file_format='netcdf',wout_file='{wf}',{grid},ntheta=256,slimit=1.0 /\n")
 for pad in [10,12,15]:(d/f'mklim-{pad}.input').write_text(f"&nlinp lim_form='netcdf',lim_file='limiter-{pad}.nc',vessel_model='3D',vessel_file='../wall-{pad:02d}cm.dat',{grid} /\n")
