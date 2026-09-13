"""Extract compact, independently computed snapshot checks; no convergence flag."""
import argparse,json,hashlib
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();d=a.run
fpath=next(d.glob('*.mag.h5'));rows=[];previous=None;mu0=4e-7*np.pi
with h5py.File(fpath,'r') as f:
 nr,nz,ntor=(int(f[x][()]) for x in ['nr','nz','ntor']);R=np.linspace(float(f['rminb'][()]),float(f['rmaxb'][()]),nr);Z=np.linspace(float(f['zminb'][()]),float(f['zmaxb'][()]),nz)
 dr=R[1]-R[0];dz=Z[1]-Z[0];dv=R[None,None,:]*dr*dz*2*np.pi/ntor
 keys=sorted((k for k in f if isinstance(f[k],h5py.Group)),key=float)
 for k in keys:
  g=f[k];b=np.stack([g[c][:] for c in ['B_R','B_phi','B_Z']],axis=-1);P=g['P'][:];p=P/mu0
  row={'time':float(k),'finite':bool(np.isfinite(b).all() and np.isfinite(P).all()),'max_pressure_Pa':float(p.max()),'pressure_integral_J':float(np.sum(p*dv)),'positive_pressure_volume_m3':float(np.sum((P>1e-8*P.max())*dv)),'max_field_T':float(np.linalg.norm(b,axis=-1).max())}
  if previous:
   oldb,oldp=previous;db=np.linalg.norm(b-oldb,axis=-1);row.update(max_delta_B_T=float(db.max()),rms_delta_B_T=float(np.sqrt(np.mean(db**2))),relative_pressure_l2=float(np.linalg.norm(P-oldp)/np.linalg.norm(P)))
  rows.append(row);previous=(b,P)
 np.savez_compressed(d/'final-snapshot.npz',R=R,Z=Z,nfp=int(f['mtor'][()]),phi=np.arange(ntor)*2*np.pi/(int(f['mtor'][()])*ntor),B_cyl=b,P_mu0_Pa=P)
summary={'file':str(fpath),'sha256':hashlib.file_digest(fpath.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(fpath.read_bytes()).hexdigest(),'pressure_units':'HINT P is mu0 times pressure in Pa','integration':'uniform cylindrical rectangle sum over full torus; pressure-defined support is not a magnetic boundary','convergence_certified':False,'snapshots':rows}
(d/'snapshot-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(rows[-2:],indent=2))
