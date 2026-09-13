"""Check source field orientation; reversing coils here is diagnostic only."""
from pathlib import Path
import json,numpy as np,jax
jax.config.update('jax_enable_x64',True)
from vmex.core.wout import read_wout
from vmex.core.virtual_casing import surface_field_data_from_wout
r=Path(__file__).resolve().parents[1];rows=[]
for case in ['beta0p5','beta2p5']:
 w=read_wout(next((r/'qa-source'/case/'inputs').glob('wout*')));sd=surface_field_data_from_wout(w,nphi=16,ntheta=16,use_stellsym=False);ref=np.moveaxis(np.asarray(sd.B_total),0,-1).reshape(-1,3);n=np.moveaxis(np.asarray(sd.normal),0,-1).reshape(-1,3)
 f=np.load(r/'runs/qa-geometry/gpu'/case/'boundary-q256.npz');weights=f['area_weights'];bc=f['B_coils'];bp=f['B_plasma'];out=bc+bp
 def align(x):
  norm=np.sqrt(np.sum(weights*np.sum(x*x,axis=1))*np.sum(weights*np.sum(ref*ref,axis=1)))
  return float(np.sum(weights*np.sum(x*ref,axis=1))/norm)
 row={'case':case,'total_field_alignment':align(out),'coil_field_alignment':align(bc),'reverse_all_coils_diagnostic_rms_Bn_T':float(np.sqrt(np.sum(weights*np.sum((-bc+bp)*n,axis=1)**2))),'actual_rms_Bn_T':float(np.sqrt(np.sum(weights*f['Bn']**2)))};rows.append(row)
(r/'local-support/qa-field-direction-audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
