"""Compare two completed runs at identical elapsed magnetic time and restart."""
import argparse,json
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--reference-steps',type=int,default=1000);p.add_argument('--candidate-steps',type=int,default=500);p.add_argument('--output',default='qa-cadence-comparison.json');p.add_argument('--reference-run');p.add_argument('--candidate-run');p.add_argument('--sample-tag');a=p.parse_args();r=a.root.resolve()
runs=[r/'runs'/(name or f'qa-cadence-n{n}-001') for name,n in zip((a.reference_run,a.candidate_run),(a.reference_steps,a.candidate_steps))]
provenance=[json.loads((d/'restart-provenance.json').read_text()) for d in runs]
assert provenance[0]['restart_sha256']==provenance[1]['restart_sha256']
summaries=[json.loads((d/'pilot-summary.json').read_text()) for d in runs]
final=[s['snapshots'][-1] for s in summaries]
assert np.isclose(final[0]['time'],final[1]['time'],rtol=0,atol=1e-12)
fields=[np.load(d/'final-snapshot.npz') for d in runs]
for key in ['R','Z','phi']:np.testing.assert_array_equal(fields[0][key],fields[1][key])
with Dataset(runs[0]/'limiter-12.nc') as f:wall=np.asarray(f['limiter'][:])>.5
interior=np.zeros(wall.shape,dtype=bool);interior[:,2:-2,2:-2]=True
w=np.broadcast_to(fields[0]['R'][None,None,:],wall.shape)
delta=np.linalg.norm(fields[1]['B_cyl']-fields[0]['B_cyl'],axis=-1)
checks={}
for label,mask in [('inside_wall',interior&wall),('outside_wall',interior&~wall),('all_interior',interior)]:
 checks[label]={'rms_delta_B_T':float(np.sqrt(np.average(delta[mask]**2,weights=w[mask]))),'max_delta_B_T':float(delta[mask].max())}
dp=fields[1]['P_mu0_Pa']-fields[0]['P_mu0_Pa'];checks['relative_pressure_L2_full_grid']=float(np.sqrt(np.sum(w*dp**2)/np.sum(w*fields[0]['P_mu0_Pa']**2)))
rows=[]
for d,snapshot in zip(runs,final):
 status=json.loads((r/'runs/job-records'/d.name/'status.json').read_text());assert status['state']=='finished' and status['returncode']==0
 rows.append({'run':d.name,'elapsed_seconds':status['elapsed_seconds'],'time':snapshot['time'],'pressure_current_A':snapshot['current_by_support']['positive_pressure']['mean_current_A'],'wall_current_A':snapshot['current_by_support']['inside_wall']['mean_current_A'],'force_ratio_squared_native_form':snapshot['force_by_support']['positive_pressure_interior']['ratio_to_grad_pressure_squared_plus_jxb_squared']})
report={'restart_sha256':provenance[0]['restart_sha256'],'comparison':f'Matched magnetic-time comparison: {runs[0].name} vs {runs[1].name}; inspect input/restart provenance for changed numerical controls','runs':rows,'differences':checks,'qualified_equilibrium':False,'limitation':'Coarse transient numerical-control sensitivity only; no order estimate, reference exterior-field agreement, wall/grid sensitivity or current convergence certificate.'}
out=r/'local-support'/a.output;out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
