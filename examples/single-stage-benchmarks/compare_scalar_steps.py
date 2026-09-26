#!/usr/bin/env python
"""Plot two accepted_steps.csv files on the same accepted-iteration axis."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('control', type=Path)
parser.add_argument('free', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
cache = args.output.resolve()/'cache'; cache.mkdir()
os.environ.update(MPLCONFIGDIR=str(cache), MPLBACKEND='Agg')
import matplotlib.pyplot as plt

def read(path):
    with path.open() as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]

control, free = read(args.control), read(args.free)
radius_key = 'major_radius_m' if all('major_radius_m' in rows[0] for rows in (control,free)) else 'rbc00_m'
last = min(int(control[-1]['step']), int(free[-1]['step']))
control, free = control[:last+1], free[:last+1]
assert [row['step'] for row in control] == [row['step'] for row in free] == list(range(last+1))
fields = ('step','qa','objective','coil_step_u_l2','coil_displacement_max_m',
          'normal_field_rms','min_abs_iota','aspect_error',radius_key)
with (args.output/'comparison.csv').open('w', newline='') as stream:
    names = ['step']+[f'{arm}_{field}' for field in fields[1:] for arm in ('control','free')]
    writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader()
    for a,b in zip(control,free):
        row = {'step':int(a['step'])}
        row.update({f'{arm}_{key}':values[key] for key in fields[1:] for arm,values in [('control',a),('free',b)]})
        writer.writerow(row)

fig, axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
panels = [('qa','Raw QA = sum of squared QS residuals',1,False),
          ('coil_step_u_l2','Accepted coil increment ||Δu_coil||₂',1,True),
          ('coil_displacement_max_m','Maximum coil displacement per step [mm]',1000,True),
          ('normal_field_rms','Area-weighted RMS B·n / |B|',1,False),
          ('min_abs_iota','Minimum |iota|',1,False),
          (radius_key,'Physical major radius [m]' if radius_key=='major_radius_m' else 'RBC(0,0) [m]',1,False)]
for ax,(key,label,scale,log) in zip(axes.flat,panels):
    for data,name,color in [(control,'Fixed control','#2166ac'),(free,'Free boundary','#d95f02')]:
        data=data[1:] if log else data
        ax.plot([r['step'] for r in data],[scale*r[key] for r in data],'.-',label=name,color=color)
    if log:
        ax.set_yscale('log')
    if key == 'min_abs_iota':
        ax.axhline(.19, color='black', ls='--', lw=1, label='Floor 0.19')
    if key=='major_radius_m':
        ax.axhspan(.99,1.01,color='green',alpha=.08)
        ax.axhline(.99,color='black',ls='--',lw=1); ax.axhline(1.01,color='black',ls='--',lw=1)
    ax.set(xlabel='Accepted iteration',title=label)
    ax.grid(alpha=.25)
axes[0,0].legend(); axes[1,1].legend()
fig.suptitle('Scalar single-stage comparison at equal accepted iteration')
fig.savefig(args.output/'comparison.png',dpi=180)
plt.close(fig)
summary = dict(accepted_steps=last,control=control[-1],free=free[-1],
    sources={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.control,args.free)},
    control_initial=control[0],free_initial=free[0],
    maximum_coil_step_mm={name:1000*max(r['coil_displacement_max_m'] for r in data)
                         for name,data in [('control',control),('free',free)]},
    caveat='The control has independent boundary and coil variables; the free equilibrium determines its boundary from coils. See run provenance for active constraints.')
(args.output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
