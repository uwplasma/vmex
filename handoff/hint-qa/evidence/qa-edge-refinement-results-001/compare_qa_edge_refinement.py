"""Compare matched-time fresh QA grid pilots through integrated diagnostics only.

Grid-index rectangles are deliberately not compared across grids. Distance shells
use fixed meter bins but retain a grid-dependent pressure mask and two-layer trim.
Run sample_qa_cadence_targets.py separately to add common physical target fields.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
p.add_argument('--reference-run', required=True)
p.add_argument('--candidate-run', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
reports = []
for name in (a.reference_run, a.candidate_run):
    run = a.root / 'runs' / name
    summary = json.loads((run / 'pilot-summary.json').read_text())
    shell = json.loads((run / 'return-current-diagnostic-001.json').read_text())
    reports.append(dict(run=name, grid=summary['grid_validation'],
        snapshots=summary['snapshots'], pressure_distance_shells=shell['pressure_distance_shells'],
        field_sha256=summary['source_sha256']))
times = [[row['time'] for row in report['snapshots']] for report in reports]
if len(times[0]) != len(times[1]) or any(abs(x-y)>1e-12 for x,y in zip(*times)):
    raise ValueError('Snapshot times differ; cannot claim matched-time comparison')
for key in ('rminb', 'rmaxb', 'zminb', 'zmaxb', 'mtor'):
    if abs(reports[0]['grid'][key]-reports[1]['grid'][key]) > 1e-12:
        raise ValueError('Physical domains or field periods differ')
result = dict(convergence_certified=False, matched_snapshot_times=times[0], runs=reports,
    interpretation='Matched fresh startup comparison; neither run is a mature equilibrium and two grids cannot establish asymptotic convergence.',
    limitations=['Pressure-distance bins have common meter bounds, but pressure-selected supports vary with grid and evolution.',
                 'Two excluded R/Z layers correspond to different physical margins; all-interior integrals have different supports.',
                 'Native target fields, when appended, share physical coordinates and sampler but are not certified exterior to relaxed current support.',
                 'Full current minus parallel drive contains required pressure-driven perpendicular structure and is not a zero target.'])
with a.output.open('x') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
print(a.output)
