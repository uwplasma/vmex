> Archived context, 13 September 2026. The top-level handoff gives current status; older job statements below are historical. Machine locations were removed; unbundled raw data are not implied to be available.

# QA magnetic-step cadence audit

Source and input inspection, 2026-09-12. This audit launched no simulations. The
250/500/1000 runs share a restart and add 0.2 magnetic time, but changing
`nstepb` changes more than tracing cadence. The observed approximately 6.1 mT
maximum outside-wall difference is not causally attributed to any one mechanism
by this inspection.

## Velocity smoothing reference schedule

In `src/HINT/stepb_mod.f90`, `stepb` calls `refer` unconditionally at outer
entry (line 333), starts the local magnetic-step loop at 1 (346), smooths at
`MOD(i,nsmoothb)==0` (407), and refreshes references at
`MOD(i,nrefb)==0` (409). `refer` copies the current velocity and magnetic
perturbations into reference arrays (2192–2197).

The active smoother subtracts velocity references, filters those increments,
and adds the references back (1823–1844). The corresponding magnetic-field
operations are commented out (1826–1828, 1837–1839, 1845–1847). Its spatial
filter uses fixed coefficients (2126–2131). Velocity enters magnetic evolution
through `v x B` in `faraday` (1309–1311).

With the actual `nrefb=100, nsmoothb=1` inputs:

| Magnetic steps per outer call | Global reference refresh epochs after restart | Reference interval pattern |
| --- | --- | --- |
| 250 | 100, 200, 250, 350, 450, 500, ... | 100, 100, 50, repeating |
| 500 or 1000 | 100, 200, 300, 400, 500, ... | uniform 100 |
| proposed 200 | 100, 200, 300, 400, 500, ... | uniform 100 |

For aligned chunks, the unconditional entry refresh duplicates the preceding
end-of-chunk periodic refresh. The 250-step choice instead truncates and
restarts the reference schedule. This is a definite smoothing-reference
confound, but the inspection does not measure its contribution to field
differences.

## Physical state is preserved across outer calls

Although `preprocessing` zeroes scratch arrays, it reloads total magnetic field
minus the prescribed field (720–722) and evolved velocity (732–734).
`postprocessing` gathers the evolved magnetic perturbation and velocity,
restores total magnetic field, and broadcasts both (765–773). It does not
zero the evolved physical velocity or magnetic field at every outer call.

The RKG auxiliary accumulators are updated by `progress` (1800–1806); the
fourth stage has `c2=c3=0` (390–402), so they already reset after every complete
magnetic substep. Their preprocessing reset is not an additional loss of
multi-step integrator memory.

## Other coupled outer updates and actual deck checks

`src/HINT/evolve.f90` calls `stepa` and `stepb` on each outer iteration
(89–95). Field-line pressure redistribution overwrites `p1` in
`src/HINT/stepa_modules/fline_method_mod.f90` (300–310), then broadcasts it
(350). At each `stepb` entry, `cal_s` calls `cal_axis` and `cal_flux`
(`stepb_mod.f90`, 2468–2476). `src/HINT/flux_mod.f90` clears and reconstructs
flux from current pressure (128–136). Thus the input `flx_type='file'` does
not freeze the subsequent flux reconstruction. `preprocessing` loads this
flux into `ss` (`stepb_mod.f90`, 738), which the linear prescribed-current
profile consumes (2775–2782). Current normalization itself is recomputed at
each RKG-stage `cal_j` call through `cal_netj` and `cal_volume` (865, 2719).

The inspected `runs/qa-cadence-n250-001/hint.input` uses
`run_mode='follow', lresetp=.false., npset=0, npchg=4`, leaves `nvreset` at
its default zero, and sets `eta1=0`. These imply:

- No pressure ramp in this continuation: `src/HINT/cal_press.f90` sets
  `npchg=0, pfac=1` and returns for follow without pressure reset (143–145),
  despite the deck's retained `npchg=4` and initial-pilot comment.
- No periodic pressure or velocity reset: `evolve.f90` maps nonpositive
  reset periods beyond the outer iteration count (80–81), with reset calls
  guarded at 91–92. `nvreset=0` is the default in `module.f90` (38).
- Uniform resistivity `eta0`, from `stepb_mod.f90` (2816–2824).
- Zero background rotation defaults, `module.f90` (56–57); `cal_s` sets
  background velocity to zero and copies pressure in this case
  (`stepb_mod.f90`, 2480–2483). This is distinct from evolved velocity.

`local-support/prepare_qa_cadence.py` changes only `nstep` and `nstepb`, with
`nstep=2000/nstepb`, retaining `dt_b=1e-4` and the same restart. The resulting
experiment couples pressure redistribution, pressure-derived flux/current
profile refresh, and (for 250) the velocity-reference schedule. Its existing
"tracing cadence" description should not be interpreted as isolating tracing.

## Controlled next comparison and diagnostic caveat

Use `nstepb=200, nstep=10`, retaining `nrefb=100`, `nsmoothb=1`,
`dt_b=1e-4`, the identical restart, and other parameters. This preserves
global smoothing-reference epochs shared with 500/1000 while adding the same
0.2 magnetic time. Other outer pressure/current-profile refresh changes remain
the intended coupled experiment. Describe the result as **outer-update cadence
sensitivity with aligned smoothing references**, not trace-only convergence.

The input `nenergyb=100` also matters: energy output is conditional on local
step number and occurs before smoothing (`stepb_mod.f90`, 406–407). In a
250-step chunk, the last history sample is at step 200, leaving the final 50
steps unsampled. At the final chunk this is 0.005 magnetic time before the
final field. Use final field artifacts for matched-time field comparisons;
do not treat the final history row as the final evolved state.
