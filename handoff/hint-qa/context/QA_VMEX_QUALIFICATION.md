> Archived context, 13 September 2026. The top-level handoff gives current status; older job statements below are historical. Machine locations were removed; unbundled raw data are not implied to be available.

# Finite-beta QA VMEX qualification

## Certified CPU root and cold repeat, 13 September 2026

The original 0.5% finite-beta QA input was solved twice in independent Python processes at ns31, MPOL7, NTOR6, NFP2, with current mode ncurr=1. VMEX source14360d179af4534aa9bb638b172c8724ee289d9b and the explicitly imported SOLVAX e4b5071 archive were held fixed. Full Python source, harness, input and reference hashes match between runs.

Both returned status0 with valid geometry and actual assembled projected residual norm4.4748087001405284e-11 below primal_tol1e-10. The states are bitwise identical. Solve plus certification took21.64 and26.57 seconds; these are two observations, not a statistical performance benchmark. Memo repeats also match exactly, but are not counted as independent solves.

The computed edge-extrapolated toroidal current is -95778.34986522896 A, equal to reference WOUT. Input CURTOR is -95673.86795711746 A; the104.481908 A difference is retained explicitly under the finite-mesh edge extrapolation convention. The pressure half-mesh relative maximum error is2.05256e-16. The reference beta0.005038845418346838 is read from WOUT; it is not a newly computed VMEX beta measurement.

The earlier host convergence history has FSQ1.52487e-11 and ratio1.52487. The actual-state certificate after refinement has FSQ1.56670e-19 and ratio1.56670e-8. Historical host metrics and the current-state certificate are intentionally reported separately.

This establishes one fixed-boundary CPU root and repeatability diagnostic. It does not establish mesh convergence, free-boundary agreement, exterior-field accuracy, derivative accuracy or an optimized design. GPU qualification and a live parameterized energy derivative diagnostic are being evaluated separately.

Evidence: `qa-root-cpu-repeat-14360d17.json`, `runs/qa-root-cpu-14360d17-001/qualification.json`, and `runs/qa-root-cpu-14360d17-002/qualification.json`. Each run contains the exact source manifest, state and constraint report.

## Independent GPU-environment root comparison

Remote validation host GPU0 completed one cold qualification using an immutable VMEX archive at the same 14360d179af4534aa9bb638b172c8724ee289d9b commit and explicitly imported SOLVAX e4b5071 archive. All 75 VMEX Python files, 23 SOLVAX Python files, actual imported SOLVAX source hashes, input, WOUT and helper hashes match CPU run001 exactly. The runner selected `CUDA_VISIBLE_DEVICES=0` and `JAX_PLATFORMS=cuda,cpu`; the helper reported `cuda:0`. GPU1 was not used by this task.

The GPU-environment result returned status0, valid geometry and assembled projected residual norm4.4749422624231e-11 below1e-10. Its coefficient-state relative L2 difference from CPU001 is4.179039223460897e-12; maximum absolute difference is3.020691335953174e-12. Computed toroidal current is -95778.34986522907 A, differing from reference WOUT by -1.1641532182693481e-10 A. Pressure relative maximum error is2.0525629619308096e-16. The memo repeat is exact. This is cross-device-environment root agreement at one resolution; no derivative or exterior-field accuracy follows from it.

Placement is supported by source inspection, with a measurement limitation. In `vmex/core/implicit.py`, `_host_solve_and_mask_status` enters the explicit configuration device context, and both the initial solve and fixed-point refinement execute within that context. `_host_solve` passes `device=AUTO`; `vmex/core/device.py:resolve_device` preserves the explicit JAX platform/default-device selection rather than forcibly selecting CPU. The method name “host solve” therefore describes Python orchestration, not proof of CPU-only numerical execution. GPU allocation was observed at608MiB, but no profiler trace or per-kernel placement record was collected. These artifacts should be described as GPU-environment qualification, not a demonstration that every equilibrium operation runs on GPU.

Solve plus certification took130.30 seconds and total bounded-run time136.99 seconds, versus CPU observations21.64 and26.57 seconds. Cold compilation, transfer and dispatch were not timed separately. Phase instrumentation is needed before interpreting this difference as an algorithmic regression or a meaningful performance comparison.

Evidence: `local-support/qa-root-cpu-gpu-14360d17-parity.json` and `runs/qa-root-gpu-14360d17-001/` (qualification manifest, state, report and runner status). No additional GPU jobs were launched after this qualification.

PR302 snapshot after this run: draft at14360d179af4534aa9bb638b172c8724ee289d9b, three checks successful (scope selection, documentation link check and two-device placement/AD),21 queued, no current-head failures reported. Broad CI and independent approval remain outstanding; this numerical record does not replace those gates.


## CPU live-solve energy derivative, 13 September 2026

At VMEX `14360d179af4534aa9bb638b172c8724ee289d9b` and the same SOLVAX
archive, the bounded CPU derivative harness completed in 147.54 seconds.
It uses the actual `solve_implicit(params, cfg)` custom VJP and recomputed
runtime, not the cached Equilibrium snapshot-field path. Only RBC(m=1,n=0)
varies: its derivative with respect to dimensionless q is R00 =
1.016712828943485 metres. Pressure coefficients, current mode, CURTOR and
PHIEDGE remain fixed. The objective is VMEX magnetic energy `wb` in its native
normalization; this is one energy direction, not an exterior-field objective.

The ordinary reverse directional derivative is -2.42065706831589. A separate
matrix-free forward linear solve produced -2.4206570683224817. The state
cotangent/state tangent and parameter direction/state pullback pairings agree
with absolute error 6.592e-12 and relative error 2.723e-12. Explicit metric
parameter dependence is accounted for separately (zero in this instance).
The independently recomputed tangent equation residual is 5.837e-10, below
its 5.864e-9 certificate bound. No full radial block matrix was assembled.

Eight perturbed live solves all passed actual-state primal certification,
including matching parameter and returned-state hashes. Their projected
residual norms range from 1.722e-12 to 4.536e-11, below 1e-10.

| q step | Maximum first-order Taylor remainder | Observed halving order | Central-difference relative derivative error |
| --- | --- | --- | --- |
| 1e-4 | 2.491e-7 | — | 4.215e-7 |
| 5e-5 | 6.317e-8 | 1.979 | 4.103e-6 |
| 2.5e-5 | 1.621e-8 | 1.963 | 5.522e-6 |
| 1.25e-5 | 4.539e-9 | 1.836 | 1.085e-5 |

The remainder is approximately second order over this range, with degradation
at the smallest step. Central-difference error does not decrease monotonically;
the increasing error at smaller steps is compatible with amplification of
solver/value noise, but this run does not isolate its cause. Do not claim
asymptotic finite-difference convergence or blanket derivative qualification.

Phase times were 22.96 seconds for the certified base, 41.53 seconds for the
ordinary reverse energy gradient, and 22.61 seconds for the matrix-free tangent
and duality calculation. Each perturbed primal plus value took 6.73–8.88
seconds. Compilation and execution within those phases are not separated.
These are single-run costs, not a statistical benchmark or a GPU speedup claim.

Evidence: `runs/qa-derivatives-cpu-14360d17-001/{report.json,provenance.json,output.log}`
and `runs/job-records/qa-derivatives-cpu-14360d17-001/status.json`. The manifest
records exact input, VMEX, SOLVAX and both helper hashes. The root helper's
subsequent status1-placeholder guard was added after this run; the recorded
CPU helper bytes remain available for an exactly matched GPU comparison.
No optimization run or additional derivative simulation follows from this
record. GPU derivative validation and other objectives remain outstanding.

## GPU derivative follow-up prepared, not launched

The exact CPU derivative harness and its original root-helper dependency are archived under `runs/qa-derivatives-source-14360d17/` and transferred to the same remote validation host directory name. Their hashes match the completed CPU derivative provenance. The later local root-helper status1 diagnostic change is deliberately excluded from this frozen pair. The existing immutable VMEX14360d17 archive, explicit SOLVAXe4b archive and original finite-beta0.5 input are reused.

A GPU0-only run is prepared with the unchanged step1e-4,600-second worker limit and630-second outer limit. Full source/input/helper hashes, executable argument vector, shell-safe SSH command and prelaunch checks are in `local-support/qa-derivatives-gpu-14360d17-preparation.json`. GPU0 was occupied by PID4183060 at95% utilization during preparation and remained occupied at97% on the final check at2026-09-13T00:32:11Z (963MiB total,940MiB process allocation). The run was not launched. No background polling or automation was created. A fresh process/occupancy check and the shared resource lock are required before a later launch.

This follow-up will compare the same CPU direction and objective, retaining partial phase diagnostics on timeout or failure. Neither CPU completion nor future CPU/GPU agreement alone qualifies all finite-beta QA derivatives. No GPU derivative performance claim exists while this run is pending.


## Synthetic energy calibration: bounded incomplete attempt

The host safeguarded-Newton example used the certified q=5e-5 energy as a
synthetic target, with bounds +/-1e-4 and an absolute energy tolerance of 1e-9.
It retained the same pressure, current and flux parameters. This is an inverse
calibration exercise, not a physical reactor optimization. The ordinary VMEX
custom VJP supplies the derivative; SOLVAX is used by VMEX's refinement and
adjoint. The nonlinear iteration is explicitly host-orchestrated, because
SOLVAX's existing nonlinear JVP wrappers cannot directly linearize the
custom-VJP-only equilibrium solve.

The first certified value/gradient took 147.37 seconds and gave the same
-2.42065706831589 derivative as the preceding directional check. One certified
Newton trial was accepted at q=4.997390217341553e-5. Its energy residual fell
from 1.2096967952740512e-4 to 6.310907169071456e-8, with primal residual
1.7239783603628866e-12. The optimizer worker then reached its 180-second cap
during the second value/gradient evaluation. The 1e-9 calibration tolerance
was not reached, and the independent fresh-process final root was not run.
There is no completed or independently verified calibration result.

A one-second native process sample during the first evaluation captured JAX
CompileAndLoad activity. It does not separate total compilation from execution
or establish why tracing the scalar closure cost more than the earlier full
parameter-gradient path. The next bounded attempt should first separate primal
and gradient phase timings, then reuse the already exercised full parameter
gradient contracted with the fixed boundary direction. An accepted candidate
whose certified value already meets the target should be checked before
computing an unnecessary next gradient. Do not raise the time limit or relax
the calibration/primal tolerances merely to report success.

Evidence: `runs/qa-calibration-cpu-14360d17-001/{optimization.json,provenance.json,optimize.log}`
and `runs/job-records/qa-calibration-cpu-14360d17-001/status.json` (exit124).
The reusable attempt is `local-support/calibrate_finite_beta_qa_energy.py`.
Its sibling `qualify_finite_beta_qa_root.py` supplies provenance utilities;
a portable handoff must include both, the target derivative report/provenance,
and the exact input/source hashes. GPU derivative work is separately pending
resource availability; this timeout triggered no additional simulation.
