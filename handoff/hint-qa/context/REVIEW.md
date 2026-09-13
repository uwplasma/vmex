> Archived context, 13 September 2026. The top-level handoff gives current status; older job statements below are historical. Machine locations were removed; unbundled raw data are not implied to be available.

# Maintained-branch findings and evidence

Baseline: `current` commit bf31fc39f7179bdd91d84319c51c68e2f6fff25f. Review in progress, 12 September 2026. This is not an exhaustive audit. Earlier master-only findings remain separate in the revised plan.

| Finding | Evidence and correction |
|---|---|
| HMAG callback declared as a real function but invoked as a subroutine | gfortran rejects `libnr.f90`; original failure log saved. Removed the incorrect real declaration of external `funcs`. All seven tools subsequently compile in both configurations. |
| HINT pressure-to-flux interpolation uses a dimensional pressure difference as its weight | Actual `cal_flux` fails the independent pressure-rescaling regression, reference error 0.23952. Normalizing by bin width and clamping endpoints gives reference error 1.13e-14 and pressure-rescaling error 1.44e-14. This reproduces the issue on maintained code. |
| Undefined default logical controls and initial pressure reference | Explicitly initialized `lpedge/lpfix/lpcut` and `p0`, which is broadcast after allocation. Static correctness fixes; no isolated runtime failure is claimed. |
| HMAG acquires a write lock while reading equilibrium data | Concurrent HMAG/GPRTS access reproduced HDF5 lock failure, followed by a bounds error because the read error is unchecked. The reader performs no writes; changed its normal HDF5 open to read-only. Missing-file/error propagation still needs broader review. |
| GPRTS divides by zero before applying its diagnostic mask | Bounds/FPE-enabled native LHD postprocessing crashed. Ratios now evaluate only where the original pressure-gradient mask allows and denominators are positive; the same native case completes afterward. The historical small-gradient masking policy is retained, not endorsed as a global force-convergence criterion. Independent physical residuals remain required. |
| HMAG periodic interpolation requests an out-of-range quadratic stencil | Native profile tracing with `mr=4`, `mcirc=10` reproduced index 103 into `thetak(-1:102)` at `cal_flxqnt.f90:428`. The duplicate 2π endpoint is now copied from zero without interpolation, and the search stays within valid ghost stencils. The same debug case then completes with profile and eight Poincaré outputs. |

Native LHD smoke tests use the supplied vacuum WOUT, mgrid and vessel. The one/two-rank and pressure-preserving restart comparisons exercise HDF5 solver input/output. They do not establish correctness of every HDF5 error path or of physical convergence.

Outstanding high-impact checks: source-wide HDF5 handle/error handling; pressure reset and mid-ramp restart semantics; current-profile and current-unit conventions; new scheduled tracing controls and broadcasts; finite-pressure edge sensitivity to grid/optimizer; small-gradient diagnostic masking; zero-pressure divisions; independent force/divergence validation; native QA flux interpolation and numerical wall. The maintained axis and pressure-tracing code already supersedes some master defects; those old patches were not transplanted.

## September 12 numerical-diagnostic follow-up

The native force/current diagnostic subtracts the sampled vacuum-field curl (`GPRTS/cal_check.f90` uses `jvec1-jvec0`). A coarse total-field finite-difference residual therefore does not reproduce it: it also measures vacuum-grid differentiation error. Independent fourth-order response-field differentiation agrees closely with the completed LHD run's force ratio; doubling the native vacuum grid reduces its curl/divergence errors about sixteenfold. This is a convention/refinement finding, not a newly patched solver defect.

The Python on-surface virtual-casing path can allocate a large automatic quadrature setup before target chunk limits take effect. A GPU run reproduced a 4.25 GiB allocation failure; explicit quadrature and a 16×16 target grid completed. This is a resource limitation of the tested setup, not evidence of a HINT solver failure. The pinned CPU/GPU field checks agree at rounding level. The supplied 2.5% coil/WOUT boundary mismatch also reproduces across backends and quadrature refinements; source hashes match. Do not attribute it to HINT or silently reverse coil currents.


## Continued QA audit and fixes

The power-series/linear pressure diagnostic printed an uninitialized `fpeak`.
It now prints `N/A` for branches that do not compute that quantity, retaining
the parabolic diagnostic. The local debug regression has identical pressure
arrays and magnetic differences below 5.4e-15 T. See
`pressure-diagnostic-regression.json`.

A duplicated `k+2` in the fourth-order toroidal velocity-diffusion stencil is
corrected to `k+1`. The active build does not define VISCOS2ND. An analytic
Fourier regression extracted from the source gives orders 3.985, 3.996 and
3.999 after correction; the upstream error grows under refinement. Debug Mac
and optimized remote validation host smokes with nonzero `nu1=0.001` remain finite. The QA
pilot uses the default `nu1=0`, so this defect does not explain its slow current
relaxation. See `toroidal-viscosity-stencil.json` and the viscosity smoke records.
The installed native version is now `current-bf31fc3-local3`, with earlier
versions retained.

The maintained current drive is `C profile(s) Btotal`, without local B² division.
Normalizing that drive to source `ctor` is not proof of attained total current
or exact absolute source <J.B>. Actual current is curl-derived and must be
checked across pressure and vacuum regions. The source profile evolves through
pressure-reconstructed flux labels. The initial 8-step pilot's pressure-mask
current is -2289.087 A; its wall integral is +22.655 A due to pressure-free
return current. The independent pressure-mask diagnostic matches native HINT.

The limiter terminates pressure tracing but does not clamp Bresponse. With
`eta1=0`, active resistivity is uniform `eta0`. `eta_diff` is inactive without
BDIFF2ND, which these builds do not define. Outer box boundaries fix normal
response increments and extrapolate tangential increments; limiter clearance
and outer-box magnetic sensitivity are separate acceptance gates.

Companion Solvax nonfinite-root fixes are proposed in PR104. VMEX now has a
worktree implementation of actual-primal admission and matching refined-state
materialization, with focused CPU evidence. Broader qualification and history
independence remain separate. See `CONTINUATION_PLAN.md` and the new primary
source synthesis `LITERATURE_GAP_REVIEW.md`.
