> Archived context, 13 September 2026. The top-level handoff gives current status; older job statements below are historical. Machine locations were removed; unbundled raw data are not implied to be available.

# QA input contract and remaining validation

The outer-boundary rmnc/zmns coefficients are exactly identical between the supplied WOUTs; interior geometry and the pressure/current/coil constraints differ. The original WOUT/input/coil files under `QA_INPUTS/{beta0p5,beta2p5}/inputs` remain the reference. A two-outer-step local debug smoke now exercises the 0.5% native input; it is not a relaxed equilibrium.

| Quantity | 0.5% label | 2.5% label |
|---|---:|---:|
| Source volume beta | 0.5038845418% | 2.5469063197% |
| Axis pressure | 144195.89706 Pa | 720979.4853 Pa |
| Reference field used for normalization | 5.2315858878 T | 5.1352849085 T |
| HINT input axis beta fraction | 0.01324116031 | 0.06871217253 |
| Source toroidal current | −95778.349865 A | −272398.496522 A |
| Axis seed R at phi=0 | 1.1785170057 m | 1.1881566187 m |

Pressure conversion is `P_HDF5 = mu0 * p_Pa`. The HINT input beta is `2*mu0*p_axis/B_reference²`; substituting 0.005 or 0.025 directly would change the source pressure. Report actual relaxed pressure and the chosen volume-beta definition afterward.

Use native MKFLX for initial normalized toroidal flux. All three input grids must share R/Z coordinates, toroidal sampling and nfp=2. HDF5/NetCDF arrays read through Python are ordered `(phi,Z,R)` and cylindrical vectors `(B_R,B_phi,B_Z)`; Cartesian targets are in meters and fields in tesla.

HINT's imposed parallel current is proportional to `lambda(s) B`. The existing provisional shape uses `<J·B>/<B²>` from WOUT, with a uniform s table and normalization to the input toroidal current. Do not substitute `dI/ds` for this shape. In the maintained linear branch, `cal_volume` integrates the shape times toroidal field on the selected poloidal cut, and `cal_netj` normalizes with `mu0*inet0/(Rmaj0*B0)`. This establishes the code's normalization contract, not bootstrap self-consistency after topology changes. Verify discrete integrated current on all cuts and the relaxed current profile before qualifying either case.

There is no physical QA vessel in the supplied data. Numerical walls have now been constructed with 0.10, 0.12 and 0.15 m outward **poloidal R/Z buffers**, not constant three-dimensional normal offsets. Each has 128 toroidal sections per field period and 256 poloidal vertices. All 192 fixed exterior targets and every initial plasma grid cell lie inside all three walls. Native MKFLX/MKLIM preprocessing completed on the common 64×64×32 grid for both cases.

The complete Fourier filament curves were checked using derivative bounds between 2048 coil samples and conservative bounds across interpolated wall patches. The minimum clearance lower bounds over both cases are 0.1417, 0.1207 and 0.0893 m for the three buffers. These bounds concern zero-thickness filaments and the numerical wall, not a physical vessel or conductor radius. See `runs/qa-native-inputs/wall-validation.json` and `prepare_qa_wall.py`. Initial containment does not certify containment of relaxed plasma-current support.

Both 64×64×32 and 128×128×64 prescribed coil grids were generated on remote validation host GPU. At 192 common targets, native MAGVAL spline evaluation on the coarse grid has RMS errors of 0.0002346 T and 0.0004874 T, maximum errors 0.001692 T and 0.004142 T, against direct coil evaluation. The native 128×128×64 spline check is now complete: RMS errors are 4.577e-6 T and 9.248e-6 T, with maxima 3.281e-5 T and 7.275e-5 T. The RMS reductions are about 51× and 53× at the same 192 targets. Two levels do not establish an asymptotic order or an error bound everywhere. The remote validation host coarse results reproduce the local debug values to rounding. See `native-vacuum-refinement.json`; keep the independent trilinear check separate.

The discrete initial imposed parallel-current component varies across toroidal cuts by at most 0.2833% and 1.1166% relative to its phi=0 normalization. The normalization itself is algebraic. This does not validate total relaxed current or bootstrap self-consistency. See `qa-native-input-validation.json`.

The 0.5% debug smoke completed in 11.2 seconds with all saved field, velocity and pressure arrays finite. Its pressure ramp reaches a grid maximum of 144.493 kPa against the 144.196 kPa axis target (a grid maximum is not an axis interpolation). The two magnetic updates per outer step leave current far below target; no force or current convergence is claimed. See `qa-debug-smoke.json`. The eight-step pilot input is prepared but has not been run.

Before a QA relaxation: demonstrate vacuum-grid interpolation/refinement, native flux-map consistency, current normalization, pressure conversion, wall clearance and resource limits. Start with the 0.5% case. A normal-field coil-fit residual is part of the given-boundary application and must not be silently removed by retuning the reference.

The bounded boundary audit finds about 0.1160% and 4.2327% area-weighted RMS Bn/B_reference for the original pairs, respectively. The latter is confirmed on CPU and GPU. Do not describe the 2.5% supplied coils as an exact or demonstrated close match to the fixed WOUT. A diagnostic global current reversal reduces Bn but reverses the dominant field; it is not an authorized input correction.


## Subsequent pilot and source audit

The eight-step, 100-magnetic-step-per-trace remote validation host pilot completed in 184.4 s.
All arrays are finite and pressure remains inside the numerical wall. Native
and independent P>0 current integrals agree at -2289.087 A, far from the
-95778.350 A source target. Whole-wall current is +22.655 A, exposing return
current in pressure-free cells; it must not be substituted for the native
pressure-mask diagnostic. The native-form force-ratio-squared is 0.478804.

The 128-grid initial current-cut variation is 0.3639% and 0.8900% for the two
cases, versus 0.2833% and 1.1166% at 64. This does not establish converged total
current. See `qa-current-refinement.json`.

Maintained code imposes a resistive `C lambda(s) Btotal` drive. A provisional
flux-integral calculation from the unnormalized mapped profile gives about
-95.339 kA and -268.229 kA, versus total WOUT currents -95.778 and -272.399 kA.
These are finite-mesh diagnostics, not correction factors. Preserve the source
fixtures and require attained curl current and reconstructed <J.B> profiles
before claiming matched constraints.

The active build's eta_diff parameter does not apply unless BDIFF2ND is
compiled. eta1=0 makes eta0 uniform even outside pressure support. The limiter
and the outer-box magnetic boundary are distinct: pressure tracing terminates
at the limiter, whereas the box fixes normal response increments. Both require
sensitivity checks. A pressure-preserving follow-up with 1000 magnetic steps
per trace completed in 390.05 s. At magnetic time 0.88, pressure-supported
current is -15.905 kA, whole-wall current -0.774 kA, and the native-form force
ratio squared is 0.016635 (previous snapshot 0.014777). This is a bounded
relaxation-efficiency diagnostic, not a settled equilibrium or matched-current
certificate. Step A and Step B consumed 183.44 s and 204.71 s, respectively.
See `runs/qa-follow-beta0p5-001/pilot-summary.json`.

## Matched-time cadence diagnostic

At common magnetic time 1.08 from an identical t=0.88 restart, 2×1000 steps
took 90.70 s versus 129.74 s for 4×500. These are single timings
on a shared host, not a timing distribution. The pressure-supported currents are
-17.6924 and -17.6797 kA (12.76 A difference); native-form force ratios squared
are 0.0140061 and 0.0141422. Full-grid pressure L2 difference is 0.3341%.

The field difference outside the numerical wall is 0.721 µT RMS, 9.291 µT max.
At the 192 fixed given-boundary comparison targets, using native MAGVAL
interpolation, it is 19.520 µT RMS, 61.484 µT max. These targets lie inside the
numerical wall; outside-wall agreement cannot substitute for their accuracy.
The faster cadence is useful for exploratory relaxation but is not qualified
for final comparisons. A 250-step cadence from the same restart is the next
controlled temporal refinement; assess the target-field differences before
choosing a production interval. Current/time/grid/wall gates remain unmet.
See `qa-cadence-comparison.json`, the two run summaries, native target CSVs
and restart SHA-256 b801cd321fce9473efd668143409f06efdc7166003fcd1a47e7b8ebb3a38ff62.

## Current-drive support audit

For the present pedge=0.01 and jcuts=1 settings, initial MKFLX exterior labels
are s=1 (`MKFLX/flxout.f90:213`). Step A clears pressure and fills only
limiter-selected cells (`fline_method_mod.f90:303`). Before Step B, cal_s
rebuilds labels: pressure below paxis*pedge maps to psi(1)/psi(1)=1, assuming
valid normalization (`flux_mod.f90:184`). cal_netj first zeros its arrays, then
uses the profile only for strict s<jcuts (`stepb_mod.f90:2776`). Therefore the
nonzero profile endpoint does not inject drive into zero-pressure cells here.

This excludes one possible explanation for exterior current; it does not
certify a vacuum exterior. Curl-derived response current evolves through
J-Jnet in the resistive term (`stepb_mod.f90:1317`) and is not clamped to zero.

## Further cadence refinement

The 250-step run completed in 235.00 s at the same magnetic time 1.08. Relative
to 500 steps, pressure differs 0.5692% and target fields 69.393 µT RMS / 188.171 µT
maximum. Exterior-to-wall grid differences reach 6.119 mT. This is not a
convergent cadence sequence. The interaction between nstepb and the retained
nrefb=100 reference/smoothing schedule is under source review before another
refinement is selected. No production cadence has been certified.

The aligned 200-step run completed in 257.12 s. Relative to 500 steps, target fields differ 37.344 µT RMS / 92.824 µT maximum, pressure 0.8322%, and pressure-mask current 12.56 A. Outside-wall differences are 1.623 µT RMS / 17.720 µT maximum. These are coupled relaxation transients, not an established temporal convergence order or equilibrium error estimate. See QA_CADENCE_AUDIT.md and qa-cadence-500-200.json. The next prepared study varies eta0 to 0.002 / 0.005 against the original 0.001 reference at otherwise identical controls; qa-resistivity-sensitivity-plan.json records inputs and acceptance checks. It has not been launched.
