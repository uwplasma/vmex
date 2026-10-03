# Coil-constraint single-stage benchmarks

A vacuum rotating ellipse (1 m major radius) with three independent order-16
coils, optimized for QA, or for QH or QI with `COIL_CASE` (below). The case
and its limits are in `parameters.py`: minimum |iota| >= 0.41, aspect ratio in
[4.9, 5.1], major radius 1 +/- 0.01 m, coil length <= 5 m,
curvature <= 5 /m, mean squared curvature <= 5 /m^2, coil-coil distance
>= 0.15 m and coil-plasma distance >= 0.20 m.

- `free_boundary_single_stage_optimization.py` varies the coil shapes and one
  common current factor. Every trial is a free-boundary equilibrium of those
  coils, and SLSQP minimizes quasisymmetry with all the limits as hard
  inequalities, through `vmex.optimize.FreeBoundaryProblem`.
- `single_stage_optimization.py` is the fixed-boundary counterpart: SLSQP on
  the boundary and the coils, with the same hard limits and a B.n/|B| term in
  the objective.
- `postprocess.py` reads a finished run of either script and writes the loss
  and constraint histories, a GIF of the coils and LCFS, and a dense (NS201)
  free-boundary solve of the final coils with the `vmex.plot_wout` figures;
  `--trace` adds the `vmex --trace` alpha losses of that equilibrium at
  ARIES-CS size, and `--poincare 2000` a Poincare section of the final coils'
  field, seeded on the run's own flux surfaces and drawn over them.

In vacuum only the flux per ampere sets the plasma size. Holding both the coil
currents and PHIEDGE fixed pins it, and at this iota the aspect ratio then sits
at its 4.9 floor (QA ~0.02); a free current factor reaches aspect 5.1 (QA
~0.004). `SHARED_CURRENT = False` and `FLUX_TOLERANCE` restore the pinned case
in the free and fixed scripts.

`COIL_CASE=qa3`, `qh` and `qi` run the same comparison from rotating
ellipses at R = 1 m and B0 ~ 1 T built by `seed_input`: QA at nfp 3 and
aspect 6, QH (helicity (1, -1)) at nfp 4 and aspect 6, and QI at nfp 4 and
aspect 8, with aspect bands of +/- 0.1. The iota floors keep the vacuum field
off low-order rationals, where it breaks into islands that the nested-surface
equilibrium cannot see: min |iota| >= 0.41 (QA), >= 1.1 (QH, between iota = 1
and 8/7, from a b = 0.9 a_eff seed at iota 1.016) and >= 0.51 (QI).
QI minimizes the constructed QI residual of
`examples/optimization/QI_optimization.py` with the edge mirror ratio <= 0.21
as one more hard limit. Each has stage-two coils, `coils.<case>.json`: three
order-8 coils per half period, <= 3.5 m long, curvature <= 8 /m,
MSC <= 10 /m^2, 0.08 m apart and 0.15 m from the plasma, fitted from circles
to B.n = 0 on the seed (rms B.n/|B| 3.7e-3, 4.4e-3 and 5e-4) with currents
set to enclose PHIEDGE. They start up to 5 % over the length and curvature
limits, which SLSQP then enforces:

    COIL_CASE=qi python free_boundary_single_stage_optimization.py --steps 5 --output runs/free-qi
    COIL_CASE=qi python single_stage_optimization.py --steps 5 --output runs/fixed-qi
    COIL_CASE=qi python postprocess.py runs/free-qi

`input.rotating_ellipse` is the seed and `coils.initial.json` the coils fitted
to it. `_coil_constraints.py` evaluates the coil inequalities and the
independent endpoint checks. The scripts need ESSOS (`pip install "vmex[coils]"`)
and a GPU for the default resolution:

    python free_boundary_single_stage_optimization.py --steps 5 --output runs/free
    python single_stage_optimization.py --steps 5 --output runs/fixed
    python postprocess.py runs/free
