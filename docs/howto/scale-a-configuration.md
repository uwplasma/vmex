# Scale a configuration to reactor size

`vmex --scale` applies the ideal-MHD dimensional similarity transform to an
input deck or wout file — by explicit `B_scale R_scale` factors, or to
ARIES-CS size when no factors are given. The default target,
`--scale-target volavgB`, is the ARIES-CS wout's own VMEC `volavgB = 5.8646 T`
and `Aminor_p = 1.7044 m`, the normalisation of Bader et al. (NF 2021),
Landreman, Buller & Drevlak (PoP 2022) and Paul et al. (NF 2022) and of the
reactor-scale wouts ESSOS and SIMSOPT ship, which therefore map to factors
of 1. `--scale-target axis` is the Landreman & Paul (PRL 2022) convention:
Boozer `B00 = 5.7 T` on the magnetic axis and `a = 1.7 m`. The two differ by
up to about 10 % in field on real configurations, so state the convention
with any loss fraction. `vmex --trace` applies the same scaling in memory. Use it for physical orbit studies where particle energy
and Larmor radius are fixed (e.g. 3.5 MeV alpha calculations).

## Scale a deck or a wout

```console
vmex --scale input.case            # target ARIES-CS volavgB and Aminor_p
vmex --scale input.case 1.2 0.8    # explicit B_scale=1.2, R_scale=0.8
vmex --scale wout_case.nc          # scale a finished equilibrium directly
```

Output names gain `_scaled`. A positive magnetic factor preserves the flux
direction. Then solve and plot the scaled deck as usual:

```console
vmex input.case_scaled --plot --booz
```

## What transforms how

Normalized flux, rotational transform, beta, aspect ratio, mode numbers, and
profile shapes do not change. Dimensional quantities scale as:

| quantity | factor |
|----------|--------|
| boundary, axis, major/minor radii | $R_s$ |
| volume, Jacobian Fourier coefficients | $R_s^3$ |
| magnetic field | $B_s$ |
| covariant field, total/coil currents | $B_s R_s$ |
| contravariant field | $B_s/R_s$ |
| toroidal and poloidal flux | $B_s R_s^2$ |
| pressure and $B^2$ | $B_s^2$ |
| magnetic/pressure energy | $B_s^2 R_s^3$ |
| $\mathbf{J}\cdot\mathbf{B}$ | $B_s^2/R_s$ |
| wout `DMerc`, `DShear`, `DWell`, `DCurr`, `DGeod` | $B_s^{-2}R_s^{-4}$ |
| `IonLarmor` | $B_s^{-1}$ |

For inputs, pressure scales once through `PRES_SCALE`; `AM`, `AM_AUX_F`,
current/iota profiles, and every normalized spline coordinate remain shape
data. `CURTOR` scales only in prescribed-current mode.

A wout records no `PRES_SCALE`, so `scale_wout` puts $B_s^2$ into the
coefficients it echoes: every `AM_AUX_F` knot value, and the entries of `AM`
that multiply the profile for its `PMASS_TYPE` (all of a power series, only
the leading coefficient of `two_power`, the numerator of `rational`), so the
profile a scaled wout describes evaluates to its `presf`. `AC`, `AI`, and the
knot positions stay shape data: iota is dimensionless, and every current
profile is normalized to `ctor`, which scales as $B_s R_s$.

## ARIES-CS targets from an input

A wout contains `volavgB`, the axis spectrum and `Aminor_p`, so its factors
are exact. A fixed boundary gives `Aminor_p` directly by Fourier quadrature,
but the field depends
on the converged internal field, so VMEX runs a bounded radial probe at
`ns <= 9` and `ns <= 17` (final probe `ftol = 1e-10`) rather than the full
ladder. The command prints both resolutions and the coarse-to-fine changes —
those changes are the declared target uncertainty.

## Free-boundary decks

The probe also determines the final minor radius, and the scaled input and
mgrid sidecar are written together. Per-ampere (`mgrid_mode = S`) field
tables scale as $R_s^{-1}$; raw tables scale as $B_s$, while their recorded
currents and `EXTCUR` scale as $B_s R_s$. Direct-coil inputs are rejected:
their geometry and currents must be scaled before field tabulation.

## The validation contract

The defining test is commutation: (1) solve the original input and scale its
wout; (2) scale the input (and mgrid when present) and reconverge it;
(3) compare every physical wout scalar, profile, Fourier coefficient,
Mercier term, and NESTOR potential/surface field. The pressure coefficients
are compared through the profile they evaluate to, because the input path
carries $B_s^2$ in `PRES_SCALE` and the wout path in `AM`. VMEX runs this
check for finite-pressure prescribed-current fixed boundary and for symmetric
and LASYM free-boundary NESTOR cases. The structured functions
{func}`vmex.core.scaling.scale_input`,
{func}`vmex.core.scaling.scale_mgrid`, and
{func}`vmex.core.scaling.scale_wout` use parsed objects; they never edit
namelist text by prefix.
