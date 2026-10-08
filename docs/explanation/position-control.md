# Position control: why a tokamak needs a vertical field

This page covers the physics behind `PositionControl` and the
`LPOSITION_CONTROL` input option: the vertical field a current-carrying
plasma needs, the radial and vertical stability window, the feedback law, the
vacuum-harmonic correction field, and what the reported control field means.
Usage is in {doc}`/howto/free-boundary`.

## The vertical field of a current-carrying plasma

A toroidal current `I` along the plasma ring is pushed outward by its own
hoop force and by the pressure of the plasma. A vertical field `B_V`
produces the opposite, inward force `I B_V` per unit length when
`B_V` points up for a current along `+phi`, so an equilibrium at major radius
`R` and minor radius `a` needs the Shafranov vertical field

$$
B_V = \frac{\mu_0 I}{4\pi R}\left[\ln\frac{8R}{a} + \beta_p + \frac{\ell_i}{2} - \frac{3}{2}\right],
$$

with the poloidal beta `beta_p` and the internal inductance `l_i`. For a
DIII-D-like plasma this is a few percent of the toroidal field. A free-boundary
solve takes the field from the coils and the current from `CURTOR`, so a coil
set whose vertical field is wrong has no equilibrium at the intended
position. Too little vertical field and the plasma slides inboard; too much and
it is pushed outboard. Nothing in the force-balance iteration restores it,
because the iteration only follows the force and the force changes sign with
the displacement in the unstable direction.

## The stability window

Write the vertical field near the plasma as `B_Z(R)` and define the decay
index

$$
n = -\frac{R}{B_Z}\frac{\partial B_Z}{\partial R}.
$$

For a ring current in a purely axisymmetric field the equilibrium is
vertically stable for `n > 0` and radially stable for `n < 3/2`, so a
position without feedback exists only for `0 < n < 3/2`. Outside that window
(strongly sheared current-driven plasmas driven by coils fitted to match a
field, for example, can reach `n` of 2 to 10) the iteration drifts until a
Jacobian reset and never settles. Tokamaks that need `n` outside the window,
or that need to track a moving target, close the loop with a fast
vertical-field coil. `PositionControl` is that loop for the iteration.

## The feedback law

Every `POSITION_INTERVAL` iterations the steady-state vacuum loop returns to
the host. The measured quantity is the phi-averaged magnetic-axis radius
`R_00` (the `(m, n) = (0, 0)` axis coefficient), with error
`e = R_00 - R_target`. The control field is

$$
B_Z^{\rm ctrl} = -\sigma\, g_0\left[k_p\, e + k_i \sum e\,\Delta t + k_d\,\frac{\Delta e}{\Delta t}\right],
\qquad g_0 = \frac{\mu_0 |I|}{4\pi R_0^2},
$$

where `sigma` is the sign of the plasma current, `Delta t` is the number of
iterations between updates and `g_0` is a field gradient in T/m. The gains are
dimensionless multiples of `g_0`, which lets one setting transfer between
machines. The deck gain `POSITION_GAIN` is `k_p`; the integral and derivative
gains keep the ratios `k_i = 0.0025 k_p` and `k_d = 50 k_p` (the derivative
damps the oscillation that a proportional law alone sustains). Each update
is also limited in slew rate and clipped in amplitude. The integral term
removes the steady-state error, so at convergence `R_00` sits on the target
and `B_Z^ctrl` is the field that holds it there.

## The correction field

The added field must be curl-free and divergence-free so that NESTOR's vacuum
solution stays consistent. The default is a uniform vertical field,
`B_Z = b_0`. With `POSITION_NMAX >= 1` the field also includes helical
vacuum harmonics, with `k = n N_fp`,

$$
\Phi = b_n\, Z \left(\frac{R}{R_0}\right)^{k}\cos k\phi
\quad\text{and}\quad
\Phi = a_n\,\frac{R_0}{k}\left(\frac{R}{R_0}\right)^{k}\sin k\phi ,
$$

which are exact solutions of Laplace's equation, `B = grad Phi`. They respect
`N_fp` periodicity and stellarator symmetry, and are driven by the axis
coefficients `R_0n` and `Z_0n`. The helical channels are experimental; they
reduced the residual of the Landreman sheared-iota equilibria driven by
ESSOS coils from 1.8e-4 to about 1e-7 but did not converge them.

## What the reported field means

`result.position_control.vertical_field` is the uniform `B_Z` that the
equilibrium needs on top of the supplied coil field to sit at the target
radius. A machine would supply it with its vertical-field coil, so it is the
vertical-field coil current expressed as a field: a coil set that needs a
large correction is not holding the intended equilibrium. In the DIII-D-like
example with a 40 mT deficit baked into the field, the converged
`B_Z^ctrl` is 39.8 mT, and with a 40 mT excess it is -39.8 mT.
