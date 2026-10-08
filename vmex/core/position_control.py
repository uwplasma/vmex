"""Position control for free-boundary solves.

A current-carrying plasma in the field of its coils is not guaranteed a stable
position.  For an axisymmetric vertical field the radial position is stable
only if the decay index ``n = -(R/B_Z) dB_Z/dR`` is below 3/2; coil sets that
shape a strongly sheared 3D equilibrium can exceed that and also bend the
plasma column helically, so the free-boundary iteration drifts and never
settles.  Tokamak equilibrium codes cure this with feedback on the vertical
field.  This module provides the same for VMEX as an opt-in: a small
curl-free correction field is added to the external field and its amplitudes
follow a proportional-integral-derivative law on the error between the
measured axis position and a target.

The correction is built from exact vacuum harmonics (scalar potentials), for
``k = n * nfp``::

    B_Z-type  (n = 0 .. nmax):  Phi = b_n Z (R/R0)^k cos(k phi)
    B_R-type  (n = 1 .. nmax):  Phi = a_n (R0/k) (R/R0)^k sin(k phi)

``n = 0`` is the uniform vertical field ``B_Z = b_0`` and is the only channel
active by default (``nmax = 0``).  The ``n >= 1`` channels act on the
helical bending of the axis, ``R_axis ~ cos(k phi)`` (via ``b_n``) and
``Z_axis ~ sin(k phi)`` (via ``a_n``), and respect stellarator symmetry.  The
converged amplitudes are returned in ``result.position_control`` so the user
can judge how much correction the coils needed; a good coil set gives small
values.  Nothing changes unless a :class:`PositionControl` is passed.

Sign convention: a plasma current ``I > 0`` along ``+phi`` is pushed outward
by ``B_Z > 0`` and downward by ``B_R > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PositionControl:
    """Configuration of the position feedback.

    ``measure`` is ``"axis"`` (the magnetic-axis Fourier coefficients) or
    ``"boundary"`` (those of the edge).  The controlled quantities are the
    mean radius ``R_00`` and, for ``nmax >= 1``, the helical coefficients
    ``R_0n`` and ``Z_0n``.  ``target`` is the mean radius in metres (``None``
    keeps the initial state's values, which is also the target of every
    ``n >= 1`` coefficient).  The gains are dimensionless multiples of
    ``g0 = mu0 |I| / (4 pi R0^2)`` [T/m], the field gradient scale of the
    plasma current ``I`` at the target radius ``R0`` (so one setting transfers
    between machines): ``gain`` multiplies the error, ``integral_gain`` its
    integral over iterations and ``derivative_gain`` its rate per iteration
    (the derivative damps oscillations; its time constant is
    ``derivative_gain / gain`` iterations), ``interval`` is the number of
    iterations between updates, ``max_field`` clips each amplitude [T] and
    ``max_step`` limits its change per update [T] (a power-supply slew rate;
    it also keeps one-off jumps of the measurement after a solver restart from
    kicking the plasma).  Inside ``deadband`` (metres, per coefficient) an amplitude
    is held, so a converged run is not disturbed by the feedback.  ``current_sign`` is the sign of the toroidal current
    (``None`` reads it from ``curtor``).
    """

    measure: str = "axis"
    target: float | None = None
    nmax: int = 0
    gain: float = 4.0
    integral_gain: float = 0.01
    derivative_gain: float = 200.0
    interval: int = 20
    max_field: float = 0.5
    max_step: float = 0.05
    deadband: float = 0.0
    current_sign: float | None = None

    def __post_init__(self) -> None:
        if self.measure not in ("axis", "boundary"):
            raise ValueError("measure must be 'axis' or 'boundary'")
        if int(self.interval) < 1:
            raise ValueError("interval must be a positive iteration count")
        if int(self.nmax) < 0:
            raise ValueError("nmax must be non-negative")


@dataclass(frozen=True)
class PositionControlResult:
    """State of the control, returned as ``result.position_control``.

    ``bz[n]`` are the ``B_Z``-type amplitudes [T] (``bz[0]`` is the uniform
    vertical field) and ``br[n - 1]`` the ``B_R``-type amplitudes; ``target``
    and ``measured`` list ``(R_00 .. R_0nmax, Z_01 .. Z_0nmax)`` in metres;
    ``history`` holds one row per update: ``iteration, fsqr + fsqz, bz..., br...,
    measured...``.
    """

    bz: np.ndarray
    br: np.ndarray
    target: np.ndarray
    measured: np.ndarray
    integral: np.ndarray
    error: np.ndarray
    r0: float
    history: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))

    @property
    def vertical_field(self) -> float:
        """The uniform vertical field ``B_Z^ctrl`` [T]."""
        return float(self.bz[0])


class ControlledField:
    """External field plus the control harmonics (a JAX pytree).

    ``ctl_bz``, ``ctl_br``, ``ctl_r0`` and ``stop`` are traced leaves (named so they
    never shadow the tabulated field's own ``br``/``bz`` attributes), so updating them
    never recompiles: ``stop`` makes the steady-state vacuum loop exit at that
    iteration so the host can update the amplitudes.
    """

    def __init__(self, base: Any, bz: Any, br: Any, r0: Any, stop: Any, nfp: int):
        self.base, self.ctl_bz, self.ctl_br, self.ctl_r0 = base, bz, br, r0
        self.stop, self.nfp_ctrl = stop, nfp

    def b_cyl(self, r: Any, phi: Any, z: Any):
        """``(B_r, B_phi, B_z)`` of the base field plus the control harmonics."""
        b_r, b_p, b_z = self.base.b_cyl(r, phi, z)
        x = r / self.ctl_r0
        for n in range(self.ctl_bz.shape[0]):
            k = n * self.nfp_ctrl
            c, s = jnp.cos(k * phi), jnp.sin(k * phi)
            b_z = b_z + self.ctl_bz[n] * x**k * c
            if k:
                b_r = b_r + self.ctl_bz[n] * z * k * x ** (k - 1) / self.ctl_r0 * c
                b_p = b_p - self.ctl_bz[n] * z * k * x**k / r * s
        for n in range(1, self.ctl_br.shape[0] + 1):
            k = n * self.nfp_ctrl
            c, s = jnp.cos(k * phi), jnp.sin(k * phi)
            b_r = b_r + self.ctl_br[n - 1] * x ** (k - 1) * s
            b_p = b_p + self.ctl_br[n - 1] * x ** (k - 1) * c
        return b_r, b_p, b_z

    __call__ = b_cyl

    def __getattr__(self, name: str) -> Any:  # tabulated-field attributes (br, extcur, ...)
        if name in ("base", "ctl_bz", "ctl_br", "ctl_r0", "stop", "nfp_ctrl") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)


jax.tree_util.register_pytree_node(
    ControlledField,
    lambda f: ((f.base, f.ctl_bz, f.ctl_br, f.ctl_r0, f.stop), f.nfp_ctrl),
    lambda nfp, c: ControlledField(*c, nfp),
)


def wrap_field(base: Any, ctl: PositionControlResult, stop: int = 2**30) -> ControlledField:
    """Wrap ``base`` (or re-wrap a :class:`ControlledField`) with the control state."""
    if isinstance(base, ControlledField):
        base = base.base
    dt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    return ControlledField(base, jnp.asarray(ctl.bz, dtype=dt), jnp.asarray(ctl.br, dtype=dt),
                           jnp.asarray(ctl.r0, dtype=dt), jnp.asarray(stop, dtype=jnp.int32),
                           int(base.nfp))


def measure_position(state: Any, modes: Any, cfg: PositionControl) -> np.ndarray:
    """``(R_00 .. R_0nmax, Z_01 .. Z_0nmax)`` of the axis (row 0) or edge (last row)."""
    m, n = np.asarray(modes.m), np.asarray(modes.n)
    row = 0 if cfg.measure == "axis" else -1
    R, Z = np.asarray(state.R_cos)[row], np.asarray(state.Z_sin)[row]
    idx = [int(np.flatnonzero((m == 0) & (n == j))[0]) for j in range(int(cfg.nmax) + 1)]
    return np.concatenate([R[idx], Z[idx[1:]]])


def initial_control(cfg: PositionControl, state: Any, modes: Any) -> PositionControlResult:
    """Zero amplitudes with the targets read from the initial state."""
    nmax = int(cfg.nmax)
    if nmax > int(np.max(np.asarray(modes.n))):
        raise ValueError(f"nmax = {nmax} exceeds the deck's NTOR = {int(np.max(np.asarray(modes.n)))}")
    measured = measure_position(state, modes, cfg)
    target = measured.copy()
    if cfg.target is not None:
        target[0] = float(cfg.target)
    return PositionControlResult(
        bz=np.zeros(nmax + 1), br=np.zeros(nmax), target=target, measured=measured,
        integral=np.zeros_like(measured), error=np.zeros_like(measured), r0=float(target[0]))


def update_control(cfg: PositionControl, ctl: PositionControlResult, measured: np.ndarray,
                   iteration: int, sign: float, fsq: float = 0.0,
                   scale: float = 1.0) -> PositionControlResult:
    """One PID update of every amplitude from the measured axis coefficients.

    ``scale`` is the gain unit ``g0`` [T/m] (see :class:`PositionControl`)."""
    nmax = int(cfg.nmax)
    first = ctl.history.shape[0] == 0
    dt = float(cfg.interval) if first else max(1.0, float(iteration) - float(ctl.history[-1, 0]))
    err = np.asarray(measured) - ctl.target
    integral = ctl.integral + err * dt
    deriv = np.zeros_like(err) if first else (err - ctl.error) / dt
    pid = scale * (cfg.gain * err + cfg.integral_gain * integral + cfg.derivative_gain * deriv)
    held = np.abs(err) < cfg.deadband
    new = np.concatenate([-sign * pid[: nmax + 1], -sign * pid[nmax + 1:]])
    old = np.concatenate([ctl.bz, ctl.br])
    new = np.clip(np.clip(new, old - cfg.max_step, old + cfg.max_step), -cfg.max_field, cfg.max_field)
    new = np.where(held, old, new)
    integral = np.where(held, ctl.integral, integral)
    row = np.concatenate([[iteration, fsq], new, np.asarray(measured)])
    hist = row[None] if first else np.vstack([ctl.history, row])
    return replace(ctl, bz=new[: nmax + 1], br=new[nmax + 1:], measured=np.asarray(measured),
                   integral=integral, error=err, history=hist)


def gain_scale(curtor: float, r0: float) -> float:
    """``g0 = mu0 |I| / (4 pi R0^2)`` [T/m]; ``0.1`` for a current-free deck."""
    g0 = 1.0e-7 * abs(float(curtor)) / float(r0) ** 2
    return g0 if g0 > 0.0 else 0.1
