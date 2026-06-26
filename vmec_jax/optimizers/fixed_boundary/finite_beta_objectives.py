"""Finite-beta, Mercier, Glasser, and profile objectives for optimization.

The classes in this module turn differentiable VMEC diagnostics into
least-squares terms: magnetic well, volume-averaged field, total beta, Mercier
``D_Merc``, Glasser resistive-interchange ``D_R``, bootstrap-current mismatch,
and profile-shape targets.  They are intentionally small adapters around the
physics kernels in ``finite_beta.py`` and ``mercier.py`` so user scripts can
assemble objectives by listing `(term, target, weight)` tuples.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..._compat import jnp
from ...field import b_cartesian_from_state
from ...finite_beta import (
    finite_beta_scalars_from_state,
    magnetic_well_from_vp,
    mercier_terms_from_state,
    redl_bootstrap_mismatch_from_state,
)
from ...mercier import glasser_resistive_interchange_from_mercier_terms
from .objective_terms import ObjectiveTerm
from .objective_terms import StageContext


def _smooth_positive_part(value, *, softness: float):
    value = jnp.asarray(value, dtype=jnp.float64)
    softness = float(softness)
    if softness <= 0.0:
        return jnp.maximum(value, 0.0)
    return softness * jnp.logaddexp(value / softness, 0.0)


def _target_is_zero(target) -> bool:
    arr = np.asarray(target, dtype=float)
    return bool(np.allclose(arr, 0.0))


class MagneticWell:
    """Smooth lower-bound objective for the VMEC magnetic-well proxy."""

    name = "magnetic_well"

    def __init__(self, *, minimum: float = 0.0, softness: float = 1.0e-3):
        """Evaluate this object for fixed-boundary VMEC solve and implicit differentiation."""
        self.minimum = float(minimum)
        self.softness = float(softness)

    def well(self, ctx: StageContext, state):
        """Evaluate well for fixed-boundary VMEC solve and implicit differentiation."""
        scalars = finite_beta_scalars_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
        )
        return magnetic_well_from_vp(scalars["vp"])

    def J(self, ctx: StageContext, state):
        """Evaluate the scalar objective contribution for the current VMEC state."""
        deficit = float(self.minimum) - self.well(ctx, state)
        return _smooth_positive_part(deficit, softness=self.softness)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        """Convert this user-facing objective into packed least-squares residuals."""
        if not _target_is_zero(target):
            raise ValueError("MagneticWell is a lower-bound penalty and requires target=0.")
        return ObjectiveTerm(self.name, self.J, target=0.0, weight=residual_weight)


class VolavgB:
    """Volume-averaged magnetic-field objective for finite-beta studies."""

    name = "volavgB"

    def J(self, ctx: StageContext, state):
        """Evaluate the scalar objective contribution for the current VMEC state."""
        return finite_beta_scalars_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
        )["volavgB"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        """Convert this user-facing objective into packed least-squares residuals."""
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class BetaTotal:
    """Total-beta objective for finite-beta studies."""

    name = "betatotal"

    def J(self, ctx: StageContext, state):
        """Evaluate the scalar objective contribution for the current VMEC state."""
        return finite_beta_scalars_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
        )["betatotal"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        """Convert this user-facing objective into packed least-squares residuals."""
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class DMerc:
    """Smooth lower-bound objective for VMEC Mercier stability."""

    name = "DMerc"

    def __init__(
        self,
        *,
        minimum: float = 0.0,
        softness: float = 1.0e-3,
        mmax_force: int | None = None,
        nmax_force: int | None = None,
    ):
        """Evaluate this object for fixed-boundary VMEC solve and implicit differentiation."""
        self.minimum = float(minimum)
        self.softness = float(softness)
        self.mmax_force = None if mmax_force is None else int(mmax_force)
        self.nmax_force = None if nmax_force is None else int(nmax_force)

    def terms(self, ctx: StageContext, state):
        """Evaluate terms for fixed-boundary VMEC solve and implicit differentiation."""
        return mercier_terms_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            mmax_force=self.mmax_force,
            nmax_force=self.nmax_force,
        )

    def J(self, ctx: StageContext, state):
        """Evaluate the scalar objective contribution for the current VMEC state."""
        dmerc = jnp.asarray(self.terms(ctx, state)["DMerc"], dtype=jnp.float64)
        active = dmerc[1:-1] if int(dmerc.shape[0]) > 2 else jnp.zeros((0,), dtype=dmerc.dtype)
        deficit = float(self.minimum) - active
        return _smooth_positive_part(deficit, softness=self.softness)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        """Convert this user-facing objective into packed least-squares residuals."""
        if not _target_is_zero(target):
            raise ValueError("DMerc is a lower-bound penalty and requires target=0.")
        return ObjectiveTerm(self.name, self.J, target=0.0, weight=residual_weight)


class GlasserResistiveInterchange:
    """Smooth upper-bound objective for the Glasser resistive criterion."""

    name = "D_R"

    def __init__(
        self,
        *,
        maximum: float = 0.0,
        softness: float = 1.0e-3,
        shear_epsilon: float = 0.0,
        mmax_force: int | None = None,
        nmax_force: int | None = None,
    ):
        """Evaluate this object for fixed-boundary VMEC solve and implicit differentiation."""
        self.maximum = float(maximum)
        self.softness = float(softness)
        self.shear_epsilon = float(shear_epsilon)
        self.mmax_force = None if mmax_force is None else int(mmax_force)
        self.nmax_force = None if nmax_force is None else int(nmax_force)

    def terms(self, ctx: StageContext, state):
        """Evaluate terms for fixed-boundary VMEC solve and implicit differentiation."""
        terms = mercier_terms_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            mmax_force=self.mmax_force,
            nmax_force=self.nmax_force,
        )
        if self.shear_epsilon == 0.0:
            return terms
        return {
            **terms,
            **glasser_resistive_interchange_from_mercier_terms(
                DMerc=terms["DMerc"],
                shear=terms["shear"],
                H=terms["H"],
                shear_epsilon=self.shear_epsilon,
            ),
        }

    def J(self, ctx: StageContext, state):
        """Evaluate the scalar objective contribution for the current VMEC state."""
        d_r = jnp.asarray(self.terms(ctx, state)["D_R"], dtype=jnp.float64)
        active = d_r[1:-1] if int(d_r.shape[0]) > 2 else jnp.zeros((0,), dtype=d_r.dtype)
        excess = active - float(self.maximum)
        return _smooth_positive_part(excess, softness=self.softness)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        """Convert this user-facing objective into packed least-squares residuals."""
        if not _target_is_zero(target):
            raise ValueError("GlasserResistiveInterchange is an upper-bound penalty and requires target=0.")
        return ObjectiveTerm(self.name, self.J, target=0.0, weight=residual_weight)


class _MercierProfileObjective:
    """Base object for differentiable VMEC JXBFORCE profile objectives."""

    name = "mercier_profile"
    profile_key = ""

    def __init__(
        self,
        *,
        surfaces: Sequence[float] | None = None,
        normalize: float = 1.0,
        mmax_force: int | None = None,
        nmax_force: int | None = None,
    ):
        """Evaluate this object for fixed-boundary VMEC solve and implicit differentiation."""
        self.surfaces = None if surfaces is None else tuple(float(s) for s in surfaces)
        self.normalize = float(normalize)
        self.mmax_force = None if mmax_force is None else int(mmax_force)
        self.nmax_force = None if nmax_force is None else int(nmax_force)

    def terms(self, ctx: StageContext, state):
        """Evaluate terms for fixed-boundary VMEC solve and implicit differentiation."""
        return mercier_terms_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            mmax_force=self.mmax_force,
            nmax_force=self.nmax_force,
        )

    def _select_profile(self, ctx: StageContext, profile):
        profile = jnp.asarray(profile, dtype=jnp.float64)
        if self.surfaces is None:
            return profile[1:-1] if int(profile.shape[0]) > 2 else jnp.zeros((0,), dtype=profile.dtype)
        s = np.asarray(getattr(ctx.static, "s"), dtype=float)
        indices = [int(np.argmin(np.abs(s - float(surface)))) for surface in self.surfaces]
        return profile[jnp.asarray(indices, dtype=jnp.int32)]

    def J(self, ctx: StageContext, state):
        """Evaluate the scalar objective contribution for the current VMEC state."""
        profile = self.terms(ctx, state)[self.profile_key]
        values = self._select_profile(ctx, profile)
        return values / float(self.normalize)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        """Convert this user-facing objective into packed least-squares residuals."""
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class JDotB(_MercierProfileObjective):
    """VMEC ``jdotb`` profile objective from the differentiable JXBFORCE path."""

    name = "jdotb"
    profile_key = "jdotb"


class BDotB(_MercierProfileObjective):
    """VMEC ``bdotb`` profile objective from the differentiable JXBFORCE path."""

    name = "bdotb"
    profile_key = "bdotb"


class BDotGradV(_MercierProfileObjective):
    """VMEC ``bdotgradv`` profile objective from the differentiable JXBFORCE path."""

    name = "bdotgradv"
    profile_key = "bdotgradv"


class BVector:
    """Cartesian magnetic-field vector objective on one radial surface."""

    name = "B_vector"

    def __init__(self, *, s_index: int = -1, normalize: float = 1.0):
        """Evaluate this object for fixed-boundary VMEC solve and implicit differentiation."""
        self.s_index = int(s_index)
        self.normalize = float(normalize)

    def J(self, ctx: StageContext, state):
        """Evaluate the scalar objective contribution for the current VMEC state."""
        field = b_cartesian_from_state(
            state,
            ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            s_index=self.s_index,
        )
        return jnp.ravel(jnp.asarray(field, dtype=jnp.float64)) / float(self.normalize)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        """Convert this user-facing objective into packed least-squares residuals."""
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class JVector(_MercierProfileObjective):
    """Flux-coordinate current-density vector objective from JXBFORCE channels."""

    name = "J_vector"

    def J(self, ctx: StageContext, state):
        """Evaluate the scalar objective contribution for the current VMEC state."""
        terms = mercier_terms_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            mmax_force=self.mmax_force,
            nmax_force=self.nmax_force,
            include_channels=True,
        )
        sqrtg = jnp.asarray(terms["sqrtg"], dtype=jnp.float64)
        sqrtg_safe = jnp.where(sqrtg != 0.0, sqrtg, jnp.asarray(1.0, dtype=sqrtg.dtype))
        jtheta = jnp.where(sqrtg != 0.0, jnp.asarray(terms["itheta"], dtype=jnp.float64) / sqrtg_safe, 0.0)
        jzeta = jnp.where(sqrtg != 0.0, jnp.asarray(terms["izeta"], dtype=jnp.float64) / sqrtg_safe, 0.0)
        vector = jnp.stack([jtheta, jzeta], axis=-1)
        values = self._select_profile(ctx, vector)
        return jnp.ravel(values) / float(self.normalize)

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        """Convert this user-facing objective into packed least-squares residuals."""
        return ObjectiveTerm(self.name, self.J, target=target, weight=residual_weight)


class ToroidalCurrent(_MercierProfileObjective):
    """Integrated toroidal-current profile from VMEC's Mercier path."""

    name = "torcur"
    profile_key = "torcur"


class ToroidalCurrentGradient(_MercierProfileObjective):
    """Radial derivative of ``ToroidalCurrent`` used by VMEC Mercier terms."""

    name = "torcur_prime"
    profile_key = "ip"


class RedlBootstrapMismatch(_MercierProfileObjective):
    """Redl bootstrap-current mismatch objective for finite-beta studies."""

    name = "redl_bootstrap_mismatch"

    def __init__(
        self,
        *,
        helicity_n: int,
        ne_coeffs,
        Te_coeffs,
        Ti_coeffs=None,
        Zeff_coeffs=1.0,
        surfaces: Sequence[float] | None = None,
        n_lambda: int = 32,
        mmax_force: int | None = None,
        nmax_force: int | None = None,
    ):
        """Evaluate this object for fixed-boundary VMEC solve and implicit differentiation."""
        super().__init__(surfaces=surfaces, normalize=1.0, mmax_force=mmax_force, nmax_force=nmax_force)
        self.helicity_n = int(helicity_n)
        self.ne_coeffs = tuple(float(x) for x in np.ravel(np.asarray(ne_coeffs, dtype=float)))
        self.Te_coeffs = tuple(float(x) for x in np.ravel(np.asarray(Te_coeffs, dtype=float)))
        if Ti_coeffs is None:
            self.Ti_coeffs = None
        else:
            self.Ti_coeffs = tuple(float(x) for x in np.ravel(np.asarray(Ti_coeffs, dtype=float)))
        self.Zeff_coeffs = tuple(float(x) for x in np.ravel(np.atleast_1d(np.asarray(Zeff_coeffs, dtype=float))))
        self.n_lambda = int(n_lambda)

    def _evaluate(self, ctx: StageContext, state):
        return redl_bootstrap_mismatch_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
            helicity_n=self.helicity_n,
            ne_coeffs=self.ne_coeffs,
            Te_coeffs=self.Te_coeffs,
            Ti_coeffs=self.Ti_coeffs,
            Zeff_coeffs=self.Zeff_coeffs,
            surfaces=self.surfaces,
            n_lambda=self.n_lambda,
            mmax_force=self.mmax_force,
            nmax_force=self.nmax_force,
        )

    def J(self, ctx: StageContext, state):
        """Evaluate the scalar objective contribution for the current VMEC state."""
        return self._evaluate(ctx, state)["residuals1d"]

    def total(self, ctx: StageContext, state):
        """Evaluate total for fixed-boundary VMEC solve and implicit differentiation."""
        return self._evaluate(ctx, state)["total"]

    def to_objective_term(self, *, target, residual_weight: float) -> ObjectiveTerm:
        """Convert this user-facing objective into packed least-squares residuals."""
        if not _target_is_zero(target):
            raise ValueError("RedlBootstrapMismatch is already normalized and requires target=0.")
        return ObjectiveTerm(
            self.name,
            self.J,
            target=0.0,
            weight=residual_weight,
            total=lambda ctx, state: float(residual_weight) ** 2 * self.total(ctx, state),
        )


__all__ = [
    "BDotB",
    "BDotGradV",
    "BVector",
    "BetaTotal",
    "DMerc",
    "GlasserResistiveInterchange",
    "JDotB",
    "JVector",
    "MagneticWell",
    "RedlBootstrapMismatch",
    "ToroidalCurrent",
    "ToroidalCurrentGradient",
    "VolavgB",
]
