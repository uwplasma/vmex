"""Fourier coil coordinates and conservative physical step bounds.

Coordinates are relative changes in selected base currents, followed by
additive Cartesian Fourier coefficients in metres (coil, coordinate, mode).
The nominal arrays are copied; symmetry copies never become independent DOFs.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
import jax.numpy as jnp
from .transforms import register_pytree_dataclass


@dataclass(frozen=True, eq=False)
class DirectCoilField:
    """Filament-quadrature magnetic field with JAX derivatives."""

    gamma: Any
    gamma_dash: Any
    currents: Any

    def b_cyl(self, r: Any, phi: Any, z: Any) -> tuple[Any, Any, Any]:
        """Evaluate the filament field in cylindrical coordinates, in tesla."""
        rr, pp, zz = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
        cosine, sine = jnp.cos(pp), jnp.sin(pp)
        xyz = jnp.stack((rr * cosine, rr * sine, zz), axis=-1)
        displacement = xyz[..., None, None, :] - jnp.asarray(self.gamma)
        radius2 = jnp.sum(displacement * displacement, axis=-1)
        inv_radius3 = jnp.maximum(radius2, 1.0e-30) ** -1.5
        differential = jnp.cross(jnp.asarray(self.gamma_dash), displacement)
        differential = differential * inv_radius3[..., None]
        point_ndim = xyz.ndim - 1
        current_shape = (1,) * point_ndim + (-1, 1, 1)
        weighted = differential * jnp.reshape(jnp.asarray(self.currents), current_shape)
        bxyz = 1.0e-7 * jnp.mean(jnp.sum(weighted, axis=-3), axis=-2)
        br = cosine * bxyz[..., 0] + sine * bxyz[..., 1]
        bphi = -sine * bxyz[..., 0] + cosine * bxyz[..., 1]
        return br, bphi, bxyz[..., 2]


register_pytree_dataclass(DirectCoilField)


class CoilParameters:
    """Map a finite design vector to ESSOS coils without an equilibrium solve."""

    def __init__(
        self,
        coefficients,
        currents,
        *,
        current_dofs,
        max_coil_mode=None,
        nfp=2,
        stellsym=True,
        n_segments=75,
        scales=None,
    ):
        if np.iscomplexobj(coefficients) or np.iscomplexobj(currents):
            raise ValueError("coil coefficients and currents must be real")
        coefficients, currents = np.asarray(coefficients, dtype=float), np.asarray(currents, dtype=float)
        if (
            coefficients.ndim != 3
            or coefficients.shape[0] < 1
            or coefficients.shape[1] != 3
            or coefficients.shape[2] % 2 != 1
            or currents.shape != (coefficients.shape[0],)
            or not np.all(np.isfinite(coefficients))
            or not np.all(np.isfinite(currents))
        ):
            raise ValueError("finite coefficients (coils, 3, 2*order+1) and base currents required")
        order = (coefficients.shape[2] - 1) // 2
        mode = order if max_coil_mode is None else max_coil_mode
        if int(mode) != mode or not 0 <= mode <= order:
            raise ValueError("max_coil_mode must be an integer within the input coil order")
        self.current_dofs = tuple(current_dofs)
        if len(set(self.current_dofs)) != len(self.current_dofs) or any(
            int(i) != i or not 0 <= i < len(currents) for i in self.current_dofs
        ):
            raise ValueError("current_dofs must be unique base-coil indices")
        self.current_dofs = tuple(int(i) for i in self.current_dofs)
        if any(currents[i] == 0 for i in self.current_dofs):
            raise ValueError("relative current coordinates require nonzero nominal currents")
        if int(nfp) != nfp or nfp < 1 or int(n_segments) != n_segments or n_segments < 3:
            raise ValueError("positive nfp and at least three segments required")
        # Immutable NumPy backing data prevents changes to the coordinate chart.
        self.coefficients = np.frombuffer(coefficients.tobytes(), dtype=float).reshape(coefficients.shape)
        self.currents = np.frombuffer(currents.tobytes(), dtype=float)
        self.mode = int(mode)
        self.nfp, self.stellsym, self.n_segments = int(nfp), bool(stellsym), int(n_segments)
        self.curve_shape = (len(currents), 3, 2 * self.mode + 1)
        self.ncurrent = len(self.current_dofs)
        self.size = self.ncurrent + int(np.prod(self.curve_shape))
        self.x0 = np.zeros(self.size)
        names = [f"current[{i}]/nominal" for i in self.current_dofs]
        modes = ["constant"] + [f"{kind}({k})" for k in range(1, self.mode + 1) for kind in ("sin", "cos")]
        names += [f"coil[{i}].{axis}.{mode}" for i in range(len(currents)) for axis in "xyz" for mode in modes]
        self.dof_names = tuple(names)
        if scales is None:
            mode_scales = [0.002] + [0.002 / k**2 for k in range(1, self.mode + 1) for _ in range(2)]
            scales = np.r_[np.full(self.ncurrent, 0.06), np.tile(mode_scales, 3 * len(currents))]
        scales = np.asarray(scales, dtype=float)
        if scales.shape != (self.size,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("one positive finite scale per coordinate required")
        self.scales = np.frombuffer(scales.tobytes(), dtype=float)

    @classmethod
    def from_coils(cls, coils, **kwargs):
        """Import physical geometry/current arrays through standard ESSOS APIs.

        ESSOS exposes scaled curve DOFs but raw currents. Convert the curve
        coefficients once here; our design coordinates are always in metres.
        Both ESSOS main and the research branch support these accessors.
        """
        return cls(
            np.asarray(coils.dofs_curves) / np.asarray(coils.curves.scaling)[None, None, :],
            coils.dofs_currents_raw,
            nfp=coils.nfp,
            stellsym=coils.stellsym,
            n_segments=coils.n_segments,
            **kwargs,
        )

    def _parameters(self, x):
        values = jnp.asarray(x)
        if values.shape != (self.size,) or not jnp.issubdtype(values.dtype, jnp.floating):
            raise ValueError(f"expected floating parameter vector of shape ({self.size},)")
        return values

    def base_currents_at(self, x):
        """Return physical base-coil currents in amperes."""
        x = self._parameters(x)
        currents = jnp.asarray(self.currents)
        for local, base in enumerate(self.current_dofs):
            currents = currents.at[base].add(x[local] * self.currents[base])
        return currents

    def curve_dofs_at(self, x):
        """Return full Cartesian Fourier coefficients in metres."""
        x = self._parameters(x)
        return (
            jnp.asarray(self.coefficients)
            .at[:, :, : 2 * self.mode + 1]
            .add(x[self.ncurrent :].reshape(self.curve_shape))
        )

    def coils_from_x(self, x):
        """Construct coils without mutating the nominal input coils."""
        from essos.coils import Coils, Curves

        return Coils(Curves(self.curve_dofs_at(x), self.n_segments, self.nfp, self.stellsym), self.base_currents_at(x))

    def __call__(self, x):
        """Return the same filament-quadrature field used by the maintained case."""
        coils = self.coils_from_x(x)
        return DirectCoilField(jnp.asarray(coils.gamma), jnp.asarray(coils.gamma_dash), jnp.asarray(coils.currents))

    def motion_bounds(self, delta):
        """Bound coil displacement at every angle and nominal-current fraction."""
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (self.size,) or not np.all(np.isfinite(delta)):
            raise ValueError("finite coordinate displacement required")
        coefficients = delta[self.ncurrent :].reshape(self.curve_shape)
        bound = np.linalg.norm(coefficients[:, :, 0], axis=1)
        for k in range(1, self.mode + 1):
            bound += np.linalg.svd(coefficients[:, :, 2 * k - 1 : 2 * k + 1], compute_uv=False)[:, 0]
        return float(np.max(bound)), float(np.max(np.abs(delta[: self.ncurrent]), initial=0.0))
