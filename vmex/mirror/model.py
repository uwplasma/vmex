"""Mirror input contracts and differentiable state containers.

The supported open-end model is a finite equilibrium domain between two
fixed, flux-carrying cuts.  These cuts are not periodic and are not
plasma-vacuum interfaces.  The lateral ``s=1`` surface is the fixed or free
plasma boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

MIRROR_INPUT_SCHEMA = "vmex.mirror.input/2"
MIRROR_OUTPUT_SCHEMA = "vmex.mirror.mout/1"

Array = Any


@dataclass(frozen=True)
class MirrorResolution:
    """Static resolution for ``(s, theta, xi)`` mirror coordinates.

    ``mpol`` is the largest retained theta Fourier mode. Axisymmetry uses
    ``mpol=0``. The collocation size is derived as ``2*mpol+1`` so no
    undeclared or Nyquist mode enters the state.
    """

    ns: int = 17
    mpol: int = 0
    nxi: int = 33

    def __post_init__(self) -> None:
        if self.ns < 3:
            raise ValueError("mirror ns must be >= 3 for second-order radial differences")
        if self.mpol < 0:
            raise ValueError("mirror mpol must be >= 0")
        if self.nxi < 2:
            raise ValueError("mirror nxi must be >= 2")

    @property
    def ntheta(self) -> int:
        """Number of nodal values required to represent modes through ``mpol``."""

        return 2 * self.mpol + 1

    @property
    def axisymmetric(self) -> bool:
        """Whether theta dependence is absent."""

        return self.mpol == 0


@dataclass(frozen=True)
class MirrorConfig:
    """Numerical and boundary contract for a mirror equilibrium.

    Geometry and normal flux are fixed at both axial cuts while field lines
    may cross them. End losses, sheaths, sources, and transport are outside
    this equilibrium model.

    The default nonlinear tolerance is the requested component-wise physical
    force tolerance.  It is not an optimizer objective tolerance.
    """

    resolution: MirrorResolution = MirrorResolution()
    z_min: float = -1.0
    z_max: float = 1.0
    ftol: float = 1.0e-12
    max_iterations: int = 2000

    def __post_init__(self) -> None:
        if not self.z_max > self.z_min:
            raise ValueError("z_max must be greater than z_min")
        if not self.ftol > 0.0:
            raise ValueError("mirror ftol must be positive")
        if self.max_iterations < 1:
            raise ValueError("mirror max_iterations must be >= 1")

    def build_grid(self) -> "MirrorGrid":
        """Build immutable collocation and quadrature data."""

        from .basis import build_mirror_grid

        return build_mirror_grid(self)


@dataclass(frozen=True)
class MirrorBoundary:
    """Lateral boundary scale ``a(theta, xi)`` in ``r=sqrt(s)*a``.

    ``radius_scale`` has shape ``(ntheta, nxi)``.  It is a differentiable JAX
    leaf so fixed-boundary shape derivatives do not require another boundary
    representation.
    """

    radius_scale: Array

    @classmethod
    def from_radius(cls, radius: Array, grid: "MirrorGrid") -> "MirrorBoundary":
        """Broadcast scalar, axial, or full theta-axial radii to the grid."""

        value = jnp.asarray(radius)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.asarray(1.0).dtype)
        if value.ndim == 0:
            value = jnp.broadcast_to(value, (grid.ntheta, grid.nxi))
        elif value.shape == (grid.nxi,):
            value = jnp.broadcast_to(value[None, :], (grid.ntheta, grid.nxi))
        elif value.shape != (grid.ntheta, grid.nxi):
            raise ValueError(
                f"boundary radius shape {value.shape} must be scalar, ({grid.nxi},), or ({grid.ntheta}, {grid.nxi})"
            )
        return cls(radius_scale=value)

    @classmethod
    def from_axis_field(
        cls,
        axial_flux_derivative: Array,
        on_axis_bz: Array,
        grid: "MirrorGrid",
        *,
        radius_floor: float = 0.0,
    ) -> "MirrorBoundary":
        """Build the leading-order flux tube ``a=sqrt(2*Psi'/|Bz|)``.

        This paraxial relation is an initializer and analytic validation
        fixture, not a replacement for a finite-radius equilibrium solve.
        """

        bz = jnp.asarray(on_axis_bz)
        if bz.shape != (grid.nxi,):
            raise ValueError(f"on_axis_bz shape {bz.shape} must be ({grid.nxi},)")
        flux = jnp.asarray(axial_flux_derivative, dtype=bz.dtype)
        if flux.ndim != 0:
            raise ValueError("flux-tube boundary requires a scalar axial_flux_derivative")
        tiny = jnp.finfo(bz.dtype).tiny
        radius = jnp.sqrt(2.0 * flux / jnp.maximum(jnp.abs(bz), tiny))
        radius = jnp.maximum(radius, jnp.asarray(radius_floor, dtype=bz.dtype))
        return cls.from_radius(radius, grid)


@dataclass(frozen=True)
class MirrorState:
    """Differentiable mirror geometry and field-line state.

    The first two arrays have shape ``(ns, ntheta, nxi)``. ``radius_scale`` defines
    ``r=sqrt(s)*radius_scale``; storing the regular scale rather than ``r``
    avoids evolving a singular radial derivative at the magnetic axis.
    ``lambda_stream`` is the divergence-free field stream function and uses a
    zero surface-average gauge in the solver lane.
    """

    radius_scale: Array
    lambda_stream: Array

    @classmethod
    def from_boundary(cls, boundary: MirrorBoundary, grid: "MirrorGrid") -> "MirrorState":
        """Construct the radial self-similar initial state for a boundary."""

        boundary_radius = jnp.asarray(boundary.radius_scale)
        expected = (grid.ntheta, grid.nxi)
        if boundary_radius.shape != expected:
            raise ValueError(f"boundary shape {boundary_radius.shape} does not match {expected}")
        shape = (grid.ns, grid.ntheta, grid.nxi)
        return cls(
            radius_scale=jnp.broadcast_to(boundary_radius[None, :, :], shape),
            lambda_stream=jnp.zeros(shape, dtype=boundary_radius.dtype),
        )

    def validate_shape(self, grid: "MirrorGrid") -> None:
        """Raise when state arrays do not match the static grid."""

        expected = (grid.ns, grid.ntheta, grid.nxi)
        if self.radius_scale.shape != expected:
            raise ValueError(f"radius_scale shape {self.radius_scale.shape} does not match {expected}")
        if self.lambda_stream.shape != expected:
            raise ValueError(f"lambda_stream shape {self.lambda_stream.shape} does not match {expected}")


@dataclass(frozen=True)
class MirrorInput:
    """One open-mirror equilibrium: resolution, boundary, flux, profiles, coils.

    The mirror counterpart of :class:`~vmex.VmecInput`, solved by
    :func:`vmex.mirror.solve_mirror`. The lateral boundary is the polar radius
    ``a(theta, z) = sum_m rbc[m, k] cos(m theta) + rbs[m, k] sin(m theta)``
    about the straight axis at the axial stations ``zb[k]``, joined by a cubic
    spline; empty ``zb`` means stations spaced uniformly over
    ``[z_min, z_max]``. ``phiedge`` is the axial flux through the boundary
    [Wb]; the pressure is ``pres_scale * sum_i am[i] s**i`` [Pa] and
    ``current_derivative`` is the axial-current profile ``I'(s)``.

    ``lfreeb=True`` solves the free boundary held by the circular coils
    ``coil_radius``/``coil_z``/``coil_current`` (or an ``external_field``
    passed to the solve); the start is the vacuum flux tube of ``phiedge``,
    so the boundary table is not used, and finite pressure is reached by a
    continuation from vacuum. ``elements`` is the number of axial B-spline
    elements; ``nxi`` sizes the free-boundary axial collocation grid.
    """

    ns: int = 7
    mpol: int = 0
    elements: int = 6
    nxi: int = 17
    z_min: float = -1.0
    z_max: float = 1.0
    phiedge: float | None = None
    zb: Any = ()
    rbc: Any = ()
    rbs: Any = ()
    pres_scale: float = 0.0
    am: Any = (1.0, -1.0)
    current_derivative: float = 0.0
    gamma: float = 5.0 / 3.0
    ftol: float = 1.0e-12
    niter: int = 1000
    lfreeb: bool = False
    coil_radius: Any = ()
    coil_z: Any = ()
    coil_current: Any = ()
    exterior_ntheta: int = 12
    exterior_order: int = 6

    _ARRAYS = frozenset({"zb", "am", "coil_radius", "coil_z", "coil_current"})

    @classmethod
    def from_text(cls, text: str) -> "MirrorInput":
        """Parse a ``&MIRROR ... /`` namelist (see :meth:`from_file`)."""

        import re
        from dataclasses import fields

        from vmex.core.input import _find_assignments, _parse_key, _parse_scalar, _strip_fortran_comments, _tokenize_values

        start = re.search(r"&\s*MIRROR\b", text, flags=re.IGNORECASE)
        end = re.search(r"^\s*/\s*$", text[start.end():], flags=re.MULTILINE) if start else None
        if end is None:
            raise ValueError("no &MIRROR ... / namelist found")
        block = "\n".join(_strip_fortran_comments(line) for line in text[start.end(): start.end() + end.start()].splitlines())
        known = {field.name for field in fields(cls)}
        values: dict[str, Any] = {}
        rows: dict[str, dict[int, list[float]]] = {"rbc": {}, "rbs": {}}
        matches = _find_assignments(block)
        for index, match in enumerate(matches):
            name, designator = _parse_key(match.group("key"))
            name = name.lower()
            stop = matches[index + 1].start() if index + 1 < len(matches) else len(block)
            parsed = [_parse_scalar(token) for token in _tokenize_values(block[match.end(): stop].strip())]
            if name in rows:
                if designator is None or len(designator) != 1 or not isinstance(designator[0], int):
                    raise ValueError(f"{name.upper()} takes one poloidal index: {name.upper()}(m) = values at ZB")
                rows[name][designator[0]] = parsed
            elif name not in known or designator is not None:
                raise ValueError(f"unknown &MIRROR variable {match.group('key').strip()}")
            else:
                values[name] = parsed if name in cls._ARRAYS else parsed[0]
        for name, table in rows.items():
            if table:
                width = {len(row) for row in table.values()}
                if len(width) != 1:
                    raise ValueError(f"every {name.upper()} row needs one value per boundary station")
                array = np.zeros((max(table) + 1, width.pop()))
                for m, row in table.items():
                    array[m] = row
                values[name] = array
        return cls(**values)

    @classmethod
    def from_file(cls, path: Any) -> "MirrorInput":
        """Read a mirror input deck: one ``&MIRROR`` namelist.

        Every field of this class is a (case-insensitive) variable of the
        same name; ``RBC(m) = ...`` and ``RBS(m) = ...`` give the Fourier
        mode ``m`` at the ``ZB`` stations. The format is documented in
        ``docs/reference/mirror-input.rst``.
        """

        from pathlib import Path

        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    def to_file(self, path: Any) -> Any:
        """Write this input as a ``&MIRROR`` deck and return the path."""

        from dataclasses import fields
        from pathlib import Path

        def text(value: Any) -> str:
            if isinstance(value, bool):
                return "T" if value else "F"
            return " ".join(repr(float(item)) for item in np.ravel(value)) if np.ndim(value) else repr(value)

        lines = ["&MIRROR"]
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in ("rbc", "rbs"):
                for m, row in enumerate(np.atleast_2d(np.asarray(value, dtype=float)) if np.size(value) else ()):
                    lines.append(f"  {field.name.upper()}({m}) = {text(row)}")
            elif value is not None and not (field.name in self._ARRAYS and not np.size(value)):
                lines.append(f"  {field.name.upper()} = {text(value)}")
        Path(path).write_text("\n".join(lines + ["/", ""]), encoding="utf-8")
        return Path(path)

    @property
    def config(self) -> MirrorConfig:
        """The numerical contract of the solve."""

        return MirrorConfig(
            resolution=MirrorResolution(ns=int(self.ns), mpol=int(self.mpol), nxi=int(self.nxi)),
            z_min=float(self.z_min),
            z_max=float(self.z_max),
            ftol=float(self.ftol),
            max_iterations=int(self.niter),
        )

    def pressure(self, s: Array) -> Array:
        """Return ``p(s) = pres_scale * sum_i am[i] s**i`` in pascals."""

        coefficients = np.asarray(self.am, dtype=float)[::-1]
        return float(self.pres_scale) * np.polyval(coefficients, np.asarray(s, dtype=float))

    def boundary_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(zb, rbc, rbs)`` as arrays of shape ``(K,)``, ``(mpol+1, K)``."""

        rbc = np.atleast_2d(np.asarray(self.rbc, dtype=float))
        if rbc.size == 0:
            raise ValueError("a fixed-boundary mirror needs boundary coefficients rbc")
        modes, stations = int(self.mpol) + 1, rbc.shape[1]
        if rbc.shape[0] > modes:
            raise ValueError(f"rbc has {rbc.shape[0]} poloidal rows but mpol={self.mpol}")
        rbc = np.pad(rbc, ((0, modes - rbc.shape[0]), (0, 0)))
        rbs = np.atleast_2d(np.asarray(self.rbs, dtype=float)) if np.size(self.rbs) else np.zeros((1, stations))
        if rbs.shape[1] != stations or rbs.shape[0] > modes:
            raise ValueError("rbs must have at most mpol+1 rows and one column per station")
        rbs = np.pad(rbs, ((0, modes - rbs.shape[0]), (0, 0)))
        zb = np.asarray(self.zb, dtype=float)
        if zb.size == 0:
            zb = np.linspace(float(self.z_min), float(self.z_max), stations)
        if zb.shape != (stations,) or np.any(np.diff(zb) <= 0.0):
            raise ValueError("zb must be increasing with one entry per boundary column")
        span = 1.0e-12 * (float(self.z_max) - float(self.z_min))
        if stations < 2 or zb[0] > float(self.z_min) + span or zb[-1] < float(self.z_max) - span:
            raise ValueError("at least two boundary stations zb must cover [z_min, z_max]")
        return zb, rbc, rbs

    def boundary_radius(self, theta: Array, z: Array) -> np.ndarray:
        """Evaluate the boundary polar radius on the ``theta x z`` tensor grid."""

        from scipy.interpolate import CubicSpline

        zb, rbc, rbs = self.boundary_table()
        z = np.asarray(z, dtype=float)
        cosine, sine = CubicSpline(zb, rbc, axis=1)(z), CubicSpline(zb, rbs, axis=1)(z)
        modes = np.arange(rbc.shape[0])[:, None] * np.asarray(theta, dtype=float)[None, :]
        return np.cos(modes).T @ cosine + np.sin(modes).T @ sine

    def with_boundary(self, radius, stations: Array | None = None) -> "MirrorInput":
        """Tabulate a boundary function ``radius(theta, z)`` into ``rbc``/``rbs``.

        The Fourier coefficients are exact on the solver's ``2*mpol+1`` theta
        nodes; the default stations are ``8*elements+1`` uniform ``z`` values.
        """

        count = 2 * int(self.mpol) + 1
        theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        if stations is None:
            stations = np.linspace(float(self.z_min), float(self.z_max), 8 * int(self.elements) + 1)
        zb = np.asarray(stations, dtype=float)
        values = np.broadcast_to(np.asarray(radius(theta[:, None], zb[None, :]), dtype=float), (count, zb.size))
        modes = np.arange(int(self.mpol) + 1)[:, None] * theta[None, :]
        rbc = 2.0 / count * np.cos(modes) @ values
        rbc[0] *= 0.5
        return replace(self, zb=zb, rbc=rbc, rbs=2.0 / count * np.sin(modes) @ values)


def _regularize_axis_radius(radius_scale: Array) -> Array:
    """Remove odd leading poloidal radius modes at the magnetic axis.

    Even modes describe the centered limiting cross-section. Odd radial-shape
    modes translate that section and must vanish as ``sqrt(s)`` for a
    single-valued axis.
    """

    radius_scale = jnp.asarray(radius_scale)
    ntheta = int(radius_scale.shape[1])
    if ntheta == 1:
        return radius_scale.at[0].set(radius_scale[1])
    modes = jnp.rint(jnp.fft.fftfreq(ntheta, d=1.0 / ntheta)).astype(int)
    axis_modes = jnp.fft.fft(radius_scale[1], axis=0)
    centered = jnp.where((jnp.abs(modes) % 2 == 0)[:, None], axis_modes, 0.0)
    return radius_scale.at[0].set(jnp.fft.ifft(centered, axis=0).real)


def project_fixed_boundary_state(
    state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
) -> MirrorState:
    """Apply the side boundary, axis regularity, and lambda gauge.

    The input state's endpoint profiles are prescribed cut data. Keeping them
    intact permits finite-radius flux surfaces instead of forcing every cut
    to be a scaled copy of the LCFS. The lambda surface mean is a pure gauge.
    """

    state.validate_shape(grid)
    boundary_radius = jnp.asarray(boundary.radius_scale)
    if boundary_radius.shape != (grid.ntheta, grid.nxi):
        raise ValueError("boundary shape does not match mirror grid")
    radius_scale = jnp.asarray(state.radius_scale)
    radius_scale = radius_scale.at[-1].set(boundary_radius)
    radius_scale = _regularize_axis_radius(radius_scale)

    lam = jnp.asarray(state.lambda_stream)
    lam = lam.at[0].set(lam[1])
    theta_weights = jnp.asarray(grid.theta_basis.weights)
    xi_weights = jnp.asarray(grid.axial_basis.weights)
    denominator = jnp.sum(theta_weights) * jnp.sum(xi_weights)
    surface_mean = jnp.einsum("j,k,ijk->i", theta_weights, xi_weights, lam) / denominator
    lam = lam - surface_mean[:, None, None]
    return MirrorState(radius_scale=radius_scale, lambda_stream=lam)


jax.tree_util.register_dataclass(MirrorBoundary, data_fields=["radius_scale"], meta_fields=[])
jax.tree_util.register_dataclass(
    MirrorState,
    data_fields=["radius_scale", "lambda_stream"],
    meta_fields=[],
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
