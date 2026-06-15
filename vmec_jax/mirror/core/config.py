"""Configuration objects for mirror workflows."""

from __future__ import annotations

from dataclasses import dataclass

from .grids import MirrorGrid, make_mirror_grid


@dataclass(frozen=True)
class MirrorResolution:
    """Resolution for the first mirror fixed-boundary discretization."""

    ns: int = 17
    ntheta: int = 1
    nxi: int = 33
    mpol: int = 0


@dataclass(frozen=True)
class MirrorConfig:
    """Minimal mirror configuration for grid and basis construction."""

    resolution: MirrorResolution = MirrorResolution()
    z_min: float = -1.0
    z_max: float = 1.0

    def build_grid(self) -> MirrorGrid:
        """Build the static mirror grid described by this configuration."""
        return make_mirror_grid(
            ns=self.resolution.ns,
            ntheta=self.resolution.ntheta,
            nxi=self.resolution.nxi,
            mpol=self.resolution.mpol,
            z_min=self.z_min,
            z_max=self.z_max,
        )
