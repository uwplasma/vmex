"""Per-case physical constraint diagnostics, shared by fixed-boundary workflows."""


class PhysicalConstraints:
    def __init__(self, *, IOTA_FLOOR=0.19, RADIUS_TARGET=1.0, RADIUS_TOLERANCE=0.01,
                 IOTA_MARGIN=0.0005, RADIUS_MARGIN=0.001, FORCE_TOLERANCE=1e-11):
        self.IOTA_FLOOR = IOTA_FLOOR
        self.RADIUS_TARGET = RADIUS_TARGET
        self.RADIUS_TOLERANCE = RADIUS_TOLERANCE
        self.IOTA_MARGIN = IOTA_MARGIN
        self.RADIUS_MARGIN = RADIUS_MARGIN
        self.FORCE_TOLERANCE = FORCE_TOLERANCE

    @classmethod
    def from_settings(cls, settings):
        names = ("IOTA_FLOOR", "RADIUS_TARGET", "RADIUS_TOLERANCE", "IOTA_MARGIN", "RADIUS_MARGIN")
        return cls(**{name: getattr(settings, name) for name in names},
                   FORCE_TOLERANCE=settings.EQUILIBRIUM_FTOL)

    @staticmethod
    def physical_values(state, runtime):
        import jax.numpy as jnp
        from vmex import optimize as opt
        return jnp.stack((opt.min_abs_iota(state, runtime), opt.major_radius(state, runtime)))

    def inequalities(self, values):
        """Dimensionless favourable-positive rows: iota, radius lower, radius upper."""
        import jax.numpy as jnp
        iota, radius = values
        width = self.RADIUS_TOLERANCE - self.RADIUS_MARGIN
        return jnp.stack(((iota-self.IOTA_FLOOR-self.IOTA_MARGIN)/self.IOTA_FLOOR,
                          (radius-self.RADIUS_TARGET+width)/self.RADIUS_TOLERANCE,
                          (self.RADIUS_TARGET+width-radius)/self.RADIUS_TOLERANCE))

    def diagnostics(self, state, runtime):
        import numpy as np
        iota, radius = map(float, self.physical_values(state, runtime))
        return dict(major_radius_m=radius, radius_error_m=radius-self.RADIUS_TARGET,
            iota_constraint_slack=iota-self.IOTA_FLOOR,
            radius_constraint_slack_m=self.RADIUS_TOLERANCE-abs(radius-self.RADIUS_TARGET),
            constraints_feasible=int(iota >= self.IOTA_FLOOR and abs(radius-self.RADIUS_TARGET) <= self.RADIUS_TOLERANCE),
            optimizer_constraints_feasible=int(np.min(self.inequalities((iota,radius))) >= -1e-8))
