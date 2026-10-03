"""Typed exception taxonomy for vmex (zero-crash policy).

Every physics or input failure maps to one of these exceptions instead of a
crash, a bare traceback, or ``sys.exit``.  Each exception carries the
diagnostic state needed to understand the failure (iteration counters, force
residuals, offending surface).  The CLI catches :class:`VmecError` and prints
the VMEC2000-style termination message from :data:`WERROR_MESSAGES` plus a
one-line remedy hint.

VMEC2000 counterpart: the ``ier_flag`` error codes defined in
``Sources/General/vmec_params.f`` and the ``werror`` message table printed by
``Sources/Input_Output/fileout.f``.  See §2.5.
"""

from __future__ import annotations

from dataclasses import dataclass

# VMEC2000 ier_flag values (Sources/General/vmec_params.f).
NORM_TERM_FLAG = 0
BAD_JACOBIAN_FLAG = 1
MORE_ITER_FLAG = 2
JAC75_FLAG = 4
INPUT_ERROR_FLAG = 5
PHIEDGE_ERROR_FLAG = 7
NS_ERROR_FLAG = 8
MISC_ERROR_FLAG = 9
SUCCESSFUL_TERM_FLAG = 11

# Internal-only loop status.  VMEC2000 has no dedicated ``ier_flag`` for a
# non-finite force evaluation; callers still receive ``MISC_ERROR_FLAG`` via
# :class:`VmecNumericalError`, while this distinct carry value lets
# ``solver._finalize`` distinguish NaN/Inf from a Jacobian-retry failure.
NONFINITE_FLAG = 90

# Internal-only eqsolve control transfer.  VMEC2000 communicates this as
# ``irst = 4`` (not an ier_flag): with ``LMOVE_AXIS=T``, a finite first force
# sum above 1e2 returns to eqsolve so ``guess_axis`` can rebuild the initial
# profiles before any momentum step is taken.  A distinct carry status lets
# the jitted VMEX loop make the same host-side control transfer.
AXIS_REGUESS_FLAG = 91

#: VMEC2000 termination messages, keyed by ier_flag
#: (Sources/Input_Output/fileout.f, ``werror`` table).
WERROR_MESSAGES: dict[int, str] = {
    NORM_TERM_FLAG: "EXECUTION TERMINATED NORMALLY",
    BAD_JACOBIAN_FLAG: "INITIAL JACOBIAN CHANGED SIGN!",
    MORE_ITER_FLAG: "MORE ITERATIONS REQUIRED",
    JAC75_FLAG: "MORE THAN 75 JACOBIAN ITERATIONS (DECREASE DELT)",
    INPUT_ERROR_FLAG: "ERROR READING INPUT FILE OR NAMELIST",
    PHIEDGE_ERROR_FLAG: "PHIEDGE HAS WRONG SIGN IN VACUUM REGION",
    NS_ERROR_FLAG: "NS ARRAY MUST NOT BE ALL ZEROES",
    MISC_ERROR_FLAG: "ERROR IN INPUT VALUES",
    SUCCESSFUL_TERM_FLAG: "EXECUTION TERMINATED NORMALLY",
}


@dataclass
class VmecError(Exception):
    """Base class for all vmex failures.

    Attributes
    ----------
    message:
        Human-readable description (VMEC2000-style where applicable).
    hint:
        One-line remedy suggestion shown by the CLI.
    ier_flag:
        The matching VMEC2000 ``ier_flag`` code, for wout/status parity.
    """

    message: str
    hint: str = ""
    ier_flag: int = MISC_ERROR_FLAG

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


@dataclass
class VmecInputError(VmecError):
    """Invalid or unreadable input (INDATA / JSON / arguments).

    Raised host-side, before or during setup, whenever a deck or an API
    argument cannot be turned into a well-posed run: an unparsable or missing
    INDATA namelist or ``_vmex`` JSON block, an unsupported or contradictory
    option, an out-of-range resolution or profile specification, a restart
    ``wout`` that does not match the requested run, and the ``APHI``
    toroidal-flux derivative reversing sign inside ``s`` in ``[0, 1]``
    (:func:`vmex.core.setup.validate_torflux_monotone`).  Nothing here is
    traced, so the check costs a valid deck nothing.

    It adds no attributes of its own beyond pinning ``ier_flag``; the
    ``message`` names the offending key or file and the ``hint`` says what to
    change.

    VMEC2000: ``input_error_flag`` paths in ``readin.f``.

    Attributes
    ----------
    ier_flag:
        Always ``INPUT_ERROR_FLAG`` (VMEC2000 ``ier_flag = 5``).
    """

    ier_flag: int = INPUT_ERROR_FLAG


@dataclass
class VmecJacobianError(VmecError):
    """The flux-surface Jacobian changed sign and could not be recovered.

    Raised after the VMEC2000 escalation ladder is exhausted (axis re-guess,
    time-step resets at ijacob = 25/50, abort at 75 → ``jac75_flag``).
    VMEC2000: ``Sources/General/jacobian.f`` (irst=2) and ``eqsolve.f``.
    Raised from the solver's finalize step for every terminal ``ier_flag``
    that is neither convergence nor a non-finite force, so it also carries
    ``BAD_JACOBIAN_FLAG`` (a sign change already in the initial geometry).
    The usual remedies are a smaller ``DELT`` or a better axis guess.

    Attributes
    ----------
    ier_flag:
        The terminal VMEC2000 code, ``JAC75_FLAG`` (4) for the exhausted
        ladder or ``BAD_JACOBIAN_FLAG`` (1) for an initial sign change; any
        code without a ``WERROR_MESSAGES`` entry is reported as
        ``JAC75_FLAG``.
    iteration:
        The solver iteration counter reached when the run gave up.
    jacobian_resets:
        VMEC2000's ``ijacob``: how many times the Jacobian sign change forced
        a restart with a reduced time step.
    fsq:
        The final ``(fsqr, fsqz, fsql)`` invariant force residuals, or
        ``None`` if unavailable.  These are the dimensionless normalised
        sums of squares of ``getfsq.f``, on the same scale as ``ftol``.
    """

    ier_flag: int = JAC75_FLAG
    iteration: int = 0
    jacobian_resets: int = 0
    fsq: tuple[float, float, float] | None = None  # (fsqr, fsqz, fsql)


@dataclass
class VmecConvergenceError(VmecError):
    """The force residuals did not reach ftol within the iteration budget.

    VMEC2000: ``more_iter_flag`` from ``eqsolve.f``.  Carries the residual
    history tail so callers can decide whether to continue (hot restart).
    Convergence requires ``fsqr``, ``fsqz`` and ``fsql`` to be at or below
    ``ftol`` simultaneously; this error means the iteration budget
    (``NITER``) ran out first while everything stayed finite.  Comparing
    ``fsq`` against ``ftol`` shows which of the three components -- ``R``,
    ``Z`` or ``lambda`` -- is the laggard.

    Attributes
    ----------
    ier_flag:
        Always ``MORE_ITER_FLAG`` (VMEC2000 ``ier_flag = 2``).
    iteration:
        The iteration counter at which the budget was exhausted.
    ftol:
        The convergence threshold that was in force, dimensionless.
    fsq:
        The final ``(fsqr, fsqz, fsql)`` invariant force residuals of
        ``getfsq.f`` -- normalised sums of squares of the R, Z and lambda
        forces, dimensionless and directly comparable with ``ftol`` -- or
        ``None`` if unavailable.
    """

    ier_flag: int = MORE_ITER_FLAG
    iteration: int = 0
    fsq: tuple[float, float, float] | None = None
    ftol: float = 0.0


@dataclass
class VmecNumericalError(VmecError):
    """A force evaluation produced NaN or infinity.

    This is intentionally a fail-fast error: once a non-finite value reaches
    the Richardson momentum state, later iterations cannot diagnose or repair
    its source.  Common first-iteration causes are zero effective toroidal
    flux (``PHIEDGE``/``APHI``), a singular or sign-changing initial geometry,
    and non-finite profile values.

    Internally the loop carries the distinct status ``NONFINITE_FLAG`` so
    ``solver._finalize`` can tell NaN/Inf apart from a Jacobian-retry
    failure, but callers still see the generic ``MISC_ERROR_FLAG`` for
    wout/status parity with VMEC2000, which has no dedicated code for this.
    This class is also the base of the strong-force polish errors.

    Attributes
    ----------
    ier_flag:
        Always ``MISC_ERROR_FLAG`` (VMEC2000 ``ier_flag = 9``).
    iteration:
        The iteration counter at which the non-finite value was detected.
    fsq:
        The last ``(fsqr, fsqz, fsql)`` invariant force residuals of
        ``getfsq.f``, dimensionless, or ``None``.  They are themselves
        typically NaN or Inf by the time this is raised, and the polish
        subclasses never populate this field.
    """

    ier_flag: int = MISC_ERROR_FLAG
    iteration: int = 0
    fsq: tuple[float, float, float] | None = None


@dataclass
class AdjointSolveError(VmecNumericalError):
    """The implicit-adjoint Krylov solve returned an unconverged ``lambda``.

    Raised by the reverse pass of :func:`vmex.core.implicit.solve_implicit`
    (and the multi-RHS pullback) when the GCROT(m, k) adjoint solve exhausts
    its budget with a residual above the acceptance threshold.  An
    unconverged adjoint is a *silently wrong* gradient — plausible magnitude,
    wrong value — so it is never returned: host-eager reverse passes raise
    this typed error; traced reverse passes (a ``jax.jit`` around the whole
    gradient) NaN-poison the adjoint instead, which the optimize drivers'
    finite-gradient guards catch.

    Attributes
    ----------
    iterations:
        Total inner Krylov (Arnoldi) iterations the solve performed.
    residual_norm:
        The true residual norm ``||b - A^T lambda||`` the solve reached.
    tolerance:
        The acceptance threshold it failed to meet
        (``slack * adjoint_tol * ||b||``).

    Remedies: raise ``adjoint_maxiter``/``adjoint_gcrot_m``/
    ``adjoint_gcrot_k`` (more Krylov budget) or loosen ``adjoint_tol``.
    """

    iterations: int = 0
    residual_norm: float = 0.0
    tolerance: float = 0.0


@dataclass
class StrongForceCertificationError(VmecNumericalError):
    """A force-balance polish missed its force or stationarity tolerance.

    Raised by :func:`vmex.core.polish.polish_legacy_solution` when
    :class:`~vmex.core.polish.PolishConfig` has ``fail_policy="raise"``
    (the default); with ``fail_policy="return_unpolished"`` the driver returns
    the unpolished state and its report instead.

    Attributes
    ----------
    force_norm:
        Independent volume-RMS force over ``volavgB**2 / (mu0 Aminor_p)``.
    force_tolerance:
        ``PolishConfig.force_tolerance``.
    stationarity:
        Frobenius-scaled projected gradient ``eta`` of the final state.
    stationarity_tolerance:
        ``PolishConfig.stationarity_tolerance``.
    """

    force_norm: float = float("inf")
    force_tolerance: float = 0.0
    stationarity: float = float("inf")
    stationarity_tolerance: float = 0.0


@dataclass
class MgridNotFoundError(VmecError):
    """A free-boundary run referenced an mgrid file that cannot be read.

    The solver catches this and falls back to a fixed-boundary solve with a
    warning (behavior VMEC2000 has and VMEC++ dropped — §2.5); it is
    re-raised only when the caller explicitly requires free-boundary.

    Raised by :func:`vmex.core.mgrid.read_mgrid` in two cases: the resolved
    path is not an existing file, or the file exists but netCDF4 cannot open
    it as a MAKEGRID dataset.  The check is host-side, so nothing here is
    traced.

    Attributes
    ----------
    ier_flag:
        Always ``INPUT_ERROR_FLAG`` (VMEC2000 ``ier_flag = 5``).
    path:
        The expanded filesystem path that was tried, as a string.  Deck
        paths from ``MGRID_FILE`` are resolved relative to the working
        directory, which is the usual reason this differs from what the
        deck says.
    """

    ier_flag: int = INPUT_ERROR_FLAG
    path: str = ""


@dataclass
class TrialRejected(VmecError):
    """An optimizer trial point could not be solved or certified.

    Raised by :class:`~vmex.core.freeboundary_problem.FreeBoundaryProblem`
    for an expected numerical rejection of a trial: a step beyond the
    continuation budget, an ordinary solve that does not converge, or a
    failed certification or polish (the wrapped :class:`VmecError` is the
    ``__cause__``).  The accepted equilibrium is unchanged, so a backtracking
    caller may try a smaller step; :func:`vmex.core.optimize.minimize` stops
    at the accepted point with ``stop_reason="equilibrium_trial_rejected"``.

    It adds no attributes of its own; the ``message`` says why the trial was
    rejected.
    """
