"""Optimizer-neutral objective and derivative callables.

The classes in this module contain no optimization algorithms.  They expose
the small function contracts consumed by SciPy, JAXopt, Optax, and user code.
VMEC-specific construction is imported lazily so this module remains usable in
lightweight tests and does not introduce an import cycle with
``vmex.core.optimize``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import sys
from threading import Event, RLock, Thread
import time
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np


Array = Any
HostFun = Callable[[np.ndarray], Any]


def _nonlinear_constraint(values, jacobian, lower, upper, scales):
    from scipy.optimize import NonlinearConstraint

    lo, hi, scale = np.broadcast_arrays(np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float), np.asarray(scales, dtype=float))
    if np.any(np.isnan(lo) | np.isnan(hi) | (lo > hi)) or np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError("ordered bounds and finite positive constraint scales required")
    return NonlinearConstraint(lambda x: np.asarray(values(x))/scale, lo/scale, hi/scale,
                               jac=lambda x: np.asarray(jacobian(x))/scale[..., None])


def _slice_bounds(bounds: Any, free: np.ndarray, size: int) -> Any:
    """Restrict carried box constraints to the ``free`` decision variables.

    Scalar entries apply to every variable and are carried through; a
    per-variable array is sliced.  Anything else is rejected rather than
    handed on at the wrong length, which an optimizer would apply to the
    wrong variables.
    """
    if bounds is None:
        return None

    def one(entry: Any) -> Any:
        array = np.asarray(entry, dtype=float)
        if array.ndim == 0:
            return entry
        if array.size == size:
            return array.reshape(-1)[free]
        raise ValueError(
            "bounds must be scalar or carry one entry per decision variable "
            f"({size}), got {array.size}")

    lower = getattr(bounds, "lb", None)
    if lower is not None:
        return type(bounds)(one(lower), one(bounds.ub))
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        return (one(bounds[0]), one(bounds[1]))
    raise TypeError(
        "bounds must be None, a (lower, upper) pair, or expose lb/ub")


def _run_with_progress(
    function: Callable[[], Any],
    *,
    action: str,
    complete: str,
    progress: bool,
    report_interval: float,
    stream: Any = None,
    announce: bool = True,
) -> Any:
    """Run one operation with a low-overhead elapsed-time heartbeat.

    With ``announce=False`` nothing is printed unless the call outlives the
    first interval, so a per-evaluation heartbeat stays silent through the
    fast calls and speaks up only for the slow ones.
    """
    interval = float(report_interval)
    if interval <= 0.0:
        raise ValueError("report_interval must be positive")
    if not progress:
        return function()
    stream = sys.stdout if stream is None else stream
    spoke = [False]

    def announce_once() -> None:
        if not spoke[0]:
            spoke[0] = True
            print(f"{action}...", file=stream, flush=True)

    if announce:
        announce_once()
    started = time.perf_counter()
    finished = Event()

    def heartbeat() -> None:
        while not finished.wait(interval):
            announce_once()
            elapsed = time.perf_counter() - started
            print(f"  {elapsed:.1f} s elapsed.", file=stream, flush=True)

    reporter = Thread(target=heartbeat, name="vmex-progress", daemon=True)
    reporter.start()
    try:
        result = function()
    except Exception:
        elapsed = time.perf_counter() - started
        print(f"Failed after {elapsed:.1f} s.", file=stream, flush=True)
        raise
    finally:
        finished.set()
        reporter.join()
    if spoke[0]:
        elapsed = time.perf_counter() - started
        print(f"{complete} in {elapsed:.1f} s.", file=stream, flush=True)
    return result


def _branch_on_status(status: Any, accepted: Callable[[Any], Any],
                      rejected: Callable[[Any], Any]) -> Any:
    """``lax.cond(status == 0, ...)``, or a Python branch when status is concrete.

    Outside ``jax.jit`` (an eager ``jax.value_and_grad`` over a state-composed
    objective) ``lax.cond`` is traced and compiled on every call, because each
    call's equilibrium state enters both branches as fresh constants: three XLA
    programs per call even on a 5-surface deck, and 20-40 s per trial on the
    single-stage free-boundary example. The status is concrete there, so the
    taken branch alone is evaluated; under ``jit`` or ``vmap`` the status is
    abstract and ``lax.cond`` runs as before.
    """
    import jax

    try:
        taken = int(status) == 0
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerIntegerConversionError,
            TypeError):
        return jax.lax.cond(status == 0, accepted, rejected, operand=None)
    return accepted(None) if taken else rejected(None)


@dataclass(frozen=True)
class Evaluation:
    """Values and diagnostics produced at one decision vector.

    Fields that were not requested or are unavailable are ``None``.  ``status``
    is a short machine-readable value such as ``"success"`` or
    ``"failed_solve"``; ``message`` is intended for a human.  Optimizers use
    the ordinary callable methods and do not need to understand this object.

    Returned by :meth:`FunctionProblem.evaluate` and the two ``compile_*``
    helpers.  It is a report, not a cache: nothing here is consulted by a
    later evaluation.

    Attributes
    ----------
    x:
        The decision vector the values were produced at, as an owned float
        copy with the shape of ``problem.x0``.
    value:
        Scalar objective at ``x``.  For a residual-only problem this is the
        least-squares cost ``0.5 * r @ r``, matching SciPy's
        ``OptimizeResult.cost``.  ``None`` when the problem exposes neither
        a scalar objective nor residuals.
    gradient:
        Gradient of ``value`` with respect to ``x``, reshaped to ``x``'s
        shape.  For a residual problem it is ``J.T @ r``.  ``None`` when
        ``derivatives=False`` was requested or no gradient lane exists.
    residual:
        The flattened residual vector ``r(x)``, or ``None`` for a
        scalar-only problem.
    jacobian:
        The residual Jacobian ``dr_i/dx_j`` with shape
        ``(residual.size, x.size)``, or ``None`` when derivatives were not
        requested or the problem provides no Jacobian.
    status:
        ``"success"``, or — from a VMEC-backed problem —
        ``"failed_solve"`` when the equilibrium solve at ``x`` raised, and
        ``"under_converged"`` when it returned but its force residual
        exceeds the threshold below which implicit derivatives are
        certified.
    message:
        Human-readable explanation, empty on success; the solver
        exception's text for ``"failed_solve"``.
    diagnostics:
        Extra per-evaluation values.  Empty for a plain
        :class:`FunctionProblem`.  A :class:`VmecProblem` adds the
        cumulative ``failed_trials`` and ``derivative_fallbacks`` counters,
        a ``solve_stats`` mapping when the implicit lane recorded one, and
        — when the equilibrium at ``x`` could be materialised — the summed
        force residual ``fsq``, its ``fsq_ratio`` to the solve tolerance,
        the configured ``max_fsq_ratio``, and the boolean
        ``derivative_certified``.  A failed solve also carries
        ``exception_type``.
    """

    x: np.ndarray
    value: float | None = None
    gradient: np.ndarray | None = None
    residual: np.ndarray | None = None
    jacobian: np.ndarray | None = None
    status: str = "success"
    message: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the evaluation completed without a recoverable failure."""
        return self.status == "success"


class FunctionProblem:
    """A decision vector plus optimizer-compatible objective callables.

    Parameters are explicit and immutable from the caller's perspective.
    Supplying combined value/gradient or residual/Jacobian functions enables a
    one-entry exact-key cache, so the common SciPy call sequence does not repeat
    expensive work.  The cache is protected by a lock; JAX-native callables do
    not use host state and remain suitable for tracing.

    This class deliberately does not provide ``solve(method=...)``.  Pass its
    methods directly to the optimizer of choice.

    At least one of ``fun``, ``value_and_grad``, ``residual``, or
    ``residual_and_jac`` is required; every other callable is optional and
    the matching method raises :exc:`AttributeError` when its lane is
    absent.  The host callables receive one contiguous float NumPy array and
    are free to be opaque; the ``jax_*`` callables receive whatever the
    caller traces and must stay traceable.

    Two independent one-entry caches, each keyed on the exact bytes of ``x``
    (shape, dtype, and contents — no tolerance), avoid repeating work across
    the split calls an optimizer makes at one iterate.  The scalar cache is
    filled by :meth:`value_and_grad` and so covers ``value_and_grad`` alone,
    ``fun`` together with ``grad``, or ``residual_and_jac``.  The
    least-squares cache is filled by :meth:`residual_and_jac` and so covers
    ``residual_and_jac``, or ``residual`` together with ``residual_jac``.
    Whether the cache actually pays depends on which callables were
    supplied: with ``residual_and_jac``, SciPy's separate ``fun(x)`` then
    ``jac(x)`` calls both route through it and the second is free, whereas
    separately supplied ``residual`` and ``residual_jac`` are each invoked
    directly and share nothing.  Cached arrays are copied out, so a caller
    may mutate what it receives.  The caches are guarded by a re-entrant
    lock; the ``jax_*`` lane touches no host state and stays safe to trace.

    Parameters
    ----------
    x0:
        Initial decision vector.  Copied to a float array; its size fixes
        the number of degrees of freedom and the expected Jacobian column
        count, and its shape is the shape gradients are reshaped to.
    fun:
        ``x -> float``, the scalar objective.  Called directly by
        :meth:`fun` without touching the cache.
    grad:
        ``x -> array``, the objective gradient.  Used only in combination
        with ``fun``; on its own it does not enable :meth:`grad`.
    value_and_grad:
        ``x -> (float, array)``.  The preferred scalar lane: it is one call
        for both quantities and it fills the scalar cache.  The method that
        serves it is also reachable under SciPy's ``fun_and_grad`` name.
    residual:
        ``x -> array``, the least-squares residual vector ``r(x)``.  It is
        flattened, and defines the scalar objective ``0.5 * r @ r`` when the
        problem has no scalar lane of its own (no ``fun``, ``grad``, or
        ``value_and_grad``).
    residual_jac:
        ``x -> array``, the Jacobian ``dr_i/dx_j``.  It must have one
        column per decision variable; anything else raises
        :exc:`ValueError`.
    residual_and_jac:
        ``x -> (array, array)``, the preferred least-squares lane: one call
        for both, filling the cache that :meth:`residual` and
        :meth:`residual_jac` then read.  The Jacobian shape is checked
        exactly against ``(r.size, x0.size)``.
    jax_fun, jax_value_and_grad, jax_residual, jax_residual_jac:
        Traceable counterparts of the four callables above, returned
        unwrapped by the matching ``jax_*`` methods.  They are never
        cached and never see host state, so they remain usable inside
        :func:`jax.jit` and :func:`jax.grad`.  ``jax_fun`` falls back to the
        first element of ``jax_value_and_grad`` when it is not supplied.
    names:
        One name per decision variable, in order, surfaced as
        :attr:`dof_names` for labelling output.  The default is
        ``x[0], x[1], ...``; a length other than ``x0.size`` raises
        :exc:`ValueError`.
    bounds:
        Box constraints stored verbatim for the optimizer to consume — a
        SciPy :class:`~scipy.optimize.Bounds`, or a ``(lower, upper)``
        pair.  This class neither interprets nor enforces them.
    scales:
        Positive finite per-variable scale factors with the shape of
        ``x0``, defaulting to ones.  Like ``bounds`` they are carried, not
        applied: pass them to the optimizer (SciPy's ``x_scale``).  A
        non-finite or non-positive entry raises :exc:`ValueError`.
    metadata:
        Free-form mapping copied onto the instance.  VMEX-built problems
        use it to carry the named residual slices, the solver
        configuration, the mutable solve counters, and the traceable
        state accessors that :class:`VmecProblem` reads.
    evaluation_progress:
        Print an elapsed-time heartbeat around long evaluations.  It stays
        silent until a call outlives the first interval, so fast calls
        print nothing.  It wraps the standalone :meth:`residual` and
        :meth:`residual_jac` calls only, which is where a production deck
        spends minutes; the combined and scalar lanes are unaffected.
    report_interval:
        Seconds between heartbeat lines.  Must be positive.
    """

    def __init__(
        self,
        x0: Array,
        *,
        fun: HostFun | None = None,
        grad: HostFun | None = None,
        value_and_grad: HostFun | None = None,
        residual: HostFun | None = None,
        residual_jac: HostFun | None = None,
        residual_and_jac: HostFun | None = None,
        jax_fun: Callable[[Array], Array] | None = None,
        jax_value_and_grad: Callable[[Array], tuple[Array, Array]] | None = None,
        jax_residual: Callable[[Array], Array] | None = None,
        jax_residual_jac: Callable[[Array], Array] | None = None,
        names: Sequence[str] | None = None,
        bounds: Any = None,
        scales: Array | None = None,
        metadata: Mapping[str, Any] | None = None,
        evaluation_progress: bool = False,
        report_interval: float = 10.0,
    ) -> None:
        self.x0 = np.asarray(x0, dtype=float).copy()
        self.names = tuple(names or (f"x[{i}]" for i in range(self.x0.size)))
        if len(self.names) != self.x0.size:
            raise ValueError("names must contain one entry per decision variable")
        self.bounds = bounds
        self.scales = np.ones_like(self.x0) if scales is None else np.asarray(scales, dtype=float)
        if self.scales.shape != self.x0.shape:
            raise ValueError("scales must have the same shape as x0")
        if np.any(~np.isfinite(self.scales)) or np.any(self.scales <= 0.0):
            raise ValueError("scales must be finite and positive")
        self.metadata = dict(metadata or {})
        # A single residual or Jacobian evaluation is minutes of silence on a
        # production deck; the heartbeat says which one is running and for how
        # long, so a slow linear solve is distinguishable from a hang.
        self.evaluation_progress = bool(evaluation_progress)
        self.report_interval = float(report_interval)

        self._fun = fun
        self._grad = grad
        self._value_and_grad = value_and_grad
        self._residual = residual
        self._residual_jac = residual_jac
        self._residual_and_jac = residual_and_jac
        self._jax_fun = jax_fun
        self._jax_value_and_grad = jax_value_and_grad
        self._jax_residual = jax_residual
        self._jax_residual_jac = jax_residual_jac

        if fun is None and value_and_grad is None and residual is None and residual_and_jac is None:
            raise ValueError("provide a scalar objective, residual function, or combined callable")
        self._lock = RLock()
        self._vg_cache: tuple[tuple[Any, ...], tuple[float, np.ndarray]] | None = None
        self._rj_cache: tuple[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] | None = None
        # Filled in by :meth:`subproblem` on the view it returns, and left
        # ``None``/empty on a problem that is not one.
        self.parent: "FunctionProblem | None" = None
        self.embed: Callable[[Array], np.ndarray] | None = None
        self.free_indices: np.ndarray | None = None
        self.frozen_indices: np.ndarray | None = None
        self.frozen_names: tuple[str, ...] = ()
        self.frozen_values: np.ndarray | None = None

    @property
    def dof_names(self) -> tuple[str, ...]:
        """Ordered names corresponding one-to-one with entries of a decision vector."""
        return self.names

    @classmethod
    def from_functions(cls, x0: Array, **kwargs: Any) -> "FunctionProblem":
        """Build a problem from user-supplied x-level callables."""
        return cls(x0, **kwargs)

    @staticmethod
    def _x(x: Array) -> np.ndarray:
        return np.asarray(x, dtype=float)

    @staticmethod
    def _key(x: np.ndarray) -> tuple[Any, ...]:
        contiguous = np.ascontiguousarray(x)
        return contiguous.shape, contiguous.dtype.str, contiguous.tobytes()

    def value_and_grad(self, x: Array) -> tuple[float, np.ndarray]:
        """Return scalar value and gradient for SciPy ``jac=True``."""
        xh = self._x(x)
        key = self._key(xh)
        with self._lock:
            if self._vg_cache is not None and self._vg_cache[0] == key:
                value, gradient = self._vg_cache[1]
                return value, gradient.copy()
            if self._value_and_grad is not None:
                value, gradient = self._value_and_grad(xh)
            elif self._fun is not None and self._grad is not None:
                value, gradient = self._fun(xh), self._grad(xh)
            elif self._residual_and_jac is not None:
                residual, jacobian = self._residual_and_jac(xh)
                residual = np.asarray(residual, dtype=float).ravel()
                jacobian = np.asarray(jacobian, dtype=float)
                value = 0.5 * float(residual @ residual)
                gradient = jacobian.T @ residual
            else:
                raise AttributeError("this problem does not provide a scalar gradient")
            pair = float(np.asarray(value)), np.asarray(gradient, dtype=float).reshape(self.x0.shape)
            self._vg_cache = key, pair
            return pair[0], pair[1].copy()

    # SciPy commonly calls this form ``fun_and_grad``; JAX uses
    # ``value_and_grad``.  Both names intentionally share one implementation.
    fun_and_grad = value_and_grad

    def fun(self, x: Array) -> float:
        """Return the scalar objective value."""
        if self._fun is not None:
            return float(np.asarray(self._fun(self._x(x))))
        if self._value_and_grad is not None or self._grad is not None:
            return self.value_and_grad(x)[0]
        residual = self.residual(x)
        return 0.5 * float(residual @ residual)

    def grad(self, x: Array) -> np.ndarray:
        """Return the scalar objective gradient."""
        return self.value_and_grad(x)[1]

    def residual_and_jac(self, x: Array) -> tuple[np.ndarray, np.ndarray]:
        """Return the residual vector and its Jacobian."""
        xh = self._x(x)
        key = self._key(xh)
        with self._lock:
            if self._rj_cache is not None and self._rj_cache[0] == key:
                residual, jacobian = self._rj_cache[1]
                return residual.copy(), jacobian.copy()
            if self._residual_and_jac is not None:
                residual, jacobian = self._residual_and_jac(xh)
            elif self._residual is not None and self._residual_jac is not None:
                residual, jacobian = self._residual(xh), self._residual_jac(xh)
            else:
                raise AttributeError("this problem does not provide a residual Jacobian")
            pair = (np.asarray(residual, dtype=float).ravel(),
                    np.asarray(jacobian, dtype=float))
            if pair[1].shape != (pair[0].size, self.x0.size):
                raise ValueError(
                    "residual Jacobian must have shape "
                    f"({pair[0].size}, {self.x0.size}), got {pair[1].shape}"
                )
            self._rj_cache = key, pair
            return pair[0].copy(), pair[1].copy()

    def _timed(self, action: str, function: Callable[[], Any]) -> Any:
        """Run one optimizer evaluation under the elapsed-time heartbeat."""
        if not self.evaluation_progress:
            return function()
        return _run_with_progress(
            function, action=action, complete=f"{action} done",
            progress=True, report_interval=self.report_interval,
            announce=False)

    def residual(self, x: Array) -> np.ndarray:
        """Return the residual vector."""
        if self._residual_and_jac is not None:
            return self.residual_and_jac(x)[0]
        function = self._residual
        if function is not None:
            return self._timed(
                "residual",
                lambda: np.asarray(function(self._x(x)), dtype=float).ravel())
        raise AttributeError("this problem does not provide residuals")

    def residual_jac(self, x: Array) -> np.ndarray:
        """Return the residual Jacobian."""
        if self._residual_and_jac is not None:
            return self.residual_and_jac(x)[1]
        function = self._residual_jac
        if function is not None:
            jacobian = np.asarray(
                self._timed("Jacobian", lambda: function(self._x(x))),
                dtype=float)
            if jacobian.shape[1:] != (self.x0.size,):
                raise ValueError(
                    "residual Jacobian must have one column per decision variable"
                )
            return jacobian
        raise AttributeError("this problem does not provide a residual Jacobian")

    def J(self, x: Array) -> float:
        """SIMSOPT-style alias for :meth:`fun`."""
        return self.fun(x)

    def dJ(self, x: Array) -> np.ndarray:
        """SIMSOPT-style alias for :meth:`grad`."""
        return self.grad(x)

    def jax_fun(self, x: Array) -> Array:
        """Return the traceable scalar objective."""
        if self._jax_fun is not None:
            return self._jax_fun(x)
        if self._jax_value_and_grad is not None:
            return self._jax_value_and_grad(x)[0]
        raise AttributeError("this problem does not provide a JAX scalar objective")

    def jax_value_and_grad(self, x: Array) -> tuple[Array, Array]:
        """Return the traceable scalar value and gradient."""
        if self._jax_value_and_grad is None:
            raise AttributeError("this problem does not provide a JAX value-and-gradient")
        return self._jax_value_and_grad(x)

    def jax_residual(self, x: Array) -> Array:
        """Return the traceable residual vector."""
        if self._jax_residual is None:
            raise AttributeError("this problem does not provide JAX residuals")
        return self._jax_residual(x)

    def jax_residual_jac(self, x: Array) -> Array:
        """Return the traceable residual Jacobian."""
        if self._jax_residual_jac is None:
            raise AttributeError("this problem does not provide a JAX residual Jacobian")
        return self._jax_residual_jac(x)

    def evaluate(self, x: Array, *, derivatives: bool = True) -> Evaluation:
        """Evaluate available scalar and residual quantities at ``x``."""
        xh = self._x(x).copy()
        value = gradient = residual = jacobian = None
        if self._fun is not None or self._value_and_grad is not None:
            if derivatives and (self._value_and_grad is not None or self._grad is not None):
                value, gradient = self.value_and_grad(xh)
            else:
                value = self.fun(xh)
        if self._residual is not None or self._residual_and_jac is not None:
            if derivatives and (self._residual_and_jac is not None or self._residual_jac is not None):
                residual, jacobian = self.residual_and_jac(xh)
            else:
                residual = self.residual(xh)
            if value is None:
                value = 0.5 * float(residual @ residual)
                if derivatives and jacobian is not None:
                    gradient = jacobian.T @ residual
        return Evaluation(
            x=xh, value=value, gradient=gradient,
            residual=residual, jacobian=jacobian,
        )

    def compile_residual_and_jacobian(
        self,
        x: Array | None = None,
        *,
        progress: bool = True,
        report_interval: float = 10.0,
        stream: Any = None,
    ) -> Evaluation:
        """Compile and cache the least-squares residual and Jacobian.

        This call is optional: an optimizer compiles on its first evaluation
        if it is omitted.  Calling it explicitly provides elapsed-time output
        during a potentially long first JAX compilation.  Later calls at the
        same ``x`` use the normal one-entry problem cache.
        """
        xh = self.x0.copy() if x is None else self._x(x).copy()

        def compile_callables() -> Evaluation:
            residual, jacobian = self.residual_and_jac(xh)
            return Evaluation(
                x=xh,
                value=0.5 * float(residual @ residual),
                gradient=jacobian.T @ residual,
                residual=residual,
                jacobian=jacobian,
            )

        return _run_with_progress(
            compile_callables,
            action="Compiling residual and Jacobian (first call may take a minute)",
            complete="Residual and Jacobian ready",
            progress=progress,
            report_interval=report_interval,
            stream=stream,
        )

    def _embedding(
        self, active: Any, x: Array | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[Array], np.ndarray]]:
        """Resolve ``active`` and build the frozen-dof embedding for it.

        Returns ``(base, free, frozen, embed)``: the full vector the frozen
        variables are held at, the positions that stay free, the positions
        that are held, and ``embed(u) -> full x``.
        """
        base = (self.x0 if x is None else self._x(x)).astype(float).copy()
        if base.shape != self.x0.shape:
            raise ValueError(
                f"x must have shape {self.x0.shape}, got {base.shape}")
        entries = list(active)
        if any(isinstance(entry, str) for entry in entries):
            if not all(isinstance(entry, str) for entry in entries):
                raise TypeError("active must be all names or all indices")
            lookup: dict[str, int] = {}
            for position, name in enumerate(self.names):
                lookup.setdefault(name, position)
            missing = [name for name in entries if name not in lookup]
            if missing:
                raise ValueError(
                    "not decision variables of this problem: "
                    + ", ".join(map(str, missing)))
            free = np.array([lookup[name] for name in entries], dtype=int)
        else:
            free = np.asarray(entries, dtype=int).reshape(-1)
        if free.size and (free.min() < 0 or free.max() >= self.x0.size):
            raise ValueError("active contains an out-of-range index")
        if np.unique(free).size != free.size:
            raise ValueError("active repeats a degree of freedom")
        frozen = np.setdiff1d(np.arange(self.x0.size, dtype=int), free)

        def embed(u: Array) -> np.ndarray:
            values = np.asarray(u, dtype=float).reshape(-1)
            if values.size != free.size:
                raise ValueError(
                    f"expected {free.size} free variables, got {values.size}")
            full = base.copy()
            full[free] = values
            return full

        return base, free, frozen, embed

    def subproblem(
        self,
        active: Sequence[int] | Sequence[str],
        *,
        x: Array | None = None,
    ) -> "FunctionProblem":
        """Return a problem over a subset of these decision variables.

        Every variable outside ``active`` is held at its value in ``x``
        (default :attr:`x0`), and the returned problem's decision vector is
        the ``active`` sub-vector, in the order given.  The callables,
        compiled graphs, caches, and metadata of *this* problem are reused
        unchanged: a sub-problem is a view, not a rebuild.

        This is what a ``max_mode`` continuation ladder wants.  Building a
        fresh problem per stage retraces and recompiles every graph, because
        the decision vector changes length even when the underlying
        equilibrium arrays do not; one problem built at the largest stage,
        sub-setted per stage, compiles once.

        Parameters
        ----------
        active:
            The variables that stay free, given as positions in this
            problem's decision vector or as entries of :attr:`dof_names`.
            Duplicates and out-of-range entries are rejected.
        x:
            The full vector the frozen variables are held at.  The default
            is this problem's :attr:`x0`; a continuation stage passes the
            previous stage's result, embedded with :meth:`embed`.

        Returns
        -------
        A problem of this class whose ``x0``, ``scales``, ``bounds``, and
        ``dof_names`` are the ``active`` entries of this one's.  It carries
        ``embed``, ``free_indices``, ``frozen_indices``, ``frozen_names``,
        and ``parent`` so a caller can map a sub-vector back to a full one
        and check what was held.
        """
        base, free, frozen, embed = self._embedding(active, x)
        sub = type(self)(base[free], **self._subproblem_kwargs(base, free, embed))
        return self._as_view(sub, base, free, frozen, embed)

    def _as_view(
        self, sub: Any, base: np.ndarray, free: np.ndarray, frozen: np.ndarray,
        embed: Callable[[Array], np.ndarray],
    ) -> Any:
        """Record on ``sub`` which variables it frees and which it holds."""
        sub.embed = embed
        sub.parent = self
        sub.free_indices = free
        sub.frozen_indices = frozen
        sub.frozen_names = tuple(self.names[position] for position in frozen)
        sub.frozen_values = base[frozen].copy()
        return sub

    def _subproblem_kwargs(
        self, base: np.ndarray, free: np.ndarray,
        embed: Callable[[Array], np.ndarray],
    ) -> dict[str, Any]:
        """Constructor arguments for :meth:`subproblem`'s frozen-dof view."""

        def host(function: HostFun | None) -> HostFun | None:
            return None if function is None else (lambda u: function(embed(u)))

        def columns(function: HostFun | None) -> HostFun | None:
            if function is None:
                return None

            def sliced(u: Array) -> Any:
                return np.asarray(function(embed(u)), dtype=float)[..., free]

            return sliced

        def gradient(function: HostFun | None) -> HostFun | None:
            if function is None:
                return None

            def restricted(u: Array) -> Any:
                return np.asarray(function(embed(u)), dtype=float).reshape(-1)[free]

            return restricted

        def pair(function: HostFun | None) -> HostFun | None:
            if function is None:
                return None

            def both(u: Array) -> Any:
                first, second = function(embed(u))
                return first, np.asarray(second, dtype=float)[..., free]

            return both

        def scalar_pair(function: HostFun | None) -> HostFun | None:
            if function is None:
                return None

            def both(u: Array) -> Any:
                value, grad = function(embed(u))
                return value, np.asarray(grad, dtype=float).reshape(-1)[free]

            return both

        def traced(u: Array) -> Array:
            import jax.numpy as jnp  # local: keep this module importable bare

            return jnp.asarray(base).at[free].set(u)

        def jax_value(function: Any) -> Any:
            return None if function is None else (lambda u: function(traced(u)))

        def jax_columns(function: Any) -> Any:
            return (None if function is None
                    else (lambda u: function(traced(u))[..., free]))

        def jax_scalar_pair(function: Any) -> Any:
            if function is None:
                return None

            def both(u: Array) -> Any:
                value, grad = function(traced(u))
                return value, grad.reshape(-1)[free]

            return both

        return dict(
            fun=host(self._fun),
            grad=gradient(self._grad),
            value_and_grad=scalar_pair(self._value_and_grad),
            residual=host(self._residual),
            residual_jac=columns(self._residual_jac),
            residual_and_jac=pair(self._residual_and_jac),
            jax_fun=jax_value(self._jax_fun),
            jax_value_and_grad=jax_scalar_pair(self._jax_value_and_grad),
            jax_residual=jax_value(self._jax_residual),
            jax_residual_jac=jax_columns(self._jax_residual_jac),
            names=[self.names[position] for position in free],
            bounds=_slice_bounds(self.bounds, free, self.x0.size),
            scales=self.scales[free],
            metadata=self.metadata,
            evaluation_progress=self.evaluation_progress,
            report_interval=self.report_interval,
        )

    def nonlinear_constraint(self, lower, upper, *, scales=1.0):
        """Bound residual values in their original units, with analytic derivatives.

        Return a SciPy NonlinearConstraint for ``opt.minimize(problem, ...)``
        or SciPy. Positive row scales condition values, bounds and Jacobians
        together without changing the feasible set.
        """
        return _nonlinear_constraint(self.residual, self.residual_jac, lower, upper, scales)

    def with_acceptance(self, on_accept):
        """Compose a host objective with an accepted-state owner.

        Return a view for ``opt.minimize``. ``on_accept(x)`` runs only after
        optimizer acceptance, never on trial evaluations. It must either
        commit successfully or raise without changing its accepted state.
        This is useful when adding coil costs/coordinates to a stateful VMEC
        objective; x0 must map to that owner's already-accepted state.
        Scales, constraints and callbacks retain original units.
        Ordinary FunctionProblem instances remain stateless and unchanged.
        """
        if getattr(self, "accept_x", None) is not None:
            raise ValueError("this problem already owns accepted-state promotion")
        return _AcceptedFunctionProblem(self, on_accept)

    def compile_value_and_gradient(
        self,
        x: Array | None = None,
        *,
        progress: bool = True,
        report_interval: float = 10.0,
        stream: Any = None,
    ) -> Evaluation:
        """Compile and cache the scalar value and gradient.

        This optional call makes the first JAX compilation visible before
        BFGS, L-BFGS-B, Adam, or another gradient optimizer starts.
        """
        xh = self.x0.copy() if x is None else self._x(x).copy()

        def compile_callables() -> Evaluation:
            value, gradient = self.value_and_grad(xh)
            return Evaluation(x=xh, value=value, gradient=gradient)

        return _run_with_progress(
            compile_callables,
            action="Compiling value and gradient (first call may take a minute)",
            complete="Value and gradient ready",
            progress=progress,
            report_interval=report_interval,
            stream=stream,
        )


class VmecProblem(FunctionProblem):
    """A :class:`FunctionProblem` backed by a VMEX equilibrium solve.

    Adds the maps between the optimizer's decision vector and VMEC objects:
    the input deck, the converged equilibrium, and the boundary coefficient
    arrays.  It keeps the same optimizer contract as its base class, so the
    same methods go to SciPy, JAXopt, or Optax unchanged, and it enriches
    :meth:`evaluate` with the solve and adjoint status of the underlying
    equilibrium.

    Build one with :meth:`from_tuples`, :meth:`from_loss`, or
    :meth:`from_input` rather than calling this constructor: they route
    through ``vmex.core.optimize.make_problem``, which is what assembles the
    four callables below along with the objective, the derivative lane, the
    degree-of-freedom names, and the metadata.

    Parameters
    ----------
    *args:
        Forwarded positionally to :class:`FunctionProblem`; in practice the
        decision vector ``x0``.
    **kwargs:
        Forwarded to :class:`FunctionProblem`: the objective callables,
        ``names``, ``bounds``, ``scales``, and ``metadata``.
    input_from_x:
        Required.  ``x -> VmecInput``: a new input deck carrying the
        boundary coefficients — and the current degrees of freedom, when
        the problem parameterizes them — of this decision vector.  Nothing
        is solved.
    x_from_input:
        Required.  The inverse, ``VmecInput -> array``: the decision vector
        that reproduces a given deck, which is the normal starting point of
        a continuation stage.  :meth:`x_from_input` rejects a result whose
        shape differs from ``x0``.
    equilibrium_from_x:
        Optional ``x -> Equilibrium``, the converged equilibrium at ``x``.
        Implicit problems return the accepted state the objective already
        computed rather than cold-solving the boundary again, which matters
        for strongly shaped boundaries whose cold axis guess can produce a
        sign-changing initial Jacobian.  ``None`` makes
        :meth:`equilibrium_from_x` raise :exc:`AttributeError`.  A callable
        that accepts a ``newton_iterations`` keyword receives it only when
        the caller asks for something other than the default 10, so a
        closure without that keyword still works.
    boundary_from_x:
        Optional ``x -> tuple of arrays``, the traceable boundary
        coefficients: ``(rbc, zbs)`` for a stellarator-symmetric input and
        ``(rbc, zbs, rbs, zbc)`` when ``lasym``.  Each is a full dense
        INDATA-layout array of shape ``(2 * ntor + 1, mpol)`` indexed
        ``[n + ntor, m]`` in metres — not the trimmed decision vector — and
        is a JAX array, so it composes with coil or surface objectives
        under :func:`jax.grad`.  ``None`` makes :meth:`boundary_from_x`
        raise :exc:`AttributeError`.
    """

    def __init__(
        self,
        *args: Any,
        input_from_x: Callable[[Array], Any],
        x_from_input: Callable[[Any], Array],
        equilibrium_from_x: Callable[..., Any] | None = None,
        boundary_from_x: Callable[[Array], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._input_from_x = input_from_x
        self._x_from_input = x_from_input
        self._equilibrium_from_x = equilibrium_from_x
        self._boundary_from_x = boundary_from_x

    @classmethod
    def from_tuples(
        cls,
        inp: Any,
        objective_terms: Sequence[tuple[Callable[..., Any], Any, float]],
        **kwargs: Any,
    ) -> "VmecProblem":
        """Build a VMEC least-squares problem from weighted objective tuples.

        The README entry point.  Each tuple is ``(function, target,
        weight)`` and contributes one or more rows to a single residual
        vector; the rows of all terms are concatenated in the order given
        and the scalar cost is ``0.5 * r @ r``.  Named row ranges are
        recorded in the problem metadata, which is what lets
        :class:`~vmex.core.monitoring.OptimizationMonitor` report per-term
        costs without re-solving anything.

        Parameters
        ----------
        inp:
            The starting :class:`~vmex.core.input.VmecInput`.  Its boundary
            supplies the initial decision vector and its resolution and
            profiles are held fixed apart from the parameterized degrees of
            freedom.
        objective_terms:
            The ``(function, target, weight)`` triples.

            ``function`` is normally a traceable
            ``function(state, runtime) -> scalar or vector``, evaluated on
            the converged equilibrium — the two arguments are also spelled
            ``(equilibrium_state, solver_context)``.  An objective *object*
            exposing a ``residuals_state`` method may be passed instead
            (whole instance or bound method), in which case its full
            pointwise residual vector becomes this term's rows.  Under
            ``derivative_method="finite_difference"`` a one-argument host
            callable taking the whole ``Equilibrium`` is accepted too; the
            implicit lane rejects it, since it cannot be traced.

            ``target`` is the value the term is driven toward, coerced with
            :func:`float`, so it must be scalar even when ``function``
            returns a vector — the same target is then subtracted from
            every row.

            ``weight`` is a non-negative scalar, or a one-dimensional array
            with one entry per residual row of that term.  Under the
            default ``weight_semantics="cost"`` it multiplies the squared
            cost, so the row is ``sqrt(weight) * (function - target)``;
            with ``weight_semantics="residual"`` the row is
            ``weight * (function - target)`` and a negative entry is then
            allowed.
        **kwargs:
            Passed through to ``vmex.core.optimize.make_problem``: which
            boundary modes vary (``max_mode``, ``vary_major_radius``,
            ``current_dofs``), the derivative lane
            (``derivative_method``, ``implicit_jacobian_method``,
            ``jacobian_batch_size``), the forward solve controls, the
            variable scaling (``use_ess``, ``ess_alpha``, ``bounds``), and
            ``weight_semantics``.

        Returns
        -------
        A :class:`VmecProblem` whose ``residual``/``residual_jac`` pair,
        ``x0``, and ``scales`` are ready for
        :func:`scipy.optimize.least_squares`.  A non-finite or empty
        residual at the initial point raises :exc:`FloatingPointError`
        rather than starting an optimization that cannot recover.
        """
        from .optimize import make_problem
        return make_problem(inp, objective_terms=objective_terms, problem_class=cls, **kwargs)

    @classmethod
    def from_loss(cls, inp: Any, loss: Callable[..., Any], **kwargs: Any) -> "VmecProblem":
        """Build a VMEC scalar problem from a traceable state/runtime loss.

        Parameters
        ----------
        inp:
            The starting :class:`~vmex.core.input.VmecInput`, as for
            :meth:`from_tuples`.
        loss:
            ``loss(state, runtime) -> scalar``, evaluated on the converged
            equilibrium and already carrying its own weights.  It must
            return a single value: a vector-valued objective belongs in
            :meth:`from_tuples`, or must be reduced here explicitly.
            Unlike an ``objective_terms`` entry it is used exactly as
            written — an object's ``residuals_state`` is never substituted
            for it.
        **kwargs:
            As for :meth:`from_tuples`.

        Returns
        -------
        A :class:`VmecProblem` exposing only the scalar lane —
        :meth:`~FunctionProblem.fun`, :meth:`~FunctionProblem.grad`, and
        their traceable counterparts — for a gradient optimizer such as
        BFGS, L-BFGS-B, or Adam.  It provides no residual or Jacobian.
        """
        from .optimize import make_problem
        return make_problem(inp, loss=loss, problem_class=cls, **kwargs)

    @classmethod
    def from_input(cls, inp: Any, **kwargs: Any) -> "VmecProblem":
        """Parameterize an input for field VJPs without defining an objective.

        Builds the same machinery as :meth:`from_loss` around an
        identically zero loss, so there is nothing to minimize.  Use it when
        what you want is the parameterization itself: the decision vector
        and its names, :meth:`input_from_x` and :meth:`boundary_from_x`, and
        the differentiable :meth:`interior_field` and :meth:`exterior_field`
        with exact VJPs in these degrees of freedom.  ``inp`` and
        ``**kwargs`` are as for :meth:`from_tuples`.
        """
        from .optimize import make_problem

        return make_problem(
            inp, loss=lambda _state, _runtime: 0.0,
            problem_class=cls, **kwargs)

    def subproblem(  # type: ignore[override]
        self,
        active: Sequence[int] | Sequence[str] | None = None,
        *,
        max_mode: int | None = None,
        x: Array | None = None,
    ) -> "VmecProblem":
        """Return this problem restricted to one continuation stage.

        Pass ``max_mode`` to free exactly the boundary harmonics a problem
        built at that ``max_mode`` would have had — plus every non-boundary
        degree of freedom, such as the current dofs, which no stage freezes
        — and hold every higher harmonic at its value in ``x``.  Pass
        ``active`` instead to choose the free variables directly, as for
        :meth:`FunctionProblem.subproblem`.

        A ``max_mode`` ladder built this way compiles once.  Building a
        fresh problem per stage does not: the decision vector changes
        length, so every jitted residual, Jacobian and predictor graph is a
        cache miss, even though ``MINIMUM_MPOL``-style examples keep the
        equilibrium resolution — and therefore every array shape inside
        the solve — identical from stage to stage.  A stage that *does*
        raise the resolution still needs its own problem; group the stages
        that share one.

        The returned problem maps its sub-vector through the same
        :meth:`input_from_x`, :meth:`x_from_input`, :meth:`equilibrium_from_x`
        and :meth:`boundary_from_x` as this one, so the deck and equilibrium
        it produces carry the frozen harmonics unchanged.
        """
        if (active is None) == (max_mode is None):
            raise TypeError("pass exactly one of active= or max_mode=")
        if max_mode is not None:
            active = self.stage_dof_names(max_mode)
        base, free, frozen, embed = self._embedding(active, x)
        kwargs = self._subproblem_kwargs(base, free, embed)
        if max_mode is not None:
            # The stage's own max_mode, so a further subproblem() of this one
            # cuts from the right ladder rung; everything else -- the config,
            # the shared solve-counter holder, the residual slices -- is the
            # parent's, because it is the same compiled problem.
            kwargs["metadata"] = {**self.metadata, "max_mode": int(max_mode)}
        kwargs["input_from_x"] = lambda u: self._input_from_x(embed(u))
        kwargs["x_from_input"] = lambda deck: np.asarray(
            self._x_from_input(deck), dtype=float).reshape(-1)[free]
        if self._equilibrium_from_x is not None:
            source = self._equilibrium_from_x
            kwargs["equilibrium_from_x"] = (
                lambda u, **kw: source(embed(u), **kw))
        if self._boundary_from_x is not None:
            boundary = self._boundary_from_x
            kwargs["boundary_from_x"] = lambda u: boundary(embed(u))
        return self._as_view(
            VmecProblem(base[free], **kwargs), base, free, frozen, embed)

    def stage_dof_names(self, max_mode: int) -> list[str]:
        """Names of the degrees of freedom free at continuation ``max_mode``.

        The boundary harmonics a problem built at ``max_mode`` would carry,
        in this problem's own order, followed by every degree of freedom
        that is not a boundary harmonic of this problem (the optional
        current dofs), which a ``max_mode`` stage does not freeze.
        """
        from .optimize import boundary_dof_names

        deck = self.metadata.get("input")
        built_at = self.metadata.get("max_mode")
        if deck is None or built_at is None:
            raise AttributeError(
                "this problem does not record the input deck and max_mode a "
                "continuation stage would be cut from; pass active= instead")
        if int(max_mode) > int(built_at):
            raise ValueError(
                f"max_mode={int(max_mode)} exceeds the max_mode this problem "
                f"was built at ({int(built_at)})")
        vary = bool(self.metadata.get("vary_major_radius", False))
        names = list(boundary_dof_names(
            deck, int(max_mode), vary_major_radius=vary))
        boundary = len(boundary_dof_names(
            deck, int(built_at), vary_major_radius=vary))
        return names + list(self.names[boundary:])

    def input_from_x(self, x: Array) -> Any:
        """Return a new :class:`VmecInput` containing decision vector ``x``."""
        return self._input_from_x(self._x(x))

    def x_from_input(self, inp: Any) -> np.ndarray:
        """Return this problem's decision vector for ``inp``.

        This is the inverse of :meth:`input_from_x` for the boundary and any
        optional current degrees of freedom selected when the problem was
        constructed.  It is the normal continuation-stage starting vector.
        """
        x = np.asarray(self._x_from_input(inp), dtype=float)
        if x.shape != self.x0.shape:
            raise ValueError(
                f"input produced decision-vector shape {x.shape}, "
                f"expected {self.x0.shape}"
            )
        return x

    def equilibrium_from_x(self, x: Array, *, newton_iterations: int = 10) -> Any:
        """Return the converged equilibrium evaluated at ``x``.

        Implicit problems reuse the accepted optimizer state instead of
        cold-solving the optimized boundary again.  This matters for strongly
        shaped boundaries whose cold magnetic-axis guess may have a
        sign-changing initial Jacobian.
        """
        if self._equilibrium_from_x is None:
            raise AttributeError("this problem does not provide equilibria")
        if int(newton_iterations) == 10:
            return self._equilibrium_from_x(self._x(x))
        return self._equilibrium_from_x(
            self._x(x), newton_iterations=int(newton_iterations))

    def boundary_from_x(self, x: Array) -> Any:
        """Return traceable boundary coefficient arrays for decision vector ``x``."""
        if self._boundary_from_x is None:
            raise AttributeError("this problem does not provide boundary arrays")
        return self._boundary_from_x(x)

    def restart_from(self, equilibrium):
        """Seed the next host solve from a compatible equilibrium.

        This changes only the warm initial guess; it neither evaluates nor
        accepts a point. Resolution and symmetry must match. Any pending
        perturbation predictor is discarded so it cannot override this seed.
        Use ``with_accepted_state`` to apply this policy to every host trial.
        """
        cfg = self.metadata.get("config")
        if cfg is None or not cfg.hot_restart:
            raise ValueError("restart seeding requires an implicit problem with hot_restart=True")
        for name in ("mpol", "ntor", "nfp", "lasym"):
            if getattr(equilibrium.inp, name) != getattr(cfg.inp, name):
                raise ValueError(f"restart equilibrium differs in {name}")
        from .restart import restart_state
        from . import implicit as imp

        state = restart_state(equilibrium.state, cfg.inp)
        fields = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
        shape = np.shape(state.R_cos)
        if (len(shape) != 2 or shape[0] != cfg.resolution.ns
                or any(np.shape(getattr(state, name)) != shape
                       or not np.all(np.isfinite(getattr(state, name))) for name in fields)):
            raise ValueError("restart requires finite state arrays at the problem resolution")
        imp._HOT_CACHE[cfg] = state
        imp._PERTURB_SEED.pop(cfg, None)
        self.metadata.get("holder", {}).update(lin=None)

    def with_accepted_state(self):
        """Return a host view seeded from its last optimizer-accepted equilibrium.

        ``view.accepted`` exposes parameters and equilibrium; ``accept_x``
        promotes a usable candidate. Trial values, gradients and residuals do
        not advance that anchor. Use ``opt.minimize(view, ...)`` directly or
        compose a larger objective with ``FunctionProblem.with_acceptance``.
        The source and view share caches and must not run concurrently.
        JAX-traced optimization is deliberately not exposed by this host view.
        """
        if isinstance(self, _AcceptedVmecProblem):
            return self
        return _AcceptedVmecProblem(self)

    def jax_objective_from_state(
        self,
        x: Array,
        extra_costs: Callable[[Array, Any], Array],
        *,
        n_extra_terms: int,
    ) -> tuple[Array, tuple[Array, Array]]:
        """Combine the VMEX least-squares cost with state-dependent costs.

        ``extra_costs(state, runtime)`` returns one already-weighted scalar
        cost per added objective term. The auxiliary result contains the VMEX
        residual rows and those added costs, ready to pass as auxiliary data to
        :func:`jax.value_and_grad`. Failed equilibrium trials receive the same
        smooth finite rejection cost as the base problem, so driver scripts do
        not need their own accepted/rejected branches.
        """
        import jax.numpy as jnp

        state_runtime_status = self.metadata.get("jax_state_runtime_status")
        residual_from_state = self.metadata.get("jax_residual_from_state")
        failure_value = self.metadata.get("jax_failure_value")
        residual_size = self.metadata.get("residual_size")
        if any(item is None for item in (
                state_runtime_status, residual_from_state, failure_value,
                residual_size)):
            raise AttributeError(
                "state-composed objectives require an implicit VMEC "
                "least-squares problem")
        if n_extra_terms < 1:
            raise ValueError("n_extra_terms must be positive")

        state_runtime_status = cast(Callable[..., Any], state_runtime_status)
        residual_from_state = cast(Callable[..., Any], residual_from_state)
        failure_value = cast(Callable[..., Any], failure_value)
        residual_size = int(residual_size)

        state, runtime, status = state_runtime_status(x)

        def accepted(_):
            residual = residual_from_state(state, runtime)
            costs = jnp.atleast_1d(extra_costs(state, runtime))
            if costs.shape != (n_extra_terms,):
                raise ValueError(
                    f"extra_costs returned shape {costs.shape}, expected "
                    f"({n_extra_terms},)")
            return (0.5 * jnp.vdot(residual, residual) + jnp.sum(costs),
                    (residual, costs))

        def rejected(_):
            return (failure_value(x),
                    (jnp.zeros(residual_size), jnp.zeros(n_extra_terms)))

        return _branch_on_status(status, accepted, rejected)

    def jax_extra_costs_from_state(
        self,
        x: Array,
        extra_costs: Callable[[Array, Any], Array],
        *,
        n_extra_terms: int,
    ) -> tuple[Array, Array]:
        """Evaluate additive state-dependent costs only at valid VMEC trials.

        This is the split-compilation counterpart of
        :meth:`jax_objective_from_state`. It returns zero extra cost at a
        rejected trial, leaving the base problem to supply its certified
        rejection wall. Splitting a large virtual-casing or coil graph from
        the VMEC objective substantially lowers peak XLA compilation memory.
        """
        import jax.numpy as jnp

        state_runtime_status = self.metadata.get("jax_state_runtime_status")
        if state_runtime_status is None:
            raise AttributeError(
                "state-dependent costs require an implicit VMEC problem")
        if n_extra_terms < 1:
            raise ValueError("n_extra_terms must be positive")
        state_runtime_status = cast(Callable[..., Any], state_runtime_status)
        state, runtime, status = state_runtime_status(x)

        def accepted(_):
            costs = jnp.atleast_1d(extra_costs(state, runtime))
            if costs.shape != (n_extra_terms,):
                raise ValueError(
                    f"extra_costs returned shape {costs.shape}, expected "
                    f"({n_extra_terms},)")
            return jnp.sum(costs), costs

        return _branch_on_status(
            status, accepted,
            lambda _: (jnp.asarray(0.0), jnp.zeros(n_extra_terms)))

    def jax_quantity_from_state(
        self,
        x: Array,
        quantity: Callable[[Array, Any], Array],
    ) -> tuple[Array, Array]:
        """Evaluate a differentiable floating-point quantity and solve status.

        ``quantity(state, runtime)`` may return any fixed-shape JAX array.
        Rejected equilibrium trials return an array of NaNs with that shape,
        so an invalid state cannot look like a usable diagnostic. The scalar
        status is zero only for an accepted equilibrium.
        """
        import jax.numpy as jnp

        state_runtime_status = self.metadata.get("jax_state_runtime_status")
        if state_runtime_status is None:
            raise AttributeError(
                "state quantities require an implicit VMEX problem")
        state_runtime_status = cast(Callable[..., Any], state_runtime_status)
        state, runtime, status = state_runtime_status(x)
        value = jnp.asarray(quantity(state, runtime))
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("state quantities must have a floating-point dtype")
        return jnp.where(status == 0, value, jnp.full_like(value, jnp.nan)), status

    def exterior_field(
        self,
        x: Array,
        *,
        external_field: Any | None = None,
        external_parameters: Array | None = None,
        external_field_from_parameters: Callable[[Array], Any] | None = None,
        external_dof_names: tuple[str, ...] = (),
        nphi: int | None = None,
        ntheta: int | None = None,
        digits: int = 6,
        accuracy_check: str = "warn",
        levels: tuple[tuple[int, int], ...] | None = None,
        chunk_size: int | str = "auto",
        target_chunk_size: int | str = "auto",
    ) -> Any:
        """Return the exterior field and exact VJPs in this problem's DOFs.

        Query points must lie outside the last closed flux surface and away
        from coil filaments.  The returned field follows the stored-point API:
        ``field.set_points(xyz); field.B(); field.B_vjp(cotangent)``.
        Set the source and target chunk sizes only to cap virtual-casing
        memory; ``"auto"`` is the tuned default.  ``nphi`` and ``ntheta``
        default to the boundary-sized source grid the ``VmecExtender``
        classmethods use, so a high-aspect boundary is not extended on a grid
        four orders too coarse for its ``digits``; pass them to override.
        """
        state_runtime = self.metadata.get("jax_state_runtime")
        inp = self.metadata.get("input")
        if state_runtime is None or inp is None:
            raise AttributeError(
                "this problem does not expose a differentiable equilibrium field")
        from . import virtual_casing as vc
        from .extender import VmecExtender, _source_nphi_for_digits

        parameters = self._x(x)
        chosen = _source_nphi_for_digits(inp, digits)
        source_nphi = chosen if nphi is None else int(nphi)
        source_ntheta = chosen if ntheta is None else int(ntheta)

        def surface_data(p):
            state, runtime = state_runtime(p)
            return vc.surface_field_data_from_state(
                inp, state, runtime=runtime,
                nphi=source_nphi, ntheta=source_ntheta)

        return VmecExtender.from_parameterized_surface_data(
            surface_data, parameters, external_field=external_field,
            external_parameters=external_parameters,
            external_field_from_parameters=external_field_from_parameters,
            external_dof_names=external_dof_names,
            digits=digits, levels=levels, chunk_size=chunk_size,
            target_chunk_size=target_chunk_size, dof_names=self.dof_names,
            accuracy_check=accuracy_check)

    def interior_field(
        self, x: Array, *, newton_iterations: int = 10
    ) -> Any:
        """Return the interior field and exact VJPs in this problem's DOFs."""
        state_runtime = self.metadata.get("jax_state_runtime")
        inp = self.metadata.get("input")
        if state_runtime is None or inp is None:
            raise AttributeError(
                "this problem does not expose a differentiable equilibrium field")
        from .extender import VmecInteriorField

        return VmecInteriorField.from_parameterized_state(
            inp, state_runtime, self._x(x), dof_names=self.dof_names,
            newton_iterations=newton_iterations)

    def surface_field_values(
        self,
        x: Array,
        quantity: str,
        *,
        external_field: Any | None = None,
        nphi: int = 32,
        ntheta: int = 32,
        digits: int = 4,
        precision: Any | None = None,
    ) -> Array:
        """Return ``|B|`` or ``B.n/B`` on a trial boundary for plotting.

        ``B.n/B`` is evaluated on the exterior side using the supplied coil or
        MGRID field plus the plasma-current virtual-casing field. This helper
        keeps optional movie coloring out of optimization driver code; it is
        not used by the objective or optimizer, so it reads a plain forward
        solve when the problem offers one (``host_state_runtime``): the
        derivative anchor of the differentiable lane changes nothing a figure
        can show and cost 83% of each single-stage movie frame.
        """
        import jax.numpy as jnp

        if quantity not in ("absB", "B.n/B"):
            raise ValueError('quantity must be "absB" or "B.n/B"')
        state_runtime = (self.metadata.get("host_state_runtime")
                         or self.metadata.get("jax_state_runtime"))
        inp = self.metadata.get("input")
        if state_runtime is None or inp is None:
            raise AttributeError("surface fields require an implicit VMEC problem")
        from . import virtual_casing as vc

        state, runtime = state_runtime(self._x(x))
        data = vc.surface_field_data_from_state(
            inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
        Bmag = jnp.linalg.norm(data.B_total, axis=0)
        if quantity == "absB":
            return Bmag
        if external_field is None:
            raise ValueError("B.n/B requires external_field")
        interface = vc.PlasmaVacuumInterface.from_surface_data(
            data, digits=digits, precision=precision)
        return interface.bnormal_residual(external_field) / Bmag

    def evaluate(self, x: Array, *, derivatives: bool = True) -> Evaluation:
        """Evaluate and attach VMEC solve/adjoint status diagnostics."""
        evaluation = super().evaluate(x, derivatives=derivatives)
        cfg = self.metadata.get("config")
        if cfg is None:
            return evaluation
        from . import implicit as imp

        # A cached value/Jacobian can be revisited after an unrelated rejected
        # trial.  Confirm that the requested point still maps to the cached
        # converged equilibrium before consulting the process-local last-error
        # slot, otherwise that older failure would incorrectly mark this
        # successful evaluation as failed.
        equilibrium = None
        try:
            equilibrium = self.equilibrium_from_x(evaluation.x)
        except (AttributeError, RuntimeError):
            pass
        else:
            imp._LAST_STATUS_ERROR.pop(cfg, None)

        diagnostics = dict(evaluation.diagnostics)
        stats = imp._SOLVE_STATS.get(cfg)
        if stats is not None:
            diagnostics["solve_stats"] = dict(stats)
        holder = self.metadata.get("holder", {})
        diagnostics["failed_trials"] = int(holder.get("failed_trials", 0))
        diagnostics["derivative_fallbacks"] = int(
            holder.get("derivative_fallbacks", 0)
        )
        if equilibrium is not None:
            result = equilibrium.result
            fsq = float(result.fsqr) + float(result.fsqz) + float(result.fsql)
            ratio = fsq / float(cfg.ftol)
            diagnostics.update(
                fsq=fsq,
                fsq_ratio=ratio,
                max_fsq_ratio=float(cfg.max_fsq_ratio),
                derivative_certified=bool(result.converged or ratio <= cfg.max_fsq_ratio),
            )
            if not diagnostics["derivative_certified"]:
                return replace(
                    evaluation,
                    status="under_converged",
                    message="FSQ exceeds the implicit-derivative threshold",
                    diagnostics=diagnostics,
                )
        error = imp._LAST_STATUS_ERROR.get(cfg)
        if error is None:
            return replace(evaluation, diagnostics=diagnostics)
        diagnostics["exception_type"] = type(error).__name__
        return replace(
            evaluation,
            status="failed_solve",
            message=str(error),
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class _AcceptedPoint:
    parameters: np.ndarray
    equilibrium: Any = None


class _AcceptedIterates:
    def subproblem(self, *args, **kwargs):
        raise NotImplementedError("create the subproblem before adding accepted-state ownership")

    def _init_acceptance(self, on_accept):
        if not callable(on_accept):
            raise TypeError("on_accept must be callable")
        self._on_accept = on_accept
        self._accepted_parameters = self.x0.copy()
        self.accepted_step = 0

    @property
    def accepted(self):
        return _AcceptedPoint(self._accepted_parameters.copy(),
                              getattr(self, "_accepted_equilibrium", None))

    def accept_x(self, x):
        from .errors import TrialRejected

        point = np.asarray(x, dtype=float).copy()
        if point.shape != self.x0.shape or not np.all(np.isfinite(point)):
            raise ValueError("accepted point must be finite and match x0")
        with self._lock:
            if np.array_equal(point, self._accepted_parameters):
                return self.accepted
            value, gradient = self.value_and_grad(point)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise TrialRejected("cannot accept a nonfinite objective or gradient")
            self._on_accept(point.copy())
            self._accepted_parameters = point
            self.accepted_step += 1
            return self.accepted


def _host_view_options(source, wrap):
    return dict(fun=wrap(source.fun), value_and_grad=wrap(source.value_and_grad),
        residual=wrap(source.residual) if source._residual is not None or source._residual_and_jac is not None else None,
        residual_jac=wrap(source.residual_jac) if source._residual_jac is not None or source._residual_and_jac is not None else None,
        names=source.names, bounds=source.bounds, scales=source.scales.copy(), metadata=source.metadata,
        evaluation_progress=source.evaluation_progress, report_interval=source.report_interval)


class _AcceptedFunctionProblem(_AcceptedIterates, FunctionProblem):
    def __init__(self, source, on_accept):
        super().__init__(source.x0, **_host_view_options(source, lambda function: function))
        self._init_acceptance(on_accept)


class _AcceptedVmecProblem(_AcceptedIterates, VmecProblem):
    def __init__(self, source):
        self._source = source
        equilibrium = source.equilibrium_from_x(source.x0)
        self._check_equilibrium(equilibrium)
        source.restart_from(equilibrium)
        self._accepted_equilibrium = equilibrium

        def seeded(function):
            def call(x, **kwargs):
                source.restart_from(self._accepted_equilibrium)
                return function(x, **kwargs)
            return call

        super().__init__(source.x0, input_from_x=source.input_from_x,
            x_from_input=source.x_from_input, boundary_from_x=source.boundary_from_x,
            equilibrium_from_x=seeded(source.equilibrium_from_x),
            **_host_view_options(source, seeded))
        self._init_acceptance(self._promote_equilibrium)

    def _check_equilibrium(self, equilibrium):
        from .errors import TrialRejected

        cfg = self._source.metadata.get("config")
        if cfg is None:
            raise ValueError("accepted-state views require an implicit VMEC problem")
        result = equilibrium.result
        forces = np.asarray([result.fsqr, result.fsqz, result.fsql], dtype=float)
        if (not np.all(np.isfinite(forces)) or np.any(forces < 0)
                or not (result.converged or forces.sum() <= cfg.ftol * cfg.max_fsq_ratio)):
            raise TrialRejected("cannot accept an equilibrium outside the derivative force gate")

    def _promote_equilibrium(self, x):
        from .errors import TrialRejected

        try:
            equilibrium = self.equilibrium_from_x(x)
        except RuntimeError as error:
            raise TrialRejected(str(error)) from error
        self._check_equilibrium(equilibrium)
        # Validate compatibility and seed arrays before changing the anchor.
        self._source.restart_from(equilibrium)
        self._accepted_equilibrium = equilibrium


__all__ = ["Evaluation", "FunctionProblem", "VmecProblem"]
