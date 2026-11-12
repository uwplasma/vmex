"""One trust-region stage of a boundary-optimization ladder.

:func:`run_stage` wraps :func:`scipy.optimize.least_squares` for any object with ``x0``,
``residual`` and ``residual_jac`` (a :class:`~vmex.core.problem.VmecProblem` stage or a
``FreeMapProblem``).  Every shipped ladder makes the same call; this module holds the choices
that matter when the seed is not tuned:

* The variables are rewritten in step units, ``u = (x - x0) / step``.  SciPy then starts the
  trust region at one step instead of at ``||x0 / x_scale||``, the whole boundary in step
  units, and grows it by doubling after every full step.
* ``max_nfev`` is a ceiling, not a quota.  The stage stops as soon as ``patience`` accepted
  iterations lower the cost by less than 2 % (after at least a quarter of the ceiling), so a
  seed far from the optimum can use the whole ceiling to finish its stage while a stage at its
  local minimum returns early.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = ["run_stage"]


def run_stage(
    problem: Any,
    *,
    max_nfev: int,
    step: Any = 1.0,
    change: Any = None,
    patience: int = 3,
    ftol: float = 1.0e-6,
    xtol: float = 1.0e-10,
    callback: Callable[[Any], None] | None = None,
    verbose: int = 0,
):
    """Minimize ``0.5 |problem.residual(x)|^2`` over one continuation stage.

    ``step`` is the scale of one variable and ``change`` limits the stage to that many steps
    either side of ``problem.x0``; both are a scalar or a per-variable array, and ``change=None``
    leaves the stage unbounded.  Returns the SciPy ``OptimizeResult`` with ``x`` in the
    problem's own units.
    """
    from scipy.optimize import OptimizeResult, least_squares

    x0 = np.asarray(problem.x0, dtype=float)
    step = np.broadcast_to(np.asarray(step, dtype=float), x0.shape)
    costs: list[float] = []

    def monitor(intermediate_result):
        costs.append(float(intermediate_result.cost))
        if callback is not None:  # the callback sees the problem's own coordinates
            seen = OptimizeResult(intermediate_result)
            seen.x = x0 + step * intermediate_result.x
            if "jac" in seen:
                seen.jac = intermediate_result.jac / step
            if "grad" in seen:
                seen.grad = intermediate_result.grad / step
            callback(seen)
        slow = len(costs) > patience and costs[-patience - 1] - costs[-1] < 0.02 * costs[-patience - 1]
        if slow and len(costs) >= max_nfev // 4:
            raise StopIteration

    bound = np.broadcast_to(np.inf if change is None else np.asarray(change, dtype=float), x0.shape)
    result = least_squares(
        lambda u: problem.residual(x0 + step * u), np.zeros_like(x0),
        jac=lambda u: problem.residual_jac(x0 + step * u) * step,
        bounds=(-bound, bound), max_nfev=max_nfev, ftol=ftol, xtol=xtol,
        verbose=verbose, callback=monitor)
    result.x = x0 + step * result.x
    return result
