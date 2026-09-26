Basic API
=========

The API you use daily: the lazily imported ``vmex`` top-level exports
(``import vmex as vj``) plus the three modules behind gradient-based work.
Everything else — the per-module solver internals and the mirror lane — is
in :doc:`advanced`.

Top-level package
-----------------

Every name in ``vmex.__all__`` is listed below by group, each linked to the
module that documents it; this page is an index, not a second copy of every
docstring.

.. automodule:: vmex
   :no-members:

Inputs
------

.. automodule:: vmex.core.input
   :members:

.. automodule:: vmex.core.desc
   :members:

Run directives (``!@VMEX`` comment lines and the JSON ``_vmex`` section) are
execution metadata and never become :class:`~vmex.core.input.VmecInput`
fields; they are parsed and resolved here.  The precedence rule is stated
with the polishing options in :doc:`advanced`.

.. automodule:: vmex.core.run_options
   :members:

Differentiation and optimization
--------------------------------

A converged :class:`vmex.core.optimize.Equilibrium` exposes
``equilibrium.solution`` (the spectral equilibrium arrays) and
``equilibrium.solver_context`` (read-only grids, profiles, and constants).
The shorter ``state`` and ``runtime`` attribute names remain compatible with
existing code; ``runtime`` never means elapsed wall-clock time.

.. automodule:: vmex.core.implicit
   :members:

.. automodule:: vmex.core.optimize
   :members:

The optimizer-neutral callables behind ``vj.VmecProblem.from_tuples`` in the
README: value, residual, and derivative functions with the contracts SciPy,
JAXopt, Optax, and user code consume, and no optimization algorithm of their
own.

Both fixed- and free-boundary host problems can use ``opt.minimize`` and
``problem.nonlinear_constraint``. For fixed-boundary optimization,
``VmecProblem.with_accepted_state()`` returns a view that seeds each trial from
the last accepted equilibrium. ``FunctionProblem.with_acceptance(on_accept)``
connects a composed plasma/coil objective to that owner without promoting trial
evaluations. The free-boundary problem owns this lifecycle directly. Solver
backends remain specific to each formulation.

.. automodule:: vmex.core.problem
   :members:

.. automodule:: vmex.core.monitoring
   :members:

Coil optimization through a free-boundary equilibrium uses the same objective
interface and shared host optimizer, with explicit physical target bands.

.. automodule:: vmex.core.coil_parameters
   :members:

.. automodule:: vmex.core.freeboundary_problem
   :members:

.. automodule:: vmex.core.parallel
   :members:

Outputs
-------

.. automodule:: vmex.core.wout
   :members:

Optional neoclassical diagnostics
---------------------------------

.. automodule:: vmex.core.neoclassical
   :members:

Optional alpha-particle tracing
-------------------------------

.. automodule:: vmex.core.tracing
   :members:
