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

.. automodule:: vmex.core.problem
   :members:

.. automodule:: vmex.core.monitoring
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

Optional helium-ash objective
------------------------------

Requires the optional residence API in `ESSOS #167
<https://github.com/uwplasma/ESSOS/pull/167>`_. Given a callback
``transport_response(state, runtime)`` returning the evaluator's SI keyword
arguments, compose its positive cost with existing least-squares terms:

.. code-block:: python

   from vmex import optimize as opt

   ash = opt.HeliumAshObjective(transport_response, reference_occupation_s,
                                "D/V only; geometry, source and Er fixed")
   rows = opt.residuals_from_tuples(state, runtime, [(ash, 0.0, 1.0)])

The reference is fixed and positive. Derivatives cover only traced callback
dependencies; physical acceptance needs independent qualification.

.. automodule:: vmex.core.ash
   :members:

Optional alpha-particle tracing
-------------------------------

.. automodule:: vmex.core.tracing
   :members:
