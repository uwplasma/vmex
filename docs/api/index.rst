API reference
=============

The toroidal production API is :mod:`vmec_jax.core`; the open-field-line API
is :mod:`vmec_jax.mirror`. Modules are grouped as in :doc:`/architecture`.

Inputs and profiles
-------------------

.. automodule:: vmec_jax.core.input
   :members:

.. automodule:: vmec_jax.core.profiles
   :members:

Spectral representation and physics kernels
-------------------------------------------

.. automodule:: vmec_jax.core.fourier
   :members:

.. automodule:: vmec_jax.core.transforms
   :members:

.. automodule:: vmec_jax.core.geometry
   :members:

.. automodule:: vmec_jax.core.fields
   :members:

.. automodule:: vmec_jax.core.forces
   :members:

.. automodule:: vmec_jax.core.residuals
   :members:

Solver
------

.. automodule:: vmec_jax.core.setup
   :members:

.. automodule:: vmec_jax.core.preconditioner
   :members:

.. automodule:: vmec_jax.core.preconditioner_2d
   :members:

.. automodule:: vmec_jax.core.step
   :members:

.. automodule:: vmec_jax.core.solver
   :members:

.. automodule:: vmec_jax.core.multigrid
   :members:

.. automodule:: vmec_jax.core.device
   :members:

Free boundary
-------------

.. automodule:: vmec_jax.core.vacuum
   :members:

.. automodule:: vmec_jax.core.freeboundary
   :members:

.. automodule:: vmec_jax.core.freeboundary_diff
   :members:

.. automodule:: vmec_jax.core.mgrid
   :members:

.. automodule:: vmec_jax.core.coils
   :members:

Differentiation and optimization
--------------------------------

.. automodule:: vmec_jax.core.implicit
   :members:

.. automodule:: vmec_jax.core.optimize
   :members:

Outputs
-------

.. automodule:: vmec_jax.core.wout
   :members:

.. automodule:: vmec_jax.core.nyquist
   :members:

.. automodule:: vmec_jax.core.postprocess
   :members:

.. automodule:: vmec_jax.core.printing
   :members:

.. automodule:: vmec_jax.core.plotting
   :members:

.. automodule:: vmec_jax.core.boozer
   :members:

Straight-axis mirrors
---------------------

.. automodule:: vmec_jax.mirror.model
   :members:

.. automodule:: vmec_jax.mirror.solver
   :members:

.. automodule:: vmec_jax.mirror.free_boundary
   :members:

.. automodule:: vmec_jax.mirror.implicit
   :members:

.. automodule:: vmec_jax.mirror.output
   :members:

Errors and CLI
--------------

.. automodule:: vmec_jax.core.errors
   :members:

.. automodule:: vmec_jax.core.cli
   :members:
