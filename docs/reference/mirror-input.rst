Mirror input file
=================

An open-mirror equilibrium is one ``&MIRROR`` namelist, run like a VMEC deck:

.. code-block:: console

   vmex examples/data/input.mirror_axisymmetric --plot

or ``vj.solve_file(path)`` from Python, which returns a
:class:`vmex.mirror.MirrorSolution` and writes ``mout_<case>.nc`` beside the
deck. :meth:`vmex.mirror.MirrorInput.from_file` and
:meth:`~vmex.mirror.MirrorInput.to_file` read and write the same format; a
file is recognised as a mirror deck by its ``&MIRROR`` line. ``!`` starts a
comment, names are case-insensitive, and an unknown name is an error.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Variable
     - Default
     - Meaning
   * - ``NS``
     - 7
     - Radial surfaces (``s = 0 ... 1``, ``r = sqrt(s) a``).
   * - ``MPOL``
     - 0
     - Largest poloidal mode of the boundary and the state; 0 is axisymmetric.
   * - ``ELEMENTS``
     - 6
     - Axial cubic B-spline elements between the end cuts.
   * - ``NXI``
     - 17
     - Axial collocation nodes of the free-boundary exterior grid.
   * - ``Z_MIN``, ``Z_MAX``
     - -1, 1
     - Axial positions of the two fixed-flux end cuts [m].
   * - ``PHIEDGE``
     - required
     - Axial magnetic flux through the plasma boundary [Wb].
   * - ``ZB``
     - uniform
     - Axial stations of the boundary table [m]; omitted means uniform over
       ``[Z_MIN, Z_MAX]``. The stations must cover the domain.
   * - ``RBC(m)``, ``RBS(m)``
     - 0
     - Boundary polar radius ``a(theta, z) = sum_m RBC(m) cos(m theta) +
       RBS(m) sin(m theta)`` [m], one value per station, joined along ``z``
       by a cubic spline. Fixed boundary only.
   * - ``PRES_SCALE``, ``AM``
     - 0, ``1 -1``
     - Pressure ``p(s) = PRES_SCALE * sum_i AM(i) s**i`` [Pa].
   * - ``CURRENT_DERIVATIVE``
     - 0
     - Axial-current profile ``I'(s)`` (fixed boundary only).
   * - ``GAMMA``
     - 5/3
     - Adiabatic index of the mass-conserving pressure.
   * - ``FTOL``, ``NITER``
     - 1e-12, 1000
     - Normalised variational-force tolerance and iteration budget.
   * - ``LFREEB``
     - F
     - Free boundary: the plasma sits in the vacuum field of the coils below,
       starts from the vacuum flux tube of ``PHIEDGE`` and is continued from
       vacuum to the central pressure of ``PRES_SCALE``.
   * - ``COIL_RADIUS``, ``COIL_Z``, ``COIL_CURRENT``
     - none
     - Coaxial circular loops [m, m, A] of a free-boundary deck; their field
       is the exact elliptic-integral solution
       (:class:`vmex.mirror.CircularCoils`).
   * - ``EXTERIOR_NTHETA``, ``EXTERIOR_ORDER``
     - 12, 6
     - Resolution of the exterior boundary-integral vacuum solve.

The shipped decks are ``examples/data/input.mirror_axisymmetric`` (a flux
surface of an exact vacuum mirror, reproduced to 8e-4) and
``examples/data/input.mirror_two_coil_free_boundary`` (the geometry of the
independent Pleiades reference, at 10 % central beta).
