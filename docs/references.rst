References
==========

Background and canonical references for VMEC and related equilibrium methods:

1. S. P. Hirshman and J. C. Whitson, “Steepest-descent moment method for
   three-dimensional magnetohydrodynamic equilibria,” *Physics of Fluids* 26
   (1983).

2. S. P. Hirshman, W. I. van Rij, and P. Merkel, “Three-dimensional free
   boundary calculations using a spectral Green’s function method,” *Computer
   Physics Communications* 43 (1986).

3. P. Merkel, “Solution of stellarator boundary value problems with external
   currents,” *Nuclear Fusion* 27 (1987).

4. VMEC2000 reference documentation and ``wout`` file format notes (VMEC/LIBSTELL
   distribution and Princeton VMEC resources).

5. VMEC++ numerics notes (local copy):
   ``vmecpp/docs/the_numerics_of_vmecpp.pdf``.

6. VMEC++ Fourier basis implementation note (local copy):
   ``vmecpp/docs/fourier_basis_implementation.md``.

7. VMEC2000 solver core (residuals, bcovar, preconditioner):
   ``STELLOPT/VMEC2000/Sources/General/funct3d.f`` and
   ``STELLOPT/VMEC2000/Sources/General/bcovar.f``.

8. VMEC2000 time-step control and restart logic:
   ``STELLOPT/VMEC2000/Sources/TimeStep/evolve.f`` and
   ``STELLOPT/VMEC2000/Sources/TimeStep/restart.f``.

9. VMEC2000 diagnostic scalars and Mercier stability:
   ``STELLOPT/VMEC2000/Sources/Input_Output/eqfor.f`` and
   ``STELLOPT/VMEC2000/Sources/Input_Output/mercier.f``.

10. A. H. Glasser, J. M. Greene, and J. L. Johnson, “Resistive instabilities
    in general toroidal plasma configurations,” *Physics of Fluids* 18(7),
    875-888 (1975).

11. M. Landreman and R. Jorge, “Magnetic well and Mercier stability of
    stellarators near the magnetic axis,” *Journal of Plasma Physics* 86(5),
    905860510 (2020), arXiv:2006.14881.

12. VMEC++ solver/restart structure and parity-relevant control flow:
    ``vmecpp/src/vmecpp/cpp/vmecpp/vmec/vmec/vmec.cc``.

13. VMEC++ output-quantity and near-axis extrapolation notes:
    ``vmecpp/src/vmecpp/cpp/vmecpp/vmec/output_quantities/output_quantities.cc``.
