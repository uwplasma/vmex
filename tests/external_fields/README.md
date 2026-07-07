# External-Field Tests

This folder covers differentiable external magnetic-field providers:

- pure-JAX Fourier coil curves and Biot-Savart sampling,
- JAX mgrid interpolation compatibility,
- optional ESSOS adapter behavior,
- robust coil perturbation utilities.

Free-boundary solver tests that consume these providers live in
`tests/free_boundary/`.
