# External field providers

This package contains magnetic-field providers used by free-boundary VMEC:

- `coils_jax.py`: pure-JAX Fourier coil curves and Biot-Savart sampling.
- `mgrid_jax.py`: JAX-compatible interpolation of VMEC mgrid data.
- `essos_adapter.py`: optional ESSOS bridge.
- `base.py`: shared provider contracts.

Provider code should remain differentiable where practical and should avoid
writing intermediate mgrid files unless explicitly requested by an example or
validation workflow.
