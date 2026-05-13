Installation
============

Requirements
------------

- Python 3.10+
- NumPy (core requirement)
- JAX + jaxlib (for performance + autodiff)

Recommended:

- ``netCDF4`` to read VMEC2000 ``wout_*.nc`` files for validation
- ``booz_xform_jax`` for quasi-isodynamic (QI) optimization objectives

Install from source
-------------------

From the repo root (non-editable install)::

  python -m pip install -U pip
  python -m pip install .

Enable JAX::

  python -m pip install ".[jax]"

VMEC relies heavily on float64. JAX defaults to float32 unless x64 is enabled.
We recommend setting::

  export JAX_ENABLE_X64=1

Enable netCDF support::

  python -m pip install ".[netcdf]"

Enable QI optimization support::

  python -m pip install ".[qi]"

Editable install (recommended for development)::

  python -m pip install -e .

Build docs locally
------------------

Install doc dependencies::

  python -m pip install ".[docs]"

Then build docs::

  LANG=C LC_ALL=C python -m sphinx -b html docs docs/_build/html

To reproduce the stricter CI / release build locally, use warnings as errors
and nitpicky cross-reference checks::

  LANG=C LC_ALL=C python -m sphinx -W -n -b html docs docs/_build/html

Read the Docs builds the full user guide with warnings treated as errors.  For
local edit cycles where only the landing page is needed, use the explicit fast
mode::

  SPHINX_FAST=1 LANG=C LC_ALL=C python -m sphinx -b html docs docs/_build/html_fast
