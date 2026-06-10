Installation
============

Requirements
------------

- Python 3.10+
- NumPy (core requirement)
- JAX + jaxlib (for performance + autodiff)
- ``netCDF4`` to read and write VMEC ``wout_*.nc`` files
- ``matplotlib`` for plotting helpers and examples
- ``booz_xform_jax`` for differentiable Boozer-coordinate QI objectives

From PyPI
---------

``vmec-jax`` is available on `PyPI <https://pypi.org/project/vmec-jax/>`_::

  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip setuptools wheel packaging
  python -m pip install vmec-jax

PyPI can lag repository tags.  Check the package-index version before pinning or
advertising an exact release.

The plain install includes plotting support and the differentiable
``booz_xform_jax`` dependency used by the QI optimization examples.  There is
no separate plotting or QI extra.

Use ``python -m pip`` from the active environment rather than a bare ``pip``.
This prevents the common failure mode where Homebrew/system Python packages,
user-site packages, and the active interpreter are mixed.  If an install or
backend import fails, run::

  vmec --doctor

and include that report when asking for help.

GPU-enabled JAX is intentionally not forced by ``vmec-jax`` because the correct
wheel depends on platform, CUDA/ROCm version, and driver support.  Install the
CPU package above first, or install/upgrade JAX for your accelerator using the
official JAX installation matrix:
https://docs.jax.dev/en/latest/installation.html.

From conda-forge
----------------

``vmec-jax`` can be installed as a conda package from `conda-forge
<https://github.com/conda-forge/vmec-jax-feedstock>`_ into a particular project
with `Pixi <https://pixi.prefix.dev/>`_::

  pixi add vmec-jax

or into a conda environment with `conda
<https://docs.conda.io/projects/conda/>`_::

  conda install --channel conda-forge vmec-jax

The feedstock may lag both PyPI and the repository tag; verify the available
conda-forge version when documenting a release.

From source
-----------

From the repo root (non-editable install)::

  python -m pip install -U pip setuptools wheel packaging
  python -m pip install .

VMEC relies heavily on float64. JAX defaults to float32 unless x64 is enabled.
We recommend setting::

  export JAX_ENABLE_X64=1

Editable install (recommended for development)::

  python -m pip install -U pip setuptools wheel packaging
  python -m pip install -e .

Optional validation assets
--------------------------

Generated VMEC reference ``wout`` fixtures and full-size free-boundary
``mgrid`` files are release assets rather than tracked git blobs.  A fresh
clone is enough to run the input decks and small bundled examples; the example
``wout_*.nc`` files are generated on demand.  Fetch the optional released
assets only when running full physics/parity gates or regenerating docs panels
that consume stored WOUTs::

  python tools/fetch_assets.py --list
  python tools/fetch_assets.py

After installation, ``vmec --test`` runs a packaged quick-start input with
``FTOL_ARRAY = 1e-12`` and therefore works even outside a source checkout.

Build docs locally
------------------

Install doc dependencies::

  python -m pip install ".[docs]"

Then build docs::

  LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -b html docs docs/_build/html

To reproduce the current strict CI / release build locally, use warnings as
errors::

  LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html

The full documentation is not yet nitpicky-clean; use ``-n`` only when working
specifically on cross-reference cleanup.  Read the Docs builds the full user
guide with warnings treated as errors.  For local edit cycles where only the
landing page is needed, use the explicit fast mode::

  SPHINX_FAST=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html_fast
