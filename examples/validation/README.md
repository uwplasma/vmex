# Validation Examples

This folder contains optional user-facing validation scripts that compare
vmec_jax against external tools such as VMEC2000.

These examples may require local executables, optional fetched assets, or longer
runtimes than the basic examples. They are useful for reproducing parity checks
but should not be confused with the compact CI tests in `tests/`.

Generated WOUT, mgrid, Boozer, JSON, and plot files should stay out of git
unless they are compact reviewed documentation artifacts.
