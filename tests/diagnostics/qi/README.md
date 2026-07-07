# QI Diagnostics Tests

Tests in this folder cover developer diagnostics under `tools/diagnostics/qi/`:
seed audits, basin surveys, filter searches, landscape scans, parameter probes,
and policy scans.

Keep production quasi-isodynamic objectives, public helper APIs, and example
workflow tests outside this folder unless they only exercise a diagnostics
script. The goal is to make QI tooling coverage easy to find without hiding the
physics validation tests.
