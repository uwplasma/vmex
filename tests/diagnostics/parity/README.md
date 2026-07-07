# Parity Diagnostics Tests

Tests in this folder cover developer diagnostics under
`tools/diagnostics/parity/`: VMEC2000 trace parsers, manifest validation,
bounded parity command construction, and diagnostic report helpers.

Keep actual physics and VMEC2000 execution gates in the main test suite unless
they only exercise a diagnostics parser or renderer. This split keeps
developer-tool coverage easy to find without hiding production parity tests.
