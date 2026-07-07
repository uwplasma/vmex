# Repository Health Diagnostics Tests

Tests in this folder cover developer-only repository health tools under
`tools/diagnostics/repo_health/`.

These tests protect CI bucket construction, local release gates, source-size
checks, docstring gates, and repository-size checks. They should stay fast and
avoid importing the VMEC solver unless a repository-health tool explicitly needs
that behavior.
