#!/usr/bin/env python
"""Compatibility wrapper for the direct-start QA/QH sweep.

The unified sweep driver now handles both continuation and direct-start
policies. This wrapper preserves the old entry point while writing results to
the same backend-aware layout as ``generate_qs_ess_sweep.py``.
"""

from __future__ import annotations

import sys

from generate_qs_ess_sweep import main as _main


def main() -> None:
    args = sys.argv[1:]
    has_policy = "--policy" in args or any(arg.startswith("--policy=") for arg in args)
    if not has_policy:
        args = ["--policy", "direct", *args]
    sys.argv = [sys.argv[0], *args]
    _main()


if __name__ == "__main__":
    main()
