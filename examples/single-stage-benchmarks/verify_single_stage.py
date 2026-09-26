#!/usr/bin/env python
"""Qualify the fixed objective and constraints separately from production."""
import single_stage_optimization_scalar as example
from single_stage_support import verification


def main(argv=None):
    return verification.main(example, argv)


if __name__ == "__main__":
    raise SystemExit(main())
