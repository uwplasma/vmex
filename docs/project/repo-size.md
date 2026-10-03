# Repository size

Keep reproducible benchmarks and scientific tests; consolidate repeated machinery and shorten explanatory text.

## Current tree

Tracked files after updating this branch to main (2026-09-30):

| Directory | Files | Source/documentation lines |
| --- | ---: | ---: |
| `vmex/` | 76 | 65,424 |
| `tests/` | 137 | 50,219 |
| `docs/` | 110 | 12,804 |
| `examples/` | 131 | 15,629 |
| `benchmarks/` | 61 | 7,933 |

The full clone measured 45 MB on 2026-09-27; shallow and blobless clones measured 5.9 and 8.5 MB. Historical paths accounted for about 75% of packed blob bytes.

## This change

Four static figures shrink from 1,269,764 to 686,830 bytes after WebP re-encoding. Dimensions and plotted data stay unchanged; the provenance manifest records the new hashes. Updated main's mirror figures are retained.

The manifest updater reads documentation once per run instead of once per figure. Existing provenance tests check hashes, source records and references.

## Next changes

- Reuse solver and diagnostic kernels before adding new implementations.
- Combine tests with the same fixtures and purpose; retain independent numerical checks.
- Keep benchmark summaries and reproducible drivers in git; link large raw archives separately.
- Keep small shared modules when consolidation would duplicate logic or break imports. `_netcdf` serves three writers.

History changes and branch deletion require a separate maintainer decision.
