# Asset and Fixture Tests

This folder validates repository metadata for optional external assets:

- documented downloader behavior,
- external VMEC fixture manifest integrity,
- bundled validation asset gates and WOUT round trips.

The tests should remain metadata-focused and fast. Large numerical comparisons
belong in `tests/parity/` and should fetch assets explicitly when needed.
