# Example data

This folder contains VMEC input decks and small checked-in fixtures used by
examples, tests, and documentation.

- `input.*`: VMEC input decks.
- `single_grid/`: single-grid benchmark and parity inputs used by the README,
  docs, and optional VMEC2000/VMEC++ comparisons.
- Large reference WOUT, mgrid, Boozer, and JXB files are ignored by git and are
  fetched on demand with `python tools/fetch_assets.py`.

Keep new example inputs small.  Put generated output files in ignored output
directories, not in this data folder.
