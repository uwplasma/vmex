# Validation manifests

This folder stores validation manifests and compact reviewed evidence.  It is
not a dumping ground for generated solver outputs.

- `external_vmec_asset_manifest.toml`: optional external assets used by local or
  nightly validation.
- `qi_seed_robustness_plan.py`: structured QI seed-robustness validation plan.
- `artifacts/`: compact, reviewed JSON artifacts only.

Large WOUT, mgrid, Boozer, and benchmark files should stay out of git and be
fetched or regenerated as needed.
