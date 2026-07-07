from __future__ import annotations

from pathlib import Path
import sys

from tools.diagnostics.assets.external_vmec_asset_inventory import build_inventory

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


MANIFEST = Path(__file__).resolve().parents[2] / "validation" / "external_vmec_asset_manifest.toml"


def _load_manifest() -> dict:
    with MANIFEST.open("rb") as stream:
        return tomllib.load(stream)


def test_external_vmec_asset_manifest_is_metadata_only_and_pinned() -> None:
    data = _load_manifest()

    assert data["version"] == 1
    assert "assets" in data
    assert "repositories" in data

    repositories = data["repositories"]
    assert repositories["simsopt"]["license"] == "MIT"
    assert repositories["landreman_vmec_equilibria"]["license"] == "none visible in GitHub metadata"

    for repo in repositories.values():
        assert len(repo["ref"]) == 40
        assert repo["url"].startswith("https://github.com/")

    for asset in data["assets"]:
        repo = repositories[asset["repository"]]
        assert repo["ref"] in asset["url"], asset["id"]
        assert repo["ref"] in asset["raw_url"], asset["id"]
        assert asset["url"].startswith(repo["url"] + "/blob/"), asset["id"]
        assert asset["raw_url"].startswith("https://raw.githubusercontent.com/"), asset["id"]
        assert asset["path"] not in {"", "."}, asset["id"]
        assert asset["recommendation"], asset["id"]
        assert asset["families"], asset["id"]


def test_external_vmec_asset_manifest_covers_required_physics_families() -> None:
    assets = _load_manifest()["assets"]
    family_sets = {asset["id"]: set(asset["families"]) for asset in assets}
    all_families = set().union(*family_sets.values())

    assert {"tokamak", "stellarator"} <= all_families
    assert {"fixed_boundary", "free_boundary"} <= all_families
    assert {"lasym_false", "lasym_true"} <= all_families
    assert {"axisymmetric", "nonaxisymmetric"} <= all_families
    assert {"qa", "qh", "qhs", "w7x", "ncsx"} <= all_families

    assert any({"tokamak", "axisymmetric", "fixed_boundary"} <= families for families in family_sets.values())
    assert any({"stellarator", "nonaxisymmetric", "fixed_boundary"} <= families for families in family_sets.values())
    assert any({"stellarator", "nonaxisymmetric", "free_boundary"} <= families for families in family_sets.values())
    assert any({"lasym_true", "fixed_boundary"} <= families for families in family_sets.values())


def test_license_unclear_assets_are_not_recommended_for_bundling() -> None:
    data = _load_manifest()
    repositories = data["repositories"]

    for asset in data["assets"]:
        repo = repositories[asset["repository"]]
        recommendation = str(asset["recommendation"])
        if repo["license"] == "none visible in GitHub metadata":
            assert "reference_or_explicit_fetch_only" in recommendation, asset["id"]
            assert "small_fixture" not in recommendation, asset["id"]


def test_free_boundary_candidates_state_mgrid_availability() -> None:
    assets = _load_manifest()["assets"]
    free_boundary_assets = [asset for asset in assets if "free_boundary" in set(asset["families"])]

    assert free_boundary_assets
    assert any(asset["recommendation"] == "fetched_or_generated_asset" for asset in free_boundary_assets)
    assert any("missing_mgrid" in asset["recommendation"] for asset in free_boundary_assets)

    for asset in free_boundary_assets:
        notes = str(asset["notes"]).lower()
        recommendation = str(asset["recommendation"])
        if "missing_mgrid" in recommendation:
            assert "not present" in notes or "missing" in notes, asset["id"]
        if recommendation == "fetched_or_generated_asset":
            assert "generated" in notes or "writes" in notes, asset["id"]


def test_external_vmec_asset_inventory_filters_and_checks_local_paths(tmp_path: Path) -> None:
    data = _load_manifest()
    simsopt_root = tmp_path / "simsopt"
    simsopt_root.mkdir()
    (simsopt_root / "tests" / "test_files").mkdir(parents=True)
    (simsopt_root / "tests" / "test_files" / "input.circular_tokamak").write_text("&INDATA\n/\n")
    (simsopt_root / "tests" / "test_files" / "wout_circular_tokamak_reference.nc").write_bytes(b"netcdf")

    inventory = build_inventory(
        data,
        source_roots={"simsopt": simsopt_root},
        repository="simsopt",
        family={"tokamak", "axisymmetric", "fixed_boundary"},
    )

    assert inventory["selected_count"] >= 1
    assert "simsopt_fixed_axisym_circular_tokamak" in inventory["present"]
    assert "tokamak" in inventory["family_counts"]
    circular = next(asset for asset in inventory["assets"] if asset["id"] == "simsopt_fixed_axisym_circular_tokamak")
    assert all(circular["path_status"].values())
