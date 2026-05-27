from __future__ import annotations

import pytest

import vmec_jax


class _FakePyproject:
    def __init__(self, *, exists: bool, text: str = ""):
        self._exists = bool(exists)
        self._text = str(text)

    def exists(self) -> bool:
        return self._exists

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        return self._text


class _FakeRoot:
    def __init__(self, pyproject: _FakePyproject):
        self._pyproject = pyproject

    def __truediv__(self, name: str):
        assert name == "pyproject.toml"
        return self._pyproject


class _FakePath:
    def __init__(self, _path, pyproject: _FakePyproject):
        self.parents = [object(), _FakeRoot(pyproject)]

    def resolve(self):
        return self


def test_source_tree_version_handles_missing_project_table_and_version(monkeypatch):
    monkeypatch.setattr(vmec_jax, "_Path", lambda path: _FakePath(path, _FakePyproject(exists=False)))
    assert vmec_jax._source_tree_version() is None

    monkeypatch.setattr(
        vmec_jax,
        "_Path",
        lambda path: _FakePath(path, _FakePyproject(exists=True, text="[project]\nname = \"vmec-jax\"\n[tool.pytest]\n")),
    )
    assert vmec_jax._source_tree_version() is None

    monkeypatch.setattr(
        vmec_jax,
        "_Path",
        lambda path: _FakePath(path, _FakePyproject(exists=True, text="[project]\nname = \"vmec-jax\"\n")),
    )
    assert vmec_jax._source_tree_version() is None


def test_lazy_public_api_and_dir_behaviour():
    assert vmec_jax.__getattr__("api") is vmec_jax.api
    with pytest.raises(AttributeError, match="definitely_missing"):
        vmec_jax.__getattr__("definitely_missing")
    public_names = vmec_jax.__dir__()
    assert "api" in public_names
    assert "run_fixed_boundary" in public_names
    assert "run_booz_xform" in public_names
    assert "qi_stage_modes" in public_names
