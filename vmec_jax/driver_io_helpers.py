"""Driver helpers for bundled example paths and lightweight I/O wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


def example_paths(
    case: str,
    *,
    root: str | Path | None = None,
    package_file: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Return ``(input_path, wout_path)`` for a bundled example case."""

    if root is None:
        if package_file is None:
            package_file = __file__
        root_path = Path(package_file).resolve().parents[1]
    else:
        root_path = Path(root)
    data_dir = root_path / "examples" / "data"
    input_path = data_dir / f"input.{case}"
    wout_path = data_dir / f"wout_{case}_reference.nc"
    if not wout_path.exists():
        wout_path = data_dir / f"wout_{case}.nc"
    if not wout_path.exists():
        wout_path = None
    return input_path, wout_path


def load_example(
    case: str,
    *,
    root: str | Path | None = None,
    with_wout: bool = True,
    grid=None,
    example_data_type: Callable[..., Any],
    example_paths_func: Callable[..., tuple[Path, Path | None]],
    load_config_func: Callable[..., tuple[Any, Any]],
    free_boundary_static_inputs_func: Callable[..., tuple[Any, Any]],
    build_static_func: Callable[..., Any],
    read_wout_func: Callable[..., Any],
    state_from_wout_func: Callable[..., Any],
):
    """Load a bundled example case using driver-injected dependencies."""

    input_path, wout_path = example_paths_func(case, root=root)
    cfg, indata = load_config_func(str(input_path))
    fb_meta, fb_extcur = free_boundary_static_inputs_func(cfg, load_fields=False, strict=False)
    static = build_static_func(cfg, grid=grid, mgrid_metadata=fb_meta, free_boundary_extcur=fb_extcur)
    if with_wout and wout_path is not None:
        wout = read_wout_func(wout_path)
        state = state_from_wout_func(wout)
    else:
        wout = None
        state = None
    return example_data_type(
        input_path=input_path,
        wout_path=wout_path,
        cfg=cfg,
        indata=indata,
        static=static,
        wout=wout,
        state=state,
    )


def load_input(path: str | Path, *, load_config_func: Callable[..., tuple[Any, Any]]):
    """Convenience wrapper around the configured input loader."""

    return load_config_func(str(path))


def load_wout(path: str | Path, *, read_wout_func: Callable[..., Any]):
    """Convenience wrapper around the configured ``wout`` reader."""

    return read_wout_func(path)


def save_npz(path: str | Path, **arrays) -> Path:
    """Save arrays into a NumPy ``.npz`` file and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path
