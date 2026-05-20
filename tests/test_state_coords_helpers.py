import numpy as np
import pytest

from vmec_jax.coords import Coords
import vmec_jax.state as state_module
from vmec_jax.state import StateLayout, VMECState, pack_state, unpack_state, zeros_state


def test_state_layout_pack_split_roundtrip_and_shape_errors():
    layout = StateLayout(ns=2, K=3, lasym=True)
    blocks = [np.full((layout.ns, layout.K), float(i)) for i in range(layout.n_fields)]

    flat = layout.pack(*blocks)
    assert flat.shape == (layout.size,)

    for actual, expected in zip(layout.split(flat), blocks, strict=True):
        np.testing.assert_allclose(actual, expected)

    with pytest.raises(ValueError, match="Expected flat vector"):
        layout.split(np.zeros((layout.size, 1)))
    with pytest.raises(ValueError, match="Expected flat vector"):
        layout.split(np.zeros(layout.size - 1))


def test_state_pack_unpack_and_zero_state_contract():
    layout = StateLayout(ns=2, K=2, lasym=False)
    base = np.arange(layout.size, dtype=float)

    state = unpack_state(base, layout)
    np.testing.assert_allclose(pack_state(state), base)

    children, aux = state.tree_flatten()
    rebuilt = VMECState.tree_unflatten(aux, children)
    np.testing.assert_allclose(pack_state(rebuilt), base)
    assert rebuilt.layout == layout

    zero = zeros_state(layout, like=base)
    assert zero.layout == layout
    for block in zero.tree_flatten()[0]:
        np.testing.assert_allclose(block, np.zeros((layout.ns, layout.K)))


def test_coords_pytree_roundtrip_preserves_children():
    arrays = tuple(np.full((2, 3, 4), float(i)) for i in range(9))
    coords = Coords(*arrays)

    children, aux = coords.tree_flatten()
    rebuilt = Coords.tree_unflatten(aux, children)

    for actual, expected in zip(rebuilt.tree_flatten()[0], arrays, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_state_backend_detection_jax_disabled_and_duck_typed(monkeypatch):
    monkeypatch.setattr(state_module, "has_jax", lambda: False)
    assert state_module._is_jax_array(object()) is False
    assert state_module._xp_from(object()) is np

    class DuckJaxArray:
        __array_priority__ = 1000
        __module__ = "jax.synthetic"

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "jax":
            raise RuntimeError("jax.Array unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(state_module, "has_jax", lambda: True)
    monkeypatch.setattr("builtins.__import__", fake_import)

    assert state_module._is_jax_array(DuckJaxArray()) is True


def test_state_layout_pack_uses_jax_numpy_for_jax_inputs():
    if not state_module.has_jax():
        pytest.skip("JAX not available")
    import jax
    import jax.numpy as jnp

    layout = StateLayout(ns=1, K=2, lasym=False)
    block = jnp.asarray([[1.0, 2.0]])

    assert state_module._is_jax_array(block) is True
    assert state_module._xp_from(block) is jnp
    flat = layout.pack(*(block + i for i in range(layout.n_fields)))

    assert isinstance(flat, jax.Array)
    np.testing.assert_allclose(np.asarray(flat), [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7])


def test_register_pytree_node_class_safe_edge_paths(monkeypatch):
    class Example:
        pass

    monkeypatch.setattr(state_module, "has_jax", lambda: False)
    assert state_module._register_pytree_node_class_safe(Example) is Example

    monkeypatch.setattr(state_module, "has_jax", lambda: True)

    def duplicate_register(cls):
        raise ValueError("Duplicate custom PyTreeDef type registration")

    monkeypatch.setattr("jax.tree_util.register_pytree_node_class", duplicate_register)
    assert state_module._register_pytree_node_class_safe(Example) is Example

    def bad_register(cls):
        raise ValueError("different registration failure")

    monkeypatch.setattr("jax.tree_util.register_pytree_node_class", bad_register)
    assert state_module._register_pytree_node_class_safe(Example) is Example


def test_register_pytree_node_class_safe_manual_fallback(monkeypatch):
    class Example:
        @staticmethod
        def tree_unflatten(aux, children):
            return aux, children

        def tree_flatten(self):
            return (), None

    calls = []

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "jax.tree_util" and args and args[2] == ("register_pytree_node_class",):
            raise ImportError("old jax")
        return real_import(name, *args, **kwargs)

    def fake_register_node(cls, flatten, unflatten):
        calls.append((cls, flatten, unflatten))

    monkeypatch.setattr(state_module, "has_jax", lambda: True)
    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr("jax.tree_util.register_pytree_node", fake_register_node)

    assert state_module._register_pytree_node_class_safe(Example) is Example
    assert calls and calls[0][0] is Example

    def duplicate_register_node(*_args):
        raise ValueError("Duplicate custom PyTreeDef type registration")

    monkeypatch.setattr("jax.tree_util.register_pytree_node", duplicate_register_node)
    assert state_module._register_pytree_node_class_safe(Example) is Example

    def bad_register_node(*_args):
        raise ValueError("manual failure")

    monkeypatch.setattr("jax.tree_util.register_pytree_node", bad_register_node)
    with pytest.raises(ValueError, match="manual failure"):
        state_module._register_pytree_node_class_safe(Example)
