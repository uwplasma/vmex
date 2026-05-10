import numpy as np
import pytest

from vmec_jax.coords import Coords
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
