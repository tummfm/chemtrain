"""Tests for structural quantities based on neighbor lists."""

from types import SimpleNamespace

import jax.numpy as jnp
from jax_md import space
import numpy as np

from jax_md_mod import custom_quantity


def test_tetrahedral_neighbors_ignore_padded_indices():
    """A nearby padding entry cannot replace a real tetrahedral neighbor."""
    positions = jnp.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [4.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
    ])
    invalid = positions.shape[0]
    neighbor_indices = jnp.full((positions.shape[0], 5), invalid)
    neighbor_indices = neighbor_indices.at[0].set(
        jnp.asarray([1, 2, 3, 4, invalid])
    )
    neighbor = SimpleNamespace(idx=neighbor_indices)
    displacement, _ = space.free()

    nearest = custom_quantity._nearest_tetrahedral_nbrs(
        displacement, positions, neighbor
    )

    distances = jnp.sort(space.distance(nearest[0]))
    np.testing.assert_allclose(distances, jnp.asarray([1.0, 2.0, 3.0, 4.0]))
