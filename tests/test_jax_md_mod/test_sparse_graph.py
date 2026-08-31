"""Tests for fixed-size sparse graph construction."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax_md import partition, space
import numpy as np

from jax_md_mod.model import sparse_graph


def build_graph(positions, neighbor_indices, *, max_edges, max_triplets):
    """Build one fixed-size graph from dense neighbor indices."""
    neighbor = SimpleNamespace(
        format=partition.NeighborListFormat.Dense,
        idx=neighbor_indices,
    )
    displacement, _ = space.free()
    return sparse_graph.sparse_graph_from_neighborlist(
        displacement,
        positions,
        neighbor,
        r_cutoff=2.0,
        max_edges=max_edges,
        max_triplets=max_triplets,
    )


def test_sparse_graph_compacts_edges_and_reports_triplet_overflow():
    """Graph masks pad edges and count triplets before truncation."""
    positions = jnp.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    neighbor_indices = jnp.asarray([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
    ])
    compiled_build = jax.jit(
        lambda pos: build_graph(
            pos,
            neighbor_indices,
            max_edges=8,
            max_triplets=1,
        )
    )

    graph, overflow = compiled_build(positions)

    assert overflow
    assert graph.n_edges == 6
    assert graph.n_triplets == 6
    np.testing.assert_array_equal(
        graph.edge_mask, jnp.asarray([True] * 6 + [False] * 2)
    )
    np.testing.assert_array_equal(graph.idx_j[-2:], jnp.asarray([3, 3]))
    np.testing.assert_array_equal(graph.triplet_mask, jnp.asarray([True]))


def test_sparse_graph_truncates_edges_in_source_order():
    """Edge overflow retains the first valid dense-neighbor entries."""
    positions = jnp.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    neighbor_indices = jnp.asarray([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
    ])

    graph, overflow = build_graph(
        positions, neighbor_indices, max_edges=4, max_triplets=12
    )

    assert overflow
    np.testing.assert_array_equal(graph.idx_i, jnp.asarray([0, 0, 1, 1]))
    np.testing.assert_array_equal(graph.idx_j, jnp.asarray([1, 2, 0, 2]))
    np.testing.assert_array_equal(graph.edge_mask, jnp.ones(4, dtype=bool))


def test_sparse_graph_pads_an_empty_edge_set():
    """An empty graph uses only invalid fixed-size edge entries."""
    positions = jnp.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    neighbor_indices = jnp.full((3, 3), 3)

    graph, overflow = build_graph(
        positions, neighbor_indices, max_edges=4, max_triplets=4
    )

    assert not overflow
    np.testing.assert_array_equal(graph.idx_i, jnp.zeros(4, dtype=jnp.int32))
    np.testing.assert_array_equal(graph.idx_j, jnp.full(4, 3))
    np.testing.assert_allclose(graph.distance_ij, jnp.zeros(4))
    np.testing.assert_array_equal(graph.edge_mask, jnp.zeros(4, dtype=bool))
