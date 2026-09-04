# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools

import jax
from jax import numpy as jnp, Array

from jax_md_mod import custom_partition, custom_quantity
from jax_md import partition, dataclasses, space

import numpy as onp

import pytest

import networkx as nx


# A simple methane-like graph
star_graph_dense = jnp.asarray([
    [1, 2, 3, 4, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 5],
    [0, 5, 5, 5, 5, 5, 5, 5],
])

star_graph_sparse = jnp.asarray(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 5],
        [5, 5],
        [5, 5],
        [5, 5]
    ]
)

fc_graph_dense = jnp.asarray([
    [1, 2, 3, 4, 4],
    [0, 2, 3, 4, 4],
    [0, 1, 3, 4, 4],
    [0, 1, 2, 4, 4]
])

fc_graph_sparse = jnp.asarray([
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 3],
        [3, 0],
        [3, 1],
        [3, 2],
        [4, 4],
        [4, 4]
    ]
])


@dataclasses.dataclass
class NeighborIdx:
    idx: Array
    format: partition.NeighborListFormat
    reference_position: Array = None
    max_occupancy: int = None
    did_buffer_overflow: Array = False


class TestBoxConvention:
    def test_lower_triangular_box_is_not_malformed(self):
        lower = jnp.asarray([
            [2.0, 0.0, 0.0],
            [0.2, 2.1, 0.0],
            [0.1, 0.3, 2.2],
        ])
        upper = lower.T

        assert not partition.is_box_valid(lower)
        assert partition.is_box_valid(upper)
        assert not partition.is_box_valid(jnp.asarray([2.0, 2.1, 2.2]))


class TestEdgeMasking:

    def test_readout_vectors_maps_invalid_edges_to_cutoff_before_sorting(self):
        position = jnp.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ])
        neighbor = NeighborIdx(
            idx=jnp.asarray(
                [[0, 0, 3, -1], [1, 2, 1, 1]], dtype=jnp.int32
            ),
            format=partition.NeighborListFormat.Sparse,
        )
        displacement, _ = space.free()

        vectors, senders, receivers = custom_partition.readout_vectors(
            displacement,
            2.0,
            position,
            neighbor,
            mask=jnp.ones(3, dtype=bool),
        )
        onp.testing.assert_allclose(
            onp.linalg.norm(onp.asarray(vectors), axis=-1),
            onp.asarray([1.0, 2.0, 2.0, 2.0]),
        )
        onp.testing.assert_array_equal(onp.asarray(senders), [0, 3, 3, 3])
        onp.testing.assert_array_equal(onp.asarray(receivers), [1, 3, 3, 3])
        contributions = jax.ops.segment_sum(
            jnp.ones_like(senders), senders, num_segments=position.shape[0]
        )
        onp.testing.assert_array_equal(onp.asarray(contributions), [1, 0, 0])

        sorted_vectors, _, _ = custom_partition.readout_vectors(
            displacement,
            2.0,
            position,
            neighbor,
            mask=jnp.ones(3, dtype=bool),
            max_edges=2,
        )
        onp.testing.assert_allclose(
            onp.linalg.norm(onp.asarray(sorted_vectors), axis=-1),
            onp.asarray([1.0, 2.0]),
        )

        gradient = jax.grad(
            lambda pos: jnp.sum(custom_partition.readout_vectors(
                displacement,
                2.0,
                pos,
                neighbor,
                mask=jnp.ones(3, dtype=bool),
            )[0])
        )(position)
        assert jnp.all(jnp.isfinite(gradient))

    def test_tetrahedral_neighbors_ignore_dense_padding(self):
        position = jnp.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.1, 0.0, 0.0],
        ])
        padding = position.shape[0]
        neighbor = NeighborIdx(
            idx=jnp.asarray([
                [1, 2, 3, 4, padding],
                [padding, padding, padding, padding, padding],
                [padding, padding, padding, padding, padding],
                [padding, padding, padding, padding, padding],
                [padding, padding, padding, padding, padding],
            ], dtype=jnp.int32),
            format=partition.NeighborListFormat.Dense,
        )
        displacement, _ = space.free()

        nearest = jax.jit(
            lambda pos: custom_quantity._nearest_tetrahedral_nbrs(
                displacement, pos, neighbor
            )
        )(position)
        norms = onp.sort(onp.linalg.norm(onp.asarray(nearest[0]), axis=-1))
        onp.testing.assert_allclose(norms, [0.1, 1.0, 2.0, 3.0])
    @pytest.mark.parametrize(
        "partitions, capacity, expected",
        (
            ([0, 0, 1, 1], 4, [[0, 2, 3, 4], [1, 3, 2, 4]]),
            ([0, 0, 1, 1], 1, [[0], [1]]),
            ([0, 1, 2, 3], 3, [[4, 4, 4], [4, 4, 4]]),
        ),
    )
    def test_partition_compacts_and_pads_sparse_edges(
        self, partitions, capacity, expected
    ):
        """Sparse partitioning keeps source order and pads after valid edges."""
        neighbor = NeighborIdx(
            idx=jnp.asarray([
                [0, 0, 2, 0, 4, 3],
                [1, 2, 3, 3, 4, 2],
            ]),
            format=partition.NeighborListFormat.Sparse,
            reference_position=jnp.zeros((4, 3)),
            max_occupancy=6,
        )
        apply_partition = jax.jit(
            lambda groups: custom_partition.partition_neighbor_list(
                neighbor, groups, max_capacity=capacity
            ).idx
        )

        result = apply_partition(jnp.asarray(partitions))

        onp.testing.assert_array_equal(result, jnp.asarray(expected))


    def test_mask_neighbor_list_map_masks_every_list(self):
        """Particle masks apply to every named neighbor list."""
        position = jnp.zeros((3, 2))
        default = NeighborIdx(
            idx=jnp.asarray([[0, 1, 2, 3], [1, 0, 0, 3]]),
            format=partition.NeighborListFormat.Sparse,
            reference_position=position,
        )
        longer_range = NeighborIdx(
            idx=jnp.asarray([[0, 2, 1, 3], [2, 0, 2, 3]]),
            format=partition.NeighborListFormat.Sparse,
            reference_position=position,
        )
        neighbors = custom_partition.NeighborListMap({
            "default": default,
            "longer_range": longer_range,
        })

        masked = custom_partition.mask_neighbor_list(
            neighbors, jnp.asarray([True, False, True])
        )

        onp.testing.assert_array_equal(
            masked["default"].idx,
            jnp.asarray([[3, 3, 2, 3], [3, 3, 0, 3]]),
        )
        onp.testing.assert_array_equal(
            masked["longer_range"].idx,
            jnp.asarray([[0, 2, 3, 3], [2, 0, 3, 3]]),
        )

    def test_neighbor_list_map_reports_overflow_from_every_list(self):
        """Generic simulation checks also see non-default-list overflow."""
        position = jnp.zeros((2, 3))
        edge_index = jnp.asarray([[0, 2], [1, 2]])
        neighbors = custom_partition.NeighborListMap({
            "default": NeighborIdx(
                idx=edge_index,
                format=partition.NeighborListFormat.Sparse,
                reference_position=position,
                did_buffer_overflow=jnp.asarray([False, False]),
            ),
            "longer_range": NeighborIdx(
                idx=edge_index,
                format=partition.NeighborListFormat.Sparse,
                reference_position=position,
                did_buffer_overflow=jnp.asarray([False, True]),
            ),
        })

        overflow = jax.jit(lambda neighbor: neighbor.did_buffer_overflow)(
            neighbors
        )

        onp.testing.assert_array_equal(overflow, jnp.asarray([False, True]))

    @pytest.mark.parametrize("graph", (star_graph_dense, fc_graph_dense))
    @pytest.mark.parametrize("exclude", (
        [[0, 1], [1, 2]],
        [[0, 1], [0, 1]],
        [[0, 1], [1, 0]]
    ))
    @pytest.mark.parametrize("mask", (
        [True, True],
        [False, True]
    ))
    def test_exclude_dense(self, graph, exclude, mask):
        exclude = jnp.asarray(exclude)
        mask = jnp.asarray(mask)


        neighbor = NeighborIdx(
            idx=graph, format=partition.NeighborListFormat.Dense)

        # Substract from graph
        neighbor = custom_partition.exclude_from_neighbor_list(
            neighbor, exclude, mask
        )

        print(f"New neighbor list")
        print(neighbor)

        # Check
        for (i, j), mask in zip(exclude, mask):
            if not mask: continue

            assert i not in neighbor.idx[j]
            assert j not in neighbor.idx[i]

    @pytest.mark.parametrize("graph", (star_graph_sparse, fc_graph_sparse))
    @pytest.mark.parametrize("exclude", (
        [[0, 1], [1, 2]],
        [[0, 1], [0, 1]],
        [[0, 1], [1, 0]]
    ))
    @pytest.mark.parametrize("mask", (
        [True, True],
        [False, True]
    ))
    def test_exclude_sparse(self, graph, exclude, mask):
        exclude = jnp.asarray(exclude)
        mask = jnp.asarray(mask)

        neighbor = NeighborIdx(
            idx=graph, format=partition.NeighborListFormat.Sparse)

        # Substract from graph
        neighbor = custom_partition.exclude_from_neighbor_list(
            neighbor, exclude, mask
        )

        print(f"New neighbor list")
        print(neighbor)

        # Check
        for (i, j), mask in zip(exclude, mask):
            if not mask: continue

            assert not jnp.any(
                jnp.logical_and(
                    i == neighbor.idx[0, :],
                    j == neighbor.idx[1, :]
                )
            )
            assert not jnp.any(
                jnp.logical_and(
                    j == neighbor.idx[0, :],
                    i == neighbor.idx[1, :]
                )
            )


class TestTriplets:

    def test_dense_triplets_mask_all_out_of_bounds_indices(self):
        n_particles = 4
        idx = jnp.asarray([
            [1, 2, -1, n_particles + 1],
            [0, 2, n_particles, n_particles],
            [0, 1, n_particles, n_particles],
            [n_particles, n_particles, n_particles, n_particles],
        ], dtype=jnp.int32)

        @jax.jit
        def triplets(dense_idx):
            neighbor = NeighborIdx(
                idx=dense_idx,
                format=partition.NeighborListFormat.Dense,
            )
            return custom_partition.get_triplet_indices(neighbor)

        ij, kj, mask = triplets(idx)
        valid_ij = onp.asarray(ij)[onp.asarray(mask)]
        valid_kj = onp.asarray(kj)[onp.asarray(mask)]
        assert valid_ij.size > 0
        assert onp.all((0 <= valid_ij) & (valid_ij < n_particles))
        assert onp.all((0 <= valid_kj) & (valid_kj < n_particles))

    @pytest.mark.parametrize("graph", (star_graph_dense, fc_graph_dense))
    def test_get_dense(self, graph):
        neighbor = NeighborIdx(
            idx=graph, format=partition.NeighborListFormat.Dense)

        ij, kj, mask = custom_partition.get_triplet_indices(neighbor)
        mask = onp.asarray(mask)

        valid_count = int(onp.sum(mask))
        invalid_idx = graph.shape[0]
        onp.testing.assert_array_equal(
            mask,
            onp.arange(mask.size) < valid_count,
        )
        assert onp.all(onp.asarray(ij[valid_count:]) == invalid_idx)
        assert onp.all(onp.asarray(kj[valid_count:]) == invalid_idx)

        # Remove all invalid edges
        ij = ij[mask, :]
        kj = kj[mask, :]

        # Add the graph to nx and check whether all found triplets exist
        graph = custom_partition.to_networkx(neighbor)

        for i, j, k in zip(ij[:, 0], ij[:, 1], kj[:, 0]):
            assert nx.is_simple_path(graph, [int(i), int(j), int(k)])

        print(onp.append(ij, kj[:, (0,)], axis=1))
        print(mask)

        # Search all triplets in the graph
        for a, b in itertools.combinations(graph.nodes, 2):
            for path in nx.all_simple_paths(graph, a, b, cutoff=3):
                if len(path) != 3: continue

                # Check whether triplets were also found by jax algorithm
                found = False
                for i, j, k in zip(ij[:, 0], ij[:, 1], kj[:, 0]):
                    if (int(i), int(j), int(k)) == tuple(path):
                        print(f"Found {(int(i), int(j), int(k))} == {path}")
                        found = True
                        break

                if not found:
                    print(f"Missing {path}")

                assert found


class TestClusters:

    def test_find_sparse_clusters_with_isolated_particle_jit(self):
        neighbor_idx = jnp.asarray([
            [0, 1],
            [1, 0],
        ], dtype=jnp.int32)
        reference_position = jnp.zeros((3, 2))
        mask = jnp.ones(3, dtype=bool)

        @jax.jit
        def find_cluster_count(idx, position, particle_mask):
            neighbor = NeighborIdx(
                idx=idx,
                format=partition.NeighborListFormat.Sparse,
                reference_position=position,
            )
            return custom_partition.find_clusters(neighbor, particle_mask)[1]

        assert find_cluster_count(neighbor_idx, reference_position, mask) == 2
