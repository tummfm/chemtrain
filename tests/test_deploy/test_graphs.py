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
import inspect

import jax
from jax import export
from jax import numpy as jnp, Array

from chemtrain.deploy import graphs

import numpy as onp

import pytest


def _required_directed_edges(senders, receivers, local, order):
    """Independent direction-neutral dependency oracle."""
    senders = onp.asarray(senders)
    receivers = onp.asarray(receivers)
    reached = onp.asarray(local, dtype=bool).copy()
    valid = (
        (senders >= 0) & (senders < reached.size)
        & (receivers >= 0) & (receivers < reached.size)
    )
    for _ in range(max(order - 1, 0)):
        next_reached = reached.copy()
        for sender, receiver in zip(senders[valid], receivers[valid]):
            if reached[sender] or reached[receiver]:
                next_reached[sender] = True
                next_reached[receiver] = True
        reached = next_reached
    required = valid & (reached[senders.clip(0, reached.size - 1)]
                        | reached[receivers.clip(0, reached.size - 1)])
    if order == 0:
        required[:] = False
    return [
        (int(sender), int(receiver))
        for sender, receiver in zip(senders[required], receivers[required])
    ]

class TestPruning:

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_sparse_pruning_matches_direction_neutral_oracle(self, order):
        senders = jnp.asarray([0, 1, 1, 2, 2, 3, 4, 5, 5, 6, -1, 7])
        receivers = jnp.asarray([1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 0, 0])
        local = jnp.asarray([True, False, False, False, False, False, False])
        graph = graphs.SimpleSparseNeighborList(
            senders, receivers, max_edges=senders.size,
            pair_type=jnp.arange(senders.size, dtype=jnp.int32) + 1,
        )

        pruned, n_valid = graphs.prune_neighbor_list(
            graph, local, max_edges=senders.size + 2,
            nbr_order=order, half_list=False,
        )
        valid = onp.asarray(pruned.max_edges)
        actual = list(zip(
            onp.asarray(pruned.senders)[valid].tolist(),
            onp.asarray(pruned.receivers)[valid].tolist(),
        ))
        expected = _required_directed_edges(
            senders, receivers, local, order)
        assert actual == expected
        assert int(n_valid) == len(expected)
        assert onp.all(onp.asarray(pruned.pair_type)[~valid] == 0)

    def test_sparse_half_list_matches_direction_neutral_oracle(self):
        senders = jnp.asarray([0, 1, 2, 4])
        receivers = jnp.asarray([1, 2, 3, 5])
        pair_type = jnp.asarray([1, 2, 3, 4], dtype=jnp.int32)
        local = jnp.asarray([True, False, False, False, False, False])
        graph = graphs.SimpleSparseNeighborList(
            senders, receivers, max_edges=senders.size,
            pair_type=pair_type,
        )

        pruned, n_valid = graphs.prune_neighbor_list(
            graph, local, max_edges=2 * senders.size,
            nbr_order=2, half_list=True,
        )
        directed_senders = jnp.concat([senders, receivers])
        directed_receivers = jnp.concat([receivers, senders])
        expected = _required_directed_edges(
            directed_senders, directed_receivers, local, 2)
        valid = onp.asarray(pruned.max_edges)
        actual = list(zip(
            onp.asarray(pruned.senders)[valid].tolist(),
            onp.asarray(pruned.receivers)[valid].tolist(),
        ))
        assert actual == expected
        assert int(n_valid) == len(expected)
        assert set(onp.asarray(pruned.pair_type)[valid].tolist()) == {1, 2}

    def test_sparse_compaction_reports_overflow_and_pads_with_sentinel(self):
        graph = graphs.SimpleSparseNeighborList(
            jnp.asarray([0, 1, 1, 2]),
            jnp.asarray([1, 0, 2, 1]),
            max_edges=4,
            pair_type=jnp.asarray([1, 1, 2, 2], dtype=jnp.int32),
        )
        local = jnp.asarray([True, False, False])

        overflowed, n_valid = graphs.prune_neighbor_list(
            graph, local, max_edges=1, nbr_order=2, half_list=False)
        assert int(n_valid) == 4
        assert onp.array_equal(onp.asarray(overflowed.senders), [0])
        assert onp.array_equal(onp.asarray(overflowed.receivers), [1])

        padded, n_valid = graphs.prune_neighbor_list(
            graph, local, max_edges=6, nbr_order=2, half_list=False)
        assert int(n_valid) == 4
        assert onp.array_equal(onp.asarray(padded.senders)[4:], [3, 3])
        assert onp.array_equal(onp.asarray(padded.receivers)[4:], [3, 3])
        assert onp.all(~onp.asarray(padded.max_edges)[4:])
        assert onp.all(onp.asarray(padded.pair_type)[4:] == 0)

    def test_sparse_create_rejects_invalid_and_padding_endpoints(self):
        graph, _ = graphs.SimpleSparseNeighborList.create_from_args(
            5.0, 1,
            jnp.zeros((4, 3)),
            jnp.asarray([True, False, False, False]),
            jnp.asarray([True, True, False, False]),
            True,
            jnp.asarray([0, 1, 2, 4, -1]),
            jnp.asarray([1, 0, 0, 0, 0]),
            jnp.asarray([1, 1, 2, 3, 3], dtype=jnp.int32),
            jnp.ones(5, dtype=bool),
        )
        valid = onp.asarray(graph.max_edges)
        assert list(zip(
            onp.asarray(graph.senders)[valid].tolist(),
            onp.asarray(graph.receivers)[valid].tolist(),
        )) == [(0, 1), (1, 0)]
        assert onp.all(onp.asarray(graph.pair_type)[~valid] == 0)

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_dense_pruning_matches_direction_neutral_oracle(self, order):
        invalid = 6
        nbrs = jnp.asarray([
            [1, invalid], [0, 2], [1, 3],
            [2, invalid], [5, invalid], [4, invalid],
        ])
        pair_type = jnp.where(nbrs < invalid, 1, 7).astype(jnp.int32)
        local = jnp.asarray([True, False, False, False, False, False])
        graph = graphs.SimpleDenseNeighborList(
            nbrs, jnp.ones(12, dtype=bool), jnp.ones(24, dtype=bool),
            pair_type,
        )

        pruned, (n_edges, _) = graphs.prune_neighbor_list_dense(
            graph, local, nbr_order=order)
        rows = onp.repeat(onp.arange(invalid), nbrs.shape[1])
        expected = _required_directed_edges(
            rows, onp.asarray(nbrs).ravel(), local, order)
        actual = [
            (row, int(neighbor))
            for row, row_nbrs in enumerate(onp.asarray(pruned.nbrs))
            for neighbor in row_nbrs if neighbor < invalid
        ]
        assert actual == expected
        assert int(n_edges) == len(expected)
        assert onp.all(onp.asarray(pruned.pair_type)[
            onp.asarray(pruned.nbrs) >= invalid] == 0)

    @pytest.mark.parametrize("newton_pair", [False, True])
    def test_sparse_symbolic_export_contains_no_sort(self, newton_pair):
        n_atoms, max_buffers, max_edges = export.symbolic_shape(
            "n_atoms,max_buffers,max_edges",
            constraints=(
                "n_atoms >= 1", "max_buffers >= 1", "max_edges >= 1",
                "max_edges <= 2 * max_buffers",
            ),
        )

        def build(position, local, valid, senders, receivers, output):
            graph, _ = graphs.SimpleSparseNeighborList.create_from_args(
                5.0, 2, position, local, valid, newton_pair,
                senders, receivers, output,
            )
            return graph.senders, graph.receivers, graph.max_edges

        exported = export.export(jax.jit(build))(
            jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32),
            jax.ShapeDtypeStruct((n_atoms,), jnp.bool_),
            jax.ShapeDtypeStruct((n_atoms,), jnp.bool_),
            jax.ShapeDtypeStruct((max_buffers,), jnp.int32),
            jax.ShapeDtypeStruct((max_buffers,), jnp.int32),
            jax.ShapeDtypeStruct((max_edges,), jnp.bool_),
        )
        stablehlo = str(exported.mlir_module())
        assert "stablehlo.case" not in stablehlo
        assert "stablehlo.sort" not in stablehlo

    @pytest.mark.parametrize("scatter_to_receivers", [False, True])
    @pytest.mark.parametrize("newton", [False, True])
    def test_sparse_pruning_preserves_directional_model_forces(
            self, scatter_to_receivers, newton):
        n_atoms = 7
        senders = jnp.asarray([
            0, 1, 1, 2, 2, 3, 3, 4, 5, 6,
        ])
        receivers = jnp.asarray([
            1, 0, 2, 1, 3, 2, 4, 3, 6, 5,
        ])
        local = jnp.asarray(
            [True, False, False, False, False, False, False])
        positions = jnp.arange(n_atoms, dtype=jnp.float32)[:, None]
        full = graphs.SimpleSparseNeighborList(
            senders, receivers, jnp.ones(senders.shape, dtype=bool))
        layers = 2
        order = layers if newton else 2 * layers
        pruned, _ = graphs.prune_neighbor_list(
            full, local, max_edges=senders.size,
            nbr_order=order, half_list=False)

        def per_atom_energy(position, graph):
            valid = (
                graph.max_edges
                & (graph.senders >= 0) & (graph.senders < n_atoms)
                & (graph.receivers >= 0) & (graph.receivers < n_atoms)
            )
            edge_senders = jnp.clip(graph.senders, 0, n_atoms - 1)
            edge_receivers = jnp.clip(graph.receivers, 0, n_atoms - 1)
            features = position[:, 0]
            targets = edge_receivers if scatter_to_receivers else edge_senders
            for _ in range(layers):
                displacement = (
                    position[edge_senders, 0]
                    - position[edge_receivers, 0]
                )
                messages = valid * (
                    0.3 * features[edge_senders] + displacement ** 2)
                updates = jax.ops.segment_sum(
                    messages, targets, num_segments=n_atoms)
                features = features + jnp.tanh(updates)
            return features

        full_energy = per_atom_energy(positions, full)
        pruned_energy = per_atom_energy(positions, pruned)
        assert onp.allclose(
            onp.asarray(full_energy[local]),
            onp.asarray(pruned_energy[local]),
            rtol=1.0e-6, atol=1.0e-6,
        )

        def force_objective(position, graph):
            energy = per_atom_energy(position, graph)
            return jnp.sum(jnp.where(
                local if newton else jnp.ones_like(local), energy, 0.0))

        full_force = -jax.grad(force_objective)(positions, full)
        pruned_force = -jax.grad(force_objective)(positions, pruned)
        compared_rows = jnp.ones_like(local) if newton else local
        assert onp.allclose(
            onp.asarray(full_force[compared_rows]),
            onp.asarray(pruned_force[compared_rows]),
            rtol=1.0e-5, atol=1.0e-5,
        )

    def test_device_sparse_uses_exporter_graph_argument_contract(self):
        parameters = list(inspect.signature(
            graphs.DeviceSparseNeighborList.create_from_args
        ).parameters)
        assert parameters[:6] == [
            "r_cutoff", "nbr_order", "positions", "local_mask",
            "valid_mask", "newton_pair",
        ]

    def test_sparse_pair_types_follow_half_list_mirroring_and_pruning(self):
        graph, _ = graphs.SimpleSparseNeighborList.create_from_args(
            5.0,
            1,
            jnp.asarray([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [2.0, 0.0, 0.0]]),
            jnp.asarray([True, True, True]),
            jnp.asarray([True, True, True]),
            False,
            jnp.asarray([0, 1]),
            jnp.asarray([1, 2]),
            jnp.asarray([1, 3], dtype=jnp.int32),
            jnp.ones(4, dtype=bool),
        )

        valid = onp.asarray(graph.max_edges)
        actual = set(zip(
            onp.asarray(graph.senders)[valid].tolist(),
            onp.asarray(graph.receivers)[valid].tolist(),
            onp.asarray(graph.pair_type)[valid].tolist(),
        ))
        assert actual == {(0, 1, 1), (1, 0, 1), (1, 2, 3), (2, 1, 3)}

    def test_sparse_pair_types_are_zero_for_cutoff_and_padding(self):
        graph, _ = graphs.SimpleSparseNeighborList.create_from_args(
            1.5,
            1,
            jnp.asarray([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [4.0, 0.0, 0.0]]),
            jnp.asarray([True, True, True]),
            jnp.asarray([True, True, True]),
            False,
            jnp.asarray([0, 1]),
            jnp.asarray([1, 2]),
            jnp.asarray([1, 3], dtype=jnp.int32),
            jnp.ones(4, dtype=bool),
        )

        assert onp.all(onp.asarray(graph.pair_type)[~onp.asarray(graph.max_edges)] == 0)

    def test_dense_pair_types_follow_pruning_mask(self):
        neighbor_list = graphs.SimpleDenseNeighborList(
            nbrs=jnp.asarray([[1, 3], [0, 2], [1, 3]]),
            max_edges=jnp.ones(4, dtype=bool),
            max_triplets=jnp.ones(2, dtype=bool),
            pair_type=jnp.asarray([[1, 0], [1, 2], [2, 0]], dtype=jnp.int32),
        )
        pruned, _ = graphs.prune_neighbor_list_dense(
            neighbor_list,
            jnp.asarray([True, False, False]),
            nbr_order=1,
        )

        valid = onp.asarray(pruned.nbrs) < 3
        assert onp.all(onp.asarray(pruned.pair_type)[~valid] == 0)
        assert onp.array_equal(
            onp.asarray(pruned.pair_type)[valid],
            onp.asarray(neighbor_list.pair_type)[valid],
        )

    def test_dense_pair_type_payload_masks_cutoff_and_padding(self):
        graph, _ = graphs.SimpleDenseNeighborList.create_from_args(
            1.5,
            1,
            jnp.asarray([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [4.0, 0.0, 0.0]]),
            jnp.asarray([True, True, True]),
            jnp.asarray([True, True, True]),
            True,
            jnp.asarray([[1, 3], [0, 2], [1, 3]], dtype=jnp.int32),
            jnp.asarray([[1, 3], [1, 2], [2, 3]], dtype=jnp.int32),
            jnp.ones(6, dtype=bool),
            jnp.ones(6, dtype=bool),
        )

        nbrs = onp.asarray(graph.nbrs)
        pair_type = onp.asarray(graph.pair_type)
        assert pair_type[0, 0] == 1
        assert pair_type[1, 0] == 1
        assert onp.all(pair_type[nbrs >= 3] == 0)

    def test_prune_truncates_in_source_order(self):
        """Boolean compaction retains the first valid edges on overflow."""
        neighbor_list = graphs.SimpleSparseNeighborList(
            senders=jnp.asarray([0, 0, 0]),
            receivers=jnp.asarray([1, 2, 3]),
            max_edges=3,
        )
        local = jnp.asarray([True, False, False, False])

        pruned, n_valid = graphs.prune_neighbor_list(
            neighbor_list,
            local,
            max_edges=2,
            nbr_order=1,
            half_list=False,
        )

        assert n_valid == 3
        onp.testing.assert_array_equal(pruned.senders, jnp.asarray([0, 0]))
        onp.testing.assert_array_equal(pruned.receivers, jnp.asarray([1, 2]))
        onp.testing.assert_array_equal(
            pruned.max_edges, jnp.asarray([True, True])
        )

    def test_prune_keeps_ghost_sources_of_local_receivers(self):
        neighbor_list = graphs.SimpleSparseNeighborList(
            senders=jnp.asarray([2, 3]),
            receivers=jnp.asarray([0, 3]),
            max_edges=2,
        )
        local = jnp.asarray([True, False, False, False])

        pruned, n_valid = graphs.prune_neighbor_list(
            neighbor_list,
            local,
            max_edges=neighbor_list.max_edges,
            nbr_order=1,
            half_list=False,
        )

        assert n_valid == 1
        assert onp.any(
            (onp.asarray(pruned.senders) == 2)
            & (onp.asarray(pruned.receivers) == 0)
        )

    def test_prune_keeps_ghost_receivers_of_local_senders(self):
        neighbor_list = graphs.SimpleSparseNeighborList(
            senders=jnp.asarray([0, 3]),
            receivers=jnp.asarray([2, 3]),
            max_edges=2,
        )
        local = jnp.asarray([True, False, False, False])

        pruned, n_valid = graphs.prune_neighbor_list(
            neighbor_list,
            local,
            max_edges=neighbor_list.max_edges,
            nbr_order=1,
            half_list=False,
        )

        assert n_valid == 1
        assert onp.any(
            (onp.asarray(pruned.senders) == 0)
            & (onp.asarray(pruned.receivers) == 2)
        )

    @pytest.mark.parametrize(
        "list, local, order, pruned", [
            (
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 2]),
                    jnp.asarray([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 0]),
                    max_edges=12
                ),
                jnp.asarray([1, 0, 0, 0, 0, 0], dtype=bool), 1,
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([0, 1, 0, 2, 6, 6, 6, 6, 6, 6, 6, 6]),
                    jnp.asarray([1, 0, 2, 0, 6, 6, 6, 6, 6, 6, 6, 6]),
                    max_edges=4
                ),
            ),
            (
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 2]),
                    jnp.asarray([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 0]),
                    max_edges=12
                ),
                jnp.asarray([0, 0, 0, 1, 0, 0], dtype=bool), 1,
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([2, 3, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6]),
                    jnp.asarray([3, 2, 4, 3, 6, 6, 6, 6, 6, 6, 6, 6]),
                    max_edges=4
                ),
            ),
            (
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 2]),
                    jnp.asarray([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 2, 0]),
                    max_edges=12
                ),
                jnp.asarray([0, 0, 0, 0, 4, 0], dtype=bool), 2,
                graphs.SimpleSparseNeighborList(
                    jnp.asarray([2, 3, 3, 4, 4, 5, 6, 6, 6, 6, 6, 6]),
                    jnp.asarray([3, 2, 4, 3, 5, 4, 6, 6, 6, 6, 6, 6]),
                    max_edges=6
                ),
            )
        ]
    )
    def test_prune_unreachable(self, list, local, order, pruned):

        pruned_list, max_edges = graphs.prune_neighbor_list(
            list, local, max_edges=list.max_edges, nbr_order=order, half_list=False)

        print(f"Senders:   {pruned_list.senders}")
        print(f"Receivers: {pruned_list.receivers}")

        assert onp.all(pruned_list.senders == pruned.senders)
        assert onp.all(pruned_list.receivers == pruned.receivers)
        assert max_edges == pruned.max_edges
