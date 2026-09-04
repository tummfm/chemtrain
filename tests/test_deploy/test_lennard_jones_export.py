# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
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

"""Tests for the Lennard-Jones deployment example."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from chemtrain.deploy import graphs


EXAMPLE_PATH = (
    Path(__file__).parents[2]
    / "examples"
    / "chemtrain-deploy"
    / "export_lennard_jones.py"
)
SPEC = importlib.util.spec_from_file_location("export_lennard_jones", EXAMPLE_PATH)
LJ_EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LJ_EXAMPLE)


def expected_single_pair_energy(position):
    squared_distance = jnp.sum((position[0] - position[1]) ** 2)
    pair_energy = LJ_EXAMPLE._pair_energy(squared_distance, True)
    return jnp.array((0.5 * pair_energy, 0.5 * pair_energy, 0.0))


def assert_finite_energy_and_gradient(model, position, graph):
    def total_energy(coordinates):
        return jnp.sum(model.energy_fn(coordinates, {}, graph))

    energy, gradient = jax.value_and_grad(total_energy)(position)
    assert jnp.isfinite(energy)
    assert jnp.all(jnp.isfinite(gradient))


def test_sparse_lennard_jones_masks_padding_endpoints():
    position = jnp.array(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (4.0, 0.0, 0.0)),
        dtype=jnp.float32,
    )
    graph = graphs.SimpleSparseNeighborList(
        senders=jnp.array((0, 1, -1, 0), dtype=jnp.int32),
        receivers=jnp.array((1, 0, 0, 3), dtype=jnp.int32),
        max_edges=jnp.ones((4,), dtype=jnp.bool_),
    )
    model = LJ_EXAMPLE.SparseLennardJones()

    energy = model.energy_fn(position, {}, graph)

    np.testing.assert_allclose(energy, expected_single_pair_energy(position))
    assert_finite_energy_and_gradient(model, position, graph)


def test_dense_lennard_jones_masks_padding_neighbors():
    position = jnp.array(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (4.0, 0.0, 0.0)),
        dtype=jnp.float32,
    )
    graph = graphs.SimpleDenseNeighborList(
        nbrs=jnp.array(
            ((1, -1, 3), (0, 3, -1), (3, -1, -1)), dtype=jnp.int32
        ),
        max_edges=jnp.ones((9,), dtype=jnp.bool_),
        max_triplets=jnp.ones((1,), dtype=jnp.bool_),
    )
    model = LJ_EXAMPLE.DenseLennardJones()

    energy = model.energy_fn(position, {}, graph)

    np.testing.assert_allclose(energy, expected_single_pair_energy(position))
    assert_finite_energy_and_gradient(model, position, graph)
