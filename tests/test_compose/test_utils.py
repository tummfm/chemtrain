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

"""Tests for model composition utilities."""

import jax
import jax.numpy as jnp
import pytest

from chemtrain.compose import utils


def graph_model(params, senders, receivers, edge_features, node_features):
    """Return node values with incoming edge contributions."""
    del senders
    (edge_weights,) = edge_features
    (nodes,) = node_features
    messages = params["edge_scale"] * edge_weights
    incoming = jax.ops.segment_sum(
        messages, receivers, num_segments=nodes.shape[0]
    )
    return params["node_scale"] * nodes + incoming


@pytest.fixture
def graph_batch():
    """Provide two equally sized graphs, including one padded edge."""
    params = {
        "edge_scale": jnp.asarray(0.5),
        "node_scale": jnp.asarray(2.0),
    }
    senders = jnp.asarray([[0, 1, 3], [1, 2, 3]])
    receivers = jnp.asarray([[1, 2, 3], [0, 1, 3]])
    edge_weights = jnp.asarray([[1.0, 2.0, 50.0], [3.0, 4.0, 60.0]])
    nodes = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return params, senders, receivers, edge_weights, nodes


def test_batch_apply_matches_vmap_and_reverse_mode(graph_batch):
    """The supergraph path preserves values and reverse-mode derivatives."""
    params, senders, receivers, edge_weights, nodes = graph_batch
    wrapped = utils.batch_apply_fn(graph_model)

    reference = jax.vmap(graph_model, in_axes=(None, 0, 0, (0,), (0,)))
    combined = jax.vmap(wrapped, in_axes=(None, 0, 0, (0,), (0,)))
    expected = reference(params, senders, receivers, (edge_weights,), (nodes,))
    actual = jax.jit(combined)(
        params, senders, receivers, (edge_weights,), (nodes,)
    )

    assert jnp.allclose(actual, expected)

    def loss(model, node_values):
        values = combined(
            model, senders, receivers, (edge_weights,), (node_values,)
        )
        return jnp.sum(values**2)

    expected_grad = jax.grad(
        lambda model, node_values: jnp.sum(
            reference(
                model,
                senders,
                receivers,
                (edge_weights,),
                (node_values,),
            )
            ** 2
        ),
        argnums=(0, 1),
    )(params, nodes)
    actual_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))(params, nodes)
    assert jax.tree.all(
        jax.tree.map(jnp.allclose, actual_grad, expected_grad)
    )


def test_batch_apply_supports_mapped_parameters(graph_batch):
    """Mapped parameters use the sequential fallback at that map level."""
    params, senders, receivers, edge_weights, nodes = graph_batch
    params = jax.tree.map(lambda value: jnp.stack((value, 2 * value)), params)
    wrapped = utils.batch_apply_fn(graph_model)

    expected = jax.vmap(graph_model, in_axes=(0, 0, 0, (0,), (0,)))(
        params, senders, receivers, (edge_weights,), (nodes,)
    )
    actual = jax.jit(
        jax.vmap(wrapped, in_axes=(0, 0, 0, (0,), (0,)))
    )(params, senders, receivers, (edge_weights,), (nodes,))

    assert jnp.allclose(actual, expected)


def test_batch_apply_requires_node_features():
    """A graph needs node features to determine its node count."""
    wrapped = utils.batch_apply_fn(graph_model)
    with pytest.raises(ValueError, match="node feature"):
        wrapped({}, jnp.asarray([]), jnp.asarray([]), (), ())
