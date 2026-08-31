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

"""Utilities to connect models to chemtrain."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.custom_derivatives import SymbolicZero

from typing import Protocol, Any, Tuple


class ApplyFn(Protocol):
    """GNN apply function protocol."""
    def __call__(
        self,
        params: Any,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        edge_features: Tuple[jnp.ndarray],
        node_features: Tuple[jnp.ndarray],
    ) -> jnp.ndarray: ...


def batch_apply_fn(_apply_fn: ApplyFn) -> ApplyFn:
    """Combine vmapped graphs into disconnected supergraphs.

    A mapped parameter leaf cannot be shared by one supergraph evaluation. In
    that case the affected map level is evaluated sequentially, so a model
    primitive is never asked to accept a parameter batch dimension.

    The returned function supports arbitrary nesting of ``vmap`` and
    reverse-mode differentiation. JAX does not support forward-mode
    differentiation of ``custom_vjp`` functions, so applying ``jax.jvp`` to
    this wrapper raises ``TypeError``.
    """

    def apply_fn(params, senders, receivers, edge_features, node_features):
        if not jax.tree.leaves(node_features):
            raise ValueError("At least one node feature array is required")
        return _apply_fn(
            params, senders, receivers, edge_features, node_features
        )

    def make_batch_rule(wrapped):
        def wrapped_batch(
            axis_size,
            in_batched,
            params,
            senders,
            receivers,
            edge_features,
            node_features,
        ):
            in_batched = tuple(in_batched)
            params_batched, *graph_batched = in_batched

            if any(jax.tree.leaves(params_batched)):
                mapped_args = jax.tree.map(
                    lambda value, is_batched: value
                    if is_batched
                    else jnp.broadcast_to(
                        value, (axis_size,) + value.shape
                    ),
                    (
                        params,
                        senders,
                        receivers,
                        edge_features,
                        node_features,
                    ),
                    in_batched,
                )
                output = lax.map(lambda args: wrapped(*args), mapped_args)
                return output, jax.tree.map(lambda _: True, output)

            flat_graph = flatten_graph(
                axis_size,
                graph_batched,
                senders,
                receivers,
                edge_features,
                node_features,
            )
            output = wrapped(params, *flat_graph)
            node_batched_leaves = jax.tree.leaves(graph_batched[-1])
            node_feature_leaves = jax.tree.leaves(node_features)
            if not node_feature_leaves:
                raise ValueError("At least one node feature array is required")
            nodes_batched = node_batched_leaves[0]
            first_node_feature = node_feature_leaves[0]
            num_nodes = (
                first_node_feature.shape[1]
                if nodes_batched
                else first_node_feature.shape[0]
            )
            output = jax.tree.map(
                lambda value: value.reshape(
                    (axis_size, num_nodes) + value.shape[1:]
                ),
                output,
            )
            return output, jax.tree.map(lambda _: True, output)

        return wrapped_batch

    return _reverse_closed(apply_fn, make_batch_rule)


def _sequential_vmap_rule(wrapped):
    """Build a sequential custom-vmap rule for an arbitrary pytree call."""

    def rule(axis_size, in_batched, *args):
        in_batched = tuple(in_batched)
        mapped_args = jax.tree.map(
            lambda value, is_batched: value
            if is_batched
            else jnp.broadcast_to(value, (axis_size,) + value.shape),
            args,
            in_batched,
        )
        output = lax.map(lambda values: wrapped(*values), mapped_args)
        return output, jax.tree.map(lambda _: True, output)

    return rule


def _reverse_closed(fun, make_vmap_rule):
    """Close a function recursively under custom vmap and reverse-mode AD."""

    @jax.custom_vjp
    @jax.custom_batching.custom_vmap
    def wrapped(*args):
        return fun(*args)

    wrapped.fun.def_vmap(make_vmap_rule(wrapped))

    def wrapped_fwd(*args):
        activity = jax.tree.map(lambda value: value.perturbed, args)
        values = jax.tree.map(lambda value: value.value, args)
        return wrapped.fun(*values), (values, activity)

    def wrapped_bwd(residual, output_cotangent):
        args, activity = residual
        flat_args, args_tree = jax.tree.flatten(args)
        flat_activity, activity_tree = jax.tree.flatten(activity)
        if activity_tree != args_tree:
            raise TypeError("Argument and activity pytrees must match")
        active = tuple(
            index for index, perturbed in enumerate(flat_activity) if perturbed
        )

        cotangent_leaves = jax.tree.leaves(
            output_cotangent,
            is_leaf=lambda value: isinstance(value, SymbolicZero),
        )
        if not active or all(
            isinstance(value, SymbolicZero) for value in cotangent_leaves
        ):
            return jax.tree.unflatten(
                args_tree,
                tuple(
                    SymbolicZero.from_primal_value(value)
                    for value in flat_args
                ),
            )

        def materialize(value):
            if isinstance(value, SymbolicZero):
                return jnp.zeros(value.shape, value.dtype)
            return value

        output_cotangent = jax.tree.map(
            materialize,
            output_cotangent,
            is_leaf=lambda value: isinstance(value, SymbolicZero),
        )

        def pullback(packed):
            local_args, local_cotangent = packed
            local_flat_args = jax.tree.leaves(local_args)

            def active_fun(*active_values):
                values = list(local_flat_args)
                for index, value in zip(active, active_values):
                    values[index] = value
                return fun(*jax.tree.unflatten(args_tree, values))

            active_args = tuple(local_flat_args[index] for index in active)
            return jax.vjp(active_fun, *active_args)[1](local_cotangent)

        active_cotangents = _reverse_closed(
            pullback,
            lambda child: _sequential_vmap_rule(child),
        )((args, output_cotangent))
        active_cotangents = iter(active_cotangents)
        cotangents = [
            next(active_cotangents)
            if perturbed
            else SymbolicZero.from_primal_value(value)
            for value, perturbed in zip(flat_args, flat_activity)
        ]
        return jax.tree.unflatten(args_tree, cotangents)

    wrapped.defvjp(wrapped_fwd, wrapped_bwd, symbolic_zeros=True)
    return wrapped


def flatten_graph(
    axis_size,
    in_batched,
    senders,
    receivers,
    edge_features,
    node_features,
):
    """Flatten one mapped graph axis into a disconnected supergraph."""

    bsenders, breceivers, bedge_features, bnode_features = in_batched
    num_graphs = axis_size
    node_feature_leaves = jax.tree.leaves(node_features)
    node_batched_leaves = jax.tree.leaves(bnode_features)
    if not node_feature_leaves:
        raise ValueError("At least one node feature array is required")
    first_node_feature = node_feature_leaves[0]
    natoms = (
        first_node_feature.shape[1]
        if node_batched_leaves[0]
        else first_node_feature.shape[0]
    )

    if bool(bsenders) != bool(breceivers):
        raise ValueError("senders and receivers must be batched together")
    if not bsenders:
        senders = jnp.broadcast_to(
            senders, (num_graphs,) + senders.shape
        )
        receivers = jnp.broadcast_to(
            receivers, (num_graphs,) + receivers.shape
        )

    # Relabel graph-local indices before flattening graph and edge axes. One
    # invalid endpoint invalidates the whole edge and sends both endpoints to
    # the same trailing sentinel node.
    offsets = natoms * jnp.arange(
        num_graphs, dtype=senders.dtype
    )[:, None]
    valid_index = jnp.logical_and(
        jnp.logical_and(senders >= 0, senders < natoms),
        jnp.logical_and(receivers >= 0, receivers < natoms),
    )
    sentinel = num_graphs * natoms
    senders = jnp.where(
        valid_index, senders + offsets, sentinel
    ).reshape(-1)
    receivers = jnp.where(
        valid_index, receivers + offsets, sentinel
    ).reshape(-1)

    def flatten_features(features, batched):
        return jax.tree.map(
            lambda feature, is_batched: (
                feature.reshape((-1,) + feature.shape[2:])
                if is_batched
                else jnp.broadcast_to(
                    feature, (num_graphs,) + feature.shape
                ).reshape((-1,) + feature.shape[1:])
            ),
            features,
            batched,
        )

    return (
        senders,
        receivers,
        flatten_features(edge_features, bedge_features),
        flatten_features(node_features, bnode_features),
    )
