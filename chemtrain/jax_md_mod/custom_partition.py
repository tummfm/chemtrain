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

"""Custom functions to analyze the neighborlist graph."""
import importlib
from typing import Union, Tuple, Callable

import functools

import jax
import jax.numpy as jnp

import numpy as onp

from jax_md import space, util, partition


def exclude_from_neighbor_list(neighbor: partition.NeighborList,
                               exclude_idx,
                               exclude_mask) -> partition.NeighborList:
    """Function excluding edges from the neighbor list.

    Args:
        neighbor: Neighbor list
        exclude_idx: Indices of the edges that should be excluded, if contained.
        exclude_mask: Boolean array whether specified edges should be excluded.

    Returns:
        Returns a neighbor list of the same format with excluded edges.

    """

    @functools.partial(jax.vmap, in_axes=(0, 0, None, None, None, None))
    def _exclude(particle_nbrs, idx, invalid, mask, ref_i, ref_j):
        # Check for all neighbors whether they are already part of the bond
        # list.
        exclude = jax.vmap(
            # Map over all pairs i, j that are part of the neighbor list
            lambda i: jnp.any(
                # Check whether these pairs appear in the bond list
                jnp.logical_and(
                    mask,
                    jnp.logical_or(
                        jnp.logical_and(i == ref_i, idx == ref_j),
                        jnp.logical_and(i == ref_j, idx == ref_i)
                    )
                )
            )
        )(particle_nbrs)
        return jnp.where(exclude, invalid, particle_nbrs)

    @functools.partial(jax.vmap, in_axes=(1, None, None, None, None),
                       out_axes=1)
    def _exclude_sparse(nbr_idx, invalid, ref_i, ref_j, mask):
        # Map through the entries in the neighbor list and exclude them if
        # they are part of the bond list
        exclude = jnp.logical_or(
            jnp.logical_and(nbr_idx[0] == ref_i, nbr_idx[1] == ref_j),
            jnp.logical_and(nbr_idx[0] == ref_j, nbr_idx[1] == ref_i)
        )
        # Mask out invalid bonds or angles
        exclude = jnp.any(jnp.logical_and(exclude, mask))
        return jnp.where(exclude, invalid, nbr_idx)

    # Call the respective function depending on the format of the neighbor list

    invalid_idx = neighbor.idx.shape[0]
    if neighbor.format == partition.NeighborListFormat.Dense:
        new_idx = _exclude(
            neighbor.idx, jnp.arange(invalid_idx), invalid_idx, exclude_mask,
            exclude_idx[:, 0], exclude_idx[:, 1]
        )
    else:
        new_idx = _exclude_sparse(
            neighbor.idx, invalid_idx, exclude_idx[:, 0], exclude_idx[:, 1],
            exclude_mask
        )

    return neighbor.set(idx=new_idx)

def get_triplet_indices(neighbor: partition.NeighborList):
    """Returns indices for all triplets of the neighbor list."""

    @functools.partial(jax.vmap, in_axes=(None, 0), out_axes=-1)
    def _get_triplets(idx, j):
        max_nbrs = idx.shape[1]

        # Return all bonds idx to j
        to_j = idx[j, :]

        # All permutations (remove diagonal entries)
        diagonal_mask = onp.mod(onp.arange(max_nbrs ** 2), max_nbrs + 1) != 0
        ik_to_j = jnp.stack(jnp.meshgrid(to_j, to_j, indexing='ij'), axis=0)
        ik_to_j = ik_to_j.reshape((2, -1))[:, diagonal_mask]

        # Add the reference to the center atom
        ij = ik_to_j.at[1, :].set(j)
        jk = ik_to_j.at[0, :].set(j)

        return ij, jk

    if neighbor.format == neighbor.format.Dense:
        invalid_idx = neighbor.idx.shape[0]
        ij, jk = _get_triplets(neighbor.idx, jnp.arange(neighbor.idx.shape[0]))
        ij = ij.reshape((2, -1)).swapaxes(0, 1)
        jk = jk.reshape((2, -1)).swapaxes(0, 1)

        # Mask out all invalid triplets
        mask = jax.vmap(jnp.logical_and)(
            jnp.all(ij != invalid_idx, axis=-1),
            jnp.all(jk != invalid_idx, axis=-1),
        )

        # Sort (simpler to later prune the triplet array)
        order = jnp.argsort(-1.0 * mask)
        ij = ij[order, :]
        jk = jnp.flip(jk, axis=-1)[order, :]
        mask = mask[order]

        return ij, jk, mask
    else:
        raise NotImplementedError(
            f"Neighbor list format {neighbor.format} not yet supported."
        )


def to_networkx(neighbor: partition.NeighborList):
    nx = importlib.import_module('networkx')

    graph = nx.Graph()

    if neighbor.format == partition.NeighborListFormat.Dense:
        num_particles, max_neighbors = neighbor.idx.shape
        for i in range(num_particles):
            for j in range(max_neighbors):
                if neighbor.idx[i, j] == num_particles: continue

                graph.add_edge(int(i), int(neighbor.idx[i, j]))
    else:
        raise NotImplementedError(
            f"Neighbor list format {neighbor.format} not yet supported."
        )

    return graph
