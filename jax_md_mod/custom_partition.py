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
import warnings
from typing import Union, Tuple, Callable

import functools

import jax
import jax.numpy as jnp
from jax import Array

import numpy as onp

from jax_md import partition

@functools.wraps(partition.neighbor_list)
def masked_neighbor_list(displacement_or_metric,
                         box,
                         r_cutoff: float,
                         dr_threshold: float = 0.0,
                         capacity_multiplier: float = 1.25,
                         disable_cell_list: bool = False,
                         fractional_coordinates: bool = False,
                         format = partition.NeighborListFormat.Dense,
                         **static_kwargs) -> partition.NeighborFn:
    """Extension of JAX, M.D. neighbor list with masking functionality."""

    if dr_threshold > 0.0:
        warnings.warn(
            "Mask will only be applied if neighbor list must be re-computed. "
            "Setting a too high threshold might lead to unexpected behavior."
        )

    def custom_neighbor_list_fn(position, neighbors=None, mask=None, **kwargs):
        # Enforce neighbor list re-computation if mask changes
        if mask is not None:
            position = jnp.where(mask[:, None], position, 0.0)

        @jax.jit
        def custom_mask_fn(idx):
            # Mask out edges to self
            self_mask = idx == jnp.reshape(
                jnp.arange(idx.shape[0]),(idx.shape[0], 1))

            # Only mask edges to self
            if mask is None:
                return jnp.where(self_mask, idx.shape[0], idx)

            # Mask out all senders
            sender_mask = jax.vmap(
                jnp.logical_or, in_axes=(None, 1), out_axes=1
            )(jnp.logical_not(mask), self_mask)

            # Mask out all receivers
            total_mask = jax.vmap(jnp.logical_or)(
                sender_mask, jnp.logical_not(mask[idx])
            )

            return jnp.where(total_mask, idx.shape[0], idx)

        neighbor_fns = partition.neighbor_list(
            displacement_or_metric,
            box,
            r_cutoff,
            dr_threshold,
            capacity_multiplier,
            disable_cell_list,
            False,
            custom_mask_function=custom_mask_fn,
            fractional_coordinates=fractional_coordinates,
            format=format,
            **static_kwargs
        )

        return neighbor_fns.allocate(position, neighbors=neighbors, **kwargs)

    def init_neighbor_fn(position, extra_capacity: int = 0, mask=None, **kwargs):
        return custom_neighbor_list_fn(position, extra_capacity=extra_capacity, mask=mask, **kwargs)

    @jax.jit
    def update_neighbor_fn(position, nbrs, mask=None, **kwargs):
        return custom_neighbor_list_fn(position, neighbors=nbrs, mask=mask, **kwargs)

    return partition.NeighborListFns(init_neighbor_fn, update_neighbor_fn)


def mask_neighbor_list(nbrs: partition.NeighborList,
                       mask: Array) -> partition.NeighborList:
    """Masks the neighbor list indices.

    Args:
        nbrs: Dense or sparse neighbor list.
        mask: Boolean array masking valid particles (True). Edges from and to
            invalid particles (False) are removed from the neighbor list.

    Returns:
        Returns a neighbor list without edges to invalid particles.

    """

    def mask_dense(idx, mask):
        # Mask out edges to self
        self_mask = idx == jnp.reshape(
            jnp.arange(idx.shape[0]), (idx.shape[0], 1))

        # Only mask edges to self
        if mask is None:
            return jnp.where(self_mask, idx.shape[0], idx)

        # Mask out all senders
        sender_mask = jax.vmap(
            jnp.logical_or, in_axes=(None, 1), out_axes=1
        )(jnp.logical_not(mask), self_mask)

        # Mask out all receivers
        total_mask = jax.vmap(jnp.logical_or)(
            sender_mask, jnp.logical_not(mask[idx])
        )

        return jnp.where(total_mask, idx.shape[0], idx)

    def mask_sparse(idx, mask):
        # Mask out all invalid edges
        edge_mask = jnp.logical_or(
            jnp.logical_not(mask[idx[0]]),
            jnp.logical_not(mask[idx[1]])
        )

        return jnp.where(edge_mask[:, None], nbrs.reference_position.shape[0], idx)

    if nbrs.format == partition.NeighborListFormat.Dense:
        new_idx = mask_dense(nbrs.idx, mask)
    else:
        new_idx = mask_sparse(nbrs.idx, mask)

    new_position = jnp.where(mask[:, None], nbrs.reference_position, 0.0)

    return nbrs.set(idx=new_idx, reference_position=new_position)


def exclude_from_neighbor_list(neighbor: partition.NeighborList,
                               exclude_idx,
                               exclude_mask) -> partition.NeighborList:
    """Function excluding edges from the neighbor list.

    Args:
        neighbor: Neighbor list
        exclude_idx: Indices of the edges that should be excluded, if contained.
        exclude_mask: Boolean array whether specified edges should be excluded.

    Example:

        >>> from pathlib import Path
        >>> root = Path.cwd().parent

        >>> import mdtraj
        >>> from jax_md import space
        >>> from jax import numpy as jnp
        >>> from jax_md_mod.custom_partition import masked_neighbor_list

        >>> pdb = mdtraj.load(root / "examples/data/ethane.pdb")
        >>> r_init = jnp.asarray(pdb.xyz[0], dtype=jnp.float32)
        >>> box = jnp.array(1.0)

        >>> displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=True)
        >>> neighbor_fns = masked_neighbor_list(
        ...     displacement_fn, box, r_cutoff=1.0, dr_threshold=0.05, disable_cell_list=False
        ... )

        We can now exclude, e.g., the first C atom from the neighbor list

        >>> mask = jnp.array([0, 1, 1, 1, 1, 1, 1, 1])
        >>> nbrs_init = neighbor_fns.allocate(r_init, mask=mask)
        >>>
        >>> print(nbrs_init.idx)
        [[8 8 8 8 8 8 8]
         [2 3 4 5 6 7 8]
         [1 3 4 5 6 7 8]
         [1 2 4 5 6 7 8]
         [1 2 3 5 6 7 8]
         [1 2 3 4 6 7 8]
         [1 2 3 4 5 7 8]
         [1 2 3 4 5 6 8]]


        Whenever the neighbor list must be recomputed (dR threshold), a new
        mask is applied

        >>> mask = jnp.array([1, 0, 1, 1, 1, 1, 1, 1])
        >>> print(neighbor_fns.update(r_init, nbrs_init, mask=mask).idx)
        [[2 3 4 5 6 7 8]
         [8 8 8 8 8 8 8]
         [0 3 4 5 6 7 8]
         [0 2 4 5 6 7 8]
         [0 2 3 5 6 7 8]
         [0 2 3 4 6 7 8]
         [0 2 3 4 5 7 8]
         [0 2 3 4 5 6 8]]


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
        for i, j in neighbor.idx.T:
            if i == neighbor.reference_position.shape[0]: continue
            if j == neighbor.reference_position.shape[0]: continue

            graph.add_edge(int(i), int(j))

    return graph
