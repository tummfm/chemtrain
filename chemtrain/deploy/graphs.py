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

"""Graphs for exporting potential and force models."""

import abc
import functools
import typing
from itertools import product

import numpy as onp

import jax
from jax import export, numpy as jnp, lax

import jax_md_mod
from jax_md import partition, dataclasses, smap, space

from typing import NamedTuple, Tuple

from . import util
from ._protobuf import model_pb2 as model_proto


# Does not have to be typed
ListStatistics = typing.Dict


@dataclasses.dataclass
class NeighborList(metaclass=abc.ABCMeta):
    """Abstract class for neighbor list graphs."""

    @staticmethod
    @abc.abstractmethod
    def set_properties(
        proto: model_proto.Model, *, include_pair_type=False, newton_pair=True
    ):
        """Assigns the graph type to the protobuf message."""
        pass

    @staticmethod
    @util.define_symbols("")
    @abc.abstractmethod
    def create_symbolic_input_format(*args, **kwargs):
        """Create the capacity-polymorphic raw inputs for this graph type.

        Concrete graph types declare their symbolic capacity arguments with
        :func:`~chemtrain.deploy.util.define_symbols`. Returned arrays form the
        neighbor arrays consumed by :meth:`create_from_args`.
        """

    @staticmethod
    def create_from_args(r_cutoff,
                         nbr_order,
                         position,
                         local_mask,
                         valid_mask,
                         newton_pair,
                         *args) -> Tuple["NeighborList", "ListStatistics"]:
        """Build a model graph from engine-provided neighbor arrays.

        ``local_mask`` selects atoms owned by the current rank, while
        ``valid_mask`` also includes its real ghost atoms. ``newton_pair`` is
        fixed for one exported model variant.
        """


@dataclasses.dataclass
class SimpleSparseNeighborList(NeighborList):
    """Represents a precomputed neighbor list as a sparse graph.

    A simulation engine such as LAMMPS supplies the graph instead of asking
    chemtrain to infer neighbors from positions. Before model evaluation,
    chemtrain removes edges beyond the model cutoff and ghost-only edges that
    cannot affect owned forces. Pruning reduces both computation and data
    transfer.

    Attributes:
        senders: The sender indices of the edges.
        receivers: The receiver indices of the edges.
        max_edges: Internally computed Boolean mask selecting relevant edges.
            The corresponding exported input is only a shape carrier: its
            length sets the static pruning capacity and its values are ignored.
        pair_type: Optional topology category aligned with ``senders`` and
            ``receivers``. Invalid and padding edges have category zero.

    """

    senders: jax.Array
    receivers: jax.Array

    max_edges: jax.Array
    pair_type: jax.Array | None = None

    @staticmethod
    def set_properties(
        proto: model_proto.Model, *, include_pair_type=False, newton_pair=True
    ):
        proto.neighbor_list.type = model_proto.Model.NeighborListType.SIMPLE_SPARSE
        proto.neighbor_list.half_list = not newton_pair
        proto.neighbor_list.include_pair_type = include_pair_type
        requirements = getattr(proto.neighbor_list, "capacity_requirements", None)
        if requirements is not None:
            raw_edges = requirements.add()
            raw_edges.role = 1  # RAW_EDGES
            raw_edges.symbol = "max_buffers"
            pruned_edges = requirements.add()
            pruned_edges.role = 3  # PRUNED_EDGES
            pruned_edges.symbol = "max_edges"
            pruned_edges.constraints.append("max_edges <= 2 * max_buffers")

    @staticmethod
    @util.define_symbols(
        "max_buffers, max_edges",
        ["max_edges <= 2 * max_buffers"]
    )
    def create_symbolic_input_format(max_buffers, max_edges, *, include_pair_type=False, **kwargs):

        senders = jax.ShapeDtypeStruct((max_buffers,), jnp.int32)
        receivers = jax.ShapeDtypeStruct((max_buffers,), jnp.int32)
        buffer = jax.ShapeDtypeStruct((max_edges,), jnp.bool_)

        if include_pair_type:
            pair_type = jax.ShapeDtypeStruct((max_buffers,), jnp.int32)
            return senders, receivers, pair_type, buffer
        return senders, receivers, buffer

    @staticmethod
    def create_from_args(r_cutoff,
                         nbr_order,
                         position,
                         local_mask,
                         valid_mask,
                         newton_pair,
                         *args) -> Tuple["SimpleSparseNeighborList",
                                         "NeighborListStatistics"]:
        invalid_idx = position.shape[0]

        # The engine ABI stores the neighbor-list row/central atom as sender
        # and the listed neighbor as receiver. Newton on is treated as a full
        # directed list. Newton off is treated as a half-list representation
        # that pruning expands below.
        if len(args) == 4:
            senders, receivers, pair_type, m = args
        else:
            senders, receivers, m = args
            pair_type = None
        max_edges = m.size

        # Sanitize engine padding before gathering positions. In-range capacity
        # padding is excluded by valid_mask
        in_bounds = (
            (senders >= 0)
            & (senders < position.shape[0])
            & (receivers >= 0)
            & (receivers < position.shape[0])
        )
        safe_senders = jnp.where(in_bounds, senders, 0)
        safe_receivers = jnp.where(in_bounds, receivers, 0)
        valid_endpoints = (
            in_bounds
            & valid_mask[safe_senders]
            & valid_mask[safe_receivers]
        )

        # Remove all edges that are longer than the cutoff distance.
        dists = jnp.linalg.norm(
            position[safe_senders] - position[safe_receivers], axis=-1)
        invalid = ~valid_endpoints | (dists >= r_cutoff)

        vs = jnp.where(invalid, invalid_idx, senders)
        vr = jnp.where(invalid, invalid_idx, receivers)
        if pair_type is not None:
            pair_type = jnp.where(invalid, jnp.int32(0), pair_type)

        # Prune all irrelevant edges. In the newton setting, the provided
        # neighbor list is a full list.
        graph = SimpleSparseNeighborList(vs, vr, m, pair_type)
        graph, max_neighbors = prune_neighbor_list(
            graph,
            local_mask,
            max_edges=max_edges,
            nbr_order=nbr_order,
            half_list=not newton_pair,
        )

        statistics = NeighborListStatistics(
            max_neighbors=max_neighbors,
            overlong=jnp.sum(~invalid),
        )

        return graph, statistics

    def to_neighborlist(self):
        idx = jnp.stack([self.senders, self.receivers], axis=0)
        nbrs = partition.NeighborList(
            idx, None, None, None, None, partition.Sparse, None, None, None)
        return nbrs


@dataclasses.dataclass
class SimpleDenseNeighborList(NeighborList):
    """Represents a precomputed neighbor list as a dense graph.

    A simulation engine such as LAMMPS supplies one row per central atom
    instead of asking chemtrain to infer neighbors from positions. Before
    model evaluation, chemtrain removes entries beyond the model cutoff and
    ghost-only entries that cannot affect owned forces. Pruning reduces both
    computation and data transfer.

    Attributes:
        nbrs: Dense matrix of receiver indices, one row per sender.
        max_edges: Internally computed Boolean edge mask. The corresponding
            exported input only carries its static capacity. Values are ignored.
        max_triplets: Internally computed Boolean triplet mask. The corresponding
            exported input only carries its static capacity. Values are ignored.
        pair_type: Optional topology category with the same shape as ``nbrs``.
            Invalid and padding entries have category zero.

    """

    nbrs: jax.Array

    max_edges: jax.Array
    max_triplets: jax.Array
    pair_type: jax.Array | None = None

    @staticmethod
    def set_properties(
        proto: model_proto.Model, *, include_pair_type=False, newton_pair=True
    ):
        proto.neighbor_list.type = proto.NeighborListType.SIMPLE_DENSE
        proto.neighbor_list.half_list = False
        proto.neighbor_list.include_pair_type = include_pair_type
        requirements = getattr(proto.neighbor_list, "capacity_requirements", None)
        if requirements is not None:
            max_nbrs = requirements.add()
            max_nbrs.role = 2  # MAX_NEIGHBORS_PER_ATOM
            max_nbrs.symbol = "max_nbrs"
            edge_mask = requirements.add()
            edge_mask.role = 3  # PRUNED_EDGES
            edge_mask.symbol = "max_edges"
            edge_mask.constraints.append("max_edges <= n_atoms * max_nbrs")
            triplet_mask = requirements.add()
            triplet_mask.role = 4  # PRUNED_TRIPLETS
            triplet_mask.symbol = "max_triplets"
            triplet_mask.constraints.append("max_triplets <= max_edges * max_nbrs")

    @staticmethod
    @util.define_symbols(
        "max_nbrs, max_edges, max_triplets",
        [
            "max_nbrs <= n_atoms",
            "max_edges <= n_atoms * max_nbrs",
            "max_triplets <= max_edges * max_nbrs"
        ]
    )
    def create_symbolic_input_format(max_nbrs, max_edges, max_triplets, *, include_pair_type=False, **kwargs):

        nbrs = jax.ShapeDtypeStruct((kwargs["n_atoms"], max_nbrs), jnp.int32)
        max_edges = jax.ShapeDtypeStruct((max_edges,), jnp.bool_)
        max_triplets = jax.ShapeDtypeStruct((max_triplets,), jnp.bool_)

        if include_pair_type:
            pair_type = jax.ShapeDtypeStruct(nbrs.shape, jnp.int32)
            return nbrs, pair_type, max_edges, max_triplets
        return nbrs, max_edges, max_triplets

    @staticmethod
    def create_from_args(r_cutoff,
                         nbr_order,
                         position,
                         local_mask,
                         valid_mask,
                         newton_pair,
                         *args) -> Tuple["SimpleDenseNeighborList",
                                         "NeighborListStatistics"]:
        invalid_idx = position.shape[0]

        # Each engine row is the central/sender atom, and its entries are
        # receiver neighbors. Dense rows preserve this orientation in both
        # Newton modes. Pruning follows graph dependencies in both directions.
        if len(args) == 4:
            nbrs, pair_type, max_edges, max_triplets = args
        else:
            nbrs, max_edges, max_triplets = args
            pair_type = None

        rows = jnp.arange(nbrs.shape[0])[:, None]
        in_bounds = (nbrs >= 0) & (nbrs < position.shape[0])
        safe_nbrs = jnp.where(in_bounds, nbrs, 0)
        valid_endpoints = (
            in_bounds
            & valid_mask[:, None]
            & valid_mask[safe_nbrs]
        )

        # Remove all edges that are longer than the cutoff distance.
        dists = jax.vmap(
            jax.vmap(
                lambda i, j: jnp.linalg.norm(position[i] - position[j]),
                in_axes=(None, 0)
            ), in_axes=(0, 0)
        )(rows[:, 0], safe_nbrs)
        invalid = ~valid_endpoints | (dists >= r_cutoff)

        nbrs = jnp.where(invalid, invalid_idx, nbrs)
        if pair_type is not None:
            pair_type = jnp.where(invalid, jnp.int32(0), pair_type)

        # Prune all irrelevant edges. In the newton setting, the provided
        # neighbor list is a full list.
        graph = SimpleDenseNeighborList(nbrs, max_edges, max_triplets, pair_type)
        graph, (max_edges, max_triplets) = prune_neighbor_list_dense(
            graph, local_mask, nbr_order=nbr_order
        )

        statistics = NeighborListStatistics(
            max_neighbors=max_edges,
            overlong=max_triplets
        )

        return graph, statistics

    def to_neighborlist(self):
        nbrs = partition.NeighborList(
            self.nbrs, None, None, None, None, partition.Dense, None, None, None)
        return nbrs



class DeviceSparseNeighborListArgs(NamedTuple):
    update: jax.Array | jax.ShapeDtypeStruct

    xcells: jax.Array | jax.ShapeDtypeStruct
    ycells: jax.Array | jax.ShapeDtypeStruct
    zcells: jax.Array | jax.ShapeDtypeStruct
    capacity: jax.Array | jax.ShapeDtypeStruct

    # ref_pos: jax.Array | jax.ShapeDtypeStruct

    # cutoff: jax.Array | jax.ShapeDtypeStruct
    # skin: jax.Array | jax.ShapeDtypeStruct

    senders: jax.Array | jax.ShapeDtypeStruct
    receivers: jax.Array | jax.ShapeDtypeStruct


@dataclasses.dataclass
class DeviceSparseNeighborList(NeighborList):
    """Creates the neighbor list graph on the device using a cell list.

    The implementation is experimental, remains a work in progress, and is
    not supported by the current connector.

    """

    @staticmethod
    def set_properties(
        proto: model_proto.Model, *, include_pair_type=False, newton_pair=True
    ):
        if include_pair_type:
            raise ValueError(
                "DeviceSparseNeighborList does not support pair types"
            )
        proto.neighbor_list.type = proto.NeighborListType.DEVICE_SPARSE

    @staticmethod
    @util.define_symbols(
        "max_neighbors, nx, ny, nz, c",
        ["c <= n_atoms", "27*c^2*nx*ny*nz >= max_neighbors"]
    )
    def create_symbolic_input_format(max_neighbors, nx, ny, nz, c, *, n_atoms, **kwargs):

        # Currently, JAX can only infer dimensions from array shapes but not the
        # input
        update = jax.ShapeDtypeStruct((1,), jnp.bool)

        xcells = jax.ShapeDtypeStruct((nx,), jnp.bool)
        ycells = jax.ShapeDtypeStruct((ny,), jnp.bool)
        zcells = jax.ShapeDtypeStruct((nz,), jnp.bool)

        capacity = jax.ShapeDtypeStruct((c,), jnp.bool)

        # We pass reference positions from the previous build to skip the
        # neighbor list construction if smaller than the input
        # ref_pos = jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32)

        # Increase cutoff by this value to reuse neighbor list when particle
        # move less than half this distance
        # skin = jax.ShapeDtypeStruct(tuple(), jnp.float32)
        # cutoff = skin

        senders = jax.ShapeDtypeStruct((max_neighbors,), jnp.int32)
        # receivers = jax.ShapeDtypeStruct((max_neighbors,), jnp.int32)

        return (
            update, xcells, ycells, zcells, capacity, senders, senders
        )

    @staticmethod
    def create_from_args(r_cutoff, nbr_order, positions, local_mask,
                         valid_mask, newton_pair, *args):
        nargs = DeviceSparseNeighborListArgs(*args)

        buffer = jnp.zeros(
            (
                nargs.xcells.size,
                nargs.ycells.size,
                nargs.zcells.size,
                nargs.capacity.size
            ),
            dtype=jnp.int32
        )

        # TODO: Skip the recomputation for now
        # recompute = jnp.max(
        #     jnp.sum((positions - nargs.ref_pos) ** 2.0, axis=-1)
        # ) < (nargs.skin / 2) ** 2

        update_fn = functools.partial(
            compute_neighbor_list, positions, buffer, nargs.senders,
            cutoff=r_cutoff + 2.0, mask=valid_mask # Hard-coded skin size
        )

        def reuse_fn():
            # Return the statistics from the previous build
            statistics = NeighborListStatistics(
                min_cell_capacity=nargs.capacity.size,
                cell_too_small=0,
                max_neighbors=nargs.senders.size)

            return (nargs.senders, nargs.receivers), statistics


        graph, statistics = lax.cond(nargs.update.squeeze(), update_fn, reuse_fn)

        return SimpleSparseNeighborList(*graph), (*statistics.tuple, *graph)


class DeviceListStatistics(typing.TypedDict, total=True):
    """Statistics for the :class:`DeviceSparseNeighborList`."""
    min_cell_capacity: typing.Required[int]
    cell_too_small: typing.Required[int]
    max_neighbors: typing.Required[int]


class NeighborListStatistics(typing.TypedDict, total=True):
    """Capacity statistics for supported engine-provided neighbor lists.

    ``max_neighbors`` is the retained directed-edge count. For sparse graphs,
    ``overlong`` is the valid edge count before dependency pruning. For dense
    graphs, ``overlong`` is the retained ordered-triplet count. The field names
    are retained for model-format compatibility.
    """
    max_neighbors: typing.Required[int]
    overlong: typing.Required[int]


@jax.jit
def compute_cell_list(position, id_buffer, cutoff, mask=None, eps=1e-3):
    """Assigns particle indices to a three-dimensional cell grid.

    Follows the JAX-MD cell-list construction while using only shape
    information from the input arguments.

    Args:
        position: Particle positions with shape ``(n_particles, 3)``.
        id_buffer: Shape carrier with dimensions ``(nx, ny, nz, capacity)``.
            The first three dimensions set the cell counts, and the final
            dimension sets the maximum particles stored per cell.
        cutoff: Minimum cell length in the same units as ``position``.
        mask: Boolean array selecting particles included in the cell list.
        eps: Reserved tolerance for avoiding boundary classification errors.

    Returns:
        The populated cell buffer and experimental construction statistics.

    """
    assert mask is not None, "Requires mask argument!"

    if mask is None:
        mask = jnp.ones(position.shape[0], dtype=bool)

    *cell_counts, capacity = id_buffer.shape

    # Shift the positions to be in the range [0, box]. First, we shift
    # the masked particles positions to not have an influence on the range.
    # Then we shift the positions to be positive.
    mean_position = jnp.mean(mask[:, jnp.newaxis] * position, axis=0, keepdims=True)
    position = jnp.where(mask[:, jnp.newaxis], position, mean_position)
    position -= jnp.min(position, axis=0, keepdims=True)

    # TODO: How big should the tolerance be?
    box = jnp.diag(jnp.max(position, axis=0) + 0.5 * cutoff)

    # Generally, the minimum cell dimension must be larger than the cutoff,
    # such that all potential neighbors are contained in the neighboring cells.
    # Potential workaround: Increase box dimension such that smallest cell size
    # is as large as the cutoff. Will work if cell capacity is big enough
    cell_sizes = jnp.diag(box) / jnp.asarray(cell_counts)
    cell_too_small = jnp.sum((cell_sizes < cutoff) * 2 ** jnp.arange(3))

    cell_too_small = jnp.sum(1 - mask)

    # Scale the box dimensions such that all cell sizes are larger than the cutoff
    cell_sizes *= 1 + (cell_sizes < cutoff) * ((cutoff - cell_sizes) / cell_sizes)

    # Get the cell IDs for each particle in every dimension and transform them
    # into flat IDs. Assign excluded particles an invalid cell ID so they do
    # not belong to any cell.
    nx, ny, nz = cell_counts
    max_cell_ids = 1
    for n_in_dim in cell_counts:
        max_cell_ids *= n_in_dim

    cell_ids = jnp.int32(jnp.floor(position / cell_sizes[jnp.newaxis, :]))
    cell_ids = jnp.sum(cell_ids * jnp.asarray([[nz * ny, nz, 1]]), axis=-1)
    cell_ids = jnp.where(mask, cell_ids, max_cell_ids)

    # We can now count how often a particle appears in each cell
    cell_occupancy = jax.ops.segment_sum(jnp.int32(mask), cell_ids, cell_ids.size + 1)
    min_cell_capacity = jnp.max(cell_occupancy)

    # Sort particles by cell ID, then assign a unique slot within each cell.
    # A sufficient capacity ensures that no cell segment wraps and overwrites
    # an earlier particle.
    sort_idx = jnp.argsort(cell_ids)
    particle_ids = jnp.arange(position.shape[0])
    unique_id_per_segment = jnp.mod(lax.iota(jnp.int32, position.shape[0]), capacity)

    new_id_buffer = jnp.full((max_cell_ids + 1, capacity), position.shape[0])
    new_id_buffer = new_id_buffer.at[cell_ids[sort_idx], unique_id_per_segment].set(particle_ids[sort_idx])
    new_id_buffer = new_id_buffer[:-1, :].reshape(id_buffer.shape)

    statistics = DeviceListStatistics(min_cell_capacity, cell_too_small, 0)
    return new_id_buffer, statistics


@jax.jit
def compute_neighbor_list(position, id_buffer, senders, cutoff, mask=None, eps=1e-3):
    """Builds an experimental sparse neighbor list from a cell list.

    Args:
        position: Particle positions with shape ``(n_particles, 3)``.
        id_buffer: Shape carrier for cell counts and per-cell capacity.
        senders: Shape carrier whose length sets the edge capacity.
        cutoff: Maximum neighbor distance in the units of ``position``.
        mask: Boolean array selecting particles included in the graph.
        eps: Reserved tolerance for avoiding boundary classification errors.

    Returns:
        Sender and receiver arrays together with experimental construction
        statistics. Unused entries contain the invalid particle index.

    """
    assert mask is not None, "Requires mask argument!"

    if mask is None:
        mask = jnp.ones(position.shape[0], dtype=bool)

    invalid_idx = position.shape[0]

    # Compute the offsets of all neighboring cells
    offset_in_dim = jnp.arange(3) - 1
    xn, yn, zn = jnp.meshgrid(offset_in_dim, offset_in_dim, offset_in_dim, indexing='ij')
    nx, ny, nz, capacity = id_buffer.shape

    total_edges = 27 * (nx * ny * nz) * (capacity ** 2)

    id_buffer, statistics = compute_cell_list(
        position, id_buffer, cutoff, mask=mask, eps=eps)

    # Build the neighbor list for all cells
    @functools.partial(jax.vmap, in_axes=(0, None, None))
    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    @functools.partial(jax.vmap, in_axes=(None, None, 0))
    def cell_candidate_fn(cx, cy, cz):
        # Get the ids of all neighboring cells. For at least
        # three cells, this should not count edges double
        all_cx = jnp.mod(cx + xn, nx).ravel()
        all_cy = jnp.mod(cy + yn, ny).ravel()
        all_cz = jnp.mod(cz + zn, nz).ravel()

        # These are the indices of all particles that could be neighbors.
        # Sender rows come from the current cell, while receivers come from
        # all adjacent cells.
        receiver_idxs = id_buffer[all_cx, all_cy, all_cz, :]
        sender_idxs = id_buffer[cx, cy, cz, :]

        # Transform to sparse list
        cell_senders, cell_receivers = jnp.meshgrid(
            sender_idxs, receiver_idxs.ravel(), indexing='ij')
        cell_senders = cell_senders.ravel()
        cell_receivers = cell_receivers.ravel()

        sender_pos = position[cell_senders, :]
        receiver_pos = position[cell_receivers, :]

        # Compute all the distances (senders, receivers)
        dist_sq = jnp.sum((receiver_pos - sender_pos) ** 2, axis=-1)
        cut_sq = jnp.square(cutoff)

        # Select valid neighbors within cutoff that are not self
        cell_mask = dist_sq < cut_sq

        # Remove edges from or to invalid receivers
        cell_mask = jnp.logical_and(cell_mask, mask[cell_senders])
        cell_mask = jnp.logical_and(cell_mask, mask[cell_receivers])

        # Remove edges to self
        cell_mask = jnp.logical_and(cell_mask, cell_senders != cell_receivers)

        # Exclude invalid indices from both endpoints.
        cell_mask = jnp.logical_and(cell_mask, cell_senders < invalid_idx)
        cell_mask = jnp.logical_and(cell_mask, cell_receivers < invalid_idx)

        # Apply mask to neighbor list
        cell_senders = jnp.where(cell_mask, cell_senders, invalid_idx)
        cell_receivers = jnp.where(cell_mask, cell_receivers, invalid_idx)

        print(
            f"Senders: {cell_senders.shape}, Receivers: {cell_receivers.shape}")

        return cell_senders, cell_receivers

    new_senders, new_receivers = cell_candidate_fn(
        jnp.arange(nx), jnp.arange(ny), jnp.arange(nz)
    )
    new_senders, new_receivers = new_senders.ravel(), new_receivers.ravel()

    max_neighbors = senders.size
    valid_neighbors = jnp.sum(new_receivers < invalid_idx)

    _, prune_idx = lax.top_k(-new_receivers, max_neighbors)

    valid_pruned_neighbors = jnp.sum(new_receivers[prune_idx] < invalid_idx)


    statistics = statistics.set(
        max_neighbors=valid_neighbors, cell_too_small=valid_pruned_neighbors)

    return (new_senders[prune_idx], new_receivers[prune_idx]), statistics


def prune_neighbor_list(list, local, max_edges, nbr_order: int, half_list: bool = False):
    """Prunes a sparse graph to dependencies of owned atoms.

    Starting from ``local``, the function follows both edge directions for the
    requested graph depth. Edges incident to the reachable atoms are retained.
    A half list is expanded first so model code receives the same directed
    graph representation in both Newton modes.

    Args:
        list: Sparse neighbor list to prune.
        local: Mask specifying the local atoms.
        max_edges: Maximum number of edges in the pruned list.
        nbr_order: Maximum order of neighbors required for the force computation.
        half_list: If True, the engine supplied only one directed edge for each
            physical pair. Pruning adds the reverse edge so exported models see
            the same full directed graph in both Newton modes.

    Returns:
        The pruned neighbor list and its number of valid edges.

    """

    if half_list:
        # Make a full list from the half list
        senders = jnp.concat([list.senders, list.receivers], axis=0)
        receivers = jnp.concat([list.receivers, list.senders], axis=0)
        pair_type = (None if list.pair_type is None else
                     jnp.concat([list.pair_type, list.pair_type], axis=0))
    else:
        # lax.cond requires matching branch outputs, not matching intermediate
        # shapes. Full lists therefore do not need padding to the doubled size
        # used while expanding half lists.
        senders = list.senders
        receivers = list.receivers
        pair_type = list.pair_type
    list = list.set(senders=senders, receivers=receivers, pair_type=pair_type)

    valid = (
        (list.senders >= 0)
        & (list.senders < local.size)
        & (list.receivers >= 0)
        & (list.receivers < local.size)
    )
    safe_senders = jnp.where(valid, list.senders, 0)
    safe_receivers = jnp.where(valid, list.receivers, 0)

    def _update(reachable, _):
        # Expand through both endpoints so pruning is independent of whether a
        # model scatters to senders or receivers.
        reachable_from_receivers = jax.ops.segment_max(
            valid & reachable[safe_receivers], safe_senders, reachable.size)
        reachable_from_senders = jax.ops.segment_max(
            valid & reachable[safe_senders], safe_receivers, reachable.size)
        reachable |= reachable_from_receivers | reachable_from_senders
        return reachable, _

    # For k message-passing steps, retain every edge incident to an atom that is
    # reachable from an owned atom in at most k - 1 steps. Following both
    # endpoints supports models that collect neighbor messages at either
    # endpoint. An edge joining two atoms in the outermost shell cannot affect
    # an owned output and may be removed.
    if nbr_order > 0:
        reachable, _ = lax.scan(
            _update, local, jnp.arange(nbr_order - 1))
        mask = valid & (
            reachable[safe_senders] | reachable[safe_receivers])
    else:
        mask = jnp.zeros_like(valid)
    n_valid = jnp.sum(mask)

    # Fixed-size nonzero lowers to scan/scatter rather than a general GPU sort.
    # The appended invalid particle index fills unused output slots.
    candidate_count = mask.size
    select = jnp.nonzero(
        mask, size=max_edges, fill_value=candidate_count)[0]
    output_mask = select < candidate_count
    invalid_index = jnp.asarray([local.size], dtype=list.senders.dtype)
    senders = jnp.concat([list.senders, invalid_index])[select]
    receivers = jnp.concat([list.receivers, invalid_index])[select]
    pair_type = (None if list.pair_type is None else jnp.where(
        output_mask,
        jnp.concat([
            list.pair_type,
            jnp.asarray([0], dtype=list.pair_type.dtype),
        ])[select],
        jnp.int32(0),
    ))

    return SimpleSparseNeighborList(
        senders, receivers, output_mask, pair_type), n_valid


def prune_neighbor_list_dense(list, local, nbr_order: int):
    """Prunes a dense graph to dependencies of owned atoms.

    Args:
        list: Dense neighbor list to prune.
        local: Mask specifying the local atoms.
        nbr_order: Maximum order of neighbors required for the force computation.

    Returns:
        The pruned neighbor list, number of valid edges, and number of
        triplets formed by the valid edges.

    """

    valid = (list.nbrs >= 0) & (list.nbrs < local.size)
    safe_nbrs = jnp.where(valid, list.nbrs, 0)

    def _update(reachable, _):
        # Dense rows and neighbor entries are the two edge endpoints. Expand
        # reachability through both endpoints so models may collect messages at
        # either one.
        rows_reached_from_nbrs = jnp.any(
            valid & reachable[safe_nbrs], axis=1)
        nbrs_reached_from_rows = jax.ops.segment_max(
            (valid & reachable[:, None]).ravel(),
            safe_nbrs.ravel(),
            local.size,
        )
        reachable |= rows_reached_from_nbrs | nbrs_reached_from_rows
        return reachable, _

    if nbr_order > 0:
        reachable, _ = lax.scan(
            _update, local, jnp.arange(nbr_order - 1))
        edge_mask = valid & (
            reachable[:, None] | reachable[safe_nbrs])
    else:
        edge_mask = jnp.zeros_like(valid)

    nbrs = jnp.where(edge_mask, list.nbrs, local.size)
    pair_type = (None if list.pair_type is None else
                 jnp.where(edge_mask, list.pair_type, jnp.int32(0)))

    nbrs_per_atom = jnp.sum(edge_mask, axis=1)
    max_edges = jnp.sum(nbrs_per_atom)
    max_triplets = jnp.sum(nbrs_per_atom * (nbrs_per_atom - 1))

    return list.set(nbrs=nbrs, pair_type=pair_type), (max_edges, max_triplets)
