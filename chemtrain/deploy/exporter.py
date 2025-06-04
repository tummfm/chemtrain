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

"""Exporting potential models to MLIR."""

import abc
import functools

import jax
from fontTools.misc.cython import returns
from jax import numpy as jnp, export, lax

from typing import Dict, NamedTuple, Any, List, Tuple, Callable, NoReturn

import jax_md_mod
from jax_md import util as md_util, space

from . import graphs, util
from ._protobuf import model_pb2 as model_proto


class Exporter(metaclass=abc.ABCMeta):
    """Exports a potential model to an MLIR module.

    To export a potential model, subclass this class, select an appropriate
    graph type and define the energy function:

    Usage:

        >>> import jax.numpy as jnp
        >>> from jax_md_mod import custom_energy
        >>> from jax_md import partition, space
        >>> from chemtrain.deploy import exporter, graphs
        >>> class LennardJonesExport(exporter.Exporter):
        ...
        ...     graph_type = graphs.SimpleSparseNeighborList
        ...     r_cutoff = 5.0
        ...     unit_style = "real"
        ...     nbr_order = [1, 1]
        ...
        ...     def energy_fn(self, pos, species, graph):
        ...
        ...         neighbors = partition.NeighborList(
        ...             jnp.stack((graph.senders, graph.receivers)),
        ...             pos, None, None, graph.senders.size, partition.Sparse,
        ...             None, None, None
        ...         )
        ...
        ...         assert neighbors.idx.shape[0] == 2, "Wrong shape"
        ...
        ...         displacement_fn, _ = space.free()
        ...         apply_fn = custom_energy.customn_lennard_jones_neighbor_list(
        ...             displacement_fn, None, None,
        ...             sigma=jnp.asarray([3.165]), epsilon=jnp.asarray([1.0]),
        ...             r_onset=4.0, r_cutoff=5.0,
        ...             initialize_neighbor_list=False,
        ...             per_particle=True # Important for export
        ...         )
        ...
        ...         return apply_fn(pos, neighbors, species=species)
        ...
        >>> model = LennardJonesExport()
        >>> try:
        ...     model.save("out.ptb")
        ... except AssertionError as e:
        ...     # We need to call the export method first
        ...     print(f"Error: {e}")
        Error: Model has not been exported yet. Please call `export()` first.
        >>>
        >>> model.export()


    Attributes:
        graph_type: Specifies the required neighborhood representation and
            how to generate it from the input data.
            See ref:`chemtrain.deploy.graphs`.
        nbr_order: List of two integers specifying the number of neighbors
            required for the newton and non-newton setting to correctly
            compute forces.
        r_cutoff: Cutoff radius for the potential.
        unit_style: Specifies the units in which the potential requires
            positions and returns energies. The force units depend solely
            on the length and energy units.

    """

    # Use the default graph containing the full neighbor indices
    graph_type: graphs.NeighborList = graphs.SimpleSparseNeighborList

    # Order to which neighbors are required for a correct force computation
    # in the newton and the non-newton setting
    nbr_order: List[int] = [1, 1]

    r_cutoff: float

    unit_style: str = "real"

    _symbols: List[str] = []
    _constraints: List[str] = []
    _init_fns: List[Callable] = []
    _proto: model_proto.Model = None

    @abc.abstractmethod
    def energy_fn(self, position, species, graph):
        """Computes the energy for positions and a graph representation.

        Args:
            position: (N, dim) Array of particle positions, including ghost
                atoms that are not within the local domain.
            species: (N) Array of atoms species.
            graph: Graph representation of the neighborhood around atoms.

        Returns:
            Must return an energy contribution associated to each particle.

        """
        pass

    @staticmethod
    @util.define_symbols("n_atoms")
    def _define_position_shapes(n_atoms, **kwargs):
        shape_defs = (
            jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32),
            jax.ShapeDtypeStruct((n_atoms,), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32), # n_local
            jax.ShapeDtypeStruct((), jnp.int32), # n_ghost
            jax.ShapeDtypeStruct((), jnp.bool_), # newton flag
        )

        return shape_defs

    def _add_shapes(self, init_fn):
        init_fn(self._symbols, self._constraints, self._init_fns)

    def _create_shapes(self):
        all_symbols = ",".join(self._symbols)
        symbols = {
            key: symb for key, symb in zip(
                self._symbols,
                export.symbolic_shape(all_symbols, constraints=self._constraints),
            )
        }
        shapes = []
        for init_fn in self._init_fns:
            shapes.extend(init_fn(**symbols))

        # Reset
        self._symbols, self._constraints, self._init_fns = [], [], []
        return shapes


    def _energy_fn(self, position, species, n_local, n_ghost, newton, *graph_args):
        # Expects particles to be sorted by local, ghost, and padding atoms

        valid_mask = jnp.arange(position.shape[0]) < (n_local + n_ghost)
        ghost_mask = jnp.arange(position.shape[0]) < n_local

        graph, build_statistics = self.graph_type.create_from_args(
            self.r_cutoff, self.nbr_order, position, species,
            ghost_mask, valid_mask, newton, *graph_args)
        graph = lax.stop_gradient(graph)

        @functools.partial(jax.grad, has_aux=True)
        def force_and_aux(pos):
            per_atom_energies = self.energy_fn(pos, species, graph)

            assert per_atom_energies.shape == ghost_mask.shape, (
                f"Per particle energies have shape {per_atom_energies.shape}, "
                f"but should have shape {ghost_mask.shape}."
            )

            # Attention: Force is negative gradient of potential.
            # Depending on the newton flag, we either compute:
            # - the gradient of the _total potential_ w.r.t. the _local atoms_
            # - the gradient of the _local potential_ w.r.t. _all atoms_
            # The latter case equals newton=true and requires additional
            # communication to sum up the forces on the ghost atoms.
            total_energy = md_util.high_precision_sum(
                jnp.where(valid_mask, per_atom_energies, jnp.float32(0.0)))
            local_energy = md_util.high_precision_sum(
                jnp.where(ghost_mask, per_atom_energies, jnp.float32(0.0))
            )

            force_energy = jnp.where(newton, local_energy, total_energy)
            force_energy = jnp.negative(force_energy)

            # Differentiate w.r.t. the total potential in the box, but exclude
            # ghost atom contributions to the total potential
            aux = local_energy, *build_statistics
            return force_energy, aux

        return force_and_aux(position)

    def export(self) -> None:
        """Exports the potential model to an MLIR module."""

        proto = model_proto.Model()

        proto.neighbor_list.cutoff = self.r_cutoff
        proto.unit_style = self.unit_style

        assert len(self.nbr_order) == 2, (
            "The nbr_order must contain the order of required neighbors for "
            "the newton and non-newton setting."
        )
        proto.neighbor_list.nbr_order.extend(self.nbr_order)

        self.graph_type.set_properties(proto)

        # Using the ghost mask in the last layer we can compute correct forces
        # by accounting for their contribution to the gradient but
        # mask them out when we compute the total potential to not count
        # them double.
        self._add_shapes(self._define_position_shapes)
        self._add_shapes(self.graph_type.create_symbolic_input_format)

        shapes = self._create_shapes()

        exp: export.Exported = export.export(
            jax.jit(self._energy_fn), platforms=["cuda"])(*shapes)

        proto.mlir_module = exp.mlir_module()

        self._proto = proto

    def __str__(self):
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        return str(self._proto)

    def save(self, file: str) -> None:
        """Saves the exported protobuffer to a file.

        Args:
            file: Path to the file where the model should be saved.

        """

        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        with open(file, "wb") as f:
            f.write(self._proto.SerializeToString())
