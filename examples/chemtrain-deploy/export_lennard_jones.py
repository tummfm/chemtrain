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

"""Exports sparse and dense smoothly truncated Lennard-Jones models."""

from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp

from chemtrain.deploy import exporter, graphs


R_ONSET = 2.0
R_CUTOFF = 2.5


def _pair_energy(squared_distance, valid):
    r"""Computes a smoothly truncated Lennard-Jones pair energy.

    The unsmoothed energy is :math:`U(r) = 4(r^{-12} - r^{-6})` in reduced
    Lennard-Jones units with :math:`\epsilon = \sigma = 1`. Between
    ``R_ONSET`` and ``R_CUTOFF``, the energy is multiplied by the standard C1 switching
    function used by JAX-MD and LAMMPS ``lj/charmm/coul/charmm``. The function
    returns zero outside the cutoff and for invalid pairs. A temporary unit
    squared distance keeps invalid intermediate values finite before masking.
    """
    within_cutoff = valid & (squared_distance < R_CUTOFF**2)
    finite_squared_distance = jnp.where(within_cutoff, squared_distance, 1.0)
    inverse_sixth = finite_squared_distance ** -3
    energy = 4.0 * (inverse_sixth**2 - inverse_sixth)

    onset_squared = R_ONSET**2
    cutoff_squared = R_CUTOFF**2
    switch = (
        (cutoff_squared - finite_squared_distance) ** 2
        * (
            cutoff_squared
            + 2.0 * finite_squared_distance
            - 3.0 * onset_squared
        )
        / (cutoff_squared - onset_squared) ** 3
    )
    switch = jnp.where(finite_squared_distance < onset_squared, 1.0, switch)
    return jnp.where(within_cutoff, switch * energy, 0.0)


class SparseLennardJones(exporter.Exporter):
    """Exports a Lennard-Jones model with a sparse directed graph."""

    graph_type = graphs.SimpleSparseNeighborList
    r_cutoff = R_CUTOFF
    unit_style = "lj"
    # Newton on differentiates owned particle energies and returns ghost force
    # contributions to their owners. Newton off has no reverse force
    # communication, so the force derivative also includes valid ghost
    # energies and needs one additional graph shell.
    nbr_order = [1, 2]

    def energy_fn(self, position, particle_data, graph, comm=None):
        del particle_data, comm
        n_atoms = position.shape[0]
        # The sender is the central atom of a neighbor row, and the receiver is
        # the listed neighbor. Graph pruning marks padding with n_atoms. Append
        # one zero position so every graph index is valid before masking.
        valid = (
            (graph.senders >= 0)
            & (graph.senders < n_atoms)
            & (graph.receivers >= 0)
            & (graph.receivers < n_atoms)
        )
        padded_position = jnp.concatenate(
            (position, jnp.zeros((1, position.shape[1]), dtype=position.dtype))
        )
        displacement = (
            padded_position[graph.senders]
            - padded_position[graph.receivers]
        )
        edge_energy = _pair_energy(jnp.sum(displacement**2, axis=-1), valid)
        # Both edge directions are present, so each directed edge carries half
        # of one physical pair energy. Splitting the edge energy equally between
        # its sender and receiver gives one quarter to each endpoint.
        endpoint_energy = 0.25 * edge_energy
        return (
            jax.ops.segment_sum(
                endpoint_energy, graph.senders, num_segments=n_atoms
            )
            + jax.ops.segment_sum(
                endpoint_energy, graph.receivers, num_segments=n_atoms
            )
        )


class DenseLennardJones(exporter.Exporter):
    """Exports a Lennard-Jones model with a dense directed graph."""

    graph_type = graphs.SimpleDenseNeighborList
    r_cutoff = R_CUTOFF
    unit_style = "lj"
    # Newton on differentiates owned particle energies and returns ghost force
    # contributions to their owners. Newton off has no reverse force
    # communication, so the force derivative also includes valid ghost
    # energies and needs one additional graph shell.
    nbr_order = [1, 2]

    def energy_fn(self, position, particle_data, graph, comm=None):
        del particle_data, comm
        n_atoms = position.shape[0]
        # Each dense row belongs to its central atom, and each entry is a
        # neighbor. Append one zero position for the padding index.
        valid = (graph.nbrs >= 0) & (graph.nbrs < n_atoms)
        padded_position = jnp.concatenate(
            (position, jnp.zeros((1, position.shape[1]), dtype=position.dtype))
        )
        displacement = position[:, None, :] - padded_position[graph.nbrs]
        edge_energy = _pair_energy(jnp.sum(displacement**2, axis=-1), valid)
        # Both edge directions are present, so each directed edge carries half
        # of one physical pair energy. Splitting the edge energy equally between
        # its sender and receiver gives one quarter to each endpoint.
        endpoint_energy = 0.25 * edge_energy
        return (
            jnp.sum(endpoint_energy, axis=1)
            + jax.ops.segment_sum(
                endpoint_energy.reshape(-1),
                graph.nbrs.reshape(-1),
                num_segments=n_atoms,
            )
        )


def export_models(output_directory):
    """Exports sparse and dense bundles with CPU and CUDA implementations."""
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, model_type in (
        ("lennard_jones_sparse.ptb", SparseLennardJones),
        ("lennard_jones_dense.ptb", DenseLennardJones),
    ):
        model = model_type()
        model.export(platforms=("cpu", "cuda"))
        model.save(output_directory / name)


def main():
    """Parses the output directory and exports both models."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_directory",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="directory receiving the two .ptb files",
    )
    args = parser.parse_args()
    export_models(args.output_directory)


if __name__ == "__main__":
    main()
