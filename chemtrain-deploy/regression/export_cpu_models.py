#!/usr/bin/env python3
# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
# SPDX-License-Identifier: Apache-2.0
"""Export the small CPU bundles used by the public JCN regression.

The regression intentionally creates its fixtures instead of checking binary
StableHLO files into the repository. Run this script with the chemtrain source
tree on ``PYTHONPATH`` before starting the Bazel test. It exports the existing
smooth Lennard-Jones examples and one deliberately asymmetric message-passing
model. The latter gives a numerically visible result only when sparse senders
are central particles and receivers are their neighbors.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

from chemtrain.deploy import exporter, graphs


def load_lennard_jones_example():
    """Load the maintained example without making ``examples`` a package."""
    source_root = Path(__file__).resolve().parents[2]
    path = source_root / "examples" / "chemtrain-deploy" / "export_lennard_jones.py"
    specification = importlib.util.spec_from_file_location(
        "chemtrain_deploy_lennard_jones", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load Lennard-Jones example from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class AsymmetricMessagePassing(exporter.Exporter):
    """Export three directed message-passing layers with host communication.

    Every sparse edge is interpreted as ``sender <- receiver``: the sender is
    the central particle and receives a weighted neighbor feature. The three
    different layer weights make reversing that direction observable. The
    communication variant refreshes ghost features before every layer and
    reduces the rank-local energy once, which exercises both callback types.
    """

    graph_type = graphs.SimpleSparseNeighborList
    r_cutoff = 3.0
    unit_style = "lj"
    nbr_order = [1, 1]
    communication_required = True

    def energy_fn(self, position, particle_data, graph, comm=None):
        """Return per-particle directed-message energies in reduced units."""
        n_atoms = position.shape[0]
        valid = (
            (graph.senders >= 0)
            & (graph.senders < n_atoms)
            & (graph.receivers >= 0)
            & (graph.receivers < n_atoms)
        )
        features = 0.5 + position[:, 0] - 0.25 * position[:, 1]
        for weight in (0.25, 0.5, 0.75):
            if comm is not None:
                features = comm.gather(features[:, None])[:, 0]
            padded_features = jnp.concatenate(
                (features, jnp.zeros((1,), dtype=features.dtype))
            )
            messages = jnp.where(
                valid, weight * padded_features[graph.receivers], 0.0
            )
            features = features + jax.ops.segment_sum(
                messages, graph.senders, num_segments=n_atoms
            )
        energy = 0.1 * features**2
        if comm is not None:
            # Reduce only owned energy and distribute the collective term over
            # all owned particles. Ghost and padding rows must not make the
            # physical result depend on adapter capacity or decomposition.
            local_mask = particle_data["local_mask"]
            local_energy = jnp.sum(jnp.where(local_mask, energy, 0.0))
            local_count = jnp.sum(local_mask.astype(energy.dtype))
            reduced_energy = comm.reduce(local_energy)
            global_count = comm.reduce(local_count)
            collective_energy = 0.01 * reduced_energy / jnp.maximum(
                global_count, 1.0
            )
            energy = energy + jnp.where(local_mask, collective_energy, 0.0)
        return energy


def export_models(output_directory: Path) -> None:
    """Write all CPU regression bundles into ``output_directory``."""
    output_directory.mkdir(parents=True, exist_ok=True)
    example = load_lennard_jones_example()
    for name, model_type in (
        ("lennard_jones_dense.ptb", example.DenseLennardJones),
        ("lennard_jones_sparse.ptb", example.SparseLennardJones),
    ):
        model = model_type()
        model.export(platforms=("cpu",))
        model.save(output_directory / name)

    message_passing = AsymmetricMessagePassing()
    message_passing.export(communication=True, platforms=("cpu",))
    message_passing.save(output_directory / "asymmetric_message_passing.ptb")

    class Float64LennardJones(example.SparseLennardJones):
        """Export the same model with float64 internal computations."""

        position_dtype = jnp.float64

    # This exporter is a short-lived subprocess. Enabling x64 globally for its
    # final fixture works with every supported JAX release and cannot affect a
    # later regression stage.
    jax.config.update("jax_enable_x64", True)
    float64_model = Float64LennardJones()
    float64_model.export(platforms=("cpu",))
    float64_model.save(output_directory / "lennard_jones_sparse_x64.ptb")


def main() -> None:
    """Parse the output location and export reproducible regression fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    export_models(parser.parse_args().output_directory)


if __name__ == "__main__":
    main()
