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

"""Common operations to pre-process datasets for machine learning potentials.

Typical pre-processing steps for molecular reference data are subsampling,
splitting into training, validation and testing sets, as well as scaling
the positions into fractional coordinates.

Examples:

    An example of loading a small subset of a heavy-atom trajectory of alanine:

    >>> from pathlib import Path
    >>> root = Path.cwd().parent

    >>> import jax.numpy as jnp
    >>> from jax_md_mod import io
    >>> from chemtrain.data.data_loaders import init_dataloaders
    >>> from chemtrain.data.preprocessing import (
    ...     get_dataset, scale_dataset_fractional, train_val_test_split )

    We only get a subset of 10 conformations from the training data and scale the
    conformations to fractional coordinates:

    >>> box = jnp.ones(3)
    >>> position_data = get_dataset(
    ...     root / "examples/data/positions_ethane.npy",
    ...     retain=10)
    >>> force_data = get_dataset(
    ...     root / "examples/data/forces_ethane.npy",
    ...     retain=10)
    >>> position_data = scale_dataset_fractional(position_data, box)

    We split the dataset into a training, validation and testing set:

    >>> train, val, test = train_val_test_split(position_data, train_ratio=0.8, shuffle=False)
    >>> bool(jnp.all(train[0, 4, :] == position_data[0, 4, :]))
    True

    Alternatively, we can directly instanciate ``jax_sgmc`` data-loaders based
    on the split datasets by using:
    >>> dataset = {"positions": position_data, "forces": force_data}
    >>> train_loader, val_loader, test_loader = init_dataloaders(dataset)
    >>> print(train_loader.static_information)
    {'observation_count': 7}

"""

import functools

import numpy as onp

import jax
from jax import numpy as jnp, lax, tree_util

from jax_md_mod import custom_space, custom_partition
from jax_md import partition, space
from chemtrain import util

from typing import Tuple, Optional



def get_dataset(data_location_str, retain=0, subsampling=1, offset=0):
    """Loads dataset from a ``"*.npy"`` array file.

    Args:
        data_location_str: String of ``"*.npy"`` data location
        retain: Number of samples to keep in the dataset. All by default.
        subsampling: Only keep every n-th sample of the data.
        offset: Select which part of data to be used. Last part by default.

    Returns:
        Sub-sampled array of reference data.

    """
    loaded_data = onp.load(data_location_str)
    if offset == 0:
        assert retain <= loaded_data.shape[0], (
            f"Cannot retain more than {loaded_data.shape[0]} samples, got "
            f"retain = {retain}."
        )
        loaded_data = loaded_data[-retain::subsampling]
    else:
        loaded_data = loaded_data[-retain-offset:-offset:subsampling]
        assert retain + offset <= offset <= loaded_data.shape, (
            f"Cannot retain more than {loaded_data.shape[0] - offset} given "
            f"an offset of {offset}. Got retain = {retain}."
        )
    return loaded_data


def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.1, shuffle=False,
                         shuffle_seed=0):
    """Split data into disjoint subsets for training, validation and testing.

    Splitting works on arbitrary pytrees, including chex.dataclasses,
    dictionaries, and single arrays.

    The function splits the pytree leaves along their first dimension.

    If a subset ratio ratios is ``0``, returns ``None`` for the respective subset.

    Args:
        dataset: Dataset as pytree. Samples are assumed to be stacked along
            the first dimension of the pytree leaves.
        train_ratio: Fraction of dataset to use for training.
        val_ratio: Fraction of dataset to use for validation.
        shuffle: If True, shuffles data before splitting into train-val-test.
            Shuffling copies the dataset.
        shuffle_seed: PRNG Seed for data shuffling

    Returns:
        Returns a tuple ``(train_data, val_data, test_data)``, where each tuple
        element has the same pytree structure as the input pytree.

    """
    assert train_ratio + val_ratio <= 1., 'Distribution of data exceeds 100%.'
    leaves, _ = tree_util.tree_flatten(dataset)
    dataset_size = leaves[0].shape[0]
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)

    if shuffle:
        dataset_idxs = onp.arange(dataset_size)
        numpy_rng = onp.random.default_rng(shuffle_seed)
        numpy_rng.shuffle(dataset_idxs)

        def retreive_datasubset(idxs):
            data_subset = util.tree_take(dataset, idxs, axis=0)
            subset_leaves, _ = tree_util.tree_flatten(data_subset)
            subset_size = subset_leaves[0].shape[0]
            if subset_size == 0:
                data_subset = None
            return data_subset

        train_data = retreive_datasubset(dataset_idxs[:train_size])
        val_data = retreive_datasubset(dataset_idxs[train_size:
                                                    val_size + train_size])
        test_data = retreive_datasubset(dataset_idxs[val_size + train_size:])

    else:
        def retreive_datasubset(start, end):
            data_subset = util.tree_get_slice(dataset, start, end,
                                              to_device=False)
            subset_leaves, _ = tree_util.tree_flatten(data_subset)
            subset_size = subset_leaves[0].shape[0]
            if subset_size == 0:
                data_subset = None
            return data_subset

        train_data = retreive_datasubset(0, train_size)
        val_data = retreive_datasubset(train_size, train_size + val_size)
        test_data = retreive_datasubset(train_size + val_size, None)
    return train_data, val_data, test_data


def scale_dataset_fractional(positions, reference_box=None, box=None):
    """Scales a dataset of positions from real space to fractional coordinates.

    Args:
        positions: An array with shape ``(N_snapshots, N_particles, 3)`` with
            particle positions.
        reference_box: A 1 or 2-dimensional ``jax_md`` box. If not provided,
            the box is assumed to be dynamic.
        box: An array of 1 or 2-dimensional boxes, corresponding to the
            individual samples.

    Returns:
        Returns an array with  shape ``(N_snapshots, N_particles, 3)`` with
        particle positions in fractional coordinates.

    """
    if reference_box is None:
        reference_box = jnp.eye(positions.shape[-1])

    _, scale_fn = custom_space.init_fractional_coordinates(reference_box)
    if box is not None:
        return jax.vmap(lambda R, box: scale_fn(R, box=box))(positions, box)
    else:
        return jax.vmap(scale_fn)(positions)


def map_dataset(position_dataset,
                displacement_fn,
                shift_fn,
                c_map,
                d_map=None,
                force_dataset=None):
    """
    Maps fine-scaled positions and forces to a coarser scale.

    Uses the linear mapping from [Noid2008]_ to map fine-scaled positions and
    forces to coarse grained positions and forces via the relations:

    .. math::
        \\mathbf R_I = \\sum_{i \\in \\mathcal I_I} c_{Ii} \\mathbf r_i,\\quad \\text{and}

        \\mathbf{F}_I = \\sum_{i \\in \\mathcal I_I} \\frac{d_{Ii}}{c_{Ii}} \\mathbf f_i.

    Args:
        position_dataset: Dataset of fine-scaled positions.
        displacement_fn: Function to compute the displacement between two
            sets of coordinates. Necessary to handle boundary conditions.
        shift_fn: Ensures that the produced coordinates remain in the
            box.
        c_map: Matrix $c_{Ii}$ defining the linear mapping of positions.
        d_map: Matrix $d_{Ii}$ defining the linear mapping of forces in combination
            with $c_{Ii}$.
        force_dataset: Dataset of fine-scaled forces.

    Returns:
        Returns the coarse-grained positions and, if provided, coarse-grained
        forces.

    References:
        .. [Noid2008] W. G. Noid, Jhih-Wei Chu, Gary S. Ayton, Vinod Krishna,
           Sergei Izvekov, Gregory A. Voth, Avisek Das, Hans C. Andersen;
           *The multiscale coarse-graining method. I. A rigorous bridge between
           atomistic and coarse-grained models*. J. Chem. Phys. 28 June 2008;
           128 (24): 244114. https://doi-org.eaccess.tum.edu/10.1063/1.2938860
    """
    # Normalise mapping weights
    c_norm = c_map / jnp.sum(c_map, axis=1, keepdims=True)
    if d_map is not None:
        d_norm = d_map / jnp.sum(d_map, axis=1, keepdims=True)
    else:
        d_norm = None

    def _map_single(ipt, shift_fn, displacement_fn, c_norm, d_norm):
        pos, forc = ipt

        # Choose reference for each CG bead
        ref_idx = jnp.argmax(c_map, axis=1)
        ref_positions = pos[ref_idx, :]
        
        # Compute displacements for each reference position and map 
        disp = jax.vmap(
            lambda r: jax.vmap(lambda p: displacement_fn(p, r))(pos)
        )(ref_positions)
        cg_disp = jnp.einsum('Ii,Iid->Id', c_map, disp)
        cg_positions = jax.vmap(shift_fn)(ref_positions, cg_disp)
        
        
        if (forc is not None) and (d_norm is not None):
            mask = (c_norm > 0.0)
            safe_c = jnp.where(mask, c_norm, 1.0)
            cg_forces = jnp.einsum('Ii, id->Id', mask * d_norm / safe_c, forc)
        else:
            cg_forces = None

        return cg_positions, cg_forces

    _map_single = functools.partial(_map_single,
                                    shift_fn=shift_fn,
                                    displacement_fn=displacement_fn,
                                    c_norm=c_norm,
                                    d_norm=d_norm)

    if force_dataset is None:
        # map positions only
        return lax.map(lambda pos: _map_single((pos, None))[0], position_dataset)
    else:
        return lax.map(_map_single, (position_dataset, force_dataset))
    

def allocate_neighborlist(dataset,
                          displacement: space.DisplacementOrMetricFn,
                          box: space.Box,
                          r_cutoff: float,
                          capacity_multiplier: float = 1.0,
                          disable_cell_list: bool = True,
                          fractional_coordinates: bool = True,
                          format: partition.NeighborListFormat = partition.NeighborListFormat.Dense,
                          pairwise_distances: bool = True,
                          box_key: str = None,
                          mask_key: str = None,
                          reps_key: str = None,
                          batch_size: int = 1000,
                          init_kwargs: dict = None,
                          count_triplets: bool = False,
                          **static_kwargs
                          ) -> Tuple[partition.NeighborList,
                                     Tuple[int, int, float, Optional[int]]]:
    """Allocates an optimally sized neighbor list.

    .. doctest::
                                     
    >>> import jax.numpy as jnp
    >>> import jax_md_mod
    >>> from jax_md import space

    >>> # Example Dataset
    >>> 


    Args:
        dataset: A dictionary containing the dataset with key ``"R"`` for
            positions.
        displacement: A function `d(R_a, R_b)` that computes the displacement
            between pairs of points.
        box: Either a float specifying the size of the box, an array of
            shape `[spatial_dim]` specifying the box size for a cubic box in
            each spatial dimension, or a matrix of shape
            `[spatial_dim, spatial_dim]` that is _upper triangular_ and
            specifies the lattice vectors of the box.
        r_cutoff: A scalar specifying the neighborhood radius.
        capacity_multiplier: A floating point scalar specifying the fractional
            increase in maximum neighborhood occupancy we allocate compared with
            the maximum in the example positions.
        disable_cell_list: An optional boolean. If set to `True` then the
            neighbor list is constructed using only distances. This can be
            useful for debugging but should generally be left as `False`.
        fractional_coordinates: An optional boolean. Specifies whether positions
            will be supplied in fractional coordinates in the unit cube,
            :math:`[0, 1]^d`. If this is set to True then the `box_size` will be
            set to `1.0` and the cell size used in the cell list will be set to
            `cutoff / box_size`.
        format: The format of the neighbor list; see the
            :meth:`NeighborListFormat` enum for details about the different
            choices for formats. Defaults to `Dense`.
        pairwise_distances: Computes pairwise distances between every particles
            for every sample.
        box_key: The key in the dataset dictionary that contains the box. If
            not provided, uses the box argument.
        mask_key: The key in the dataset dictionary that contains the mask. If
            not provided, all particles are considered valid.
        reps_key: The key in the dataset dictionary that contains the number of
            replicas a supercell. If set, the neighborlist will only contain
            edge senders from the first appearing replica.
        batch_size: Evaluate multiple samples in parallel.
        init_kwargs: Keyword arguments passed to the neighbor list allocation,
            e.g., to specify a capacity multiplier.
        count_triplets: An optional boolean. If set to `True`, the function will
            return the maximum number of triplets, similar to the maximum
            number of edges.
        **static_kwargs: kwargs that get threaded through the calculation of
            example positions.

    Returns:
        Returns a neighbor list that fits the dataset.

    """

    # We use the masked neighbor list to avoid interference of masked particles
    # and required neighbor list capacity.
    neighbor_fn = custom_partition.masked_neighbor_list(
        displacement, box, r_cutoff, dr_threshold=0.0,
        capacity_multiplier=capacity_multiplier,
        disable_cell_list=disable_cell_list,
        fractional_coordinates=fractional_coordinates, format=format,
        **static_kwargs
    )

    assert pairwise_distances, (
        "Currently, this function only works when computing distances between "
        "all pairs of particles (``pairwise_distances=True``)."
    )

    @jax.jit
    def find_max_neighbors_and_edges(dataset):
        def number_of_neighbors(input):
            position, box, mask, reps = input

            if box is None:
                metric = space.canonicalize_displacement_or_metric(displacement)
            else:
                metric = space.canonicalize_displacement_or_metric(
                    functools.partial(displacement, box=box))

            pair_distances = space.map_product(metric)(position, position)

            # Find neighbors, discarding self-interactions and masked particles.
            is_neighbor = pair_distances <= r_cutoff
            is_neighbor = jnp.logical_and(
                is_neighbor, ~jnp.eye(is_neighbor.shape[0], dtype=jnp.bool_))

            # Invalid particles cannot receive or send edges.
            if mask is not None:
                is_neighbor = jnp.logical_and(is_neighbor, mask[jnp.newaxis, :])
                is_neighbor = jnp.logical_and(is_neighbor, mask[:, jnp.newaxis])

            # Remove all replicated receivers
            if reps is not None:
                max_local = jnp.sum(mask) // reps
                include = max_local < jnp.arange(is_neighbor.shape[0])
                is_neighbor = jnp.where(
                    include[:, jnp.newaxis], is_neighbor, False)

            # Sets the number of neighbors to 0 for masked particles
            neighbors = jnp.sum(is_neighbor, axis=1)
            if mask is not None:
                neighbors *= mask

            # Compute the number of triplets.
            # First, we evaluate whether the pair of nodes are connected by an
            # edge to the same node.
            ji, jk = jax.vmap(
                functools.partial(jnp.meshgrid, indexing="ij")
            )(is_neighbor, is_neighbor)

            extra_out = []
            if count_triplets:
                # We mask out pairs of identical edges.
                is_triplet = jnp.logical_and(ji, jk)
                is_triplet = jnp.logical_and(
                    is_triplet,
                    ~jnp.eye(
                        is_triplet.shape[0],
                        dtype=jnp.bool_
                    )[jnp.newaxis, ...]
                )

                extra_out += [jnp.sum(is_triplet)]

            avg_neighbors = jnp.mean(neighbors)
            if mask is not None:
                avg_neighbors /= jnp.mean(mask)

            max_neighbors = jnp.max(neighbors)
            max_edges = jnp.sum(neighbors)

            return max_neighbors, max_edges, avg_neighbors, *extra_out

        # We find the sample with the maximum number of neighbors or edges
        return util.batch_map(
            number_of_neighbors,
            (
                dataset["R"],
                dataset.get(box_key),
                dataset.get(mask_key),
                dataset.get(reps_key)
            ),
            batch_size=batch_size
        )

    n_neighbors, n_edges, avg_neighbors, *extra = find_max_neighbors_and_edges(
        dataset)

    print(
        f"The dataset has max. {jnp.max(n_neighbors)} neighbors per particle "
        f"and max. {jnp.max(n_edges)} edges in total.")

    if format == partition.Dense:
        # The maximum neighbors per particle determine the capacity of the
        # neighbor list.
        sample_idx = jnp.argmax(n_neighbors)
    elif format == partition.Sparse:
        # The maximum number of edges determine the capacity of the neighbor list.
        sample_idx = jnp.argmax(n_edges)
    else:
        raise ValueError(f"Unsupported neighbor list format: {format}")

    extra_out = []
    if count_triplets:
        n_triplets, = extra
        extra_out += [jnp.max(n_triplets)]

    if init_kwargs is None:
        init_kwargs = {}
    if box_key is not None:
        init_kwargs['box'] = jnp.asarray(dataset[box_key][sample_idx])
    if mask_key is not None:
        init_kwargs['mask'] = jnp.asarray(dataset[mask_key][sample_idx])

    nbrs_init = neighbor_fn.allocate(
        jnp.asarray(dataset["R"][sample_idx]), **init_kwargs)

    return nbrs_init, (n_neighbors.max(), n_edges.max(), avg_neighbors.mean(), *extra_out)
