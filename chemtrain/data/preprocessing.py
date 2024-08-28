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
    >>> # Print the coordinates of the Calpha atom
    >>> print(test[0, 4, :])
    [0.8808514  0.72712505 0.25590205]

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
from jax import lax, tree_util
import jax.numpy as jnp

from jax_md_mod import custom_space
from chemtrain import util


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
                force_dataset = None):
    """Maps fine-scaled positions and forces to a coarser scale.

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
    # Compute the mapping via displacements to take care of periodic
    # boundary conditions

    disp_fn = jax.vmap(displacement_fn, in_axes=(None, 0))

    ref_positions = jnp.zeros_like(position_dataset[0, 0, :])
    displacements = lax.map(
        functools.partial(disp_fn, ref_positions),
        position_dataset
    )

    c_map /= jnp.sum(c_map, axis=1, keepdims=True)

    cg_dislacements = lax.map(
        functools.partial(jnp.einsum, 'Ii..., id->Id', c_map),
        -displacements
    )

    cg_positions = lax.map(
        functools.partial(jax.vmap(shift_fn, in_axes=(None, 0)), ref_positions),
        cg_dislacements
    )

    # Map forces if provided

    if force_dataset is None:
        return cg_positions

    d_map /= jnp.sum(d_map, axis=1, keepdims=True)

    # Avoid division by zero.
    mask = (c_map > 0.0)
    safe_c = jnp.where(mask, c_map, 1.0)

    cg_forces = lax.map(
        functools.partial(jnp.einsum, 'Ii..., id->Id', mask * d_map / safe_c),
        force_dataset
    )

    return cg_positions, cg_forces
