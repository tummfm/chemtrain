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

"""Functions that facilitate common data processing operations for machine
learning.
"""
import numpy as onp
from jax import lax, tree_util
from jax_sgmc.data import numpy_loader

from chemtrain.jax_md_mod import custom_space
from chemtrain import util


def get_dataset(data_location_str, retain=None, subsampling=1):
    """Loads .pyy numpy dataset.

    Args:
        data_location_str: String of .npy data location
        retain: Number of samples to keep in the dataset. All by default.
        subsampling: Only keep every subsampled sample of the data, e.g. 2.

    Returns:
        Subsampled data array
    """
    loaded_data = onp.load(data_location_str)
    loaded_data = loaded_data[:retain:subsampling]
    return loaded_data


def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.1, shuffle=False,
                         shuffle_seed=0):
    """Train-validation-test split for datasets. Works on arbitrary pytrees,
    including chex.dataclasses, dictionaries and single arrays.

    If a subset ratio ratios is 0, returns None for the respective subset.

    Args:
        dataset: Dataset pytree. Samples are assumed to be stacked along
                 axis 0.
        train_ratio: Percantage of dataset to use for training.
        val_ratio: Percantage of dataset to use for validation.
        shuffle: If True, shuffles data before splitting into train-val-test.
                 Shuffling copies the dataset.
        shuffle_seed: PRNG Seed for data shuffling

    Returns:
        Tuple (train_data, val_data, test_data) with the same shape as the input
        pytree, but split along axis 0.
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


def init_dataloaders(dataset, train_ratio=0.7, val_ratio=0.1, shuffle=False):
    """Splits dataset and initializes dataloaders.

    If the validation or test ratios are 0, returns None for the respective
    dataloader.

    Args:
        dataset: Dictionary containing the whole dataset. The NumpyDataLoader
                 returns batches with the same kwargs as provided in dataset.
        train_ratio: Percantage of dataset to use for training.
        val_ratio: Percantage of dataset to use for validation.
        shuffle: Whether to shuffle data before splitting into train-val-test.

    Returns:
        A tuple (train_loader, val_loader, test_loader) of NumpyDataLoaders.
    """
    def init_subloader(data_subset):
        if data_subset is None:
            loader = None
        else:
            loader = numpy_loader.NumpyDataLoader(**data_subset, copy=False)
        return loader

    train_set, val_set, test_set = train_val_test_split(
        dataset, train_ratio, val_ratio, shuffle=shuffle)
    train_loader = init_subloader(train_set)
    val_loader = init_subloader(val_set)
    test_loader = init_subloader(test_set)
    return train_loader, val_loader, test_loader


def scale_dataset_fractional(traj, box):
    """Scales a dataset of positions from real space to fractional coordinates.

    Args:
        traj: A (N_snapshots, N_particles, 3) array of particle positions
        box: A 1 or 2-dimensional jax_md box

    Returns:
        A (N_snapshots, N_particles, 3) array of particle positions in
        fractional coordinates.
    """
    _, scale_fn = custom_space.init_fractional_coordinates(box)
    scaled_traj = lax.map(scale_fn, traj)
    return scaled_traj
