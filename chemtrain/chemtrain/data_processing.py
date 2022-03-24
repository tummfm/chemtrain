"""Functions that facilitate common data processing operations for machine
learning.
"""
import numpy as onp
from jax import tree_flatten, lax
from jax_sgmc import data

from chemtrain.jax_md_mod import custom_space
from chemtrain import util


def get_dataset(data_location_str, retain=None, subsampling=1):
    """Loads .pyy numpy dataset.

    Args:
        data_location_str: String of .npy data location
        retain: Number of samples to keep in the dataset
        subsampling: Only keep every subsampled sample of the data, e.g. 2.

    Returns:
        Subsampled data array
    """
    loaded_data = onp.load(data_location_str)
    loaded_data = loaded_data[:retain:subsampling]
    return loaded_data


def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.1):
    """Train-validation-test split for datasets. Works on arbitrary pytrees,
    including chex.dataclasses, dictionaries and single arrays.

    Args:
        dataset: Dataset pytree. Samples are assumed to be stacked along
                 axis 0.
        train_ratio: Percantage of dataset to use for training.
        val_ratio: Percantage of dataset to use for validation.

    Returns:
        Tuple (train_data, val_data, test_data) with the same shape as the input
        pytree, but split along axis 0.
    """
    leaves, _ = tree_flatten(dataset)
    dataset_size = leaves[0].shape[0]
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    train_data = util.tree_get_slice(dataset, 0, train_size, to_device=False)
    val_data = util.tree_get_slice(dataset, train_size, train_size + val_size,
                                   to_device=False)
    test_data = util.tree_get_slice(dataset, train_size + val_size, None,
                                    to_device=False)
    return train_data, val_data, test_data


def init_dataloaders(dataset, train_ratio=0.7, val_ratio=0.1):
    """Splits dataset and initializes dataloaders.

    Args:
        dataset: Dictionary containing the whole dataset. The NumpyDataLoader
                 returns batches with the same kwargs as provided in dataset.
        train_ratio: Percantage of dataset to use for training.
        val_ratio: Percantage of dataset to use for validation.

    Returns:
        A tuple (train_loader, val_loader, test_loader) of NumpyDataLoaders.
    """
    train_set, val_set, test_set = train_val_test_split(
        dataset, train_ratio, val_ratio)
    train_loader = data.NumpyDataLoader(**train_set)
    val_loader = data.NumpyDataLoader(**val_set)
    test_loader = data.NumpyDataLoader(**test_set)
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
