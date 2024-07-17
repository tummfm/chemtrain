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

from jax_sgmc.data import numpy_loader, core

from chemtrain.data.preprocessing import train_val_test_split

from typing import NamedTuple


class DataLoaders(NamedTuple):
    train_loader: core.DataLoader
    val_loader: core.DataLoader
    test_loader: core.DataLoader


def init_dataloaders(dataset, train_ratio=0.7, val_ratio=0.1, shuffle=False):
    """Splits dataset and initializes dataloaders.

    If the validation or test ratios are 0, returns None for the respective
    dataloaders.

    Args:
        dataset: Dictionary containing the whole dataset. The NumpyDataLoader
            returns batches with the same kwargs as provided in dataset.
        train_ratio: Fraction of dataset to use for training.
        val_ratio: Fraction of dataset to use for validation.
        shuffle: Whether to shuffle data before splitting into train-val-test.

    Returns:
        Returns a tuple ``(train_loader, val_loader, test_loader)`` of
        NumpyDataLoaders.

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
    return DataLoaders(train_loader, val_loader, test_loader)
