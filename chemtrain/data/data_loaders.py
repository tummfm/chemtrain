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

import jax.numpy as jnp
from jax_sgmc.data import numpy_loader, core

from chemtrain.data.preprocessing import train_val_test_split
from chemtrain import util

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


def init_batch_functions(data_loader: core.HostDataLoader,
                         mb_size: int,
                         cache_size: int = 1,
                         ) -> core.RandomBatch:
    """Initializes reference data access outside jit-compiled functions.

    Randomly draw batches from a given dataset on the host or the device.

    Args:
        data_loader: Reads data from storage.
        cache_size: Number of batches in the cache. A larger number is
            faster, but requires more memory.
        mb_size: Size of the data batch.

    Returns:
      Returns a tuple of functions to initialize a new reference data state, get
      a minibatch from the reference data state and release the data loader after
      the last computation.
    """

    hcb_format, mb_information = data_loader.batch_format(
        cache_size, mb_size=mb_size)
    mask_shape = (cache_size, mb_size)

    def init_fn(random: bool = True, **kwargs) -> core.CacheState:

        if random:
            chain_id = data_loader.register_random_pipeline(
                cache_size=cache_size, mb_size=mb_size, **kwargs
            )
        else:
            print(f"Initialize full data pipeline")
            chain_id = data_loader.register_ordered_pipeline(
                cache_size=cache_size, mb_size=mb_size, **kwargs
            )

        initial_state, initial_mask = data_loader.get_batches(chain_id)
        if initial_mask is None:
            initial_mask = jnp.ones((cache_size, mb_size), dtype=jnp.bool_)

        inital_cache_state = core.CacheState(
            cached_batches=initial_state,
            cached_batches_count=jnp.array(cache_size),
            current_line=jnp.array(0),
            chain_id=jnp.array(chain_id),
            valid=initial_mask
        )

        return inital_cache_state

    def _new_cache_fn(state: core.CacheState,
                      ) -> core.Batch:
        new_data, masks = data_loader.get_batches(state.chain_id)

        if masks is None:
            # Assume all samples to be valid.
            masks = jnp.ones(mask_shape, dtype=jnp.bool_)

        new_state = core.CacheState(
            cached_batches_count=state.cached_batches_count,
            cached_batches=new_data,
            current_line=jnp.array(0),
            chain_id=state.chain_id,
            valid=masks,
            callback_uuid=state.callback_uuid
        )

        return new_state

    def batch_fn(data_state: core.CacheState,
                 information: bool = False,
                 ) -> core.Batch:
        """Draws a new random batch.

        Args:
            data_state: State with cached samples
            information: Whether to return batch information
            device_count: Number of parallel programs calling the batch function

        Returns:
            Returns the new data state and the next batch. Optionally an additional
            struct containing information about the batch can be returned.

        """
        # Refresh the cache if necessary, after all cached batches have been used.
        if data_state.current_line == data_state.cached_batches_count:
            data_state = _new_cache_fn(data_state)

        current_line = jnp.mod(
            data_state.current_line, data_state.cached_batches_count)

        # Read the current line from the cache and add the mask containing
        # information about the validity of the individual samples
        mini_batch = util.tree_get_single(data_state.cached_batches, current_line)
        mask = data_state.valid[current_line, :]

        current_line = current_line + 1

        new_state = core.CacheState(
            cached_batches=data_state.cached_batches,
            cached_batches_count=data_state.cached_batches_count,
            current_line=current_line,
            chain_id=data_state.chain_id,
            valid=data_state.valid
        )

        info = core.MiniBatchInformation(
            observation_count = mb_information.observation_count,
            batch_size = mb_information.batch_size,
            mask = mask)

        if information:
            return new_state, (mini_batch, info)
        else:
            return new_state, mini_batch

    def release():
        pass

    return init_fn, batch_fn, release
