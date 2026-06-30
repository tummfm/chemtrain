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
import functools
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, NamedTuple, Optional, cast
import warnings

try:
    import mpi4py
except ImportError:
    mpi4py = None

try:
    import h5py
except ImportError:
    h5py = None

import jax
from jax import numpy as jnp, random
import numpy as onp

from jax_sgmc.data import numpy_loader, core

from chemtrain.data.preprocessing import train_val_test_split
from chemtrain import util
from chemtrain import config as chemtrain_config

PyTree = Any


class HDF5ParallelDataLoader(numpy_loader.NumpyDataLoader):
    """DataLoader that can be used in distributed settings and reads data from HDF5 files.

    This DataLoader is designed to be used in distributed settings, where multiple
    processes are running in parallel. It ensures that each process gets a different
    subset of the data, and that the data is shuffled differently for each process.

    Args:
        file: HDF file containing the entries of the dataset as root datasets.
        strict_order: Whether to strictly enforce the order of the data.
            If False, the indices for the batches are redistributed.

    """

    def __init__(self, file, strict_order: bool = False, *, close_comm: bool = True):
        # The sample is necessary to return the observations in the correct format.
        super().__init__()

        if isinstance(file, h5py.File):
            self._dataset = file
        else:
            self._dataset = h5py.File(name=file, mode="r")

        root_datasets = {
            key: val.shape[0] for key, val in self._dataset.items()
            if isinstance(val, h5py.Dataset)
        }

        assert len(set(root_datasets.values())) == 1, \
            "All datasets in the HDF5 file must have the same length."

        self._observation_count = list(root_datasets.values())[0]
        self._keys = list(root_datasets.keys())

        self._format_cache = {
            key: jax.ShapeDtypeStruct(
                dtype=onp.dtype(cast(h5py.Dataset, self._dataset[key]).dtype),
                shape=tuple(int(s) for s in cast(h5py.Dataset, self._dataset[key]).shape[1:]),
            )
            for key in self._keys
        }

        self._strict_order = strict_order
        self._close_comm = bool(close_comm)
        self._owns_comm = False

        # Clone to use communicator with mpi4py
        comm = util.get_communicator()
        if comm is not None:
            self._comm = comm.Clone()
            self._owns_comm = True
        else:
            self._comm = None

    def is_root(self):
        if self._comm is None:
            return False
        return self._comm.Get_rank() == 0

    def _mpi_rank(self) -> int:
        if self._comm is None:
            return 0
        return int(self._comm.Get_rank())

    def _mpi_size(self) -> int:
        if self._comm is None:
            return 1
        return int(self._comm.Get_size())

    def register_random_pipeline(
        self,
        cache_size: int = 1,
        mb_size: Optional[int] = None,
        in_epochs: bool = False,
        shuffle: bool = False,
        **kwargs: Any,
    ) -> int:
        """Register a pipeline with local mb_size per rank.

        Under MPI, the underlying index pipeline is registered with
        ``mb_size_global = mb_size_local * world_size`` so each rank can take a
        disjoint slice of a global batch.
        """
        if mb_size is None:
            raise ValueError("mb_size must be provided")

        world_size = self._mpi_size()
        chain_id = super().register_random_pipeline(
            cache_size=cache_size,
            mb_size=mb_size * world_size,
            in_epochs=in_epochs,
            shuffle=shuffle,
            **kwargs,
        )
        self._chains[chain_id]["local_mb_size"] = int(mb_size)
        self._chains[chain_id]["world_size"] = int(world_size)
        return chain_id

    def register_ordered_pipeline(
        self,
        cache_size: int = 1,
        mb_size: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Register a pipeline with local mb_size per rank.

        See :meth:`register_random_pipeline`.
        """
        if mb_size is None:
            raise ValueError("mb_size must be provided")

        world_size = self._mpi_size()
        chain_id = super().register_ordered_pipeline(
            cache_size=cache_size,
            mb_size=mb_size * world_size,
            **kwargs,
        )
        self._chains[chain_id]["local_mb_size"] = int(mb_size)
        self._chains[chain_id]["world_size"] = int(world_size)
        return chain_id

    def get_batches(self, chain_id: int) -> PyTree:
        """Draws a batch from a chain.

        Args:
        chain_id: ID of the chain, which holds the information about the form of
            the batch and the process of assembling.

        Returns:
        Returns a superbatch as registered by :func:`register_random_pipeline` or
        :func:`register_ordered_pipeline` with `cache_size` batches holding
        `mb_size` observations.

        """
        # Data slicing is the same for all methods of random and ordered access,
        # only the indices for slicing differ. The method _get_indices find the
        # correct method for the chain.

        if self._comm is None:
            selections_idx, selections_mask = self._get_indices(chain_id)
            selections_idx = onp.asarray(selections_idx, dtype=onp.int32)
            selections_mask = onp.asarray(selections_mask, dtype=onp.bool_)
        else:
            rank = self._mpi_rank()
            world_size = self._mpi_size()

            if self.is_root():
                selections_idx, selections_mask = self._get_indices(chain_id)
                selections_idx = onp.ascontiguousarray(onp.asarray(selections_idx, dtype=onp.int32))
                selections_mask = onp.ascontiguousarray(onp.asarray(selections_mask, dtype=onp.bool_))
                global_shape = selections_idx.shape
            else:
                selections_idx = None
                selections_mask = None
                global_shape = None

            global_shape = self._comm.bcast(global_shape, root=0)
            if global_shape is None:
                raise RuntimeError("Failed to broadcast global index shape from root process.")

            if not self.is_root():
                selections_idx = onp.empty(global_shape, dtype=onp.int32)
                selections_mask = onp.empty(global_shape, dtype=onp.bool_)

            # Broadcast the global index tables. This is small compared to the
            # actual HDF5 payload and ensures all ranks see a consistent
            # partitioning.
            assert selections_idx is not None
            assert selections_mask is not None
            self._comm.Bcast(selections_idx, root=0)
            self._comm.Bcast(selections_mask, root=0)

            # Each rank takes a strided slice along the *batch* dimension.
            # This matches util.mpi_tree_gather/mpi_tree_mean conventions.
            selections_idx = cast(onp.ndarray, selections_idx)[:, rank::world_size]
            selections_mask = cast(onp.ndarray, selections_mask)[:, rank::world_size]

        restore_shape = selections_idx.shape
        unique, restore = onp.unique(selections_idx.ravel(), return_inverse=True)

        # Slice the data and transform into pytree
        selected_observations = {
            leaf_name: jnp.asarray(
                cast(h5py.Dataset, self._dataset[leaf_name])[unique][restore].reshape(
                    restore_shape + self._format_cache[leaf_name].shape
                )
            )
            for leaf_name in self._keys
        }

        return selected_observations, jnp.array(selections_mask, dtype=jnp.bool_)

    def save_state(self, chain_id: int) -> PyTree:
        raise NotImplementedError("Saving of the DataLoader state is not supported.")

    def load_state(self, chain_id: int, data) -> None:
        raise NotImplementedError("Loading of the DataLoader state is not supported.")

    @property
    def _format(self):
        """Returns shape and dtype of a single observation."""
        return self._format_cache

    @property
    def static_information(self):
        """Returns information about total samples count and batch size. """
        information = {
            "observation_count": self._observation_count
        }
        return information

    def close(self):
        try:
            self._dataset.close()
        finally:
            if self._close_comm and self._owns_comm and self._comm is not None:
                try:
                    self._comm.Free()
                except Exception:
                    # Best-effort cleanup.
                    pass
                self._comm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def use_mpi(self) -> bool:
        """Whether this loader implements an MPI sharding strategy."""
        return self._comm is not None and self._mpi_size() > 1





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
                         *,
                         prefetch: bool = False,
                         use_mpi: bool = False,
                         ) -> core.RandomBatch:
    """Initializes reference data access outside jit-compiled functions.

    Randomly draw batches from a given dataset on the host or the device.
    If ``rng_seed=<seed>`` is passed to the ``init_fn``, a ``jax.random.PRNGKey``,
    will be added to the batch.

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

    comm = None
    rank = 0
    world_size = 1
    if use_mpi and util.use_mpi():
        comm = util.get_communicator()
        if comm is not None:
            rank = int(comm.Get_rank())
            world_size = int(comm.Get_size())

    # Check whether the loader supports mpi sharding and whether sharding is
    # active. If one is False, we fallback to one rank loading all data and
    # then distribute the data to all ranks.

    def _loader_uses_mpi() -> bool:
        attr = getattr(data_loader, "use_mpi", None)
        if not callable(attr):
            return False
        try:
            return bool(attr())
        except Exception:
            return False

    loader_uses_mpi = _loader_uses_mpi()
    fallback_shard = comm is not None and world_size > 1 and not loader_uses_mpi

    if prefetch and comm is not None and world_size > 1:
        warnings.warn(
            "prefetch=True uses a background thread that may call MPI "
            "operations. If you observe hangs or non-deterministic crashes, "
            "disable prefetch."
        )

    # We need to make this distinction because if the data loader does not 
    # support loading of sharded batches, one rank has to load batches for
    # all the devices and then shard them manually.
    loader_mb_size = mb_size if not fallback_shard else mb_size * world_size

    hcb_format, mb_information = data_loader.batch_format(
        cache_size, mb_size=loader_mb_size
    )
    rng_batch_size = mb_information.batch_size if not fallback_shard else mb_size
    mask_shape = (cache_size, mb_size)

    prefetch_futures: Dict[int, Future] = {}
    executor = None
    if prefetch:
        executor = ThreadPoolExecutor(max_workers=1)

    def _chain_id_as_int(chain_id) -> int:
        if isinstance(chain_id, (int, onp.integer)):
            return int(chain_id)
        return int(jax.device_get(chain_id))

    # Helper function that can be submitted to a thread pool.
    def _prefetch_once(chain_id: int):
        return data_loader.get_batches(chain_id)

    # Helper function that submits the prefetch only if prefetching is
    # enabled.
    def _submit_prefetch(chain_id: int) -> None:
        if not prefetch:
            return
        if executor is None:
            raise RuntimeError("Prefetch requested but executor is not initialized.")

        prefetch_futures[chain_id] = executor.submit(_prefetch_once, chain_id)

    def init_fn(random: bool = True, rng_seed=None, **kwargs) -> core.CacheState:

        if random:
            chain_id = data_loader.register_random_pipeline(
                cache_size=cache_size, mb_size=loader_mb_size, **kwargs
            )
        else:
            chain_id = data_loader.register_ordered_pipeline(
                cache_size=cache_size, mb_size=loader_mb_size, **kwargs
            )

        initial_state, initial_mask = _prefetch_once(chain_id)

        if fallback_shard:
            # Rank 0 selects the (global) batch; broadcast via mpi4jax, then slice
            # into disjoint per-rank minibatches.
            initial_state = util.mpi_tree_broadcast(initial_state, root=0)
            if initial_mask is None:
                initial_mask = jnp.ones((cache_size, loader_mb_size), dtype=jnp.bool_)
            initial_mask = util.mpi_tree_broadcast(initial_mask, root=0)

            initial_state_swapped = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), initial_state
            )
            initial_state_swapped, _ = util.mpi_tree_slice(initial_state_swapped)
            initial_state = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), initial_state_swapped
            )

            initial_mask_swapped = jnp.swapaxes(initial_mask, 0, 1)
            initial_mask_swapped, _ = util.mpi_tree_slice(initial_mask_swapped)
            initial_mask = jnp.swapaxes(initial_mask_swapped, 0, 1)

        if initial_mask is None:
            initial_mask = jnp.ones(mask_shape, dtype=jnp.bool_)

        _submit_prefetch(chain_id)

        initial_internal_state = {}
        if rng_seed is not None:
            initial_internal_state['rng'] = jax.random.PRNGKey(rng_seed)

        # Perform the sharding here! Avoids thread-safety issues 

        inital_cache_state = core.CacheState(
            cached_batches=initial_state,
            cached_batches_count=jnp.array(cache_size),
            current_line=jnp.array(0),
            chain_id=jnp.array(chain_id),
            valid=initial_mask,
            state=initial_internal_state,
        )

        return inital_cache_state

    def _new_cache_fn(state: core.CacheState,
                      ) -> core.CacheState:
        chain_id = _chain_id_as_int(state.chain_id)

        if prefetch:
            future = prefetch_futures.pop(chain_id, None)
            if future is None:
                new_data, masks = data_loader.get_batches(chain_id)
            else:
                new_data, masks = future.result()
        else:
            new_data, masks = data_loader.get_batches(chain_id)

        # Scatter in the main thread (after prefetch).
        if fallback_shard:
            new_data = util.mpi_tree_broadcast(new_data, root=0)
            if masks is None:
                masks = jnp.ones((cache_size, loader_mb_size), dtype=jnp.bool_)
            masks = util.mpi_tree_broadcast(masks, root=0)

            new_data_swapped = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), new_data
            )
            new_data_swapped, _ = util.mpi_tree_slice(new_data_swapped)
            new_data = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), new_data_swapped
            )

            masks_swapped = jnp.swapaxes(masks, 0, 1)
            masks_swapped, _ = util.mpi_tree_slice(masks_swapped)
            masks = jnp.swapaxes(masks_swapped, 0, 1)

        if masks is None:
            # Assume all samples to be valid.
            masks = jnp.ones(mask_shape, dtype=jnp.bool_)

        new_state = core.CacheState(
            cached_batches_count=state.cached_batches_count,
            cached_batches=new_data,
            current_line=jnp.array(0),
            chain_id=state.chain_id,
            valid=masks,
            callback_uuid=state.callback_uuid,
            state=state.state
        )

        # Already load the next batch in the background.
        _submit_prefetch(chain_id)

        return new_state
        
    @jax.jit
    def _split_batch(data_state: core.CacheState):
        current_line = jnp.mod(
            data_state.current_line, data_state.cached_batches_count)

        # Read the current line from the cache and add the mask containing
        # information about the validity of the individual samples
        mini_batch = util.tree_get_single(data_state.cached_batches, current_line)
        mask = data_state.valid[current_line, :]

        # Add a random key if required
        internal_state = data_state.state
        if 'rng' in internal_state.keys():
            key, split = random.split(internal_state['rng'])
            mini_batch['rng'] = random.split(split, rng_batch_size)
            internal_state['rng'] = key

        current_line = current_line + 1

        new_state = core.CacheState(
            cached_batches=data_state.cached_batches,
            cached_batches_count=data_state.cached_batches_count,
            current_line=current_line,
            chain_id=data_state.chain_id,
            valid=data_state.valid,
            state=internal_state
        )
        
        info = core.MiniBatchInformation(
            observation_count = mb_information.observation_count,
            batch_size = rng_batch_size,
            mask = mask)
            
        return new_state, mini_batch, info

    def batch_fn(data_state: core.CacheState,
                 information: bool = False,
                 device_count: int = 1,
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

        new_state, mini_batch, info = _split_batch(data_state)

        if information:
            return new_state, (mini_batch, info)
        else:
            return new_state, mini_batch

    def release():
        for future in prefetch_futures.values():
            future.cancel()
        prefetch_futures.clear()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    return init_fn, batch_fn, release
