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
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, NamedTuple, Optional, cast

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
from chemtrain.parallel import DataParallelContext, resolve_parallelism

PyTree = Any


class HDF5ParallelDataLoader(numpy_loader.NumpyDataLoader):
    """Read parts of an HDF5 batch directly on each process.

    All processes select the same global batch. Each MPI process or JAX device
    reads only its part of that batch. Distributed loading requires a file path
    so that each process can open the file independently. HDF5 reads return
    NumPy arrays, which may be prefetched before the main thread places them on
    the JAX mesh.

    Args:
        file: HDF file containing the entries of the dataset as root datasets.
        strict_order: Whether to strictly enforce the order of the data.
            If False, the indices for the batches are redistributed.
        close_comm: Deprecated compatibility argument. The loader never owns or
            closes the process communicator.

    """

    def __init__(
        self, file, strict_order: bool = False, *, close_comm: bool = True
    ):
        # The sample is necessary to return the observations in the correct format.
        super().__init__()

        if h5py is None:
            raise ImportError("h5py is required for HDF5ParallelDataLoader.")

        if isinstance(file, h5py.File):
            self._dataset = file
            self._owns_file = False
        else:
            self._dataset = h5py.File(name=file, mode="r")
            self._owns_file = True

        root_datasets = {
            key: val.shape[0] for key, val in self._dataset.items()
            if isinstance(val, h5py.Dataset)
        }

        if not root_datasets:
            raise ValueError("The HDF5 file has no root datasets.")
        if len(set(root_datasets.values())) != 1:
            raise ValueError(
                "All root datasets in the HDF5 file must have the same length."
            )

        self._observation_count = list(root_datasets.values())[0]
        self._keys = list(root_datasets.keys())

        self._format_cache = {
            key: jax.ShapeDtypeStruct(
                dtype=onp.dtype(cast(h5py.Dataset, self._dataset[key]).dtype),
                shape=tuple(
                    int(s)
                    for s in cast(h5py.Dataset, self._dataset[key]).shape[1:]
                ),
            )
            for key in self._keys
        }

        self._strict_order = strict_order
        if not close_comm:
            warnings.warn(
                "close_comm is deprecated; HDF5ParallelDataLoader never owns "
                "the process communicator.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._parallel_context = DataParallelContext("single", rank=0, size=1)
        self._closed = False

    def configure_parallel(self, context: DataParallelContext) -> None:
        """Select which part of each batch this loader reads.

        The training code calls this method when a loader is assigned to a
        trainer. Users normally do not call it directly. MPI and distributed
        JAX runs require a file path instead of an open ``h5py.File`` so every
        process can open and close its own file.
        """
        if context.mode != "single" and not self._owns_file:
            raise ValueError(
                "Distributed HDF5 loading requires a file path so every process "
                "can own an independent read-only handle."
            )
        self._parallel_context = context

    def is_root(self):
        return self._parallel_context.is_root

    def register_random_pipeline(
        self,
        cache_size: int = 1,
        mb_size: Optional[int] = None,
        in_epochs: bool = False,
        shuffle: bool = False,
        **kwargs: Any,
    ) -> int:
        """Register a pipeline whose ``mb_size`` is the global batch size."""
        if mb_size is None:
            raise ValueError("mb_size must be provided")

        chain_id = super().register_random_pipeline(
            cache_size=cache_size, mb_size=mb_size,
            in_epochs=in_epochs,
            shuffle=shuffle,
            **kwargs,
        )
        return chain_id

    def register_ordered_pipeline(
        self,
        cache_size: int = 1,
        mb_size: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Register an ordered pipeline using a global batch size."""
        if mb_size is None:
            raise ValueError("mb_size must be provided")

        chain_id = super().register_ordered_pipeline(
            cache_size=cache_size, mb_size=mb_size, **kwargs,
        )
        return chain_id

    def get_batches(self, chain_id: int) -> PyTree:
        """Draws a batch from a chain.

        Args:
        chain_id: ID of the chain, which holds the information about the form of
            the batch and the process of assembling.

        Returns:
        Returns a superbatch as registered by :func:`register_random_pipeline`
        or :func:`register_ordered_pipeline`. Single-process loading returns
        ``cache_size`` batches of ``mb_size`` observations. MPI and JAX return
        this process's NumPy slice. JAX placement later gives that slice its
        global sharded shape.

        """
        # Data slicing is the same for all methods of random and ordered access,
        # only the indices for slicing differ. The method _get_indices find the
        # correct method for the chain.

        selections_idx, selections_mask = self._get_indices(chain_id)
        selections_idx = onp.asarray(selections_idx, dtype=onp.int32)
        selections_mask = onp.asarray(selections_mask, dtype=onp.bool_)

        if self._parallel_context.mode == "jax":
            # Read the contiguous part owned by this process as NumPy. Device
            # placement happens later on the thread consuming the cache.
            sharding = self._parallel_context.batch_sharding(2, cached=True)
            device_indices = sharding.addressable_devices_indices_map(
                selections_idx.shape
            ).values()
            local_slices = sorted(
                (index[1].start, index[1].stop) for index in device_indices
            )
            start, stop = local_slices[0][0], local_slices[-1][1]
            if any(
                left[1] != right[0]
                for left, right in zip(local_slices, local_slices[1:])
            ):
                raise ValueError(
                    "Each process must own a contiguous part of the data mesh."
                )
            selections_idx = selections_idx[:, start:stop]
            selections_mask = selections_mask[:, start:stop]

        if self._parallel_context.mode == "mpi":
            if selections_idx.shape[1] % self._parallel_context.size != 0:
                raise ValueError("Global HDF5 batch is not divisible by MPI size.")
            local_size = selections_idx.shape[1] // self._parallel_context.size
            start = self._parallel_context.rank * local_size
            stop = start + local_size
            selections_idx = selections_idx[:, start:stop]
            selections_mask = selections_mask[:, start:stop]

        # HDF5 fancy indices must be sorted and unique. Restore shuffled order
        # and repeated padding indices after the one physical read per leaf.
        restore_shape = selections_idx.shape
        unique, restore = onp.unique(selections_idx.ravel(), return_inverse=True)
        selected_observations = {}
        for leaf_name in self._keys:
            values = cast(h5py.Dataset, self._dataset[leaf_name])[unique]
            selected_observations[leaf_name] = values[restore].reshape(
                restore_shape + self._format_cache[leaf_name].shape
            )
        return selected_observations, selections_mask

    def place_on_mesh(self, observations, mask, global_batch_size: int):
        """Place a process-local cache on the configured JAX mesh.

        ``observations`` and ``mask`` have local ``[cache, batch, ...]``
        shapes. ``global_batch_size`` restores the global batch dimension.
        JAX mode returns globally shaped sharded arrays; other modes return the
        inputs unchanged.
        """
        if self._parallel_context.mode != "jax":
            return observations, mask

        def place(array):
            global_shape = (array.shape[0], global_batch_size, *array.shape[2:])
            sharding = self._parallel_context.batch_sharding(
                len(global_shape), cached=True
            )
            return jax.make_array_from_process_local_data(
                sharding, array, global_shape=global_shape
            )

        observations = jax.tree_util.tree_map(place, observations)
        mask = place(mask)
        return observations, mask

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
        if self._closed:
            return
        self._closed = True
        if self._owns_file:
            self._dataset.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def use_mpi(self) -> bool:
        """Whether this loader implements an MPI sharding strategy."""
        return self._parallel_context.mode == "mpi"





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
                         parallel_context: Optional[DataParallelContext] = None,
                         ) -> core.RandomBatch:
    """Initializes reference data access outside jit-compiled functions.

    Randomly draw batches from a given dataset on the host or the device.
    If ``rng_seed=<seed>`` is passed to the ``init_fn``, a ``jax.random.PRNGKey``,
    will be added to the batch.

    Args:
        data_loader: Reads data from storage.
        cache_size: Number of batches in the cache. A larger number is
            faster, but requires more memory.
        mb_size: Global batch size. MPI returns one rank-local slice; JAX mode
            returns a globally shaped array sharded across the data mesh.
        prefetch: Load the next cache while the current cache is in use. JAX
            device placement still runs on the thread consuming the cache.
        use_mpi: Deprecated switch for callers without ``parallel_context``.
        parallel_context: Selected parallel training method.

    Returns:
      Returns a tuple of functions to initialize a new reference data state, get
      a minibatch from the reference data state and release the data loader after
      the last computation.
    """

    # A supplied context takes precedence over the old use_mpi argument.
    if parallel_context is None:
        if use_mpi and util.use_mpi():
            parallel_context = resolve_parallelism("mpi")
        else:
            # Preserve legacy callers whose own mapping code manages devices.
            parallel_context = DataParallelContext("single", rank=0, size=1)

    configure_parallel = getattr(data_loader, "configure_parallel", None)
    if callable(configure_parallel):
        configure_parallel(parallel_context)
    place_on_mesh = getattr(data_loader, "place_on_mesh", None)

    world_size = parallel_context.size

    loader_mpi_setting = getattr(data_loader, "use_mpi", None)
    loader_uses_mpi = (
        bool(loader_mpi_setting()) if callable(loader_mpi_setting) else False
    )
    split_loaded_batch = parallel_context.mode == "mpi" and not loader_uses_mpi

    # General loaders cannot read a separate batch part on each MPI process.
    # Rank zero reads the full cache and sends one host slice to every rank.
    loader_mb_size = mb_size
    returned_mb_size = (
        mb_size // world_size if parallel_context.mode == "mpi" else mb_size
    )
    if mb_size % world_size != 0:
        raise ValueError(
            f"Global batch {mb_size} is not divisible by parallel size {world_size}."
        )

    _, mb_information = data_loader.batch_format(
        cache_size, mb_size=returned_mb_size
    )
    rng_batch_size = mb_information.batch_size
    mask_shape = (cache_size, returned_mb_size)

    prefetch_futures: Dict[int, Future] = {}
    executor = None
    if prefetch and (not split_loaded_batch or parallel_context.is_root):
        executor = ThreadPoolExecutor(max_workers=1)

    def _chain_id_as_int(chain_id) -> int:
        """Copy a data-chain identifier to Python."""
        if isinstance(chain_id, (int, onp.integer)):
            return int(chain_id)
        return int(jax.device_get(chain_id))

    def _submit_prefetch(chain_id: int) -> None:
        """Start reading the next cache when background loading is enabled."""
        if not prefetch or executor is None:
            return

        prefetch_futures[chain_id] = executor.submit(
            data_loader.get_batches, chain_id
        )

    def _scatter_mpi_cache(observations, mask):
        """Send one contiguous host cache slice from rank zero to each rank."""
        comm = util.get_communicator()
        if comm is None:
            raise RuntimeError("MPI communicator is unavailable.")

        if parallel_context.is_root:
            if mask is None:
                mask = onp.ones((cache_size, loader_mb_size), dtype=onp.bool_)
            local_size = loader_mb_size // world_size
            observations = [
                jax.tree_util.tree_map(
                    lambda value: onp.asarray(value)[
                        :, rank * local_size:(rank + 1) * local_size
                    ],
                    observations,
                )
                for rank in range(world_size)
            ]
            mask = [
                onp.asarray(mask)[:, rank * local_size:(rank + 1) * local_size]
                for rank in range(world_size)
            ]
        else:
            observations = None
            mask = None

        return comm.scatter(observations, root=0), comm.scatter(mask, root=0)

    def _load_cache(chain_id: int):
        """Read a cache and distribute generic MPI-loader data on the host."""
        if split_loaded_batch:
            if parallel_context.is_root:
                observations, mask = data_loader.get_batches(chain_id)
            else:
                observations, mask = None, None
            return _scatter_mpi_cache(observations, mask)
        return data_loader.get_batches(chain_id)

    def init_fn(random: bool = True, rng_seed=None, **kwargs) -> core.CacheState:
        """Register a data chain and read its first cache."""

        if random:
            chain_id = data_loader.register_random_pipeline(
                cache_size=cache_size, mb_size=loader_mb_size, **kwargs
            )
        else:
            chain_id = data_loader.register_ordered_pipeline(
                cache_size=cache_size, mb_size=loader_mb_size, **kwargs
            )

        initial_state, initial_mask = _load_cache(chain_id)

        if callable(place_on_mesh):
            initial_state, initial_mask = place_on_mesh(
                initial_state, initial_mask, loader_mb_size
            )

        if initial_mask is None:
            initial_mask = jnp.ones(mask_shape, dtype=jnp.bool_)

        _submit_prefetch(chain_id)

        initial_internal_state = {}
        if rng_seed is not None:
            initial_internal_state['rng'] = jax.random.PRNGKey(rng_seed)

        # Store the first cache after all host loading and placement is complete.
        initial_cache_state = core.CacheState(
            cached_batches=initial_state,
            cached_batches_count=jnp.array(cache_size),
            current_line=jnp.array(0),
            chain_id=jnp.array(chain_id),
            valid=initial_mask,
            state=initial_internal_state,
        )

        return initial_cache_state

    def _new_cache_fn(state: core.CacheState,
                      ) -> core.CacheState:
        """Replace an exhausted cache with newly loaded batches."""
        chain_id = _chain_id_as_int(state.chain_id)

        consume_prefetch = (
            prefetch and (not split_loaded_batch or parallel_context.is_root)
        )
        if consume_prefetch:
            future = prefetch_futures.pop(chain_id, None)
            if future is None:
                new_data, masks = data_loader.get_batches(chain_id)
            else:
                new_data, masks = future.result()
        elif split_loaded_batch:
            new_data, masks = None, None
        else:
            new_data, masks = data_loader.get_batches(chain_id)

        if split_loaded_batch:
            new_data, masks = _scatter_mpi_cache(new_data, masks)

        if callable(place_on_mesh):
            new_data, masks = place_on_mesh(new_data, masks, loader_mb_size)

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
        """Take one batch and its validity mask from the current cache."""
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
            if parallel_context.mode == "mpi":
                split = random.fold_in(split, parallel_context.rank)
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
        """Finish pending reads and close resources owned by the loader."""
        try:
            for future in prefetch_futures.values():
                if not future.cancel():
                    future.result()
        finally:
            prefetch_futures.clear()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            close = getattr(data_loader, "close", None)
            if callable(close):
                close()

    return init_fn, batch_fn, release
