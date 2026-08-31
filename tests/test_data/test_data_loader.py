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

import h5py
import numpy as np
import pytest

from jax import numpy as jnp
from jax_sgmc.data.numpy_loader import NumpyDataLoader

from chemtrain.data import data_loaders
from chemtrain.parallel import DataParallelContext


class _RootCommunicator:
    """Small host-only communicator for generic-loader MPI tests."""

    def __init__(self):
        self.calls = []

    def scatter(self, values, root):
        self.calls.append(values)
        return values[0]


class _NonRootCommunicator:
    """Return prepared local values without materializing a root cache."""

    def __init__(self, values):
        self.values = iter(values)

    def scatter(self, values, root):
        assert values is None
        return next(self.values)

class TestForceMatching:

    def test_hdf5_loader_reads_global_batch(self, tmp_path):
        path = tmp_path / "dataset.h5"
        with h5py.File(path, "w") as handle:
            handle["x"] = np.arange(24, dtype=np.float32)

        loader = data_loaders.HDF5ParallelDataLoader(path)
        init_fn, get_batch_fn, release_fn = data_loaders.init_batch_functions(
            loader, mb_size=6, cache_size=1, prefetch=True
        )
        state = init_fn(random=False)
        state, (batch, info) = get_batch_fn(state, information=True)

        np.testing.assert_array_equal(batch["x"], np.arange(6, dtype=np.float32))
        assert info.batch_size == 6
        assert info.mask.shape == (6,)

        _, next_batch = get_batch_fn(state)
        np.testing.assert_array_equal(
            next_batch["x"], np.arange(6, 12, dtype=np.float32)
        )
        release_fn()
        assert not loader._dataset.id.valid

    def test_rng_key(self):

        loader = NumpyDataLoader(x=jnp.zeros((100, 10)))
    
        init_fn, get_batch_fn, release_fn = data_loaders.init_batch_functions(
            loader, 10, 2,
        )
    
        init_state = init_fn(random=True, rng_seed=11)
    
        _, batch1 = get_batch_fn(init_state)
        new_state, batch2 = get_batch_fn(init_state)
        
        assert 'rng' in batch1.keys(), "No rng key in batch."
        assert jnp.all(batch1['rng'] == batch2['rng']), "RNG key is not deterministic."
        
        _, batch3 = get_batch_fn(new_state)
        
        assert not jnp.all(batch1['rng'] == batch3['rng']), "RNG key did not change."

    def test_generic_mpi_loader_reads_only_on_root(self, monkeypatch):
        loader = NumpyDataLoader(x=np.arange(12, dtype=np.float32))
        communicator = _RootCommunicator()
        monkeypatch.setattr(
            data_loaders.util, "get_communicator", lambda: communicator
        )

        init_fn, get_batch_fn, release_fn = data_loaders.init_batch_functions(
            loader,
            mb_size=6,
            cache_size=1,
            parallel_context=DataParallelContext("mpi", rank=0, size=2),
        )
        state = init_fn(random=False)
        _, (batch, info) = get_batch_fn(state, information=True)

        np.testing.assert_array_equal(batch["x"], np.arange(3, dtype=np.float32))
        assert info.batch_size == 3
        assert len(communicator.calls) == 2
        assert communicator.calls[0][1]["x"].shape == (1, 3)
        release_fn()

    def test_generic_mpi_loader_nonroot_does_not_read(self, monkeypatch):
        loader = NumpyDataLoader(x=np.arange(12, dtype=np.float32))

        def no_read(chain_id):
            raise AssertionError("Only rank zero may read a generic MPI cache.")

        loader.get_batches = no_read
        communicator = _NonRootCommunicator(
            ({"x": np.arange(3, 6, dtype=np.float32)[None]}, np.ones((1, 3)))
        )
        monkeypatch.setattr(
            data_loaders.util, "get_communicator", lambda: communicator
        )

        init_fn, get_batch_fn, release_fn = data_loaders.init_batch_functions(
            loader,
            mb_size=6,
            cache_size=1,
            prefetch=True,
            parallel_context=DataParallelContext("mpi", rank=1, size=2),
        )
        state = init_fn(random=False)
        _, batch = get_batch_fn(state)

        np.testing.assert_array_equal(batch["x"], np.arange(3, 6, dtype=np.float32))
        release_fn()

    def test_release_closes_loader_after_prefetch_failure(self):
        loader = NumpyDataLoader(x=np.arange(12, dtype=np.float32))
        get_batches = loader.get_batches
        calls = 0
        closed = False

        def failing_get_batches(chain_id):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("read failed")
            return get_batches(chain_id)

        def close():
            nonlocal closed
            closed = True

        loader.get_batches = failing_get_batches
        loader.close = close
        init_fn, _, release_fn = data_loaders.init_batch_functions(
            loader, mb_size=3, cache_size=1, prefetch=True
        )
        init_fn(random=False)

        with pytest.raises(RuntimeError, match="read failed"):
            release_fn()
        assert closed

    def test_jax_nonroot_consumes_prefetched_cache(self):
        loader = NumpyDataLoader(x=np.arange(12, dtype=np.float32))
        get_batches = loader.get_batches
        calls = 0

        def counted_get_batches(chain_id):
            nonlocal calls
            calls += 1
            return get_batches(chain_id)

        loader.get_batches = counted_get_batches
        init_fn, get_batch_fn, release_fn = data_loaders.init_batch_functions(
            loader,
            mb_size=4,
            cache_size=1,
            prefetch=True,
            parallel_context=DataParallelContext("jax", rank=1, size=2),
        )
        state = init_fn(random=False)
        state, _ = get_batch_fn(state)
        state, _ = get_batch_fn(state)
        release_fn()

        assert calls == 3
