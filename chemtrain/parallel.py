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

"""Utilities for running data-parallel training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from chemtrain import util


Parallelism = Literal["auto", "single", "mpi", "jax"]


@dataclass(frozen=True)
class DataParallelContext:
    """Parallel training setup selected by the user.

    Attributes:
        mode: Selected parallel mode.
        rank: Process number. This is 0 for a local JAX run.
        size: Number of MPI processes in MPI mode and total number of devices
            in JAX mode.
        mesh: JAX device mesh. This is only set in JAX mode.
    """

    mode: Literal["single", "mpi", "jax"]
    rank: int
    size: int
    mesh: Mesh | None = None

    @property
    def is_root(self) -> bool:
        """Whether this process writes files and prints progress."""
        return self.rank == 0

    @property
    def replicated_sharding(self) -> NamedSharding | None:
        """Place a copy of the model state on every JAX device."""
        if self.mesh is None:
            return None
        return NamedSharding(self.mesh, PartitionSpec())

    def batch_sharding(self, ndim: int = 1, *, cached: bool = False):
        """Split the batch axis over the JAX devices.

        Args:
            ndim: Number of array dimensions.
            cached: The cache dimension comes before the batch dimension when
                True, so dimension 1 is split instead of dimension 0.
        """
        if self.mesh is None:
            return None
        batch_axis = 1 if cached else 0
        spec = [None] * ndim
        spec[batch_axis] = "data"
        return NamedSharding(self.mesh, PartitionSpec(*spec))


def resolve_parallelism(
    parallelism: Parallelism = "auto", mesh: Mesh | None = None
) -> DataParallelContext:
    """Select and check one parallel training mode.

    Args:
        parallelism: ``single`` uses one process and device, ``mpi`` uses one
            device per MPI process, and ``jax`` splits work over a JAX mesh.
            ``auto`` selects JAX distributed training, MPI, local multi-device
            JAX, or single-device training in that order.
        mesh: Optional one-dimensional JAX mesh named ``data``. It can only be
            used with JAX parallelism and must assign the same number of
            devices to every process.

    Returns:
        Checked parallel training settings.

    Raises:
        ValueError: If parallel modes are mixed or the device setup is invalid.
        RuntimeError: If MPI support is incomplete.
    """
    if parallelism not in ("auto", "single", "mpi", "jax"):
        raise ValueError(f"Unknown parallelism mode: {parallelism!r}.")

    if parallelism == "auto":
        if jax.process_count() > 1:
            parallelism = "jax"
        elif util.use_mpi():
            parallelism = "mpi"
        elif jax.device_count() > 1:
            parallelism = "jax"
        else:
            parallelism = "single"

    if parallelism == "single":
        if mesh is not None:
            raise ValueError("A device mesh is only valid for JAX parallelism.")
        if util.use_mpi() or jax.process_count() > 1 or jax.device_count() != 1:
            raise ValueError("Single parallelism requires one process and one device.")
        return DataParallelContext("single", rank=0, size=1)

    if parallelism == "mpi":
        if mesh is not None:
            raise ValueError("MPI parallelism cannot be combined with a JAX mesh.")
        if not util.has_mpi4py():
            raise RuntimeError(
                "MPI parallelism requires mpi4py; install chemtrain[mpi]."
            )
        if not util.use_mpi():
            raise ValueError(
                "MPI parallelism requires more than one MPI process."
            )
        if not util.has_mpi4jax():
            raise RuntimeError(
                "MPI parallelism requires mpi4jax; install chemtrain[mpi]."
            )
        if jax.process_count() != 1:
            raise ValueError("MPI and JAX distributed processes cannot be combined.")
        if jax.local_device_count() != 1:
            raise ValueError(
                "MPI parallelism requires exactly one visible JAX device per rank."
            )
        comm = util.get_communicator()
        if comm is None:
            raise RuntimeError("MPI communicator is unavailable.")
        return DataParallelContext(
            "mpi", rank=int(comm.Get_rank()), size=int(comm.Get_size())
        )

    if util.use_mpi() and jax.process_count() == 1:
        raise ValueError(
            "JAX parallelism was launched under MPI without initializing "
            "jax.distributed; choose MPI or initialize JAX distributed training."
        )
    if mesh is None:
        mesh = Mesh(np.asarray(jax.devices()), ("data",))
    if mesh.axis_names != ("data",):
        raise ValueError("The data-parallel mesh must have one axis named 'data'.")
    # HDF5 loading forms equal process-local batch slices before JAX places
    # them on the mesh, so every process must own the same number of devices.
    devices_per_process = {
        process: sum(
            device.process_index == process for device in mesh.devices.flat
        )
        for process in range(jax.process_count())
    }
    if not devices_per_process or min(devices_per_process.values()) == 0:
        raise ValueError(
            "The data-parallel mesh must include a device from every JAX process."
        )
    if len(set(devices_per_process.values())) != 1:
        raise ValueError(
            "The data-parallel mesh must use the same device count on every process."
        )
    return DataParallelContext(
        "jax", rank=int(jax.process_index()), size=int(mesh.size), mesh=mesh
    )
