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

import jax
from jax import Array
import jax.numpy as jnp
import jax_md.partition
from jax_md import dataclasses


def is_box_valid(box: Array) -> bool:
    """Returns whether a box violates chemtrain's lower-triangular form.

    JAX-MD uses this predicate to set ``MALFORMED_BOX``, despite the upstream
    function name. Scalars and side-length vectors are always valid.
    """
    if jnp.isscalar(box) or box.ndim == 0 or box.ndim == 1:
        return False
    if box.ndim == 2:
        return jnp.bool_(jnp.logical_not(jnp.all(jnp.tril(box) == box)))
    return True


@dataclasses.dataclass
class PartitionError:
    """Stores neighbor-list errors without forcing traced values to Python."""

    code: Array

    def update(self, bit: bytes, pred: Array) -> Array:
        """Adds an error bit where the predicate is true."""
        zero = jnp.zeros((), jnp.uint8)
        bit = jnp.array(bit, dtype=jnp.uint8)
        return PartitionError(self.code | jnp.where(pred, bit, zero))

    def __str__(self) -> str:
        """Returns the available error without failing on a traced code."""
        try:
            if not jnp.any(self.code):
                return ""
        except Exception:
            return f"Error code not available ({self.code})"

        if jnp.any(self.code & jax_md.partition.PEC.NEIGHBOR_LIST_OVERFLOW):
            return "Partition Error: Neighbor list buffer overflow."
        if jnp.any(self.code & jax_md.partition.PEC.CELL_LIST_OVERFLOW):
            return "Partition Error: Cell list buffer overflow"
        if jnp.any(self.code & jax_md.partition.PEC.CELL_SIZE_TOO_SMALL):
            return "Partition Error: Cell size too small"
        if jnp.any(self.code & jax_md.partition.PEC.MALFORMED_BOX):
            return (
                "Partition Error: Incorrect box format. Expecting lower "
                "triangular."
            )
        raise ValueError(f"Unexpected Error Code {self.code}.")

    __repr__ = __str__

jax_md.partition.is_box_valid = is_box_valid
jax_md.partition.PartitionError = PartitionError

import e3nn_jax._src.scatter

from jax_md_mod import e3nn_mod

e3nn_jax._src.scatter._distinct_but_small = e3nn_mod._distinct_but_small
