# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
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

"""Feature communication and reductions for communication-enabled models.

Outside a communication-enabled variant, ``gather`` returns the validated
input and ``reduce`` returns the rank-local value. During export, both
operations lower to FFI calls implemented by chemtrain-deploy and the
simulation-engine adapter. LAMMPS provides one such adapter. The local behavior
is useful for model initialization and single-domain testing.
"""

from __future__ import annotations

import math

import jax
from jax import numpy as jnp


FORWARD_TARGET = "chemtrain_deploy.gather_forward"
REVERSE_TARGET = "chemtrain_deploy.gather_reverse"
REDUCE_TARGET = "chemtrain_deploy.reduce"
REDUCE_TRANSPOSE_TARGET = "chemtrain_deploy.reduce_transpose"
CUSTOM_CALL_TARGETS = (
    FORWARD_TARGET,
    REVERSE_TARGET,
    REDUCE_TARGET,
    REDUCE_TRANSPOSE_TARGET,
)


class ExportCommunication:
    """Communication interface used while tracing a deployment variant.

    ``gather_widths`` and ``reduce_widths`` record the number of scalar values
    packed by every gather and reduction during the current trace. A fixed
    float32 token orders calls through the public FFI. The token value is
    irrelevant because only the dependency between consecutive calls matters.

    Width recording is a Python tracing-time side effect, not part of the
    compiled computation. The exporter therefore creates a new instance for
    every trace. Retracing replaces the complete width record instead of
    appending metadata from an earlier trace.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.gather_widths: list[int] = []
        self.reduce_widths: list[int] = []
        self.token = jnp.zeros((1,), dtype=jnp.float32)

    def gather(self, tree):
        """Gathers a non-empty, atom-leading floating-point pytree.

        All leaves need the same particle dimension and floating-point dtype.
        Disabled communication still validates the values but does not call
        the FFI or exchange ghost rows.
        """
        width = packed_width(tree)
        # Widths are recorded only while Python traces the function. Compiled
        # model code sees the FFI result and dependency token, not the Python
        # lists.
        self.gather_widths.append(width)
        if not self.enabled:
            return tree
        communicated, self.token = _gather_with_token(tree, self.token)
        return communicated

    def reduce(self, value):
        """Sums a floating scalar or vector across ranks when enabled."""
        array = jnp.asarray(value)
        if array.ndim > 1:
            raise ValueError("comm.reduce supports scalars and vectors")
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("comm.reduce supports floating-point values only")
        width = 1 if array.ndim == 0 else array.shape[0]
        self.reduce_widths.append(width)
        if not self.enabled:
            return array
        reduced, self.token = _reduce_with_token(array, self.token)
        return reduced


def packed_width(tree) -> int:
    """Returns the scalar width of a packed atom-leading floating pytree."""
    arrays, _ = _validated_arrays(tree)
    return sum(math.prod(array.shape[1:]) for array in arrays)


def _validated_arrays(tree):
    """Flattens and validates the layout required by communication."""
    leaves, treedef = jax.tree.flatten(tree)
    if not leaves:
        raise ValueError("comm.gather requires a non-empty pytree")

    arrays = [jnp.asarray(leaf) for leaf in leaves]
    first = arrays[0]
    if first.ndim < 1:
        raise ValueError("comm.gather leaves must have an atom-leading axis")
    if not jnp.issubdtype(first.dtype, jnp.floating):
        raise TypeError("comm.gather supports floating-point arrays only")

    for array in arrays[1:]:
        if array.ndim < 1:
            raise ValueError(
                "comm.gather leaves must have an atom-leading axis"
            )
        if array.dtype != first.dtype:
            raise TypeError("comm.gather leaves must have the same dtype")
        if array.shape[0] != first.shape[0]:
            raise ValueError(
                "comm.gather leaves must have the same atom-leading size"
            )
    return arrays, treedef


def _call_ffi(target, buffer, token):
    """Invokes a typed communication FFI with an ordering token."""
    result_shapes = (
        jax.ShapeDtypeStruct(buffer.shape, buffer.dtype),
        jax.ShapeDtypeStruct(token.shape, token.dtype),
    )
    return jax.ffi.ffi_call(
        target,
        result_shapes,
        has_side_effect=True,
        vmap_method="sequential",
    )(buffer, token)


@jax.custom_vjp
def _communicate(buffer, token):
    return _call_ffi(FORWARD_TARGET, buffer, token)


def _communicate_fwd(buffer, token):
    output = _call_ffi(FORWARD_TARGET, buffer, token)
    return output, None


def _communicate_bwd(_, cotangents):
    buffer_cotangent, token_cotangent = cotangents
    return _call_ffi(REVERSE_TARGET, buffer_cotangent, token_cotangent)


_communicate.defvjp(_communicate_fwd, _communicate_bwd)


@jax.custom_vjp
def _reduce(buffer, token):
    return _call_ffi(REDUCE_TARGET, buffer, token)


def _reduce_fwd(buffer, token):
    output = _call_ffi(REDUCE_TARGET, buffer, token)
    return output, None


def _reduce_bwd(_, cotangents):
    buffer_cotangent, token_cotangent = cotangents
    return _call_ffi(
        REDUCE_TRANSPOSE_TARGET, buffer_cotangent, token_cotangent
    )


_reduce.defvjp(_reduce_fwd, _reduce_bwd)


def _reduce_with_token(value, token):
    original_shape = value.shape
    vector = value.reshape((1,)) if value.ndim == 0 else value
    reduced, token = _reduce(vector, token)
    return reduced.reshape(original_shape), token


def _gather_with_token(tree, token):
    """Gathers atom-leading floating-point arrays with one FFI call.

    Leaves are packed into a single ``[n_atoms, packed_width]`` matrix. The
    reverse-mode transpose is a reverse communication call at the matching
    point in backpropagation.
    """

    arrays, treedef = _validated_arrays(tree)
    first = arrays[0]

    widths = [math.prod(array.shape[1:]) for array in arrays]

    packed = jnp.concatenate(
        [
            array.reshape((array.shape[0], width))
            for array, width in zip(arrays, widths)
        ],
        axis=1,
    )

    communicated, token = _communicate(packed, token)

    if communicated.shape != packed.shape:
        raise ValueError(
            "comm.gather changed shape from "
            f"{packed.shape} to {communicated.shape}"
        )

    unpacked = []
    start = 0
    for array, width in zip(arrays, widths):
        unpacked.append(
            communicated[:, start:start + width].reshape(array.shape)
        )
        start += width

    return jax.tree.unflatten(treedef, unpacked), token


def gather(tree):
    """Validates a pytree without communicating between ranks.

    Leaves are converted with :func:`jax.numpy.asarray`, then returned with the
    original pytree structure, shapes, and values. Deployment models receive
    :class:`ExportCommunication` explicitly. The module-level fallback keeps
    initialization and single-domain reference calculations executable without
    a registered native FFI.
    """
    arrays, treedef = _validated_arrays(tree)
    return jax.tree.unflatten(treedef, arrays)
