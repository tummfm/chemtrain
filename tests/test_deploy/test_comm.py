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

import jax
from jax import numpy as jnp
import pytest

from chemtrain.deploy import comm


def test_gather_packs_and_unpacks_pytree():
    tree = {
        "scalar": jnp.arange(4, dtype=jnp.float32),
        "tensor": jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3),
    }
    gathered = comm.gather(tree)
    assert jax.tree.structure(gathered) == jax.tree.structure(tree)
    for actual, expected in zip(
        jax.tree.leaves(gathered), jax.tree.leaves(tree)
    ):
        assert actual.shape == expected.shape
        assert jnp.array_equal(actual, expected)


def test_export_communication_records_sites_and_packed_widths():
    communication = comm.ExportCommunication()
    features = {
        "scalar": jnp.ones((4,), dtype=jnp.float32),
        "tensor": jnp.ones((4, 2, 3), dtype=jnp.float32),
    }

    communication.gather(features)
    communication.gather(jnp.ones((4, 3), dtype=jnp.float32))
    communication.reduce(jnp.ones((5,), dtype=jnp.float32))

    assert communication.token.shape == (1,)
    assert communication.gather_widths == [7, 3]
    assert communication.reduce_widths == [5]


def test_gather_vjp_uses_forward_and_reverse_side_effecting_calls():
    def loss(x):
        communication = comm.ExportCommunication(enabled=True)
        x = communication.gather(x)
        x = communication.gather(x)
        return jnp.sum(x ** 2)

    lowered = jax.jit(jax.grad(
        loss
    )).lower(jax.ShapeDtypeStruct((4, 3), jnp.float32))
    stablehlo = str(lowered.compiler_ir("stablehlo"))

    assert stablehlo.count(f"@{comm.FORWARD_TARGET}") == 2
    assert stablehlo.count(f"@{comm.REVERSE_TARGET}") == 2
    assert stablehlo.count("has_side_effect = true") == 4
    assert "!stablehlo.token" not in stablehlo

    forward = stablehlo.index(f"@{comm.FORWARD_TARGET}")
    reverse = stablehlo.index(f"@{comm.REVERSE_TARGET}")
    assert forward < reverse

    # A fixed-size array token orders calls without a private JAX effect.
    call_lines = [
        line for line in stablehlo.splitlines()
        if comm.FORWARD_TARGET in line or comm.REVERSE_TARGET in line
    ]
    assert len(call_lines) == 4
    assert all("tensor<1xf32>" in line for line in call_lines)


@pytest.mark.parametrize(
    "tree, exception, message",
    [
        ({}, ValueError, "non-empty"),
        (jnp.ones((), dtype=jnp.float32), ValueError, "atom-leading"),
        (jnp.ones((2,), dtype=jnp.int32), TypeError, "floating-point"),
        (
            (jnp.ones((2, 1)), jnp.ones((3, 1))),
            ValueError,
            "atom-leading size",
        ),
        (
            (jnp.ones((2, 1), dtype=jnp.float32),
             jnp.ones((2, 1), dtype=jnp.float16)),
            TypeError,
            "same dtype",
        ),
    ],
)
def test_gather_validation(tree, exception, message):
    with pytest.raises(exception, match=message):
        comm.gather(tree)
