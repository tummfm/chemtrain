"""Tests for the optional MACE-JAX composition helpers."""

import jax.numpy as jnp
import pytest

pytest.importorskip("mace_jax")

from chemtrain.compose import mace_jax


class ConvertedModelStub:
    """Minimal converted model that exposes the selected graph head."""

    def apply(self, params):
        del params

        def evaluate(data, **kwargs):
            del kwargs
            node_energy = jnp.full(
                data["node_attrs"].shape[0],
                data["head"][0],
                dtype=jnp.float32,
            )
            return {"node_energy": node_energy}, None

        return evaluate


@pytest.fixture
def stubbed_mace_conversion(monkeypatch):
    """Replace conversion and graph construction with deterministic stubs."""

    monkeypatch.setattr(
        mace_jax.mace_jax_from_torch,
        "convert_model",
        lambda *args, **kwargs: (ConvertedModelStub(), {}, None),
    )

    def readout_vectors(*args, **kwargs):
        del args, kwargs
        return (
            jnp.zeros((2, 3), dtype=jnp.float32),
            jnp.asarray((0, 1), dtype=jnp.int32),
            jnp.asarray((1, 0), dtype=jnp.int32),
        )

    monkeypatch.setattr(
        mace_jax.custom_partition,
        "readout_vectors",
        readout_vectors,
    )


@pytest.mark.parametrize(
    ("head", "expected_index"),
    [(None, 0), ("second", 1), (1, 1)],
)
@pytest.mark.usefixtures("stubbed_mace_conversion")
def test_mace_head_reaches_converted_graph(head, expected_index):
    """Bind the selected model head as one graph-level integer."""

    config = {
        "heads": ["first", "second"],
        "r_max": 5.0,
        "num_elements": 2,
    }
    variables, apply_fn = mace_jax.mace_jax_neighborlist_from_torch(
        config,
        object(),
        lambda left, right, **kwargs: left - right,
        max_edge_multiplier=None,
        per_particle=True,
        scale_pos=1.0,
        scale_pot=1.0,
        head=head,
    )

    energy = apply_fn(
        variables,
        jnp.zeros((2, 3), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        species=jnp.asarray((0, 1), dtype=jnp.int32),
    )

    assert jnp.array_equal(
        energy,
        jnp.full((2,), expected_index, dtype=jnp.float32),
    )


@pytest.mark.parametrize("head", ["missing", 2, -1, True, 1.5])
def test_mace_head_rejects_invalid_selection(head):
    """Reject head values that do not identify an exported model head."""

    with pytest.raises(ValueError, match="MACE head"):
        mace_jax.mace_jax_neighborlist_from_torch(
            {"heads": ["first", "second"]},
            object(),
            lambda left, right, **kwargs: left - right,
            head=head,
        )
