"""Tests for model-independent MPI tree utilities."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from chemtrain import util


def test_mpi_guard_returns_after_serial_success():
    with util.mpi_guard():
        value = 3

    assert value == 3


def test_mpi_guard_chains_serial_failure():
    with pytest.raises(RuntimeError, match="rank 0: ValueError: invalid") as error:
        with util.mpi_guard():
            raise ValueError("invalid")

    assert isinstance(error.value.__cause__, ValueError)


def test_mpi_guard_reports_remote_failures_on_every_rank(monkeypatch):
    communicator = SimpleNamespace(
        allgather=lambda error: [error, "IndexError: remote failure"]
    )
    monkeypatch.setattr(util, "use_mpi", lambda: True)
    monkeypatch.setattr(util, "get_communicator", lambda: communicator)

    with pytest.raises(
        RuntimeError, match="rank 1: IndexError: remote failure"
    ):
        with util.mpi_guard():
            pass


def test_packed_mean_groups_arrays_by_dtype(monkeypatch):
    calls = []

    monkeypatch.setattr(util, "use_mpi", lambda: True)

    def record_mean(array):
        calls.append(array)
        return array

    monkeypatch.setattr(util, "mpi_tree_mean", record_mean)
    tree = {
        "float": (jnp.arange(3, dtype=jnp.float32), jnp.ones(2, jnp.float32)),
        "half": jnp.arange(4, dtype=jnp.float16).reshape(2, 2),
    }

    result = util.mpi_tree_mean_packed(tree)

    assert len(calls) == 1
    assert calls[0].dtype == jnp.float32
    assert result["float"][0].dtype == jnp.float32
    assert result["float"][1].shape == (2,)
    assert result["half"].dtype == jnp.float16
    assert result["half"].shape == (2, 2)


def test_first_masked_uses_direct_mpi4jax_arrays(monkeypatch):
    """mpi4jax collectives return arrays without a token argument."""
    calls = []

    class Communicator:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

    def allreduce(array, *, op, comm):
        calls.append((array, op, comm))
        return array

    communicator = Communicator()
    monkeypatch.setattr(util, "use_mpi", lambda: True)
    monkeypatch.setattr(
        util, "get_mpi4jax", lambda: SimpleNamespace(allreduce=allreduce)
    )
    monkeypatch.setattr(
        util,
        "get_mpi",
        lambda: SimpleNamespace(MIN="min", SUM="sum", LOR="lor"),
    )
    monkeypatch.setattr(util, "_get_mpi4jax_communicator", lambda: communicator)

    selected, found = util.mpi_tree_first_masked(
        {"values": jnp.asarray([[4.0], [7.0]])},
        jnp.asarray([False, True]),
    )

    assert bool(found)
    assert selected["values"].item() == 7.0
    assert len(calls) == 2
