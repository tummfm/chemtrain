"""Regression tests for using ChemTrain without optional MPI packages."""

import importlib
import importlib.abc
import sys

import pytest

from chemtrain import util


class _MissingMPI(importlib.abc.MetaPathFinder):
    """Make imports behave as if mpi4py and mpi4jax are not installed."""

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "mpi4py" or fullname.startswith("mpi4py."):
            raise ModuleNotFoundError(name=fullname)
        if fullname == "mpi4jax" or fullname.startswith("mpi4jax."):
            raise ModuleNotFoundError(name=fullname)
        return None


@pytest.fixture
def missing_mpi(monkeypatch):
    """Reload the optional-MPI boundary with both dependencies unavailable."""
    blocked = _MissingMPI()
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "mpi4py"
        or name.startswith("mpi4py.")
        or name == "mpi4jax"
        or name.startswith("mpi4jax.")
    }
    for name in saved_modules:
        del sys.modules[name]
    monkeypatch.setattr(sys, "meta_path", [blocked, *sys.meta_path])
    importlib.reload(util)
    try:
        yield
    finally:
        sys.meta_path.remove(blocked)
        for name in tuple(sys.modules):
            if (
                name == "mpi4py"
                or name.startswith("mpi4py.")
                or name == "mpi4jax"
                or name.startswith("mpi4jax.")
            ):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        importlib.reload(util)


def test_non_mpi_imports_and_parallelism_are_available(missing_mpi):
    """Import serial and JAX paths after hiding both MPI dependencies."""
    from chemtrain import parallel
    from chemtrain.data import data_loaders  # noqa: F401
    from chemtrain.learn import max_likelihood  # noqa: F401
    from chemtrain import trainers  # noqa: F401

    assert not util.has_mpi4py()
    assert not util.has_mpi4jax()
    assert not util.use_mpi()
    assert parallel.resolve_parallelism("auto").mode != "mpi"


def test_serial_mpi_utilities_and_explicit_error_without_dependencies(missing_mpi):
    """MPI helpers fall back locally and explicit MPI names its dependency."""
    from chemtrain import parallel

    tree = {"values": [1, 2]}
    assert util.mpi_tree_slice(tree) == (tree, None)
    assert util.mpi_tree_gather(tree) == tree
    assert util.mpi_tree_mean(tree) == tree
    assert util.mpi_tree_broadcast(tree) == tree

    with pytest.raises(RuntimeError, match="mpi4py"):
        parallel.resolve_parallelism("mpi")


def test_mpi_packages_are_loaded_lazily(missing_mpi):
    """Reloading utilities leaves optional MPI dependencies unloaded."""
    assert "mpi4py" not in sys.modules
    assert "mpi4jax" not in sys.modules
    assert util._import_optional.cache_info().currsize == 0
    assert util.get_mpi4jax() is None
