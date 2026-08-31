"""MPI integration tests; run this module once under a two-rank launcher."""

import os
import tempfile
from collections import namedtuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from chemtrain import parallel, util
from chemtrain.data.data_loaders import HDF5ParallelDataLoader, init_batch_functions
from chemtrain.ensemble import reweighting
from chemtrain.learn import difftre as difftre_learn
from chemtrain.learn.max_likelihood import mpi_update_fn
from chemtrain.trainers import DifftreParallel, ForceMatching


pytestmark = pytest.mark.mpi


def test_two_rank_hdf5_loading_and_gradient_mean():
    """Check rank-local reads and a compiled cross-rank optimizer update."""
    mpi4jax = pytest.importorskip("mpi4jax")
    del mpi4jax
    comm = util.get_communicator()
    if comm is None or comm.Get_size() != 2:
        pytest.skip("Launch with mpiexec -n 2 and one CPU device per rank.")

    context = parallel.resolve_parallelism("mpi")
    rank = comm.Get_rank()
    path = None
    if rank == 0:
        descriptor, path = tempfile.mkstemp(suffix=".h5")
        os.close(descriptor)
        with h5py.File(path, "w") as handle:
            handle["x"] = np.arange(16, dtype=np.float32)
    path = comm.bcast(path, root=0)
    comm.Barrier()

    loader = HDF5ParallelDataLoader(path)
    init, get, release = init_batch_functions(
        loader, mb_size=4, cache_size=1, parallel_context=context
    )
    state = init(random=False)
    _, batch = get(state)
    np.testing.assert_array_equal(
        batch["x"], np.arange(rank * 2, rank * 2 + 2, dtype=np.float32)
    )
    release()

    def model(params, local_batch):
        return params["weight"] * local_batch["x"]

    def loss(prediction, local_batch):
        return jnp.mean((prediction - 2.0 * local_batch["x"]) ** 2)

    params = {"weight": jnp.array(0.0)}
    optimizer = optax.sgd(0.1)
    update = mpi_update_fn(model, loss, optimizer)
    new_params, _, value, _ = update(params, optimizer.init(params), batch)
    jax.block_until_ready((new_params, value))
    assert jnp.isclose(value, 14.0)
    assert jnp.isclose(new_params["weight"], 1.4)

    comm.Barrier()
    if rank == 0:
        os.unlink(path)


def test_force_matching_trainer_broadcasts_initial_state():
    """The trainer broadcasts its complete parameter and optimizer state."""
    comm = util.get_communicator()
    if comm is None or comm.Get_size() != 2:
        pytest.skip("Launch with mpiexec -n 2 and one CPU device per rank.")

    def energy_template(params):
        def energy(position, **kwargs):
            del kwargs
            return params["scale"] * jnp.sum(position**2)

        return energy

    # Different rank-local inputs verify that rank zero determines the shared
    # state, rather than merely checking two identical initializations.
    initial_params = {"scale": jnp.asarray(1.0 + comm.Get_rank())}
    trainer = ForceMatching(
        initial_params,
        optax.adam(1.0e-3),
        energy_template,
        nbrs_init=None,
        batch=2,
        parallelism="mpi",
        checkpoint_path=tempfile.mkdtemp(),
        log_file=None,
    )

    assert jnp.isclose(trainer.state.params["scale"], 1.0)
    assert all(
        jnp.allclose(local, root)
        for local, root in zip(
            jax.tree.leaves(trainer.state.opt_state),
            jax.tree.leaves(optax.adam(1.0e-3).init({"scale": jnp.asarray(1.0)})),
        )
    )


def test_difftre_parallel_broadcasts_initial_state(monkeypatch, tmp_path):
    """DiffTRe starts every rank from rank zero's state and RNG key."""
    comm = util.get_communicator()
    if comm is None or comm.Get_size() != 2:
        pytest.skip("Launch with mpiexec -n 2 and one CPU device per rank.")

    def initial_trajectory(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Trajectory initialization is outside this regression.")

    def batched_model(params, *args):
        del args
        return params["weight"], {}

    def batched_weights(*args):
        del args
        return None, jnp.ones(1)

    monkeypatch.setattr(
        reweighting,
        "init_pot_reweight_propagation_fns",
        lambda *args, **kwargs: (
            initial_trajectory,
            object(),
            object(),
            lambda function, **options: function,
        ),
    )
    monkeypatch.setattr(
        difftre_learn,
        "init_difftre_gradient_and_propagation",
        lambda *args, **kwargs: (batched_model, lambda *a: a[1], batched_weights),
    )

    reference_states = namedtuple("ReferenceStates", ["sim_state", "nbrs"])(
        namedtuple("SimulationState", ["position"])(jnp.zeros((2, 1))),
        None,
    )
    trainer = DifftreParallel(
        key=jax.random.PRNGKey(comm.Get_rank()),
        init_params={"weight": jnp.asarray(1.0 + comm.Get_rank())},
        optimizer=optax.sgd(0.1),
        energy_fn_template=lambda _: None,
        simulator_template=lambda _: None,
        neighbor_fn=None,
        timings=None,
        state_kwargs={"kT": jnp.ones(2)},
        quantities={},
        targets={"observable": {"target": jnp.zeros(2)}},
        observables={},
        reference_states=reference_states,
        sim_batch_size=2,
        checkpoint_path=tmp_path / "checkpoints",
    )

    np.testing.assert_allclose(trainer.params["weight"], 1.0)
    np.testing.assert_array_equal(trainer.key, jax.random.PRNGKey(0))


def test_difftre_parallel_update_uses_global_batch_mean():
    """An uneven MPI slice matches the mean update and prediction order."""
    comm = util.get_communicator()
    if comm is None or comm.Get_size() != 2:
        pytest.skip("Launch with mpiexec -n 2 and one CPU device per rank.")

    trajectory = namedtuple("Trajectory", ["position"])(
        jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    )
    trajectory_states = namedtuple("TrajectoryStates", ["trajectory"])(
        trajectory
    )
    optimizer = optax.sgd(0.1)
    params = {"weight": jnp.asarray(0.0)}

    trainer = DifftreParallel.__new__(DifftreParallel)
    trainer.parallel_context = parallel.resolve_parallelism("mpi")
    trainer.state = util.TrainerState(
        params=params, opt_state=optimizer.init(params)
    )
    trainer.optimizer = optimizer
    trainer.traj_states = trajectory_states
    trainer.targets = {
        "observable": {"target": jnp.asarray([1.0, 2.0, 3.0])}
    }
    trainer.statepoints = {"value": jnp.asarray([10.0, 20.0, 30.0])}
    trainer._traj_states_on_host = False
    trainer.reweight_ratio = 0.5
    trainer._epoch = 0
    trainer.predictions = {0: {}, 1: {}, 2: {}}
    trainer.batch_gradient_norms = []
    trainer.batch_losses = []
    trainer.batch_statepoint_counts = []
    trainer.step_size_history = []
    trainer._adaptive_step_size = None

    def weights(_, local_trajectories):
        local_size = local_trajectories.trajectory.position.shape[0]
        return None, jnp.full(local_size, 2.0)

    def model(_, local_trajectories, statepoints, targets):
        del local_trajectories
        local_targets = targets["observable"]["target"]
        predictions = {"observable": statepoints["value"]}
        local_mean = jnp.mean(local_targets)
        return (local_mean, predictions), {"weight": local_mean}

    trainer.weights = weights
    trainer.model = model
    trainer.propagate = lambda *_: pytest.fail("unexpected recomputation")

    trainer._update(jnp.asarray([0, 1, 2]))

    np.testing.assert_allclose(trainer.params["weight"], -0.2)
    np.testing.assert_allclose(trainer.batch_losses[-1], 2.0)
    predictions = [
        trainer.predictions[index][0]["observable"] for index in range(3)
    ]
    np.testing.assert_allclose(predictions, [10.0, 20.0, 30.0])
