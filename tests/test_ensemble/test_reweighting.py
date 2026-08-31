"""Tests for trajectory reweighting recovery."""

from collections import namedtuple

import jax.numpy as jnp
import numpy as np

from chemtrain.ensemble import reweighting, sampling


class SimulationState(namedtuple("SimulationStateBase", "position")):
    """Minimal simulator state with the JAX-MD update interface."""

    def set(self, **kwargs):
        return self._replace(**kwargs)


NeighborList = namedtuple("NeighborList", "max_occupancy reference_position")


class NeighborFunction:
    """Record requests made when a neighbor list is allocated."""

    def __init__(self):
        self.position = None
        self.extra_capacity = None

    def allocate(self, position, extra_capacity, **kwargs):
        del kwargs
        self.position = position
        self.extra_capacity = extra_capacity
        return NeighborList(
            max_occupancy=extra_capacity,
            reference_position=position,
        )


def test_overflow_recovery_uses_neighbor_reference_and_grows_capacity():
    """Overflow recovery allocates from the neighbor-list reference."""
    neighbor_fn = NeighborFunction()
    reference_position = jnp.ones((2, 3))
    overflowing = sampling.SimulatorState(
        sim_state=SimulationState(jnp.full((2, 3), 9.0)),
        nbrs=NeighborList(
            max_occupancy=8,
            reference_position=reference_position,
        ),
    )
    trajectory = sampling.TrajectoryState(
        sim_state=overflowing,
        trajectory=SimulationState(jnp.zeros((2, 3))),
        overflow=True,
    )

    reset = reweighting.reallocate_overflowing_trajectory(
        trajectory,
        neighbor_fn,
        capacity_factor=1.5,
        extra_capacity=2,
    )

    np.testing.assert_allclose(neighbor_fn.position, reference_position)
    assert neighbor_fn.extra_capacity == 6
    np.testing.assert_allclose(
        reset.sim_state.sim_state.position,
        reference_position,
    )


def test_safe_propagate_retries_the_wrapped_propagation():
    """A retry keeps statepoint arguments in the propagation function."""
    neighbor_fn = NeighborFunction()
    reference = sampling.SimulatorState(
        sim_state=SimulationState(jnp.ones((2, 3))),
        nbrs=NeighborList(
            max_occupancy=4,
            reference_position=jnp.ones((2, 3)),
        ),
    )
    trajectory = sampling.TrajectoryState(
        sim_state=reference,
        trajectory=SimulationState(jnp.zeros((2, 3))),
        overflow=False,
    )
    timings = sampling.TimingClass(
        t_equilib_start=jnp.empty(0),
        t_production_start=jnp.empty(0),
        t_production_end=jnp.empty(0),
        timesteps_per_printout=1,
        time_step=1.0,
    )
    _, _, _, safe_propagate = reweighting.init_pot_reweight_propagation_fns(
        lambda params: params,
        lambda energy: (energy, energy),
        neighbor_fn,
        timings,
        {"kT": 1.0},
        safe_propagation=False,
    )
    recompute_calls = []

    def propagate(params, traj_state, *, recompute=False):
        del params
        recompute_calls.append(recompute)
        return traj_state.replace(overflow=not recompute)

    result = safe_propagate(
        propagate,
        multiple_arguments=False,
        max_retry=2,
    )(None, trajectory)

    assert recompute_calls == [False, True]
    assert not result.overflow
