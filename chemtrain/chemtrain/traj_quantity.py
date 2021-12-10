"""Molecular dynamics observable functions acting on trajectories rather than
single snapshots.

Builds on the TrajectoryState object defined in traj_util.py.
"""
from jax import vmap, numpy as jnp
from jax_md import simulate, quantity


def volumes(traj_state):
    """Returns array of volumes for all boxes in a NPT trajectory.

    Args:
        traj_state: TrajectoryState containing the NPT trajectory
    """
    dim = traj_state.sim_state[0].position.shape[-1]
    boxes = vmap(simulate.npt_box)(traj_state.trajectory)
    return vmap(quantity.volume, (None, 0))(dim, boxes)


def isothermal_compressibility(traj_state, kbt):
    """Returns isothermal compressibility of a system in the NPT ensemble
    via the fluctuation formula.

    Args:
        traj_state: TrajectoryState containing the NPT trajectory
        kbt: Temperature * Boltzmann constant
    """
    volume_traj = volumes(traj_state)
    mean_volume = jnp.mean(volume_traj, axis=0)
    mean_sq_volume = jnp.mean(volume_traj**2, axis=0)
    kappa = (mean_sq_volume - mean_volume**2) / (mean_volume * kbt)
    return kappa
