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


def isothermal_compressibility_npt(volume_traj, kbt):
    """Returns isothermal compressibility of a system in the NPT ensemble
    via the fluctuation formula.

    Args:
        volume_traj: Trajectory of box volumes of a NPT simulation,
                     e.g. computed via traj_quantity.volumes.
        kbt: Temperature * Boltzmann constant
    """
    mean_volume = jnp.mean(volume_traj)
    kappa = (jnp.mean(volume_traj**2) - mean_volume**2) / (mean_volume * kbt)
    return kappa
