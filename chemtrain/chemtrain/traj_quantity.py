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


def thermal_expansion_coefficient_npt(volume_traj, energy_traj, temperature,
                                      kbt, pressure):
    """Returns the thermal expansion coefficient of a system in the NPT ensemble
    via the fluctuation formula.

    Args:
        volume_traj: Trajectory of box volumes of a NPT simulation,
                     e.g. computed via traj_quantity.volumes.
        energy_traj: Trajectory of potential energies
        temperature: Thermostat temperature
        kbt: Temperature * Boltzmann constant
        pressure: Barostat pressure
    """
    mean_volume = jnp.mean(volume_traj)
    fluctuation = (jnp.mean(volume_traj * energy_traj) - jnp.mean(energy_traj)
                   * mean_volume
                   + pressure * (jnp.mean(volume_traj**2) - mean_volume**2))
    alpha = fluctuation / temperature / kbt / mean_volume
    return alpha


def specific_heat_capacity_npt(volume_traj, energy_traj, temperature, kbt,
                               pressure, n_dof):
    """Returns the specific heat capacity of a system in the NPT ensemble via
     the fluctuation formula.

    Args:
        volume_traj: Trajectory of box volumes of a NPT simulation,
                     e.g. computed via traj_quantity.volumes.
        energy_traj: Trajectory of potential energies
        temperature: Thermostat temperature
        kbt: Temperature * Boltzmann constant
        pressure: Barostat pressure
        n_dof: Number of degrees of freedom in the system
    """
    mean_volume = jnp.mean(volume_traj)
    mean_potential = jnp.mean(energy_traj)
    fluctuation = (jnp.mean(energy_traj**2) - mean_potential**2
                   + 2. * pressure * (jnp.mean(volume_traj * energy_traj)
                                      - mean_volume * mean_potential)
                   + pressure**2 * (jnp.mean(volume_traj**2) - mean_volume**2))
    k_b = kbt / temperature
    c_p = 0.5 * n_dof * k_b + fluctuation / kbt / temperature
    return c_p
