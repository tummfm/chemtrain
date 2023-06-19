# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
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

"""Molecular dynamics observable functions acting on trajectories rather than
single snapshots.

Builds on the TrajectoryState object defined in traj_util.py.
"""
from jax import vmap, numpy as jnp
from jax_md import simulate, quantity


def init_traj_mean_fn(quantity_key):
    """Initializes the 'traj_fn' for the DiffTRe 'target' dict for simple
    trajectory-averaged observables.

    This function builds the 'traj_fn' of the DiffTRe 'target' dict for the
    common case of target observables that simply consist of a
    trajectory-average of instantaneous quantities, such as RDF, ADF, pressure
    or density.

    This function also serves as a template on how to build the 'traj_fn' for
    observables that are a general function of one or many instantaneous
    quantities, such as stiffness via the stress-fluctuation method or
    fluctuation formulas in this module. The 'traj_fn' receives a dict of all
    quantity trajectories as input under the same keys as instantaneous
    quantities are defined in 'quantities'. The 'traj_fn' then returns the
    ensemble-averaged quantity, possibly taking advantage of fluctuation
    formulas defined in the traj_quantity module.

    Args:
        quantity_key: Quantity key used in 'quantities' to generate the
                      quantity trajectory at hand, to be averaged over.

    Returns:
        The 'traj_fn' to be used in building the 'targets' dict for DiffTRe.
    """
    def traj_mean(quantity_trajs):
        quantity_traj = quantity_trajs[quantity_key]
        return jnp.mean(quantity_traj, axis=0)
    return traj_mean


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
