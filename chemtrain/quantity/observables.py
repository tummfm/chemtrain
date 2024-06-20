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
from jax import vmap, numpy as jnp, lax
from jax_md import simulate, quantity

from chemtrain.quantity import constants

from chemtrain.typing import TrajFn

def init_identity_fn(quantity_key) -> TrajFn:
    """Initialize a wrapper that returns the snapshots unchanged.

    Args:
        quantity_key: Key referencing the quantity in the computed snapshots.

    Returns:
        Returns the snapshots of the quantity.

    """

    def identity(quantity_trajs, weights=None):
        return quantity_trajs[quantity_key]

    return identity


def init_traj_mean_fn(quantity_key) -> TrajFn:
    """Initialize the computation of a (perturbed) ensemble average.

    This function builds the ``'traj_fn'`` of the DiffTRe ``target`` dict for the
    common case of target observables that simply consist of a
    trajectory-average of instantaneous quantities, such as RDF, ADF, pressure
    or density.

    .. math::

       a = \\left\\langle w(\\mathbf r) a(\\mathbf r)\\right\\rangle_{\\tilde U},\\quad \\text{where} \\\\
       w(\\mathbf r) = \\frac{e^{-\\beta (U(\\mathbf r) - \\tilde U(\\mathbf r))}}{\\left\\langle e^{-\\beta (U(\\mathbf r) - \\tilde U(\\mathbf r))}\\right\\rangle}

    Args:
        quantity_key: Key referencing the quantity in the computed snapshots.

    Returns:
        Returns a function to compute an ensemble average when given a
        dictionary of snapshots.
        Returns a perturbed ensemble average if weights are provided.

    """
    def traj_mean(quantity_trajs, weights=None):
        quantity_traj = quantity_trajs[quantity_key]
        return _traj_mean(quantity_traj, weights, linearized=False)
    return traj_mean


def _weighted_mean(quantity_traj, weights):
    if weights is not None:
        weights *= weights.size
        quantity_traj = (quantity_traj.T * weights).T
    return jnp.mean(quantity_traj, axis=0)


def _linearized_mean(quantity_traj, weights):
    # Mask out zero weights
    mask = weights > 1e-30
    weights = jnp.where(mask, weights, 1.0)
    n_nonzero = jnp.sum(mask)

    # Calculate the masked means
    quantity_mean = jnp.mean(quantity_traj, axis=0)
    correction = jnp.sum((quantity_traj.T * jnp.log(weights) * mask).T, axis=0)
    correction -= quantity_mean * jnp.sum(jnp.log(weights) * mask)
    return quantity_mean + correction / n_nonzero


def _traj_mean(quantity_traj, weights=None, linearized=False, **kwargs):
    del kwargs

    if linearized:
        assert weights is not None, (
            "The linearized trajectory average requires weights to be "
            "specified.")
        return _linearized_mean(quantity_traj, weights)
    else:
        return _weighted_mean(quantity_traj, weights)


def init_linear_traj_mean_fn(quantity_key) -> TrajFn:
    """Initializes the linear approximation of an observable average.

    This function approximates ensemble averages of instantaneous quantities
    for a perturbed ensemble based on a cumulant expansion [#imbalzano2021]_.


    .. [#imbalzano2021] Giulio Imbalzano, Yongbin Zhuang, Venkat Kapil,
       Kevin Rossi, Edgar A. Engel, Federico Grasselli, Michele Ceriotti;
       _Uncertainty estimation for molecular dynamics and sampling._
       J. Chem. Phys. 21 February 2021; 154 (7): 074102.
       https://doi.org/10.1063/5.0036522

    Args:
        quantity_key: Key referencing the quantity in the computed snapshots.

    Returns:
        Returns a function to approximate a perturbed ensemble average when
        given a dictionary of snapshots and weights.

    """

    def traj_mean(quantity_trajs, weights=None):
        quantity_traj = quantity_trajs[quantity_key]
        return _traj_mean(quantity_traj, weights, linearized=True)
    return traj_mean


def init_relative_entropy_traj_fn(ref_kbt, reference_key = 'ref_energy') -> TrajFn:
    r"""Initializes the computation of the relative entropy difference between
    the current canonical distribution and reference distribution.

    The relative entropy is given as

    .. math::
        S_\text{rel} = -\int p(x)\log\frac{p(x)}{q(x)}dx.

    For two canonical distributions defined by the potentials U_p and U_q,
    this relative entropy computes as

    .. math::

        S_\text{rel} = \beta\left\langle\left(U_p - U_q\right)\right\rangle_{U_p} - \log\left\langle e^{-\beta(U_q - U_p)}\right\rangle_{U_p}.

    Args:
        ref_kbt: Reference temperature.
        reference_key: Key corresponding to the reference energy quantity.

    Returns:
        Returns a function that computes the relative entropy given a
        dictionary of quantities along a trajectory.

    """
    beta = 1.0 / ref_kbt
    def relative_entropy_traj_fn(quantity_trajs, weights=None):
        energies = quantity_trajs['energy']
        ref_energies = quantity_trajs[reference_key]

        if weights is None:
            weights = jnp.ones_like(energies) / energies.size

        energy_difference = jnp.sum(weights * (energies - ref_energies))

        # Compute the free energy from the current to the reference potential
        # via the forward perturbation and reverse the sign
        exponents = -beta * (ref_energies - energies)
        max_exponent = jnp.max(exponents)
        exp = jnp.exp(exponents - max_exponent)

        # We use this method of free energy calculation only for the value but
        # not for the gradient.
        # The quality of the estimation via the perturbation formula depends
        # on the difference between the potentials, which is perhaps large.
        # Instead, the gradients of the free energy are quite easily computable
        # with respect to any other reference state.

        free_energy_difference = jnp.log(jnp.sum(weights * exp))
        free_energy_difference += max_exponent
        free_energy_difference *= -1.0 / beta
        free_energy_difference = lax.stop_gradient(free_energy_difference)

        # By using stop_gradient, we can ensure the contribution to the gradient
        # without a contribution to the absolute value.
        free_energy_difference -= quantity_trajs['free_energy']
        free_energy_difference += lax.stop_gradient(
            quantity_trajs['free_energy'])

        entropy_difference = energy_difference + free_energy_difference
        entropy_difference *= beta * constants.kb

        return entropy_difference
    return relative_entropy_traj_fn


def init_heat_capacity_nvt(kT, dof, **kwargs):
    """Returns the specific heat capacity of a system in the NPT ensemble via
     the fluctuation formula.

    Args:
        kT: Reference temperature
        dof: Number of degrees of freedom in the system
        kwargs: Additional arguments determining the calculation of trajectory
            averages.

    Returns:
        Returns a function to compute the (perturbed) specific heat capacity
        based on potential energy snapshots.

    References:
        `<https://journals-aps-org.eaccess.tum.edu/pre/pdf/10.1103/PhysRevE.99.012139>`

    """
    def specific_heat_capacity_traj_fn(quantity_trajs, weights=None):
        energy = quantity_trajs['energy']
        if weights is None:
            weights = jnp.ones_like(energy) / energy.size

        fluctuation = _traj_mean(energy ** 2, weights, **kwargs)
        fluctuation -= _traj_mean(energy, weights, **kwargs) ** 2

        c_v = (0.5 * dof + fluctuation / kbt ** 2) * constants.kb
        return c_v
    return specific_heat_capacity_traj_fn


def init_heat_capacity_npt(kT, dof, pressure, **kwargs):
    """Initializes the isobaric heat capacity of a system in the NPT ensemble.

    The heat capacity of the system depends on the fluctuation of the
    conformational enthalpy [#stroeker2021]_.

    Args:
        kT: Thermostat temperature
        dof: Number of degrees of freedom in the system
        kwargs: Extra arguments to the function computing the ensemble
            averages

    Returns:
        Returns a function to compute the (perturbed) specific heat capacity
        based on potential energy snapshots and volume snapshots.

    References:
        .. [#stroeker2021] P. Ströker, R. Hellmann, und K. Meier, „Systematic formulation of thermodynamic properties in the N p T ensemble“, Phys. Rev. E, Bd. 103, Nr. 2, S. 023305, Feb. 2021, doi: 10.1103/PhysRevE.103.023305.


    """

    def cp_fn(quantity_traj, weights=None):
        volume_traj = quantity_traj['volume']
        energy_traj = quantity_traj['energy']

        enthalpy = energy_traj + volume_traj * ref_pressure

        fluctuation = _traj_mean(enthalpy ** 2, weights, **kwargs)
        fluctuation -= _traj_mean(enthalpy, weights, **kwargs) ** 2

        cp = (0.5 * dof + fluctuation / kbt ** 2) * constants.kb
        return cp
    return cp_fn


# TODO: Update the remaining compute functions (see reference above for npt
#       formulations)

def volumes(traj_state):
    """Returns array of volumes for all boxes in a NPT trajectory.

    Args:
        traj_state: TrajectoryState containing the NPT trajectory
    """
    dim = traj_state.sim_state.sim_state.position.shape[-1]
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
