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

"""Implementation of the reweighting formalism.

Allows re-using existing trajectories.
p(S) = ...
"""
from abc import abstractmethod
import functools

import time
import warnings

import numpy as onp

import jax_sgmc.util
from jax import (checkpoint, lax, random, grad, tree_util, vmap, debug,
                 numpy as jnp)

import jax_md.util
from jax_md import util as jax_md_util

from chemtrain import util
from chemtrain.trajectory import traj_util
from chemtrain.learn import max_likelihood
from chemtrain.jax_md_mod import custom_quantity
from chemtrain.pickle_jit import jit
from chemtrain.quantity import constants

from typing import Dict, Any, Union, Callable, Tuple
try:
    from jax.typing import ArrayLike
except:
    ArrayLike = Any
from jax_md.partition import NeighborFn
from chemtrain.typing import EnergyFnTemplate, TargetDict
from chemtrain.trajectory.traj_util import TrajectoryState

def checkpoint_quantities(compute_fns):
    """Applies checkpoint to all compute_fns to save memory on backward pass."""
    for quantity_key in compute_fns:
        compute_fns[quantity_key] = checkpoint(compute_fns[quantity_key])


def _estimate_effective_samples(weights):
    """Returns the effective sample size after reweighting to
    judge reweighting quality.
    """
    # mask to avoid NaN from log(0) if a few weights are 0.
    weights = jnp.where(weights > 1.e-10, weights, 1.e-10)
    exponent = -jnp.sum(weights * jnp.log(weights))
    return jnp.exp(exponent)


def _build_weights(exponents):
    """Returns weights and the effective sample size from exponents
    of the reweighting formulas in a numerically stable way.
    """

    # The reweighting scheme is a softmax, where the exponent above
    # represents the logits. To improve numerical stability and
    # guard against overflow it is good practice to subtract the
    # max of the exponent using the identity softmax(x + c) =
    # softmax(x). With all values in the exponent <=0, this
    # rules out overflow and the 0 value guarantees a denominator >=1.
    exponents -= jnp.max(exponents)
    prob_ratios = jnp.exp(exponents)
    weights = prob_ratios / jax_md_util.high_precision_sum(prob_ratios)
    n_eff = _estimate_effective_samples(weights)
    return weights, n_eff


def reweight_trajectory(traj, **targets):
    """Computes weights to reweight a trajectory from one thermodynamic
    state point to another.

    This function allows re-using an existing trajectory to compute
    observables at slightly perturbed thermodynamic state points. The
    reference trajectory can be generated at a constant state point or
    at different state points, e.g. via non-equlinibrium MD. Both NVT and
    NPT trajectories are supported, however reweighting currently only
    allows reweighting into the same ensemble. For NVT, the trajectory can
    be reweighted to a different temperature. For NPT, can be reweighted to
    different kbT and/or pressure. We assume quantities not included in
    'targets' to be constant over the trajectory, however this is not ensured
    by the code.
    Implemented are cases 1. - 4. of the reference [#plumed]_.


    Args:
        traj: Reference trajectory to be reweighted
        targets: Kwargs containing the targets under 'kT' and/or 'pressure'.
            If a keyword is not provided, the qunatity is assumed to be and
            remain constant.

    Returns:
        A tuple (weights, n_eff). Weights can be used to compute
        reweighted observables and n_eff judges the expected
        statistical error from reweighting.

    References:
        .. [#plumed] `<https://www.plumed.org/doc-v2.6/user-doc/html/_r_e_w_e_i_g_h_t__t_e_m_p__p_r_e_s_s.html>`_ # pylint: disable=line-too-long

    """
    npt_ensemble = util.is_npt_ensemble(traj.sim_state[0])
    if not npt_ensemble:
        assert 'kT' in targets, 'For NVT, a "kT" target needs to be provided.'
    # Note: if temperature / pressure are supposed to remain constant and are
    # hence not provided in the targets, we set them to the respective reference
    # values. Hence, their contribution to reweighting cancels. This should
    # even be at no additional cost under jit as XLA should easily detect the
    # zero contribution. Same applies to combinations in the NPT ensemble.
    target_kbt = targets.get('kT', traj.thermostat_kbt)
    target_beta = 1. / target_kbt
    reference_betas = 1. / traj.thermostat_kbt

    # temperature reweighting
    if 'energy' not in traj.aux:
        raise ValueError('For reweighting, energies need to be provided '
                         'alongside the trajectory. Add energy to auxilary '
                         'outputs in trajectory generator.')
    exponents = -(target_beta - reference_betas) * traj.aux['energy']

    if npt_ensemble:  # correct for P * V
        assert 'kbT' in targets or 'pressure' in targets, ('At least one target'
                                                           ' needs to be given '
                                                           'for reweighting.')
        target_press = targets.get('pressure', traj.barostat_press)
        target_beta_p = target_beta * target_press
        ref_beta_p = reference_betas * traj.barostat_press
        volumes = traj_quantity.volumes(traj)

        # For constant p, reduces to -V * P_ref * (beta_target - beta_ref)
        # For constant T, reduces to -V * beta_ref * (p_target - p_ref)
        exponents -= volumes * (target_beta_p - ref_beta_p)

    return _build_weights(exponents)


def init_reference_trajectory_reweight_fns(energy_fn_template: EnergyFnTemplate,
                                           neighbor_fn: NeighborFn,
                                           target_quantities: Dict[str, Any],
                                           ref_kbt: ArrayLike,
                                           ref_pressure: ArrayLike = None,
                                           compute_fns: Dict[str, Callable] = None,
                                           energy_batch_size: int = 10,
                                           dynamic_dropout: bool = False,
                                           reference_energy_fn_template: EnergyFnTemplate = None,
                                           pressure_correction: bool = False
                                           ) -> [Callable, Callable]:
    """Initializes reweighting based on a reference trajectory.

    Instead of recomputing a trajectory, this function uses a precomputed
    trajectory and quantities to estimate quantities for a different potential
    model via the reweighing procedure.

    .. code-block :: python

        # Initialize the reference reweighting method
        init_fn, compute_fn = init_reference_trajectory_reweight_fn(
            energy_fn_template, neighbour_fn, target_quantities, ref_kbt)

        # Provide the reference trajectory and reference quantities
        state = init_fn(reference_trajectory, reference_quantities)

        # Compute the new quantities
        results = compute_fn(state, new_energy_parameters)

    Args:
        energy_fn_template: Perturbed potential model
        neighbor_fn: Neighbour list function
        target_quantities: Quantities to estimate via reweighting
        ref_kbt: Reference microscopic temperature
        ref_pressure: Reference pressure
        compute_fns: Functions to recompute quantities used in reweighting. The
            energy is automatically recomputed. It is possible to add quantities
            that are not contained in the reference quantities and required to
            recompute quantities that depdend directly on the potential, i.e.
            the pressure.
        energy_batch_size: Number of configurations to compute in parallel
        dynamic_dropout: Issues a new dropout key for every state of the
            trajectory.
        reference_energy_fn_template: Energy function to re-compute the energies
            of the potential used to generate the trajectory.
        pressure_correction: Include the pressure in the Boltzmann factor for
            the NPT ensemble.

    Returns:
        Returns a dictionary containing the quantities obtained via reweighting,
        the original quantities and the weights as well as the effective sample
        size during the reweighting procedure.

    """

    # The energy of the new potential model is necessary for reweighting
    traj_energy_fn = custom_quantity.energy_wrapper(energy_fn_template)

    if compute_fns is None:
        compute_fns = {}
    compute_fns['energy'] = traj_energy_fn


    beta = 1. / ref_kbt
    checkpoint_quantities(compute_fns)

    def init_reference_reweighting(reference_trajectory: TrajectoryState,
                                   reference_quantities: Dict[str, Any],
                                   extra_acpacity: int = 0,
                                   reference_params: Any = None,
                                   ) -> Any:
        """Inits reweighting based on a reference trajectory."""

        # Take the last frame of the reference trajectory as the simulation
        # state and initialize the neighbour list to create a state as expected
        # by the traj_util.quantity_traj function

        first_frame = util.tree_get_single(reference_trajectory)
        nbr_state = util.neighbor_allocate(
            neighbor_fn, first_frame, extra_capacity=extra_acpacity)

        n_samples = reference_quantities['energy'].size

        thermostat = jnp.ones(n_samples) * ref_kbt
        if ref_pressure is not None:
            barostat = jnp.ones(n_samples) * ref_pressure
        else:
            barostat = None

        initial_state = traj_util.TrajectoryState(
            sim_state=(first_frame, nbr_state),
            trajectory=reference_trajectory,
            overflow=False,
            thermostat_kbt=thermostat,
            barostat_press=barostat,
            aux=reference_quantities,
            energy_params=reference_params
        )

        # TODO: Rework for the new trajectory quantities
        # Without the potential, some reference quantities cannot be
        # calculated. Therefore, we just require all reference quantities
        # upfront. Otherwise, it would be necessary to also provide the
        # reference potential model. We check that all required reference
        # quantities were provided or can be recomputed by a compute_fn.
        # available_quantities = list(compute_fns.keys())
        # available_quantities += list(reference_quantities.keys())
        # missing = []
        # for target_key in target_quantities.keys():
        #     if target_key not in available_quantities:
        #         missing.append(target_key)
        # assert len(missing) == 0, f'Missing quantities, {missing}' \
        #                           f'must be provided as reference quantitiy ' \
        #                           f'or as compute_fn.'

        return initial_state

    def compute_reweighted_quantities(reference_state: TrajectoryState,
                                      energy_params: Any,
                                      dropout_key: ArrayLike = None
                                      ) -> Dict[str, Union[Any, ArrayLike]]:
        """Computes weights for the reweighting approach."""

        npt_ensemble = util.is_npt_ensemble(reference_state.sim_state[0])

        dropout_keys = None
        if dropout_key is not None:
            n_samples = reference_state.trajectory.position.shape[0]
            if dynamic_dropout:
                dropout_keys = random.split(dropout_key, n_samples)
            else:
                dropout_keys = jnp.tile(dropout_key, (n_samples, 1))

        # TODO: Recompute also the reference quantities if given a reference energy
        # template
        if reference_energy_fn_template is not None:
            print(f"Recompute reference quantities")

            reference_compute_fns = {}
            reference_compute_fns["energy"] = custom_quantity.energy_wrapper(
                reference_energy_fn_template)

            if npt_ensemble:
                reference_compute_fns["conf_pressure"] = custom_quantity.init_pressure(
                    reference_energy_fn_template, include_kinetic=False
                )


            ref_quantities = traj_util.quantity_traj(
                reference_state, reference_compute_fns, reference_state.energy_params,
                energy_batch_size, dropout_keys
            )
        else:
            ref_quantities = reference_state.aux

        if npt_ensemble:
            compute_fns['conf_pressure']  = custom_quantity.init_pressure(
                energy_fn_template, include_kinetic=False
            )

        # reweighting properties (U and pressure) under perturbed potential
        quantities = traj_util.quantity_traj(
            reference_state, compute_fns, energy_params,
            energy_batch_size, dropout_keys)


        # Note: Difference in pot. Energy is difference in total energy
        # as kinetic energy is the same and cancels.
        exponent = quantities['energy'] - ref_quantities['energy']

        # In the npt ensemble, the pressure depends on the particle positions
        # and thus contribute to the boltzmann factor similar to the potential.
        # The volume instead depends only on the particle positions and is
        # thus equivalent for both states
        if npt_ensemble and pressure_correction:
            print(f"Consider the pressure in the boltzmann factor.")

            exponent = quantities['volume'] * (
                quantities['conf_pressure'] - ref_quantities['conf_pressure']
            )

        exponent *= -beta
        weights, n_eff = _build_weights(exponent)

        # Compute raw quantities and update re-computed reference quantities.
        for key, quant in reference_state.aux.items():
            if key not in quantities.keys():
                quantities[key] = quant

        reweighted_quantities = {
            target_key: target['traj_fn'](quantities, weights=weights)
            for target_key, target in target_quantities.items()
        }

        result = {
            'weights': weights,
            'exponent': exponent,
            'n_eff': n_eff,
            'reference_quantities':  ref_quantities,
            'unweighted_quantities': quantities,
            'reweighted_quantities': reweighted_quantities
        }

        return result

    return init_reference_reweighting, compute_reweighted_quantities


def init_pot_reweight_propagation_fns(energy_fn_template, simulator_template,
                                      neighbor_fn, timings, ref_kbt,
                                      ref_press=None, reweight_ratio=0.9,
                                      npt_ensemble=False, energy_batch_size=10,
                                      entropy_approximation=False,
                                      max_iter_bar=25, safe_propagation=True,
                                      replica_kbt=None, num_chains=None,
                                      ):
    """
    Initializes all functions necessary for trajectory reweighting for
    a single state point.

    Initialized functions include a function that computes weights for a
    given trajectory and a function that propagates the trajectory forward
    if the statistical error does not allow a re-use of the trajectory.
    The propagation function also ensures that generated trajectories
    did not encounter any neighbor list overflow.
    """
    traj_energy_fn = custom_quantity.energy_wrapper(energy_fn_template)
    reweighting_quantities = {'energy': traj_energy_fn}


    bennett_free_energy = init_bar(
        energy_fn_template, ref_kbt, energy_batch_size, max_iter_bar)

    if npt_ensemble:
        # pressure currently only used to print pressure of generated trajectory
        # such that user can ensure correct statepoint of reference trajectory
        pressure_fn = custom_quantity.init_pressure(energy_fn_template)
        reweighting_quantities['pressure'] = pressure_fn

    trajectory_generator = traj_util.trajectory_generator_init(
        simulator_template, energy_fn_template, timings, reweighting_quantities,
    )

    beta = 1. / ref_kbt
    checkpoint_quantities(reweighting_quantities)

    # TODO: The following steps would be necessary to enable a
    #       parallel/vectorized version of the reweighting framework:
    #       1. Compute weights for all samples
    #       2. Select n samples (weighted) without replacement
    #       3. Shuffle the velocities and reasign randomly
    #       4. Run vectorized simulations

    def resample_new_simstate(params, traj_state):
        sim_state, nbrs = traj_state.sim_state
        momentum = traj_state.trajectory.momentum

        # Position shape ends with [n_samples, n_particles, 3]
        weights, *_ = compute_weights(params, traj_state)
        num_samples = weights.size


        # Choose new initial positions from the reweighted probability
        # distribution.
        key, split1, split2 = random.split(traj_state.key, 3)
        print(f"Weights have shape {weights.shape}, sample count is {num_samples} ")
        new_position_idx = random.choice(
            split1, jnp.arange(num_samples), shape=(num_chains,),
            replace=True, p=weights
        )

        new_sim_state = tree_util.tree_map(
            lambda x: x[new_position_idx, ...], traj_state.trajectory
        )

        # Since velocities are independent of the potential, we can simply
        # draw from the already sampled velocity distribution
        new_momentum = random.shuffle(split2, momentum)[:num_chains, ...]
        new_sim_state = new_sim_state.set(momentum=new_momentum)

        new_traj_state = traj_state.replace(
            sim_state=(new_sim_state, nbrs), key=key
        )

        print(tree_util.tree_map(jnp.shape, new_sim_state))

        return new_traj_state


    def compute_weights(params, traj_state, entropy_and_free_energy=False):
        """Computes weights for the reweighting approach."""

        # reweighting properties (U and pressure) under perturbed potential
        reweight_properties = traj_util.quantity_traj(
            traj_state, reweighting_quantities, params, energy_batch_size)

        # Note: Difference in pot. Energy is difference in total energy
        # as kinetic energy is the same and cancels
        exponent = -beta * (reweight_properties['energy']
                            - traj_state.aux['energy'])

        weights, n_eff = _build_weights(exponent)

        if not entropy_and_free_energy:
            return weights, n_eff
        else:
            # Compute the free energy difference and entropy difference to the
            # potential model that generated the trajectory
            max_exp = jnp.max(exponent)
            ratio_sum = jax_md_util.high_precision_sum(
                jnp.exp(exponent - max_exp))
            log_n = jnp.log(exponent.size)
            free_energy_diff = jnp.log(ratio_sum) + max_exp - log_n
            free_energy_diff *= -1. / beta

            if entropy_approximation:
                # Use an approximate formulation of the entropy with an
                # analytically equivalent gradient
                w_fixed = lax.stop_gradient(weights)
                entropy = jnp.sum(
                    (reweight_properties['energy'].T ** 2) * w_fixed)
                entropy -= jnp.sum(
                    reweight_properties['energy'].T * w_fixed) ** 2
                entropy -= jnp.var(traj_state.aux['energy'])
                entropy *= -constants.kb * beta ** 2
            else:
                # This is the thermodynamic entropy definition
                eng_diff = jnp.sum(reweight_properties['energy'].T * weights)
                eng_diff -= jnp.mean(traj_state.aux['energy'])
                entropy = constants.kb * beta * (eng_diff - free_energy_diff)

            # Add the differences with respect to the initial potential
            entropy += traj_state.entropy_diff
            free_energy_diff += traj_state.free_energy_diff

            return weights, n_eff, entropy, free_energy_diff

    def trajectory_identity_mapping(inputs):
        """Re-uses trajectory if no recomputation needed."""
        traj_state = inputs[1]
        return traj_state

    def recompute_trajectory(inputs):
        """Recomputes the reference trajectory, starting from the last
        state of the previous trajectory to save equilibration time.
        """
        params, traj_state = inputs
        # give kT here as additional input to be handed through to energy_fn
        # for kbt-dependent potentials

        print(f"[Recompute] Traj state has key {traj_state.key}")

        if num_chains is not None:
            traj_state = resample_new_simstate(params, traj_state)

        updated_traj = trajectory_generator(params, traj_state.sim_state,
                                            kT=replica_kbt, pressure=ref_press)
        updated_traj = updated_traj.replace(key=traj_state.key)

        # Apply the BAR procedure to obtain the free energy difference between
        # the old and new trajectory

        dfe, ds = bennett_free_energy(traj_state, updated_traj)
        updated_traj = updated_traj.replace(
            entropy_diff=traj_state.entropy_diff + ds,
            free_energy_diff=traj_state.free_energy_diff + dfe
        )

        return updated_traj


    @jit
    def propagation_fn(params, traj_state):
        """Checks if a trajectory can be re-used. If not, a new trajectory
        is generated ensuring trajectories are always valid.
        Takes params and the traj_state as input and returns a
        trajectory valid for reweighting as well as an error code
        indicating if the neighborlist buffer overflowed during trajectory
        generation.
        """
        _, n_eff = compute_weights(params, traj_state)
        n_snapshots = traj_state.aux['energy'].size

        recompute = n_eff < reweight_ratio * n_snapshots
        propagated_state = lax.cond(recompute,
                                    recompute_trajectory,
                                    trajectory_identity_mapping,
                                    (params, traj_state))
        return propagated_state


    def safe_propagate(fun, multiple_arguments=True, max_retry=3):
        """Re-executes the wrapped function until errors are resolved."""

        def wrapped(params, traj_state, *args, **kwargs):
            recompute = kwargs.pop("recompute", False)

            last_state, _ = traj_state.sim_state
            for reset_counter in range(max_retry):
                # When only propagating then only the trajectory is returned
                if multiple_arguments:
                    traj_state, *returns = fun(
                        params, traj_state, *args, **kwargs)
                else:
                    traj_state = fun(params, traj_state)
                    returns = None

                if recompute:
                    print(f"Forces recomputation")
                    traj_state = recompute_trajectory(
                        (params, traj_state))

                if traj_state.overflow:
                    last_state, _ = traj_state.sim_state
                    if last_state.position.ndim > 2:
                        single_enlarged_nbrs = util.neighbor_allocate(
                            neighbor_fn, util.tree_get_single(last_state))
                        enlarged_nbrs = vmap(util.neighbor_update, (None, 0))(
                            single_enlarged_nbrs, last_state)
                    else:
                        enlarged_nbrs = util.neighbor_allocate(
                            neighbor_fn, last_state)
                    reset_traj_state = traj_state.replace(
                        sim_state=(last_state, enlarged_nbrs))
                    traj_state = recompute_trajectory(
                        (params, reset_traj_state))
                    reset_counter += 1
                else:
                    if multiple_arguments:
                        return traj_state, *returns
                    else:
                        return traj_state

            if jnp.any(jnp.isnan(last_state.position)):
                raise RuntimeError(
                    'Last state is NaN. Currently there is no recovering from '
                    'this. Restart from the last non-overflown state might '
                    'help, but comes at the cost that the reference state is '
                    'likely not representative.')

            raise RuntimeError('Multiple neighbor list re-computations did '
                               'not yield a trajectory without overflow. '
                               'Consider increasing the neighbor list '
                               'capacity multiplier.')
        return wrapped


    def propagate(params, old_traj_state):
        """Wrapper around jitted propagation function that ensures that
        if neighbor list buffer overflowed, the trajectory is recomputed and
        the neighbor list size is increased until valid trajectory was obtained.
        Due to the recomputation of the neighbor list, this function cannot be
        jit.
        """
        new_traj_state = propagation_fn(params, old_traj_state)
        return new_traj_state

    safe_propagation_fn = safe_propagate(propagate, multiple_arguments=False)
    def init_first_traj(key, params, reference_state):
        """Initializes initial trajectory to start optimization from.

        We dump the initial trajectory for equilibration, as initial
        equilibration usually takes much longer than equilibration time
        of each trajectory. If this is still not sufficient, the simulation
        should equilibrate over the course of subsequent updates.
        """
        print(f"Before")
        print(tree_util.tree_map(jnp.shape, reference_state[0]))

        if reference_state[0].position.ndim == 2 and num_chains is not None:
            reference_state = tree_util.tree_map(
                lambda x: jnp.tile(x, [num_chains] + [1] * x.ndim), reference_state
            )

        print(f"After")
        print(tree_util.tree_map(jnp.shape, reference_state[0]))

        dump_traj = trajectory_generator(params, reference_state,
                                         kT=replica_kbt, pressure=ref_press)

        t_start = time.time()
        init_traj = trajectory_generator(params, dump_traj.sim_state,
                                         kT=replica_kbt, pressure=ref_press)
        init_traj = init_traj.replace(key=key)
        runtime = (time.time() - t_start) / 60.  # in mins


        init_traj = safe_propagation_fn(params, init_traj)

        # Use the initial trajectory as a reference for entropy and free energy
        init_traj = init_traj.replace(
            entropy_diff=0.0,
            free_energy_diff=0.0,
            energy_params=params,
            key=key
        )

        print(f"Trajectory state has key {init_traj.key}")

        return init_traj, runtime

    if safe_propagation:
        return init_first_traj, compute_weights, safe_propagation_fn
    else:
        warnings.warn(
            'Propagation function is not safe by default. '
            'Do not forget to use the wrapper around the compute function to '
            'ensure that the neighborlist does not overflow.')
        return init_first_traj, compute_weights, propagate, safe_propagate


def init_default_loss_fn(targets: TargetDict, l_H = 0.0, l_RE = None):
    """Initializes the default loss function, where MSE errors of
    destinct quantities are added.

    First, observables are computed via the reweighting scheme.
    These observables can be ndarray valued, e.g. vectors for RDF
    / ADF or matrices for stress. For each observable, the element-wise
    MSE error is computed wrt. the target provided in
    "quantities[quantity_key]['target']". This per-quantity loss
    is multiplied by gamma in "quantities[quantity_key]['gamma']".
    The final loss is then the sum over all of these weighted
    per-quantity MSE losses. This function allows both observables that
    are simply ensemble averages of instantaneously fluctuating quantities
    and observables that are more complex functions of one or more quantity
    trajectories. The function computing the observable from trajectories of
    instantaneous fluctuating quantities needs to be provided via in
    "quantities[quantity_key]['traj_fn']". For the simple, but common case of
    an average of a single quantity trajectory, 'traj_fn' is given by
    traj_quantity.init_traj_mean_fn.

    Alternatively, a custom loss_fn can be defined. The custom
    loss_fn needs to have the same input-output signuture as the loss_fn
    implemented here.

    Args:
        targets: The target dict with 'gamma', 'target' and 'traj_fn'
            for each observable defined in 'quantities'.
        l_H: Coefficient of the maximum entropy penalty
        l_RE: Interpolation coefficient between relative entropy and difftre
            loss.

    Returns:
        The loss_fn taking trajectories of fluctuating properties,
        computing ensemble averages via the reweighting scheme and
        outputs the loss and predicted observables.
    """
    # Interpolate between the default loss and the relative entropy loss
    if l_RE is None:
        alpha = 1.0
        beta = 1.0
    else:
        alpha = l_RE
        beta = 1.0 - l_RE

    def loss_fn(quantity_trajs, weights):
        loss = 0.
        predictions = {
            key: target['traj_fn'](quantity_trajs, weights=weights)
            for key, target in targets.items()
        }

        # Maximize the entropy (minimize the negative entropy)
        loss -= l_H * quantity_trajs.get('entropy', 0.0)
        loss -= alpha * predictions.get('relative_entropy', 0.0) * targets.pop('relative_entropy', {}).get('gamma', 0.0)

        # MSE loss for the remaining targets
        for target_key, target in targets.items():
            loss += beta * target['gamma'] * max_likelihood.mse_loss(
                predictions[target_key], target['target'])

        return loss, predictions
    return loss_fn


def init_rel_entropy_gradient(energy_fn_template, compute_weights, kbt,
                              vmap_batch_size=10):
    """Initializes a function that computes the relative entropy gradient.

    The computation of the gradient is batched to increase computational
    efficiency.

    Args:
        energy_fn_template: Energy function template
        compute_weights: compute_weights function as initialized from
                         init_pot_reweight_propagation_fns.
        kbt: KbT
        vmap_batch_size: Batch size for

    Returns:
        A function rel_entropy_gradient(params, traj_state, reference_batch),
        which returns the relative entropy gradient of 'params' given a
        generated trajectory saved in 'traj_state' and a reference trajectory
        'reference_batch'.
    """
    beta = 1 / kbt

    @jit
    def rel_entropy_gradient(params, traj_state, reference_batch):
        if traj_state.sim_state[0].position.ndim > 2:
            nbrs_init = util.tree_get_single(traj_state.sim_state[1])
        else:
            nbrs_init = traj_state.sim_state[1]

        def energy(params, position):
            energy_fn = energy_fn_template(params)
            # Note: nbrs update requires constant box, i.e. not yet
            # applicable to npt ensemble
            nbrs = nbrs_init.update(position)
            return energy_fn(position, neighbor=nbrs)

        def weighted_gradient(map_input):
            position, weight = map_input
            snapshot_grad = grad(energy)(params, position)  # dudtheta
            weight_gradient = lambda new_grad: weight * new_grad
            weighted_grad_snapshot = tree_util.tree_map(weight_gradient,
                                                        snapshot_grad)
            return weighted_grad_snapshot

        def add_gradient(map_input):
            batch_gradient = vmap(weighted_gradient)(map_input)
            return util.tree_sum(batch_gradient, axis=0)

        weights, _ = compute_weights(params, traj_state)

        # reshape for batched computations
        batch_weights = weights.reshape((-1, vmap_batch_size))
        traj_shape = traj_state.trajectory.position.shape
        batchwise_gen_traj = traj_state.trajectory.position.reshape(
            (-1, vmap_batch_size, traj_shape[-2], traj_shape[-1]))
        ref_shape = reference_batch.shape
        reference_batches = reference_batch.reshape(
            (-1, vmap_batch_size, ref_shape[-2], ref_shape[-1]))

        # no reweighting for reference data: weights = 1 / N
        ref_weights = jnp.ones(reference_batches.shape[:2]) / (ref_shape[0])

        ref_grad = lax.map(add_gradient, (reference_batches, ref_weights))
        mean_ref_grad = util.tree_sum(ref_grad, axis=0)
        gen_traj_grad = lax.map(add_gradient, (batchwise_gen_traj,
                                               batch_weights))
        mean_gen_grad = util.tree_sum(gen_traj_grad, axis=0)

        combine_grads = lambda x, y: beta * (x - y)
        dtheta = tree_util.tree_map(combine_grads, mean_ref_grad, mean_gen_grad)
        return dtheta
    return rel_entropy_gradient


def init_bar(energy_fn_template: EnergyFnTemplate,
             ref_kbt: ArrayLike,
             energy_batch_size: int = 10,
             iter_bisection: int = 25
             ) -> Callable[[TrajectoryState, TrajectoryState], Tuple[ArrayLike, ArrayLike]]:
    """Initializes the free energy and entropy computation via the BAR approach.

    The algorithm [#wyczalkowski2010]_ uses the BAR method to derive
    the free energy difference between two trajectories and additionally
    derives the entropy difference via the thermodynamic relation
    :math:`TdS = dU - dF`.

    This implementation relies on the bisection method to solve the implicit
    equation

    .. math ::
        \\Delta F:\ \\left\\langle\\frac{1}{1 + \\exp(\\beta\\Delta U - \\beta\\Delta F)}\\right\\rangle_0 - \\left\\langle\\frac{1}{1 + \\exp(-\\beta\\Delta U + \\beta\\Delta F)}\\right\\rangle_1 = 0.

    Args:
        energy_fn_template: Function that returns a potential model when
            called with a set of energy parameters.
        ref_kbt: Reference temperature.
        energy_batch_size: Batch size of the vectorized potential energy
            computation.
        iter_bisection: Iterations of the bisection method.

    Returns:
        Returns the new_traj with updated free energy and entropy difference.
        These differences are updated by the differences between the old
        and the new trajectory, such that these values resemble the
        differences to the first trajectory that has been generated.

    References:
         .. [#wyczalkowski2010] New Estimators for Calculating Solvation Entropy
            and Enthalpy and Comparative Assessments of Their Accuracy and
            Precision. Matthew A. Wyczalkowski, Andreas Vitalis, and Rohit V.
            Pappu in The Journal of Physical Chemistry B 2010 114 (24),
            8166-8180, DOI: 10.1021/jp103050u

     """

    traj_energy_fn = custom_quantity.energy_wrapper(energy_fn_template)
    reweighting_quantities = {'energy': traj_energy_fn}

    # Helper functions to calculate the free energy and entropy difference via
    # the iterative bar approach

    beta = 1. / ref_kbt

    def _vmap_potential_energy_differences(old_traj, new_traj):
        """Performs the reweighting procedure vectorized."""
        @jax_sgmc.util.list_vmap
        def _inner(pair):
            vmap_batch_size = max([1, energy_batch_size // 2])
            traj_state, params = pair
            reweighting_properties = traj_util.quantity_traj(
                traj_state, reweighting_quantities, params, vmap_batch_size)
            return reweighting_properties
        # The BAR method requires the energy difference between the potential
        # for both the perturbed and unperturbed trajectories. We thus have to
        # compute the potential energy of the new potential model on the old
        # trajectory and vice versa.
        return _inner(
            (old_traj, new_traj.energy_params),
            (new_traj, old_traj.energy_params))

    def _fr_free_energy(dV_p, dV_0, df):
        """Returns the forward and reverse estimators for the BAR equation."""
        exponent_p = beta * (dV_p - df)
        exponent_0 = beta * (-dV_0 + df)
        gf = 1.0 / (1 + jnp.exp(exponent_p))
        gr = 1.0 / (1 + jnp.exp(exponent_0))
        return gf, gr

    def _bar_residual(df, dV_p, dV_0):
        """Squared residual of the implicit BAR equation. """
        gf, gr = _fr_free_energy(dV_p, dV_0, df)
        # debug.print("[Solve] df = {df} with {gf} and {gr}", df=df, gf=jnp.mean(gf), gr=jnp.mean(gr))
        sum_gf = jax_md.util.high_precision_sum(gf)
        sum_gr = jax_md.util.high_precision_sum(gr)
        return sum_gf - sum_gr

    def _entropy_equation(df, V_0, V_p, rV_p, rV_0):
        """Computes the entropy difference from both trajectories."""
        dV_p = V_p - rV_0
        dV_0 = rV_p - V_0

        # Forward and reverse estimators once for the perturbed ensemble average
        # and once for the reference ensemble average
        gf_p, gr_p = _fr_free_energy(dV_p, dV_p, df)
        gf_0, gr_0 = _fr_free_energy(dV_0, dV_0, df)

        alpha_0 = jnp.mean(gf_0 * V_0) - jnp.mean(gf_0) * jnp.mean(V_0)
        alpha_0 += jnp.mean(gf_0 * gr_0 * dV_0)

        alpha_p = jnp.mean(gr_p * V_p) - jnp.mean(gr_p) * jnp.mean(V_p)
        alpha_p -= jnp.mean(gf_p * gr_p * dV_p)

        du = alpha_0 - alpha_p
        du /= jnp.mean(gf_0 * gr_0) + jnp.mean(gf_p * gr_p) + 1.e-30

        return constants.kb * beta * (du - df)

    def _init_bisection(dV_p, dV_0):
        # Helper function to find valid initial points by extending the search
        # interval if necessary
        def _expand_interval(state, _):
            # Expand the interval if both initial points have a residual with equal
            # sign

            df_p, df_0 = state
            loss_p = _bar_residual(df_p, dV_p, dV_0)
            loss_0 = _bar_residual(df_0, dV_p, dV_0)

            # Extend the search interval by a factor of four if solution is
            # not contained in the interval. Ensure that the interval is
            # increased even if the proposals are equal up to a constant
            extend = 1.5 * jnp.abs(df_p - df_0) + 0.5e-4 * jnp.abs(df_p + df_0) + 1e-8
            extend *= jnp.sign(df_p - df_0)
            extend_interval = jnp.where(
                jnp.sign(loss_p) == jnp.sign(loss_0),
                extend, 0.0)
            df_p += extend_interval
            df_0 -= extend_interval

            # debug.print("[BAR INIT] Residuals are {r_p} and {r_0} -> New search interval: [{fp}, {fo}]", r_p=loss_p, r_0=loss_0, fp=df_p, fo=df_0)

            return (df_p, df_0), None

        # Initialize the guesses
        exponent_p = -beta * dV_p
        exponent_0 = -beta * dV_0

        exp_p = jnp.exp(exponent_p - jnp.max(exponent_p))
        exp_0 = jnp.exp(exponent_0 - jnp.max(exponent_0))

        df_p = jnp.log(jax_md.util.high_precision_sum(exp_p))
        df_0 = jnp.log(jax_md.util.high_precision_sum(exp_0))

        df_p += jnp.max(exponent_p) - jnp.log(exponent_p.size)
        df_0 += jnp.max(exponent_0) - jnp.log(exponent_0.size)

        df_p *= -1 / beta
        df_0 *= -1 / beta

        # debug.print("[BAR INIT] Initialize a = {df_a} and b = {df_b}", df_a = df_p, df_b = df_0)

        (df_p, df_0), _ = lax.scan(_expand_interval, (df_p, df_0), onp.arange(10))
        res_a = _bar_residual(df_p, dV_p, dV_0)
        res_b = _bar_residual(df_0, dV_p, dV_0)

        def _bisection_step(state, _):
            df_a, df_b = state
            df_c = 0.5 * (df_a + df_b)
            loss_a = _bar_residual(df_a, dV_p, dV_0)
            loss_c = _bar_residual(df_c, dV_p, dV_0)

            # debug.print("[BAR : {idx}] Residual {residual} in [{a}, {b}] for df = {df} in [{fa}, {fb}]", idx=idx, residual=loss_c, df=df_c, a=loss_a, b=loss_b, fa=df_a, fb=df_b)

            # Keep the point A or B that is on the other side of the zero line
            # from point C,
            # i.e. check if the loss for A and C have the same sign.
            # If this is the case, search in [C, B], otherwise in [A, C].
            new_a = jnp.where(jnp.sign(loss_a) == jnp.sign(loss_c), df_c, df_a)
            new_b = jnp.where(jnp.sign(loss_a) == jnp.sign(loss_c), df_b, df_c)

            # Check if c is already the solution (up to the available precision)
            new_a = jnp.where(loss_c == 0.0, df_c, new_a)
            new_b = jnp.where(loss_c == 0.0, df_c, new_b)

            return (new_a, new_b), loss_c

        return (df_p, df_0), _bisection_step

    def bennett_free_energy(old_traj: TrajectoryState,
                            new_traj: TrajectoryState):
        """Compute the free energy and entropy difference.

         The algorithm from [#wyczalkowski2010] uses the BAR method to derive
         the free energy difference between two trajectories and additionally
         derives the entropy difference via the thermodynamic relation
         $TdS = dU - dF$. This implementation relies on the bisection method to
         solve the implicit equation.

         Args:
             old_traj: trajectory generated with the unperturbed potential
             new_traj: trajectory generated with the perturbed potential

        Returns:
            Returns the new_traj with updated free energy and entropy difference.
            These differences are updated by the differences between the old
            and the new trajectory, such that these values resemble the
            differences to the first trajectory that has been generated.

         .. [#wyczalkowski2010] New Estimators for Calculating Solvation Entropy and Enthalpy and Comparative Assessments of Their Accuracy and Precision. Matthew A. Wyczalkowski, Andreas Vitalis, and Rohit V. Pappu in The Journal of Physical Chemistry B 2010 114 (24), 8166-8180, DOI: 10.1021/jp103050u

         """

        # Calculate the differences in potential energy for both trajectories
        rew_0, rew_p = _vmap_potential_energy_differences(old_traj, new_traj)

        # Get the potential energy from both trajectory
        V_p = new_traj.aux['energy']
        V_0 = old_traj.aux['energy']
        rV_p = rew_0['energy']
        rV_0 = rew_p['energy']

        # dV_p is the energy difference between the perturbed and unperturbed
        # potential based on the perturbed ensemble, dV_0 is the same
        # difference but on the unperturbed ensemble.

        dV_p = V_p - rV_0
        dV_0 = rV_p - V_0

        init_f, update_f = _init_bisection(dV_p, dV_0)
        (dfe, df2), _ = lax.scan(update_f, init_f, onp.arange(iter_bisection))

        res_a = _bar_residual(dfe, dV_p, dV_0)
        res_b = _bar_residual(df2, dV_p, dV_0)

        ds = _entropy_equation(dfe, V_0, V_p, rV_p, rV_0)

        return dfe, ds

    return bennett_free_energy


class PropagationBase(max_likelihood.MLETrainerTemplate):
    """Trainer base class for shared functionality whenever (multiple)
    simulations are run during training. Can be used as a template to
    build other trainers. Currently used for DiffTRe and relative entropy.

    We only save the latest generated trajectory for each state point.
    While accumulating trajectories would enable more frequent reweighting,
    this effect is likely minor as past trajectories become exponentially
    less useful with changing potential. Additionally, saving long trajectories
    for each statepoint would increase memory requirements over the course of
    the optimization.
    """
    def __init__(self, init_trainer_state, optimizer, checkpoint_path,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None,
                 full_checkpoint=True, key=None):
        super().__init__(optimizer, init_trainer_state, checkpoint_path,
                         full_checkpoint, energy_fn_template)
        self.sim_batch_size = sim_batch_size
        self.reweight_ratio = reweight_ratio

        if key is None:
            self.key = random.PRNGKey(0)

        # store for each state point corresponding traj_state and grad_fn
        # save in distinct dicts as grad_fns need to be deleted for checkpoint
        self.grad_fns, self.trajectory_states, self.statepoints = {}, {}, {}
        self.n_statepoints = 0
        self.shuffle_key = random.PRNGKey(0)

    def _init_statepoint(self, reference_state, energy_fn_template,
                         simulator_template, neighbor_fn, timings, kbt,
                         set_key=None, energy_batch_size=10,
                         initialize_traj=True, ref_press=None,
                         safe_propagation=True, entropy_approximation=False,
                         replica_kbt=None, num_chains=None):
        """Initializes the simulation and reweighting functions as well
        as the initial trajectory for a statepoint."""
        # TODO ref pressure only used in print and to have barostat values.
        #  Reevaluate this parameter of barostat values not used in reweighting
        # TODO document ref_press accordingly

        if set_key is not None:
            key = set_key
            if set_key not in self.statepoints.keys():
                self.n_statepoints += 1
        else:
            key = self.n_statepoints
            self.n_statepoints += 1
        self.statepoints[key] = {'kbT': kbt}
        npt_ensemble = util.is_npt_ensemble(reference_state[0])
        if npt_ensemble: self.statepoints[key]['pressure'] = ref_press

        if replica_kbt is not None:
            assert reference_state[0].position.ndim > 2, (
                "Replica exchange requires multiple simulator states")

        gen_init_traj, *reweight_fns = init_pot_reweight_propagation_fns(
            energy_fn_template, simulator_template, neighbor_fn, timings,
            kbt, ref_press, self.reweight_ratio, npt_ensemble,
            energy_batch_size, safe_propagation=safe_propagation,
            entropy_approximation=entropy_approximation,
            replica_kbt=replica_kbt, num_chains=num_chains
        )
        if initialize_traj:
            self.key, split = random.split(self.key)
            init_traj, runtime = gen_init_traj(
                split, self.params, reference_state)
            print(f'Time for trajectory initialization {key}: {runtime} mins')
            self.trajectory_states[key] = init_traj
        else:
            print('Not initializing the initial trajectory is only valid if '
                  'a checkpoint is loaded. In this case, please be use to add '
                  'state points in the same sequence, otherwise loaded '
                  'trajectories will not match its respective simulations.')

        return key, *reweight_fns

    @abstractmethod
    def add_statepoint(self, *args, **kwargs):
        """User interface to add additional state point to train model on."""
        raise NotImplementedError()

    @property
    def params(self):
        return self.state.params

    @params.setter
    def params(self, loaded_params):
        self.state = self.state.replace(params=loaded_params)

    def get_sim_state(self, key):
        return self.trajectory_states[key].sim_state

    def _get_batch(self):
        """Helper function to re-shuffle simulations and split into batches."""
        self.shuffle_key, used_key = random.split(self.shuffle_key, 2)
        shuffled_indices = random.permutation(used_key, self.n_statepoints)
        if self.sim_batch_size == 1:
            batch_list = jnp.split(shuffled_indices, shuffled_indices.size)
        elif self.sim_batch_size == -1:
            batch_list = jnp.split(shuffled_indices, 1)
        else:
            raise NotImplementedError('Only batch_size = 1 or -1 implemented.')

        return (batch.tolist() for batch in batch_list)

    def _print_measured_statepoint(self):
        """Print meausured kbT (and pressure for npt ensemble) for all
        statepoints to ensure the simulation is indeed carried out at the
        prescribed state point.
        """
        for sim_key, traj in self.trajectory_states.items():
            print(f'[Statepoint {sim_key}]')
            statepoint = self.statepoints[sim_key]
            measured_kbt = jnp.mean(traj.aux['kbT'])
            if 'pressure' in statepoint:  # NPT
                measured_press = jnp.mean(traj.aux['pressure'])
                press_print = (f'\n\tpress = {measured_press:.2f} ref_press = '
                               f'{statepoint["pressure"]:.2f}')
            else:
                press_print = ''
            print(f'\tkbT = {measured_kbt:.3f} ref_kbt = '
                  f'{statepoint["kbT"]:.3f}' + press_print)

    def train(self, max_epochs, thresh=None, checkpoint_freq=None):
        assert self.n_statepoints > 0, ('Add at least 1 state point via '
                                        '"add_statepoint" to start training.')
        super().train(max_epochs, thresh=thresh,
                      checkpoint_freq=checkpoint_freq)

    @abstractmethod
    def _update(self, batch):
        """Implementation of gradient computation, stepping of the optimizer
        and logging of auxiliary results. Takes batch of simulation indices
        as input.
        """
