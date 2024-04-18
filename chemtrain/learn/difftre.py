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

"""Utility functions for active learning applications."""

import functools

import jax
from jax import jit, grad, tree_util, vmap, numpy as jnp, lax

from chemtrain import util
from chemtrain.learn import max_likelihood
from chemtrain.typing import TargetDict
from chemtrain.jax_md_mod import custom_quantity
from chemtrain.trajectory import reweighting, traj_util


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


def init_difftre_gradient_and_propagation(reweight_fns, loss_fn, quantities, energy_fn_template):
    """Initializes the function to compute gradients of ensemble averages via DiffTRe."""

    weights_fn, propagate_fn, safe_propagate = reweight_fns

    quantities['energy'] = custom_quantity.energy_wrapper(
        energy_fn_template)
    reweighting.checkpoint_quantities(quantities)

    def difftre_loss(params, traj_state):
        """Computes the loss using the DiffTRe formalism and
        additionally returns predictions of the current model.
        """
        weights, _, entropy, free_energy = weights_fn(
            params, traj_state, entropy_and_free_energy=True)

        quantity_trajs = traj_util.quantity_traj(
            traj_state, quantities, params)
        quantity_trajs.update(entropy=entropy, free_energy=free_energy)

        loss, predictions = loss_fn(quantity_trajs, weights)

        # Always save free energy and entropy even if they are not part of
        # the loss.
        predictions.update(entropy=entropy, free_energy=free_energy)

        return loss, predictions

    loss_grad_fn = jax.value_and_grad(difftre_loss, has_aux=True, argnums=0)

    # TODO: There is more opportunity to make this general.
    #       We could extend the propagation and gradient function to take
    #       additional args besides the first two, e.g., batch state and
    #       output additional args besides the two mandatory traj state and
    #       loss grad

    @safe_propagate
    @functools.partial(jit, donate_argnums=1)
    def difftre_grad_and_propagation(params, traj_state):
        """The main DiffTRe function that recomputes trajectories
        when needed and computes gradients of the loss wrt. energy function
        parameters for a single state point.
        """
        traj_state = propagate_fn(params, traj_state)
        (loss_val, predictions), loss_grad = loss_grad_fn(params, traj_state)
        return traj_state, loss_val, loss_grad, predictions


    return difftre_grad_and_propagation


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
