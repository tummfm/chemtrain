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

"""A collection of functions to facilitate learning maximum likelihood /
 single point estimate models.
 """
from functools import partial

import jax
from jax import (lax, vmap, pmap, value_and_grad, tree_map, device_count,
                 numpy as jnp, device_put, jit)
from jax.sharding import Mesh, PartitionSpec, NamedSharding, SingleDeviceSharding
from jax.experimental.shard_map import shard_map
from jax_sgmc import data
import optax

from chemtrain import util


def _get_param_loss_fn(loss_fn, batched_model, penalty_fn=None):
    def params_loss_fn(params, batch):
        predictions = batched_model(params, batch)
        out = loss_fn(predictions, batch)

        # Canonicalize output
        if isinstance(out, tuple):
            loss, per_target_loss = out
        else:
            loss = out
            per_target_loss = None


        # Add a penalty if provided
        if penalty_fn is not None:
            loss += penalty_fn(params)

        return loss, per_target_loss
    return params_loss_fn


def pmap_update_fn(batched_model, loss_fn, optimizer, penalty_fn=None):
    """Initializes a pmapped function for updating parameters.

    Usage:
        .. code-block :: python

            params, opt_state, loss, grad = update_fn(params, opt_state, batch)

    Loss and grad are only a single instance, no n_device replica.
    Params and opt_state need to be N_devices times duplicated along axis 0.
    Batch is reshaped by this function.

    Args:
        batched_model: A model with signature model(params, batch), which
                       predicts a batch of outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
                 value for a batch.
        optimizer: Optax optimizer

    Returns:
        A function that computes the gradient and updates the parameters via the
        optimizer.
    """
    # loss as function of params and batch for optimization
    param_loss_fn = _get_param_loss_fn(loss_fn, batched_model, penalty_fn)

    @partial(jax.pmap, in_axes=(None, None, 0), axis_name='batch')
    def pmap_batch_update(params, opt_state, batch):
        (loss, per_target_loss), grad = value_and_grad(
            param_loss_fn, has_aux=True
        )(params, batch)

        # step optimizer within pmap to minimize communication overhead
        grad = lax.psum(grad, 'batch')
        loss = lax.psum(loss, 'batch')
        per_target_loss = lax.psum(per_target_loss, 'batch')

        new_params, opt_state = step_optimizer(params, opt_state, grad,
                                               optimizer)
        return new_params, opt_state, loss, grad, per_target_loss

    def batch_update(params, opt_state, batch, per_target_loss=False):
        batch = util.tree_pmap_split(batch, len(jax.devices()))
        out = pmap_batch_update(params, opt_state, batch)
        new_params, opt_state, loss, grad = util.tree_get_single(out)

        if per_target_loss:
            return new_params, opt_state, loss, grad, per_target_loss
        else:
            return new_params, opt_state, loss, grad
    return batch_update


def shmap_update_fn(batched_model, loss_fn, optimizer, penalty_fn=None):
    """Initializes a pmapped function for updating parameters.

    Usage:
        .. code-block :: python

            params, opt_state, loss, grad = update_fn(params, opt_state, batch)

    Loss and grad are only a single instance, no n_device replica.
    Params and opt_state need to be N_devices times duplicated along axis 0.
    Batch is reshaped by this function.

    Args:
        batched_model: A model with signature model(params, batch), which
                       predicts a batch of outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
                 value for a batch.
        optimizer: Optax optimizer

    Returns:
        A function that computes the gradient and updates the parameters via the
        optimizer.
    """
    # loss as function of params and batch for optimization.
    mesh = Mesh(jax.devices(), axis_names=('batch'))
    replicate = NamedSharding(mesh, PartitionSpec())
    split = NamedSharding(mesh, PartitionSpec('batch'))

    param_loss_fn = _get_param_loss_fn(loss_fn, batched_model, penalty_fn)

    @jit
    def batch_update(params, opt_state, data):
        @partial(shard_map, mesh=mesh, in_specs=PartitionSpec('batch'),
                 out_specs=PartitionSpec())
        def _inner(batch):
            (loss, per_target_loss), grad = value_and_grad(
                param_loss_fn, has_aux=True)(params, batch)
            # step optimizer within pmap to minimize communication overhead
            grad = lax.psum(grad, axis_name='batch')
            loss = lax.psum(loss, axis_name='batch')
            per_target_loss = lax.psum(per_target_loss, axis_name='batch')

            new_params, new_opt_state = step_optimizer(
                params, opt_state, grad, optimizer)

            return new_params, new_opt_state, loss, grad, per_target_loss
        return _inner(data)

    def update_fn(params, opt_state, batch, per_target=False):
        params = device_put(params, replicate)
        opt_state = device_put(opt_state, replicate)
        batch = device_put(batch, split)

        *outs, per_target_loss = batch_update(params, opt_state, batch)

        if per_target:
            return *outs, per_target_loss
        else:
            return outs

    return update_fn


def shmap_loss_fn(batched_model, loss_fn, penalty_fn=None):
    """Initializes a shmapped function for computing a loss.

    Usage:
        .. code-block :: python

            params, opt_state, loss, grad = update_fn(params, opt_state, batch)

    Loss and grad are only a single instance, no n_device replica.
    Params and opt_state need to be N_devices times duplicated along axis 0.
    Batch is reshaped by this function.

    Args:
        batched_model: A model with signature model(params, batch), which
                       predicts a batch of outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
                 value for a batch.
        optimizer: Optax optimizer

    Returns:
        A function that computes the gradient and updates the parameters via the
        optimizer.
    """
    # loss as function of params and batch for optimization.
    mesh = Mesh(jax.devices(), axis_names=('batch'))
    replicate = NamedSharding(mesh, PartitionSpec())
    split = NamedSharding(mesh, PartitionSpec('batch'))

    param_loss_fn = _get_param_loss_fn(loss_fn, batched_model, penalty_fn)

    @jit
    def batch_update(params, data):
        @partial(shard_map, mesh=mesh, in_specs=PartitionSpec('batch'),
                 out_specs=PartitionSpec())
        def _inner(batch):
            loss, per_target_loss = param_loss_fn(params, batch)

            loss = lax.psum(loss, axis_name='batch')
            per_target_loss = lax.psum(per_target_loss, axis_name='batch')

            return loss, per_target_loss
        return _inner(data)

    def update_fn(params, batch, per_target=False):
        params = device_put(params, replicate)
        batch = device_put(batch, split)

        *outs, per_target_loss = batch_update(params, batch)

        if per_target:
            return *outs, per_target_loss
        else:
            return outs

    return update_fn


def init_val_predictions(batched_model, val_loader, batch_size=1,
                         batch_cache=10):
    """Model predictions for whole validation/test dataset.

    Usage:
        .. code-block :: python

            predictions, data_state = mapped_model_fn(params, data_state)

    Params needs to be N_devices times duplicated along axis 0.

    Args:
        batched_model: A model with signature model(params, batch), which
                       predicts a batch of outputs used in loss function.
        val_loader: Validation or test set NumpyDataLoader.
        batch_size: Total batch size that is processed in parallel
        batch_cache: Number of batches to cache.

    Returns:
        Tuple (predictions, data_state). predictions contains model predictions
        for the whole validation dataset and data_state is used to start the
        data loading in the next evaluation.
    """
    # case where validation data is very small
    batch_size = min(val_loader.static_information['observation_count']
                     // device_count(), batch_size)
    map_fun, data_release = data.full_data_mapper(val_loader, batch_cache,
                                                  batch_size)

    @jax.jit
    def single_batch(params, batch, unused_state):
        return batched_model(params, batch), unused_state

    def mapped_model_fn(params):
        params = jax.device_put(params, SingleDeviceSharding(jax.devices()[0]))
        predictions, _ = map_fun(partial(single_batch, params), None)
        return predictions
    return mapped_model_fn, data_release


def init_val_loss_fn(model, loss_fn, val_loader, val_targets_keys=None,
                     batch_size=1, batch_cache=100):
    """Initializes a pmapped loss function that computes the validation loss.

    Usage:
        .. code-block :: python

            val_loss, data_state = batched_loss_fn(params, data_state)

    Params needs to be N_devices times duplicated along axis 0.

    Args:
        model: A model with signature model(params, batch), which predicts
               outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
                 value for a batch.
        val_loader: NumpyDataLoader for validation set.
        val_targets_keys: Dict containing targets of whole val
        batch_size: Total batch size that is processed in parallel.
        batch_cache: Number of batches to cache on GPU to reduce host-device
                     communication.

    Returns:
        A pmapped function that returns the average validation loss.
    """

    # We compute the validation error over the whole dataset at once, because
    # otherwise it is non-trivial to compute the correct error for masked
    # batches with different number of masked targets without explicitly knowing
    # the mask in this function
    # If predictions and targets of the whole validation dataset does not fit
    # memory, a more specialized approach needs to be taken.

    if val_targets_keys is None:
        target_data = val_loader.reference_data
    else:
        target_data = {key: val_loader.reference_data[key]
                       for key in val_targets_keys}

    mapped_predictions_fn, data_release_fn = init_val_predictions(
        model, val_loader, batch_size, batch_cache)

    def mapped_loss_fn(params):
        predictions = mapped_predictions_fn(params)
        val_loss = loss_fn(predictions, target_data)
        return val_loss

    return mapped_loss_fn, data_release_fn


def _masked_loss(per_element_loss, mask=None):
    """Computes average loss, accounting for masked elements, if applicable."""
    if mask is None:
        return jnp.mean(per_element_loss)
    else:
        assert mask.shape == per_element_loss.shape, ('Mask requires same shape'
                                                      ' as targets.')
        return jnp.sum(per_element_loss * mask) / jnp.sum(mask)


def mse_loss(predictions, targets, mask=None):
    """Computes mean squared error loss for given predictions and targets.

    Args:
        predictions: Array of predictions
        targets: Array of respective targets. Needs to have same shape as
                 predictions.
        mask: Mask contribution of some array elements. Needs to have same shape
              as predictions. Default None applies no mask.

    Returns:
        Mean squared error loss value.
    """
    squared_differences = jnp.square(targets - predictions)
    return _masked_loss(squared_differences, mask)


def mae_loss(predictions, targets, mask=None):
    """Computes the mean absolute error for given predictions and targets.

    Args:
        predictions: Array of predictions
        targets: Array of respective targets. Needs to have same shape as
                 predictions.
        mask: Mask contribution of some array elements. Needs to have same shape
              as predictions. Default None applies no mask.

    Returns:
        Mean absolute error value.
    """
    abs_err = jnp.abs(targets - predictions)
    return _masked_loss(abs_err, mask)


def identity_loss(predictions, *args, **kwargs):
    """Considers the prediction itself as loss value.

    For example, the relative entropy can be used directly as loss in DiffTRe.

    Args:
        predictions: Array of predictions (scalar)

    Returns:
        Returns the prediction itself as loss value.

    """
    del args, kwargs
    return predictions


def step_optimizer(params, opt_state, grad, optimizer):
    """Steps optimizer and updates state using the gradient."""
    scaled_grad, new_opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, scaled_grad)
    return new_params, new_opt_state
