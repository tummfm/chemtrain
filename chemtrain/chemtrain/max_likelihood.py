"""A collection of functions to facilitate learning maximum likelihood /
 single point estimate models.
 """
from functools import partial

from jax import lax, pmap, value_and_grad

from chemtrain import util


def pmap_loss_fn(loss_fn, optimizer):
    """Initializes pmapped functions for updating parameters and computing loss
    values.

    Usage:
    params, opt_state, loss, grad = update_fn(params, opt_state, batch)
    val_loss = batched_loss_fn(params, val_batch)

    For pmap, batch needs to be reshaped to (N_devices, batch_per_device, X) and
    params needs to be N_devices times duplicated along axis 0.


    Args:
        loss_fn: Loss function to minimize
        optimizer: Optax optimizer

    Returns:
        A tuple (batch_update, batched_loss_fn) of pmapped functions. The former
        computes the gradient and updates the parameters via the optimizer.
        The latter returns the loss value, e.g. for the validation set.
    """
    @partial(pmap, axis_name='devices')
    def batched_loss_fn(params, batch):
        loss = loss_fn(params, batch)
        loss = lax.pmean(loss, axis_name='devices')
        return loss

    @partial(pmap, axis_name='devices')
    def batch_update(params, opt_state, batch):
        loss, grad = value_and_grad(loss_fn)(params, batch)

        # step optimizer within pmap to minimize communication overhead
        grad = lax.pmean(grad, axis_name='devices')
        loss = lax.pmean(loss, axis_name='devices')
        new_params, opt_state = util.step_optimizer(params, opt_state,
                                                    grad, optimizer)
        return new_params, opt_state, loss, grad
    return batch_update, batched_loss_fn
