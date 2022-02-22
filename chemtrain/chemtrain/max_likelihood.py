"""A collection of functions to facilitate learning maximum likelihood /
 single point estimate models.
 """
from functools import partial

from jax import lax, pmap, value_and_grad

from chemtrain import util


def pmap_update_fn(loss_fn, optimizer):
    """Initializes a pmapped function for updating parameters.

    Usage:
    params, opt_state, loss, grad = update_fn(params, opt_state, batch)

    For pmap, batch needs to be reshaped to (N_devices, batch_per_device, X) and
    params needs to be N_devices times duplicated along axis 0.


    Args:
        loss_fn: Loss function to minimize that takes (params, batch) as input.
        optimizer: Optax optimizer

    Returns:
        A function that computes the gradient and updates the parameters via the
        optimizer.
    """
    @partial(pmap, axis_name='devices')
    def batch_update(params, opt_state, batch):
        loss, grad = value_and_grad(loss_fn)(params, batch)

        # step optimizer within pmap to minimize communication overhead
        grad = lax.pmean(grad, axis_name='devices')
        loss = lax.pmean(loss, axis_name='devices')
        new_params, opt_state = util.step_optimizer(params, opt_state,
                                                    grad, optimizer)
        return new_params, opt_state, loss, grad
    return batch_update


def pmap_loss_fn(loss_fn):
    """Initializes a pmapped loss function.

    The loss function can be used for evaluating the validation loss.
    In priniple, any function with the same signature as loss_fn can be used.

    Usage:
    val_loss = batched_loss_fn(params, val_batch)

    For pmap, batch needs to be reshaped to (N_devices, batch_per_device, X) and
    params needs to be N_devices times duplicated along axis 0.

    Args:
        loss_fn: Loss function that takes (params, batch) as input.

    Returns:
        A pmapped function that returns the loss value.
    """
    @partial(pmap, axis_name='devices')
    def batched_loss_fn(params, batch):
        loss = loss_fn(params, batch)
        loss = lax.pmean(loss, axis_name='devices')
        return loss
    return batched_loss_fn
