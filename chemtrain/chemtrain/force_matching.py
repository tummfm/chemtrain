"""Functions for learning via direct matching of per-snapshot quantities,
such as energy, forces and virial pressure.
"""
from collections import namedtuple
from functools import partial

from coax.utils._jit import jit
from jax import vmap, lax, value_and_grad, pmap, numpy as jnp
from jax_sgmc import data
import numpy as onp

from chemtrain import util
from chemtrain.jax_md_mod import custom_quantity

# Note:
#  Computing the neighborlist in each snapshot is not efficient for DimeNet++,
#  which constructs a sparse graph representation afterwards. However, other
#  models such as the tabulated potential are inefficient if used without
#  neighbor list as many cut-off interactions are otherwise computed.
#  For the sake of a simpler implementation, the slight inefficiency
#  in the case of DimeNet++ is accepted for now.
#  A more efficient implementation is based on pre-computation of neighborlists
#  for each snapshot in the dataset.

State = namedtuple(
    'State',
    ['position']
)
State.__doc__ = """Emulates structure of simulation state for
compatibility with quantity functions.

position: atomic positions
"""


def init_dataloaders(position_data, energy_data=None, force_data=None,
                     virial_data=None, train_ratio=0.875):
    train_set_size = position_data.shape[0]
    train_size = int(train_set_size * train_ratio)

    # pylint: disable=unbalanced-tuple-unpacking
    r_train, r_val = onp.split(position_data, [train_size])
    train_dict = {'R': r_train}
    val_dict = {'R': r_val}
    if energy_data is not None:
        u_train, u_val = onp.split(energy_data, [train_size])
        train_dict['U'] = u_train
        val_dict['U'] = u_val
    if force_data is not None:
        f_train, f_val = onp.split(force_data, [train_size])
        train_dict['F'] = f_train
        val_dict['F'] = f_val
    if virial_data is not None:
        p_train, p_val = onp.split(virial_data, [train_size])
        train_dict['p'] = p_train
        val_dict['p'] = p_val

    train_loader = data.NumpyDataLoader(**train_dict)
    val_loader = data.NumpyDataLoader(**val_dict)
    return train_loader, val_loader


def init_virial_fn(virial_data, energy_fn_template, box_tensor):
    """Initializes the correct virial function depending on the target data
    type.
    """
    if virial_data is not None:
        if virial_data.ndim == 3:
            virial_fn = custom_quantity.init_virial_stress_tensor(
                energy_fn_template, box_tensor, include_kinetic=False)
        elif virial_data.ndim in [1, 2]:
            virial_fn = custom_quantity.init_pressure(
                energy_fn_template, box_tensor, include_kinetic=False)
        else:
            raise ValueError('Format of virial dataset incompatible.')
        assert box_tensor is not None, ('If the virial is to be matched, '
                                        'box_tensor is a mandatory input.')
    else:
        virial_fn = None

    return virial_fn


def init_single_prediction(nbrs_init, energy_fn_template, virial_fn=None):
    """Initialize predictions for a single snapshot. Can be used to
    parametrize potentials from per-snapshot energy, force and/or virial.
    """
    def single_prediction(params, positions):
        energy_fn = energy_fn_template(params)
        # TODO check for neighborlist overflow and hand through
        nbrs = nbrs_init.update(positions)
        energy, negative_forces = value_and_grad(energy_fn)(positions,
                                                            neighbor=nbrs)
        predictions = {'U': energy, 'F': -negative_forces}
        if virial_fn is not None:
            predictions['virial'] = virial_fn(State(positions), nbrs, params)
        return predictions
    return single_prediction


def init_update_fns(energy_fn_template, nbrs_init, optimizer, gamma_f=1.,
                    gamma_p=1.e-6, virial_fn=None):
    """Initializes update functions for energy and/or force matching.

    The returned functions are jit and can therefore not be pickled.
    """

    single_prediction = init_single_prediction(nbrs_init, energy_fn_template,
                                               virial_fn)

    def loss_fn(params, batch):
        predictions = vmap(single_prediction, in_axes=(None, 0))(params,
                                                                 batch['R'])
        loss = 0.
        if 'U' in batch.keys():  # energy loss component
            loss += util.mse_loss(predictions['U'], batch['U'])
        if 'F' in batch.keys():  # forces loss component
            loss += gamma_f * util.mse_loss(predictions['F'], batch['F'])
        if 'p' in batch.keys():  # virial loss component
            loss += gamma_p * util.mse_loss(predictions['virial'], batch['p'])
        return loss

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


def init_mae_fn(val_loader, nbrs_init, energy_fn_template, batch_size=1,
                batch_cache=1, virial_fn=None):
    """Returns a function that computes for each observable - energy, forces and
    virial (if applicable) - the individual mean absolute error on the
    validation set. These metrics are usually better interpretable than a
    (combined) MSE loss value.
    """
    single_prediction = init_single_prediction(nbrs_init, energy_fn_template,
                                               virial_fn)

    init_fun, map_fun = data.full_reference_data(val_loader, batch_cache,
                                                 batch_size)
    init_data_state = init_fun()

    def abs_error(params, batch, mask, unused_scan_carry):
        # batch = util.tree_split(batch, n_devices)  # TODO enable pmap
        predictions = vmap(single_prediction, in_axes=(None, 0))(params,
                                                                 batch['R'])
        maes = {}
        if 'U' in batch.keys():  # energy loss component
            maes['energy'] = util.mae_loss(predictions['U'], batch['U'], mask)
        if 'F' in batch.keys():  # forces loss component
            maes['forces'] = util.mae_loss(predictions['F'], batch['F'], mask)
        if 'p' in batch.keys():  # virial loss component
            maes['virial'] = util.mae_loss(predictions['virial'], batch['p'],
                                           mask)
        return maes, unused_scan_carry

    @jit
    def mean_abs_error(params, data_state):
        data_state, (batch_maes, _) = map_fun(partial(abs_error, params),
                                              data_state, None, masking=True)
        average_maes = {key: jnp.mean(values)
                        for key, values in batch_maes.items()}
        return average_maes, data_state

    return mean_abs_error, init_data_state

