"""Functions for learning via direct matching of per-snapshot quantities,
such as energy, forces and virial pressure.
"""
from collections import namedtuple
from functools import partial

from coax.utils._jit import jit
from jax import vmap, value_and_grad, numpy as jnp
from jax_sgmc import data

from chemtrain import util, max_likelihood
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


def build_dataset(position_data, energy_data=None, force_data=None,
                  virial_data=None):
    """Builds the force-matching dataset depending on available data.

    Interface of force-loss function depends on dict keys set here.
    """
    dataset = {'R': position_data}
    if energy_data is not None:
        dataset['U'] = energy_data
    if force_data is not None:
        dataset['F'] = force_data
    if virial_data is not None:
        dataset['p'] = virial_data
    return dataset


def init_virial_fn(virial_data, energy_fn_template, box_tensor):
    """Initializes the correct virial function depending on the target data
    type.
    """
    if virial_data is not None:
        assert box_tensor is not None, ('If the virial is to be matched, '
                                        'box_tensor is a mandatory input.')
        if virial_data.ndim == 3:
            virial_fn = custom_quantity.init_virial_stress_tensor(
                energy_fn_template, box_tensor, include_kinetic=False,
                pressure_tensor=True
            )
        elif virial_data.ndim in [1, 2]:
            virial_fn = custom_quantity.init_pressure(
                energy_fn_template, box_tensor, include_kinetic=False)
        else:
            raise ValueError('Format of virial dataset incompatible.')
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
            predictions['p'] = - virial_fn(State(positions), nbrs, params)
        return predictions
    return single_prediction


def init_update_fns(energy_fn_template, nbrs_init, optimizer, gamma_u=1.,
                    gamma_f=1., gamma_p=1.e-6, virial_fn=None):
    """Initializes update functions for energy and/or force matching.

    The returned functions are jit and can therefore not be pickled.

    Args:
        energy_fn_template: Energy function template
        nbrs_init: Initial neighbor list
        optimizer: Optax optimizer
        gamma_u: Weight for potential energy loss component
        gamma_f: Weight for force loss component
        gamma_p: Weight for virial loss component
        virial_fn: Function to compute virial pressure

    Returns:
        A tuple (batch_update, batched_loss_fn) of pmapped functions. The former
        computes the gradient and updates the parameters via the optimizer.
        The latter returns the loss value, e.g. for the validation set.
    """
    single_prediction = init_single_prediction(nbrs_init, energy_fn_template,
                                               virial_fn)

    def loss_fn(params, batch):
        predictions = vmap(single_prediction, in_axes=(None, 0))(params,
                                                                 batch['R'])
        loss = 0.
        if 'U' in batch.keys():  # energy loss component
            loss += gamma_u * util.mse_loss(predictions['U'], batch['U'])
        if 'F' in batch.keys():  # forces loss component
            loss += gamma_f * util.mse_loss(predictions['F'], batch['F'])
        if 'p' in batch.keys():  # virial loss component
            loss += gamma_p * util.mse_loss(predictions['p'], batch['p'])
        return loss

    batch_update = max_likelihood.pmap_update_fn(loss_fn, optimizer)
    batched_loss_fn = max_likelihood.pmap_loss_fn(loss_fn)
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
            u_mask = jnp.ones_like(predictions['U']) * mask
            maes['energy'] = util.mae_loss(predictions['U'], batch['U'], u_mask)
        if 'F' in batch.keys():  # forces loss component
            f_mask = jnp.ones_like(predictions['F']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            maes['forces'] = util.mae_loss(predictions['F'], batch['F'], f_mask)
        if 'p' in batch.keys():  # virial loss component
            p_mask = jnp.ones_like(predictions['p']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            maes['pressure'] = util.mae_loss(predictions['p'], batch['p'],
                                             p_mask)
        return maes, unused_scan_carry

    @jit
    def mean_abs_error(params, data_state):
        data_state, (batch_maes, _) = map_fun(partial(abs_error, params),
                                              data_state, None, masking=True)
        average_maes = {key: jnp.mean(values)
                        for key, values in batch_maes.items()}
        return average_maes, data_state

    return mean_abs_error, init_data_state

