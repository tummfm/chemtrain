"""Functions for learning via direct matching of per-snapshot quantities,
such as energy, forces and virial pressure.
"""
from collections import namedtuple

from jax import vmap, value_and_grad, numpy as jnp

from chemtrain import max_likelihood
from chemtrain.jax_md_mod import custom_quantity

# Note:
#  Computing the neighborlist in each snapshot is not efficient for DimeNet++,
#  which constructs a sparse graph representation afterwards. However, other
#  models such as the tabulated potential are inefficient if used without
#  neighbor list as many cut-off interactions are otherwise computed.
#  For the sake of a simpler implementation, the slight inefficiency
#  in the case of DimeNet++ is accepted for now.

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

    Interface of force-loss functions depends on dict keys set here.
    """
    dataset = {'R': position_data}
    if energy_data is not None:
        dataset['U'] = energy_data
    if force_data is not None:
        dataset['F'] = force_data
    if virial_data is not None:
        dataset['p'] = virial_data
    target_key_list = list(dataset)
    target_key_list.remove('R')
    assert target_key_list, "At least one target quantity needs to be supplied."
    return dataset, target_key_list


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


def init_model(nbrs_init, energy_fn_template, virial_fn=None):
    """Initialize predictions of energy, forces and virial (if applicable)
    for a single snapshot.

    Beware, currently overflow of neighbor list is not checked.

    Args:
        nbrs_init: Initial neighbor list.
        energy_fn_template: Energy_fn_template to get energy_fn from params.
        virial_fn: Function to compute virial pressure. If None, no virial
                   pressure is predicted.

    Returns:
        A function(params, batch) returning a dict of predictions
        containing energy ('U'), forces('F') and if applicable virial ('p').
        The batch is assumed to be a dict contain particle positions under 'R'.
    """
    def fm_model(params, batch):
        positions = batch['R']
        energy_fn = energy_fn_template(params)
        # TODO check for neighborlist overflow and hand through
        nbrs = nbrs_init.update(positions)
        energy, negative_forces = value_and_grad(energy_fn)(positions,
                                                            neighbor=nbrs)
        predictions = {'U': energy, 'F': -negative_forces}
        if virial_fn is not None:
            predictions['p'] = - virial_fn(State(positions), nbrs, params)
        return predictions
    return fm_model


def init_loss_fn(gamma_u=1., gamma_f=1., gamma_p=1.e-6,
                 error_fn=max_likelihood.mse_loss):
    """Initializes loss function for energy/force matching.

    Args:
        gamma_u: Weight for potential energy loss component
        gamma_f: Weight for force loss component
        gamma_p: Weight for virial loss component
        error_fn: Function quantifying the deviation of the model and the
                  targets. By default, a mean-squared error.

    Returns:
        loss_fn(predictions, targets), which returns a scalar loss value for a
        batch of predictions and targets.
    """
    def loss_fn(predictions, targets):
        loss = 0.
        if 'U' in targets.keys():  # energy loss component
            # TODO possibly add masks to generalize to padded species
            loss += gamma_u * error_fn(predictions['U'], targets['U'])
        if 'F' in targets.keys():  # forces loss component
            loss += gamma_f * error_fn(predictions['F'], targets['F'])
        if 'p' in targets.keys():  # virial loss component
            loss += gamma_p * error_fn(predictions['p'], targets['p'])
        return loss
    return loss_fn


def init_mae_fn(val_loader, nbrs_init, energy_fn_template, batch_size=1,
                batch_cache=1, virial_fn=None):
    """Returns a function that computes for each observable - energy, forces and
    virial (if applicable) - the individual mean absolute error on the
    validation set. These metrics are usually better interpretable than a
    (combined) MSE loss value.
    """
    # TODO refactor afterwards: Delete masks and take model input
    single_prediction = init_model(nbrs_init, energy_fn_template,
                                   virial_fn)
    # TODO refactor loss_fn such that single components can be returned.
    def abs_error(params, batch, mask):
        predictions = vmap(single_prediction, in_axes=(None, 0))(params,
                                                                 batch['R'])
        maes = {}
        if 'U' in batch.keys():  # energy loss component
            u_mask = jnp.ones_like(predictions['U']) * mask
            maes['energy'] = max_likelihood.mae_loss(predictions['U'],
                                                     batch['U'], u_mask)
        if 'F' in batch.keys():  # forces loss component
            f_mask = jnp.ones_like(predictions['F']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            maes['forces'] = max_likelihood.mae_loss(predictions['F'],
                                                     batch['F'], f_mask)
        if 'p' in batch.keys():  # virial loss component
            p_mask = jnp.ones_like(predictions['p']) * mask[:, jnp.newaxis,
                                                            jnp.newaxis]
            maes['pressure'] = max_likelihood.mae_loss(predictions['p'],
                                                       batch['p'], p_mask)
        return maes

    mean_abs_error, init_data_state = max_likelihood.val_loss_fn(
        abs_error, val_loader, batch_size, batch_cache)

    return mean_abs_error, init_data_state
