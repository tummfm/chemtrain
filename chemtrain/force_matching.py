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
    return dataset, _dataset_target_keys(dataset)


def _dataset_target_keys(dataset):
    """Dataset keys excluding particle positions for validation loss with
    possibly masked atoms.
    """
    target_key_list = list(dataset)
    target_key_list.remove('R')
    assert target_key_list, 'At least one target quantity needs to be supplied.'
    return target_key_list


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
        A function(params, single_observation) returning a dict of predictions
        containing energy ('U'), forces('F') and if applicable virial ('p').
        The single_observation is assumed to be a dict contain particle
        positions under 'R'.
    """
    def fm_model(params, single_observation):
        positions = single_observation['R']
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
                 error_fn=max_likelihood.mse_loss, individual=False):
    """Initializes loss function for energy/force matching.

    Args:
        gamma_u: Weight for potential energy loss component
        gamma_f: Weight for force loss component
        gamma_p: Weight for virial loss component
        error_fn: Function quantifying the deviation of the model and the
                  targets. By default, a mean-squared error.
        individual: Default False initializes a loss function that returns
                    scalar loss weighted by gammas. If True, returns all
                    individual components, e.g. for testing purposes. In this
                    case, gamma values are unused.

    Returns:
        loss_fn(predictions, targets), which returns a scalar loss value for a
        batch of predictions and targets.
    """
    def loss_fn(predictions, targets, mask=None):
        errors = {}
        loss_val = 0.
        if 'U' in targets.keys():  # energy loss component
            errors['energy'] = error_fn(predictions['U'], targets['U'])
            loss_val += gamma_u * errors['energy']
        if 'F' in targets.keys():  # forces loss component
            if mask is None:  # only forces need mask, U and p are unchanged
                mask = jnp.ones_like(predictions['F'])
            errors['forces'] = error_fn(predictions['F'], targets['F'], mask)
            loss_val += gamma_f * errors['forces']
        if 'p' in targets.keys():  # virial loss component
            errors['pressure'] = error_fn(predictions['p'], targets['p'])
            loss_val += gamma_p * errors['pressure']

        if individual:
            return errors
        else:
            return loss_val

    return loss_fn


def init_mae_fn(val_loader, nbrs_init, energy_fn_template, batch_size=1,
                batch_cache=1, virial_fn=None):
    """Returns a function that computes for each observable - energy, forces and
    virial (if applicable) - the individual mean absolute error on the
    validation set. These metrics are usually better interpretable than a
    (combined) MSE loss value.
    """
    model = init_model(nbrs_init, energy_fn_template, virial_fn)
    batched_model = vmap(model, in_axes=(None, 0))

    abs_error = init_loss_fn(error_fn=max_likelihood.mae_loss, individual=True)

    target_keys = _dataset_target_keys(val_loader._reference_data)
    mean_abs_error, data_release_fn = max_likelihood.init_val_loss_fn(
        batched_model, abs_error, val_loader, target_keys, batch_size,
        batch_cache)

    return mean_abs_error, data_release_fn
