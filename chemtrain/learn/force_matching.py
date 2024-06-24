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

"""Functions for direct learning of per-snapshot quantities.

Directly learnable quantities are, for example, energy, forces, or virial
pressure.

"""
from typing import NamedTuple, Callable, TypedDict, Tuple, List
from typing_extensions import Required

from jax import vmap, value_and_grad, numpy as jnp

from jax_sgmc.data.numpy_loader import NumpyDataLoader

from chemtrain.learn import max_likelihood
from jax_md_mod import custom_quantity

from chemtrain.typing import EnergyFnTemplate, ArrayLike, NeighborList, ErrorFn

# Note:
#  Computing the neighbor list in each snapshot is not efficient for DimeNet++,
#  which constructs a sparse graph representation afterward. However, other
#  models such as the tabulated potential are inefficient if used without
#  neighbor list as many cut-off interactions are otherwise computed.
#  For the sake of a simpler implementation, the slight inefficiency
#  in the case of DimeNet++ is accepted for now.

class State(NamedTuple):
    """Emulates state of a jax-md simulator.

    Args:
        position: Particle positions
    """
    position: ArrayLike = None


class AtomisticDataset(TypedDict, total=False):
    """Atomistic data for force-matching.

    Args:
        R: Particle positions
        U: Potential energies
        F: Forces
        p: Pressures
        kT: Temperatures
    """
    R: Required[ArrayLike]
    U: ArrayLike
    F: ArrayLike
    p: ArrayLike
    kT: ArrayLike


def build_dataset(position_data: ArrayLike,
                  energy_data: ArrayLike = None,
                  force_data: ArrayLike = None,
                  virial_data: ArrayLike = None,
                  kt_data: ArrayLike = None
                  ) -> Tuple[AtomisticDataset, List[str]]:
    """Builds the force-matching dataset depending on available data.

    Building the dataset involves separating the reference data into inputs
    to the model and target predictions of the model.
    For example, the positions of the particles act as input, while forces
    are targets.
    Additionally, this function canonicalizes the keys of the reference data.

    Example:

        For force matching, the reference data constist of particle positions
        and target forces.

        >>> from chemtrain.learn.force_matching import build_dataset
        >>> position_data = [...]
        >>> force_data = [...]

        The dataset for training is can be created via:

        >>> dataset, target_keys = build_dataset(
        ...     position_data=position_data, force_data=force_data)
        >>> print(dataset)
        {'R': [Ellipsis], 'F': [Ellipsis]}
        >>> print(target_keys)
        ['F']

    Args:
        position_data: Reference particle positions
        energy_data: Reference potential energies
        force_data: Reference forces
        virial_data: Reference virials
        kt_data: Reference temperatures

    Returns:
        Returns the canonicalized dataset and a list of keys specifying the
        trainable targets.

    """
    dataset = {'R': position_data}
    if energy_data is not None:
        dataset['U'] = energy_data
    if force_data is not None:
        dataset['F'] = force_data
    if virial_data is not None:
        dataset['p'] = virial_data
    if kt_data is not None:
        dataset['kT'] = kt_data
    return dataset, _dataset_target_keys(dataset)


def _dataset_target_keys(dataset):
    """Dataset keys excluding particle positions for validation loss with
    possibly masked atoms.
    """
    target_key_list = list(dataset)
    target_key_list.remove('R')
    if 'kT' in target_key_list:
        target_key_list.remove('kT')
    assert target_key_list, 'At least one target quantity needs to be supplied.'
    return target_key_list


def init_virial_fn(virial_data: ArrayLike,
                   energy_fn_template: EnergyFnTemplate,
                   box_tensor: ArrayLike):
    """Initializes the virial function corresponding to the reference data.

    The virial data can be scalar (e.g. for matching pressure) or a 3x3 tensor
    (e.g. for matching stresses).
    This function selects the correct computing function matching the shape of
    the reference data.

    Args:
        virial_data: Data determining the form of virial prediction
        energy_fn_template: Energy_fn_template to get energy_fn from params
        box_tensor: Box to initialize virial prediction

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
            if virial_data.ndim == 2:
                assert virial_data.shape[-1] == 1, (
                    'Scalar pressure is required. Otherwise, [3,3] tensor.')
            virial_fn = custom_quantity.init_pressure(
                energy_fn_template, box_tensor, include_kinetic=False)
        else:
            raise ValueError('Format of virial dataset incompatible.')
    else:
        virial_fn = None

    return virial_fn


def init_model(nbrs_init: NeighborList,
               energy_fn_template: EnergyFnTemplate,
               virial_fn = None):
    """Initialize prediction function for a single snapshot.

    The prediction function computed the energy, force, and virial (if provided)
    based on a single conformation and returns the results in a canonical
    format.

    Note:
        The prediction function does not check whether the neighbor list
        overflowed.

    Args:
        nbrs_init: Initial neighbor list.
        energy_fn_template: Energy_fn_template to get energy_fn from params.
        virial_fn: Function to compute virial pressure. If None, no virial
            pressure is predicted.

    Returns:
        A function(params, single_observation) returning a dict of predictions
        containing energy (``'U'``), forces(``'F'``) and, if applicable,
        virial (``'p'``).
        The single_observation is assumed to be a dict contain particle
        positions under 'R'.
    """
    def fm_model(params, single_observation):
        if 'kT' in single_observation:
            aux_kwargs = {'kT': single_observation['kT']}
        else:
            aux_kwargs = {}
        positions = single_observation['R']
        energy_fn = energy_fn_template(params)
        # TODO check for neighborlist overflow and hand through
        nbrs = nbrs_init.update(positions)
        energy, negative_forces = value_and_grad(energy_fn)(positions,
                                                            neighbor=nbrs,
                                                            **aux_kwargs)
        predictions = {'U': energy, 'F': -negative_forces}
        if virial_fn is not None:
            predictions['p'] = virial_fn(State(positions), nbrs, params,
                                         **aux_kwargs)
        return predictions
    return fm_model


def init_loss_fn(gamma_u: float = 1.,
                 gamma_f: float = 1.,
                 gamma_p: float = 1.e-6,
                 error_fn: ErrorFn = max_likelihood.mse_loss,
                 individual: bool = False):
    """Initializes loss function for energy/force matching.

    Args:
        gamma_u: Weight for potential energy loss component
        gamma_f: Weight for force loss component
        gamma_p: Weight for virial loss component
        error_fn: Function quantifying the deviation of the model and the
            targets. By default, a mean-squared error.
        individual: Return the loss values for the individual targets, e.g., for
            testing purposes. If False, the loss function returns a scalar loss
            value from the individual loss contributions, weighted by the
            ``gamma_`` coefficients.

    Returns:
        Returns a function ``loss_fn(predictions, targets)``, which returns a
        scalar loss value for a batch of predictions and targets.
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


def init_mae_fn(val_loader: NumpyDataLoader,
                nbrs_init: NeighborList,
                energy_fn_template: EnergyFnTemplate,
                batch_size: int = 1,
                batch_cache: int = 1,
                virial_fn: Callable = None):
    """Computes the Mean Absolute Error for each observable.

    The MAE for each observable are usually better interpretable than a
    (combined) MSE loss value.

    Args:
        val_loader: DataLoader with validation data
        nbrs_init: State of the neighbor list
        energy_fn_template: Energy_fn_template to get energy_fn from params
        batch_size: Batch site for batched loss computation
        batch_cache: Numbers of batches stored on device
        virial_fn: Function to compute virial pressure. If ``None``, no virial
            pressure is predicted.

    Returns:
        Returns a function that computes for each observable - energy, forces,
        and virial (if applicable) - the individual mean absolute error on the
        validation set.

    """
    model = init_model(nbrs_init, energy_fn_template, virial_fn)
    batched_model = vmap(model, in_axes=(None, 0))

    abs_error = init_loss_fn(error_fn=max_likelihood.mae_loss, individual=True)

    target_keys = _dataset_target_keys(val_loader.reference_data)
    mean_abs_error, data_release_fn = max_likelihood.init_val_loss_fn(
        batched_model, abs_error, val_loader, target_keys, batch_size,
        batch_cache)

    return mean_abs_error, data_release_fn
