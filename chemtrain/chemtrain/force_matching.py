"""Functions for learning via direct matching of per-snapshot quantities,
such as energy, forces and virial pressure.
"""
from collections import namedtuple
from functools import partial

from jax import vmap, lax, value_and_grad, pmap

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


def init_single_prediction(nbrs_init, energy_fn_template, virial_fn=None):
    """Initialize predictions for a single snapshot. Can be used to
    parametrize potentials from per-snapshot energy, force and/or virial.
    """
    def single_prediction(params, observation):
        energy_fn = energy_fn_template(params)
        pos = observation['R']
        # TODO check for neighborlist overflow and hand through
        nbrs = nbrs_init.update(pos)
        energy, negative_forces = value_and_grad(energy_fn)(pos, neighbor=nbrs)
        predictions = {'U': energy, 'F': -negative_forces}
        if virial_fn is not None:
            predictions['virial'] = virial_fn(State(pos), nbrs, params)
        return predictions
    return single_prediction


def init_update_fns(energy_fn_template, nbrs_init, optimizer, gamma_f=1.,
                    gamma_p=1.e-6, box_tensor=None, virial_type=False):
    """Initializes update functions for energy and/or force matching.

    The returned functions are jit and can therefore not be pickled.
    """
    if virial_type is not None:
        if virial_type == 'scalar':
            virial_fn = custom_quantity.init_pressure(
                energy_fn_template, box_tensor, include_kinetic=False)
        elif virial_type == 'tensor':
            virial_fn = custom_quantity.init_virial_stress_tensor(
                energy_fn_template, box_tensor, include_kinetic=False)
        else:
            raise ValueError(f'Virial_type {virial_type} not recognized. Should'
                             f' be "scalar" or "tensor".')
    else:
        virial_fn = None

    single_prediction = init_single_prediction(nbrs_init, energy_fn_template,
                                               virial_fn)

    def loss_fn(params, batch):
        predictions = vmap(single_prediction, in_axes=(None, 0))(params, batch)
        loss = 0.
        if 'U' in batch.keys():  # energy is loss component
            loss += util.mse_loss(predictions['U'], batch['U'])
        if 'F' in batch.keys():  # forces are loss component
            loss += gamma_f * util.mse_loss(predictions['F'], batch['F'])
        if 'p' in batch.keys():  # virial is loss component
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
        return new_params, opt_state, loss

    return batch_update, batched_loss_fn
