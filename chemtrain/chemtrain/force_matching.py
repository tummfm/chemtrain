from collections import namedtuple

import jax
from jax import jit, vmap, lax
import optax
from jax import value_and_grad
import jax.numpy as jnp
from jax_md import quantity
from chemtrain.difftre import mse_loss
import time
import numpy as onp
from jax_sgmc.data import NumpyDataLoader, random_reference_data
from chemtrain.util import TrainerTemplate
from chemtrain.jax_md_mod import custom_quantity

# TODO add parallelization over multiple GPU

"""
Note:
Computing the neighborlist in each snapshot is not efficient for DimeNet++,
which constructs a sparse graph representation afterwards. However, other 
models such as the tabulated potential are inefficient if used without neighbor 
list as many cut-off interactions are otherwise computed.
For the sake of a simpler implementation, the slight inefficiency
in the case of DimeNet++ is accepeted for now.
"""


FMState = namedtuple(
    "FMState",
    ["params",
     "opt_state",
     "train_batch_state",
     "val_batch_state"]
)

State = namedtuple(
    "State",
    ["position"]
)
State.__doc__ = """Emulates structure of simulation state for 
compatibility with quantity functions.

position: atomic positions
"""


def init_force_loss_fn(energy_fn_template, neighbor_fn, nbrs_init,
                       virial_fn=None, gamma_p=1.e-6):

    @jit
    def force_loss(params, batch):
        energy_fn = energy_fn_template(params)
        force_fn = quantity.canonicalize_force(energy_fn)

        @vmap
        def batched_forces(R):
            neighbors = neighbor_fn(R, nbrs_init)
            return force_fn(R, neighbor=neighbors)

        loss = 0.
        predicted_forces = batched_forces(batch['R'])
        loss += mse_loss(predicted_forces, batch['F'])

        if virial_fn is not None:
            # Note:
            # maybe more efficient to combine both preditions, but XLA
            # might just optimize such that neighbors computed above
            # are re-used

            @vmap
            def batched_virial(R):
                # emulate state structure, as virial_fn usually requires
                # full state to include kinetic contribution
                state = State(R)
                nbrs = neighbor_fn(R, nbrs_init)
                return virial_fn(state, nbrs, params)

            predicted_p = virial_fn(batch['R'])
            loss += gamma_p * mse_loss(predicted_p, batch['p'])

        return loss
    return force_loss


def init_update_fn(energy_fn_template, neighbor_fn, nbrs_init, optimizer,
                   get_train_batch, get_val_batch, gamma_p=1.e-6,
                   box_tensor=None, include_virial=False):

    if include_virial:
        virial_fn = custom_quantity.init_pressure(energy_fn_template,
                                                  box_tensor,
                                                  include_kinetic=False)

        loss_fn = init_force_loss_fn(energy_fn_template, neighbor_fn, nbrs_init,
                                     virial_fn, gamma_p=gamma_p)
    else:
        loss_fn = init_force_loss_fn(energy_fn_template, neighbor_fn, nbrs_init)

    def batch_update(params, opt_state, loss_inputs):
        loss, grad = value_and_grad(loss_fn)(params, loss_inputs)
        updates, opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    @jit
    def update(train_state, dummy):
        """Function to scan over, optimizing parameters and returning
        training and validation loss values.
        """
        train_batch_state, train_batch = get_train_batch(
            train_state.train_batch_state)
        val_batch_state, val_batch = get_val_batch(train_state.val_batch_state)
        params, opt_state, train_loss = batch_update(train_state.params,
                                                     train_state.opt_state,
                                                     train_batch)
        val_loss = loss_fn(params, val_batch)
        new_train_state = FMState(params, opt_state, train_batch_state,
                                  val_batch_state)
        return new_train_state, (train_loss, val_loss)

    return update


class Trainer(TrainerTemplate):
    def __init__(self, init_params, energy_fn_template, neighbor_fn, nbrs_init,
                 optimizer, position_data, force_data, virial_data=None,
                 box_tensor=None, gamma_p=1.e-6, batch_size=1, batch_cache=100,
                 train_ratio=0.875, checkpoint_folder='Checkpoints'):
        """
        Default training percentage represents 70-10-20 split, when 20 % test
        data was already deducted
        """

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        super().__init__(checkpoint_path=checkpoint_path)

        # split dataset and initialize dataloader
        n_gpu = jax.device_count()
        self.epoch = 0
        train_set_size = position_data.shape[0]
        self.train_size = int(train_set_size * train_ratio)
        R_train, R_val = onp.split(position_data, [self.train_size])
        F_train, F_val = onp.split(force_data, [self.train_size])
        train_dict = {'R': R_train, 'F': F_train}
        val_dict = {'R': R_val, 'F': F_val}
        include_virial = virial_data is not None
        if include_virial:
            # TODO: test virial matching! Predicted pressure should match
            assert box_tensor is not None, "If the virial is to be matched, " \
                                           "box_tensor is a mandatory input."
            p_train, p_val = onp.split(virial_data, [self.train_size])
            train_dict['p'] = p_train
            val_dict['p'] = p_val
        train_loader = NumpyDataLoader(batch_size, R=R_train, F=F_train)
        val_loader = NumpyDataLoader(batch_size, R=R_val, F=F_val)
        init_train_batch, get_train_batch = random_reference_data(
            train_loader, batch_cache)
        init_val_batch, get_val_batch = random_reference_data(
            val_loader, batch_cache)

        train_data_state = init_train_batch()
        val_batch_state = init_val_batch()
        opt_state = optimizer.init(init_params)  # initialize optimizer state
        self.train_losses, self.val_losses = [], []
        self.train_state = FMState(init_params, opt_state, train_data_state,
                                   val_batch_state)
        self.update = init_update_fn(energy_fn_template, neighbor_fn, nbrs_init,
                                     optimizer, get_train_batch, get_val_batch,
                                     gamma_p=gamma_p, box_tensor=box_tensor,
                                     include_virial=include_virial)

    @property
    def state(self):
        return self.train_state

    @state.setter
    def state(self, loaded_state):
        self.train_state = loaded_state

    def train(self, epochs, checkpoints=None):
        start_epoch = self.epoch
        end_epoch = self.epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            # training
            start_time = time.time()
            self.train_state, losses = lax.scan(self.update,
                                                self.train_state,
                                                jnp.arange(self.train_size))
            end_time = (time.time() - start_time) / 60.
            train_losses, val_losses = losses
            self.train_losses.extend(train_losses)
            self.val_losses.extend(val_losses)
            mean_train_loss = sum(train_losses) / len(train_losses)
            mean_val_loss = sum(val_losses) / len(val_losses)
            print('Training time for epoch', str(epoch), ',', str(end_time),
                  'min, average train loss:', mean_train_loss,
                  'average val loss:', mean_val_loss)

            self.dump_checkpoint_occasionally(epoch, frequency=checkpoints)
        return
