from collections import namedtuple
from jax import jit, vmap, lax, device_count, value_and_grad, pmap
import optax
from jax_md import quantity
from chemtrain.difftre import mse_loss
import time
import numpy as onp
from jax_sgmc.data import NumpyDataLoader, random_reference_data
from chemtrain.util import TrainerTemplate, TrainerStateTemplate, \
    tree_split, tree_get_single, tree_replicate
from chemtrain.jax_md_mod import custom_quantity
from functools import partial
from typing import Any
from chex import dataclass

# Note:
#  Computing the neighborlist in each snapshot is not efficient for DimeNet++,
#  which constructs a sparse graph representation afterwards. However, other
#  models such as the tabulated potential are inefficient if used without
#  neighbor list as many cut-off interactions are otherwise computed.
#  For the sake of a simpler implementation, the slight inefficiency
#  in the case of DimeNet++ is accepeted for now.


@partial(dataclass, frozen=True)
class FMState(TrainerStateTemplate):
    train_batch_state: Any
    val_batch_state: Any


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
                   box_tensor=None, include_virial=False, n_devices=1):

    if include_virial:
        virial_fn = custom_quantity.init_pressure(energy_fn_template,
                                                  box_tensor,
                                                  include_kinetic=False)

        loss_fn = init_force_loss_fn(energy_fn_template, neighbor_fn, nbrs_init,
                                     virial_fn, gamma_p=gamma_p)
    else:
        loss_fn = init_force_loss_fn(energy_fn_template, neighbor_fn, nbrs_init)

    @partial(pmap, axis_name='devices')
    def batched_loss_fn(params, batch):
        loss = loss_fn(params, batch)
        loss = lax.pmean(loss, axis_name='devices')
        return loss

    @partial(pmap, axis_name='devices')
    def batch_update(params, opt_state, batch):
        loss, grad = value_and_grad(loss_fn)(params, batch)
        grad = lax.pmean(grad, axis_name='devices')
        loss = lax.pmean(loss, axis_name='devices')
        updates, opt_state = optimizer.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    def update(train_state):
        """Function to scan over, optimizing parameters and returning
        training and validation loss values.
        """
        train_batch_state, train_batch = get_train_batch(
            train_state.train_batch_state)
        val_batch_state, val_batch = get_val_batch(train_state.val_batch_state)

        train_batch = tree_split(train_batch, n_devices)
        val_batch = tree_split(val_batch, n_devices)

        params, opt_state, train_loss = batch_update(train_state.params,
                                                     train_state.opt_state,
                                                     train_batch)

        val_loss = batched_loss_fn(params, val_batch)
        new_train_state = FMState(params=params,
                                  opt_state=opt_state,
                                  train_batch_state=train_batch_state,
                                  val_batch_state=val_batch_state)
        # only need loss_val from single device
        return new_train_state, train_loss[0], val_loss[0]

    return update


class Trainer(TrainerTemplate):
    # TODO save best params during training based on val loss

    # TODO end training when val loss does not decrease for certain
    #  number of epochs / update steps
    def __init__(self, init_params, energy_fn_template, neighbor_fn, nbrs_init,
                 optimizer, position_data, force_data, virial_data=None,
                 box_tensor=None, gamma_p=1.e-6, batch_per_device=1,
                 batch_cache=100, train_ratio=0.875,
                 checkpoint_folder='Checkpoints', checkpoint_format='pkl'):
        """
        Default training percentage represents 70-10-20 split, when 20 % test
        data was already deducted
        """

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        super().__init__(energy_fn_template, checkpoint_format, checkpoint_path)

        # split dataset and initialize dataloader
        n_devices = device_count()
        batch_size = n_devices * batch_per_device
        train_set_size = position_data.shape[0]
        train_size = int(train_set_size * train_ratio)
        self.batches_per_epoch = train_size // batch_size
        R_train, R_val = onp.split(position_data, [train_size])
        F_train, F_val = onp.split(force_data, [train_size])
        train_dict = {'R': R_train, 'F': F_train}
        val_dict = {'R': R_val, 'F': F_val}
        include_virial = virial_data is not None
        # TODO include energy data - maybe build more flexible datset setup:
        #  not 3 dfferent inputs
        if include_virial:
            # TODO: test virial matching! Predicted pressure should match
            assert box_tensor is not None, "If the virial is to be matched, " \
                                           "box_tensor is a mandatory input."
            p_train, p_val = onp.split(virial_data, [train_size])
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

        # replicate params and optimizer states for pmap
        init_params = tree_replicate(init_params, n_devices)
        opt_state = tree_replicate(opt_state, n_devices)

        self.__state = FMState(params=init_params,
                               opt_state=opt_state,
                               train_batch_state=train_data_state,
                               val_batch_state=val_batch_state)

        self.update = init_update_fn(energy_fn_template, neighbor_fn, nbrs_init,
                                     optimizer, get_train_batch, get_val_batch,
                                     gamma_p=gamma_p, box_tensor=box_tensor,
                                     include_virial=include_virial,
                                     n_devices=n_devices)

    @property
    def state(self):
        return self.__state

    @property
    def params(self):
        single_params = tree_get_single(self.__state.params)
        return single_params

    @state.setter
    def state(self, loaded_state):
        self.__state = loaded_state

    def train(self, epochs, checkpoint_freq=None):
        """Continue training for a number of epochs."""
        start_epoch = self.epoch
        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            start_time = time.time()
            train_losses, val_losses = [], []
            for i in range(self.batches_per_epoch):
                self.__state, train_loss, val_loss = self.update(self.__state)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            end_time = (time.time() - start_time) / 60.

            self.train_losses.extend(train_losses)
            self.val_losses.extend(val_losses)
            mean_train_loss = sum(train_losses) / len(train_losses)
            mean_val_loss = sum(val_losses) / len(val_losses)
            print('Training time for epoch', str(epoch), ',', str(end_time),
                  'min, average train loss:', mean_train_loss,
                  'average val loss:', mean_val_loss)

            self.epoch += 1
            self.dump_checkpoint_occasionally(frequency=checkpoint_freq)
        return
