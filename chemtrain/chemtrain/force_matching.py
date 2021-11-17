from collections import namedtuple
from jax import jit, vmap, lax, device_count, value_and_grad, pmap
from jax_md import quantity
import time
import numpy as onp
from jax_sgmc.data import NumpyDataLoader, random_reference_data
from chemtrain.util import MLETrainerTemplate, TrainerState, \
    tree_split, tree_get_single, tree_replicate, mse_loss, step_optimizer
from chemtrain.jax_md_mod import custom_quantity
from functools import partial

# Note:
#  Computing the neighborlist in each snapshot is not efficient for DimeNet++,
#  which constructs a sparse graph representation afterwards. However, other
#  models such as the tabulated potential are inefficient if used without
#  neighbor list as many cut-off interactions are otherwise computed.
#  For the sake of a simpler implementation, the slight inefficiency
#  in the case of DimeNet++ is accepeted for now.

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
                   gamma_p=1.e-6, box_tensor=None, include_virial=False,
                   n_devices=1):

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
        new_params, opt_state = step_optimizer(params, opt_state,
                                               grad, optimizer)
        return new_params, opt_state, loss

    def grad_fn(state, train_batch, val_batch):
        # this function should not be jitted as jit of pmap is discouraged
        train_batch = tree_split(train_batch, n_devices)
        val_batch = tree_split(val_batch, n_devices)

        params, opt_state, train_loss = batch_update(state.params,
                                                     state.opt_state,
                                                     train_batch)

        val_loss = batched_loss_fn(params, val_batch)
        new_state = state.replace(params=params, opt_state=opt_state)
        return new_state, train_loss, val_loss

    return grad_fn


class Trainer(MLETrainerTemplate):
    # TODO save best params during training based on val loss

    # TODO end training when val loss does not decrease for certain
    #  number of epochs / update steps; maybe exponentially moving average?
    def __init__(self, init_params, energy_fn_template, neighbor_fn, nbrs_init,
                 optimizer, position_data, force_data, virial_data=None,
                 box_tensor=None, gamma_p=1.e-6, batch_per_device=1,
                 batch_cache=10, train_ratio=0.875,
                 checkpoint_folder='Checkpoints', checkpoint_format='pkl'):

        # setup dataset
        self.n_devices, self.batches_per_epoch, self.get_train_batch, \
            self.get_val_batch, self.train_batch_state, self.val_batch_state = \
            self._process_dataset(position_data, train_ratio, force_data,
                                  virial_data, box_tensor, batch_per_device,
                                  batch_cache)
        self.train_losses, self.val_losses = [], []
        opt_state = optimizer.init(init_params)  # initialize optimizer state

        # replicate params and optimizer states for pmap
        init_params = tree_replicate(init_params, self.n_devices)
        opt_state = tree_replicate(opt_state, self.n_devices)

        init_state = TrainerState(params=init_params,
                                  opt_state=opt_state)

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        super().__init__(optimizer, init_state, checkpoint_path,
                         checkpoint_format, energy_fn_template)

        self.grad_fns = init_update_fn(energy_fn_template, neighbor_fn, nbrs_init,
                                     optimizer,
                                     gamma_p=gamma_p, box_tensor=box_tensor,
                                     include_virial=False,  # TODO fix
                                     n_devices=self.n_devices)

    def update_dataset(self, position_data, train_ratio=0.875, force_data=None,
                       virial_data=None, box_tensor=None, batch_per_device=1,
                       batch_cache=10):
        """Allows changing dataset on the fly, which is particularly
        useful for active learning applications.
        """
        self.n_devices, self.batches_per_epoch, self.get_train_batch, \
            self.get_val_batch, self.train_data_state, self.val_batch_state = \
            self._process_dataset(position_data, train_ratio, force_data,
                                  virial_data, box_tensor, batch_per_device,
                                  batch_cache)
        # TODO re-initialize grad_fns if targets are changed!

    @staticmethod
    def _process_dataset(position_data, train_ratio=0.875, force_data=None,
                         virial_data=None, box_tensor=None, batch_per_device=1,
                         batch_cache=10):
        # Default train_ratio represents 70-10-20 split, when 20 % test
        # data was already deducted
        n_devices = device_count()
        batch_size = n_devices * batch_per_device
        train_set_size = position_data.shape[0]
        train_size = int(train_set_size * train_ratio)
        batches_per_epoch = train_size // batch_size
        R_train, R_val = onp.split(position_data, [train_size])
        F_train, F_val = onp.split(force_data, [train_size])
        train_dict = {'R': R_train, 'F': F_train}
        val_dict = {'R': R_val, 'F': F_val}
        # TODO is there better way to handle whether to inlcude virial?
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
        train_loader = NumpyDataLoader(R=R_train, F=F_train)
        val_loader = NumpyDataLoader(R=R_val, F=F_val)
        init_train_batch, get_train_batch = random_reference_data(
            train_loader, batch_cache, batch_size)
        init_val_batch, get_val_batch = random_reference_data(
            val_loader, batch_cache, batch_size)

        train_batch_state = init_train_batch()
        val_batch_state = init_val_batch()
        return n_devices, batches_per_epoch, get_train_batch, get_val_batch, \
            train_batch_state, val_batch_state

    @property
    def params(self):
        single_params = tree_get_single(self.params)
        return single_params

    @params.setter
    def params(self, loaded_params):
        replicated_params = tree_replicate(loaded_params, self.n_devices)
        self.params = replicated_params

    def update(self):
        """Function to scan over, optimizing parameters and returning
        training and validation loss values.
        """
        self.train_batch_state, train_batch = self.get_train_batch(
            self.train_batch_state)
        self.val_batch_state, val_batch = self.get_val_batch(self.val_batch_state)

        self.state, train_loss, val_loss = self.grad_fns(self.state, train_batch, val_batch)
        # only need loss_val from single device
        self.train_losses.append(train_loss[0])
        self.val_losses.append(val_loss[0])

    def evaluate_convergence(self, duration):
        mean_train_loss = sum(self.train_losses[-self.batches_per_epoch:]) \
                          / self.batches_per_epoch
        mean_val_loss = sum(self.val_losses[-self.batches_per_epoch:]) \
                        / self.batches_per_epoch
        print(f'Epoch {self.epoch}: Average train loss: {mean_train_loss} '
              f'Average val loss: {mean_val_loss} '
              f'Elapsed time = {duration} min')
        converged = False  # TODO implement convergence test
        return converged

    def train(self, epochs, checkpoint_freq=None, thresh=None):
        """Continue training for a number of epochs."""
        start_epoch = self.epoch
        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            start_time = time.time()
            # TODO replicate params here: we only checkpoint non-replicated
            #  params, but keep communication overhead within an epoch low.
            for i in range(self.batches_per_epoch):
                self.update()
            duration = (time.time() - start_time) / 60.

            converged = self.evaluate_convergence(duration)
            self.update_times.append(duration)
            self.epoch += 1
            self.dump_checkpoint_occasionally(frequency=checkpoint_freq)

            if converged:
                break
        if thresh is not None:
            print('Maximum number of epochs reached without convergence.')
