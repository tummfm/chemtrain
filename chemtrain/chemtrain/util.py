"""Utility functions helpful in designing new trainers."""
import abc
import copy
from functools import partial
import pathlib
import time
from typing import Any

import chex
import cloudpickle as pickle
# import h5py
import numpy as onp
import optax
from jax import lax, tree_map, tree_leaves, numpy as jnp
from jax_md import util, simulate

from chemtrain.jax_md_mod import custom_space


# freezing seems to give slight performance improvement
@partial(chex.dataclass, frozen=True)
class TrainerState:
    """Each trainer at least contains the state of parameter and
    optimizer.
    """
    params: Any
    opt_state: Any


def _get_box_kwargs_if_npt(state):
    kwargs = {}
    if is_npt_ensemble(state):
        box = simulate.npt_box(state)
        kwargs['box'] = box
    return kwargs


def neighbor_update(neighbors, state):
    """Update neighbor lists irrespective of the ensemble.

    Fetches the box to the neighbor list update function in case of the
    NPT ensemble.

    Args:
        neighbors: Neighbor list to be updated
        state: Simulation state

    Returns:
        Updated neighbor list
    """
    kwargs = _get_box_kwargs_if_npt(state)
    nbrs = neighbors.update(state.position, **kwargs)
    return nbrs


def neighbor_allocate(neighbor_fn, state):
    """Re-allocates neighbor lost irrespective of ensemble. Not jitable.

    Args:
        neighbor_fn: Neighbor function to re-allocate neighbor list
        state: Simulation state

    Returns:
        Updated neighbor list
    """
    kwargs = _get_box_kwargs_if_npt(state)
    nbrs = neighbor_fn.allocate(state.position, **kwargs)
    return nbrs


def mse_loss(predictions, targets):
    """Computes mean squared error loss for given predictions and targets."""
    squared_difference = jnp.square(targets - predictions)
    mean_of_squares = util.high_precision_sum(
        squared_difference) / predictions.size
    return mean_of_squares


def step_optimizer(params, opt_state, grad, optimizer):
    """Steps optimizer and updates state using the gradient."""
    scaled_grad, new_opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, scaled_grad)
    return new_params, new_opt_state


def is_npt_ensemble(state):
    """Whether a state belongs to the NPT ensemble."""
    return hasattr(state, 'box_position')


def tree_get_single(tree):
    """Returns the first tree of a tree-replica, e.g. from pmap and and moves
    it to the default device.
    """
    single_tree = tree_map(lambda x: jnp.array(x[0]), tree)
    return single_tree


def tree_replicate(tree, n_devices):
    """Replicates a pytree along the first axis for pmap."""
    return tree_map(lambda x: jnp.array([x] * n_devices), tree)


def tree_split(tree, n_devices):
    """Splits the first axis of `tree` evenly across the number of devices."""
    assert tree_leaves(tree)[0].shape[0] % n_devices == 0, \
        'First dimension needs to be multiple of number of devices.'
    return tree_map(lambda x: jnp.reshape(x, (n_devices, x.shape[0]//n_devices,
                                              *x.shape[1:])), tree)


def tree_mean(tree_list):
    """Computes the mean a list of equal-shaped pytrees."""
    @partial(partial, tree_map)
    def tree_add_imp(*leafs):
        return jnp.mean(jnp.stack(leafs), axis=0)

    return tree_add_imp(*tree_list)


def get_dataset(configuration_str, retain=None, subsampling=1):
    data = onp.load(configuration_str)
    data = data[:retain:subsampling]
    return data


def scale_dataset_fractional(traj, box):
    _, scale_fn = custom_space.init_fractional_coordinates(box)
    scaled_traj = lax.map(scale_fn, traj)
    return scaled_traj


def jit_fn_not_found_error(e):
    raise AttributeError('Please store the (jit-compiled) function under '
                         '"self.update" or "self.grad_fns", such that it '
                         'can be deleted here as it cannot be pickled.') from e


def format_not_recognized_error(file_format):
    raise ValueError(f'File format {file_format} not recognized. '
                     f'Expected ".hdf5" or ".pkl".')


class TrainerInterface(abc.ABC):
    """Abstract class defining the user interface of trainers."""


class MLETrainerTemplate(abc.ABC):
    """Abstract class implementing common properties and methods of single
    point estimate Trainers using optax optimizers.
    """

    def __init__(self, optimizer, init_state, checkpoint_path,
                 checkpoint_format='.pkl', reference_energy_fn_template=None):
        """Forces implementation of checkpointing routines. A reference
        energy_fn_template can be provided, but is not mandatory due to
        the dependence of the template on the box via the displacement
        function.
        """
        self.state = init_state
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.check_format = checkpoint_format
        self._epoch = 0
        self.reference_energy_fn_template = reference_energy_fn_template
        self.update_times = []
        self.converged = False

    @property
    def energy_fn(self):
        return self.reference_energy_fn_template(self.params)

    def _step_optimizer(self, curr_grad):
        """Wrapper around step_optimizer that is useful whenever the
        update of the optimizer can be done outside of jit-compiled functions.
        """
        new_params, new_opt_state = step_optimizer(self.params,
                                                   self.state.opt_state,
                                                   curr_grad,
                                                   self.optimizer)
        self.state = self.state.replace(params=new_params,
                                        opt_state=new_opt_state)

    def _dump_checkpoint_occasionally(self, frequency=None):
        """Dumps a checkpoint during training, from which training can
        be resumed.
        """
        if frequency is not None:
            pathlib.Path(self.checkpoint_path).mkdir(parents=True,
                                                     exist_ok=True)
            if self._epoch % frequency == 0:  # checkpoint model
                if self.check_format == 'pkl':
                    file_path = (self.checkpoint_path +
                                 f'/epoch{self._epoch - 1}.pkl')
                    save_dict = self.__dict__.copy()
                    # jitted function cannot be pickled
                    try:
                        save_dict.pop('grad_fns')  # for difftre / rel_entropy
                    except KeyError as e:
                        jit_fn_not_found_error(e)
                    with open(file_path, 'wb') as f:
                        pickle.dump(save_dict, f)

                elif self.check_format == 'hdf5':
                    # file_path = self.checkpoint_path + f'/checkpoints.hdf5'
                    raise NotImplementedError
                    # from jax_sgmc.io import pytree_dict_keys, dict_to_pytree
                    # leaf_names = pytree_dict_keys(self.state)
                    # leafes = tree_leaves(self.state)
                    # with h5py.File(file_path, "w") as file:
                    #     for leaf_name, value in zip(leaf_names, leafes):
                    #         file[leaf_name] = value
                else:
                    format_not_recognized_error(self.check_format)

    def load_checkpoint(self, file_path):
        """Loads a saved checkpoint, if trainer was already initialized.
        Allows continuation of training."""
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as pickle_file:
                loaded_dict = pickle.load(pickle_file)
            self.__dict__.update(loaded_dict)
        elif file_path.endswith('.hdf5'):
            raise NotImplementedError
            # state = dict_to_pytree(as_dict['b'], some_tree['b'])
        else:
            format_not_recognized_error(file_path[-4:])
        self.state = tree_map(jnp.array, self.state)  # move state on device

    def save_trainer(self, save_path):
        """Saves whole trainer, e.g. for production after training."""
        trainer_copy = copy.copy(self)
        try:
            trainer_copy.__delattr__('grad_fns')
        except AttributeError as e:
            jit_fn_not_found_error(e)
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(trainer_copy, pickle_file)

    @classmethod
    def load_trainer(cls, file_path):
        """Loads a trainer saved via 'save_trainer'.
        Does not require initialization of the trainer class,
        but does not allow continuation of training because
        update function cannot be used. Save checkpoints instead,
        if re-training is needed.
        """
        with open(file_path, 'rb') as pickle_file:
            trainer = pickle.load(pickle_file)
        trainer.state = tree_map(jnp.array, trainer.state)  # move on device
        return trainer

    def save_energy_params(self, file_path, save_format='.hdf5'):
        if save_format == '.hdf5':
            raise NotImplementedError  # TODO implement hdf5
        elif save_format == '.pkl':
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(self.params, pickle_file)
        else:
            format_not_recognized_error(save_format)

    def load_energy_params(self, file_path):
        if file_path.endswith('.hdf5'):
            raise NotImplementedError
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as pickle_file:
                params = pickle.load(pickle_file)
        else:
            format_not_recognized_error(file_path[-4:])
        self.params = tree_map(jnp.array, params)  # move state on device

    def train(self, epochs, checkpoint_freq=None, thresh=None):
        """Trains for a specified number of epochs, checkpoints after a
        specified number of epochs and ends training if a convergence
        criterion is met. This function can be called multiple times to extend
        training.

        This function only implements the training sceleton by splitting the
        training into epochs and batches as well as providing checkpointing and
        ending of training if the convergence criterion is met. The specifics
        of dataloading, parameter updating and convergence criterion evaluation
        needs to be implemented in _get_batch(), _update() and
        _evaluate_convergence(), respectively, depending on the exact trainer
        details to be implemented.
        """
        self.converged = False
        start_epoch = self._epoch
        end_epoch = start_epoch + epochs
        for _ in range(start_epoch, end_epoch):
            start_time = time.time()
            for batch in self._get_batch():
                self._update(batch)
            duration = (time.time() - start_time) / 60.
            self.update_times.append(duration)
            self._evaluate_convergence(duration, thresh)
            self._epoch += 1
            self._dump_checkpoint_occasionally(frequency=checkpoint_freq)

            if self.converged:
                break
        if thresh is not None:
            print('Maximum number of epochs reached without convergence.')

    @abc.abstractmethod
    def _get_batch(self):
        """A generator that returns the next batch that will be provided to the
        _update function. The length of the generator should correspond to the
        number of batches per epoch.
        """

    @abc.abstractmethod
    def _update(self, batch):
        """Uses the current batch to updates self.state via the training scheme
        implemented in the specific trainer. Can additionally save auxilary
        optimization results, such as losses and observables, that can be
        used by _evaluate_convergence and for post-processing.
        """

    @abc.abstractmethod
    def _evaluate_convergence(self, duration, thresh):
        """Checks whether a convergence criterion has been met. Can also be
        used to print callbacks, such as time per epoch and loss vales.
        """

    @property
    @abc.abstractmethod
    def params(self):
        """Short-cut for parameters. Cannot be implemented here due to
        different parallelization schemes for different trainers.
        """

    @params.setter
    @abc.abstractmethod
    def params(self, loaded_params):
        raise NotImplementedError()
