from abc import ABC, abstractmethod
from typing import Any
import numpy as onp
from pathlib import Path
import cloudpickle as pickle
import optax
from jax import lax, tree_map, tree_leaves, numpy as jnp
import jax_md.util

from chemtrain.jax_md_mod import custom_space
import copy
from chex import dataclass
from functools import partial
# import h5py


# freezing seems to give slight performance improvement
@partial(dataclass, frozen=True)
class TrainerState:
    """Each trainer at least contains the state of parameter and
    optimizer.
    """
    params: Any
    opt_state: Any


def mse_loss(predictions, targets):
    """Computes mean squared error loss for given predictions and targets."""
    squared_difference = jnp.square(targets - predictions)
    mean_of_squares = jax_md.util.high_precision_sum(squared_difference) \
                      / predictions.size
    return mean_of_squares


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
        "First dimension needs to be multiple of number of devices."
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


def jit_fn_not_found_error():
    raise AttributeError("Please store the (jit-compiled) function under "
                         "'self.update' or 'self.grad_fns', such that it "
                         "can be deleted here as it cannot be pickled.")


def format_not_recognized_error(format):
    raise ValueError(f"File format {format} not recognized. "
                     f"Expected '.hdf5' or '.pkl'.")


class TrainerInterface(ABC):
    """Abstract class defining the user interface of trainers."""


class MLETrainerTemplate(ABC):
    """Abstract class implementing common properties and methods of single
    point estimate Trainers using optax optimizers.
    """

    def __init__(self, optimizer, init_state, checkpoint_path, checkpoint_format='.pkl',
                 reference_energy_fn_template=None):
        """Forces implementation of checkpointing routines. A reference
        energy_fn_template can be provided, but is not mandatory due to
        the dependence of the template on the box via the displacement
        function.
        """
        self.state = init_state
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.check_format = checkpoint_format
        self.epoch = 0
        self.reference_energy_fn_template = reference_energy_fn_template
        self.update_times = []

        if checkpoint_format == '.pkl':
            print('Pickle is useful for checkpointing as the whole trainer '
                  '(except for jitted functions) can be saved. However, using'
                  '(cloud-)pickle for long-term storage is highly discouraged. '
                  'Consider saving learned energy_params in a different '
                  'format, e.g. using the save_energy_params function.')

    def step_optimizer(self, curr_grad):
        """Steps optimizer and updates state using the gradient."""
        scaled_grad, opt_state = self.optimizer.update(curr_grad,
                                                       self.state.opt_state)
        new_params = optax.apply_updates(self.state.params, scaled_grad)
        self.state = self.state.replace(params=new_params, opt_state=opt_state)

    def dump_checkpoint_occasionally(self, frequency=None):
        """Dumps a checkpoint during training, from which training can
        be resumed.
        """
        if frequency is not None:
            Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)
            if self.epoch % frequency == 0:  # checkpoint model
                if self.check_format == 'pkl':
                    file_path = self.checkpoint_path + \
                                f'/epoch{self.epoch - 1}.pkl'
                    save_dict = self.__dict__.copy()
                    # jitted function cannot be pickled
                    try:
                        save_dict.pop('grad_fns')  # for difftre / rel_entropy
                    except KeyError:
                        try:
                            save_dict.pop('update')  # for force matching  # TODO unique interface?
                        except KeyError:
                            raise jit_fn_not_found_error()
                    with open(file_path, 'wb') as f:
                        pickle.dump(save_dict, f)

                elif self.check_format == 'hdf5':
                    file_path = self.checkpoint_path + f'/checkpoints.hdf5'
                    raise NotImplementedError  # TODO
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
        except KeyError:
            try:
                trainer_copy.__delattr__('update')
            except KeyError:
                raise jit_fn_not_found_error()
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

    @abstractmethod
    def train(self, epochs, checkpoints=None):
        """Training method only takes the number of epochs to run,
        runs them and saves the optimization state into self.state,
        such that calling train again will resume optimization
        from the last state.
        """

    @abstractmethod
    def evaluate_convergence(self, *args, **kwargs):
        """Implement a function that checks for convergence to break
        training loop"""

    @property
    @abstractmethod
    def params(self):
        # cannot be implemented here due to different parallelization schemes
        # for different trainers
        raise NotImplementedError()

    @params.setter
    @abstractmethod
    def params(self, loaded_params):
        raise NotImplementedError()

    @property
    def energy_fn(self):
        return self.reference_energy_fn_template(self.params)
