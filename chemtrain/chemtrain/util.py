from abc import ABC, abstractmethod
from typing import Any
import numpy as onp
from pathlib import Path
import dill as pickle
from jax import lax, tree_map, tree_leaves, numpy as jnp
from chemtrain.jax_md_mod import custom_space
import copy
from chex import dataclass
from functools import partial
# import h5py


# freezing seems to give slight performance improvement
@partial(dataclass, frozen=True)
class TrainerStateTemplate:
    """Each trainer at least contains the state of parameter and
    optimizer. For consistency, we therefore include these parameters
    here.
    """
    params: Any
    opt_state: Any


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


def get_dataset(configuration_str, retain=None, subsampling=1):
    data = onp.load(configuration_str)
    data = data[:retain:subsampling]
    return data


def scale_dataset_fractional(traj, box):
    _, scale_fn = custom_space.init_fractional_coordinates(box)
    scaled_traj = lax.map(scale_fn, traj)
    return scaled_traj


def update_not_found_error():
    return AttributeError("Please store the (jit-compiled) update "
                          "function under 'self.update', such that it "
                          "can be deleted here as it cannot be "
                          "pickled.")


def assert_distribuatable(total_samples, n_devies, vmap_per_device):
    assert total_samples % (n_devies * vmap_per_device) == 0, \
        "For parallelization, the samples need to be evenly distributed " \
        "over the devices and vmap, i.e. be a multiple of n_devices * n_vmap."


class TrainerTemplate(ABC):
    """Abstract class to define common properties and methods of Trainers."""

    def __init__(self, energy_fn_template, checkpoint_format, checkpoint_path):
        """Forces implementation of checkpointing routines and
        energy_fn_template.
        """
        self.energy_fn_template = energy_fn_template
        self.checkpoint_path = checkpoint_path
        self.check_format = checkpoint_format
        self.epoch = 0

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
                    try:
                        # jitted function cannot be pickled
                        save_dict.pop('update')
                    except AttributeError:
                        raise update_not_found_error()
                    with open(file_path, 'wb') as f:
                        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

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
                    raise ValueError('File format needs to be pkl or hdf5.')

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
            raise ValueError('Filetype not recognized. '
                             'Expected pickle .pkl or .hdf5')
        self.state = tree_map(jnp.array, self.state)  # move state on device

    def save_trainer(self, save_path):
        """Saves whole trainer, e.g. for production after training."""
        with open(save_path, 'wb') as pickle_file:
            trainer_copy = copy.copy(self)
            try:
                trainer_copy.__delattr__('update')
            except AttributeError:
                raise update_not_found_error()
            pickle.dump(trainer_copy, pickle_file, pickle.HIGHEST_PROTOCOL)

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

    @abstractmethod
    def train(self, epochs, checkpoints=None):
        """Training method only takes the number of epochs to run,
        runs them and saves the optimization state into self.state,
        such that calling train again will resume optimization
        from the last state.
        """

    @property
    @abstractmethod
    def state(self):
        """Variable to save optimization state. The state needs to
        contain a parameter 'params' containing the current energy
        parameters.
        """

    @state.setter
    @abstractmethod
    def state(self, loadstate):
        raise NotImplementedError()

    @property
    @abstractmethod
    def params(self):
        raise NotImplementedError()

    @property
    def energy_fn(self):
        return self.energy_fn_template(self.params)
