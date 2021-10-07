from abc import ABC, abstractmethod
import numpy as onp
from pathlib import Path
import pickle
from jax import lax, tree_map, tree_leaves, numpy as jnp
from chemtrain.jax_md_mod import custom_space
from functools import partial
# import h5py


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


def load_state(state_file_path):
    if state_file_path.endswith('.pkl'):
        with open(state_file_path, 'rb') as pickle_file:
            state = pickle.load(pickle_file)
    elif state_file_path.endswith('.hdf5'):
        raise NotImplementedError
        # state = dict_to_pytree(as_dict['b'], some_tree['b'])
    else:
        raise ValueError('Filetype not recognized. '
                         'Expected pickle .pkl or .hdf5')
    state = tree_map(jnp.array, state)  # move loaded state on device
    return state


def load_params(state_file_path):
    state = load_state(state_file_path)
    return state.params


class TrainerTemplate(ABC):
    """Abstract class to define common properties  methods of Trainers."""

    def __init__(self, energy_fn_template, checkpoint_format, checkpoint_path):
        """Forces implementation of checkpointing routines and
        energy_fn_template.
        """
        self.energy_fn_template = energy_fn_template
        self.checkpoint_path = checkpoint_path
        self.create_checkpoint_directory(checkpoint_path)
        self.check_format = checkpoint_format
        self.epoch = 0
        if checkpoint_format == 'hdf5':
            pass


    @staticmethod
    def create_checkpoint_directory(checkpoint_path):
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        return

    def save_state(self, file_path):
        if file_path.endswith('.pkl'):
            with open(file_path, 'wb') as output:
                pickle.dump(self.state, output, pickle.HIGHEST_PROTOCOL)
        elif file_path.endswith('hdf5'):  # TODO
            raise NotImplementedError
            # from jax_sgmc.io import pytree_dict_keys, dict_to_pytree
            # leaf_names = pytree_dict_keys(self.state)
            # leafes = tree_leaves(self.state)
            # with h5py.File(file_path, "w") as file:
            #     for leaf_name, value in zip(leaf_names, leafes):
            #         file[leaf_name] = value

        else:
            raise ValueError('File path needs to end with .pkl or .hdf5.')

    def dump_checkpoint_occasionally(self, frequency=None):
        if frequency is not None:
            if self.epoch % frequency == 0:  # checkpoint model
                if self.check_format == 'pkl':
                    file_path = self.checkpoint_path + f'/epoch{self.epoch}.pkl'
                elif self.check_format == 'hdf5':
                    file_path = self.checkpoint_path + f'/checkpoints.hdf5'
                else:
                    raise ValueError('File format needs to be pkl or hdf5.')
                self.save_state(file_path)

    @abstractmethod
    def train(self, epochs, checkpoints=None):
        """Training method only takes the numer of epochs to run,
        runs them and saves the optimization state into self.state,
        such that calling tarin again will resume optimization
        from the last state.
        """

    @property
    @abstractmethod
    def state(self):
        """Variable to save optimization state. The state needs to
        contain a parameter 'params' conatining the current energy
        parameters.
        """

    @state.setter
    @abstractmethod
    def state(self, loadstate):
        raise NotImplementedError()

    def load_checkpoint(self, file_path):
        loaded_state = load_state(file_path)
        self.state = loaded_state

    @property
    @abstractmethod
    def params(self):
        raise NotImplementedError()

    @property
    def energy_fn(self):
        return self.energy_fn_template(self.params)


