from abc import ABC, abstractmethod
import numpy as onp
from pathlib import Path
import pickle
from jax import lax, tree_map, numpy as jnp
from jax_md_mod import custom_space


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
    else:
        raise ValueError('Filetype not recognized. '
                         'Expected pickle .pkl or .hdf5')
    state = tree_map(jnp.array, state)  # pickle saving converts to ordinary np
    return state


def load_params(state_file_path):
    state = load_state(state_file_path)
    return state.params


class TrainerTemplate(ABC):
    """Abstract class to define common properties  methods of Trainers."""

    def __init__(self, checkpoint_path):
        """Forces implementation of checkpointing routines and
        optimization state.
        """
        self.checkpoint_path = checkpoint_path
        self.create_checkpoint_directory(checkpoint_path)

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
        else:
            raise ValueError('File path needs to end with .pkl or .hdf5.')

    def dump_checkpoint_occasionally(self, epoch, frequency=None,
                                     file_format='pkl'):
        if frequency is not None:
            if epoch % frequency == 0:  # checkpoint model
                file_path = self.checkpoint_path + f'/state{epoch}.{file_format}'
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
    def params(self):
        return self.state.params
