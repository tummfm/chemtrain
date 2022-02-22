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
from jax import lax, tree_map, tree_leaves, tree_flatten, numpy as jnp
from jax_md import simulate
from jax_sgmc import data

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


def neighbor_allocate(neighbor_fn, state, extra_capacity=0):
    """Re-allocates neighbor lost irrespective of ensemble. Not jitable.

    Args:
        neighbor_fn: Neighbor function to re-allocate neighbor list
        state: Simulation state
        extra_capacity: Additional capacity of new neighbor list

    Returns:
        Updated neighbor list
    """
    kwargs = _get_box_kwargs_if_npt(state)
    nbrs = neighbor_fn.allocate(state.position, extra_capacity, **kwargs)
    return nbrs


def _masked_loss(per_element_loss, mask=None):
    """Computes average loss, accounting for masked elements, if applicable."""
    if mask is None:
        return jnp.mean(per_element_loss)
    else:
        assert mask.shape == per_element_loss.shape, ('Mask requires same shape'
                                                      ' as targets.')
        real_contributors = jnp.sum(mask)
        return jnp.sum(per_element_loss * mask) / real_contributors


def mse_loss(predictions, targets, mask=None):
    """Computes mean squared error loss for given predictions and targets.

    Args:
        predictions: Array of predictions
        targets: Array of respective targets. Needs to have same shape as
                 predictions.
        mask: Mask contribution of some array elements. Needs to have same shape
              as predictions. Default None applies no mask.

    Returns:
        Mean squared error loss value.
    """
    squared_differences = jnp.square(targets - predictions)
    return _masked_loss(squared_differences, mask)


def mae_loss(predictions, targets, mask=None):
    """Computes the mean absolute error for given predictions and targets.

    Args:
        predictions: Array of predictions
        targets: Array of respective targets. Needs to have same shape as
                 predictions.
        mask: Mask contribution of some array elements. Needs to have same shape
              as predictions. Default None applies no mask.

    Returns:
        Mean absolute error value.
    """
    abs_err = jnp.abs(targets - predictions)
    return _masked_loss(abs_err, mask)


def step_optimizer(params, opt_state, grad, optimizer):
    """Steps optimizer and updates state using the gradient."""
    scaled_grad, new_opt_state = optimizer.update(grad, opt_state)
    new_params = optax.apply_updates(params, scaled_grad)
    return new_params, new_opt_state


def is_npt_ensemble(state):
    """Whether a state belongs to the NPT ensemble."""
    return hasattr(state, 'box_position')


def tree_norm(tree):
    """Returns the Euclidean norm of a PyTree."""
    leaves, _ = tree_flatten(tree)
    return sum(jnp.vdot(x, x) for x in leaves)


def tree_get_single(tree, n=0):
    """Returns the n-th tree of a tree-replica, e.g. from pmap.
    By default, the first tree is returned.
    """
    single_tree = tree_map(lambda x: jnp.array(x[n]), tree)
    return single_tree


def tree_get_slice(tree, idx_start, idx_stop, take_every=1):
    """Returns a slice of trees taken from a tree-replica along axis 0."""
    return tree_map(lambda x: jnp.array(x[idx_start:idx_stop:take_every]), tree)


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


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.

    From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.

    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.

    From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves, treedef = tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


# def convert_to_list(params):
#     """Converts parameters returned by different trainers to a standartized
#     output format that is then accepted by all post processing routines
#     """
#     n_samples = onp.shape(params._leaves[0])[0]
#     param_list = [tree_get_single(params, n) for n in range(n_samples)]
#     return param_list


def assert_distributable(total_samples, n_devies, vmap_per_device):
    assert total_samples % (n_devies * vmap_per_device) == 0, (
        'For parallelization, the samples need to be evenly distributed'
        'over the devices and vmap, i.e. be a multiple of n_devices * n_vmap.')


def get_dataset(data_location_str, retain=None, subsampling=1):
    """Loads .pyy numpy dataset.

    Args:
        data_location_str: String of .npy data location
        retain: Number of samples to keep in the dataset
        subsampling: Only keep every subsampled sample of the data, e.g. 2.

    Returns:
        Subsampled data array
    """
    loaded_data = onp.load(data_location_str)
    loaded_data = loaded_data[:retain:subsampling]
    return loaded_data


def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.1):
    """Train-validation-test split for datasets. Works on arbitrary pytrees,
    including chex.dataclasses, dictionaries and single arrays.

    Args:
        dataset: Dataset pytree. Samples are assumed to be stacked along
                 axis 0.
        train_ratio: Percantage of dataset to use for training.
        val_ratio: Percantage of dataset to use for validation.

    Returns:
        Tuple (train_data, val_data, test_data) with the same shape as the input
        pytree, but split along axis 0.
    """
    leaves, _ = tree_flatten(dataset)
    dataset_size = leaves[0].shape[0]
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    train_data = tree_get_slice(dataset, 0, train_size)
    val_data = tree_get_slice(dataset, train_size, train_size + val_size)
    test_data = tree_get_slice(dataset, train_size + val_size, None)
    return train_data, val_data, test_data


def init_dataloaders(dataset, train_ratio=0.7, val_ratio=0.1):
    """Splits dataset and initializes dataloaders.

    Args:
        dataset: Dictionary containing the whole dataset. The NumpyDataLoader
                 returns batches with the same kwargs as provided in dataset.
        train_ratio: Percantage of dataset to use for training.
        val_ratio: Percantage of dataset to use for validation.

    Returns:
        A tuple (train_loader, val_loader, test_loader, test_set) of
        NumpyDataLoaders and the test data.
    """
    train_set, val_set, test_set = train_val_test_split(
        dataset, train_ratio, val_ratio)
    train_loader = data.NumpyDataLoader(**train_set)
    val_loader = data.NumpyDataLoader(**val_set)
    test_loader = data.NumpyDataLoader(**test_set)
    return train_loader, val_loader, test_loader, test_set


def scale_dataset_fractional(traj, box):
    _, scale_fn = custom_space.init_fractional_coordinates(box)
    scaled_traj = lax.map(scale_fn, traj)
    return scaled_traj


def load_trainer(file_path):
    """Returns the trainer saved via 'trainer.save_trainer'.

    Args:
        file_path: Path of pickle file containing trainer.

    """
    with open(file_path, 'rb') as pickle_file:
        trainer = pickle.load(pickle_file)
    trainer.move_to_device()
    return trainer


def format_not_recognized_error(file_format):
    raise ValueError(f'File format {file_format} not recognized. '
                     f'Expected ".hdf5" or ".pkl".')


class TrainerInterface(abc.ABC):
    """Abstract class defining the user interface of trainers as well as
    checkpointing functionality.
    """
    def __init__(self, checkpoint_path, reference_energy_fn_template=None):
        """A reference energy_fn_template can be provided, but is not mandatory
        due to the dependence of the template on the box via the displacement
        function.
        """
        self.checkpoint_path = checkpoint_path
        self._epoch = 0
        self.reference_energy_fn_template = reference_energy_fn_template

    @property
    def energy_fn(self):
        if self.reference_energy_fn_template is None:
            raise ValueError('Cannot construct energy_fn as no reference '
                             'energy_fn_template was provided during '
                             'initialization.')
        return self.reference_energy_fn_template(self.params)

    def _dump_checkpoint_occasionally(self, frequency=None):
        """Dumps a checkpoint during training, from which training can
        be resumed.
        """
        assert self.checkpoint_path is not None
        if frequency is not None:
            pathlib.Path(self.checkpoint_path).mkdir(parents=True,
                                                     exist_ok=True)
            if self._epoch % frequency == 0:  # checkpoint model
                file_path = (self.checkpoint_path +
                             f'/epoch{self._epoch - 1}.pkl')
                self.save_trainer(file_path)

    def save_trainer(self, save_path):
        """Saves whole trainer, e.g. for production after training."""
        with open(save_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def save_energy_params(self, file_path, save_format='.hdf5'):
        if save_format == '.hdf5':
            raise NotImplementedError  # TODO implement hdf5
            # from jax_sgmc.io import pytree_dict_keys, dict_to_pytree
            # leaf_names = pytree_dict_keys(self.state)
            # leafes = tree_leaves(self.state)
            # with h5py.File(file_path, "w") as file:
            #     for leaf_name, value in zip(leaf_names, leafes):
            #         file[leaf_name] = value
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

    @property
    @abc.abstractmethod
    def params(self):
        """Short-cut for parameters. Depends on specific trainer"""

    @params.setter
    @abc.abstractmethod
    def params(self, loaded_params):
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """Training of any trainer should start by calling train."""

    @abc.abstractmethod
    def move_to_device(self):
        """Move all attributes that are expected to be on device to device to
         avoid TracerExceptions after loading trainers from disk, i.e.
         loading numpy rather than device arrays.
         """


class MLETrainerTemplate(TrainerInterface):
    """Abstract class implementing common properties and methods of single
    point estimate Trainers using optax optimizers.
    """

    def __init__(self, optimizer, init_state, checkpoint_path,
                 reference_energy_fn_template=None):
        super().__init__(checkpoint_path, reference_energy_fn_template)
        self.state = init_state
        self.optimizer = optimizer
        self.update_times = []
        self.gradient_norm_history = []
        self._converged = False
        self._diverged = False

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

    def train(self, max_epochs, thresh=None, checkpoint_freq=None):
        """Trains for a maximum number of epochs, checkpoints after a
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

        Args:
            max_epochs: Maximum number of epochs for which training is
                        continued. Training will end sooner if convergence
                        criterion is met.
            thresh: Threshold of the early stopping convergence criterion. If
                    None, no early stopping is applied. Definition of thresh
                    depends on specific convergence criterion.
                    See EarlyStopping.
            checkpoint_freq: Number of epochs after which a checkpoint is saved.
                             No checkpoints are saved by default.
        """
        self._converged = False
        start_epoch = self._epoch
        end_epoch = start_epoch + max_epochs
        for _ in range(start_epoch, end_epoch):
            start_time = time.time()
            for batch in self._get_batch():
                self._update(batch)
            duration = (time.time() - start_time) / 60.
            self.update_times.append(duration)
            self._evaluate_convergence(duration, thresh)
            self._epoch += 1
            self._dump_checkpoint_occasionally(frequency=checkpoint_freq)

            if self._converged or self._diverged:
                break
        else:
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

    def move_to_device(self):
        self.state = tree_map(jnp.array, self.state)  # move on device


class EarlyStopping:
    """A class that saves the best parameter obtained so far based on the
    validation loss and determines whether the optimization can be stopped based
    on some stopping criterion.

    The following criteria are implemented:
    * 'window_median': 2 windows are placed at the end of the loss history.
                       Stops when the median of the latter window of size
                       "thresh" exceeds the median of the prior window of the
                       same size.
    * 'PQ': Stops when the PQ criterion exceeds thresh
    * 'max_loss': Stops when the loss decreased below the maximum allowed loss
                  specified cia thresh.
    """
    def __init__(self, criterion, pq_window_size=5):
        """Initialize EarlyStopping.

        Args:
            criterion: Convergence criterion to employ
            pq_window_size: Window size for PQ method
        """
        self.criterion = criterion

        # own loss history that can be reset on the fly if needed.
        self._epoch_losses = []
        self.best_loss = 1.e16
        self.best_params = None  # need to be moved on device if loaded

        self.pq_window_size = pq_window_size

    def _is_converged(self, thresh):
        converged = False
        if thresh is not None:  # otherwise no early stopping used
            if self.criterion == 'window_median':
                window_size = thresh
                if len(self._epoch_losses) >= 2 * window_size:
                    prior_window = onp.array(
                        self._epoch_losses[-2 * window_size:-window_size])
                    latter_window = onp.array(self._epoch_losses[-window_size:])
                    converged = (onp.median(latter_window)
                                 > onp.median(prior_window))

            elif self.criterion == 'PQ':
                if len(self._epoch_losses) >= self.pq_window_size:
                    best_loss = min(self._epoch_losses)
                    loss_window = self._epoch_losses[-self.pq_window_size:]
                    gen_loss = 100. * (loss_window[-1] / best_loss - 1.)
                    window_average = sum(loss_window) / self.pq_window_size
                    window_min = min(loss_window)
                    progress = 1000. * (window_average / window_min - 1.)
                    pq = gen_loss / progress
                    converged = pq > thresh

            elif self.criterion == 'max_loss':
                converged = self._epoch_losses[-1] < thresh
            else:
                raise ValueError(f'Convergence criterion {self.criterion} '
                                 f'unknown. Select "max_loss", "ave_loss" or '
                                 f'"std".')
        return converged

    def early_stopping(self, curr_epoch_loss, thresh, params=None,
                       save_best_params=True):
        """Estimates whether convergence criterion was met and keeps track of
        best parameters obtained so far.

        Args:
            curr_epoch_loss: Validation loss of the most recent epoch
            thresh: Convergence threshold. Specific definition depends on the
                    selected convergence criterion.
            params: Optimization parameters to save in case of being best
            save_best_params: If best params are supposed to be tracked

        Returns:
            True if the convergence criterion was met, else False.
        """
        self._epoch_losses.append(curr_epoch_loss)

        if save_best_params:
            assert params is not None, ('If best params are saved, they need to'
                                        ' be provided in early_stopping.')
            improvement = self.best_loss - curr_epoch_loss
            if improvement > 0.:
                self.best_loss = curr_epoch_loss
                self.best_params = copy.copy(params)

        return self._is_converged(thresh)

    def reset_convergence_losses(self):
        """Resets loss history used for convergence estimation, e.g. to avoid
        early stopping when loss increases due to on-the-fly changes in the
        dataset or the loss fucntion.
        """
        self._epoch_losses = []
        self.best_loss = 1.e16
        self.best_params = None

    def move_to_device(self):
        """Moves best_params to device to use them after loading trainer."""
        self.best_params = tree_map(jnp.array, self.best_params)
