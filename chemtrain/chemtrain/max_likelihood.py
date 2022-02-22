"""A collection of functions to facilitate learning maximum likelihood /
 single point estimate models.
 """
import abc
import copy
from functools import partial
import time

from jax import lax, pmap, value_and_grad, tree_map, numpy as jnp
import numpy as onp
import optax

from chemtrain import util


def pmap_update_fn(loss_fn, optimizer):
    """Initializes a pmapped function for updating parameters.

    Usage:
    params, opt_state, loss, grad = update_fn(params, opt_state, batch)

    For pmap, batch needs to be reshaped to (N_devices, batch_per_device, X) and
    params needs to be N_devices times duplicated along axis 0.


    Args:
        loss_fn: Loss function to minimize that takes (params, batch) as input.
        optimizer: Optax optimizer

    Returns:
        A function that computes the gradient and updates the parameters via the
        optimizer.
    """
    @partial(pmap, axis_name='devices')
    def batch_update(params, opt_state, batch):
        loss, grad = value_and_grad(loss_fn)(params, batch)

        # step optimizer within pmap to minimize communication overhead
        grad = lax.pmean(grad, axis_name='devices')
        loss = lax.pmean(loss, axis_name='devices')
        new_params, opt_state = step_optimizer(params, opt_state, grad,
                                               optimizer)
        return new_params, opt_state, loss, grad
    return batch_update


def pmap_loss_fn(loss_fn):
    """Initializes a pmapped loss function.

    The loss function can be used for evaluating the validation loss.
    In priniple, any function with the same signature as loss_fn can be used.

    Usage:
    val_loss = batched_loss_fn(params, val_batch)

    For pmap, batch needs to be reshaped to (N_devices, batch_per_device, X) and
    params needs to be N_devices times duplicated along axis 0.

    Args:
        loss_fn: Loss function that takes (params, batch) as input.

    Returns:
        A pmapped function that returns the loss value.
    """
    @partial(pmap, axis_name='devices')
    def batched_loss_fn(params, batch):
        loss = loss_fn(params, batch)
        loss = lax.pmean(loss, axis_name='devices')
        return loss
    return batched_loss_fn


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


class MLETrainerTemplate(util.TrainerInterface):
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
