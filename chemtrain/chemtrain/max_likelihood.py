"""A collection of functions to facilitate learning maximum likelihood /
 single point estimate models.
 """
import abc
import copy
from functools import partial
import time

from jax import (jit, lax, pmap, value_and_grad, tree_map, device_count,
                 numpy as jnp)
from jax_sgmc import data
import numpy as onp
import optax

from chemtrain import util, data_processing, dropout


def pmap_update_fn(loss_fn, optimizer, n_devices):
    """Initializes a pmapped function for updating parameters.

    Usage:
    params, opt_state, loss, grad = update_fn(params, opt_state, batch)

    For pmap, params and opt_state need to be N_devices times duplicated along
    axis 0. Batch is reshaped by this function.

    Args:
        loss_fn: Loss function to minimize that takes (params, batch) as input.
        optimizer: Optax optimizer
        n_devices: Number of devices

    Returns:
        A function that computes the gradient and updates the parameters via the
        optimizer.
    """
    @partial(pmap, axis_name='devices')
    def pmap_batch_update(params, opt_state, batch):
        loss, grad = value_and_grad(loss_fn)(params, batch)

        # step optimizer within pmap to minimize communication overhead
        grad = lax.pmean(grad, axis_name='devices')
        loss = lax.pmean(loss, axis_name='devices')
        new_params, opt_state = step_optimizer(params, opt_state, grad,
                                               optimizer)
        return new_params, opt_state, loss, grad

    @jit
    def batch_update(params, opt_state, batch):
        batch = util.tree_split(batch, n_devices)
        new_params, opt_state, loss, grad = pmap_batch_update(params, opt_state,
                                                              batch)
        return new_params, opt_state, loss[0], util.tree_get_single(grad)
    return batch_update


def predict_val_data(model, val_loader, batch_size=1, batch_cache=100):
    """Model predictions for whole validation/test dataset.

    Can be used to monitor validation loss as well as postprocessing.
    """
    init_fun, map_fun = data.full_reference_data(val_loader, batch_cache,
                                                 batch_size)
    init_data_state = init_fun()

    pmap_model = pmap(model)

    def batched_model(params, batch, unused_state):
        batch = util.tree_split(batch, device_count())
        loss = pmap_model(params, batch)
        return loss, unused_state

    @jit
    def mapped_model_fn(params, data_state):
        data_state, (predictions, _) = map_fun(partial(batched_model, params),
                                               data_state, None)
        # correct for masked samples:
        # Loss function averages over batch_size, which we undo to divide
        # by the real number of samples.
        if isinstance(batch_losses, dict):
            average = {key: jnp.sum(values) * batch_size / n_val_samples
                       for key, values in batch_losses.items()}
        else:
            average = jnp.sum(batch_losses) * batch_size / n_val_samples
        return average, data_state
    return mapped_model_fn, init_data_state


def val_loss_fn(loss_fn, val_loader, batch_size=1, batch_cache=100):
    """Initializes a pmapped loss function.

    The loss function can be used for evaluating the validation or test
    loss via a full_data_map.

    This function is not equipped to deal with padded atoms.  # TODO

    Usage:
    val_loss, data_state = batched_loss_fn(params, data_state)

    For pmap, params needs to be N_devices times duplicated along axis 0.
    The data batch is reshaped accordingly by this function.

    Args:
        loss_fn: Loss function that takes (params, batch) as input.
        val_loader: NumpyDataLoader for validation set
        batch_size: Mini-batch size
        batch_cache: Number of batches to cache on GPU to reduce host-device
                     communication

    Returns:
        A pmapped function that returns the loss value.
    """
    n_val_samples = val_loader._observation_count
    init_fun, map_fun = data.full_reference_data(val_loader, batch_cache,
                                                 batch_size)
    init_data_state = init_fun()

    @partial(pmap, axis_name='devices')
    def pmap_loss_fn(*args, **kwargs):
        loss = loss_fn(*args, **kwargs)
        loss = lax.pmean(loss, axis_name='devices')
        return loss

    def batched_loss(params, batch, mask, unused_scan_carry):
        n_devices = device_count()
        batch = util.tree_split(batch, n_devices)
        mask = util.tree_split(mask, n_devices)
        # unused_scan_carry = util.tree_replicate(unused_scan_carry, n_devices)
        loss = pmap_loss_fn(params, batch, mask)
        return util.tree_get_single(loss), unused_scan_carry

    @jit
    def mapped_loss_fn(params, data_state):
        data_state, (batch_losses, _) = map_fun(partial(batched_loss, params),
                                                data_state, None, masking=True)
        # correct for masked samples:
        # Loss function averages over batch_size, which we undo to divide
        # by the real number of samples.
        if isinstance(batch_losses, dict):
            average = {key: jnp.sum(values) * batch_size / n_val_samples
                       for key, values in batch_losses.items()}
        else:
            average = jnp.sum(batch_losses) * batch_size / n_val_samples
        return average, data_state

    return mapped_loss_fn, init_data_state


def _masked_loss(per_element_loss, mask=None):
    """Computes average loss, accounting for masked elements, if applicable."""
    if mask is None:
        return jnp.mean(per_element_loss)
    else:
        assert mask.shape == per_element_loss.shape, ('Mask requires same shape'
                                                      ' as targets.')
        return jnp.sum(per_element_loss * mask) / jnp.sum(mask)


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
                if dropout.dropout_is_used(self.params):
                    params = dropout.next_dropout_params(self.params)
                    self.params = params

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
    def __init__(self, params, criterion, pq_window_size=5):
        """Initialize EarlyStopping.

        Args:
            criterion: Convergence criterion to employ
            pq_window_size: Window size for PQ method
        """
        self.criterion = criterion

        # own loss history that can be reset on the fly if needed.
        self._epoch_losses = []
        self.best_loss = 1.e16
        self.best_params = copy.copy(params)  # move on device, if loaded

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


class DataParallelTrainer(MLETrainerTemplate):
    """Trainer functionalities for MLE training based on a dataset, where
    parallelization can simply be accomplished by pmapping over batched data.
    """
    def __init__(self, dataset, loss_fn, init_params, optimizer,
                 checkpoint_path, batch_per_device, batch_cache,
                 train_ratio=0.7, val_ratio=0.1,
                 convergence_criterion='window_median',
                 energy_fn_template=None):
        n_devices = device_count()
        self._update_fn = pmap_update_fn(loss_fn, optimizer, n_devices)
        self.batch_size = batch_per_device * n_devices
        self.batch_cache = batch_cache
        self._loss_fn = loss_fn

        # replicate params and optimizer states for pmap
        opt_state = optimizer.init(init_params)  # initialize optimizer state
        init_params = util.tree_replicate(init_params)
        opt_state = util.tree_replicate(opt_state)
        init_state = util.TrainerState(params=init_params, opt_state=opt_state)

        super().__init__(
            optimizer=optimizer, init_state=init_state,
            checkpoint_path=checkpoint_path,
            reference_energy_fn_template=energy_fn_template)

        self.train_batch_losses, self.train_losses, self.val_losses = [], [], []
        self._early_stop = EarlyStopping(init_params, convergence_criterion)

        (self._batches_per_epoch, self._get_train_batch,
         self._train_batch_state, self.val_loader, self.test_loader
         ) = self._process_dataset(dataset, train_ratio, val_ratio)

        self._val_loss_fn, self._val_data_state = val_loss_fn(
            loss_fn, self.val_loader, self.batch_size, batch_cache
        )

    def update_dataset(self, train_ratio=0.1, val_ratio=0.1, **dataset_kwargs):
        """Allows changing dataset on the fly, which is particularly
        useful for active learning applications.

        Args:
            train_ratio: Percantage of dataset to use for training.
            val_ratio: Percantage of dataset to use for validation.
            **dataset_kwargs: Kwargs to supply to self._build_dataset to
                      re-build the dataset
        """
        # reset convergence criterion as loss might not be comparable
        self._early_stop.reset_convergence_losses()
        dataset = self._build_dataset(**dataset_kwargs)
        (self._batches_per_epoch, self._get_train_batch,
         self._train_batch_state, self.val_loader, self.test_loader
         ) = self._process_dataset(dataset, train_ratio, val_ratio)

        self._val_loss_fn, self._val_data_state = val_loss_fn(
            self._loss_fn, self.val_loader, self.batch_size,
            self.batch_cache
        )

    def _process_dataset(self, dataset, train_ratio=0.7, val_ratio=0.1):
        train_loader, val_loader, test_loader = \
            data_processing.init_dataloaders(dataset, train_ratio, val_ratio)
        init_train_state, get_train_batch = data.random_reference_data(
            train_loader, self.batch_cache, self.batch_size)
        train_batch_state = init_train_state()

        observation_count = train_loader._observation_count
        batches_per_epoch = observation_count // self.batch_size
        return (batches_per_epoch, get_train_batch, train_batch_state,
                val_loader, test_loader)

    def _get_batch(self):
        for _ in range(self._batches_per_epoch):
            self._train_batch_state, train_batch = self._get_train_batch(
                self._train_batch_state)
            yield train_batch

    def _update(self, batch):
        """Function to iterate, optimizing parameters and saving
        training and validation loss values.
        """
        params, opt_state, train_loss, curr_grad = self._update_fn(
            self.state.params, self.state.opt_state, batch)

        self.state = self.state.replace(params=params, opt_state=opt_state)
        self.train_batch_losses.append(train_loss)  # only from single device

        self.gradient_norm_history.append(util.tree_norm(curr_grad))

    def _evaluate_convergence(self, duration, thresh):
        """Prints progress, saves best obtained params and signals converged if
        validation loss improvement over the last epoch is less than the thesh.
        """
        mean_train_loss = sum(self.train_batch_losses[-self._batches_per_epoch:]
                              ) / self._batches_per_epoch
        self.train_losses.append(mean_train_loss)

        val_loss, self._val_data_state = self._val_loss_fn(self.state.params,
                                                           self._val_data_state)
        self.val_losses.append(val_loss)
        print(f'Epoch {self._epoch}: Average train loss: {mean_train_loss:.5f} '
              f'Average val loss: {val_loss:.5f} Gradient norm:'
              f' {self.gradient_norm_history[-1]}'
              f' Elapsed time = {duration:.3f} min')

        self._converged = self._early_stop.early_stopping(val_loss, thresh,
                                                          self.params)

    @abc.abstractmethod
    def _build_dataset(self, *args, **kwargs):
        """Function that returns the dataset as a Dictionary for the specific
        problem at hand. The data for each leaf of the dataset is assumed to be
        stacked along axis 0.
        """

    @property
    def params(self):
        single_params = util.tree_get_single(self.state.params)
        return single_params

    @params.setter
    def params(self, loaded_params):
        replicated_params = util.tree_replicate(loaded_params)
        self.state = self.state.replace(params=replicated_params)

    @property
    def best_params(self):
        return self._early_stop.best_params

    def move_to_device(self):
        super().move_to_device()
        self._early_stop.move_to_device()
