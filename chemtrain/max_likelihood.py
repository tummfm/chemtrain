# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of functions to facilitate learning maximum likelihood /
 single point estimate models.
 """
import abc
import copy
from functools import partial
import time

from jax import (lax, vmap, pmap, value_and_grad, tree_map, device_count,
                 numpy as jnp)
from jax_sgmc import data
import numpy as onp
import optax

from chemtrain import util, data_processing, dropout
from chemtrain.pickle_jit import jit


def pmap_update_fn(batched_model, loss_fn, optimizer):
    """Initializes a pmapped function for updating parameters.

    Usage:
    params, opt_state, loss, grad = update_fn(params, opt_state, batch)

    Loss and grad are only a single instance, no n_device replica.
    Params and opt_state need to be N_devices times duplicated along axis 0.
    Batch is reshaped by this function.

    Args:
        batched_model: A model with signature model(params, batch), which
                       predicts a batch of outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
                 value for a batch.
        optimizer: Optax optimizer

    Returns:
        A function that computes the gradient and updates the parameters via the
        optimizer.
    """
    # loss as function of params and batch for optimization.
    def param_loss_fn(params, batch):
        predictions = batched_model(params, batch)
        loss = loss_fn(predictions, batch)
        return loss

    @partial(pmap, axis_name='devices')
    def pmap_batch_update(params, opt_state, batch):
        loss, grad = value_and_grad(param_loss_fn)(params, batch)

        # step optimizer within pmap to minimize communication overhead
        grad = lax.pmean(grad, axis_name='devices')
        loss = lax.pmean(loss, axis_name='devices')
        new_params, opt_state = step_optimizer(params, opt_state, grad,
                                               optimizer)
        return new_params, opt_state, loss, grad

    @jit
    def batch_update(params, opt_state, batch):
        batch = util.tree_pmap_split(batch, device_count())
        new_params, opt_state, loss, grad = pmap_batch_update(params, opt_state,
                                                              batch)
        return new_params, opt_state, loss[0], util.tree_get_single(grad)
    return batch_update


def init_val_predictions(batched_model, val_loader, batch_size=1,
                         batch_cache=10):
    """Model predictions for whole validation/test dataset.

    Usage:
    predictions, data_state = mapped_model_fn(params, data_state)

    Params needs to be N_devices times duplicated along axis 0.

    Args:
        batched_model: A model with signature model(params, batch), which
                       predicts a batch of outputs used in loss function.
        val_loader: Validation or test set NumpyDataLoader.
        batch_size: Total batch size that is processed in parallel
        batch_cache: Number of batches to cache.

    Returns:
        Tuple (predictions, data_state). predictions contains model predictions
        for the whole validation dataset and data_state is used to start the
        data loading in the next evaluation.
    """
    # case where validation data is very small
    batch_size = min(val_loader.static_information['observation_count']
                     // device_count(), batch_size)
    map_fun, data_release = data.full_data_mapper(val_loader, batch_cache,
                                                  batch_size)

    pmap_model = pmap(batched_model)

    def single_batch(params, batch, unused_state):
        batch = util.tree_pmap_split(batch, device_count())
        batch_prediction = pmap_model(params, batch)
        predictions = util.tree_concat(batch_prediction)
        return predictions, unused_state

    @jit
    def mapped_model_fn(params):
        predictions, _ = map_fun(partial(single_batch, params), None)
        return predictions
    return mapped_model_fn, data_release


def init_val_loss_fn(model, loss_fn, val_loader, val_targets_keys=None,
                     batch_size=1, batch_cache=100):
    """Initializes a pmapped loss function that computes the validation loss.

    Usage:
    val_loss, data_state = batched_loss_fn(params, data_state)

    Params needs to be N_devices times duplicated along axis 0.

    Args:
        model: A model with signature model(params, batch), which predicts
               outputs used in loss function.
        loss_fn: Loss function(predictions, targets) returning the scalar loss
                 value for a batch.
        val_loader: NumpyDataLoader for validation set.
        val_targets_keys: Dict containing targets of whole val
        batch_size: Total batch size that is processed in parallel.
        batch_cache: Number of batches to cache on GPU to reduce host-device
                     communication.

    Returns:
        A pmapped function that returns the average validation loss.
    """

    # We compute the validation error over the whole dataset at once, because
    # otherwise it is non-trivial to compute the correct error for masked
    # batches with different number of masked targets without explicitly knowing
    # the mask in this function
    # If predictions and targets of the whole validation dataset does not fit
    # memory, a more specialized approach needs to be taken.

    if val_targets_keys is None:
        target_data = val_loader._reference_data
    else:
        target_data = {key: val_loader._reference_data[key]
                       for key in val_targets_keys}

    mapped_predictions_fn, data_release_fn = init_val_predictions(
        model, val_loader, batch_size, batch_cache)

    def mapped_loss_fn(params):
        predictions = mapped_predictions_fn(params)
        val_loss = loss_fn(predictions, target_data)
        return val_loss

    return mapped_loss_fn, data_release_fn


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
        self._dropout = dropout.dropout_is_used(self.params)
        # Note: make sure not to use such a construct during training as an
        # if-statement based on params forces the python part to wait for the
        # completion of the batch, hence losing the advantage of asynchronous
        # dispatch, which can become the bottleneck in high-throughput learning.

        self.release_fns = []

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
                self._update_with_dropout(batch)
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

    def _update_with_dropout(self, batch):
        """Updates params, while keeping track of Dropout."""
        self._update(batch)
        if self._dropout:
            # TODO refactor this as this needs to wait for when
            #  params will again be available, slowing down re-loading
            #  of batches. We could set dropout key as kwarg and keep
            #  track of keys in this class. Also refactor dropout in
            #  DimeNet taking advantage of haiku RNG key management and
            #  built-in dropout in MLP
            params = dropout.next_dropout_params(self.params)
            self.params = params

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

    def _release_data_references(self):
        for release in self.release_fns:
            release()
        self.release_fns = []


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
            params: Optimization parameters to save in case of being best. Make
                    sure to supply non-devive-replicated params,
                    i.e. self.params.
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
    As pmap requires constant batch dimensions, data with unequal number of
    atoms needs to be padded and to be compatible with this trainer.
    """
    def __init__(self, dataset_dict, loss_fn, model, init_params, optimizer,
                 checkpoint_path, batch_per_device, batch_cache,
                 train_ratio=0.7, val_ratio=0.1, shuffle=False,
                 convergence_criterion='window_median',
                 energy_fn_template=None):
        self.model = model
        self.batched_model = vmap(model, in_axes=(None, 0))
        self._update_fn = pmap_update_fn(self.batched_model, loss_fn, optimizer)
        self.batch_cache = batch_cache
        self._loss_fn = loss_fn
        self.batch_size = batch_per_device * device_count()

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
        self._early_stop = EarlyStopping(self.params, convergence_criterion)

        (self._batches_per_epoch, self._get_train_batch,
         self._train_batch_state, self.train_loader, self.val_loader,
         self.test_loader, self.target_keys
         ) = self._process_dataset(dataset_dict, train_ratio, val_ratio,
                                   shuffle=shuffle)

        if self.val_loader is not None:  # no validation dataset
            self._val_loss_fn, data_release_fn = init_val_loss_fn(
                self.batched_model, self._loss_fn, self.val_loader,
                self.target_keys, self.batch_size, self.batch_cache
            )
            self.release_fns.append(data_release_fn)

    def update_dataset(self, train_ratio=0.7, val_ratio=0.1, shuffle=False,
                       **dataset_kwargs):
        """Allows changing dataset on the fly, which is particularly
        useful for active learning applications.

        Args:
            train_ratio: Percentage of dataset to use for training.
            val_ratio: Percentage of dataset to use for validation.
            shuffle: Whether to shuffle data before splitting into
                     train-val-test.
            **dataset_kwargs: Kwargs to supply to self._build_dataset to
                      re-build the dataset
        """
        # reset convergence criterion as loss might not be comparable
        self._early_stop.reset_convergence_losses()

        # release all references before allocating new data to avoid memory leak
        self._release_data_references()

        (self._batches_per_epoch, self._get_train_batch,
         self._train_batch_state, self.train_loader, self.val_loader,
         self.test_loader, target_keys
         ) = self._process_dataset(dataset_kwargs, train_ratio, val_ratio,
                                   shuffle=shuffle)

        if self.val_loader is not None:
            self._val_loss_fn, data_release_fn = init_val_loss_fn(
                self.batched_model, self._loss_fn, self.val_loader, target_keys,
                self.batch_size, self.batch_cache
            )
            self.release_fns.append(data_release_fn)

    def _process_dataset(self, dataset_dict, train_ratio=0.7, val_ratio=0.1,
                         shuffle=False):
        # considers case of re-training with different number of GPUs
        dataset, target_keys = self._build_dataset(**dataset_dict)
        train_loader, val_loader, test_loader = \
            data_processing.init_dataloaders(dataset, train_ratio, val_ratio,
                                             shuffle=shuffle)
        init_train_state, get_train_batch, release = data.random_reference_data(
             train_loader, self.batch_cache, self.batch_size)
        self.release_fns.append(release)
        train_batch_state = init_train_state(shuffle=True)

        observation_count = train_loader.static_information['observation_count']
        batches_per_epoch = observation_count // self.batch_size
        return (batches_per_epoch, get_train_batch, train_batch_state,
                train_loader, val_loader, test_loader, target_keys)

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

        if self.val_loader is not None:
            val_loss = self._val_loss_fn(self.state.params)
            self.val_losses.append(val_loss)
            self._converged = self._early_stop.early_stopping(val_loss, thresh,
                                                              self.params)
        else:
            val_loss = None
        print(f'Epoch {self._epoch}: Average train loss: {mean_train_loss:.5f} '
              f'Average val loss: {val_loss} Gradient norm:'
              f' {self.gradient_norm_history[-1]}'
              f' Elapsed time = {duration:.3f} min')

    def update_with_samples(self, **sample):
        """A single params update step, where a batch is taken from the training
        set and samples of the batch are substituted by the provided samples.

        This function is useful in an active learning context to retrain
        specifically on newly labeled datapoints. The number of provided samples
        must not exceed the trainer batch size.

        Args:
            **sample: Kwargs storing data samples to supply to
                      self._build_dataset to build samples in the correct
                      pytree. Analogous usage as update_dataset, but the dataset
                      only consists of a few observations.
        """
        ordered_samples, _ = self._build_dataset(**sample)
        n_samples = util.tree_multiplicity(ordered_samples)
        assert n_samples <= self.batch_size, ('Number of provided samples must'
                                              ' not exceed trainer batch size.')
        batch = next(self._get_batch())
        updated_batch = util.tree_set(batch, ordered_samples, n_samples)
        self._update_with_dropout(updated_batch)

    def set_testloader(self, testdata):
        """Set testloader to use provided test data.

        Args:
            **testdata: Kwargs storing the sample daata to supply to
                        self._build_dataset to build the sample in the correct
                        pytree. Analogous usage as update_dataset, but the
                        dataset only consists of a single observation.
        """
        dataset, _ = self._build_dataset(**testdata)
        _, _, test_loader = data_processing.init_dataloaders(dataset, 0., 0.)
        self.test_loader = test_loader
        self._init_test_fn()  # re-init test function to adjust to new data set

    @abc.abstractmethod
    def _build_dataset(self, *args, **kwargs):
        """Function that returns a tuple (dataset, target_keys).
        The 'dataset' is a dictionary for the specific problem at hand.
        The data for each leaf of the dataset is assumed to be stacked along
        axis 0. 'target_keys' is a list of keys that are necessary to evaluate
        the loss_fn, assuming the model prediction is available. In the simplest
        case, the same keys as in 'dataset' can be provided. For a memory
        expensive dataset, keys that are only needed as model input can be
        omitted to save GPU memory.
        """

    @abc.abstractmethod
    def _init_test_fn(self):
        """Function that sets self._test_fn and self._test_state to evalaute
        test set loss.
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
        #  if no validation data given, _early_stop.best_params are simply
        #  init_params
        if self.val_loader is None:
            return self.params
        else:
            return self._early_stop.best_params

    @property
    def best_inference_params(self):
        """Returns best model params irrespective whether dropout is used."""
        if dropout.dropout_is_used(self.best_params):
            # all nodes present during inference
            params, _ = dropout.split_dropout_params(self.best_params)
        else:
            params = self.best_params
        return params

    @property
    def best_inference_params_replicated(self):
        inference_params = self.best_inference_params
        return util.tree_replicate(inference_params)

    def move_to_device(self):
        super().move_to_device()
        self._early_stop.move_to_device()
