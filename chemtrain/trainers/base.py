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

"""Abstract templates for trainers, defining common functionality and
requirements."""
import abc
import copy
import pathlib
import time
from abc import abstractmethod
from os import PathLike

import cloudpickle as pickle
import numpy as onp
from jax import tree_map, numpy as jnp, random, vmap, device_count, jit
from jax_sgmc import data

from chemtrain import util
from chemtrain.data import data_processing
from chemtrain.learn.max_likelihood import pmap_update_fn, \
    init_val_loss_fn, step_optimizer, shmap_update_fn
from chemtrain.potential import dropout
from chemtrain.trajectory.reweighting import init_pot_reweight_propagation_fns
from chemtrain.typing import EnergyFnTemplate
from chemtrain.util import format_not_recognized_error


class TrainerInterface(metaclass=abc.ABCMeta):
    """Abstract class defining the user interface of trainers as well as
    checkpointing functionality.
    """
    # TODO write protocol classes for better documentation of initialized
    #  functions
    def __init__(self,
                 checkpoint_path,
                 reference_energy_fn_template=None,
                 full_checkpoint=True):
        """A reference energy_fn_template can be provided, but is not mandatory
        due to the dependence of the template on the box via the displacement
        function.
        """
        self._statistics = {}
        self._full_checkpoint = full_checkpoint
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
                epoch = str(self._epoch).rjust(5, '0')
                file_path = (
                    pathlib.Path(self.checkpoint_path) / f'epoch{epoch}.pkl')
                self.save_trainer(file_path)

    def save_trainer(self, save_path, format='.pkl'):
        """Saves whole trainer, e.g. for production after training."""
        if self._full_checkpoint:
            data = self
        else:
            data = self._statistics
            try:
                self._statistics["trainer_state"] = self.state
            except AttributeError:
                print(f"Skipping trainer state")

        if format == '.pkl':
            with open(save_path, 'wb') as pickle_file:
                pickle.dump(data, pickle_file)
        elif format == 'none':
            return data

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
        """Short-cut for parameters. Depends on specific trainer."""

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

    def checkpoint(self, name, object):
        """Marks attribute to be saved in a partial checkpoint.

        This requires that the object to be saved is only mutated but not
        replaced during training.

        Args:
            name: Name of the statistic in the saved dictionary
            object: Mutable object to be saved via pickle

        Returns:
            Returns the original object unchanged.

        """
        self._statistics.update({name: object})
        return object


class MLETrainerTemplate(TrainerInterface):
    """Abstract class implementing common properties and methods of single
    point estimate Trainers using optax optimizers.

    Args:
        optimizer: Optax optimizer
        init_state: Initial state of optimizer and model
        checkpoint_path: Path to folder where checkpoints are saved
        full_checkpoint: Whether to save the full trainer with pickle or only
            a subset of attributes.
        reference_energy_fn_template: Function returning a concrete energy
            function for the current parameters

    Attributes:
        update_times: Computation time of each update
        gradient_norm_history: Norms of the gradient for each update

    """

    def __init__(self,
                 optimizer,
                 init_state: util.TrainerState,
                 checkpoint_path: PathLike,
                 full_checkpoint: bool = True,
                 reference_energy_fn_template: EnergyFnTemplate = None):
        super().__init__(
            checkpoint_path, reference_energy_fn_template, full_checkpoint)
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

    def _optimizer_step(self, curr_grad):
        """Wrapper around step_optimizer that is useful whenever the
        update of the optimizer can be done outside of jit-compiled functions.

        Returns:
            Returns the parameters after an update of the optimizer, but
            without updating the internal states.
        """
        new_params, _ = step_optimizer(
            self.params, self.state.opt_state, curr_grad, self.optimizer)
        return new_params

    def _step_optimizer(self, curr_grad, alpha=1.0):
        """Wrapper around step_optimizer that is useful whenever the
        update of the optimizer can be done outside of jit-compiled functions.
        """
        new_params, new_opt_state = step_optimizer(self.params,
                                                   self.state.opt_state,
                                                   curr_grad,
                                                   self.optimizer)

        # Do an optimized update
        new_params = tree_map(
            lambda old, new: old * (1 - alpha) + new * alpha,
            self.params, new_params
        )
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
            try:
                for batch in self._get_batch():
                    self._update_with_dropout(batch)
                self._dump_checkpoint_occasionally(frequency=checkpoint_freq)
                duration = (time.time() - start_time) / 60.
                self.update_times.append(duration)
                self._evaluate_convergence(duration, thresh)
                self._epoch += 1
            except RuntimeError as err:
                # In case the simulation diverges, break the optimization
                # and checkpoint the last state such that an analysis can
                # be performed.
                self._diverged = True
                if self.checkpoint_path is not None:
                    path = (self.checkpoint_path
                            + f'/epoch{self._epoch - 1}_error_state.pkl')
                    self.save_trainer(save_path=path)
                print(f'Training has been unsuccessful due to the following'
                      f' error: {err}')
                break


            if self._converged:
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


class PropagationBase(MLETrainerTemplate):
    """Trainer base class for shared functionality whenever (multiple)
    simulations are run during training. Can be used as a template to
    build other trainers. Currently used for DiffTRe and relative entropy.

    We only save the latest generated trajectory for each state point.
    While accumulating trajectories would enable more frequent reweighting,
    this effect is likely minor as past trajectories become exponentially
    less useful with changing potential. Additionally, saving long trajectories
    for each statepoint would increase memory requirements over the course of
    the optimization.
    """
    def __init__(self, init_trainer_state, optimizer, checkpoint_path,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None,
                 full_checkpoint=True, key=None):
        super().__init__(optimizer, init_trainer_state, checkpoint_path,
                         full_checkpoint, energy_fn_template)
        self.sim_batch_size = sim_batch_size
        self.reweight_ratio = reweight_ratio

        if key is None:
            self.key = random.PRNGKey(0)

        # store for each state point corresponding traj_state and grad_fn
        # save in distinct dicts as grad_fns need to be deleted for checkpoint
        self.grad_fns, self.trajectory_states, self.statepoints = {}, {}, {}
        self.n_statepoints = 0
        self.shuffle_key = random.PRNGKey(0)

    def _init_statepoint(self, reference_state, energy_fn_template,
                         simulator_template, neighbor_fn, timings, kbt,
                         set_key=None, energy_batch_size=10,
                         initialize_traj=True, ref_press=None,
                         safe_propagation=True, entropy_approximation=False,
                         replica_kbt=None, num_chains=None):
        """Initializes the simulation and reweighting functions as well
        as the initial trajectory for a statepoint."""
        # TODO ref pressure only used in print and to have barostat values.
        #  Reevaluate this parameter of barostat values not used in reweighting
        # TODO document ref_press accordingly

        if set_key is not None:
            key = set_key
            if set_key not in self.statepoints.keys():
                self.n_statepoints += 1
        else:
            key = self.n_statepoints
            self.n_statepoints += 1
        self.statepoints[key] = {'kbT': kbt}
        npt_ensemble = util.is_npt_ensemble(reference_state[0])
        if npt_ensemble: self.statepoints[key]['pressure'] = ref_press

        if replica_kbt is not None:
            assert reference_state[0].position.ndim > 2, (
                "Replica exchange requires multiple simulator states")

        gen_init_traj, *reweight_fns = init_pot_reweight_propagation_fns(
            energy_fn_template, simulator_template, neighbor_fn, timings,
            kbt, ref_press, self.reweight_ratio, npt_ensemble,
            energy_batch_size, safe_propagation=safe_propagation,
            entropy_approximation=entropy_approximation,
            replica_kbt=replica_kbt, num_chains=num_chains
        )
        if initialize_traj:
            self.key, split = random.split(self.key)
            init_traj, runtime = gen_init_traj(
                split, self.params, reference_state)
            print(f'Time for trajectory initialization {key}: {runtime} mins')
            self.trajectory_states[key] = init_traj
        else:
            print('Not initializing the initial trajectory is only valid if '
                  'a checkpoint is loaded. In this case, please be use to add '
                  'state points in the same sequence, otherwise loaded '
                  'trajectories will not match its respective simulations.')

        return key, *reweight_fns

    @abstractmethod
    def add_statepoint(self, *args, **kwargs):
        """User interface to add additional state point to train model on."""
        raise NotImplementedError()

    @property
    def params(self):
        return self.state.params

    @params.setter
    def params(self, loaded_params):
        self.state = self.state.replace(params=loaded_params)

    def get_sim_state(self, key):
        return self.trajectory_states[key].sim_state

    def _get_batch(self):
        """Helper function to re-shuffle simulations and split into batches."""
        self.shuffle_key, used_key = random.split(self.shuffle_key, 2)
        shuffled_indices = random.permutation(used_key, self.n_statepoints)
        if self.sim_batch_size == 1:
            batch_list = jnp.split(shuffled_indices, shuffled_indices.size)
        elif self.sim_batch_size == -1:
            batch_list = jnp.split(shuffled_indices, 1)
        else:
            raise NotImplementedError('Only batch_size = 1 or -1 implemented.')

        return (batch.tolist() for batch in batch_list)

    def _print_measured_statepoint(self):
        """Print meausured kbT (and pressure for npt ensemble) for all
        statepoints to ensure the simulation is indeed carried out at the
        prescribed state point.
        """
        for sim_key, traj in self.trajectory_states.items():
            print(f'[Statepoint {sim_key}]')
            statepoint = self.statepoints[sim_key]
            measured_kbt = jnp.mean(traj.aux['kbT'])
            if 'pressure' in statepoint:  # NPT
                measured_press = jnp.mean(traj.aux['pressure'])
                press_print = (f'\n\tpress = {measured_press:.2f} ref_press = '
                               f'{statepoint["pressure"]:.2f}')
            else:
                press_print = ''
            print(f'\tkbT = {measured_kbt:.3f} ref_kbt = '
                  f'{statepoint["kbT"]:.3f}' + press_print)

    def train(self, max_epochs, thresh=None, checkpoint_freq=None):
        assert self.n_statepoints > 0, ('Add at least 1 state point via '
                                        '"add_statepoint" to start training.')
        super().train(max_epochs, thresh=thresh,
                      checkpoint_freq=checkpoint_freq)

    @abstractmethod
    def _update(self, batch):
        """Implementation of gradient computation, stepping of the optimizer
        and logging of auxiliary results. Takes batch of simulation indices
        as input.
        """


class DataParallelTrainer(MLETrainerTemplate):
    """Trainer for parallelized MLE training based on a dataset.

    This trainer implements methods for MLE training on a dataset, where
    parallelization can simply be accomplished by pmapping over batched data.
    As pmap requires constant batch dimensions, data with unequal number of
    atoms needs to be padded and to be compatible with this trainer.
    """

    def __init__(self, dataset_dict, loss_fn, model, init_params, optimizer,
                 checkpoint_path, batch_per_device, batch_cache,
                 train_ratio=0.7, val_ratio=0.1, shuffle=False,
                 full_checkpoint=True, penalty_fn=None,
                 convergence_criterion='window_median',
                 energy_fn_template=None,
                 disable_shmap: bool = False):
        self.model = model
        self.batched_model = vmap(model, in_axes=(None, 0))
        if disable_shmap:
            self._update_fn = pmap_update_fn(
                self.batched_model, loss_fn, optimizer, penalty_fn)
        else:
            # shmap performs better, but some replication rules are missing
            self._update_fn = shmap_update_fn(
                self.batched_model, loss_fn, optimizer, penalty_fn)
        self.batch_cache = batch_cache
        self._loss_fn = loss_fn
        self.batch_size = batch_per_device * device_count()

        # replicate params and optimizer states for pmap
        opt_state = optimizer.init(init_params)  # initialize optimizer state
        init_state = util.TrainerState(params=init_params, opt_state=opt_state)

        super().__init__(
            optimizer=optimizer, init_state=init_state,
            checkpoint_path=checkpoint_path,
            full_checkpoint=full_checkpoint,
            reference_energy_fn_template=energy_fn_template)

        self.train_batch_losses = self.checkpoint('train_batch_losses', [])
        self.train_losses = self.checkpoint('train_losses', [])
        self.val_losses = self.checkpoint('val_losses', [])

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
            dataset_kwargs: Kwargs to supply to ``self._build_dataset`` to
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
            sample: Kwargs storing data samples to supply to
                ``self._build_dataset`` to build samples in the correct
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
            testdata: Kwargs storing the sample daata to supply to
                ``self._build_dataset`` to build the sample in the correct
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
        single_params = self.state.params
        return single_params

    @params.setter
    def params(self, loaded_params):
        self.state = self.state.replace(params=loaded_params)

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


class ProbabilisticFMTrainerTemplate(TrainerInterface):
    """Trainer template for methods that result in multiple parameter sets for
    Monte-Carlo-style uncertainty quantification, based on a force-matching
    formulation.
    """
    def __init__(self, checkpoint_path, energy_fn_template,
                 val_dataloader=None):
        super().__init__(checkpoint_path, energy_fn_template)
        self.results = []

        # TODO use val_loader for some metrics that are interesting for MCMC
        #  and SG-MCMC

    def move_to_device(self):
        params = []
        for param_set in self.params:
            params.append(tree_map(jnp.array, param_set))  # move on device
        self.params = params

    @property
    @abc.abstractmethod
    def list_of_params(self):
        """ Returns a list containing n single model parameter sets, where n
        is the number of samples. This provides a more intuitive parameter
        interface that self.params, which returns a large set of parameters,
        where n is the leading axis of each leaf. Self.params is most useful,
        if parameter sets are mapped via map or vmap in a postprocessing step.
        """


class MCMCForceMatchingTemplate(ProbabilisticFMTrainerTemplate):
    """Initializes log_posterior function to be used for MCMC with blackjax,
    including batch-wise evaluation of the likelihood and re-materialization.
    """
    def __init__(self, init_state, kernel, checkpoint_path, val_loader=None,
                 ref_energy_fn_template=None):
        super().__init__(checkpoint_path, ref_energy_fn_template, val_loader)
        self.kernel = jit(kernel)
        self.state = init_state

    def train(self, num_samples, checkpoint_freq=None, init_samples=None,
              rng_key=random.PRNGKey(0)):
        if init_samples is not None:
            # TODO implement multiple chains
            raise NotImplementedError

        for i in range(num_samples):
            start_time = time.time()
            rng_key, consumed_key = random.split(rng_key)
            self.state, info = self.kernel(consumed_key, self.state)
            self.results.append(self.state)
            print(f'Time for sample {i}: {(time.time() - start_time) / 60.}'
                  f' min.', info)
            self._epoch += 1
            self._dump_checkpoint_occasionally(frequency=checkpoint_freq)

    @property
    def list_of_params(self):
        return [state.position['params'] for state in self.results]

    @property
    def params(self):
        return util.tree_stack(self.list_of_params)

    @params.setter
    def params(self, loaded_params):
        raise NotImplementedError('Setting params seems not meaningful for MCMC'
                                  ' samplers.')


class EarlyStopping:
    """A class that saves the best parameter obtained so far based on the
    validation loss and determines whether the optimization can be stopped based
    on some stopping criterion.

    The following criteria are implemented:

        - ``'window_median'``: 2 windows are placed at the end of the loss
          history. Stops when the median of the latter window of size "thresh"
          exceeds the median of the prior window of the same size.

        - ``'PQ'``: Stops when the PQ criterion exceeds thresh

        - ``'max_loss'``: Stops when the loss decreased below the maximum allowed
          loss specified via thresh.

    Args:
        criterion: Convergence criterion to employ
        pq_window_size: Window size for PQ method

    Attributes:
        best_loss: Loss of the best performing parameters
        best_params: Parameters with best performance

    """
    def __init__(self, params, criterion, pq_window_size=5):
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
        """Estimates whether the convergence criterion was met and keeps track
        of the best parameters obtained so far.

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
        """Resets loss history used for convergence estimation, e.g., to avoid
        early stopping when loss increases due to on-the-fly changes in the
        dataset or the loss function.
        """
        self._epoch_losses = []
        self.best_loss = 1.e16
        self.best_params = None

    def move_to_device(self):
        """Moves best_params to device to use them after loading trainer."""
        self.best_params = tree_map(jnp.array, self.best_params)
