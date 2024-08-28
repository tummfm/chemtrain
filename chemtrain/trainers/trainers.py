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


"""This file contains several Trainer classes as a quickstart for users."""
import functools
import os
import pickle
import time
import warnings
from os import PathLike
from typing import Any, Mapping, Dict, Callable

import jax.tree_util
import numpy as onp
from jax import numpy as jnp, tree_util, jit
from jax_sgmc.data import numpy_loader
from jax_md_mod import custom_quantity

from chemtrain import (util)
from chemtrain.learn import (
    force_matching, max_likelihood, difftre, property_prediction
)
from chemtrain.quantity import property_prediction
from chemtrain.trainers import base as tt
from chemtrain.ensemble import sampling
from chemtrain.data import data_loaders


try:
    from jax.typing import ArrayLike
except:
    ArrayLike = Any
from optax import GradientTransformationExtraArgs
from jax_md.partition import NeighborFn, NeighborList
from chemtrain.typing import EnergyFnTemplate, TrajFn

class PropertyPrediction(tt.DataParallelTrainer):
    """Trainer for direct prediction of molecular properties."""
    def __init__(self, error_fn, prediction_model, init_params, optimizer,
                 graph_dataset, targets, batch_per_device=1, batch_cache=10,
                 train_ratio=0.7, val_ratio=0.1, test_error_fn=None,
                 shuffle=False, convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):
        # TODO documentation

        # TODO build graph on-the-fly as memory moving might be bottleneck here
        model = property_prediction.init_model(prediction_model)
        checkpoint_path = 'output/property_prediction/' + str(checkpoint_folder)
        loss_fn = property_prediction.init_loss_fn(error_fn)

        super().__init__(
            loss_fn, model, init_params, optimizer, checkpoint_path,
            batch_per_device, batch_cache,
            convergence_criterion=convergence_criterion
        )

        dataset_dict, _ = property_prediction.build_dataset(targets, graph_dataset)
        self.set_datasets(
            dataset_dict, train_ratio=train_ratio, val_ratio=val_ratio,
            shuffle=shuffle
        )

        self.test_error_fn = test_error_fn

    def predict(self, single_observation):
        """Prediction for a single input graph using the current param state."""
        batched_observation = tree_util.tree_map(
            functools.partial(jnp.expand_dims, axis=0), single_observation
        )
        batched_prediction = self.batched_model(
            self.best_inference_params, batched_observation)
        single_prediction = tree_util.tree_map(
            functools.partial(jnp.squeeze, axis=0), batched_prediction
        )
        return single_prediction

    def evaluate_testset_error(self, best_params=True):
        assert "testing" in self._batch_states.keys(), (
            'No test set available. Check train and val ratios.'
        )
        assert self._test_fn is not None, (
            '"test_error_fn" is necessary during initialization.'
        )

        params = (self.best_inference_params_replicated
                  if best_params else self.state.params)

        error = self.evaluate("testing", self._test_fn, params=params)

        print(f'Error on test set: {error}')
        return error


class ForceMatching(tt.DataParallelTrainer):
    """Force-matching trainer.

    Args:
        init_params: Initial energy parameters.
        energy_fn_template: Function that takes energy parameters and returns
            an energy function.
        nbrs_init: Initial neighbor list. The neighbor list must be large enough
            to not overflow for any sample of the dataset.
        optimizer: Optimizer from optax.
        position_data: Position data of the system.
        gammas: Coefficients for the individual targets in the weighted loss.
        weights_keys: Dictionary to entries of the dataset that contain a
            per-sample weight for the total loss.
        additional_targets: Additional snapshot targets to train on. Forces
            and energy are derived automatically from the energy_fn_template.
        feature_extract_fns: Features to extract from the data, passed to
            all snapshot functions as keyword arguments.
        batch_per_device: Number of samples to process vectorized on every
            device.
        batch_cache: Number of batches to load into the device memories.
        full_checkpoint: Save the whole trainer instead of only some statistics.
        disable_shmap: Use ``pmap`` instead of ``shmap`` for parallelization.
        penalty_fn: Penalty depending only on the parameters.
        convergence_criterion: Check convergence via
            :class:`base.EarlyStopping`.
        checkpoint_path: Path to the folder to store checkpoints.

    Warning:
        Currently neighborlist overflow is not checked.
        Make sure to build nbrs_init large enough.

    """
    def __init__(self,
                 init_params,
                 optimizer,
                 energy_fn_template: EnergyFnTemplate,
                 nbrs_init: NeighborList,
                 gammas: Dict[str, float] = None,
                 weights_keys: Dict[str, str] = None,
                 additional_targets: Dict[str, Dict] = None,
                 feature_extract_fns: Dict[str, Callable] = None,
                 batch_per_device: int = 1,
                 batch_cache: int = 10,
                 full_checkpoint: bool = False,
                 disable_shmap: bool = False,
                 penalty_fn: Callable = None,
                 convergence_criterion: str = 'window_median',
                 checkpoint_path: PathLike = 'checkpoints'):
        # Add additional trainable targets
        if gammas is None:
            gammas = {}

        # This feature extractor enables to evaluate the energy function
        # only once for all computations involving the energy and forces.
        feature_fns = {
            "energy_and_force": custom_quantity.energy_force_wrapper(
                energy_fn_template
            )
        }

        # These are common quantities to train on. The energy function is not
        # necessary, since forces and energy are pre-extracted
        quantities = {
            "F": custom_quantity.force_wrapper(None),
            "U": custom_quantity.energy_wrapper(None)
        }

        if additional_targets is not None:
            quantities.update(additional_targets)
        if feature_extract_fns is not None:
            feature_fns.update(feature_extract_fns)

        model = force_matching.init_model(
            nbrs_init, quantities, feature_extract_fns=feature_fns
        )

        loss_fn = force_matching.init_loss_fn(
            gammas=gammas, weights_keys=weights_keys)

        super().__init__(loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         disable_shmap=disable_shmap, penalty_fn=penalty_fn,
                         convergence_criterion=convergence_criterion,
                         full_checkpoint=full_checkpoint,
                         energy_fn_template=energy_fn_template)

        self._nbrs_init = nbrs_init

    def evaluate_mae_testset(self):
        """Prints the Mean Absolute Error for every target on the test set."""
        mae_loss_fn = force_matching.init_loss_fn(
                max_likelihood.mae_loss, individual=True
            )

        _, maes = self.evaluate(
            "testing", mae_loss_fn, params=self.best_inference_params
        )

        for key, mae_value in maes.items():
            print(f'{key}: MAE = {mae_value:.4f}')


class Difftre(tt.PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method.

    The trainer initialization only sets the initial trainer state
    as well as checkpointing and save-functionality. For training,
    target state points with respective simulations need to be added
    via :func:`Difftre.add_statepoint`.

    Args:
        init_params: Initial energy parameters
        optimizer: Optimizer from optax
        reweight_ratio: Ratio of reference samples required for n_eff to
            surpass to allow re-use of previous reference trajectory state.
            If trajectories should not be re-used, a value > 1 can be
            specified.
        sim_batch_size: Number of state-points to be processed as a single
            batch. Gradients will be averaged over the batch before stepping the
            optimizer.
        energy_fn_template: Function that takes energy parameters and
            initializes a new energy function. Here, the energy_fn_template
            is only a reference that will be saved alongside the trainer.
            Each state point requires its own due to the dependence on the
            box size via the displacement function, which can vary between
            state points.
        convergence_criterion: Either 'max_loss' or 'ave_loss'.
            If 'max_loss', stops if the maximum loss across all batches in
            the epoch is smaller than convergence_thresh. 'ave_loss'
            evaluates the average loss across the batch. For a single state
            point, both are equivalent. A criterion based on the rolling
            standatd deviation 'std' might be implemented in the future.
        checkpoint_folder: Name of folders to store ckeckpoints in.

    Attributes:
        weights_fn: Dictionary containing the reweighting functions for each
            statepoint.
        batch_losses: List of losses for each batch in each epoch.
        epoch_losses: List of losses for each epoch.
        step_size_history: List of step sizes for each batched update.
        gradient_norm_history: List of gradient norms for each batched update.
        predictions: Dictionary containing the predictions for each statepoint
            at each epoch.
        early_stop: Instance of EarlyStopping to check for convergence.

    Examples:

        .. code-block :: python

            trainer = trainers.Difftre(init_params, optimizer)

            # Add all statepoints
            trainer.add_statepoint(energy_fn_template, simulator_template,
                                   neighbor_fn, timings, statepoint_dict,
                                   compute_fns, reference_state, targets)
            ...

            # Optionally initialize the step size adaption
            trainer.init_step_size_adaption(allowed_reduction=0.5)

            trainer.train(num_updates)

    """

    def __init__(self,
                 init_params: Any,
                 optimizer: GradientTransformationExtraArgs,
                 reweight_ratio: ArrayLike = 1.0,
                 sim_batch_size: int = 1,
                 energy_fn_template: EnergyFnTemplate = None,
                 full_checkpoint: bool = False,
                 convergence_criterion: str = 'window_median',
                 checkpoint_path: os.PathLike = 'Checkpoints'):
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))

        # Optional: Initialized by calling trainer.init_step_size_adaption
        # after all statepoints to be considered have been set up.
        self._adaptive_step_size = None
        self._adaptive_step_size_threshold = None
        self._recompute = False

        self.state_dicts = {}
        self.weights_fn = {}
        self.targets = {}
        super().__init__(
            init_trainer_state=init_state, optimizer=optimizer,
            checkpoint_path=checkpoint_path, reweight_ratio=reweight_ratio,
            sim_batch_size=sim_batch_size, full_checkpoint=full_checkpoint,
            energy_fn_template=energy_fn_template)

        self.batch_losses = self.checkpoint("batch_losses", [])
        self.epoch_losses = self.checkpoint("epoch_losses", [])
        self.step_size_history = self.checkpoint("step_size_history", [])
        self.gradient_norm_history = self.checkpoint("gradient_norm_history", [])
        self.predictions = self.checkpoint("predictions", {})

        self.early_stop = tt.EarlyStopping(self.params,
                                        convergence_criterion)

    def add_statepoint(self,
                       energy_fn_template: EnergyFnTemplate,
                       simulator_template: Callable,
                       neighbor_fn: NeighborFn,
                       timings: sampling.TimingClass,
                       state_kwargs: Dict[str, ArrayLike],
                       quantities: Dict[str, Dict],
                       reference_state,
                       targets: Dict[str, Any] = None,
                       observables: Dict[str, TrajFn] = None,
                       target_loss_fns: Dict[str, Callable] = None,
                       loss_fn = None,
                       vmap_batch: int = 10,
                       initialize_traj: bool = True,
                       set_key: str = None,
                       entropy_approximation: bool = False,
                       resample_simstates: bool = False):
        """
        Adds a state point to the pool of simulations with respective targets.

        Each statepoints initializes a new gradient and propagation function via
        :func:`chemtrain.learn.difftre.init_difftre_gradient_and_propagation`.

        Args:
            energy_fn_template: Function that takes energy parameters and
                initializes a new energy function.
            simulator_template: Function that takes an energy function and
                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                about the trajectory length and which states to retain
            state_kwargs: Properties defining the thermodynamic state. Must
                at least contain the temperature 'kT'. For a non-exhaustive
                list, see :class:`chemtrain.ensemble.templates.StatePoint`.
            quantities: Dict containing for each observable specified by the
                key a corresponding function to compute it for each snapshot
                using traj_util.quantity_traj.
            reference_state: Tuple of initial simulation state and neighbor list
            targets: Dict containing the same keys as quantities and containing
                another dict providing 'gamma' and 'target' for each observable.
                Targets are only necessary when using the 'independent_loss_fn'.
            observables: Optional dictionary providing the observable functions
                for the targets. This is only necessary when the observable
                functions are not already contained in the targets dict.
            target_loss_fns: Optional dictionary providing the loss functions
                for the individual targets. This is only necessary when the
                loss functions are not already contained in the targets dict
                or should be different from the MSE loss.
            loss_fn: Custom loss function taking the trajectory of quantities
                and weights and returning the loss and predictions;
                By default, initializes an independent MSE loss, which computes
                reweighting averages from snapshot-based observables.
                In many applications, the default loss function will be
                sufficient. For a description, see
                :func:`chemtrain.learn.difftre.init_default_loss_fn`.
            vmap_batch: Batch size of vmapping of per-snapshot energy for weight
                computation.
            initialize_traj: True, if an initial trajectory should be generated.
                Should only be set to False if a checkpoint is loaded before
                starting any training.
            set_key: Specify a key in order to restart from same statepoint.
                By default, uses the index of the sequance statepoints are
                added, i.e. self.trajectory_states[0] for the first added
                statepoint.
                Can be used for changing the timings of the simulation during
                training.
            resample_simstates: Resample the sim states from all trajectories
                instead of simulating independent chains.

        """
        # init simulation, reweighting functions and initial trajectory
        (key, *reweight_fns) = self._init_statepoint(
            reference_state,
            energy_fn_template,
            simulator_template,
            neighbor_fn,
            timings,
            state_kwargs,
            set_key,
            vmap_batch,
            initialize_traj,
            safe_propagation=False,
            entropy_approximation=entropy_approximation,
            resample_simstates=resample_simstates
        )

        # For backwards compatibility and ease of use for a single statepoint
        if observables is None:
            observables = {
                key: target['traj_fn'] for key, target in targets.items()
            }
        if target_loss_fns is None:
            target_loss_fns = {
                key: target['loss_fn'] for key, target in targets.items()
                if 'loss_fn' in target
            }

        # Enables a greater flexibility by sorting out data from frunctions
        targets = {
            key: {k: v for k, v in target.items() if k in ['gamma', 'target']}
            for key, target in targets.items() if target.get('target') is not None
        }

        # build loss function for current state point
        if loss_fn is None:
            loss_fn = difftre.init_default_loss_fn(observables, target_loss_fns)
        else:
            print('Using custom loss function. Ignoring "target" dict.')

        difftre_grad_and_propagation = difftre.init_difftre_gradient_and_propagation(
            reweight_fns, loss_fn, quantities, energy_fn_template
        )

        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point
        self.weights_fn[key] = jax.jit(reweight_fns[0])
        self.state_dicts[key] = state_kwargs
        self.targets[key] = targets

        # Reset loss measures if new state point es added since loss values
        # are not necessarily comparable
        self.early_stop.reset_convergence_losses()

    def _update(self, batch):
        """Computes gradient averaged over the sim_batch by propagating
        respective state points. Additionally saves predictions and loss
        for postprocessing."""
        # TODO parallelization? Maybe lift batch requirement and only
        #  sync sporadically?
        # https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
        # https://github.com/mpi4jax/mpi4jax
        # TODO split gradient and loss computation from stepping optimizer for
        #  building hybrid trainers?

        # TODO is there good way to reuse this function in BaseClass?

        # Note: in principle, we could move all the use of instance attributes
        # into difftre_grad_and_propagation, which would increase re-usability
        # with relative_entropy. However, this would probably stop all
        # parallelization efforts

        losses = 0.0
        grads = None

        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]
            (new_traj_state, loss_val, curr_grad,
             state_point_predictions) = grad_fn(
                self.params, self.trajectory_states[sim_key],
                self.state_dicts[sim_key], self.targets[sim_key],
                recompute=self._recompute
            )

            self.trajectory_states[sim_key] = new_traj_state
            self.predictions[sim_key][self._epoch] = tree_util.tree_map(
                onp.asarray, state_point_predictions)

            losses += loss_val
            if grads is None:
                grads = curr_grad
            else:
                grads = util.tree_sum(grads, curr_grad)

            if jnp.isnan(loss_val):
                warnings.warn(f'Loss of state point {sim_key} in epoch '
                              f'{self._epoch} is NaN. This was likely caused by'
                              f' divergence of the optimization or a bad model '
                              f'setup causing a NaN trajectory.')
                self._diverged = True  # ends training
                break

        self.batch_losses.append(onp.asarray(losses / len(batch)))
        batch_grad = tree_util.tree_map(lambda x: x / len(batch), grads)

        if self._adaptive_step_size is not None:
            proposal = self._optimizer_step(batch_grad)
            alpha, residual = self._adaptive_step_size(
                self.params, batch_grad, proposal, self.trajectory_states)

            # Recompute all traj states if step size scale is too small
            self._recompute = alpha < self._adaptive_step_size_threshold
            print(f"[Step Size] Found optimal step size {alpha} with residual "
                  f"{residual}", flush=True)
        else:
            alpha = 1.0

        self._step_optimizer(batch_grad, alpha=alpha)

        batch_norm = util.tree_norm(batch_grad)
        self.gradient_norm_history.append(onp.asarray(batch_norm))
        self.step_size_history.append(onp.asarray(alpha))

    def init_step_size_adaption(self,
                                allowed_reduction: ArrayLike = 0.5,
                                interior_points: int = 10,
                                step_size_scale: float = 1e-5,
                                recompute_threshold: float = 1e-4
                                ) -> None:
        """Initializes a line search to tune the step size in each iteration.

        The line search optimizes step size to limit the decrease in the
        effective sample size (ESS) via the algorithm
        :func:`chemtrain.learn.difftre.init_step_size_adaption`.

        Args:
            allowed_reduction: Target reduction of the effective sample size
            interior_points: Number of interiour points
            step_size_scale: Accuracy of the found optimal interpolation
                coefficient
            recompute_threshold: Recompute the trajectory if found step size
                scale is below the threshold

        Returns:
            Returns the interpolation coefficient :math:`\\alpha`.

        """

        self._adaptive_step_size_threshold = recompute_threshold
        self._adaptive_step_size = difftre.init_step_size_adaption(
            self.weights_fn, allowed_reduction, interior_points, step_size_scale
        )

    def _evaluate_convergence(self, *args, thresh=None, **kwargs):
        # sim_batch_size = -1 means all statepoints are processed in one batch.
        if self.sim_batch_size < 0:
            batches_per_epoch = 1
        else:
            batches_per_epoch = self.n_statepoints // self.sim_batch_size

        last_losses = jnp.array(self.batch_losses[-batches_per_epoch:])
        epoch_loss = jnp.mean(last_losses)
        duration = self.update_times[self._epoch]
        self.epoch_losses.append(epoch_loss)

        print(
            f'\n[DiffTRe] Epoch {self._epoch}'
            f'\n\tEpoch loss = {epoch_loss:.5f}'
            f'\n\tGradient norm: {self.gradient_norm_history[-1]}'
            f'\n\tElapsed time = {duration:.3f} min')

        # Print scalar predictions
        for statepoint, prediction_series in self.predictions.items():
            self._print_measured_statepoint(sim_key=statepoint)
            last_predictions = prediction_series[self._epoch]
            for quantity, value in last_predictions.items():
                if value.ndim == 0:
                    print(f'\tPredicted {quantity}: {value}')

        self._converged = self.early_stop.early_stopping(
            epoch_loss, thresh, self.params)

    @property
    def best_params(self):
        """Returns the best parameters according to the early stopping criterion."""
        return self.early_stop.best_params

    def move_to_device(self):
        """Transforms the trainer states to JAX arrays."""
        super().move_to_device()
        self.early_stop.move_to_device()


class DifftreActive(tt.TrainerInterface):
    """Active learning of state-transferable potentials from experimental data
    via DiffTRe.

    The input trainer can be pre-trained or freshly initialized. Pre-training
    usually comes with the advantage that the initial training from random
    parameters is usually the most unstable one. Hence, special care can be
    taken such as training on NVT initially to fix the pressure and swapping
    to NPT afterwards. This active learning trainer then takes care of learning
    statepoint transferability.
    """
    def __init__(self, trainer, checkpoint_folder='Checkpoints',
                 energy_fn_template=None):
        checkpoint_path = 'output/difftre_active/' + str(checkpoint_folder)
        super().__init__(checkpoint_path, energy_fn_template)
        self.trainer = trainer
        # other inits

    def add_statepoint(self, *args, **kwargs):
        """Add another statepoint to the target state points.

        Predominantly used to add statepoints with more / different targets
        not covered in  the on-the-fly tepoint addition, e.g. for an extensive
        initial statepoint. Please refer to :obj:'Difftre.add_statepoint
        <chemtrain.trainers.Difftre.add_statepoint>' for the full documentation.
        """
        self.trainer.add_statepoint(*args, **kwargs)

    def train(self, max_new_statepoints=100):
        for added_statepoints in range(max_new_statepoints):
            accuracy_met = False
            if accuracy_met:
                print('Visited state space covered with accuracy target met.')
                break

            # checkpoint: call checkpoint of trainer
        else:
            warnings.warn('Maximum number of added statepoints added without '
                          'reaching target accuracy over visited state space.')

    @property
    def params(self):
        return self.trainer.params

    @params.setter
    def params(self, loaded_params):
        self.trainer.params = loaded_params


class RelativeEntropy(tt.PropagationBase):
    """Trainer for relative entropy minimization.

    The relative entropy algorithm currently assume a NVT ensemble.

    Args:
        init_params: Initial energy parameters.
        optimizer: Optimizer from optax.
        reweight_ratio: Ratio of reference samples required for n_eff to
            surpass to allow re-use of previous reference trajectory state.
            If trajectories should not be re-used, a value > 1 can be specified.
        sim_batch_size: Number of state-points to be processed as a single
            batch. Gradients will be averaged over the batch before stepping the
            optimizer.
        energy_fn_template: Function that takes energy parameters and
            initializes an new energy function. Here, the ``energy_fn_template``
            is only a reference that will be saved alongside the trainer.
            Each state point requires its own due to the dependence on the box
            size via the displacement function, which can vary between state points.
        convergence_criterion: Either ``'max_loss'`` or ``'ave_loss'``.
            If ``'max_loss'``, stops if the gradient norm cross all batches in
            the epoch is smaller than convergence_thresh.
            ``'ave_loss'`` evaluates  the average gradient norm across the batch.
            For a single state point, both are equivalent.
        checkpoint_path: Path to the folder to store ckeckpoints in.
        full_checkpoint: Save the whole trainer instead of only the inference
            data.

    Attributes:
        data_states: Dictionary containing the dataloader states for each
            state points.
        delta_re: Dictionary containing the improvement of the relative entropy
            with respect to the initial potential.
        step_size_history: List of step size scales for each batched update.
        gradient_norm_history: List of gradient norms for each batched update.
        weight_fn: Dictionary containing the reweighting functions for each
            statepoint.
        early_stop: Instance of EarlyStopping to check for convergence.

    """
    def __init__(self,
                 init_params,
                 optimizer,
                 reweight_ratio: float = 0.9,
                 sim_batch_size: int = 1,
                 energy_fn_template: EnergyFnTemplate = None,
                 convergence_criterion: str = 'window_median',
                 checkpoint_path: os.PathLike = 'Checkpoints',
                 full_checkpoint: bool = False):
        init_trainer_state = util.TrainerState(
            params=init_params, opt_state=optimizer.init(init_params))
        super().__init__(init_trainer_state, optimizer, checkpoint_path,
                         reweight_ratio, sim_batch_size, energy_fn_template,
                         full_checkpoint)

        # in addition to the standard trajectory state, we also need to keep
        # track of dataloader states for reference snapshots
        self.data_states = {}
        self.delta_re = self.checkpoint("delta_re", {})
        self.step_size_history = self.checkpoint("step_size_history", [])
        self.gradient_norm_history = self.checkpoint("gradient_norm_history", [])

        self.weight_fn = {}
        self.early_stop = tt.EarlyStopping(self.params, convergence_criterion)

        self._adaptive_step_size = None

    def _set_dataset(self, key, reference_data, reference_batch_size,
                     batch_cache=1):
        """Set dataset and loader corresponding to current state point."""
        reference_loader = numpy_loader.NumpyDataLoader(
            R=reference_data, copy=False)
        init_ref_batch, get_ref_batch, _ = data_loaders.init_batch_functions(
            data_loader=reference_loader, mb_size=reference_batch_size,
            cache_size=batch_cache
        )
        init_reference_batch_state = init_ref_batch(shuffle=True)
        self.data_states[key] = init_reference_batch_state
        return get_ref_batch

    def add_statepoint(self,
                       reference_data: ArrayLike,
                       energy_fn_template: EnergyFnTemplate,
                       simulator_template: Callable,
                       neighbor_fn: NeighborFn,
                       timings: sampling.TimingClass,
                       state_kwargs: Dict[str, ArrayLike],
                       reference_state,
                       reference_batch_size: int = None,
                       batch_cache: int = 1,
                       initialize_traj: bool = True,
                       set_key: str = None,
                       vmap_batch: int = 10,
                       resample_simstates: bool = False):
        """
        Adds a state point to the pool of simulations.

        The gradient of the relative entropy is computed via the gradient
        function initialized by
        :func:`chemtrain.learn.difftre.init_rel_entropy_gradient_and_propagation`.

        As each reference dataset / trajectory corresponds to a single
        state point, we initialize the dataloader together with the
        simulation.

        Currently only supports NVT simulations.

        Args:
            reference_data: De-correlated reference trajectory
            energy_fn_template: Function that takes energy parameters and
                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                about the trajectory length and which states to retain
            state_kwargs: Properties defining the thermodynamic state. Must
                at least contain the temperature 'kT'.
            reference_state: Tuple of initial simulation state and neighbor list
            reference_batch_size: Batch size of dataloader for reference
                trajectory. If None, will use the same number of snapshots as
                generated via the optimizer.
            batch_cache: Number of reference batches to cache in order to
                minimize host-device communication. Make sure the cached data
                size does not exceed the full dataset size.
            initialize_traj: True, if an initial trajectory should be generated.
                Should only be set to False if a checkpoint is loaded before
                starting any training.
            set_key: Specify a key in order to restart from same statepoint.
                By default, uses the index of the sequance statepoints are
                added, i.e. ``self.trajectory_states[0]`` for the first added
                statepoint. Can be used for changing the timings of the
                simulation during training.
            vmap_batch: Batch size of vmapping of per-snapshot energy and
                gradient calculation.
        """
        if reference_batch_size is None:
            print('No reference batch size provided. Using number of generated '
                  'CG snapshots by default.')
            states_per_traj = jnp.size(timings.t_production_start)
            if reference_state.sim_state.position.ndim > 2:
                n_trajectories = reference_state.sim_state.position.shape[0]
                reference_batch_size = n_trajectories * states_per_traj
            else:
                reference_batch_size = states_per_traj

        (key, *reweight_fns) = self._init_statepoint(reference_state,
                                                     energy_fn_template,
                                                     simulator_template,
                                                     neighbor_fn,
                                                     timings,
                                                     state_kwargs,
                                                     set_key,
                                                     vmap_batch,
                                                     initialize_traj,
                                                     entropy_approximation=False,
                                                     resample_simstates=resample_simstates,
                                                     safe_propagation=False)

        reference_dataloader = self._set_dataset(key,
                                                 reference_data,
                                                 reference_batch_size,
                                                 batch_cache)

        propagation_and_grad = difftre.init_rel_entropy_gradient_and_propagation(
            reference_dataloader, reweight_fns, energy_fn_template,
            state_kwargs['kT'], vmap_batch
        )

        self.grad_fns[key] = propagation_and_grad
        self.delta_re[key] = []
        self.weight_fn[key] = jax.jit(reweight_fns[0])

    def init_step_size_adaption(self,
                                allowed_reduction: ArrayLike = 0.5,
                                interior_points: int = 10,
                                step_size_scale: float = 1e-7
                                ) -> None:
        """Initializes a line search to tune the step size in each iteration.

        The line search optimizes step size to limit the decrease in the
        effective sample size (ESS) via the algorithm
        :func:`chemtrain.learn.difftre.init_step_size_adaption`.

        Args:
            allowed_reduction: Target reduction of the effective sample size
            interior_points: Number of interiour points
            step_size_scale: Accuracy of the found optimal interpolation
                coefficient

        Returns:
            Returns the interpolation coefficient :math:`\\alpha`.

        """

        self._adaptive_step_size = difftre.init_step_size_adaption(
            self.weight_fn, allowed_reduction, interior_points, step_size_scale
        )

    def _update(self, batch):
        """Updates the potential using the gradient from relative entropy."""
        grads = []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]

            self.trajectory_states[sim_key], delta_re, curr_grad, \
            self.data_states[sim_key] = grad_fn(self.params,
                                                self.trajectory_states[sim_key],
                                                self.data_states[sim_key])
            grads.append(curr_grad)
            self.delta_re[sim_key].append(delta_re)


        batch_grad = util.tree_mean(grads)

        if self._adaptive_step_size is not None:
            proposal = self._optimizer_step(batch_grad)
            alpha, residual = self._adaptive_step_size(
                self.params, batch_grad, proposal, self.trajectory_states)
            print(f"[Step Size] Found optimal step size {alpha} with residual "
                  f"{residual}", flush=True)
        else:
            alpha = 1.0

        self._step_optimizer(batch_grad, alpha=alpha)

        batch_norm = util.tree_norm(batch_grad)
        self.gradient_norm_history.append(onp.asarray(batch_norm))
        self.step_size_history.append(onp.asarray(alpha))


    def _evaluate_convergence(self, *args, thresh=None, **kwargs):
        curr_grad_norm = self.gradient_norm_history[-1]
        # Mean loss from last simbatch
        mean_delta_re = onp.mean(
            [delta_re[-1] for delta_re in self.delta_re.values()]
        )
        duration = self.update_times[self._epoch]

        print(
            f'\n[RE] Epoch {self._epoch}'
            f'\n\tMean Delta RE loss = {mean_delta_re:.5f}'
            f'\n\tGradient norm: {curr_grad_norm}'
            f'\n\tElapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        self._converged = self.early_stop.early_stopping(
            curr_grad_norm, thresh, save_best_params=False)


class SGMCForceMatching(tt.ProbabilisticFMTrainerTemplate):
    """Trainer for stochastic gradient Markov-chain Monte Carlo training
    based on force-matching.

    init_samples: A list, possibly of size 1, of sets of initial MCMC samples,
     where each spawns a dedicated MCMC chain,
    """
    def __init__(self, sgmc_solver, init_samples, val_dataloader=None,
                 energy_fn_template=None):
        # TODO: Where does alias.py get checkpoint_path info?
        super().__init__(None, energy_fn_template)
        self._params = [init_sample['params'] for init_sample in init_samples]
        self.sgmcmc_run_fn = sgmc_solver
        self.init_samples = init_samples

        # TODO use val dataloader to compute posterior predictive p value or
        #  other convergence metric. In ProbabilisticFMTrainerTemplate??

        # TODO also use test_set?

    def train(self, iterations):
        """Training of any trainer should start by calling train."""
        self.results = self.sgmcmc_run_fn(*self.init_samples,
                                          iterations=iterations)

    @property
    def params(self):
        """Get the sampled parameters from all chains."""
        if len(self.results) == 1:  # single chain
            return self.results[0]['samples']['variables']['params']
        else:
            params = []
            for chain in self.results:
                params.append(chain['samples']['variables']['params'])
            stacked_params = util.tree_stack(params)
            return util.tree_combine(stacked_params)

    @params.setter
    def params(self, loaded_params):
        raise NotImplementedError('Setting params seems not meaningful in'
                                  ' the case of SG-MCMC samplers.')

    @property
    def list_of_params(self):
        """A list of the sampled parameters."""
        return util.tree_unstack(self.params)

    def save_trainer(self, save_path):
        """Save the trainer to a file."""
        raise NotImplementedError('Saving the trainer currently does not work'
                                  ' for SGMCMC.')


class EnsembleOfModels(tt.ProbabilisticFMTrainerTemplate):
    """Train an ensemble of models by starting optimization from different
    initial parameter sets, for use in uncertainty quantification applications.

    Example:

        .. code-block:: python

           trainer_list = []
           for i in range(4):
               trainer_list.append(trainers.ForceMatching(...))
           trainer_ensemble = trainers.EnsembleOfModels(trainer_list)

           trainer_ensemble.train(*args, **kwargs)
           trained_params = trainer_ensemble.list_of_params

    """
    def __init__(self, trainers, ref_energy_fn_template=None):
        super().__init__(None, ref_energy_fn_template)
        self.trainers = trainers

    def train(self, *args, **kwargs):
        for i, trainer in enumerate(self.trainers):
            print(f'---------Starting trainer {i}-----------')
            trainer.train(*args, **kwargs)
        print('Finished training all models.')

    @property
    def params(self):
        return util.tree_stack(self.list_of_params)

    @params.setter
    def params(self, loaded_params):
        for i, params in enumerate(loaded_params):
            self.trainers[i].params = params

    @property
    def list_of_params(self):
        params = []
        for trainer in self.trainers:
            if hasattr(trainer, 'best_params'):
                params.append(trainer.best_params)
            else:
                params.append(trainer.params)
        return params


class InterleaveTrainers(tt.TrainerInterface):
    """Interleaves updates to train models using multiple algorithms.

    This special trainer allows to train models simultaneously with different
    algorithms.

    Example:

        .. code-block::

            # First initialize the base-trainers, e.g.
            fm_trainer = trainers.ForceMatching(...)

            difftre_trainer = trainers.Difftre(...)
            difftre_trainer.add_statepoint(...)

            # Now combine the trainers. The trainers are executed in the
            # order in which they are added

            trainer = trainers.InterleaveTrainers('checkpoint_folder',
                                                  energy_fn_template,
                                                  full_checkpoint=False)

            # Force matching should run 10 epochs before difftre runs 2 epochs
            trainer.add_trainer(fm_trainer, num_updates=10, name='Force Matching')
            trainer.add_trainer(difftre_trainer, num_updates=2, name='DiffTRe')

            trainer.train(100, checkpoint_frequency=10)

    Args:
        sequential: Start the next trainer directly with the optimized
            parameters of the previous trainer. In the non-sequential case,
            the trainers start their epoch on the same parameter set and
            the final update is a weighted sum of both updates.
        checkpoint_base_path: Location to store checkpoints of the trainers.
        reference_energy_fn_template: Energy function template to optionally
            return an energy function with current parameters.
        full_checkpoint: Store the complete trainer or important properties
            only.

    """


    def __init__(self,
                 sequential = True,
                 checkpoint_base_path = 'checkpoints',
                 reference_energy_fn_template=None,
                 full_checkpoint=False):
        super().__init__(checkpoint_base_path, reference_energy_fn_template,
                         full_checkpoint)
        self.sequential = sequential
        self._trainers = []
        self._epoch = 0

    def add_trainer(self, trainer, num_updates: int = 1, name: str = "trainer",
                    weight: float = 1.0, **trainer_kwargs):
        """Adds a trainer to the combined training.

        The trainers are executed in the order they are added to this instance.
        It is possible to specify how many epochs each trainer should train
        before the next trainer starts again.

        Args:
            trainer: Trainer to add to the chain.
            num_updates: Consecutive updates of the trainer in one epoch of the
                interleaved trainer.
            name: Display name of the trainer.
            weight: Weight for the interpolated update of the parameters.
            trainer_kwargs: Additional arguments for the training method
                of the trainer.

        """
        self._trainers.append(
            {'trainer': trainer, 'num_updates': num_updates, 'name': name,
             'kwargs': trainer_kwargs, 'weight': weight}
        )

    @property
    def params(self):
        return self._trainers[-1]['trainer'].params

    @params.setter
    def params(self, params):
        for trainer in self._trainers:
            trainer['trainer'].params = params

    @property
    def _all_params(self):
        return [t['trainer'].params for t in self._trainers]

    @property
    def _all_weights(self):
        return [t['weight'] for t in self._trainers]

    def _init_interpolated_update(self):
        weights = jnp.asarray(self._all_weights)
        weights /= jnp.sum(weights)
        @jit
        def update(parameters):
            # Scale the parameters
            structure = tree_util.tree_structure(parameters[0])
            leaves = [tree_util.tree_leaves(t) for t in parameters]
            concat = [jnp.concatenate(l) for l in zip(*leaves)]
            summed = [jnp.sum(weights * l, axis=0) for l in concat]
            return tree_util.tree_unflatten(structure, summed)
        return update

    def train(self, epochs, checkpoint_frequency=None):
        """Train model with combined algorithms.

        Args:
            epochs: Number of epochs, where one epoch can contain multiple
                epochs for each added trainer.
            checkpoint_frequency: Save a checkpoint in the given frequency.

        """
        interpolated_update = self._init_interpolated_update()
        self._converged = False
        start_epoch = self._epoch
        end_epoch = start_epoch + epochs
        for e in range(start_epoch, end_epoch):
            start = time.time()
            for t, trainer in enumerate(self._trainers):
                print(f'---------Starting trainer {trainer["name"]} for {trainer["num_updates"]} updates -----------')
                trainer['trainer'].train(trainer['num_updates'], **trainer['kwargs'])

                next = (t + 1) % len(self._trainers)

                if self.sequential:
                    # Pass updated parameters to the next trainer
                    self._trainers[next]['trainer'].params = trainer['trainer'].params
            if not self.sequential:
                # Update the parameters of all trainers with a weighted sum of
                # the individual parameters
                self.params = interpolated_update(self.params)

            duration = (time.time() - start) / 60.
            self._epoch += 1
            print(f'Finished epoch {e} for all trainers in {duration : .2f} minutes.')
            self._dump_checkpoint_occasionally(frequency=checkpoint_frequency)

    def move_to_device(self):
        for trainer in self._trainers:
            trainer['trainer'].move_to_device()

    def save_trainer(self, save_path, format='.pkl'):
        data = {}
        for t, trainer in enumerate(self._trainers):
            number = str(t + 1).rjust(3, '0')
            key = 'trainer_{0}_{1}'.format(trainer['name'], number)
            data[key] = trainer['trainer'].save_trainer(None, format='none')

        if format == '.pkl':
            with open(save_path, 'wb') as pickle_file:
                pickle.dump(data, pickle_file)
        elif format == 'none':
            return data
