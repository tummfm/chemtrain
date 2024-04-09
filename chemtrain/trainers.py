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
import pickle
import time

import jax.tree_util
import warnings

from blackjax import nuts, stan_warmup
from jax import random, numpy as jnp, lax, tree_util
from jax_sgmc import data
from jax_sgmc.data import numpy_loader

import numpy as onp

from chemtrain.jax_md_mod import custom_quantity
from chemtrain import (util)
from chemtrain.quantity import property_prediction
from chemtrain.learn import force_matching, max_likelihood, probabilistic
from chemtrain.trajectory import reweighting, traj_util
from chemtrain.pickle_jit import jit

from typing import Any, Mapping, Dict
try:
    from jax.typing import ArrayLike
except:
    ArrayLike = Any
from optax import GradientTransformationExtraArgs
from jax_md.partition import NeighborFn
from chemtrain.typing import EnergyFnTemplate
from chemtrain.trajectory.traj_util import TrajectoryState, TimingClass

class PropertyPrediction(max_likelihood.DataParallelTrainer):
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
        dataset_dict = {'targets': targets, 'graph_dataset': graph_dataset}
        loss_fn = property_prediction.init_loss_fn(error_fn)

        super().__init__(dataset_dict, loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio, shuffle=shuffle,
                         convergence_criterion=convergence_criterion)

        self.test_error_fn = test_error_fn
        self._init_test_fn()

    @staticmethod
    def _build_dataset(targets, graph_dataset):
        return property_prediction.build_dataset(targets, graph_dataset)

    def predict(self, single_observation):
        """Prediction for a single input graph using the current param state."""
        # TODO jit somewhere?
        return self.model(self.best_inference_params, single_observation)

    def evaluate_testset_error(self, best_params=True):
        assert self.test_loader is not None, ('No test set available. Check'
                                              ' train and val ratios.')
        assert self._test_fn is not None, ('"test_error_fn" is necessary'
                                           ' during initialization.')

        params = (self.best_inference_params_replicated
                  if best_params else self.state.params)
        error = self._test_fn(params)
        print(f'Error on test set: {error}')
        return error

    def _init_test_fn(self):
        if self.test_error_fn is not None and self.test_loader is not None:
            test_loss_fn = property_prediction.init_loss_fn(self.test_error_fn)
            self._test_fn, data_release_fn = max_likelihood.init_val_loss_fn(
                self.batched_model, test_loss_fn, self.test_loader,
                self.target_keys, self.batch_size, self.batch_cache)
            self.release_fns.append(data_release_fn)
        else:
            self._test_fn = None


class ForceMatching(max_likelihood.DataParallelTrainer):
    """Force-matching trainer.

    This implementation assumes a constant number of particles per box and
    constant box sizes for each snapshot.
    If this is not the case, please use the ForceMatchingPrecomputed trainer
    based on padded sparse neighborlists.
    Caution: Currently neighborlist overflow is not checked.
    Make sure to build nbrs_init large enough.
    # TODO generalize to padded particles and without neighborlists

    Virial data is pressure tensor, i.e. negative stress tensor

    """
    def __init__(self, init_params, energy_fn_template, nbrs_init,
                 optimizer, position_data, energy_data=None, force_data=None,
                 virial_data=None, kt_data=None, box_tensor=None, gamma_f=1.,
                 gamma_p=1.e-6, batch_per_device=1, batch_cache=10,
                 train_ratio=0.7, val_ratio=0.1, shuffle=False,
                 full_checkpoint=False,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        dataset_dict = {'position_data': position_data,
                        'energy_data': energy_data,
                        'force_data': force_data,
                        'virial_data': virial_data,
                        'kt_data': kt_data
                        }

        virial_fn = force_matching.init_virial_fn(
            virial_data, energy_fn_template, box_tensor)
        model = force_matching.init_model(nbrs_init, energy_fn_template,
                                          virial_fn)
        loss_fn = force_matching.init_loss_fn(gamma_f=gamma_f, gamma_p=gamma_p)

        super().__init__(dataset_dict, loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch_per_device, batch_cache,
                         train_ratio, val_ratio, shuffle=shuffle,
                         convergence_criterion=convergence_criterion,
                         full_checkpoint=full_checkpoint,
                         energy_fn_template=energy_fn_template)
        self._virial_fn = virial_fn
        self._nbrs_init = nbrs_init
        self._init_test_fn()

    @staticmethod
    def _build_dataset(position_data, energy_data=None, force_data=None,
                       virial_data=None, kt_data=None):
        return force_matching.build_dataset(position_data, energy_data,
                                            force_data, virial_data, kt_data)

    def evaluate_mae_testset(self):
        assert self.test_loader is not None, ('No test set available. Check'
                                              ' train and val ratios or add a'
                                              ' test_loader manually.')
        maes = self.mae_fn(self.best_inference_params_replicated)
        for key, mae_value in maes.items():
            print(f'{key}: MAE = {mae_value:.4f}')

    def _init_test_fn(self):
        if self.test_loader is not None:
            self.mae_fn, data_release_fn = force_matching.init_mae_fn(
                self.test_loader, self._nbrs_init,
                self.reference_energy_fn_template, self.batch_size,
                self.batch_cache, self._virial_fn
            )
            self.release_fns.append(data_release_fn)
        else:
            self.mae_fn = None


class Difftre(reweighting.PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method.

    The implementation assumes a NVT ensemble in weight computation.
    The trainer initialization only sets the initial trainer state
    as well as checkpointing and save-functionality. For training,
    target state points with respective simulations need to be added
    via 'add_statepoint'.

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

    Examples:

        .. code-block :: python

            trainer = trainers.Difftre(init_params, optimizer)

            # Add all statepoints
            trainer.add_statepoint(energy_fn_template, simulator_template,
                                   neighbor_fn, timings, kbt, compute_fns,
                                   reference_state, targets)
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
                 energy_fn_template: EnergyFnTemplate=None,
                 full_checkpoint=False,
                 convergence_criterion: str = 'window_median',
                 checkpoint_folder = 'Checkpoints'):
        # TODO doc: beware that for too short trajectory might have overfittet
        #  to single trajectory; if in doubt, set reweighting ratio = 1 towards
        #  end of optimization
        checkpoint_path = 'output/difftre/' + str(checkpoint_folder)
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))

        # Optional: Initialized by calling trainer.init_step_size_adaption
        # after all statepoints to be considered have been set up.
        self._adaptive_step_size = None

        self.weights_fn = {}
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
        self.lagrange_multipliers: Dict[str, Any] = self.checkpoint("lagrange_multipliers", {})

        self.early_stop = max_likelihood.EarlyStopping(self.params,
                                                       convergence_criterion)

    def add_statepoint(self,
                       energy_fn_template: EnergyFnTemplate,
                       simulator_template,
                       neighbor_fn: NeighborFn,
                       timings: TimingClass,
                       kbt: ArrayLike,
                       quantities: Dict[str, Dict],
                       reference_state: TrajectoryState,
                       targets: Dict[str, Any] = None,
                       ref_press: ArrayLike = None,
                       loss_fn = None,
                       vmap_batch: int = 10,
                       initialize_traj: bool = True,
                       set_key: str = None,
                       loss_kwargs: Mapping = None,
                       entropy_approximation: bool = False,
                       replica_kbt: ArrayLike = None,
                       num_chains: int = None):
        """
        Adds a state point to the pool of simulations with respective targets.

        Requires own energy_fn_template and simulator_template to allow
        maximum flexibility for state points: Allows different ensembles
        (NVT vs NpT), box sizes and target quantities per state point.
        The quantity dict defines the way target observations
        contribute to the loss function. Each target observable needs to be
        saved in the quantity dict via a unique key. Model predictions will
        be output under the same key. In case the default loss function should
        be employed, for each observable the 'target' dict containing
        a multiplier controlling the weight of the observable
        in the loss function under 'gamma' as well as the prediction target
        under 'target' needs to be provided.

        In many applications, the default loss function will be sufficient.
        If a target observable cannot be described directly as an average
        over instantaneous quantities (e.g. stiffness),
        a custom loss_fn needs to be defined. The signature of the loss_fn
        needs to be the following: It takes the trajectory of computed
        instantaneous quantities saved in a dict under its respective key of
        the quantities_dict. Additionally, it receives corresponding weights
        w_i to perform ensemble averages under the reweighting scheme. With
        these components, ensemble averages of more complex observables can
        be computed. The output of the function is (loss value, predicted
        ensemble averages). The latter is only necessary for post-processing
        the optimization process. See 'init_independent_mse_loss_fn' for
        an example implementation.

        Args:
            energy_fn_template: Function that takes energy parameters and
                initializes a new energy function.
            simulator_template: Function that takes an energy function and
                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                about the trajectory length and which states to retain
            kbt: Temperature in kbT
            quantities: Dict containing for each observable specified by the
                key a corresponding function to compute it for each snapshot
                using traj_util.quantity_traj.
            reference_state: Tuple of initial simulation state and neighbor list
            targets: Dict containing the same keys as quantities and containing
                another dict providing 'gamma' and 'target' for each observable.
                Targets are only necessary when using the 'independent_loss_fn'.
            loss_fn: Custom loss function taking the trajectory of quantities
                and weights and returning the loss and predictions;
                Default None initializes an independent MSE loss, which computes
                reweighting averages from snapshot-based observables.
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
            loss_kwargs: Keyword arguments that are passed to the initializer
                of the loss function.
            num_chains: Sample from vectorized simulations instead from a single
               long one.

        """
        if loss_kwargs is None:
            loss_kwargs = {}

        # init simulation, reweighting functions and initial trajectory
        (key, weights_fn, propagate, safe_propagate) = self._init_statepoint(
            reference_state,
            energy_fn_template,
            simulator_template,
            neighbor_fn,
            timings,
            kbt,
            set_key,
            vmap_batch,
            initialize_traj,
            ref_press,
            safe_propagation=False,
            entropy_approximation=entropy_approximation,
            replica_kbt=replica_kbt,
            num_chains=num_chains
        )


        # build loss function for current state point
        if loss_fn is None:
            loss_fn = reweighting.init_default_loss_fn(targets, **loss_kwargs)
        else:
            print('Using custom loss function. Ignoring "target" dict.')

        quantities['energy'] = custom_quantity.energy_wrapper(
            energy_fn_template)
        reweighting.checkpoint_quantities(quantities)

        def difftre_loss(params, traj_state):
            """Computes the loss using the DiffTRe formalism and
            additionally returns predictions of the current model.
            """
            weights, _, entropy, free_energy = weights_fn(
                params, traj_state, entropy_and_free_energy=True)

            quantity_trajs = traj_util.quantity_traj(
                traj_state, quantities, params)
            quantity_trajs.update(entropy=entropy, free_energy=free_energy)

            loss, predictions = loss_fn(quantity_trajs, weights)

            # Always save free energy and entropy even if they are not part of
            # the loss.
            predictions.update(entropy=entropy, free_energy=free_energy)

            return loss, predictions

        loss_grad_fn = jax.value_and_grad(difftre_loss, has_aux=True, argnums=0)

        @safe_propagate
        @jit
        def difftre_grad_and_propagation(params, traj_state):
            """The main DiffTRe function that recomputes trajectories
            when needed and computes gradients of the loss wrt. energy function
            parameters for a single state point.
            """
            traj_state = propagate(params, traj_state)
            (loss_val, predictions), loss_grad = loss_grad_fn(params, traj_state)
            return traj_state, loss_val, loss_grad, predictions

        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point
        self.weights_fn[key] = jax.jit(weights_fn)

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
        grads, losses = [], []
        for sim_key in batch:

            grad_fn = self.grad_fns[sim_key]
            (new_traj_state, loss_val, curr_grad,
             state_point_predictions) = grad_fn(
                self.params, self.trajectory_states[sim_key])

            self.trajectory_states[sim_key] = new_traj_state
            self.predictions[sim_key][self._epoch] = state_point_predictions

            grads.append(curr_grad)
            losses.append(loss_val)
            if jnp.isnan(loss_val):
                warnings.warn(f'Loss of state point {sim_key} in epoch '
                              f'{self._epoch} is NaN. This was likely caused by'
                              f' divergence of the optimization or a bad model '
                              f'setup causing a NaN trajectory.')
                self._diverged = True  # ends training
                break

        self.batch_losses.append(sum(losses) / self.sim_batch_size)
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
        self.gradient_norm_history.append(util.tree_norm(batch_grad))
        self.step_size_history.append(alpha)

    def init_step_size_adaption(self,
                                allowed_reduction: ArrayLike = 0.5,
                                interior_points: int = 10,
                                step_size_scale: float = 1e-7
                                ) -> None:
        """Initializes a line search to tune the step size in each iteration.

        This method interpolates linearly between the old parameters
        :math:`\\theta^{(i)}` and the paremeters :math:`\\tilde\\theta`
        proposed by the optimizer to find the optimal update

        .. math ::
            \\theta^{(i + 1)} = (1 - \\alpha) \\theta^{(i)} + \\alpha\\tilde\\theta

        that reduces the effective sample size to a predefined constant

        .. math ::
            N_\\text{eff}(\\theta^{(i+1)}) = r\cdot N_\\text{eff}(\\theta^{(i)}).

        This method uses a vectorized bisection algorithm with fixed number of
        iterations. At each iteration, the algorithm computes the effective
        sample size for a predefined number of interior points and updates the
        search interval boundaries to include the two closest points bisecting
        the residual.

        The number of required iterations computes from the number of interior
        points :math:`n_i` and the desired accuracy :math:`a` via

        .. math ::
            N = \\left\\lceil -\\log(a) / \\log(n_i + 1)\\right\\rceil.

        Args:
            allowed_reduction: Target reduction of the effective sample size
            interior_points: Number of interiour points
            step_size_scale: Accuracy of the found optimal interpolation
                coefficient

        Returns:
            Returns the interpolation coefficient :math:`\\alpha`.

        """

        iterations = int(onp.ceil(-onp.log(step_size_scale) / onp.log(interior_points + 1)))
        print(f"[Step size] Use {iterations} iterations for {interior_points} interior points.")

        def _initialize_search(params, traj_states):
            N_effs = {
                sim_key: self.weights_fn[sim_key](
                    params, traj_states[sim_key])[1]
                for sim_key in self.statepoints
            }
            return N_effs

        @functools.partial(jax.vmap, in_axes=(0, None, None, None, None, None))
        def _residual(alpha, params, N_effs, batch_grad, proposal, traj_states):
            # Find the biggest reduction among the statepoints

            new_params = jax.tree_util.tree_map(
                lambda old, new: old * (1 - alpha) + new * alpha,
                params, proposal
            )

            reductions = []
            for sim_key in self.statepoints:
                # Calculate the expected effective number of weights
                _, N_eff_new = self.weights_fn[sim_key](
                    new_params, traj_states[sim_key]
                )

                reductions.append(jnp.log(N_eff_new) - jnp.log(N_effs[sim_key]))

            min_reduction = jnp.min(jnp.array(reductions))
            # Allow a reduction of the current effective sample size
            # The minimum reduction must still be larger than the allowed reduction
            # i.e. the residual of the final alpha must be greater than zero
            return min_reduction - jnp.log(allowed_reduction)

        def _step(state, _, params=None, N_effs=None, batch_grad=None, proposal=None, traj_states=None):
            a, b, res_a, res_b = state

            # Do not re-evaluate the residual for the already computed interval
            # boundaries
            c = jnp.reshape(jnp.linspace(a, b, interior_points + 2)[1:-1], (-1,))
            res_c = _residual(c, params, N_effs, batch_grad, proposal, traj_states)

            # debug.print("[Step Size] Residuals are {res}", res=res_c)

            # Add bondary points to the possible candidates
            c = jnp.concatenate((jnp.asarray([a, b]), c))
            res_c = jnp.concatenate((jnp.asarray([res_a, res_b]), res_c))

            # Find the smallest point bigger than zero and the biggest point
            # smaller than zero
            all_positive = jnp.where(res_c < 0, jnp.max(res_c), res_c)
            all_negative = jnp.where(res_c > 0, jnp.min(res_c), res_c)
            a_idx = jnp.argmin(all_positive)
            b_idx = jnp.argmax(all_negative)
            a, res_a = c[a_idx], res_c[a_idx]
            b, res_b = c[b_idx], res_c[b_idx]

            # debug.print("[Step Size] Search interval [{a}, {b}] with residual in [{res_a}, {res_b}]", a=a, b=b, res_a=res_a, res_b=res_b)

            return (a, b, res_a, res_b), None

        @jit
        def _adaptive_step_size(params, batch_grad, proposal, traj_states):
            N_effs = _initialize_search(params, traj_states)
            a, b = 1.0e-5, 1.0
            res_a, res_b = _residual(
                jnp.asarray([a, b]),
                params, N_effs, batch_grad, proposal, traj_states)

            # Check that minimum step size is sufficiently small, else just keep
            # the minimum step size
            b = jnp.where(res_a <= 0, a, b)

            # Check whether full step does not reduce the effective step size
            # below the threshold. If this is the case do the full step
            a = jnp.where(jnp.logical_and(res_a > 0, res_b > 0), b, a)

            # In the other case, do the bisection with the unchanged initial
            # values of a and b
            _step_fn = functools.partial(
                _step, N_effs=N_effs, batch_grad=batch_grad, proposal=proposal,
                traj_states=traj_states, params=params)
            (a, b, res_a, _), _ = lax.scan(
                _step_fn,
                (a, b, res_a, res_b), onp.arange(iterations)
            )
            return a, res_a

        self._adaptive_step_size = _adaptive_step_size

    def _evaluate_convergence(self, duration, thresh):
        last_losses = jnp.array(self.batch_losses[-self.sim_batch_size:])
        epoch_loss = jnp.mean(last_losses)
        self.epoch_losses.append(epoch_loss)
        print(
            f'\n[DiffTRe] Epoch {self._epoch}'
            f'\n\tEpoch loss = {epoch_loss:.5f}'
            f'\n\tGradient norm: {self.gradient_norm_history[-1]}'
            f'\n\tElapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        # print last scalar predictions
        for statepoint, prediction_series in self.predictions.items():
            last_predictions = prediction_series[self._epoch]
            for quantity, value in last_predictions.items():
                if value.ndim == 0:
                    print(f'\tPredicted {quantity}: {value}')

        self._converged = self.early_stop.early_stopping(epoch_loss, thresh,
                                                         self.params)

    @property
    def best_params(self):
        return self.early_stop.best_params

    def move_to_device(self):
        super().move_to_device()
        self.early_stop.move_to_device()


class DifftreActive(util.TrainerInterface):
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


class RelativeEntropy(reweighting.PropagationBase):
    """Trainer for relative entropy minimization."""
    def __init__(self, init_params, optimizer,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None,
                 convergence_criterion='window_median',
                 checkpoint_folder='Checkpoints'):
        """
        Initializes a relative entropy trainer instance.

        Uses first order method optimizer as Hessian is very expensive
        for neural networks. Both reweighting and the gradient formula
        currently assume a NVT ensemble.

        Args:
            init_params: Initial energy parameters
            optimizer: Optimizer from optax
            reweight_ratio: Ratio of reference samples required for n_eff to
                            surpass to allow re-use of previous reference
                            trajectory state. If trajectories should not be
                            re-used, a value > 1 can be specified.
            sim_batch_size: Number of state-points to be processed as a single
                            batch. Gradients will be averaged over the batch
                            before stepping the optimizer.
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function. Here, the
                                energy_fn_template is only a reference that
                                will be saved alongside the trainer. Each
                                state point requires its own due to the
                                dependence on the box size via the displacement
                                function, which can vary between state points.
            convergence_criterion: Either 'max_loss' or 'ave_loss'.
                                   If 'max_loss', stops if the gradient norm
                                   across all batches in the epoch is smaller
                                   than convergence_thresh. 'ave_loss' evaluates
                                   the average gradient norm across the batch.
                                   For a single state point, both are
                                   equivalent. A criterion based on the rolling
                                   standard deviation 'std' might be implemented
                                   in the future.
            checkpoint_folder: Name of folders to store ckeckpoints in.
        """

        checkpoint_path = 'output/rel_entropy/' + str(checkpoint_folder)
        init_trainer_state = util.TrainerState(
            params=init_params, opt_state=optimizer.init(init_params))
        super().__init__(init_trainer_state, optimizer, checkpoint_path,
                         reweight_ratio, sim_batch_size, energy_fn_template)

        # in addition to the standard trajectory state, we also need to keep
        # track of dataloader states for reference snapshots
        self.data_states = {}

        self.early_stop = max_likelihood.EarlyStopping(self.params,
                                                       convergence_criterion)

    def _set_dataset(self, key, reference_data, reference_batch_size,
                     batch_cache=1):
        """Set dataset and loader corresponding to current state point."""
        reference_loader = numpy_loader.NumpyDataLoader(R=reference_data,
                                                        copy=False)
        init_reference_batch, get_ref_batch, _ = data.random_reference_data(
            reference_loader, batch_cache, reference_batch_size)
        init_reference_batch_state = init_reference_batch(shuffle=True)
        self.data_states[key] = init_reference_batch_state
        return get_ref_batch

    def add_statepoint(self, reference_data, energy_fn_template,
                       simulator_template, neighbor_fn, timings, kbt,
                       reference_state, reference_batch_size=None,
                       batch_cache=1, initialize_traj=True, set_key=None,
                       vmap_batch=10):
        """
        Adds a state point to the pool of simulations.

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
            kbt: Temperature in kbT
            reference_state: Tuple of initial simulation state and neighbor list
            reference_batch_size: Batch size of dataloader for reference
                                  trajectory. If None, will use the same number
                                  of snapshots as generated via the optimizer.
            batch_cache: Number of reference batches to cache in order to
                         minimize host-device communication. Make sure the
                         cached data size does not exceed the full dataset size.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
            set_key: Specify a key in order to restart from same statepoint.
                     By default, uses the index of the sequance statepoints are
                     added, i.e. self.trajectory_states[0] for the first added
                     statepoint. Can be used for changing the timings of the
                     simulation during training.
            vmap_batch: Batch size of vmapping of per-snapshot energy and
                        gradient calculation.
        """
        if reference_batch_size is None:
            print('No reference batch size provided. Using number of generated'
                  ' CG snapshots by default.')
            states_per_traj = jnp.size(timings.t_production_start)
            if reference_state[0].position.ndim > 2:
                n_trajctories = reference_state[0].position.shape[0]
                reference_batch_size = n_trajctories * states_per_traj
            else:
                reference_batch_size = states_per_traj

        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbt,
                                                           set_key,
                                                           vmap_batch,
                                                           initialize_traj)

        reference_dataloader = self._set_dataset(key,
                                                 reference_data,
                                                 reference_batch_size,
                                                 batch_cache)

        grad_fn = reweighting.init_rel_entropy_gradient(
            energy_fn_template, weights_fn, kbt, vmap_batch)

        def propagation_and_grad(params, traj_state, batch_state):
            """Propagates the trajectory, if necessary, and computes the
            gradient via the relative entropy formalism.
            """
            traj_state = propagate(params, traj_state)
            new_batch_state, reference_batch = reference_dataloader(batch_state)
            reference_positions = reference_batch['R']
            grad = grad_fn(params, traj_state, reference_positions)
            return traj_state, grad, new_batch_state

        self.grad_fns[key] = propagation_and_grad

    def _update(self, batch):
        """Updates the potential using the gradient from relative entropy."""
        grads = []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]

            self.trajectory_states[sim_key], curr_grad, \
            self.data_states[sim_key] = grad_fn(self.params,
                                                self.trajectory_states[sim_key],
                                                self.data_states[sim_key])
            grads.append(curr_grad)

        batch_grad = util.tree_mean(grads)
        self._step_optimizer(batch_grad)
        self.gradient_norm_history.append(util.tree_norm(batch_grad))

    def _evaluate_convergence(self, duration, thresh):
        curr_grad_norm = self.gradient_norm_history[-1]
        print(f'\nEpoch {self._epoch}: Gradient norm: '
              f'{curr_grad_norm}, Elapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        self._converged = self.early_stop.early_stopping(curr_grad_norm, thresh,
                                                         save_best_params=False)


class SGMCForceMatching(probabilistic.ProbabilisticFMTrainerTemplate):
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
        return util.tree_unstack(self.params)

    def save_trainer(self, save_path):
        raise NotImplementedError('Saving the trainer currently does not work'
                                  ' for SGMCMC.')


# TODO adjust to new blackjax interface, then allow newer version
class NUTSForceMatching(probabilistic.MCMCForceMatchingTemplate):
    """Trainer that samples from the posterior distribution of energy_params via
    the No-U-Turn Sampler (NUTS), based on a force-matching formulation.
    """
    def __init__(self, prior, likelihood, train_loader, init_sample,
                 batch_cache=1, batch_size=1, val_loader=None,
                 warmup_steps=1000, step_size=None,
                 inv_mass_matrix=None, checkpoint_folder='Checkpoints',
                 ref_energy_fn_template=None, init_prng_key=random.PRNGKey(0)):
        checkpoint_path = 'output/NUTS/' + str(checkpoint_folder)

        log_posterior_fn = probabilistic.init_log_posterior_fn(
            likelihood, prior, train_loader, batch_size, batch_cache
        )
        init_state = nuts.new_state(init_sample, log_posterior_fn)

        if step_size is None or inv_mass_matrix is None:
            def warmup_gen_fn(step, inverse_mass_matrix):
                return nuts.kernel(log_posterior_fn, step,
                                   inverse_mass_matrix)

            init_state, (step_size, inv_mass_matrix), info = stan_warmup.run(
                init_prng_key, warmup_gen_fn, init_state, warmup_steps)
            print('Finished warmup.\n', info)

        kernel = nuts.kernel(log_posterior_fn, step_size,
                             inv_mass_matrix)
        super().__init__(init_state, kernel, checkpoint_path, val_loader,
                         ref_energy_fn_template)


class EnsembleOfModels(probabilistic.ProbabilisticFMTrainerTemplate):
    """Train an ensemble of models by starting optimization from different
    initial parameter sets, for use in uncertainty quantification applications.
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


class InterleaveTrainers(util.TrainerInterface):
    """Interleaves updates to train models using multiple algorithms.

    This special trainer allows to train models simultaneously with different
    algorithms.

    Usage:

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
