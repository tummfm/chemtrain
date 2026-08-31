from chemtrain import config as chemtrain_config, util
from chemtrain.data import data_loaders
from chemtrain.ensemble import sampling
from chemtrain.learn import difftre
from chemtrain.trainers import base as tt
from chemtrain.typing import EnergyFnTemplate


from numpy.typing import ArrayLike

import jax
import numpy as onp
from jax import numpy as jnp
from jax_md.partition import NeighborFn
from jax_sgmc.data import numpy_loader

import os
from typing import Callable, Dict


class RelativeEntropy(tt.PropagationBase):
    """Trainer for relative entropy minimization.

    The Relative Entropy Minimization procedure coarse-graines potential
    models by minimizing the relative entropy between the atomistic reference
    and coarse-grained target canonical distributions [#Shell2008]_
    [#Thaler2022]_.

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

    References:
        .. [#Shell2008] Shell, M. S. The Relative Entropy Is Fundamental to
           Multiscale and Inverse Thermodynamic Problems. J. Chem. Phys. 2008,
           129 (14), 144108. https://doi.org/10.1063/1.2992060.
        .. [#Thaler2022] Thaler, S.; Stupp, M.; Zavadlav, J. Deep Coarse-Grained
           Potentials via Relative Entropy Minimization. The Journal of Chemical
           Physics 2022, 157 (24), 244103. https://doi.org/10.1063/5.0124538.

    """
    def __init__(self,
                 init_params,
                 optimizer,
                 reweight_ratio: float = 0.9,
                 sim_batch_size: int = 1,
                 energy_fn_template: EnergyFnTemplate = None,
                 convergence_criterion: str = "window_median",
                 checkpoint_path: os.PathLike = "Checkpoints",
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

        self.early_stop = tt.EarlyStopping(self.params, convergence_criterion)

    def _set_dataset(self, key, reference_data, reference_batch_size,
                     batch_cache=1):
        """Set dataset and loader corresponding to current state point."""
        reference_loader = numpy_loader.NumpyDataLoader(
            R=reference_data, copy=False)
        init_ref_batch, get_ref_batch, _ = data_loaders.init_batch_functions(
            data_loader=reference_loader, mb_size=reference_batch_size,
            cache_size=batch_cache,
            prefetch=chemtrain_config.read("async_dataloading", True),
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
                       resample_simstates: bool = False,
                       allowed_reduction: float = None,
                       adaption_kwargs: Dict = None):
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
            allowed_reduction: Allowed reduction of the effective sample size
                for the given statepoint.
            adaption_kwargs: Additional keyword arguments for the step size
                line search. For a description, see
                :func:`chemtrain.learn.difftre.init_step_size_adaption`.
        """
        if reference_batch_size is None:
            print("No reference batch size provided. Using number of generated "
                  "CG snapshots by default.")
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
            state_kwargs["kT"], vmap_batch
        )

        self.grad_fns[key] = propagation_and_grad
        self.delta_re[key] = []
        self.weight_fn[key] = jax.jit(reweight_fns[0])

        if allowed_reduction is not None:
            if adaption_kwargs is None:
                adaption_kwargs = {}

            self._adaptive_step_size[key] = difftre.init_step_size_adaption(
                self.weight_fn[key], allowed_reduction, **adaption_kwargs
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

        step_size = 1.0
        residual = None
        proposal = self._optimizer_step(batch_grad)
        for sim_key in batch:
            if sim_key not in self._adaptive_step_size: continue

            alpha, residual = self._adaptive_step_size[sim_key](
                self.params, batch_grad, proposal,
                self.trajectory_states[sim_key]
            )

            if alpha < step_size:
                step_size = alpha

        if residual is None:
            print(f"[Step Size] Using step size {step_size}", flush=True)
        else:
            print(
                f"[Step Size] Found optimal step size {step_size} with "
                f"residual {residual}",
                flush=True,
            )

        self._step_optimizer(batch_grad, alpha=step_size)

        batch_norm = util.tree_norm(batch_grad)
        self.gradient_norm_history.append(onp.asarray(batch_norm))
        self.step_size_history.append(onp.asarray(step_size))


    def _evaluate_convergence(self, *args, thresh=None, **kwargs):
        curr_grad_norm = self.gradient_norm_history[-1]
        # Mean loss from last simbatch
        mean_delta_re = onp.mean(
            [delta_re[-1] for delta_re in self.delta_re.values()]
        )
        duration = self.update_times[self._epoch]

        print(
            f"\n[RE] Epoch {self._epoch}"
            f"\n\tMean Delta RE loss = {mean_delta_re:.5f}"
            f"\n\tGradient norm: {curr_grad_norm}"
            f"\n\tElapsed time = {duration:.3f} min")

        self._print_measured_statepoint()

        self._converged = self.early_stop.early_stopping(
            curr_grad_norm, thresh, save_best_params=False)
