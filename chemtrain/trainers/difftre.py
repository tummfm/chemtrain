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

"""Trainer implementations for differentiable trajectory reweighting."""

import os
import pickle
import time
import warnings
from os import PathLike
from typing import Any, Callable, Dict

import jax
import numpy as onp
from jax import numpy as jnp, random, tree_util
from jax_md.partition import NeighborFn
from numpy.typing import ArrayLike
from optax import GradientTransformationExtraArgs

from chemtrain import parallel, util
from chemtrain.ensemble import reweighting, sampling
from chemtrain.learn import difftre
from chemtrain.trainers import base as tt
from chemtrain.typing import EnergyFnTemplate, TrajFn


class DifftreParallel(tt.MLETrainerTemplate):
    """Trainer class for parametrizing potentials via the DiffTRe method.

    This method runs independent statepoints in batches. It uses JAX
    vectorization in one process and can split each batch across MPI ranks.

    Args:
        init_params: Initial energy parameters
        optimizer: Optimizer from optax
        energy_fn_template: Function that takes energy parameters and
            initializes a new energy function.
        simulator_template: Function that takes an energy function and
            returns a simulator function.
        neighbor_fn: Neighbor function. Must be of
            :func:`jax_md_mod.custom_partition.masked_neighbor_list` if the
            statepoints have a different number of atoms.
        timings: Instance of TimingClass containing information about the
            trajectory length and which states to retain
        state_kwargs: Properties defining the thermodynamic state. Must at least
            contain the temperature 'kT'. For a non-exhaustive list, see
            :class:`chemtrain.ensemble.templates.StatePoint`.
        quantities: Dict containing for each observable specified by the key a
            corresponding function to compute it for each snapshot using
            :func:`ensemble.sampling.quantity_traj`.
        targets: Dict containing the same keys as quantities and containing
            another dict providing 'gamma' and 'target' for each observable.
        observables: Optional dictionary providing the observable functions
            for the targets.
        reference_states: Initial simulator states from which DiffTRe can
            compute the initial trajectory states.
        reweight_ratio: Ratio of reference samples required for n_eff to
            surpass to allow re-use of previous reference trajectory state.
            If trajectories should not be re-used, a value > 1 can be
            specified.
        allowed_reduction: Allowed reduction of the effective sample size
            through a parameter update.
        step_size_scale: Initial step size scale for the step size adaption.
        interior_points: Number of interior points to use for the step size
            adaption.
        sim_batch_size: Number of state-points to be processed as a single
            batch. Gradients will be averaged over the batch before stepping the
            optimizer.
        traj_states_on_host: Keep stored trajectories in host memory between
            updates.
        full_checkpoint: Save the complete trainer in single-process mode.
            MPI requires a state dictionary for synchronized in-place restore.
        target_loss_fns: Dictionary of loss functions to use for each target.
        loss_fn: Custom loss function to use for the training.
        vmap_batch: Number of samples to process simultaneously when computing
            instantaneous quantities for a trajectory.
        bucket_recompute: Groups together statepoints that need a recomputation.
        resample_simstates: Resample the sim states from all trajectories
            instead of simulating independent chains.
        convergence_criterion: Either 'max_loss' or 'ave_loss'.
            If 'max_loss', stops if the maximum loss across all batches in
            the epoch is smaller than convergence_thresh. 'ave_loss'
            evaluates the average loss across the batch. For a single state
            point, both are equivalent. A criterion based on the rolling
            standard deviation 'std' might be implemented in the future.
        checkpoint_path: Name of folders to store checkpoints in.
        log_dir: Path to the log file where to store training progress.
        parallelism: Use ``"single"`` or ``"mpi"``. ``"auto"`` selects MPI
            when launched with more than one rank. JAX sharding is unsupported.
        overflow_capacity_multiplier: Factor used to enlarge a neighbor list
            after an overflow.
        max_neighbor_retries: Maximum number of capacity increases attempted
            for one trajectory update.

    Attributes:
        batch_losses: List of losses for each batch in each epoch.
        epoch_losses: List of losses for each epoch.
        step_size_history: List of step sizes for each batched update.
        gradient_norm_history: List of gradient norms for each batched update.
        batch_gradient_norms: List of gradient norms for each batch.
        predictions: Dictionary containing the predictions for each statepoint
            at each epoch.
        early_stop: Instance of EarlyStopping to check for convergence.

    """

    def __init__(
            self,
            key: jax.Array,
            init_params: Any,
            optimizer: GradientTransformationExtraArgs,
            energy_fn_template: EnergyFnTemplate,
            simulator_template: Callable,
            neighbor_fn: NeighborFn,
            timings: sampling.TimingClass,
            state_kwargs: Dict[str, ArrayLike],
            quantities: Dict[str, Dict],
            targets: Dict[str, Any],
            observables: Dict[str, TrajFn],
            reference_states=None,
            reweight_ratio: float = 0.9,
            allowed_reduction: float = 0.95,
            step_size_scale: float = 1e-4,
            interior_points: int = 100,
            sim_batch_size: int = 1,
            traj_states_on_host: bool = False,
            full_checkpoint: bool = False,
            target_loss_fns: Dict[str, Callable] = None,
            loss_fn=None,
            vmap_batch: int = 10,
            bucket_recompute: bool = True,
            resample_simstates: bool = False,
            convergence_criterion: str = "window_median",
            checkpoint_path: os.PathLike = "Checkpoints",
            log_dir: os.PathLike = None,
            parallelism: parallel.Parallelism = "auto",
            overflow_capacity_multiplier: float = 1.25,
            max_neighbor_retries: int = 3,
    ):
        if parallelism == "auto":
            parallelism = "mpi" if util.use_mpi() else "single"
        if parallelism == "jax":
            raise ValueError(
                "DifftreParallel supports single-process and MPI execution."
            )
        self.parallel_context = parallel.resolve_parallelism(parallelism)
        if self.parallel_context.mode == "mpi" and full_checkpoint:
            raise ValueError(
                "MPI DiffTRe checkpoints store a trainer state dictionary; "
                "set full_checkpoint=False."
            )

        if overflow_capacity_multiplier <= 1.0:
            raise ValueError("overflow_capacity_multiplier must exceed 1.")
        if max_neighbor_retries < 1:
            raise ValueError("max_neighbor_retries must be positive.")
        self.overflow_capacity_multiplier = float(
            overflow_capacity_multiplier
        )
        self.max_neighbor_retries = int(max_neighbor_retries)

        init_state = util.TrainerState(
            params=init_params,
            opt_state=optimizer.init(init_params)
        )

        # Validate and normalize the statepoint axis before building vmaps.
        self.key = key
        self._reference_states = reference_states
        self.reweight_ratio = reweight_ratio
        self._bucket_recompute = bucket_recompute
        self._traj_states_on_host = traj_states_on_host

        # Determine the number of statepoints from the simulator states when
        # available. A loaded-trajectory workflow may omit them, so use the
        # first batched state or target value in that case.
        n_statepoints = None
        if reference_states is not None:
            reference_sim_state = (
                reference_states.sim_state
                if isinstance(reference_states, sampling.SimulatorState)
                else reference_states[0]
            )
            n_statepoints = int(reference_sim_state.position.shape[0])
        else:
            input_leaves = tree_util.tree_leaves((state_kwargs, targets))
            for leaf in input_leaves:
                shape = onp.shape(leaf)
                if shape:
                    n_statepoints = int(shape[0])
                    break
        if n_statepoints is None:
            raise ValueError(
                "Cannot determine the number of statepoints from scalar "
                "inputs without reference states."
            )
        self._n_statepoints = n_statepoints

        # Scalars apply to every statepoint. Other values must already carry
        # the statepoint axis first; later vmaps and MPI slicing rely on it.
        def normalize_statepoint_value(value):
            value = jnp.asarray(value)
            if value.ndim == 0:
                return jnp.broadcast_to(value, (self._n_statepoints,))
            if value.shape[0] != self._n_statepoints:
                raise ValueError(
                    "Every non-scalar statepoint leaf must use the common "
                    f"leading dimension {self._n_statepoints}."
                )
            return value

        self.statepoints = tree_util.tree_map(
            normalize_statepoint_value, state_kwargs
        )
        self.targets = tree_util.tree_map(
            normalize_statepoint_value, targets
        )

        # Merge a short final batch into the preceding batch. This keeps at
        # least one statepoint on every MPI rank throughout an epoch.
        if sim_batch_size == -1:
            sim_batch_size = self._n_statepoints
        self.batch_size = int(sim_batch_size)
        if not 1 <= self.batch_size <= self._n_statepoints:
            raise ValueError(
                "sim_batch_size must be -1 or between 1 and the number of "
                "statepoints."
            )
        if (
            self.parallel_context.mode == "mpi"
            and self.batch_size < self.parallel_context.size
        ):
            raise ValueError(
                "MPI DiffTRe requires at least one statepoint per rank in "
                "every batch."
            )

        # Trajectories are initialized explicitly or restored from a checkpoint.
        self.traj_states = None

        # Build the shared trajectory, reweighting, and gradient functions.
        gen_init_traj, *reweight_fns = reweighting.init_pot_reweight_propagation_fns(
            energy_fn_template,
            simulator_template,
            neighbor_fn,
            timings,
            self.statepoints,
            reweight_ratio,
            False,
            vmap_batch,
            safe_propagation=False,
            entropy_approximation=False,
            resample_simstates=resample_simstates,
            overflow_capacity_factor=self.overflow_capacity_multiplier,
        )

        if target_loss_fns is None:
            target_loss_fns = {}
        if loss_fn is None:
            loss_fn = difftre.init_default_loss_fn(observables, target_loss_fns)

        batched_model, batched_propagation, batched_weights = (
            difftre.init_difftre_gradient_and_propagation(
                reweight_fns,
                loss_fn,
                quantities,
                energy_fn_template,  # type: ignore
                wrapped=False,
                batched=True,
            )
        )

        self.model = jax.jit(
            jax.value_and_grad(batched_model, argnums=0, has_aux=True)
        )
        self.propagate = reweight_fns[-1](
            jax.jit(batched_propagation),
            multiple_arguments=False,
            max_retry=self.max_neighbor_retries,
        )
        self.weights = jax.jit(batched_weights)
        self._gen_init_traj = jax.jit(gen_init_traj)

        if allowed_reduction is not None:
            self._adaptive_step_size = difftre.init_step_size_adaption(
                lambda *args: (None, jnp.min(batched_weights(*args)[1])),
                allowed_reduction, step_size_scale=step_size_scale,
                interior_points=interior_points
            )
        else:
            self._adaptive_step_size = None

        super().__init__(
            init_state=init_state,
            optimizer=optimizer,
            checkpoint_path=checkpoint_path,
            full_checkpoint=full_checkpoint,
            log_file=log_dir
        )

        self.state = util.mpi_tree_broadcast(self.state, root=0)
        self.key = util.mpi_tree_broadcast(self.key, root=0)
        self.key = self.checkpoint("key", self.key)
        self.traj_states = self.checkpoint("trajectory_states", self.traj_states)
        self._reference_states = self.checkpoint(
            "reference_states", self._reference_states
        )
        self._checkpoint_parallel_size = self.checkpoint(
            "parallel_size", self.parallel_context.size
        )
        self._checkpoint_complete = self.checkpoint(
            "checkpoint_complete", False
        )
        self.batch_losses = self.checkpoint("batch_losses", [])
        self.batch_statepoint_counts = self.checkpoint(
            "batch_statepoint_counts", []
        )
        self.batch_gradient_norms = self.checkpoint("batch_gradient_norms", [])
        self.epoch_losses = self.checkpoint("epoch_losses", [])
        self.step_size_history = self.checkpoint("step_size_history", [])
        self.predictions: Dict[int, Dict[str, Any]] = self.checkpoint(
            "predictions", {}
        )  # type: ignore

        for idx in range(self.n_statepoints):
            if idx not in self.predictions.keys():
                self.predictions[idx] = {}

        self.early_stop = self.checkpoint(
            "early_stop",
            tt.EarlyStopping(self.params, convergence_criterion),
        )

    def initialize_trajstates(self, params: Any = None, *, num_runs: int = 1):
        """Initializes the trajectory states for all statepoints.

        Args:
            params: Energy parameters to use for the initial trajectories.
                If None, the current trainer parameters are used.

        """
        reference_states = self._reference_states
        if reference_states is None:
            raise ValueError(
                "Cannot initialize trajstates without reference_states. "
                "Passing initial_trajstates is not supported."
            )

        # Backwards compatibility: allow tuple (sim_state, nbrs).
        if isinstance(reference_states, tuple):
            reference_states = sampling.SimulatorState(
                sim_state=reference_states[0], nbrs=reference_states[1]
            )

        num_runs = int(num_runs)
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}.")

        if params is None:
            params = self.params
        elif self.parallel_context.mode == "mpi":
            params = util.mpi_tree_broadcast(params, root=0)

        n_statepoints = self.n_statepoints

        traj_gen_fn = jax.jit(
            jax.vmap(
                lambda k, p, r, s: self._gen_init_traj(k, p, r, num_runs=num_runs, **s),
                in_axes=(0, None, 0, 0),
            )
        )

        # Shape inference on a single statepoint to avoid huge abstract inputs.
        if n_statepoints < 1:
            raise ValueError("Cannot initialize trajstates for 0 statepoints.")

        shape_idx = onp.arange(1)
        shape_keys = random.split(self.key, 1)
        shape_reference = util.tree_take(reference_states, shape_idx, on_cpu=False)
        shape_statepoints = util.tree_take(self.statepoints, shape_idx, on_cpu=False)

        single_traj_shape = jax.eval_shape(
            traj_gen_fn, shape_keys, params, shape_reference, shape_statepoints
        )

        if self._traj_states_on_host:
            self.traj_states = util.tree_map(
                lambda x: onp.zeros((n_statepoints, *x.shape[1:]), dtype=x.dtype),
                single_traj_shape,
            )
        else:
            self.traj_states = util.tree_map(
                lambda x: jnp.zeros((n_statepoints, *x.shape[1:]), dtype=x.dtype),
                single_traj_shape,
            )

        offset = 0
        for num_states in self._epoch_batch_sizes():
            self.key, split = random.split(self.key)

            indices = onp.arange(offset, offset + num_states)
            reference_state_split = util.tree_take(
                reference_states, indices, on_cpu=False
            )
            statepoint_split = util.tree_take(
                self.statepoints, indices, on_cpu=False
            )
            splits = random.split(split, num_states)

            if self.parallel_context.mode == "mpi":
                (reference_state_split, statepoint_split, splits), dim = (
                    util.mpi_tree_slice(
                        (reference_state_split, statepoint_split, splits)
                    )
                )
            else:
                dim = None

            # Generate each rank's trajectories before gathering the batch.
            with util.mpi_guard():
                traj_states_split = traj_gen_fn(
                    splits, params, reference_state_split, statepoint_split
                )
                jax.block_until_ready(traj_states_split)

            if self.parallel_context.mode == "mpi":
                traj_states_split = util.mpi_tree_gather(traj_states_split, dim)

            if self._traj_states_on_host:
                traj_states_split_host = jax.device_get(traj_states_split)
                self.traj_states = util.tree_put(
                    self.traj_states, indices,
                    traj_states_split_host, on_cpu=True
                )
            else:
                self.traj_states = util.tree_put(
                    self.traj_states, indices,
                    traj_states_split, on_cpu=False
                )
            offset += num_states

    def load_trajstates(self, traj_states: PathLike | sampling.TrajectoryState):
        """Load replicated trajectories and reference states on rank zero.

        Args:
            traj_states: Either a file written by :meth:`save_trajstates` or a
                TrajectoryState instance.

        """
        saved_states = None
        with util.mpi_guard():
            if self.parallel_context.is_root:
                if isinstance(traj_states, sampling.TrajectoryState):
                    saved_states = {
                        "trajectory_states": traj_states,
                        "reference_states": self._reference_states,
                    }
                else:
                    with open(traj_states, "rb") as file:
                        saved_states = pickle.load(file)
        if self.parallel_context.mode == "mpi":
            comm = util.get_communicator()
            assert comm is not None
            saved_states = comm.bcast(saved_states, root=0)
        if not isinstance(saved_states, dict):
            raise ValueError(
                "Trajectory files must contain trajectories and reference "
                "states."
            )

        loaded_traj_states = saved_states["trajectory_states"]
        self._reference_states = saved_states["reference_states"]

        if self._traj_states_on_host:
            self.traj_states = jax.device_get(loaded_traj_states)
        else:
            self.traj_states = jax.device_put(loaded_traj_states)

    def save_trajstates(self, path: PathLike):
        """Save trajectories and reference states on rank zero.

        Args:
            path: Path to the pickle file to save trajstates to.
        """
        with util.mpi_guard():
            if self.parallel_context.is_root:
                saved_states = {
                    "trajectory_states": jax.device_get(self.traj_states),
                    "reference_states": jax.device_get(
                        self._reference_states
                    ),
                }
                path = os.fspath(path)
                temporary_path = f"{path}.tmp"
                with open(temporary_path, "wb") as file:
                    pickle.dump(saved_states, file)
                os.replace(temporary_path, path)

    def restore(self, checkpoint):
        """Restore one root-loaded checkpoint identically on every MPI rank."""
        checkpoint_data = checkpoint
        should_read = (
            self.parallel_context.mode != "mpi"
            or self.parallel_context.is_root
        )
        with util.mpi_guard():
            if isinstance(checkpoint, (str, PathLike)) and should_read:
                with open(checkpoint, "rb") as file:
                    checkpoint_data = pickle.load(file)
        if self.parallel_context.mode == "mpi":
            comm = util.get_communicator()
            assert comm is not None
            if not self.parallel_context.is_root:
                checkpoint_data = None
            checkpoint_data = comm.bcast(checkpoint_data, root=0)

        if not isinstance(checkpoint_data, dict):
            raise ValueError(
                "DifftreParallel exact restore requires a partial checkpoint."
            )
        if checkpoint_data.get("checkpoint_complete") is not True:
            raise ValueError(
                "Only checkpoints saved after a complete DiffTRe epoch can "
                "be resumed. Error-state checkpoints are diagnostic."
            )
        if "parallel_size" not in checkpoint_data:
            raise ValueError("Checkpoint does not contain its MPI world size.")
        saved_size = int(checkpoint_data["parallel_size"])
        if saved_size != self.parallel_context.size:
            raise ValueError(
                "Exact DiffTRe restore requires the original MPI world size."
            )
        super().restore(checkpoint_data)
        self._checkpoint_complete = False
        self.move_to_device()

    def _dump_checkpoint_occasionally(self, *args, **kwargs):
        """Mark regular end-of-epoch checkpoints as safe to resume."""
        self._checkpoint_complete = True
        try:
            super()._dump_checkpoint_occasionally(*args, **kwargs)
        finally:
            self._checkpoint_complete = False

    @property
    def params(self):
        """Current energy parameters."""
        return self.state.params

    @params.setter
    def params(self, loaded_params):
        """Replaces the current energy parameters."""
        self.state = self.state.replace(params=loaded_params)

    @property
    def n_statepoints(self):
        """Number of thermodynamic statepoints represented by the trainer."""
        return self._n_statepoints

    def _epoch_batch_sizes(self):
        """Return one epoch's batch sizes, merging a short final batch."""
        full_batches, remainder = divmod(
            self.n_statepoints, self.batch_size
        )
        batch_sizes = [self.batch_size] * full_batches
        if remainder:
            batch_sizes[-1] += remainder
        return batch_sizes

    def _get_batch(self):
        """Returns the next batch of statepoints to be processed."""
        if self.traj_states is None:
            raise ValueError(
                "Trajectory states not initialized. Call initialize_trajstates" \
                "first or load them with load_trajstates."
            )

        self.key, key = random.split(self.key)
        num_statepoints = self.n_statepoints
        mask = jnp.ones(num_statepoints)

        for draw_size in self._epoch_batch_sizes():
            key, split = random.split(key)

            # If bucketing is no longer possible or disabled, return back
            # a random batch of statepoints. Otherwise, return a full batch of
            # statepoints that either need or don't need reweighting.
            if not self._bucket_recompute or 2 * draw_size > jnp.sum(mask):
                batches = random.choice(
                    split, num_statepoints, (draw_size,),
                    replace=False, p=mask
                )
            else:
                # Compute the effective sample size for twice as many samples
                candidates = random.choice(
                    split, num_statepoints, (2 * draw_size,),
                    replace=False, p=mask
                )

                if self._traj_states_on_host:
                    candidates_np = onp.asarray(candidates)
                    trajstates_host = util.tree_take(
                        self.traj_states, candidates_np, on_cpu=True
                    )
                    trajstates = jax.device_put(trajstates_host)
                else:
                    trajstates = util.tree_take(
                        self.traj_states, candidates, on_cpu=False
                    )

                if self.parallel_context.mode == "mpi":
                    trajstates, dim = util.mpi_tree_slice(trajstates)
                else:
                    dim = None

                n_eff = self._effective_sample_sizes(trajstates, dim)
                min_n_eff = self.traj_states.trajectory.position.shape[
                                1] * self.reweight_ratio

                recompute = n_eff < min_n_eff

                # Select samples only from the largest class. At least one
                # of the conditions should be fulfilled:
                # a) At least BS samples must be recomputed
                # b) At least BS samples do not need a recomputation
                key, split = random.split(key)
                if jnp.sum(recompute) > draw_size:
                    select = jnp.float32(recompute)
                else:
                    select = jnp.float32(~recompute)

                # Select from the class at random. The not selected samples
                # should remain in the pool.
                batches = random.choice(
                    split, candidates, (draw_size,),
                    replace=False, p=select
                )

            # Mark the samples as drawn
            mask = mask.at[batches].set(0.0)
            yield batches

    def _effective_sample_sizes(self, trajstates, dim):
        """Compute and gather effective sample sizes for a local slice."""
        with util.mpi_guard():
            _, local_n_eff = self.weights(self.params, trajstates)
            jax.block_until_ready(local_n_eff)
        return util.mpi_tree_gather(local_n_eff, dim)

    def _prepare_batch(self, batch):
        """Load, split, and refresh one batch of trajectory data."""
        if self.traj_states is None:
            raise ValueError("Trajectory states have not been initialized.")

        # Load all aligned inputs before applying the same rank-local slice.
        if self._traj_states_on_host:
            batch_indices = onp.asarray(batch)
            trajstates = jax.device_put(
                util.tree_take(
                    self.traj_states, batch_indices, on_cpu=True
                )
            )
        else:
            trajstates = util.tree_take(
                self.traj_states, batch, on_cpu=False
            )
        targets = util.tree_take(self.targets, batch, on_cpu=False)
        statepoints = util.tree_take(self.statepoints, batch, on_cpu=False)
        (local_trajstates, local_targets, local_statepoints), dim = (
            util.mpi_tree_slice((trajstates, targets, statepoints))
        )

        n_eff = self._effective_sample_sizes(local_trajstates, dim)
        min_n_eff = (
            self.traj_states.trajectory.position.shape[1]
            * self.reweight_ratio
        )
        if self.parallel_context.is_root:
            print(
                "[DifftreParallel] Effective sample sizes "
                f"(limit: {min_n_eff})"
            )
            for statepoint_index, effective_size in zip(batch, n_eff):
                action = (
                    "-> recompute" if effective_size < min_n_eff else ""
                )
                print(
                    f"\t[Statepoint {statepoint_index}] Effective sample "
                    f"size: {effective_size:.2f} {action}"
                )

        # Store regenerated trajectories globally so later batches reuse them.
        if onp.any(n_eff < min_n_eff):
            if self.parallel_context.is_root:
                print("[DifftreParallel] Recomputing trajectories...")
            start = time.time()
            with util.mpi_guard():
                local_trajstates = self.propagate(
                    self.params, local_trajstates, local_statepoints
                )
                jax.block_until_ready(local_trajstates)
            trajstates = util.mpi_tree_gather(local_trajstates, dim)
            if self._traj_states_on_host:
                self.traj_states = util.tree_put(
                    self.traj_states,
                    onp.asarray(batch),
                    jax.device_get(trajstates),
                    on_cpu=True,
                )
            else:
                self.traj_states = util.tree_put(
                    self.traj_states, batch, trajstates, on_cpu=False
                )
            if self.parallel_context.is_root:
                print(
                    "[DifftreParallel] Recomputed trajectories in "
                    f"{(time.time() - start) / 60.:.2f} min"
                )

        return (
            local_trajstates,
            local_targets,
            local_statepoints,
            targets,
            statepoints,
            dim,
        )

    def _update(self, batch):
        """Update parameters from one batch and record its results."""
        (
            local_trajstates,
            local_targets,
            local_statepoints,
            targets,
            _,
            dim,
        ) = self._prepare_batch(batch)

        # Average the local loss and gradient over the global batch.
        if self.parallel_context.is_root:
            print("[DifftreParallel] Computing loss...")
        start = time.time()
        with util.mpi_guard():
            model_result = self.model(
                self.params,
                local_trajstates,
                local_statepoints,
                local_targets,
            )
            jax.block_until_ready(model_result)
            (local_loss, local_predictions), local_grad = model_result

        loss = util.mpi_tree_mean(local_loss, dim)
        grad = util.mpi_tree_mean(local_grad, dim)
        state_point_predictions = util.mpi_tree_gather(
            local_predictions, dim
        )

        batch_norm = util.tree_norm(grad)
        self.batch_gradient_norms.append(onp.asarray(batch_norm))
        if self.parallel_context.is_root:
            print(
                f"[DifftreParallel] Computed loss {loss} in "
                f"{(time.time() - start) / 60.:.2f} min"
            )

        proposal = self._optimizer_step(grad)
        start = time.time()

        if self._adaptive_step_size is None:
            alpha = jnp.asarray(1.0)
            residual = None
        else:
            # Every rank uses the smallest safe step proposed by any rank.
            with util.mpi_guard():
                local_alpha, local_residual = self._adaptive_step_size(
                    self.params, grad, proposal, local_trajstates
                )
                jax.block_until_ready((local_alpha, local_residual))
                alpha_value = float(onp.asarray(local_alpha))
                residual_value = float(onp.asarray(local_residual))
            candidates = [(alpha_value, residual_value, 0)]
            if self.parallel_context.mode == "mpi":
                comm = util.get_communicator()
                assert comm is not None
                candidates = comm.allgather(
                    (alpha_value, residual_value, self.parallel_context.rank)
                )
            alpha_value, residual_value, _ = min(
                candidates, key=lambda candidate: (candidate[0], candidate[2])
            )
            alpha = jnp.asarray(alpha_value)
            residual = jnp.asarray(residual_value)

        if self.parallel_context.is_root:
            print(
                f"[Step Size] Found optimal step size {alpha} with residual "
                f"{residual} in {(time.time() - start):.1f} s",
                flush=True,
            )

        self._step_optimizer(grad, alpha=alpha)

        # Record globally ordered results for this batch.
        if self.parallel_context.is_root:
            print("[DifftreParallel] Predictions:")
        for idx, b in enumerate(batch):
            self.predictions[int(b)][self._epoch] = {
                key: onp.asarray(val[idx])
                for key, val in state_point_predictions.items()
            }

            # Print scalar predictions
            if self.parallel_context.is_root:
                print(f"\t[Statepoint {b}]")
            for key, value in state_point_predictions.items():
                if jnp.shape(value[idx]) == ():
                    target = ""
                    if key in targets:
                        target = f"(target: {targets[key]['target'][idx]})"

                    if self.parallel_context.is_root:
                        print(f"\t\t{key} = {value[idx]} {target}")

        # Save the loss and gradient norm
        self.batch_losses.append(onp.asarray(loss))
        self.batch_statepoint_counts.append(len(batch))
        self.step_size_history.append(onp.asarray(alpha))

    def predict(self, batch):
        """Evaluate a batch on local MPI slices and return it in batch order.

        Regenerate and store trajectories first if the effective sample size
        is too low.
        """
        (
            local_trajstates,
            local_targets,
            local_statepoints,
            _,
            statepoints,
            dim,
        ) = self._prepare_batch(batch)

        if self.parallel_context.is_root:
            print("[DifftreParallel] Start predictions...")
            for idx, b in enumerate(batch):
                print(f"\t[Statepoint {b}]")
                for key, val in statepoints.items():
                    if jnp.isscalar(val[idx]):
                        print(f"\t\t{key} = {val[idx]}")

        with util.mpi_guard():
            model_result = self.model(
                self.params,
                local_trajstates,
                local_statepoints,
                local_targets,
            )
            jax.block_until_ready(model_result)
            (_, local_predictions), _ = model_result
        return util.mpi_tree_gather(local_predictions, dim)

    def _evaluate_convergence(
        self, *args, convergence_thresh=None, **kwargs
    ):
        """Update count-weighted epoch statistics and early stopping."""
        batches_per_epoch = len(self._epoch_batch_sizes())
        last_losses = jnp.asarray(self.batch_losses[-batches_per_epoch:])
        last_counts = jnp.asarray(
            self.batch_statepoint_counts[-batches_per_epoch:]
        )
        epoch_loss = jnp.sum(last_losses * last_counts) / jnp.sum(last_counts)
        duration = self.update_times[self._epoch]
        self.epoch_losses.append(epoch_loss)
        last_norms = jnp.asarray(
            self.batch_gradient_norms[-batches_per_epoch:]
        )
        mean_norm = jnp.sum(last_norms * last_counts) / jnp.sum(last_counts)
        self.gradient_norm_history.append(onp.asarray(mean_norm))

        if self.parallel_context.is_root:
            print(
                f"\n[DiffTRe] Epoch {self._epoch}"
                f"\n\tEpoch loss = {epoch_loss:.5f}"
                f"\n\tGradient norm: {self.gradient_norm_history[-1]}"
                f"\n\tElapsed time = {duration:.3f} min"
            )

        self._converged = self.early_stop.early_stopping(
            epoch_loss, convergence_thresh, self.params
        )

    @property
    def best_params(self):
        """Returns the best parameters according to the early stopping criterion."""
        return self.early_stop.best_params

    def move_to_device(self):
        """Transforms the trainer states to JAX arrays."""
        super().move_to_device()
        self.early_stop.move_to_device()
        if self._traj_states_on_host:
            self.traj_states = jax.device_get(self.traj_states)


class Difftre(tt.PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method.

    The Differentiable Trajectory Reweighting (DiffTRe) method [#Thaler2021]_
    is a method to compute the gradients of ensemble averages without
    differentiating through the simulation. Therefore, the method can
    efficiently train potential models on macroscopic observables.

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
            standard deviation 'std' might be implemented in the future.
        checkpoint_folder: Name of folders to store ckeckpoints in.

    Attributes:
        weight_fn: Dictionary containing the reweighting functions for each
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

    References:
        .. [#Thaler2021] Thaler, S.; Zavadlav, J. Learning Neural Network
           Potentials from Experimental Data via Differentiable Trajectory
           Reweighting. Nat Commun **2021**, 12 (1), 6884.
           https://doi.org/10.1038/s41467-021-27241-4.

    """

    def __init__(self,
                 init_params: Any,
                 optimizer: GradientTransformationExtraArgs,
                 reweight_ratio: ArrayLike = 1.0,
                 adaptive_step_size_threshold: float = 1e-4,
                 sim_batch_size: int = 1,
                 energy_fn_template: EnergyFnTemplate = None,
                 full_checkpoint: bool = False,
                 convergence_criterion: str = "window_median",
                 checkpoint_path: os.PathLike = "Checkpoints",
                 log_dir: os.PathLike = None):
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))

        # Optional: Initialized by calling trainer.init_step_size_adaption
        # after all statepoints to be considered have been set up.
        self._recompute = False

        self._adaptive_step_size_threshold = adaptive_step_size_threshold

        self.state_dicts = {}
        self.weight_fn = {}
        self.targets = {}
        super().__init__(
            init_trainer_state=init_state, optimizer=optimizer,
            checkpoint_path=checkpoint_path, reweight_ratio=reweight_ratio,
            sim_batch_size=sim_batch_size, full_checkpoint=full_checkpoint,
            energy_fn_template=energy_fn_template, log_dir=log_dir)

        self.batch_losses = self.checkpoint("batch_losses", [])
        self.epoch_losses = self.checkpoint("epoch_losses", [])
        self.step_size_history = self.checkpoint("step_size_history", [])
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
                       resample_simstates: bool = False,
                       allowed_reduction: ArrayLike = None,
                       adaption_kwargs: Dict = None
                       ):
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
                using :func:`ensemble.sampling.quantity_traj`.
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
            allowed_reduction: Allowed reduction of the effective sample size
                for the given statepoint.
            adaption_kwargs: Additional keyword arguments for the step size
                line search. For a description, see
                :func:`chemtrain.learn.difftre.init_step_size_adaption`.

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
            entropy_approximation=False,
            resample_simstates=resample_simstates
        )

        # For backwards compatibility and ease of use for a single statepoint
        if observables is None:
            observables = {
                key: target["traj_fn"] for key, target in targets.items()
            }
        if target_loss_fns is None:
            target_loss_fns = {
                key: target["loss_fn"] for key, target in targets.items()
                if "loss_fn" in target
            }

        # Enables a greater flexibility by sorting out data from frunctions
        targets = {
            key: {k: v for k, v in target.items() if k in ["gamma", "target"]}
            for key, target in targets.items() if target.get("target") is not None
        }

        # build loss function for current state point
        if loss_fn is None:
            loss_fn = difftre.init_default_loss_fn(observables, target_loss_fns)
        else:
            print("Using custom loss function. Ignoring 'target' dict.")

        difftre_grad_and_propagation = difftre.init_difftre_gradient_and_propagation(
            reweight_fns, loss_fn, quantities, energy_fn_template
        )

        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point
        self.weight_fn[key] = jax.jit(reweight_fns[0])
        self.state_dicts[key] = state_kwargs
        self.targets[key] = targets

        if allowed_reduction is not None:
            if adaption_kwargs is None:
                adaption_kwargs = {}

            self._adaptive_step_size[key] = difftre.init_step_size_adaption(
                self.weight_fn[key], allowed_reduction, **adaption_kwargs
            )

        # Reset loss measures if new state point es added since loss values
        # are not necessarily comparable
        self.early_stop.reset_convergence_losses()

    def predict(self, *, key: int):
        """Get predictions for a specific statepoint.

        This method predicts the target quantities for a specific
        statepoint. If necessary, the statepoint performs a trajectory
        regeneration.

        Args:
            key: The key of the statepoint to predict.

        Returns:
            Returns a dictionary containing the predicted observables
            given the current parameter values.

        """
        traj_state = self.trajectory_states[key]
        try:
            traj_state.overflow
        except:
            start = time.time()
            traj_state = traj_state()
            compute_time = (time.time() - start) / 60.

            print(
                f"Delayed initialization of trajectory state in {compute_time :.2f} min.")

        grad_fn = self.grad_fns[key]
        (new_traj_state, *_, state_point_predictions) = grad_fn(
            self.params, traj_state, self.state_dicts[key], self.targets[key],
            recompute=self._recompute
        )

        self.trajectory_states[key] = new_traj_state
        return state_point_predictions

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
            traj_state = self.trajectory_states[sim_key]
            try:
                traj_state.overflow
            except:
                start = time.time()
                traj_state = traj_state()
                compute_time = (time.time() - start) / 60.

                print(f"Delayed initialization of trajectory state in {compute_time :.2f} min.")

            grad_fn = self.grad_fns[sim_key]
            (new_traj_state, loss_val, curr_grad,
             state_point_predictions) = grad_fn(
                self.params, traj_state,
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

            # Print scalar predictions and statepoint measurements
            self._print_measured_statepoint(sim_key=sim_key)
            last_predictions = self.predictions[sim_key][self._epoch]
            for quantity, value in last_predictions.items():
                if value.ndim == 0:
                    if quantity in self.targets[sim_key]:
                        target = f"({self.targets[sim_key][quantity]['target']})"
                    else:
                        target = ""
                    print(f"\tPredicted {quantity}: {value} {target}")

            if jnp.isnan(loss_val):
                warnings.warn(f"Loss of state point {sim_key} in epoch "
                              f"{self._epoch} is NaN. This was likely caused by"
                              f" divergence of the optimization or a bad model "
                              f"setup causing a NaN trajectory.")
                self._diverged = True  # ends training
                break

        self.batch_losses.append(onp.asarray(losses / len(batch)))
        batch_grad = tree_util.tree_map(lambda x: x / len(batch), grads)

        step_size = 1.0
        recompute = False
        proposal = self._optimizer_step(batch_grad)
        for sim_key in batch:
            if sim_key not in self._adaptive_step_size: continue

            alpha, residual = self._adaptive_step_size[sim_key](
                self.params, batch_grad, proposal, self.trajectory_states[sim_key]
            )

            recompute |= alpha < self._adaptive_step_size_threshold

            print(f"[Step Size] Found optimal step size for {alpha} for statepoint {sim_key} with residual "
                  f"{residual}", flush=True)

            if alpha < step_size:
                step_size = alpha

        # self._recompute = recompute
        self._step_optimizer(batch_grad, alpha=step_size)

        batch_norm = util.tree_norm(batch_grad)
        self.gradient_norm_history.append(onp.asarray(batch_norm))
        self.step_size_history.append(onp.asarray(step_size))


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
            f"\n[DiffTRe] Epoch {self._epoch}"
            f"\n\tEpoch loss = {epoch_loss:.5f}"
            f"\n\tGradient norm: {self.gradient_norm_history[-1]}"
            f"\n\tElapsed time = {duration:.3f} min")

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
