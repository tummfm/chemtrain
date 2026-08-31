from chemtrain import util
from chemtrain.learn import force_matching, max_likelihood
from chemtrain.trainers import base as tt
from chemtrain.typing import EnergyFnTemplate
from jax_md_mod import custom_partition, custom_quantity


import numpy as onp
import jax
from jax import numpy as jnp
from jax_md.partition import NeighborList, NeighborListFns


from os import PathLike
from typing import Any, Callable, Dict, Mapping, Optional


class ForceMatching(tt.DataParallelTrainer):
    """Parametrizes potential models via the Force Matching method.

    The Force Matching method can be used to learn atomistic [#Ercolessi1994]_
    and coarse-grained [#Noid2008]_ models from first-principle or atomistic
    reference data.

    Args:
        init_params: Initial energy parameters.
        energy_fn_template: Function that takes energy parameters and returns
            an energy function.
        nbrs_init: Initial neighbor list. The neighbor list must be large enough
            to not overflow for any sample of the dataset.
        optimizer: Optimizer from optax.
        gammas: Coefficients for the individual targets in the weighted loss.
        weights_keys: Dictionary to entries of the dataset that contain a
            per-sample weight for the total loss.
        additional_targets: Additional snapshot targets to train on. Forces
            and energy are derived automatically from the energy_fn_template.
        feature_extract_fns: Features to extract from the data, passed to
            all snapshot functions as keyword arguments.
        energy_fn_has_aux: Energy function has an auxiliary output. The
            energy function will be called with argument ``mode="with_aux"``
            and should return a tuple ``(pot, aux)``.
        batch: Global batch size across MPI ranks and local devices.
        batch_per_device: Legacy batch per data-parallel device. It is
            multiplied by the global data-parallel size to derive ``batch``.
        batch_cache: Number of batches to load into the device memories.
        full_checkpoint: Save the whole trainer instead of only some statistics.
        disable_shmap: Use ``pmap`` instead of ``shmap`` for parallelization.
        parallelism: Data-parallel backend: ``"auto"``, ``"single"``,
            ``"mpi"``, or ``"jax"``.
        mesh: Optional one-dimensional JAX mesh named ``"data"``.
        penalty_fn: Penalty depending only on the parameters.
        convergence_criterion: Check convergence via
            :class:`base.EarlyStopping`.
        checkpoint_path: Path to the folder to store checkpoints.
        log_file: Path to file where to log training progress.

    Warning:
        With ``neighbor_fns``, the trainer reallocates lists after an
        overflow. With ``nbrs_init``, the caller must provide enough capacity.

    References:
        .. [#Ercolessi1994] Ercolessi, F.; Adams, J. B. Interatomic Potentials
           from First-Principles Calculations: The Force-Matching Method.
           Europhys. Lett. 1994, 26 (8), 583–588.
           https://doi.org/10.1209/0295-5075/26/8/005.
        .. [#Noid2008] Noid, W. G.; Chu, J.-W.; Ayton, G. S.; Krishna, V.;
           Izvekov, S.; Voth, G. A.; Das, A.; Andersen, H. C. The Multiscale
           Coarse-Graining Method. I. A Rigorous Bridge between Atomistic and
           Coarse-Grained Models. J Chem Phys 2008, 128 (24), 244114.
           https://doi.org/10.1063/1.2938860.

    """
    def __init__(self,
                 init_params,
                 optimizer,
                 energy_fn_template: EnergyFnTemplate,
                 nbrs_init: None | NeighborList = None,
                 neighbor_fns: None | NeighborListFns | dict[str, NeighborListFns] = None,
                 neighbor_fns_kwargs: Optional[Mapping[str, Any]] = None,
                 gammas: Dict[str, float] = None,
                 error_fns: Dict[str, Callable] = None,
                 weights_keys: Dict[str, str] = None,
                 additional_targets: Dict[str, Dict] = None,
                 feature_extract_fns: Dict[str, Callable] = None,
                 energy_fn_has_aux: bool = False,
                 batch: Optional[int] = None,
                 batch_per_device: Optional[int] = None,
                 batch_cache: int = 10,
                 full_checkpoint: bool = False,
                 disable_shmap: bool = False,
                 penalty_fn: Callable = None,
                 convergence_criterion: str = "window_median",
                 log_file: str = "force_matching.log",
                 checkpoint_path: PathLike = "checkpoints",
                 *,
                 parallelism: str = "auto",
                 mesh=None):

        if nbrs_init is not None and neighbor_fns is not None:
            raise ValueError(
                "Only one of `nbrs_init` and `neighbor_fns` may be provided. "
                "Prefer `neighbor_fns` for dynamic neighbor-list resizing."
            )

        if neighbor_fns is None:
            normalized_neighbor_fns = None

        elif isinstance(neighbor_fns, Mapping):
            if not neighbor_fns:
                raise ValueError("`neighbor_fns` must not be an empty mapping.")
            if "default" not in neighbor_fns:
                raise ValueError(
                    "A neighbor-list mapping requires a 'default' entry."
                )

            normalized_neighbor_fns = dict(neighbor_fns)

        else:
            # Single NeighborListFns-like object.
            normalized_neighbor_fns = {
                "default": neighbor_fns
            }

        self.neighbor_fns = normalized_neighbor_fns
        self.neighbor_kwargs = {
            "extra_capacity": 1,
            **(neighbor_fns_kwargs or {}),
        }

        self.nbrs_state = None

        # Add additional trainable targets
        if gammas is None:
            gammas = {}

        # This feature extractor enables to evaluate the energy function
        # only once for all computations involving the energy and forces.
        feature_fns = {
            "energy_and_force": custom_quantity.energy_force_wrapper(
                energy_fn_template, has_aux=energy_fn_has_aux
            )
        }

        # These are common quantities to train on. The energy function is not
        # necessary, since forces and energy are pre-extracted
        quantities = {
            "F": custom_quantity.force_wrapper(None),
            "U": custom_quantity.energy_wrapper(None),
            "overflow": custom_quantity.neighbor_buffer_overflow
        }

        if additional_targets is not None:
            quantities.update(additional_targets)
        if feature_extract_fns is not None:
            feature_fns.update(feature_extract_fns)

        model = force_matching.init_model(
            nbrs_init, quantities, feature_extract_fns=feature_fns
        )

        loss_fn = force_matching.init_loss_fn(
            error_fns=error_fns, gammas=gammas, weights_keys=weights_keys)

        super().__init__(loss_fn, model, init_params, optimizer,
                         checkpoint_path, batch=batch, batch_cache=batch_cache,
                         batch_per_device=batch_per_device,
                         disable_shmap=disable_shmap, penalty_fn=penalty_fn,
                         convergence_criterion=convergence_criterion,
                         full_checkpoint=full_checkpoint,
                         log_file=log_file,
                         energy_fn_template=energy_fn_template,
                         parallelism=parallelism, mesh=mesh)

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
            print(f"{key}: MAE = {mae_value:.4f}")


    def _pre_batch_stage(self, batch, stage="training"):
        """Allocate neighbor lists initially or after an overflow."""

        if self.neighbor_fns is None:
            return batch

        # Do not mutate the original batch.
        batch = dict(batch)
        batch.pop("neighbor", None)

        initial_allocation = self.nbrs_state is None

        if initial_allocation:
            allocate_keys = set(self.neighbor_fns)

            allocation_batch = batch
            if (
                self.parallel_context.mode == "jax"
                and jax.process_count() > 1
            ):
                from jax.experimental import multihost_utils

                # Neighbor-list shapes are static. Every host must therefore
                # allocate from the same global sample.
                allocation_batch = multihost_utils.process_allgather(
                    allocation_batch, tiled=True
                )

            init_batch = util.tree_get_single(allocation_batch, 0)

            if util.use_mpi():
                init_batch = util.mpi_tree_broadcast(init_batch)

            # No old lists to retain.
            nbrs = {}

        else:
            old_nbrs, overflow = self.nbrs_state

            # Globally synchronized decision for reallocation
            allocate_keys = {
                key
                for key in self.neighbor_fns
                if onp.any(util.mpi_any(overflow[key]))
            }

            if not allocate_keys:
                return {**batch, "neighbor": old_nbrs}

            allocation_batch = batch
            if (
                self.parallel_context.mode == "jax"
                and jax.process_count() > 1
            ):
                from jax.experimental import multihost_utils

                # Reallocation is rare, so replicate the batch only when an
                # overflowing global sample must be selected consistently.
                allocation_batch = multihost_utils.process_allgather(
                    allocation_batch, tiled=True
                )

            # Keep lists that did not overflow.
            nbrs = {
                key: old_nbrs[key]
                for key in self.neighbor_fns
                if key not in allocate_keys
            }

        for key in allocate_keys:

            if initial_allocation:
                sample = init_batch
            else:
                # Every MPI rank reaches this call for the same set of keys,
                # but supplies its own local overflow mask.
                sample, valid = util.mpi_tree_first_masked(
                    allocation_batch,
                    overflow[key],
                )

                if not valid:
                    raise RuntimeError(
                        f"Neighbor list '{key}' overflowed, but no "
                        "overflowing sample was found."
                    )

            position = sample["R"]

            allocation_kwargs = {
                k: v
                for k, v in sample.items()
                if k not in ("R", "neighbor", "F", "U")
            }
            allocation_kwargs.update(self.neighbor_kwargs)

            print(f"[ForceMatching] Allocating neighbor list '{key}'...")

            nbrs[key] = util.tree_replicate(
                self.neighbor_fns[key].allocate(
                position,
                **allocation_kwargs,
                ),
                util.tree_multiplicity(batch)
            )

            print(
                f"[ForceMatching] Neighbor list '{key}' occupancy: "
                f"{nbrs[key].max_occupancy}"
            )

        nbrs = custom_partition.NeighborListMap(neighbors=nbrs)
        overflow = {
            key: jnp.zeros(
                util.tree_multiplicity(batch),
                dtype=bool,
            )
            for key in self.neighbor_fns
        }

        self.nbrs_state = (nbrs, overflow)

        return {**batch, "neighbor": nbrs}

    def _post_batch_stage(self, out, stage="training"):
        """Check whether any neighbor list overflowed."""

        # Neighborlist size  is static
        if self.neighbor_fns is None:
            return super()._post_batch_stage(out, stage)

        overflow = out.predictions.pop("overflow")

        if (
            self.parallel_context.mode == "jax"
            and jax.process_count() > 1
        ):
            from jax.experimental import multihost_utils

            # The overflow flags are small and need to be visible on every
            # host before the next batch decides whether to reallocate.
            overflow = multihost_utils.process_allgather(
                overflow, tiled=True
            )

        # Every rank executes collectives in the same key order.
        global_overflow = {
            key: onp.any(util.mpi_any(overflow[key]))
            for key in self.neighbor_fns
        }

        if any(global_overflow.values()):
            print("[ForceMatching] Neighborlist overflow detected.")
            self.nbrs_state = (self.nbrs_state[0], overflow)

            return False

        return super()._post_batch_stage(out, stage)


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
        self._params = [init_sample["params"] for init_sample in init_samples]
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
            return self.results[0]["samples"]["variables"]["params"]
        else:
            params = []
            for chain in self.results:
                params.append(chain["samples"]["variables"]["params"])
            stacked_params = util.tree_stack(params)
            return util.tree_combine(stacked_params)

    @params.setter
    def params(self, loaded_params):
        raise NotImplementedError("Setting params seems not meaningful in"
                                  " the case of SG-MCMC samplers.")

    @property
    def list_of_params(self):
        """A list of the sampled parameters."""
        return util.tree_unstack(self.params)

    def save_trainer(self, save_path):
        """Save the trainer to a file."""
        raise NotImplementedError("Saving the trainer currently does not work"
                                  " for SGMCMC.")
