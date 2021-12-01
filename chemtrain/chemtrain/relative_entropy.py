from jax import jit, tree_map, tree_multimap, grad, lax, numpy as jnp
from jax_sgmc import data

from chemtrain import util, difftre


def init_rel_entropy_gradient(energy_fn_template, compute_weights, kbt):
    """Initializes a function that computes the relative entropy
    gradient given a trajectory and a batch of reference snapshots.
    """
    beta = 1 / kbt

    @jit
    def rel_entropy_gradient(params, traj_state, reference_batch):
        nbrs_init = traj_state.sim_state[1]

        def energy(params, R):
            energy_fn = energy_fn_template(params)
            nbrs = nbrs_init.update(R)
            return energy_fn(R, neighbor=nbrs)

        def weighted_gradient(grad_carry, input):
            dUdtheta = grad(energy)  # gradient wrt. params
            R = input[0]
            weight = input[1]
            snapshot_grad = dUdtheta(params, R)
            # sum over weights represents average
            update_carry = lambda carry, new_grad: carry + weight * new_grad
            updated_carry = tree_multimap(update_carry,
                                          grad_carry,
                                          snapshot_grad)
            return updated_carry, 0

        CG_traj = traj_state.trajectory.position
        CG_weights, _ = compute_weights(params, traj_state)
        ref_batchsize = reference_batch.shape[0]
        AA_weights = jnp.ones(ref_batchsize) / ref_batchsize  # no reweighting

        # TODO implement with vmap
        # Note:
        # would be more efficient to partially batch here, however
        # sequential trajectory generation dominates and batching
        # here would therefore not result in a significant speed-up.
        # The average via scan avoids linear memory scaling with
        # the trajectory length, which would become prohibitive for
        # memory consuming models such as neural network potentials.
        initial_grad = tree_map(jnp.zeros_like, params)
        mean_AA_grad, _ = lax.scan(weighted_gradient,
                                   initial_grad,
                                   (reference_batch, AA_weights))
        mean_CG_grad, _ = lax.scan(weighted_gradient,
                                   initial_grad,
                                   (CG_traj, CG_weights))

        combine_grads = lambda x, y: beta * (x - y)
        dtheta = tree_multimap(combine_grads, mean_AA_grad, mean_CG_grad)
        return dtheta
    return rel_entropy_gradient


class Trainer(difftre.PropagationBase):
    def __init__(self, init_params, optimizer,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None,
                 checkpoint_folder='Checkpoints', checkpoint_format='pkl'):
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
            checkpoint_folder: Name of folders to store ckeckpoints in.
            checkpoint_format: Checkpoint format, currently only .pkl supported.
        """

        checkpoint_path = 'output/rel_entropy/' + str(checkpoint_folder)
        init_trainer_state = util.TrainerState(
            params=init_params, opt_state=optimizer.init(init_params))
        super().__init__(init_trainer_state, optimizer, checkpoint_path,
                         reweight_ratio, sim_batch_size, checkpoint_format,
                         energy_fn_template)

        # in addition to the standard trajectory state, we also need to keep
        # track of dataloader states for reference snapshots
        self.data_states = {}

    def _set_dataset(self, key, reference_data, reference_batch_size,
                     batch_cache=1):
        """Set dataset and loader corresponding to current state point."""
        reference_loader = data.NumpyDataLoader(R=reference_data)
        cache_size = batch_cache * reference_batch_size
        init_reference_batch, get_reference_batch = data.random_reference_data(
            reference_loader, cache_size, reference_batch_size)
        init_reference_batch_state = init_reference_batch()
        self.data_states[key] = init_reference_batch_state
        return get_reference_batch

    def add_statepoint(self, reference_data, energy_fn_template,
                       simulator_template, neighbor_fn, timings, kbT,
                       reference_state, reference_batch_size=None,
                       batch_cache=1, initialize_traj=True):
        """
        Adds a state point to the pool of simulations.

        As each reference dataset / trajectory corresponds to a single
        state point, we initialize the dataloader together with the
        simulation.

        Args:
            reference_data: De-correlated reference trajectory
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbT: Temperature in kbT
            reference_state: Tuple of initial simulation state and neighbor list
            reference_batch_size: Batch size of dataloader for reference
                                  trajectory. If None, will use the same number
                                  of snapshots as generated via the optimizer.
            batch_cache: Number of reference batches to cache in order to
                         minimize host-device communication. Make sure the
                         cached data size does not exceed the full dataset size.
            npt_ensemble: Runs in NPT ensemble if True, default False is NVT.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
        """

        if reference_batch_size is None:
            # use same amount of snapshots as generated in trajectory by default
            reference_batch_size = jnp.size(timings.t_production_start)

        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbT,
                                                           initialize_traj)

        reference_dataloader = self._set_dataset(key,
                                                 reference_data,
                                                 reference_batch_size,
                                                 batch_cache)

        grad_fn = init_rel_entropy_gradient(energy_fn_template, weights_fn, kbT)

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

    def _evaluate_convergence(self, duration, thresh):
        print(f'Epoch {self._epoch}: Elapsed time = {duration} min')
        converged = False  # TODO implement convergence test
        if thresh is not None:
            raise NotImplementedError('Currently there is no convergence '
                                      'criterion implemented for relative '
                                      'entropy minimization. A possible '
                                      'implementation might be based on the '
                                      'variation of params or reweigting '
                                      'effective sample size.')
        return converged
