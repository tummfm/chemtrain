from abc import abstractmethod
import time
import warnings

from jax import value_and_grad, checkpoint, jit, lax, random, \
    numpy as jnp
from jax_md import util as jax_md_util

from chemtrain import util, traj_util
from chemtrain.jax_md_mod import custom_quantity


def _independent_mse_loss_fn_init(quantities):
    """Initializes the default loss function, where MSE errors of
    destinct quantities are added.

    First, observables are computed via the reweighting scheme.
    These observables can be ndarray valued, e.g. vectors for RDF
    / ADF or matrices for stress. For each observable, the element-wise
    MSE error is computed wrt. the target provided in
    "quantities[quantity_key]['target']". This per-quantity loss
    is multiplied by gamma in "quantities[quantity_key]['gamma']".
    The final loss is then the sum over all of these weighted
    per-quantity MSE losses. A pre-requisite for using this function is
    that observables are simply ensemble averages of instantaneously
    fluctuating quantities. If this is not the case, a custom loss_fn
    needs to be defined. The custom loss_fn needs to have the same
    input-output signuture as the loss_fn implemented here.


    Args:
        quantities: The quantity dict with 'compute_fn', 'gamma' and
                    'target' for each observable

    Returns:
        The loss_fn taking trajectories of fluctuating properties,
        computing ensemble averages via the reweighting scheme and
        outputs the loss and predicted observables.
    """
    def loss_fn(quantity_trajs, weights):
        loss = 0.
        predictions = {}
        for quantity_key in quantities:
            quantity_snapshots = quantity_trajs[quantity_key]
            weighted_snapshots = (quantity_snapshots.T * weights).T
            average = jax_md_util.high_precision_sum(weighted_snapshots, axis=0)
            predictions[quantity_key] = average
            loss += quantities[quantity_key]['gamma'] * util.mse_loss(
                average, quantities[quantity_key]['target'])
        return loss, predictions
    return loss_fn


def _estimate_effective_samples(weights):
    """Returns the effective sample size after reweighting to
    judge reweighting quality.
    """
    # mask to avoid NaN from log(0) if a few weights are 0.
    weights = jnp.where(weights > 1.e-10, weights, 1.e-10)
    exponent = -jnp.sum(weights * jnp.log(weights))
    return jnp.exp(exponent)


def _build_weights(exponents):
    """Returns weights and the effective sample size from exponents
    of the reweighting formulas in a numerically stable way.
    """

    # The reweighting scheme is a softmax, where the exponent above
    # represents the logits. To improve numerical stability and
    # guard against overflow it is good practice to subtract the
    # max of the exponent using the identity softmax(x + c) =
    # softmax(x). With all values in the exponent <=0, this
    # rules out overflow and the 0 value guarantees a denominator >=1.
    exponents -= jnp.max(exponents)
    prob_ratios = jnp.exp(exponents)
    weights = prob_ratios / jax_md_util.high_precision_sum(prob_ratios)
    n_eff = _estimate_effective_samples(weights)
    return weights, n_eff


def reweight_trajectory(traj, targets, npt_ensemble=False):
    """Computes weights to reweight a trajectory from one thermodynamic
    state point to another.

    This function allows re-using an existing trajectory to compute
    observables at slightly perturbed thermodynamic state points. The
    reference trajectory can be generated at a constant state point or
    at different state points, e.g. via non-equlinibrium MD. Both NVT and
    NPT trajectories are supported, however reweighting currently only
    allows reweighting into the same ensemble. For NVT, the trajectory can
    be reweighted to a different temperature. For NPT, can be reweighted to
    different kbT and/or pressure. We assume quantities not included in
    'targets' to be constant over the trajectory, however this is not ensured
    by the code.
    For reference, implemented are cases 1. - 4. in
    https://www.plumed.org/doc-v2.6/user-doc/html/_r_e_w_e_i_g_h_t__t_e_m_p__p_r_e_s_s.html.

    Args:
        traj: Reference trajectory to be reweighted
        targets: A dict containing the targets under 'kbT' and/or 'pressure'.
                 If a keyword is not provided, the qunatity is assumed to be
                 and remain constant.
        npt_ensemble: Whether 'traj' was generated in NPT, default False is NVT.

    Returns:
        A tuple (weights, n_eff). Weights can be used to compute
        reweighted observables and n_eff judges the expected
        statistical error from reweighting.
    """
    if not npt_ensemble:
        assert 'kbT' in targets, 'For NVT, a kbT target needs to be provided.'
    # Note: if temperature / pressure are supposed to remain constant and are
    # hence not provided in the targets, we set them to the respective reference
    # values. Hence, their contribution to reweighting cancels. This should
    # even be at no additional cost under jit as XLA should easily detect the
    # zero contribution. Same applies to combinations in the NPT ensemble.
    target_kbt = targets.get('kbT', traj.thermostat_kbT)
    target_beta = 1. / target_kbt
    reference_betas = 1. / traj.thermostat_kbT

    # temperature reweighting
    if 'energy' not in traj.aux:
        raise ValueError('For reweighting, energies need to be provided '
                         'alongside the trajectory. Add energy to auxilary '
                         'outputs in trajectory generator.')
    exponents = -(target_beta - reference_betas) * traj.aux['energy']

    if npt_ensemble:  # correct for P * V
        assert 'kbT' in targets or 'pressure' in targets, ('At least one target'
                                                           ' needs to be given '
                                                           'for reweighting.')
        target_press = targets.get('pressure', traj.barostat_press)
        target_beta_p = target_beta * target_press
        ref_beta_p = reference_betas * traj.barostat_press
        volumes = traj_util.volumes(traj)

        # For constant p, reduces to -V * P_ref * (beta_target - beta_ref)
        # For constant T, reduces to -V * beta_ref * (p_target - p_ref)
        exponents -= volumes * (target_beta_p - ref_beta_p)

    return _build_weights(exponents)


def init_pot_reweight_propagation_fns(energy_fn_template, simulator_template,
                                      neighbor_fn, timings, ref_kbT,
                                      reweight_ratio=0.9, npt_ensemble=False):
    """
    Initializes all functions necessary for trajectory reweighting for
    a single state point.

    Initialized functions include a function that computes weights for a
    given trajectory and a function that propagates the trajectory forward
    if the statistical error does not allow a re-use of the trajectory.
    The propagation function also ensures that generated trajectories
    did not encounter any neighbor list overflow.
    """
    traj_energy_fn = custom_quantity.energy_wrapper(energy_fn_template)
    # TODO refactor 'compute_fn' and write function that checkpoints compute_fn
    reweighting_quantities = {'energy': {'compute_fn': traj_energy_fn}}

    if npt_ensemble:
        pressure_fn = 0  # TODO initialize and modify to
        reweighting_quantities['pressure'] = {'compute_fn': pressure_fn}

    trajectory_generator = traj_util.trajectory_generator_init(
        simulator_template, energy_fn_template, neighbor_fn, timings,
        reweighting_quantities)

    beta = 1. / ref_kbT
    # checkpoint energy and pressure functions as saving whole difftre
    # backward pass is too memory consuming
    for quantity_key in reweighting_quantities:
        reweighting_quantities[quantity_key]['compute_fn'] = checkpoint(
            reweighting_quantities[quantity_key]['compute_fn'])

    def compute_weights(params, traj_state):
        """Computes weights for the reweighting approach."""

        # reweighting properties (U and pressure) under perturbed potential
        reweight_properties = traj_util.quantity_traj(traj_state,
                                                      reweighting_quantities,
                                                      neighbor_fn,
                                                      params)

        # Note: Difference in pot. Energy is difference in total energy
        # as kinetic energy is the same and cancels
        exponent = -beta * (reweight_properties['energy']
                            - traj_state.aux['energy'])

        if npt_ensemble:  # we need to correct for the change in pressure
            # TODO test this
            volumes = traj_util.volumes(traj_state)
            exponent -= beta * volumes * (reweight_properties['pressure']
                                          - traj_state.aux['pressure'])
        return _build_weights(exponent)

    def trajectory_identity_mapping(inputs):
        """Re-uses trajectory if no recomputation needed."""
        traj_state = inputs[1]
        return traj_state

    def recompute_trajectory(inputs):
        """Recomputes the reference trajectory, starting from the last
        state of the previous trajectory to save equilibration time.
        """
        params, traj_state = inputs
        updated_traj = trajectory_generator(params, traj_state.sim_state)
        return updated_traj

    @jit
    def propagation_fn(params, traj_state):
        """Checks if a trajectory can be re-used. If not, a new trajectory
        is generated ensuring trajectories are always valid.
        Takes params and the traj_state as input and returns a
        trajectory valid for reweighting as well as an error code
        indicating if the neighborlist buffer overflowed during trajectory
        generation.
        """
        weights, n_eff = compute_weights(params, traj_state)
        n_snapshots = traj_state.aux['energy'].size
        recompute = n_eff < reweight_ratio * n_snapshots
        propagated_state = lax.cond(recompute,
                                    recompute_trajectory,
                                    trajectory_identity_mapping,
                                    (params, traj_state))
        return propagated_state

    def propagate(params, old_traj_state):
        """Wrapper around jitted propagation function that ensures that
        if neighbor list buffer overflowed, the trajectory is recomputed and
        the neighbor list size is increased until valid trajectory was obtained.
        Due to the recomputation of the neighbor list, this function cannot be
        jit.
        """
        new_traj_state = propagation_fn(params, old_traj_state)

        reset_counter = 0
        while new_traj_state.overflow:
            warnings.warn('Neighborlist buffer overflowed. '
                          'Initializing larger neighborlist.')
            if reset_counter == 3:  # still overflow after multiple resets
                raise RuntimeError('Multiple neighbor list re-computations did '
                                   'not yield a trajectory without overflow. '
                                   'Consider increasing the neighbor list '
                                   'capacity multiplier.')
            last_sim_snapshot, _ = new_traj_state.sim_state
            enlarged_nbrs = neighbor_fn(last_sim_snapshot.position)
            reset_traj_state = old_traj_state.replace(
                sim_state=(last_sim_snapshot, enlarged_nbrs))
            new_traj_state = recompute_trajectory((params, reset_traj_state))
            reset_counter += 1
        return new_traj_state

    return trajectory_generator, compute_weights, propagate


class PropagationBase(util.MLETrainerTemplate):
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
                 reweight_ratio=0.9, sim_batch_size=1, checkpoint_format='pkl',
                 energy_fn_template=None):
        super().__init__(optimizer, init_trainer_state, checkpoint_path,
                         checkpoint_format, energy_fn_template)
        self.sim_batch_size = sim_batch_size
        self.reweight_ratio = reweight_ratio

        # store for each state point corresponding traj_state and grad_fn
        # save in distinct dicts as grad_fns need to be deleted for checkpoint
        self.grad_fns, self.trajectory_states = {}, {}
        self.n_statepoints = 0
        self.shuffle_key = random.PRNGKey(0)

    def _init_statepoint(self, reference_state, energy_fn_template,
                         simulator_template, neighbor_fn, timings, kbT,
                         npt_ensemble=False, initialize_traj=True):
        """Initializes the simulation and reweighting functions as well
        as the initial trajectory for a statepoint."""

        # is there a better differentiator? kbT could be same for 2 simulations
        key = self.n_statepoints
        self.n_statepoints += 1

        initial_traj_generator, compute_weights, propagate = \
            init_pot_reweight_propagation_fns(energy_fn_template,
                                              simulator_template,
                                              neighbor_fn,
                                              timings,
                                              kbT,
                                              self.reweight_ratio,
                                              npt_ensemble)
        if initialize_traj:
            assert reference_state is not None, "If a new trajectory needs " \
                                                "to be generated, an initial " \
                                                "state needs to be provided."
            t_start = time.time()
            init_traj = initial_traj_generator(self.params, reference_state)
            runtime = (time.time() - t_start) / 60.
            print(f'Time for trajectory initialization {key}: {runtime} mins')
            self.trajectory_states[key] = init_traj
        else:
            print('Not initializing the initial trajectory is only valid if '
                  'a checkpoint is loaded. In this case, please be use to add '
                  'state points in the same sequence, otherwise loaded '
                  'trajectories will not match its respective simulations.')
        return key, compute_weights, propagate

    @abstractmethod
    def add_statepoint(self, *args, **kwargs):
        """User interface to add additional state point to train model on."""
        raise NotImplementedError()

    @property
    def params(self):
        return self.state.params

    @params.setter
    def params(self, loaded_params):
        self.state.params = loaded_params

    def _simulation_batches(self):
        """Helper function to re-shuffle simulations and split into batches."""
        self.shuffle_key, used_key = random.split(self.shuffle_key, 2)
        shuffled_indices = random.permutation(used_key, self.n_statepoints)
        if self.sim_batch_size == 1:
            batch_list = jnp.split(shuffled_indices, shuffled_indices.size)
        elif self.sim_batch_size == -1:
            batch_list = jnp.split(shuffled_indices, 1)
        else:
            raise NotImplementedError('Only batch_size = 1 or -1 implemented. '
                                      'Unclear how to deal with case, where '
                                      'batch_size > n_state_points.')
        return batch_list

    def train(self, epochs, checkpoint_freq=None, thresh=None):
        assert self.n_statepoints > 0, "Add at least 1 state point via " \
                                       "'add_statepoint' to start training."
        start_epoch = self.epoch
        end_epoch = start_epoch + epochs
        for epoch in range(start_epoch, end_epoch):
            start_time = time.time()
            batchlist = self._simulation_batches()
            for batch in batchlist:
                self._update(batch)

            duration = (time.time() - start_time) / 60.
            self.update_times.append(duration)
            converged = self._evaluate_convergence(duration, thresh)
            self.epoch += 1
            self.dump_checkpoint_occasionally(frequency=checkpoint_freq)

            if converged:
                break
        if thresh is not None:
            print('Maximum number of epochs reached without convergence.')

    @abstractmethod
    def _update(self, batch):
        """Implementation of gradient computation, stepping of the optimizer
        and logging of auxiliary results."""
        raise NotImplementedError()


class Trainer(PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method."""
    # TODO add NpT ensemble
    def __init__(self, init_params, optimizer, reweight_ratio=0.9,
                 sim_batch_size=1, energy_fn_template=None,
                 convergence_criterion='max_loss',
                 checkpoint_folder='Checkpoints', checkpoint_format='pkl'):
        """Initializes a DiffTRe trainer instance.

        The implementation assumes a NVT ensemble in weight computation.
        The trainer initialization only sets the initial trainer state
        as well as checkpointing and save-functionality. For training,
        target state points with respective simulations need to be added
        via 'add_statepoint'.

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
                                   If 'max_loss', stops if the maximum loss
                                   across all batches in the epoch is smaller
                                   than convergence_thresh. 'ave_loss' evaluates
                                   the average loss across the batch. For a
                                   single state point, both are equivalent.
                                   A criterion based on the rolling standatd
                                   deviation 'std' might be implemented in the
                                   future.
            checkpoint_folder: Name of folders to store ckeckpoints in.
            checkpoint_format: Checkpoint format, currently only .pkl supported.
        """

        self.batch_losses, self.epoch_losses = [], []
        self.predictions = {}
        # TODO doc: beware that for too short trajectory might have overfittet
        #  to single trajectory; if in doubt, set reweighting ratio = 1 towards
        #  end of optimization
        self.best_params = None
        self.criterion = convergence_criterion
        checkpoint_path = 'output/difftre/' + str(checkpoint_folder)
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))
        super(Trainer, self).__init__(init_state,
                                      optimizer,
                                      checkpoint_path,
                                      reweight_ratio,
                                      sim_batch_size,
                                      checkpoint_format,
                                      energy_fn_template)

    def add_statepoint(self, energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbT, quantities,
                       reference_state=None, loss_fn=None,
                       npt_ensemble=False, initialize_traj=True):
        """
        Adds a state point to the pool of simulations with respective targets.

        Requires own energy_fn_template and simulator_template to allow
        maximum flexibility for state points: Allows different ensembles
        (NVT vs NpT), box sizes and target quantities per state point.
        The quantity dict defines the way target observations
        contribute to the loss function. Each target observable needs to be
        saved in the quantity dict via a unique key. Model predictions will
        be output under the same key. Each unique observable needs to provide
        another dict containing a function computing the observable under
        'compute_fn', a multiplier controlling the weight of the observable
        in the loss function under 'gamma' as well as the prediction target
        under 'target'. The later 2 entries are not requires in case of a
        custom loss_fn.

        In many applications, the default loss function will be sufficient.
        If a target observable cannot be described directly as an average
        over instantaneous quantities (e.g. stiffness in the diamond example),
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
                                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbT: Temperature in kbT
            quantities: The quantity dict with 'compute_fn', 'gamma' and
                        'target' for each observable
            reference_state: Tuple of initial simulation state and neighbor list
            loss_fn: Custom loss function taking the trajectory of quantities
                     and weights and returning the loss and predictions;
                     Default None initializes an independent MSE loss, which
                     computes reweighting averages from snapshot-based
                     observables.
            npt_ensemble: Runs in NPT ensemble if True, default False is NVT.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
        """

        # init simulation, reweighting functions and initial trajectory
        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbT,
                                                           npt_ensemble,
                                                           initialize_traj)

        # build loss function for current state point
        if loss_fn is None:
            # TODO refactor this: Separate 'compute_fn' from targets and add
            #  checkpointing here
            loss_fn = _independent_mse_loss_fn_init(quantities)
        else:
            print('Using custom loss function. '
                  'Ignoring \'gamma\' and \'target\' in  \"quantities\".')

        def difftre_loss(params, traj_state):
            """Computes the loss using the DiffTRe formalism and
            additionally returns predictions of the current model.
            """
            weights, _ = weights_fn(params, traj_state)
            quantity_trajs = traj_util.quantity_traj(traj_state,
                                                     quantities,
                                                     neighbor_fn,
                                                     params)
            loss, predictions = loss_fn(quantity_trajs, weights)
            return loss, predictions

        statepoint_grad_fn = jit(value_and_grad(difftre_loss, has_aux=True))

        def difftre_grad_and_propagation(params, traj_state):
            """The main DiffTRe function that recomputes trajectories
            when needed and computes gradients of the loss wrt. energy function
            parameters for a single state point.
            """
            traj_state = propagate(params, traj_state)
            outputs, grad = statepoint_grad_fn(params, traj_state)
            loss_val, predictions = outputs
            return traj_state, grad, loss_val, predictions

        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point

    def _update(self, batch):
        """Computes gradient averaged over the sim_batch by propagating
        respective state points. Additionally saves predictions and loss
        for postprocessing."""
        # TODO parallelization? Maybe lift batch requirement and only sync sporadically?
        # https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
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
            new_traj_state, curr_grad, loss_val, state_point_predictions = \
                grad_fn(self.params, self.trajectory_states[sim_key])

            self.trajectory_states[sim_key] = new_traj_state
            self.predictions[sim_key][self.epoch] = state_point_predictions
            grads.append(curr_grad)
            losses.append(loss_val)
            if jnp.isnan(loss_val):
                warnings.warn(f'Loss of state point {sim_key} in epoch '
                              f'{self.epoch} is NaN. This was likely caused by '
                              f'divergence of the optimization or a bad model '
                              f'setup causing a NaN trajectory.')
                break

        self.batch_losses.append(sum(losses) / self.sim_batch_size)
        batch_grad = util.tree_mean(grads)
        self.step_optimizer(batch_grad)

    def _evaluate_convergence(self, duration, thresh):
        last_losses = jnp.array(self.batch_losses[-self.sim_batch_size:])
        epoch_loss = jnp.mean(last_losses)
        self.epoch_losses.append(epoch_loss)
        print(f'Epoch {self.epoch}: Epoch loss = {epoch_loss}, Elapsed time = '
              f'{duration} min')

        # save parameter set that resulted in smallest loss up to this point
        if jnp.argmin(jnp.array(self.epoch_losses)) == len(self.epoch_losses)-1:
            self.best_params = self.params

        converged = False
        if thresh is not None:
            if self.criterion == 'max_loss':
                if max(last_losses) < thresh: converged = True
            elif self.criterion == 'ave_loss':
                if epoch_loss < thresh: converged = True
            elif self.criterion == 'std':
                raise NotImplementedError('Currently, there is no criterion '
                                          'based on the std of the loss '
                                          'implemented.')
            else:
                raise ValueError(f'Convergence criterion {self.criterion} '
                                 f'unknown. Select "max_loss", "ave_loss" or '
                                 f'"std".')
        return converged
