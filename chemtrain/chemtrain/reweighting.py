"""Implementation of the reweighting formalism.

Allows re-using existing trajectories.
p(S) = ...
"""
from abc import abstractmethod
import time
import warnings

from coax.utils._jit import jit
from jax import (checkpoint, lax, random, grad, tree_multimap, tree_map,
                 numpy as jnp)
from jax_md import util as jax_md_util

from chemtrain import util, traj_util, traj_quantity
from chemtrain.jax_md_mod import custom_quantity


def checkpoint_quantities(compute_fns):
    """Applies checkpoint to all compute_fns to save memory on backward pass."""
    for quantity_key in compute_fns:
        compute_fns[quantity_key] = checkpoint(compute_fns[quantity_key])


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


def reweight_trajectory(traj, **targets):
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
        targets: Kwargs containing the targets under 'kT' and/or 'pressure'.
                 If a keyword is not provided, the qunatity is assumed to be
                 and remain constant.

    Returns:
        A tuple (weights, n_eff). Weights can be used to compute
        reweighted observables and n_eff judges the expected
        statistical error from reweighting.
    """
    npt_ensemble = util.is_npt_ensemble(traj.sim_state[0])
    if not npt_ensemble:
        assert 'kT' in targets, 'For NVT, a "kT" target needs to be provided.'
    # Note: if temperature / pressure are supposed to remain constant and are
    # hence not provided in the targets, we set them to the respective reference
    # values. Hence, their contribution to reweighting cancels. This should
    # even be at no additional cost under jit as XLA should easily detect the
    # zero contribution. Same applies to combinations in the NPT ensemble.
    target_kbt = targets.get('kT', traj.thermostat_kbt)
    target_beta = 1. / target_kbt
    reference_betas = 1. / traj.thermostat_kbt

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
        volumes = traj_quantity.volumes(traj)

        # For constant p, reduces to -V * P_ref * (beta_target - beta_ref)
        # For constant T, reduces to -V * beta_ref * (p_target - p_ref)
        exponents -= volumes * (target_beta_p - ref_beta_p)

    return _build_weights(exponents)


def init_pot_reweight_propagation_fns(energy_fn_template, simulator_template,
                                      neighbor_fn, timings, ref_kbt,
                                      ref_press=None, reweight_ratio=0.9,
                                      npt_ensemble=False):
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
    reweighting_quantities = {'energy': traj_energy_fn}

    if npt_ensemble:
        # kinetic energy contribution to pressure cancels in reweighting
        pressure_fn = custom_quantity.init_pressure(energy_fn_template)
        reweighting_quantities['pressure'] = pressure_fn

    trajectory_generator = traj_util.trajectory_generator_init(
        simulator_template, energy_fn_template, timings, reweighting_quantities)

    beta = 1. / ref_kbt
    checkpoint_quantities(reweighting_quantities)

    def compute_weights(params, traj_state):
        """Computes weights for the reweighting approach."""

        # reweighting properties (U and pressure) under perturbed potential
        reweight_properties = traj_util.quantity_traj(traj_state,
                                                      reweighting_quantities,
                                                      params)

        # Note: Difference in pot. Energy is difference in total energy
        # as kinetic energy is the same and cancels
        exponent = -beta * (reweight_properties['energy']
                            - traj_state.aux['energy'])

        if npt_ensemble:  # we need to correct for the change in pressure
            volumes = traj_quantity.volumes(traj_state)
            kappa = traj_quantity.isothermal_compressibility_npt(volumes,
                                                                 ref_kbt)
            # TODO prune these computations
            case = 'neglect'
            if case == 'pressure':
                exponent -= beta * volumes * (reweight_properties['pressure']
                                              - traj_state.aux['pressure'])
            elif case == 'neglect':
                pass
            elif case == 'both_var_pressure':  # volume-based
                scaling_factor = kappa * (reweight_properties['pressure']
                                          - traj_state.aux['pressure'])
                new_volumes = volumes * scaling_factor
                exponent -= beta * traj_state.aux['pressure'] * (
                        new_volumes - volumes)
            elif case == 'only_first_baro':
                scaling_factor = kappa * (reweight_properties['pressure']
                                          - traj_state.aux['pressure'])
                new_volumes = volumes * scaling_factor
                exponent -= beta * traj_state.barostat_press * (
                        new_volumes - volumes)
            elif case == 'both_baro':
                scaling_factor = kappa * (reweight_properties['pressure']
                                          - traj_state.barostat_press)
                new_volumes = volumes * scaling_factor
                exponent -= beta * traj_state.barostat_press * (
                        new_volumes - volumes)
            elif case == 'both_volume_scale':
                scaling_factor = kappa * (reweight_properties['pressure']
                                          - traj_state.barostat_press)
                new_volumes = volumes * scaling_factor

                ref_scaling = kappa * (traj_state.aux['pressure']
                                       - traj_state.barostat_press)
                ref_volumes = volumes * ref_scaling
                exponent -= beta * traj_state.barostat_press * (
                        new_volumes - ref_volumes)
            else:
                raise NotImplementedError
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
        # give kT here as additional input to be handed through to energy_fn
        # for kbt-dependent potentials
        updated_traj = trajectory_generator(params, traj_state.sim_state,
                                            kT=ref_kbt, pressure=ref_press)
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
        _, n_eff = compute_weights(params, traj_state)
        n_snapshots = traj_state.aux['energy'].size
        recompute = n_eff < reweight_ratio * n_snapshots
        propagated_state = lax.cond(recompute,
                                    recompute_trajectory,
                                    trajectory_identity_mapping,
                                    (params, traj_state))
        return propagated_state

    def safe_propagate_traj(params, traj_state):
        """Recomputes trajectory and neighbor list until a trajectory without
        overflow was obtained.
        """
        reset_counter = 0
        while traj_state.overflow:
            warnings.warn('Neighborlist buffer overflowed. '
                          'Initializing larger neighborlist.')
            if reset_counter == 3:  # still overflow after multiple resets
                raise RuntimeError('Multiple neighbor list re-computations did '
                                   'not yield a trajectory without overflow. '
                                   'Consider increasing the neighbor list '
                                   'capacity multiplier.')

            # Note: We restart the simulation from the last trajectory, where
            # we know that no overflow has occured. Re-starting from a state
            # that was generated with overflow is dangerous as overflown
            # particles could cause exploding forces once re-considered.
            # TODO is this smart? Last simstate could be very bad state as it
            #  was computed with overflowing neighborlist
            last_state, _ = traj_state.sim_state
            if jnp.any(jnp.isnan(last_state.position)):
                raise RuntimeError('Last state is NaN. Currently there is no '
                                   'recovering from this. Restart from the last'
                                   ' non-overflown state might help, but comes'
                                   ' at the cost that the reference state is '
                                   'likely not representative.')
            enlarged_nbrs = util.neighbor_allocate(neighbor_fn, last_state)
            reset_traj_state = traj_state.replace(
                sim_state=(last_state, enlarged_nbrs))
            traj_state = recompute_trajectory((params, reset_traj_state))
            reset_counter += 1
        return traj_state

    def propagate(params, old_traj_state):
        """Wrapper around jitted propagation function that ensures that
        if neighbor list buffer overflowed, the trajectory is recomputed and
        the neighbor list size is increased until valid trajectory was obtained.
        Due to the recomputation of the neighbor list, this function cannot be
        jit.
        """
        new_traj_state = propagation_fn(params, old_traj_state)
        new_traj_state = safe_propagate_traj(params, new_traj_state)
        return new_traj_state

    def init_first_traj(params, reference_state):
        """Initializes initial trajectory to start optimization from.

        We dump the initial trajectory for equilibration, as initial
        equilibration usually takes much longer than equilibration time
        of each trajectory. If this is still not sufficient, the simulation
        should equilibrate over the course of subsequent updates.
        """
        dump_traj = trajectory_generator(params, reference_state,
                                         kT=ref_kbt, pressure=ref_press)

        t_start = time.time()
        init_traj = trajectory_generator(params, dump_traj.sim_state,
                                         kT=ref_kbt, pressure=ref_press)
        runtime = (time.time() - t_start) / 60.  # in mins
        init_traj = safe_propagate_traj(params, init_traj)
        return init_traj, runtime

    return init_first_traj, compute_weights, propagate


def init_default_loss_fn(targets):
    """Initializes the default loss function, where MSE errors of
    destinct quantities are added.

    First, observables are computed via the reweighting scheme.
    These observables can be ndarray valued, e.g. vectors for RDF
    / ADF or matrices for stress. For each observable, the element-wise
    MSE error is computed wrt. the target provided in
    "quantities[quantity_key]['target']". This per-quantity loss
    is multiplied by gamma in "quantities[quantity_key]['gamma']".
    The final loss is then the sum over all of these weighted
    per-quantity MSE losses. This function allows both observables that
    are simply ensemble averages of instantaneously fluctuating quantities
    and observables that are more complex functions of one or more quantity
    trajectories. The function computing the observable from trajectories of
    instantaneous fluctuating quantities needs to be provided via in
    "quantities[quantity_key]['traj_fn']". For the simple, but common case of
    an average of a single quantity trajectory, 'traj_fn' is given by
    traj_quantity.init_traj_mean_fn.

    Alternatively, a custom loss_fn can be defined. The custom
    loss_fn needs to have the same input-output signuture as the loss_fn
    implemented here.

    Args:
        targets: The target dict with 'gamma', 'target' and 'traj_fn'
        for each observable defined in 'quantities'.

    Returns:
        The loss_fn taking trajectories of fluctuating properties,
        computing ensemble averages via the reweighting scheme and
        outputs the loss and predicted observables.
    """
    def loss_fn(quantity_trajs, weights):
        loss = 0.
        predictions = {}
        # multiply weights by N such that averages can be computed with jnp.mean
        # rather than jnp.sum to allow for a unified formulation of
        # functions that compute observables from a trajectory, whether they
        # use weights or not (e.g. in postprocessing)
        weights *= weights.size

        weighted_quant_trajs = {
            quantity_key: (quantity_snapshots.T * weights).T
            for quantity_key, quantity_snapshots in quantity_trajs.items()
        }

        for target_key in targets:
            average = targets[target_key]['traj_fn'](weighted_quant_trajs)
            loss += targets[target_key]['gamma'] * util.mse_loss(
                average, targets[target_key]['target'])
            predictions[target_key] = average
        return loss, predictions
    return loss_fn


def init_rel_entropy_gradient(energy_fn_template, compute_weights, kbt):
    """Initializes a function that computes the relative entropy
    gradient given a trajectory and a batch of reference snapshots.
    """
    beta = 1 / kbt

    @jit
    def rel_entropy_gradient(params, traj_state, reference_batch):
        nbrs_init = traj_state.sim_state[1]

        def energy(params, position):
            energy_fn = energy_fn_template(params)
            # Note: nbrs update requires constant box, i.e. not yet
            # applicable to npt ensemble
            nbrs = nbrs_init.update(position)
            return energy_fn(position, neighbor=nbrs)

        def weighted_gradient(grad_carry, scan_input):
            dudtheta = grad(energy)  # gradient wrt. params
            position = scan_input[0]
            weight = scan_input[1]
            snapshot_grad = dudtheta(params, position)
            # sum over weights represents average
            update_carry = lambda carry, new_grad: carry + weight * new_grad
            updated_carry = tree_multimap(update_carry,
                                          grad_carry,
                                          snapshot_grad)
            return updated_carry, 0

        generated_traj = traj_state.trajectory.position
        weights, _ = compute_weights(params, traj_state)
        ref_batchsize = reference_batch.shape[0]
        ref_weights = jnp.ones(ref_batchsize) / ref_batchsize  # no reweighting

        # TODO implement with vmap
        # Note:
        # would be more efficient to partially batch here, however
        # sequential trajectory generation dominates and batching
        # here would therefore not result in a significant speed-up.
        # The average via scan avoids linear memory scaling with
        # the trajectory length, which would become prohibitive for
        # memory consuming models such as neural network potentials.
        initial_grad = tree_map(jnp.zeros_like, params)
        mean_ref_grad, _ = lax.scan(weighted_gradient,
                                    initial_grad,
                                    (reference_batch, ref_weights))
        mean_gen_grad, _ = lax.scan(weighted_gradient,
                                    initial_grad,
                                    (generated_traj, weights))

        combine_grads = lambda x, y: beta * (x - y)
        dtheta = tree_multimap(combine_grads, mean_ref_grad, mean_gen_grad)
        return dtheta
    return rel_entropy_gradient


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
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None):
        super().__init__(optimizer, init_trainer_state, checkpoint_path,
                         energy_fn_template)
        self.sim_batch_size = sim_batch_size
        self.reweight_ratio = reweight_ratio

        # store for each state point corresponding traj_state and grad_fn
        # save in distinct dicts as grad_fns need to be deleted for checkpoint
        self.grad_fns, self.trajectory_states, self.statepoints = {}, {}, {}
        self.n_statepoints = 0
        self.shuffle_key = random.PRNGKey(0)

    def _init_statepoint(self, reference_state, energy_fn_template,
                         simulator_template, neighbor_fn, timings, kbt,
                         ref_press=None, initialize_traj=True):
        """Initializes the simulation and reweighting functions as well
        as the initial trajectory for a statepoint."""

        # is there a better differentiator? kbT could be same for 2 simulations
        key = self.n_statepoints
        self.n_statepoints += 1
        self.statepoints[key] = {'kbT': kbt}
        npt_ensemble = util.is_npt_ensemble(reference_state[0])
        if npt_ensemble: self.statepoints[key]['pressure'] = ref_press
        # TODO ref pressure only used in print and to have barostat values.
        #  Reevaluate this parameter of barostat values not used in reweighting
        # TODO document ref_press accordingly

        gen_init_traj, compute_weights, propagate = \
            init_pot_reweight_propagation_fns(energy_fn_template,
                                              simulator_template,
                                              neighbor_fn,
                                              timings,
                                              kbt,
                                              ref_press,
                                              self.reweight_ratio,
                                              npt_ensemble)
        if initialize_traj:
            init_traj, runtime = gen_init_traj(self.params, reference_state)
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

        return (batch for batch in batch_list)

    def _print_measured_statepoint(self):
        """Print meausured kbT (and pressure for npt ensemble) for all
        statepoints to ensure the simulation is indeed carried out at the
        prescribed state point.
        """
        for sim_key, traj in self.trajectory_states.items():
            statepoint = self.statepoints[sim_key]
            measured_kbt = jnp.mean(traj.aux['kbT'])
            if 'pressure' in statepoint:  # NPT
                measured_press = jnp.mean(traj.aux['pressure'])
                press_print = (f' press = {measured_press:.2f} ref_press = '
                               f'{statepoint["pressure"]:.2f}')
            else:
                press_print = ''
            print(f'Statepoint {sim_key}: kbT = {measured_kbt:.3f} ref_kbt = '
                  f'{statepoint["kbT"]:.3f}' + press_print)

        print('')  # to visually differentiate between epochs

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
