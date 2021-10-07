from collections import namedtuple
from jax import value_and_grad, checkpoint, jit, lax, numpy as jnp
import optax
import time
import warnings

from jax_md import util, dataclasses
from chemtrain.util import TrainerTemplate


DifftreState = namedtuple(
    "DifftreState",
    ["params",
     "traj_state",
     "opt_state"]
)

TrajectoryState = namedtuple(
    "TrajectoryState",
    ["sim_state",
     "trajectory",
     "energies"]
)


@dataclasses.dataclass
class TimingClass:
    """A dataclass containing runtimes of simulation

    Attributes:
    num_printouts_production: Number of states to save during production run
    num_dumped: Number of states to drop for equilibration
    timesteps_per_printout: Number of simulation timesteps to run for
                            each for each printout
    """
    num_printouts_production: int
    num_dumped: int
    timesteps_per_printout: int


def process_printouts(time_step, total_time, t_equilib, print_every):
    """Initializes a dataclass containing information for the simulator
    on saving states.

    Args:
        time_step: Time step size
        total_time: Total simulation time
        t_equilib: Equilibration time
        print_every: Time after which a state is saved

    Returns:
        A class containing information for the simulator
        on which states to save.
    """
    assert total_time > 0. and t_equilib > 0., "Times need to be positive."
    assert total_time > t_equilib, "Totoal time needs to exceed " \
                                   "equilibration time, otherwise no " \
                                   "trajectory will be sampled."
    timesteps_per_printout = int(print_every / time_step)
    num_printouts_production = int((total_time - t_equilib) / print_every)
    num_dumped = int(t_equilib / print_every)
    timings_struct = TimingClass(num_printouts_production,
                                 num_dumped,
                                 timesteps_per_printout)
    return timings_struct


def run_to_next_printout_neighbors(apply_fn, neighbor_fn, steps_per_printout):
    """Initializes a function that runs simulation to next printout 
    state and returns that state.

    Run simulation forward to each printout point and return state.
    Used to sample a specified number of states

    Args:
      apply_fn: Apply function from initialization of simulator
      neighbor_fn: Neighbor function
      steps_per_printout: Time steps to run for each printout state

    Returns:
      A function that takes the current simulation state, runs the 
      simulation forward to the next printout state and returns it.
    """
    def do_step(cur_state, t):
        state, nbrs = cur_state
        new_state = apply_fn(state, neighbor=nbrs)
        nbrs = neighbor_fn(new_state.position, nbrs)
        new_sim_state = (new_state, nbrs)
        return new_sim_state, t

    @jit
    def run_small_simulation(start_state, dummy):
        printout_state, _ = lax.scan(do_step, start_state,
                                     xs=jnp.arange(steps_per_printout))
        cur_state, _ = printout_state
        return printout_state, cur_state
    return run_small_simulation


def energy_trajectory(trajectory, init_nbrs, neighbor_fn, energy_fn):
    """Computes potential energy values for all states in a trajectory.

    Args:
        trajectory: Trajectory of states from simulation
        init_nbrs: A reference neighbor list to recompute neighbors
                   for each snapshot allowing jit
        neighbor_fn: Neighbor function
        energy_fn: Energy function

    Returns:
        An array of potential energy values containing the energy of
        each state in a trajectory.
    """
    def energy_snapshot(dummy_carry, state):
        R = state.position
        nbrs = neighbor_fn(R, init_nbrs)
        energy = energy_fn(R, neighbor=nbrs)
        return dummy_carry, energy

    _, U_traj = lax.scan(energy_snapshot,
                         jnp.array(0., dtype=jnp.float32),
                         trajectory)
    return U_traj


def quantity_traj(traj_state, quantities, neighbor_fn, energy_params=None):
    """Computes quantities of interest for all states in a trajectory.

    Arbitrary quantity functions can be provided via the quantities dict.
    The quantities dict should provide each quantity function via its own 
    key that contains another dict containing the function under the 
    'compute_fn' key. The resulting quantity trajectory will be saved in 
    a dict under the same key as the input quantity function.

    Args:
        traj_state: DiifftreState as output from trajectoty generator
        quantities: The quantity dict containing for each target quantity 
                    a dict containing the quantity function under 'compute_fn'
        neighbor_fn: Neighbor function
        energy_params: Energy params for energy_fn_template to initialize 
                       the current energy_fn

    Returns:
        A dict of quantity trajectories saved under the same key as the 
        input quantity function.
    """

    _, fixed_reference_nbrs = traj_state.sim_state

    @jit
    def quantity_trajectory(dummy_carry, state):
        R = state.position
        nbrs = neighbor_fn(R, fixed_reference_nbrs)
        computed_quantities = {quantity_fn_key: quantities[quantity_fn_key]
                               ['compute_fn'](state,
                                              neighbor=nbrs,
                                              energy_params=energy_params)
                               for quantity_fn_key in quantities}
        return dummy_carry, computed_quantities

    _, quantity_trajs = lax.scan(quantity_trajectory, 0., traj_state.trajectory)
    return quantity_trajs


def trajectory_generator_init(simulator_template, energy_fn_template,
                              neighbor_fn, timings_struct):
    """Initializes a trajectory_generator function that computes a new
    trajectory stating at the last state. Additionally computes energy
    values for each state used during the reweighting step.

    Args:
        simulator_template: Function returning new simulator given
                            current energy function
        energy_fn_template: Energy function template
        neighbor_fn: neighbor_fn
        timings_struct: Instance of TimingClass containing information
                        about which states to retain

    Returns:
        A function taking energy params and the current state (including
        neighbor list) that runs the simulation forward generating the
        next TrajectoryState.
    """
    num_printouts_production, num_dumped, timesteps_per_printout = \
        dataclasses.astuple(timings_struct)

    def generate_reference_trajectory(params, sim_state):
        energy_fn = energy_fn_template(params)
        _, apply_fn = simulator_template(energy_fn)
        run_to_printout = run_to_next_printout_neighbors(apply_fn,
                                                         neighbor_fn,
                                                         timesteps_per_printout)

        sim_state, _ = lax.scan(run_to_printout,
                                sim_state,
                                xs=jnp.arange(num_dumped))  # equilibrate
        new_sim_state, traj = lax.scan(run_to_printout,
                                       sim_state,
                                       xs=jnp.arange(num_printouts_production))
        final_state, nbrs = new_sim_state

        # always recompute neighbor list from last, fixed neighbor list.
        # Note:
        # one could save all the neighbor lists at the printout times,
        # if memory permits. In principle, this energy computation
        # could be omitted altogether if ones saves the energy  from
        # the simulator for each printout state. We did not opt for
        # this optimization to keep compatibility with Jax MD simulators.
        energies = energy_trajectory(traj, nbrs, neighbor_fn, energy_fn)
        return TrajectoryState(new_sim_state, traj, energies)

    return generate_reference_trajectory


def weight_computation_init(energy_fn_template, neighbor_fn, kbT):
    """Initializes a function that computes weights for the
    reweighting approach in the NVT ensemble.

    Args:
        energy_fn_template: Energy function template
        neighbor_fn: Neighbor function
        kbT: Temperature in kbT

    Returns:
        A function computing weights and the effective sample size
        given current energy params and TrajectoryState.
    """

    def estimate_effective_samples(weights):
        # mask to avoid NaN from log(0) if a few weights are 0.
        weights = jnp.where(weights > 1.e-10, weights, 1.e-10)
        exponent = - jnp.sum(weights * jnp.log(weights))
        return jnp.exp(exponent)

    def compute_weights(params, traj_state):
        _, nbrs = traj_state.sim_state

        # checkpointing: whole backward pass too memory consuming
        energy_fn = checkpoint(energy_fn_template(params))
        energies_new = energy_trajectory(traj_state.trajectory, nbrs,
                                         neighbor_fn, energy_fn)

        # Difference in pot. Energy is difference in total energy
        # as kinetic energy is the same and cancels
        exponent = -(1. / kbT) * (energies_new - traj_state.energies)
        # The reweighting scheme is a softmax, where the exponent above
        # represents the logits. To improve numerical stability and
        # guard against overflow it is good practice to subtract the
        # max of the exponent using the identity softmax(x + c) =
        # softmax(x). With all values in the exponent <=0, this
        # rules out overflow and the 0 value guarantees a denominator >=1.
        exponent -= jnp.max(exponent)
        prob_ratios = jnp.exp(exponent)
        weights = prob_ratios / util.high_precision_sum(prob_ratios)
        n_eff = estimate_effective_samples(weights)
        return weights, n_eff

    return compute_weights


def mse_loss(predictions, targets):
    """Computes mean squared error loss for given predictions and targets."""
    squared_difference = jnp.square(targets - predictions)
    mean_of_squares = util.high_precision_sum(squared_difference) \
                      / predictions.size
    return mean_of_squares


def independent_mse_loss_fn_init(quantities):
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
            average = util.high_precision_sum(weighted_snapshots, axis=0)
            predictions[quantity_key] = average
            loss += quantities[quantity_key]['gamma'] \
                * mse_loss(average, quantities[quantity_key]['target'])
        return loss, predictions
    return loss_fn


def propagation_fn_init(trajectory_generatior, compute_weights, reweight_ratio):
    """Initialize a function that checks if a trajectory can be re-used.
    If not, a new trajectory is generated ensuring trajectories are always
    valid.

    Args:
        trajectory_generatior: Initialized trajectory generator function
                               from trajectory_generator_init
        compute_weights: Initialized weight compute function from
                         weight_computation_init
        reweight_ratio: Ratio of reference samples required for n_eff to
                        surpass to allow re-use of previous trajectory state

    Returns:
        A function that takes params and the traj_state and returns a
        trajectory valid for reweighting as well as an error code
        indicating if the neighborlist buffer overflowed during trajectory
        generation.
    """

    def trajectory_identity_mapping(input):
        """Re-uses trajectory if no recomputation needed."""
        traj_state = input[1]
        return traj_state, 0

    def recompute_trajectory(input):
        """Recomputes the reference trajectory, starting from the last
        state of the previous trajectory to save equilibration time.
        """
        params, traj_state = input
        updated_traj = trajectory_generatior(params,
                                             traj_state.sim_state)
        _, nbrs = updated_traj.sim_state
        error_code = lax.cond(nbrs.did_buffer_overflow,
                              lambda _: 1, lambda _: 0, None)
        return updated_traj, error_code

    def propagation(params, traj_state):
        weights, n_eff = compute_weights(params, traj_state)
        n_snapshots = traj_state.energies.size
        recompute = n_eff < reweight_ratio * n_snapshots
        propagated_state, error_code = lax.cond(recompute,
                                                recompute_trajectory,
                                                trajectory_identity_mapping,
                                                (params, traj_state))
        return propagated_state, error_code
    return propagation


def difftre_gradient_init(compute_weights, trajectory_generation_fn,
                          loss_fn, quantities, neighbor_fn,
                          reweight_ratio=0.9):
    """
    Initializes the main DiffTRe function that recomputes trajectories
    when needed and computes gradients of the loss wrt. energy function
    parameters.
    
    Args:
        compute_weights: Initialized weight compute function from
                         weight_computation_init
        trajectory_generation_fn: Initialized trajectory generator
                                  function from trajectory_generator_init
        loss_fn: Loss function, e.g. from init_independent_mse_loss_fn
                 or custom implementation
        quantities: The quantity dict containing for each target quantity
                    a dict containing at least the quantity compute function
                    under 'compute_fn'.
        neighbor_fn: Neighbor function
        reweight_ratio: Ratio of reference samples required for n_eff to
                        surpass to allow re-use of previous reference
                        trajectory state

    Returns:
        A function that takes current DiffTReState and returns the new
       DifftreState, the gradient of the loss wrt. energy parameters,
       the loss value, an error code indicating neighbor list buffer
       overflow and predicted quantities of the current model state.
    """
    def reweighting_loss(params, traj_state):
        """Computes the loss using the DiffTRe formalism and additionally
        returns predictions of the current model.
        """
        weights, _ = compute_weights(params, traj_state)
        quantity_trajs = quantity_traj(traj_state,
                                       quantities,
                                       neighbor_fn,
                                       params)
        loss, predictions = loss_fn(quantity_trajs, weights)
        return loss, predictions

    # initialize function to recompute trajectory when necessary
    propagation_fn = propagation_fn_init(trajectory_generation_fn,
                                         compute_weights, reweight_ratio)

    @jit
    def difftre_grad(state):
        traj_state, error_code = propagation_fn(state.params, state.traj_state)
        outputs, curr_grad = value_and_grad(reweighting_loss, has_aux=True)(
            state.params, traj_state)
        loss_val, predictions = outputs
        return traj_state, curr_grad, loss_val, error_code, predictions

    return difftre_grad


def init_step_optimizer(optimizer, neighbor_fn):
    """
    Helper function to only update parameters if simulation did not result
    in neighbor list overflow. Otherise increases neighborlist for next
    iteration. This function cannot be jit therefore! This function is
    re-used in relative entropy optimization.

    Args:
        optimizer: Optimizer from optax
        neighbor_fn: Neighbor function

    Returns:
        Function updating params if neighborlist did not overflow
    """
    def step_optimizer(state, curr_grad, error_code, new_traj_state):
        if error_code == 1:
            warnings.warn('Neighborlist buffer overflowed. '
                          'Initializing larger neighborlist.')
            last_sim_snapshot, _ = state.traj_state.sim_state
            enlarged_nbrs = neighbor_fn(last_sim_snapshot.position)
            new_traj_state = TrajectoryState((last_sim_snapshot, enlarged_nbrs),
                                             state.traj_state.trajectory,
                                             state.traj_state.energies)
            new_params = state.params
            new_opt_state = state.opt_state
        else:  # only use gradient if neighbor list did not overflowed
            scaled_grad, new_opt_state = optimizer.update(curr_grad,
                                                          state.opt_state,
                                                          state.params)
            new_params = optax.apply_updates(state.params, scaled_grad)

        return DifftreState(new_params, new_traj_state, new_opt_state)
    return step_optimizer


def difftre_update_init(propagation_fn, optimizer, neighbor_fn):
    """Initializes the update function that is called iteratively to
    updates the energy parameters.

    The returned function computes the gradient used by the optimizer
    to update the energy parameters and the opt_state and returns loss
    and predictions at the current step. Additionally it handles the
    case of neighborlist buffer overflow by recomputing the neighborlist
    from the last state of the new reference trajectory and resets the
    trajectoty state such that the update is only performed with a proper
    neighbor list. The update function is not jittable for this reason.

    Args:
        propagation_fn: DiffTRe gradient and propagation function as
                        initialized from gradient_and_propagation_init
        optimizer: Optimizer from optax
        neighbor_fn: Neighbor function

    Returns:
        A function that will be called iteratively by the user to update
        energy params via the optimizer and output model predictions at
        the current state.
    """
    step_optimizer = init_step_optimizer(optimizer, neighbor_fn)

    def update(state):
        new_traj_state, curr_grad, loss_val, error_code, predictions = \
            propagation_fn(state)

        new_state = step_optimizer(state, curr_grad, error_code, new_traj_state)
        return new_state, loss_val, predictions
    return update


def difftre_init(simulator_template, energy_fn_template, neighbor_fn,
                 timings_struct, quantities, kbT, init_params, reference_state,
                 optimizer, loss_fn=None, reweight_ratio=0.9):
    """Initializes all functions for DiffTRe and returns an update
    function and the first reference trajectory.

    The current implementation assumes a NVT ensemble in weight computation.
    This function needs definition of all parameters of the simulation
    and DiffTRe. The quantity dict defines the way target observations
    contribute to the loss function. Each target observable needs to be
    saved in the quantity dict via a unique key. Model predictions will
    be output under the same key. Each unique observable needs to provide
    another dict containing a function computing the observable under
    'compute_fn', a multiplier controlling the weight of the observable
    in the loss function under 'gamma' as well as the prediction target
    under 'target'.

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
        simulator_template: Function that takes an energy function and
                            returns a simulator function.
        energy_fn_template: Function that takes energy parameters and
                            initializes an new energy function.
        neighbor_fn: Neighbor function
        timings_struct: Instance of TimingClass containing information
                        about which states to retain
        quantities: The quantity dict with 'compute_fn', 'gamma' and
                    'target' for each observable
        kbT: Temperature in kbT
        init_params: Initial energy parameters
        reference_state: Tuple of initial siumlation state and neighbor list
        optimizer: Optimizer from optax
        loss_fn: Custom loss function taking the trajectory of quantities
                 and weights and returning the loss and predictions;
                 Default None initializes independent MSE loss
        reweight_ratio: Ratio of reference samples required for n_eff to
                        surpass to allow re-use of previous reference
                        trajectory state

    Returns:
        An update function and the first reference trajectory
    """
    if loss_fn is None:
        loss_fn = independent_mse_loss_fn_init(quantities)
    else:
        print('Using custom loss function. '
              'Ignoring \'gamma\' and \'target\' in  \"quantities\".')

    trajectory_generation_fn = trajectory_generator_init(simulator_template,
                                                         energy_fn_template,
                                                         neighbor_fn,
                                                         timings_struct)
    compute_weights = weight_computation_init(energy_fn_template, neighbor_fn,
                                              kbT)
    propagation_fn = difftre_gradient_init(compute_weights,
                                           trajectory_generation_fn,
                                           loss_fn,
                                           quantities,
                                           neighbor_fn,
                                           reweight_ratio)
    update_fn = difftre_update_init(propagation_fn, optimizer, neighbor_fn)

    t_start = time.time()
    traj_initstate = trajectory_generation_fn(init_params, reference_state)
    runtime = (time.time() - t_start) / 60.
    print('Time for a single trajectory generation:', runtime, 'mins')

    return update_fn, traj_initstate


class Trainer(TrainerTemplate):
    def __init__(self, init_params, quantities, simulator_template,
                 energy_fn_template, neighbor_fn, reference_state,
                 timings_struct, optimizer, kbT, loss_fn=None,
                 reweight_ratio=0.9, checkpoint_folder='Checkpoints'):

        checkpoint_path = 'output/difftre/' + str(checkpoint_folder)
        super().__init__(energy_fn_template, checkpoint_path=checkpoint_path)

        # TODO implement optimization on multiple state points serial and
        #  in parallel
        # https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices

        self.epoch = 0
        self.losses, self.preditions, self.update_times = [], [], []
        opt_state = optimizer.init(init_params)

        self.update_fn, init_traj_state = difftre_init(simulator_template,
                                                       energy_fn_template,
                                                       neighbor_fn,
                                                       timings_struct,
                                                       quantities,
                                                       kbT,
                                                       init_params,
                                                       reference_state,
                                                       optimizer,
                                                       loss_fn,
                                                       reweight_ratio)

        self.__state = DifftreState(init_params, init_traj_state, opt_state)

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, loaded_state):
        self.__state = loaded_state

    @property
    def params(self):
        return self.__state.params

    def train(self, epochs, checkpoints=None):
        start_epoch = self.epoch
        end_epoch = self.epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            # training
            start_time = time.time()
            self.__state, loss, predictions = self.update_fn(self.__state)
            duration = (time.time() - start_time) / 60.
            print('Update', str(epoch), ': Loss =', str(loss),'Elapsed time =',
                  str(duration), 'min')
            self.losses.append(loss)
            self.preditions.append(predictions)
            self.update_times.append(duration)

            if jnp.isnan(loss):
                warnings.warn('Loss is NaN. This was likely caused by '
                              'divergence of the optimization or a bad '
                              'model setup causing a NaN trajectory.')
                break

            self.dump_checkpoint_occasionally(epoch, frequency=checkpoints)
        return
