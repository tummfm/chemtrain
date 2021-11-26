"""Utility functions to process whole MD trajectories rather than
single snapshots.
"""
from functools import partial
from typing import Any
from chex import dataclass
from jax import jit, lax, vmap, numpy as jnp
from jax_md import util, quantity

Array = util.Array


@partial(dataclass, frozen=True)
class TimingClass:
    """A dataclass containing run-times for the simulation.

    Attributes:
    t_equilib_start: Starting time of all printouts that will be dumped
                     for equilibration
    t_production_start: Starting time of all runs that result in a printout
    timesteps_per_printout: Number of simulation timesteps to run forward
                            from each starting time
    time_step: Simulation time step
    """
    t_equilib_start: Array
    t_production_start: Array
    timesteps_per_printout: int
    time_step: float


@partial(dataclass, frozen=True)
class TrajectoryState:
    """A dataclass storing information of a generated trajectory.

    Attributes:
    sim_state: Last simulation state, a tuple of last state and nbrs
    trajectory: Generated trajectory
    energies: Potential energy value for each snapshot in trajectory
    overflow: True if neighbor list overflowed during trajectory generation
    """
    sim_state: Any
    trajectory: Any
    energies: Array = None
    overflow: Array = False


def process_printouts(time_step, total_time, t_equilib, print_every):
    """Initializes a dataclass containing information for the simulator
    on simulation time and saving states.

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
    assert total_time > t_equilib, "Total time needs to exceed " \
                                   "equilibration time, otherwise no " \
                                   "trajectory will be sampled."
    timesteps_per_printout = int(print_every / time_step)
    n_production = int((total_time - t_equilib) / print_every)
    n_dumped = int(t_equilib / print_every)
    equilibration_t_start = jnp.arange(n_dumped) * print_every
    production_t_start = jnp.arange(n_production) * print_every + t_equilib
    timings = TimingClass(t_equilib_start=equilibration_t_start,
                          t_production_start=production_t_start,
                          timesteps_per_printout=timesteps_per_printout,
                          time_step=time_step)
    return timings


def run_to_next_printout_neighbors(apply_fn, neighbor_fn, timings,
                                   kt_schedule=None):
    """Initializes a function that runs simulation to next printout
    state and returns that state.

    Run simulation forward to each printout point and return state.
    Used to sample a specified number of states

    Args:
      apply_fn: Apply function from initialization of simulator
      neighbor_fn: Neighbor function
      timings: Instance of TimingClass containing information
               about which states to retain and simulation time
      kt_schedule: A function mapping simulation time within the
             trajectory to target kbT as enforced by thermostat

    Returns:
      A function that takes the current simulation state, runs the
      simulation forward to the next printout state and returns it.
    """
    def do_step(cur_state, t):
        state, nbrs = cur_state
        if kt_schedule is None:
            new_state = apply_fn(state, neighbor=nbrs)
        else:
            new_state = apply_fn(state, neighbor=nbrs, kT=kt_schedule(t))
        nbrs = neighbor_fn(new_state.position, nbrs)
        new_sim_state = (new_state, nbrs)
        return new_sim_state, t

    @jit
    def run_small_simulation(start_state, t_start=0.):
        simulation_time_points = jnp.arange(timings.timesteps_per_printout) \
                                 * timings.time_step + t_start
        printout_state, _ = lax.scan(do_step,
                                     start_state,
                                     xs=simulation_time_points)
        cur_state, _ = printout_state
        return printout_state, cur_state
    return run_small_simulation


def trajectory_generator_init(simulator_template, energy_fn_template,
                              neighbor_fn, timings, with_energy=False):
    """Initializes a trajectory_generator function that computes a new
    trajectory stating at the last state.

    Args:
        simulator_template: Function returning new simulator given
                            current energy function
        energy_fn_template: Energy function template
        neighbor_fn: neighbor_fn
        timings: Instance of TimingClass containing information
                 about which states to retain
        with_energy: If True, trajectory also contains energy values
                     for each printout state. Possibly induces as slight
                     computational overhead.

    Returns:
        A function taking energy params and the current state (including
        neighbor list) that runs the simulation forward generating the
        next TrajectoryState.
    """
    def generate_reference_trajectory(params, sim_state, kt_schedule=None):
        energy_fn = energy_fn_template(params)
        _, apply_fn = simulator_template(energy_fn)
        run_to_printout = run_to_next_printout_neighbors(apply_fn,
                                                         neighbor_fn,
                                                         timings,
                                                         kt_schedule)

        sim_state, _ = lax.scan(run_to_printout,  # equilibrate
                                sim_state,
                                xs=timings.t_equilib_start)
        new_sim_state, traj = lax.scan(run_to_printout,  # production
                                       sim_state,
                                       xs=timings.t_production_start)

        if with_energy:
            ref_nbrs = new_sim_state[1]
            energies = energy_trajectory(traj, ref_nbrs, neighbor_fn, energy_fn)
        else:
            energies = None

        return TrajectoryState(sim_state=new_sim_state,
                               trajectory=traj,
                               energies=energies,
                               overflow=new_sim_state[1].did_buffer_overflow)

    return generate_reference_trajectory

# TODO vectorization of energy and quantity_trajectory might provide some
#  computational gains, at the expense of providing an additional parameter
#  for batch-size, which can lead to OOM errors if not chosen properly.


def volumes(traj_state):
    dim = traj_state.sim_state[0].position.shape[-1]
    return vmap(quantity.volume, (None, 0))(dim, traj_state.boxes)


def press_trajectory(params, traj_state):
    #  TODO --> unify with computation of energy
    pressure = quantity_traj(traj_state)
    return pressure


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
        traj_state: DifftreState as output from trajectory generator
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
