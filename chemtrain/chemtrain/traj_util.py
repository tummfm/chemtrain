"""Utility functions to process whole MD trajectories rather than
single snapshots.
"""
from functools import partial
from typing import Any, Dict

import chex
from jax import jit, lax, vmap, numpy as jnp
from jax_md import util, quantity, simulate

Array = util.Array


@partial(chex.dataclass, frozen=True)
class TimingClass:
    """A dataclass containing run-times for the simulation.

    Attributes:
    t_equilib_start: Starting time of all printouts that will be dumped
                     for equilibration
    t_production_start: Starting time of all runs that result in a printout
    t_production_end: Generation time of all printouts
    timesteps_per_printout: Number of simulation timesteps to run forward
                            from each starting time
    time_step: Simulation time step
    """
    t_equilib_start: Array
    t_production_start: Array
    t_production_end: Array
    timesteps_per_printout: int
    time_step: float


@partial(chex.dataclass, frozen=True)
class TrajectoryState:
    """A dataclass storing information of a generated trajectory.

    Attributes:
    sim_state: Last simulation state, a tuple of last state and nbrs
    trajectory: Generated trajectory
    overflow: True if neighbor list overflowed during trajectory generation
    thermostat_kbT: Target thermostat kbT at time of respective snapshots
    barostat_press: Target barostat pressure at time of respective snapshots
    aux: Dict of auxilary per-snapshot quantities as defined by quantities
         in trajectory generator.
    """
    sim_state: Any
    trajectory: Any
    overflow: Array = False
    thermostat_kbT: Array = None
    barostat_press: Array = None
    aux: Dict = None


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
    production_t_end = production_t_start + print_every
    timings = TimingClass(t_equilib_start=equilibration_t_start,
                          t_production_start=production_t_start,
                          t_production_end=production_t_end,
                          timesteps_per_printout=timesteps_per_printout,
                          time_step=time_step)
    return timings


def _run_to_next_printout_neighbors(apply_fn, timings, **schedule_kwargs):
    """Initializes a function that runs simulation to next printout
    state and returns that state.

    Run simulation forward to each printout point and return state.
    Used to sample a specified number of states

    Args:
      apply_fn: Apply function from initialization of simulator
      neighbor_fn: Neighbor function
      timings: Instance of TimingClass containing information
               about which states to retain and simulation time
      schedule_kwargs: Kwargs to supply 'kT' and/or 'pressure' time-dependent
                       functions to allow for non-equilibrium MD

    Returns:
      A function that takes the current simulation state, runs the
      simulation forward to the next printout state and returns it.
    """
    def do_step(cur_state, t):
        apply_kwargs = {}
        if 'kt_schedule' in schedule_kwargs:
            apply_kwargs['kT'] = schedule_kwargs['kT'](t)
        if 'press_schedule' in schedule_kwargs:
            apply_kwargs['pressure'] = schedule_kwargs['pressure'](t)

        state, nbrs = cur_state
        new_state = apply_fn(state, neighbor=nbrs, **apply_kwargs)
        nbrs = nbrs.update(new_state.position)
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


def _canonicalize_dynamic_state_kwargs(state_kwargs, t_snapshots, *keys):
    """Converts constant state_kwargs, such as 'kT' and 'pressure' to constant
    lambda functions. Additionally return the values of state_kwargs at
    production printout times.
    """
    state_point_vals = []
    for key in keys:
        if key in state_kwargs:
            if jnp.isscalar(state_kwargs[key]):
                state_kwargs[key] = lambda t: state_kwargs[key]
            state_points = vmap(state_kwargs[key])(t_snapshots)
        else:
            state_points = None
        state_point_vals.append(state_points)
    return state_kwargs, tuple(state_point_vals)


def trajectory_generator_init(simulator_template, energy_fn_template,
                              timings, quantities=None):
    """Initializes a trajectory_generator function that computes a new
    trajectory stating at the last state.

    Args:
        simulator_template: Function returning new simulator given
                            current energy function
        energy_fn_template: Energy function template
        timings: Instance of TimingClass containing information
                 about which states to retain
        quantities: Quantities dict to compute and store auxilary quantities
                    alongside trajectory. This is particularly helpful for
                    storing energy and pressure in a reweighting context.

    Returns:
        A function taking energy params and the current state (including
        neighbor list) that runs the simulation forward generating the
        next TrajectoryState.
    """
    if quantities is None:
        quantities = {}

    def generate_reference_trajectory(params, sim_state, **kwargs):
        """
        Returns a new TrajectoryState with auxilary variables.

        Args:
            params: Energy function parameters
            sim_state: Initial simulation state (state)
            **kwargs: Kwargs to supply 'kT' and/or 'pressure' to change these
                      thermostat/barostat values on the fly. Can be constant
                      or function of t.

        Returns:
            TrajectoryState object containing the newly generated trajectory
        """

        kwargs, (kbt, barostat_press) = _canonicalize_dynamic_state_kwargs(
            kwargs, timings.t_production_end, 'kT', 'pressure')
        energy_fn = energy_fn_template(params)
        _, apply_fn = simulator_template(energy_fn)
        run_to_printout = _run_to_next_printout_neighbors(apply_fn, timings,
                                                          **kwargs)
        sim_state, _ = lax.scan(run_to_printout,  # equilibrate
                                sim_state,
                                xs=timings.t_equilib_start)
        new_sim_state, traj = lax.scan(run_to_printout,  # production
                                       sim_state,
                                       xs=timings.t_production_start)

        state = TrajectoryState(sim_state=new_sim_state,
                                trajectory=traj,
                                overflow=new_sim_state[1].did_buffer_overflow,
                                thermostat_kbT=kbt,
                                barostat_press=barostat_press)

        aux_trajectory = quantity_traj(state, quantities, params)
        return state.replace(aux=aux_trajectory)

    return generate_reference_trajectory


def volumes(traj_state):
    dim = traj_state.sim_state[0].position.shape[-1]
    boxes = vmap(simulate.npt_box)(traj_state.trajectory)
    return vmap(quantity.volume, (None, 0))(dim, boxes)


def quantity_traj(traj_state, quantities, energy_params=None):
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
        energy_params: Energy params for energy_fn_template to initialize
                       the current energy_fn

    Returns:
        A dict of quantity trajectories saved under the same key as the
        input quantity function.
    """

    _, fixed_reference_nbrs = traj_state.sim_state

    @jit
    def quantity_trajectory(dummy_carry, state):
        nbrs = fixed_reference_nbrs.update(state.position)
        computed_quantities = {quantity_fn_key: quantities[quantity_fn_key]
                               ['compute_fn'](state,
                                              neighbor=nbrs,
                                              energy_params=energy_params)
                               for quantity_fn_key in quantities}
        return dummy_carry, computed_quantities

    # TODO vectorization of might provide some computational gains at the
    #  expense of providing an additional parameter for batch-size, which
    #  can lead to OOM errors if not chosen properly.
    _, quantity_trajs = lax.scan(quantity_trajectory, 0., traj_state.trajectory)
    return quantity_trajs
