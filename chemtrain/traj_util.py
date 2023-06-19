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

"""Utility functions to process whole MD trajectories rather than
single snapshots.
"""
from functools import partial
from typing import Any, Dict

import chex
from jax import lax, vmap, numpy as jnp
from jax_md import simulate, util as jax_md_util

from chemtrain import util
from chemtrain.jax_md_mod import custom_quantity
from chemtrain.pickle_jit import jit

Array = jax_md_util.Array


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
    thermostat_kbt: Array = None
    barostat_press: Array = None
    aux: Dict = None


def process_printouts(time_step, total_time, t_equilib, print_every, t_start=0):
    """Initializes a dataclass containing information for the simulator
    on simulation time and saving states.

    This function is not jitable as array sizes depend on input values.

    Args:
        time_step: Time step size
        total_time: Total simulation run length
        t_equilib: Equilibration run length
        print_every: Time after which a state is saved
        t_start: Starting time. Only relevant for time-dependent
                 thermostat/barostat.

    Returns:
        A class containing information for the simulator
        on which states to save.
    """
    assert total_time > 0. and t_equilib > 0., 'Times need to be positive.'
    assert total_time > t_equilib, ('Total time needs to exceed equilibration '
                                    'time, otherwise no trajectory will be '
                                    'sampled.')
    timesteps_per_printout = int(print_every / time_step)
    n_production = int((total_time - t_equilib) / print_every)
    n_dumped = int(t_equilib / print_every)
    equilibration_t_start = jnp.arange(n_dumped) * print_every + t_start
    production_t_start = (jnp.arange(n_production) * print_every
                          + t_equilib + t_start)
    production_t_end = production_t_start + print_every
    timings = TimingClass(t_equilib_start=equilibration_t_start,
                          t_production_start=production_t_start,
                          t_production_end=production_t_end,
                          timesteps_per_printout=timesteps_per_printout,
                          time_step=time_step)
    return timings


def _run_to_next_printout_neighbors(apply_fn, timings, **kwargs):
    """Initializes a function that runs simulation to next printout
    state and returns that state.

    Run simulation forward to each printout point and return state.
    Used to sample a specified number of states

    Args:
      apply_fn: Apply function from initialization of simulator
      neighbor_fn: Neighbor function
      timings: Instance of TimingClass containing information
               about which states to retain and simulation time
      kwargs: Kwargs to supply 'kT' and/or 'pressure' time-dependent
                       functions to allow for non-equilibrium MD

    Returns:
      A function that takes the current simulation state, runs the
      simulation forward to the next printout state and returns it.
    """
    def do_step(cur_state, t):
        apply_kwargs = {}
        if 'kT' in kwargs:
            apply_kwargs['kT'] = kwargs['kT'](t)
        if 'pressure' in kwargs:
            apply_kwargs['pressure'] = kwargs['pressure'](t)

        state, nbrs = cur_state
        new_state = apply_fn(state, neighbor=nbrs, **apply_kwargs)
        nbrs = util.neighbor_update(nbrs, new_state)
        new_sim_state = (new_state, nbrs)
        return new_sim_state, t

    # @jit  # this triggers bug in JAX
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
    functions over time and deletes None kwargs. Additionally, return the
    values of state_kwargs at production printout times.
    """
    def constant_fn(_, c):
        return c

    state_point_vals = []
    for key in keys:
        if key in state_kwargs:
            if state_kwargs[key] is None:
                state_kwargs.pop(key)  # ignore kwarg if None is provided
                state_points = None
            else:
                if jnp.isscalar(state_kwargs[key]):
                    state_kwargs[key] = partial(constant_fn,
                                                c=state_kwargs[key])
                state_points = vmap(state_kwargs[key])(t_snapshots)
        else:
            state_points = None
        state_point_vals.append(state_points)
    return state_kwargs, tuple(state_point_vals)


def _traj_replicate_if_not_none(thermostat_values, n_traj):
    """Replicates thermostat targets to multiple trajectories, if not None."""
    if thermostat_values is not None:
        thermostat_values = jnp.tile(thermostat_values, n_traj)
    return thermostat_values


def trajectory_generator_init(simulator_template, energy_fn_template,
                              ref_timings=None, quantities=None, vmap_batch=10):
    """Initializes a trajectory_generator function that computes a new
    trajectory stating at the last traj_state.

    Args:
        simulator_template: Function returning new simulator given
                            current energy function
        energy_fn_template: Energy function template
        ref_timings: Instance of TimingClass containing information about the
                     times states need to be retained
        quantities: Quantities dict to compute and store auxilary quantities
                    alongside trajectory. This is particularly helpful for
                    storing energy and pressure in a reweighting context.
        vmap_batch: Batch size for computation of auxillary quantities.

    Returns:
        A function taking energy params and the current traj_state (including
        neighbor list) that runs the simulation forward generating the
        next TrajectoryState.
    """
    if quantities is None:
        quantities = {}

    # temperature is inexpensive and generally useful: compute it by default
    quantities['kbT'] = custom_quantity.temperature

    def generate_reference_trajectory(params, sim_state, **kwargs):
        """
        Returns a new TrajectoryState with auxilary variables.

        Args:
            params: Energy function parameters
            sim_state: Initial simulation state(s). Mulriple states can be
                       provided to run multiple trajectories in parallel.
            **kwargs: Kwargs to supply 'kT' and/or 'pressure' to change these
                      thermostat/barostat values on the fly. Can be constant
                      or function of t.

        Returns:
            TrajectoryState object containing the newly generated trajectory
        """
        # TODO unify with dyn_box
        timings = kwargs.pop('timings', ref_timings)
        assert timings is not None

        kwargs, (kbt, barostat_press) = _canonicalize_dynamic_state_kwargs(
            kwargs, timings.t_production_end, 'kT', 'pressure')

        energy_fn = energy_fn_template(params)
        _, apply_fn = simulator_template(energy_fn)
        run_to_printout = _run_to_next_printout_neighbors(apply_fn, timings,
                                                          **kwargs)

        if sim_state[0].position.ndim > 2:
            def run_trajectory(state, starting_time):
                state, trajectory = lax.scan(
                    run_to_printout, state, xs=starting_time)
                return state, trajectory

            if timings.t_equilib_start.size > 0:
                sim_state, _ = vmap(run_trajectory, (0, None))(  # equilibration
                    sim_state, timings.t_equilib_start)

            new_sim_state, traj = vmap(run_trajectory, (0, None))(  # production
                sim_state, timings.t_production_start)

            # combine parallel trajectories to single large one for streamlined
            # postprocessing via traj_quantity, DiffTRe, relative entropy, etc.
            traj = util.tree_combine(traj)
            overflow = jnp.any(new_sim_state[1].did_buffer_overflow)
            n_traj = sim_state[0].position.shape[0]
            kbt = _traj_replicate_if_not_none(kbt, n_traj)
            barostat_press = _traj_replicate_if_not_none(barostat_press, n_traj)

        else:
            if timings.t_equilib_start.size > 0:
                sim_state, _ = lax.scan(  # equilibration
                    run_to_printout, sim_state, xs=timings.t_equilib_start)

            new_sim_state, traj = lax.scan(  # production
                run_to_printout, sim_state, xs=timings.t_production_start)
            overflow = new_sim_state[1].did_buffer_overflow

        traj_state = TrajectoryState(sim_state=new_sim_state,
                                     trajectory=traj,
                                     overflow=overflow,
                                     thermostat_kbt=kbt,
                                     barostat_press=barostat_press)

        aux_trajectory = quantity_traj(traj_state, quantities, params,
                                       vmap_batch)
        return traj_state.replace(aux=aux_trajectory)

    return generate_reference_trajectory


def quantity_traj(traj_state, quantities, energy_params=None, batch_size=1):
    """Computes quantities of interest for all states in a trajectory.

    Arbitrary quantity functions can be provided via the quantities dict.
    The quantities dict should provide each quantity function via its own
    key that contains another dict containing the function under the
    'compute_fn' key. The resulting quantity trajectory will be saved in
    a dict under the same key as the input quantity function.

    Args:
        traj_state: TrajectoryState as output from trajectory generator
        quantities: The quantity dict containing for each target quantity
                    the snapshot compute function
        energy_params: Energy params for energy_fn_template to initialize
                       the current energy_fn
        batch_size: Number of batches for vmap

    Returns:
        A dict of quantity trajectories saved under the same key as the
        input quantity function.
    """
    if traj_state.sim_state[0].position.ndim > 2:
        last_state, fixed_reference_nbrs = util.tree_get_single(
            traj_state.sim_state)
    else:
        last_state, fixed_reference_nbrs = traj_state.sim_state
    npt_ensemble = util.is_npt_ensemble(last_state)

    @jit
    def single_state_quantities(single_snapshot):
        state, kbt = single_snapshot
        nbrs = util.neighbor_update(fixed_reference_nbrs, state)
        kwargs = {'neighbor': nbrs, 'energy_params': energy_params, 'kT': kbt}
        if npt_ensemble:
            box = simulate.npt_box(state)
            kwargs['box'] = box

        computed_quantities = {
            quantity_fn_key: quantities[quantity_fn_key](state, **kwargs)
            for quantity_fn_key in quantities
        }
        return computed_quantities

    batched_traj = util.tree_vmap_split(traj_state.trajectory, batch_size)
    if traj_state.thermostat_kbt is not None:
        thermo_kbt = traj_state.thermostat_kbt.reshape((-1, batch_size))
    else:
        thermo_kbt = traj_state.thermostat_kbt

    bachted_quantity_trajs = lax.map(
        vmap(single_state_quantities), (batched_traj, thermo_kbt)
    )
    quantity_trajs = util.tree_combine(bachted_quantity_trajs)
    return quantity_trajs


def average_predictions(quantity_trajs):
    """Computes average quantities for per-state quantiity_trajectories.

    Args:
        quantity_trajs: A dict containing per-state-quantities, e.g. as
                        generated from traj_util.quantity_traj.

    Returns:
        A dict containing ensemble-averaged quantities under the same keys as
        quantity_trajs.
    """
    average_quantities = {
        quantity_key: jnp.mean(quant_traj, axis=0)
        for quantity_key, quant_traj in quantity_trajs.items()
    }
    return average_quantities
