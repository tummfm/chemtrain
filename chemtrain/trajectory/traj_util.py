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
import functools
from functools import partial
from typing import Any, Dict, Callable, Mapping, Tuple

import numpy as onp

import chex

from jax import lax, jit, vmap, numpy as jnp, random, tree_util

from jax_md import simulate, util as jax_md_util
from jax_md.partition import NeighborList

from chemtrain import util
from chemtrain.jax_md_mod import custom_quantity

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
        thermostat_kbt: Target thermostat kbT at time of respective snapshots
        barostat_press: Target barostat pressure at time of respective snapshots
        aux: Dict of auxilary per-snapshot quantities as defined by quantities
            in trajectory generator.
        key: PRNGKey of the trajectory state.
        energy_params: Energy parameters used to generate the trajectory.
        entropy_diff: Entropy difference estimated for the trajectory, e.g.,
            via DiffTRe optimization
        free_energy_diff: Free energy difference estimated for the trajectory,
            e.g., via DiffTRe optimization
    """
    sim_state: Tuple[Any, Any]
    trajectory: Any
    overflow: Array = False
    thermostat_kbt: Array = None
    barostat_press: Array = None
    aux: Dict[str, Any] = None
    key: Array = None
    energy_params: Any = None
    entropy_diff: Array = 0.0
    free_energy_diff: Array = 0.0


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


def initialize_simulator_template(init_simulator_fn,
                                  shift_fn: Callable,
                                  nbrs: NeighborList = None,
                                  init_with_PRNGKey: bool = True,
                                  extra_simulator_kwargs: Mapping = None):
    """Initializes the simulator template and reference state.

    Args:
        init_simulator_fn: Function returning a ``(init_fn, apply_fn)`` tuple
            when provided with a potential energy function.
        shift_fn: Function shifting positions back into the simulation box
            after a simulation update step.
        nbrs: Neighbor list to allocate new neighbor list based on particle
            positions.
        init_with_PRNGKey: Whether simulator init function takes an PRNGKey,
            should be set to False e.g. for the Gradient Descend energy
            minimization routine.
        extra_simulator_kwargs: Additional arguments when initializing the
            simulator state.

    Returns:
        Returns a function to initialize the simulator state and the
        corresponding simulator template.

    """
    if extra_simulator_kwargs is None:
        extra_simulator_kwargs = {}

    simulator_template = functools.partial(
        init_simulator_fn, shift_fn=shift_fn, **extra_simulator_kwargs)

    def init_reference_state(key, r_init, energy_or_force_fn,
                             init_sim_kwargs=None, init_nbrs_kwargs=None):
        if init_nbrs_kwargs is None:
            init_nbrs_kwargs = {}
        if init_sim_kwargs is None:
            init_sim_kwargs = {}

        init_simulator, _ = simulator_template(energy_or_force_fn)

        def _single_init_fn(key, r_init):
            if r_init.ndim > 2:
                # Initialize vectorized by calling function recursively until
                # only single conformation is left
                splits = random.split(key, r_init.shape[0])
                return vmap(_single_init_fn, in_axes=0)(splits, r_init)

            nonlocal nbrs
            if nbrs is not None:
                nbrs = nbrs.update(r_init, **init_nbrs_kwargs)

            if init_with_PRNGKey:
                init_state = init_simulator(key, r_init, **init_sim_kwargs)
            else:
                init_state = init_simulator(r_init, **init_sim_kwargs)

            return init_state, nbrs

        # Check no overflow during neighborlist update
        init_state = _single_init_fn(key, r_init)

        assert not onp.any(init_state[1].did_buffer_overflow), (
            "Overflow during initialization of trajectories. Provided a "
            "neighbor list with more capacity."
        )

        return init_state

    return init_reference_state, simulator_template


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
                                     barostat_press=barostat_press,
                                     energy_params=params,
                                     )

        aux_trajectory = quantity_traj(traj_state, quantities, params,
                                       vmap_batch)
        return traj_state.replace(aux=aux_trajectory)

    return generate_reference_trajectory


def quantity_traj(traj_state, quantities, energy_params=None, batch_size=1):
    """Computes quantities of interest for all states in a trajectory.

    Arbitrary quantity functions can be provided via the quantities-dict.
    The quantities dict provides the function to compute the quantity on
    a single snapshot. The resulting quantity trajectory will be saved in
    a dict under the same key as the input quantity function.

    Example usage:
        .. code-block:: python

            def custom_compute_fn(state, neighbor=None, **kwargs):
                ...
                return quantity_snapshot


            quantities = {
                'energy': custom_quantity.energy_wrapper(energy_template_fn),
                'custom_quantity': custom_compute_fn
            }

            quantity_trajs = quantity_traj(traj_state, quantities, energy_params)
            custom_quantity = quantity_trajs['custom_quantity']


    Args:
        traj_states: TrajectoryStates as output from trajectory generator
        quantities: The quantity dict containing for each target quantity
            the snapshot compute function
        energy_params: Energy params for energy_fn_template to initialize
            the current energy_fn
        batch_size: Number of batches for vmap

    Returns:
        A dict of quantity trajectories saved under the same key as the
        input quantity function.
    """
    return quantity_traj_multimap(
        traj_state, quantities=quantities,
        energy_params=energy_params, batch_size=batch_size)


def quantity_traj_multimap(*traj_states, quantities=None, energy_params=None, batch_size=1):
    """Computes quantities of interest for all states in a trajectory.

    This function extends :func:`quantity_traj`
    to quantities with respect to multiple reference states.
    Therefore, the quantity function signature changes to

    .. code-block:: python

            def quantity_fn(*states, neighbor=None, energy_params=None, **kwargs):
                ...

    The keywords arguments, i.e. the neighbor list, are with respect to the
    first state of `*states`.

    Args:
        *traj_states: TrajectoryStates as output from trajectory generator
        quantities: The quantity dict containing for each target quantity
                    the snapshot compute function
        energy_params: Energy params for energy_fn_template to initialize
                       the current energy_fn
        batch_size: Number of batches for vmap

    Returns:
        A dict of quantity trajectories saved under the same key as the
        input quantity function.
    """
    # Check that all states have the same format
    assert len(traj_states) > 0, 'Need at least one trajectory state.'
    ref_leaves, ref_struct = tree_util.tree_flatten(traj_states[0])
    for state in traj_states:
        assert ref_struct == tree_util.tree_structure(state), (
            "All trajectory states must have the same tree structure."
        )
        assert onp.all([
            jnp.shape(l) == jnp.shape(r)
            for r, l in zip(ref_leaves, tree_util.tree_leaves(state))
        ]), "All trajectory state leaves must be of identical shape."


    print(f"Traj states in beginning of multimap")
    print(tree_util.tree_map(jnp.shape, traj_states))

    if traj_states[0].sim_state[0].position.ndim > 2:
        last_state, fixed_reference_nbrs = util.tree_get_single(
            traj_states[0].sim_state)
    else:
        last_state, fixed_reference_nbrs = traj_states[0].sim_state
    npt_ensemble = util.is_npt_ensemble(last_state)

    @jit
    def single_state_quantities(single_snapshot):
        states, kbt = single_snapshot
        nbrs = util.neighbor_update(fixed_reference_nbrs, states[0])
        kwargs = {'neighbor': nbrs, 'energy_params': energy_params, 'kT': kbt}
        if npt_ensemble:
            box = simulate.npt_box(states[0])
            kwargs['box'] = box

        print(states)

        if len(states) == 1:
            computed_quantities = {
                quantity_fn_key: quantities[quantity_fn_key](states[0], **kwargs)
                for quantity_fn_key in quantities
            }
        else:
            computed_quantities = {
                quantity_fn_key: quantities[quantity_fn_key](*states, **kwargs)
                for quantity_fn_key in quantities
            }
        return computed_quantities

    batched_trajs = [
        util.tree_vmap_split(state.trajectory, batch_size)
        for state in traj_states
    ]

    if traj_states[0].thermostat_kbt is not None:
        thermo_kbt = traj_states[0].thermostat_kbt.reshape((-1, batch_size))
    else:
        thermo_kbt = traj_states[0].thermostat_kbt

    print(f"Batched traj states")
    print(tree_util.tree_map(jnp.shape, (batched_trajs, thermo_kbt)))

    bachted_quantity_trajs = lax.map(
        vmap(single_state_quantities), (batched_trajs, thermo_kbt)
    )
    quantity_trajs = util.tree_combine(bachted_quantity_trajs)
    return quantity_trajs
