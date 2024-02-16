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

import numpy as onp

import chex
from jax import lax, vmap, pmap, numpy as jnp, tree_util, random, debug
try:
    from jax.typing import ArrayLike
except:
    ArrayLike = Any
from jax_md import simulate, util as jax_md_util

from chemtrain import util
from chemtrain.jax_md_mod import custom_quantity
from chemtrain.pickle_jit import jit
from chemtrain.typing import QuantityDict

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
    entropy_diff: Entropy difference towards a reference point
    free_energy_diff: Free energy difference towards a reference point
    energy_params: Parameters of the potential model used to generate the
        trajectory
    aux: Dict of auxilary per-snapshot quantities as defined by quantities
         in trajectory generator.
    """
    sim_state: Any
    trajectory: Any
    overflow: Array = False
    thermostat_kbt: Array = None
    barostat_press: Array = None
    entropy_diff: Array = 0.0
    free_energy_diff: Array = 0.0
    energy_params: Any = None
    aux: Dict = None


def committee_energy_fn(energy_fn_template,
                        strategy="vmap",
                        evaluation="mean",
                        weights=0.0,
                        prior_energy_fn=None,
                        ref_ssq=None):
    """Initializes committee potential model.

    Args:
        energy_fn_template: Potential model of an individual committee member
        strategy: Evaluate the potential energies vectorized
            (``strategy="vmap"``) or in parallel (``strategy="pmap"``)
        evaluation: Strategy to combine the committee member energies, e.g. via
            calculating the mean (``evaluation="mean"``) or selection the median
            energy (``evaluation="median"``)
        weights: Weights of the energies

    Returns:
        Returns a new energy function template that takes a list of individual
        committee member parameters.

    """

    # Create a function that can be parallelized over the energy parameters.
    # Collecting args and kwargs of the energy function simplifies the
    # implementation.
    def energy_fn(flat_params, rest, structure=None):
        params = tree_util.tree_unflatten(structure, flat_params)
        args, kwargs = rest
        parametrized_template = energy_fn_template(params)
        return parametrized_template(*args, **kwargs)

    # New template function
    def mean_energy_fn_template(params):
        # Concatenate all leaves along the first axis such that vmapping or
        # pmapping is possible
        structure = tree_util.tree_structure(params[0])
        leaves = [tree_util.tree_leaves(t) for t in params]
        concat_leaves = [jnp.stack(ls, axis=0) for ls in zip(*leaves)]
        concrete_energy_fn = partial(energy_fn, structure=structure)

        # Evaluate the energy for all potentials
        if strategy == "vmap":
            batched_energy_fn = vmap(concrete_energy_fn, in_axes=(0, None))
        elif strategy == "pmap":
            batched_energy_fn = pmap(concrete_energy_fn, in_axes=(0, None))
        elif strategy == "sequential":
            batched_energy_fn = lambda energy_params, arg: lax.map(
                partial(concrete_energy_fn, rest=arg), energy_params)
        else:
            assert NotImplementedError("Only vmap and pmap are currently "
                                       "implemented.")

        def committee_energy(*args, **kwargs):
            energies = batched_energy_fn(concat_leaves, (args, kwargs))
            if evaluation == "mean":
                return jnp.mean(energies)
            elif evaluation == "weighted":
                return jnp.sum(weights * energies) / jnp.sum(weights)
            if evaluation == "robust":
                baseline_energy = prior_energy_fn(*args, **kwargs)
                sqf =  jnp.var(energies, ddof=1)
                scale = 1 / ((1. / sqf) + (1. / ref_ssq))

                committee_energy = baseline_energy / sqf
                committee_energy += jnp.mean(energies) / ref_ssq
                committee_energy *= scale
                return committee_energy
            else:
                raise NotImplementedError(f"Unknown method '{evaluation} to "
                                          f"combine the individual energies.")

        return committee_energy
    return mean_energy_fn_template


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


def init_dynamic_key(timings: TimingClass, init_key: Array=None):
    """Initializes a function to get the time-dependent key.

    Args:
        timings: Timings of the simulation, necessary to infer the total number
            of simulation steps.
        init_key: Initial key from which the time-dependent keys are split.

    Returns:
        Returns a functions that returns a time-dependent PRNGKey.

    """

    if init_key is None:
        init_key = random.PRNGKey(0)

    t_start = timings.t_equilib_start[0]
    t_end = timings.t_production_end[-1]
    total_steps = int((t_end - t_start) / timings.time_step)

    def get_key(t, keys=None):
        n = jnp.int_((t - t_start) / timings.time_step)
        return keys[n, :]

    return partial(get_key, keys=random.split(init_key, total_steps))


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
    def get_apply_kwargs(t, chain=None):
        apply_kwargs = {}
        if 'kT' in kwargs:
            apply_kwargs['kT'] = kwargs['kT'](t, chain=chain)
        if 'pressure' in kwargs:
            apply_kwargs['pressure'] = kwargs['pressure'](t)
        if 'dropout_key' in kwargs:
            apply_kwargs['dropout_key'] = kwargs['dropout_key'](t)
        return apply_kwargs

    def do_step(cur_state, t, chain=None):
        apply_kwargs = get_apply_kwargs(t, chain)

        state, nbrs = cur_state
        new_state = apply_fn(state, neighbor=nbrs, **apply_kwargs)
        nbrs = util.neighbor_update(nbrs, new_state)
        new_sim_state = (new_state, nbrs)
        return new_sim_state, t

    # @jit  # this triggers bug in JAX
    def run_small_simulation(start_state, t_start=0., chain=None):
        simulation_time_points = jnp.arange(timings.timesteps_per_printout) \
                                 * timings.time_step + t_start
        printout_state, _ = lax.scan(
            partial(do_step, chain=chain), start_state,
            xs=simulation_time_points)

        cur_state, _ = printout_state
        t_printout = simulation_time_points[-1] + timings.time_step
        return printout_state, cur_state, get_apply_kwargs(t_printout, chain)
    return run_small_simulation


def _canonicalize_dynamic_state_kwargs(state_kwargs, t_snapshots, *keys):
    """Converts constant state_kwargs, such as 'kT' and 'pressure' to constant
    functions over time and deletes None kwargs. Additionally, return the
    values of state_kwargs at production printout times.
    """
    def constant_fn(_, c, chain=None):
        # In the case of multiple temperatures
        if chain is None or jnp.isscalar(c):
            return c
        else:
            return c[chain]

    state_point_vals = {}
    for key in keys:
        if key in state_kwargs:
            if state_kwargs[key] is None:
                state_kwargs.pop(key)  # ignore kwarg if None is provided
                state_points = None
            else:
                if not callable(state_kwargs[key]):
                    state_kwargs[key] = partial(constant_fn,
                                                c=state_kwargs[key])
                state_points = vmap(state_kwargs[key])(t_snapshots)
        else:
            state_points = None
        state_point_vals[key] = state_points
    return state_kwargs, state_point_vals


def _traj_replicate_if_not_none(thermostat_values, n_traj):
    """Replicates thermostat targets to multiple trajectories, if not None."""
    if thermostat_values is not None:
        if thermostat_values.ndim == 1:
            thermostat_values = jnp.tile(thermostat_values, n_traj)
        else:
            thermostat_values = jnp.ravel(thermostat_values)
    return thermostat_values


def trajectory_generator_init(simulator_template, energy_fn_template,
                              ref_timings=None, quantities=None, vmap_batch=10,
                              replica_exchange=False):
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
        replica_exchange: Accelerate exploration by simulating at multiple
            temperatures.

    Returns:
        A function taking energy params and the current traj_state (including
        neighbor list) that runs the simulation forward generating the
        next TrajectoryState.
    """
    if quantities is None:
        quantities = {}

    # temperature is inexpensive and generally useful: compute it by default
    quantities['kbT'] = custom_quantity.temperature
    # compute total energy required for the exchange of replicas
    quantities['energy'] = custom_quantity.energy_wrapper(energy_fn_template)
    # save the dropout key used for the model evaluation
    quantities['dropout_key'] = lambda state, dropout_key=None, **kwargs: dropout_key

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

        kwargs, printout_vals = _canonicalize_dynamic_state_kwargs(
            kwargs, timings.t_production_end, 'kT', 'pressure', 'dropout_key')

        energy_fn = energy_fn_template(params)
        _, apply_fn = simulator_template(energy_fn)
        run_to_printout = _run_to_next_printout_neighbors(apply_fn, timings,
                                                          **kwargs)

        if replica_exchange:
            kbt = printout_vals['kT'][:, 0]
        else:
            kbt = printout_vals['kT']
        barostat_press = printout_vals['pressure']

        if replica_exchange:
            nchains = sim_state[0].position.shape[0]

            @partial(vmap, in_axes=(0, 0, None))
            def proposal_fn(chain, state, starting_time):
                # Do a forward step
                printout_state, sample, apply_kwargs = run_to_printout(
                    state, starting_time, chain)
                sim_state, nbrs = printout_state

                energy = energy_fn_template(params)(
                    sim_state.position, neighbor=nbrs, **apply_kwargs)

                return printout_state, sample, apply_kwargs, energy

            def sample_fn(states, starting_time):
                states, samples, apply_kwargs, energies = proposal_fn(
                    jnp.arange(nchains), states, starting_time
                )

                # Perform the swap steps
                (states, _), swaps = lax.scan(
                    partial(replica_exchange_fn, apply_kwargs=apply_kwargs),
                    (states, energies), onp.arange(nchains - 1)
                )

                #debug.print("Swaps {w} with apply kwargs{apply} and energies {energies}", w=swaps, apply=apply_kwargs, energies=energies)

                # Only return the samples from the untempered chain
                samples = util.tree_get_single(samples, 0)

                return states, (samples, swaps)

            if timings.t_equilib_start.size > 0:
                sim_state, _ = lax.scan(  # equilibration
                    sample_fn, sim_state, xs=timings.t_equilib_start)

            new_sim_state, (traj, swaps) = lax.scan(  # production
                sample_fn, sim_state, xs=timings.t_production_start)
            overflow = jnp.any(new_sim_state[1].did_buffer_overflow)

            mean_swap = jnp.mean(swaps, axis=0)
            std_swap = jnp.std(swaps, axis=0)

            debug.print("Swaps per temperatures:\n\tmu: {mu}\n\tstd: {std}", mu=mean_swap, std=std_swap)


        elif sim_state[0].position.ndim > 2:

            # Ignore the energies
            @partial(vmap, in_axes=(0, (None, 0)))
            def sample_fn(state, xs):
                starting_time, chain = xs
                *args, _ = run_to_printout(state, starting_time, chain=chain)
                return args

            def run_trajectory(state, starting_time):
                chains = jnp.arange(sim_state[0].position.shape[0])
                chains = jnp.tile(chains, (starting_time.size, 1))
                xs = (starting_time, chains)
                state, trajectory = lax.scan(sample_fn, state, xs=xs)
                return state, trajectory

            # TODO: The run to printout function must be vectorized not the scan function
            # Equilibration
            if timings.t_equilib_start.size > 0:
                sim_state, _ = run_trajectory(
                    sim_state, timings.t_equilib_start)

            new_sim_state, traj = run_trajectory(
                sim_state, timings.t_production_start)


            # combine parallel trajectories to single large one for streamlined
            # postprocessing via traj_quantity, DiffTRe, relative entropy, etc.
            traj = util.tree_combine(traj)
            overflow = jnp.any(new_sim_state[1].did_buffer_overflow)
            n_traj = sim_state[0].position.shape[0]
            kbt = _traj_replicate_if_not_none(kbt, n_traj)
            barostat_press = _traj_replicate_if_not_none(barostat_press, n_traj)
        else:
            def sample_fn(state, starting_time):
                *args, _ = run_to_printout(state, starting_time)
                return args

            if timings.t_equilib_start.size > 0:
                sim_state, _ = lax.scan(  # equilibration
                    sample_fn, sim_state, xs=timings.t_equilib_start)

            new_sim_state, traj = lax.scan(  # production
                sample_fn, sim_state, xs=timings.t_production_start)
            overflow = new_sim_state[1].did_buffer_overflow

        traj_state = TrajectoryState(sim_state=new_sim_state,
                                     trajectory=traj,
                                     overflow=overflow,
                                     thermostat_kbt=kbt,
                                     barostat_press=barostat_press,
                                     energy_params=params,
                                     )

        aux_trajectory = quantity_traj(traj_state, quantities, params,
                                       vmap_batch, printout_vals.get("dropout_key"))
        return traj_state.replace(aux=aux_trajectory)

    return generate_reference_trajectory


def quantity_traj(traj_state: TrajectoryState,
                  quantities: QuantityDict,
                  energy_params: Any = None,
                  batch_size: int = 1,
                  dropout_key: ArrayLike = None):
    """Computes quantities of interest for all states in a trajectory.

    Arbitrary quantity functions can be provided via the quantities-dict.
    The quantities-dict should provide each quantity function via its own
    key that contains another dict containing the function under the
    ``"compute_fn"`` key. The resulting quantity trajectory will be saved in
    a dict under the same key as the input quantity function.

    Args:
        traj_state: TrajectoryState as output from trajectory generator
        quantities: The quantity dict containing for each target quantity
            the snapshot compute function
        energy_params: Energy params for energy_fn_template to initialize
            the current energy_fn
        batch_size: Number of batches for vmap
        dropout_key: Predict energies while using dropout.

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
        state, dyn_kwargs = single_snapshot
        nbrs = util.neighbor_update(fixed_reference_nbrs, state)
        kwargs = {'neighbor': nbrs, 'energy_params': energy_params}
        kwargs.update(dyn_kwargs)
        if npt_ensemble:
            box = simulate.npt_box(state)
            kwargs['box'] = box

        computed_quantities = {
            quantity_fn_key: quantities[quantity_fn_key](state, **kwargs)
            for quantity_fn_key in quantities
        }
        return computed_quantities

    if traj_state.thermostat_kbt is not None:
        thermo_kbt = traj_state.thermostat_kbt.reshape((-1, batch_size))
    else:
        thermo_kbt = traj_state.thermostat_kbt

    dynamic_kwargs = {"kT": thermo_kbt}
    if dropout_key is not None:
        print(f"Warning: Dropout key is used")
        num_samples = traj_state.trajectory.position.shape[0]
        if dropout_key.shape == (num_samples, 2):
            keys = dropout_key.reshape((-1, batch_size, 2))
        else:
            keys = jnp.tile(dropout_key, (num_samples // batch_size, batch_size, 1))
            # keys = random.split(dropout_key, num_samples)
            # keys = jnp.reshape(keys, (num_samples // batch_size, batch_size, -1))
        dynamic_kwargs["dropout_key"] = keys

    batched_traj = util.tree_vmap_split(traj_state.trajectory, batch_size)
    batched_quantity_trays = lax.map(
        vmap(single_state_quantities), (batched_traj, dynamic_kwargs)
    )
    quantity_trajs = util.tree_combine(batched_quantity_trays)
    return quantity_trajs


def average_predictions(quantity_trajs):
    """Computes average quantities for per-state quantity_trajectories.

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

def replica_exchange_fn(state, idx, apply_kwargs=None):
    """Exchange replicas from simulations at different temperatures.

    """
    (simstates, nbrs), energies = state
    sim1 = util.tree_get_single(simstates, idx)
    sim2 = util.tree_get_single(simstates, idx + 1)
    split = random.split(sim1.rng, 1)

    e1 = energies[idx]
    e2 = energies[idx + 1]

    kT1 = apply_kwargs['kT'][idx]
    kT2 = apply_kwargs['kT'][idx + 1]
    beta1 = 1. / kT1
    beta2 = 1. / kT2

    delta = (beta1 - beta2) * (e2 - e1)
    swap = -jnp.log(random.uniform(split)) > delta

    pos1 = jnp.where(swap, sim2.position, sim1.position)
    pos2 = jnp.where(swap, sim1.position, sim2.position)
    mom1 = jnp.where(swap, jnp.sqrt(kT1 / kT2) * sim2.momentum, sim1.momentum)
    mom2 = jnp.where(swap, jnp.sqrt(kT2 / kT1) * sim1.momentum, sim2.momentum)
    eng1 = jnp.where(swap, e2, e1)
    eng2 = jnp.where(swap, e1, e2)

    # Update the positions and momenta
    pos = simstates.position
    mom = simstates.momentum

    pos = pos.at[idx].set(pos1)
    pos = pos.at[idx + 1].set(pos2)
    mom = mom.at[idx].set(mom1)
    mom = mom.at[idx + 1].set(mom2)
    energies = energies.at[idx].set(eng1)
    energies = energies.at[idx + 1].set(eng2)

    simstates = simstates.set(momentum=mom)
    simstates = simstates.set(position=pos)

    return ((simstates, nbrs), energies), swap
