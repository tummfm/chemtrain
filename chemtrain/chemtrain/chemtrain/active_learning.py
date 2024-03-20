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

"""Utility functions for active learning applications."""
import functools
import gc
from importlib import reload

from jax import jit, numpy as jnp
from jax.experimental import host_callback

from chemtrain import traj_util, reweighting


def _continue_md(uncertainties, uncertainty_threshold, time, t_end):
    max_uncertainty = jnp.max(uncertainties)
    small_uncertainty = max_uncertainty < uncertainty_threshold
    end_not_reached = time < t_end
    return jnp.alltrue(jnp.stack([small_uncertainty, end_not_reached]))


def _forward_prod_times(old_timings, t_production):
    """Increase simulation time for next part of non-equilibrium MD."""
    new_t_start = old_timings.t_production_start + t_production
    new_t_end = old_timings.t_production_end + t_production
    return old_timings.replace(t_production_start=new_t_start,
                               t_production_end=new_t_end)


def init_estimate_uncertainty(quantities, statepoint_grid, uncertainty_fn=None):

    if uncertainty_fn is None:
        uncertainty_fn = reweighting.init_default_loss_fn

    def statepoint_uncertainty(traj, quantity_traj, target_kbt, target_press):
        weights = reweighting.reweight_trajectory(traj, kT=target_kbt,
                                                  pressure=target_press)
        uncertainty, predictions = uncertainty_fn(quantity_traj, weights, targets)


    def estimate_uncertainty(traj, energy_params):
        quantity_traj = traj_util.quantity_traj(traj, quantities, energy_params)



        return statepoints, uncertainties

    return estimate_uncertainty


def init_uq_md(energy_fn_template, simulator_template, timings, t_end,
               uncertainty_fn, uncertainty_threshold, kt_schedule=None,
               press_schedule=None):
    # Doc: if returned statepoint=None: Simulation ran through without

    trajectory_generator = traj_util.trajectory_generator_init(
        simulator_template, energy_fn_template)
    t_production = (timings.t_production_start[0]
                    - timings.t_production_end[-1])
    zero_array = jnp.array([], dtype=jnp.float32)

    # Not jitting the while-loop gives more freedom in postprocessing results,
    # i.e. storing all uncertainty values over the whole non-equilibrium MD
    # trajectory. We therefore only jit the expensive body functions
    trajectory_generator = jit(trajectory_generator)

    # TODO orient initialization more like trainers

    def uq_md(params, init_sim_state):

        # init trajectory responsible for initial equilibration
        equilib_timings = timings.replace(t_production_start=zero_array)
        traj = trajectory_generator(
            params, init_sim_state, timings=equilib_timings, kT=kt_schedule,
            pressure=press_schedule)

        # after initial, no more equilibration needed during non-equilibrium MD
        prod_timings = timings.replace(t_equilib_start=zero_array)

        continue_md = True
        while continue_md:
            traj = trajectory_generator(
                params, traj.sim_state, timings=prod_timings,  kT=kt_schedule,
                pressure=press_schedule)
            prod_timings = _forward_prod_times(prod_timings, t_production)
            cur_time = prod_timings.t_production_start[0]
            statepoints, uncertainties = uncertainty_fn(traj)
            continue_md = _continue_md(uncertainties, uncertainty_threshold,
                                       cur_time, t_end)


        converged = (cur_time >= t_end)

        return state, converged

    return uq_md
