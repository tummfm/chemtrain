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

"""Implementation if IBI in JAX M.D. For simplicity, we assume same resolution and support for computed RDFs and
   Potentials on a homogeneous grid."""

# TODO update to new versions and draw runfile out!

import os
import sys
import time

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

from jax import tree_util, numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from scipy import interpolate as sci_interpolate

from jax_md_mod import io
from chemtrain import util
from chemtrain.trajectory import traj_util
from chemtrain.learn import ibi, max_likelihood
from util import Postprocessing, Initialization

########   user input

def plot_verification_PMF(x_vals, energy_PMF_comparison):
    def load_table(location, x_vals):
        """

        Assumes that r values are stored in first column and corresponding
        values in the second column.
        """
        tabulated_array = onp.loadtxt(location)
        y_interpolator = sci_interpolate.interp1d(tabulated_array[:, 0],
                                                  tabulated_array[:, 1],
                                                  kind='cubic')
        y_vals = onp.array(y_interpolator(x_vals))
        return y_vals

    initial_guess_table_loc = 'data/IBI_Initial_guess.csv'
    u_vals_reference = load_table(initial_guess_table_loc, x_vals)
    plt.figure()
    plt.plot(x_vals, energy_PMF_comparison, label='My_PMF')
    plt.plot(x_vals, u_vals_reference, label='Reference', linestyle='--')
    plt.legend()
    plt.ylim(-2.5, 10.)
    plt.savefig('Figures/PMF.png')
    return

def plot_potential_iterations(potentials):
    plt.figure()
    for i, u_vals in enumerate(potentials):
        plt.plot(u_vals, label=str(i))
    plt.ylim(-3., 3.)
    plt.legend()
    plt.savefig('Figures/IBI_Iterations.png')
    return

type_fn = util.f32
file = 'data/confs/SPC_FW_3nm.gro'  # 905 particles
# file = 'data/confs/SPC_FW_2nm.gro'  # 229 particles

target_rdf = 'SPC'
model = 'Tabulated'

kbT = 2.49435321
mass = 18.0154
time_step = 0.002

total_time = 20.
t_equilib = 3.  # equilibration time before sampling RDF
print_every = 0.1

n_updates = 20

# Note: One might also need to change the model in Initialization.py

#############################################

box, R, _, _ = io.load_box(file)  # initial configuration
simulation_data = Initialization.InitializationClass(R, box, kbT, mass, time_step)
rdf_struct = Initialization.select_target_rdf(target_rdf)
timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

# TODO look that resoluion of reference RDF / x_vals matches the tabulated model
#  --> needs PMF_init=False in Initialization file

# if we want to use different table resolution from reference RDF resolution
# RDF_target = onp.where(RDF_target < 1.e-7, 0., RDF_target)  # fixes non-zeros introduced due to interpolation
# RDF_target = np.array(RDF_target)  # move to jax.np

x_vals = rdf_struct.rdf_bin_centers  # same support for tabulated potential and computed RDF --> simplifies IBI update
u_vals = jnp.array(ibi.initial_guess(rdf_struct.reference, kbT))

plot_verification_PMF(x_vals, u_vals)
u_vals = tree_util.tree_map(type_fn, u_vals)  # to convert to 64 bit from default 32 bit when needed

simulation_data, rdf_struct, timings = tree_util.tree_map(  # cast all inputs to wanted datatype
    type_fn, (simulation_data, rdf_struct, timings))

target_dict = {'rdf': rdf_struct}  # add all target values here here
reference_state, _, simulation_fns, quantity_fns, _ = \
    Initialization.initialize_simulation(simulation_data, model, target_dict,
                                         x_vals=x_vals)
simulator_template, energy_fn_template, neighbor_fn = simulation_fns
trajectory_generator = traj_util.trajectory_generator_init(simulator_template,
                                                           energy_fn_template,
                                                           timings)

# IBI Loop
errors = onp.zeros(n_updates)
potentials = []
potentials.append(u_vals)  # also save initial guess
RDF_list = []
traj_state = trajectory_generator(u_vals, reference_state)
for i in range(n_updates):
    t_start = time.time()
    traj_state = trajectory_generator(u_vals, traj_state.sim_state)
    quantity_traj = traj_util.quantity_traj(traj_state, quantity_fns, u_vals)
    rdf_traj = quantity_traj['rdf']
    cur_RDF = jnp.mean(rdf_traj, axis=0)  # TODO check this works
    # cur_RDF = np.mean(reweighting.compute_RDF_snapshots(R_traj, rdf_fn), axis=0)
    RDF_list.append(cur_RDF)
    errors[i] = max_likelihood.mse_loss(cur_RDF, rdf_struct.reference)
    u_vals = ibi.update_potential(cur_RDF, rdf_struct.reference, u_vals, kbT)
    potentials.append(u_vals)
    print('Iteration time:', ((time.time() - t_start) / 60.), 'mins. Error:', errors[i])

# plot results
Postprocessing.plot_initial_and_predicted_rdf(rdf_struct.rdf_bin_centers, RDF_list[-1], model, visible_device,
                                              rdf_struct.reference, RDF_list[0])
Postprocessing.plot_loss_and_gradient_history(errors, [], visible_device=visible_device)
plot_potential_iterations(potentials)
