"""Implementation if IBI in JAX M.D. For simplicity, we assume same resolution and support for computed RDFs and
   Potentials on a homogeneous grid."""

import os
import sys


# TODO update to new versions and draw runfile out!

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)  # controls on which gpu the program runs

import numpy as onp

import jax.numpy as np
from jax import tree_util
# config.update("jax_debug_nans", True)
import time

from jax_md import util
from chemtrain.jax_md_mod import io
from chemtrain import ibi, difftre

from util import Postprocessing, Initialization

########   user input

type_fn = util.f32
file = '../examples/data/confs/SPC_FW_3nm.gro'  # 905 particles
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

R, v, box = io.load_configuration(file)  # initial configuration
simulation_data = Initialization.InitializationClass(R, v, box, kbT, mass, time_step)
rdf_struct = Initialization.select_target_RDF(target_rdf, bin_width_multiplier=1.)
timings_struct = difftre.process_printouts(time_step, total_time, t_equilib, print_every)

# TODO look that resoluion of reference RDF / x_vals matches the tabulated model
#  --> needs PMF_init=False in Initialization file

# if we want to use different table resolution from reference RDF resolution
# RDF_target = onp.where(RDF_target < 1.e-7, 0., RDF_target)  # fixes non-zeros introduced due to interpolation
# RDF_target = np.array(RDF_target)  # move to jax.np

x_vals = rdf_struct.rdf_bin_centers  # same support for tabulated potential and computed RDF --> simplifies IBI update
u_vals = np.array(ibi.initial_guess(rdf_struct.reference_rdf, kbT))

ibi.plot_verification_PMF(x_vals, u_vals)
u_vals = tree_util.tree_map(type_fn, u_vals)  # to convert to 64 bit from default 32 bit when needed

simulation_data, rdf_struct, timings_struct = tree_util.tree_map(  # cast all inputs to wanted datatype
    type_fn, (simulation_data, rdf_struct, timings_struct))

target_dict = {'rdf': rdf_struct}  # add all target values here here
reference_state, _, simulation_fns, quantity_fns, targets = \
    Initialization.initialize_simulation(simulation_data, model, target_dict, x_vals=x_vals)
simulator_template, energy_fn_template, neighbor_fn = simulation_fns
trajectory_generator = difftre.trajectory_generator_init(simulator_template,
                                                         energy_fn_template,
                                                         neighbor_fn,
                                                         timings_struct)

# IBI Loop
errors = onp.zeros(n_updates)
potentials = []
potentials.append(u_vals)  # also save initial guess
state = reference_state
RDF_list = []
for i in range(n_updates):
    t_start = time.time()
    traj_state = trajectory_generator(u_vals, state)
    state, traj, U_traj = traj_state
    quantity_traj = difftre.quantity_traj(traj_state, quantity_fns, neighbor_fn, u_vals)
    rdf_traj = quantity_traj['rdf']
    cur_RDF = np.mean(rdf_traj, axis=0)  # TODO check this works
    # cur_RDF = np.mean(reweighting.compute_RDF_snapshots(R_traj, rdf_fn), axis=0)
    RDF_list.append(cur_RDF)
    errors[i] = difftre.mse_loss(cur_RDF, rdf_struct.reference_rdf)
    u_vals = ibi.update_potential(cur_RDF, rdf_struct.reference_rdf, u_vals, kbT)
    potentials.append(u_vals)
    print('Iteration time:', ((time.time() - t_start) / 60.), 'mins. Error:', errors[i])

# plot results
Postprocessing.plot_initial_and_predicted_rdf(rdf_struct.rdf_bin_centers, RDF_list[-1], model, visible_device,
                                              rdf_struct.reference_rdf, RDF_list[0])
Postprocessing.plot_loss_and_gradient_history(errors, [], visible_device=visible_device)
ibi.plot_potential_iterations(potentials)
