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

"""Currently only forward simulation of AT water model."""
import os
import sys

import mdtraj

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)
os.environ['CHEMTRAIN_VERBOSE'] = str(True)

sys.path.append("../../")

import cloudpickle as pickle
from pathlib import Path
import time

from jax import vmap, random, tree_util, numpy as jnp
from jax_md import space
import numpy as onp

from chemtrain.jax_md_mod import io, custom_space, custom_quantity
from chemtrain import util, data_processing, traj_util
from util import Initialization
import visualization
from chemtrain.potential.prior import ForceField, Topology, init_prior_potential


save_name = 'prior'
folder_name = 'output/prior'
labels = ['Reference', 'Predicted']

file_topology = 'data/confs/heavy_2_7nm.gro'
configuration_str = '../../../../Datasets/Alanine/confs_heavy_100ns.npy'
used_dataset_size = 500000
n_trajectory = 10
top = mdtraj.load_topology(file_topology)

model = 'Tabulated'

# saved_trainer_path = 'output/force_matching/trained_model.pkl'
saved_trainer_path = None

saved_params_path = 'output/force_matching/best_params.pkl'
saved_params_path = 'output/rel_entropy/trained_params_alanine.pkl'

system_temperature = 300  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant
time_step = 0.001

total_time = 1100
t_equilib = 100.
print_every = 0.1
target_dict = None

###############################

Path(f'output/postprocessing/{folder_name}').mkdir(parents=True,
                                                   exist_ok=True)
Path('output/trajectories').mkdir(parents=True, exist_ok=True)

box, _, _, _ = io.load_box(file_topology)
position_data = data_processing.get_dataset(configuration_str)
key = random.PRNGKey(0)
r_init = random.choice(key, position_data, (n_trajectory,), replace=False)



force_field = ForceField.load_ff("data/force_fields/heavy.toml")
topology = Topology.from_mdtraj(top, mapping=force_field.mapping(by_name=True))
energy_params = force_field

masses = force_field.get_nonbonded_params(topology.get_atom_species())[0][:, 0]
species = topology.get_atom_species()

displacement, _ = space.periodic_general(box, fractional_coordinates=True)

prior_fn_template = init_prior_potential(displacement, mask_bonded=True)

def energy_fn_template(force_field):
    return prior_fn_template(topology, force_field)


simulation_data = Initialization.InitializationClass(
    r_init=r_init, box=box, kbt=kbt, masses=masses, dt=time_step,
    species=species)
timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

reference_state, _, simulation_fns, _, _ = \
    Initialization.initialize_simulation(simulation_data,
                                         model,
                                         integrator='Langevin',
                                         prior_constants={},
                                         prior_idxs={})

simulator_template, _, neighbor_fn = simulation_fns


trajectory_generator = traj_util.trajectory_generator_init(simulator_template,
                                                           energy_fn_template,
                                                           timings)

# compute trajectory and quantities
t_start = time.time()
traj_state = trajectory_generator(energy_params, reference_state, kT=kbt)
t_end = time.time() - t_start
print('total runtime:', t_end)

assert not traj_state.overflow, ('Neighborlist overflow during trajectory '
                                 'generation. Increase capacity and re-run.')

# Postprocessing
traj_positions = traj_state.trajectory.position
jnp.save(f'output/trajectories/confs_alanine_{save_name}', traj_positions)

box_tensor, _ = custom_space.init_fractional_coordinates(box)
displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)
position_data = data_processing.scale_dataset_fractional(position_data, box)

# dihedrals
dihedral_idxs = jnp.array([[1, 3, 4, 6], [3, 4, 6, 8]])  # 0: phi    1: psi
batched_dihedrals = vmap(custom_quantity.dihedral_displacement, (0, None, None))

dihedrals_ref = batched_dihedrals(position_data, displacement, dihedral_idxs)
dihedral_angles = batched_dihedrals(traj_positions, displacement, dihedral_idxs)

phi = dihedral_angles[:, 0].reshape((n_trajectory, -1))
psi = dihedral_angles[:, 1].reshape((n_trajectory, -1))

# mean squared error (in rad)
nbins = 60
dihedrals_ref_rad = jnp.deg2rad(dihedrals_ref)
dihedral_angles_rad = jnp.deg2rad(dihedral_angles)
h_ref, y1, y2 = onp.histogram2d(
    dihedrals_ref_rad[:, 0], dihedrals_ref_rad[:, 1], bins=nbins, density=True)
h_pred, _, _ = onp.histogram2d(
    dihedral_angles_rad[:, 0], dihedral_angles_rad[:, 1], bins=nbins,
    density=True)

mse = onp.mean((h_ref - h_pred)**2)

# to analyze sampling uncertainty
dihedral_angles_split = dihedral_angles.reshape((n_trajectory, -1, 2))

# Plots
phi_angles_ref = onp.load('../../../../Datasets/Alanine/phi_angles_r100ns.npy')
psi_angles_ref = onp.load('../../../../Datasets/Alanine/psi_angles_r100ns.npy')
# dihedral histograms
visualization.plot_histogram_dihedral(dihedral_angles, save_name + '_predicted',
                                      folder=folder_name)
visualization.plot_histogram_dihedral(dihedrals_ref, save_name + '_REF',
                                      folder=folder_name)


visualization.plot_histogram_density(
    dihedral_angles, save_name + '_predicted_', folder=folder_name)
visualization.plot_histogram_density(dihedrals_ref, save_name + '_REF',
                                     folder=folder_name)

# for i in range(n_trajectory):  # visualize single trajectories
#     visualization.plot_histogram_density(dihedral_angles_split[i, :],
#                                          save_name + f'_FM_{i}',
#                                          folder=folder_name)

visualization.plot_1d_dihedral(
    [phi_angles_ref, phi], 'phi_' + save_name, labels=labels[0:2],
    folder=folder_name)
visualization.plot_1d_dihedral(
    [psi_angles_ref, psi], 'psi_' + save_name, location='upper left',
    labels=labels[0:2], xlabel='$\psi$ in deg', folder=folder_name)

visualization.plot_compare_1d_free_energy(
    [dihedral_angles[:, 0]], dihedrals_ref[:, 0], 'phi_' + save_name,
    labels[1:], kbt, folder=folder_name)
visualization.plot_compare_1d_free_energy(
    [dihedral_angles[:, 1]], dihedrals_ref[:, 1], 'psi_' + save_name,
    labels[1:], kbt, folder=folder_name, xlabel='$\psi$ in rad')
visualization.plot_histogram_free_energy(dihedral_angles, save_name + '_pred',
                                         kbt, folder=folder_name)
visualization.plot_histogram_free_energy(dihedrals_ref, save_name + '_REF',
                                         kbt, folder=folder_name)

visualization.plot_compare_histogram_free_energy(
    [dihedrals_ref, dihedral_angles], save_name, kbt, titles=labels,
    folder=folder_name)

# pplot.plot_histogram_free_energy_1x3([dihedrals_ref, dihedral_angles],
#                                      save_name, titles=labels,
#                                      folder=folder_name)
# pplot.plot_histogram_density_2x2([dihedrals_ref, dihedral_angles],
#                                  save_name, titles=labels, folder=folder_name)
# pplot.plot_1D_dihedral_1x2([phi, phi_angles_ref], [psi, psi_angles_ref],
#                            save_name, labels=[labels[1], labels[0]])
