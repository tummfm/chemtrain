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

"""Input file to run DiffTRe training."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 3
    # controls on which gpu the program runs
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

import matplotlib.pyplot as plt
import optax
from pathlib import Path

from chemtrain.jax_md_mod import io
from chemtrain import trainers, util
from chemtrain.trajectory import traj_util
from util import Postprocessing, Initialization

from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform)



# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)
# config.update('jax_numpy_rank_promotion','warn')
# config.update('jax_enable_x64', True)

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/difftre').mkdir(parents=True, exist_ok=True)

################### user input ###########:

# conversion_factor = 1.66054021  # to kg/m^3 from u
# water: 18.008 u
# pressure_conversion_factor = 16.6054  # from kJ/mol nm^-3 to bar
# experimental 0.10007 atoms / A^3 = 997.87 g/l at 23C = 296.15K
# --> pressure = 1bar = 0.06022 kJ / mol nm^3 = 2135 molecules in 4nm box
file = 'data/confs/Water_experimental_3nm.gro'  # 901 particles
# file = 'data/confs/Water_experimental_4nm.gro'  # 2132 particles
# file = 'data/confs/SPC_FW_3nm.gro'  # 905 particles
# file = 'data/confs/SPC_955_3nm.gro'  # 862 particles
# file = 'data/confs/SPC_FW_2nm.gro'  # 229 particles

# model = 'LJ'
model = 'Tabulated'
# model = 'PairNN'
# model = 'CGDimeNet'

kbt_dependent = False
dropout_init_seed = 42
dropout_init_seed = None  # if no dropout should be used

save_path = 'output/difftre/trained_model.pkl'
# checkpoint = 'output/difftre/Checkpoints/epoch2.pkl'
checkpoint = None

# integrator = 'NPT'
integrator = 'Nose_Hoover'

system_temperature = 296.15  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant
# kbT = 2.49435321  # 300K
mass = 18.0154
time_step = 0.01

# lower bound from noise for 2/20: 0.015
num_chains = 6
total_time = 61.
t_equilib = 1.
print_every = 0.1

# Recompute effective traj length
if num_chains is not None:
    total_time = ((total_time - t_equilib) / num_chains) + t_equilib
    print(f"Vectorized total time is {total_time}")

# optimization parameters:
check_freq = 10
num_updates = 300
# negative lr: step towards negative gradient direction
if model == 'Tabulated':
    initial_lr = -0.1
elif model == 'PairNN':
    initial_lr = -0.005
else:
    initial_lr = -0.003  # in npt ensemble no more than 0.002; maybe even 0.001
lr_schedule = optax.exponential_decay(initial_lr, num_updates, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule)
)

# target_rdf = 'LJ'
# target_rdf = 'SPC'
target_rdf = 'Water_Ox'
pressure_target = 1. / 16.6054  # 1 bar in kJ / mol nm^3
rdf_struct = Initialization.select_target_rdf(target_rdf)
adf_struct = Initialization.select_target_adf('Water_Ox', 0.318)
target_dict = {'rdf': rdf_struct, # 'adf': adf_struct,
               'pressure': pressure_target}
# target_dict = {'rdf': rdf_struct, 'pressure': pressure_target}
# target_dict = {'rdf': rdf_struct}

# Prior potential
constants = {'repulsive': (0.3165, 1., 0.5)}
idxs = {}

#############################################

# preprocess user input
# initial configuration
box, r_init, _, _ = io.load_box(file)  # initial configuration
simulation_data = Initialization.InitializationClass(
    r_init=r_init, box=box, kbt=kbt, masses=mass, dt=time_step,
    ref_press=pressure_target
)

timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

reference_state, init_params, simulation_fns, compute_fns, targets = \
    Initialization.initialize_simulation(simulation_data, model, target_dict,
                                         integrator=integrator,
                                         kbt_dependent=kbt_dependent,
                                         dropout_init_seed=dropout_init_seed,
                                         prior_constants=constants,
                                         prior_idxs=idxs)
simulator_template, energy_fn_template, neighbor_fn = simulation_fns

trainer = trainers.Difftre(init_params,
                           optimizer,
                           energy_fn_template=energy_fn_template)

trainer.add_statepoint(energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, compute_fns, reference_state,
                       targets, pressure_target, num_chains=num_chains)

trainer.init_step_size_adaption(0.25)

if checkpoint is not None:  # restart from a previous checkpoint
    trainer = util.load_trainer(checkpoint)

trainer.train(num_updates, checkpoint_freq=check_freq)
trainer.save_trainer(save_path)

# Postprocessing
predicted_quantities = trainer.predictions[0]  # predictions for state point 0
Postprocessing.plot_loss_and_gradient_history(trainer.batch_losses,
                                              visible_device)

if model == 'Tabulated':
    # make sure x_vals is the same as in the tabulated model definition
    x_vals = Initialization.default_x_vals(0.9, 0.1)
    Postprocessing.plot_and_save_tabulated_potential(
        x_vals, trainer.params, init_params, model, visible_device)

if 'rdf' in predicted_quantities[0]:
    RDF_save_dict = {'reference': rdf_struct.reference,
                     'x_vals': rdf_struct.rdf_bin_centers}
    rdf_series = [predicted_quantities[epoch]['rdf']
                  for epoch in predicted_quantities]
    RDF_save_dict['series'] = rdf_series
    # rdf_pickle_file_path = 'output/Gif/RDFs_' + model + str(visible_device) + '.pkl'
    # with open(rdf_pickle_file_path, 'wb') as f:
    #     pickle.dump(RDF_save_dict, f)
    # Postprocessing.visualize_time_series('RDFs_' + model + str(visible_device) + '.pkl')
    Postprocessing.plot_initial_and_predicted_rdf(
        rdf_struct.rdf_bin_centers, rdf_series[-1], model, visible_device,
        rdf_struct.reference, rdf_series[0])

if 'adf' in predicted_quantities[0]:
    ADF_save_dict = {'reference': adf_struct.reference,
                     'x_vals': adf_struct.adf_bin_centers}
    adf_series = [predicted_quantities[epoch]['adf']
                  for epoch in predicted_quantities]
    ADF_save_dict['series'] = adf_series
    # adf_pickle_file_path = 'output/Gif/ADFs_' + model + str(visible_device) + '.pkl'
    # with open(adf_pickle_file_path, 'wb') as f:
    #     pickle.dump(ADF_save_dict, f)
    # Postprocessing.visualize_time_series('ADFs_' + model + str(visible_device) + '.pkl')
    Postprocessing.plot_initial_and_predicted_adf(
        adf_struct.adf_bin_centers, adf_series[-1], model, visible_device,
        adf_struct.reference, adf_series[0])

if 'pressure' in predicted_quantities[0]:
    pressure_series = [predicted_quantities[epoch]['pressure']
                       for epoch in predicted_quantities]
    Postprocessing.plot_pressure_history(pressure_series, model, visible_device,
                                         reference_pressure=pressure_target)

plt.figure()
plt.plot(trainer.update_times)
plt.savefig('output/figures/difftre_time_per_epoche.png')
