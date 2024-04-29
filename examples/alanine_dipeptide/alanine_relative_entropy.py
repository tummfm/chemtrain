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

"""Training a CG model of alanine dipeptide via relative entropy minimization.
"""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)
sys.path.append("../../")
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

from pathlib import Path

from jax import random
import optax

from chemtrain import util, trainers
from chemtrain.data import data_processing
from chemtrain.trajectory import traj_util
from chemtrain.jax_md_mod import io
from util import Initialization

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/rel_entropy').mkdir(parents=True, exist_ok=True)

# user input
mapping = 'heavy'
name = 'RE_LJ_2ns_4fs_150up'
file_topology = f'data/confs/{mapping}_2_7nm.gro'
save_params_path = f'output/rel_entropy/{name}_trained_params_alanine.pkl'
save_trainer_path = f'output/rel_entropy/{name}_trainer_alanine.pkl'

configuration_str = f'../../../Datasets/Alanine/confs_{mapping}_100ns.npy'

used_dataset_size = 400000

saved_params_path = None

# simulation parameters
system_temperature = 300  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant

time_step = 0.002
total_time = 21.
t_equilib = 1.
print_every = 0.1
num_chains = 100

model = 'CGDimeNet'

# checkpoint = 'output/rel_entropy/Checkpoints/epoch2.pkl'
checkpoint = None
check_freq = None

num_updates = 300

initial_lr = 0.003
lr_schedule = optax.exponential_decay(-initial_lr, num_updates, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule)
)

timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

# initial configuration
box, _, masses, _ = io.load_box(file_topology)

priors = ['bond', 'angle', 'dihedral']  # 'LJ
species, prior_idxs, prior_constants = Initialization.select_protein(
    'heavy_alanine_dipeptide', priors)

position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size)

# Random starting configurations
key = random.PRNGKey(0)
r_init = random.choice(key, position_data, (num_chains,), replace=False)

simulation_data = Initialization.InitializationClass(
    r_init=r_init, box=box, kbt=kbt, masses=masses, dt=time_step,
    species=species)

init_sim_states, init_params, simulation_fns, _, _ = \
    Initialization.initialize_simulation(simulation_data,
                                         model,
                                         integrator='Langevin',
                                         prior_constants=prior_constants,
                                         prior_idxs=prior_idxs)

simulator_template, energy_fn_template, neighbor_fn = simulation_fns

reference_data = data_processing.scale_dataset_fractional(position_data, box)

trainer = trainers.RelativeEntropy(init_params, optimizer, reweight_ratio=1.1,
                                   energy_fn_template=energy_fn_template)

trainer.add_statepoint(
    reference_data, energy_fn_template, simulator_template, neighbor_fn,
    timings, kbt, init_sim_states, reference_batch_size=used_dataset_size,
    vmap_batch=num_chains, num_chains=num_chains)

trainer.init_step_size_adaption(0.25)

if checkpoint is not None:
    trainer = util.load_trainer(checkpoint)

trainer.train(num_updates, checkpoint_freq=check_freq)

# save_epochs = [0, 10, 50, 100]
# for k in range(num_updates):
#     trainer.train(1, checkpoint_freq=check_freq)
#     if k in save_epochs:
#         trainer.save_energy_params(save_checkpoint+f'_epoch{k}.pkl', '.pkl')

save_params_path = Path(save_params_path)
save_trainer_path = Path(save_trainer_path)

save_params_path.parent.mkdir(exist_ok=True, parents=True)
save_trainer_path.parent.mkdir(exist_ok=True, parents=True)

trainer.save_energy_params(save_params_path, '.pkl')
trainer.save_trainer(save_trainer_path, '.pkl')
