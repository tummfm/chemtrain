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

"""Training a CG water model via relative entropy minimization."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(-1)

from pathlib import Path

import cloudpickle as pickle
from jax import tree_util, random, numpy as jnp
import numpy as onp
import optax

from chemtrain import util, trainers
from chemtrain.data import data_processing
from chemtrain.trajectory import traj_util
from util import Initialization

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/rel_entropy').mkdir(parents=True, exist_ok=True)

# dataset
configuration_str = '../../../Datasets/TIP4P/conf_COM_10k.npy'
box_str = '../../../Datasets/TIP4P/box.npy'

training_name = 'RE'
save_path = f'output/rel_entropy/trained_model_{training_name}.pkl'
save_param_path = f'output/rel_entropy/trained_params_{training_name}.pkl'
used_dataset_size = 15

# saved_params_path = 'output/force_matching/trained_params_water.pkl'
saved_params_path = None

# simulation parameters
system_temperature = 298  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant
mass = 18.01528
model_init_key = random.PRNGKey(0)

time_step = 0.01
total_time = 75.
t_equilib = 5.
print_every = 0.1

# model = 'CGDimeNet'
model = 'Tabulated'

# checkpoint = 'output/rel_entropy/Checkpoints/epoch2.pkl'
checkpoint = None
check_freq = None

num_updates = 300
if model == 'Tabulated':
    initial_lr = 0.1
elif model == 'CGDimeNet':
    initial_lr = 0.003
else:
    raise NotImplementedError

lr_schedule = optax.exponential_decay(-initial_lr, num_updates, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(lr_schedule)
)

timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

box_length = onp.load(box_str)
box = jnp.ones(3) * box_length

position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size)
r_init = position_data[0]

constants = {'repulsive': (0.3165, 1., 0.5)}
idxs = {}

simulation_data = Initialization.InitializationClass(
    r_init=r_init, box=box, kbt=kbt, masses=mass, dt=time_step)
reference_state, init_params, simulation_fns, _, _ = \
    Initialization.initialize_simulation(
        simulation_data, model, model_init_key=model_init_key,
        prior_constants=constants, prior_idxs=idxs)
simulator_template, energy_fn_template, neighbor_fn = simulation_fns

if saved_params_path is not None:
    print('using saved params')
    with open(saved_params_path, 'rb') as pickle_file:
        params = pickle.load(pickle_file)
        init_params = tree_util.tree_map(jnp.array, params)

reference_data = data_processing.scale_dataset_fractional(position_data, box)
trainer = trainers.RelativeEntropy(
    init_params, optimizer, energy_fn_template=energy_fn_template,
    reweight_ratio=1.1)

trainer.add_statepoint(reference_data, energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, reference_state)

if checkpoint is not None:  # restart from a previous checkpoint
    trainer = util.load_trainer(checkpoint)

trainer.train(num_updates, checkpoint_freq=check_freq)
trainer.save_energy_params(save_param_path, '.pkl')
trainer.save_trainer(save_path)
