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

"""Training a CG model for alanine dipeptide via force matching."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

import cloudpickle as pickle
from pathlib import Path

from jax import random
from jax_md import space
import matplotlib.pyplot as plt
import optax

from chemtrain import trainers, data_processing, util
from chemtrain.jax_md_mod import custom_space, io
from util import Initialization

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/force_matching').mkdir(parents=True, exist_ok=True)

# user input
mapping = 'heavy'
name = 'FM_50epochs_440k'
file_topology = f'data/confs/{mapping}_2_7nm.gro'
save_path = f'output/force_matching/model_alanine_{mapping}_{name}.pkl'
save_params_path = f'output/force_matching/best_params_{mapping}_{name}.pkl'
save_params_path2 = f'output/force_matching/params_{mapping}_{name}.pkl'
save_plot = f'output/figures/FM_losses_alanine_{mapping}_{name}.png'
configuration_str = f'../../../../Datasets/Alanine/confs_{mapping}_100ns.npy'
force_str = f'../../../../Datasets/Alanine/forces_{mapping}_100ns.npy'

used_dataset_size = 500000
train_ratio = 0.7
val_ratio = 0.1
batch_per_device = 512
batch_cache = 50

initial_lr = 0.001
epochs = 100
check_freq = 10

system_temperature = 300  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant

model = 'CGDimeNet'
checkpoint = None
# checkpoint = 'output/force_matching/Checkpoints/epoch2.pkl'

key = random.PRNGKey(0)
model_init_key, shuffle_key = random.split(key, 2)

# build datasets
position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size)
force_data = data_processing.get_dataset(force_str, retain=used_dataset_size)

box, _, masses, _ = io.load_box(file_topology)
box_tensor, _ = custom_space.init_fractional_coordinates(box)
displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)

position_data = data_processing.scale_dataset_fractional(position_data,
                                                         box_tensor)
r_init = position_data[0]

lrd = int(used_dataset_size / batch_per_device * epochs)
lr_schedule = optax.exponential_decay(-initial_lr, lrd, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule)
)

priors = ['bond', 'angle', 'dihedral']  # 'LJ
species, prior_idxs, prior_constants = Initialization.select_protein(
    'heavy_alanine_dipeptide', priors)

energy_fn_template, _, init_params, nbrs_init = \
    Initialization.select_model(
        model, r_init, displacement, box, model_init_key, kbt, fractional=True,
        species=species, prior_constants=prior_constants, prior_idxs=prior_idxs)

trainer = trainers.ForceMatching(init_params, energy_fn_template, nbrs_init,
                                 optimizer, position_data,
                                 force_data=force_data,
                                 batch_per_device=batch_per_device,
                                 box_tensor=box_tensor,
                                 batch_cache=batch_cache,
                                 train_ratio=train_ratio)

if checkpoint is not None:  # restart from a previous checkpoint
    trainer = util.load_trainer(checkpoint)

trainer.train(epochs, checkpoint_freq=check_freq)

plt.figure()
plt.plot(trainer.train_losses, label='Train')
plt.plot(trainer.val_losses, label='Val')
plt.legend()
plt.ylabel('MSE Loss')
plt.xlabel('Updates')
plt.savefig(save_plot)

trainer.save_trainer(save_path)
best_params = trainer.best_params
with open(save_params_path, 'wb') as pickle_file:
    pickle.dump(best_params, pickle_file)

with open(save_params_path2, 'wb') as pickle_file2:
    pickle.dump(trainer.params, pickle_file2)
