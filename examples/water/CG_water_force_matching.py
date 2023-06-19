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

"""Train a CG water model via force matching."""
import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str('2,3')
# os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

import cloudpickle as pickle
from pathlib import Path

from jax import random, numpy as jnp
from jax_md import space
import matplotlib.pyplot as plt
import numpy as onp
import optax

from chemtrain import trainers, util, data_processing
from chemtrain.jax_md_mod import custom_space
from util import Initialization

Path('output/figures').mkdir(parents=True, exist_ok=True)
Path('output/force_matching').mkdir(parents=True, exist_ok=True)


# user input
configuration_str = '../../../../Datasets/TIP4P/conf_COM_10k.npy'
force_str = '../../../../Datasets/TIP4P/forces_COM_10k.npy'
box_str = '../../../../Datasets/TIP4P/box.npy'

training_name = 'FM'
save_path = f'output/force_matching/trained_model_{training_name}.pkl'
save_plot = f'output/figures/force_matching_losses_{training_name}.png'
save_params_path = f'output/force_matching/best_params_{training_name}.pkl'
save_params_path2 = f'output/force_matching/params_{training_name}.pkl'

used_dataset_size = 1100
train_ratio = 0.7
val_ratio = 0.1
batch_per_device = 5
batch_cache = 10
epochs = 50
check_freq = 10

box_length = onp.load(box_str)
box = jnp.ones(3) * box_length

model = 'CGDimeNet'
# model = 'Tabulated'

checkpoint = None
# checkpoint = 'output/force_matching/Checkpoints/epoch2.pkl'

# build datasets
position_data = data_processing.get_dataset(configuration_str,
                                            retain=used_dataset_size)
force_data = data_processing.get_dataset(force_str, retain=used_dataset_size)

dataset_size = position_data.shape[0]
print('Dataset size:', dataset_size)

if model == 'Tabulated':
    initial_lr = 0.1
elif model == 'CGDimeNet':
    initial_lr = 0.001
else:
    raise NotImplementedError

decay_length = int(dataset_size / batch_per_device * epochs)
lr_schedule = optax.exponential_decay(-initial_lr, decay_length, 0.01)
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule)
)

box_tensor, _ = custom_space.init_fractional_coordinates(box)
displacement, _ = space.periodic_general(box_tensor,
                                         fractional_coordinates=True)
position_data = data_processing.scale_dataset_fractional(position_data,
                                                         box_tensor)
R_init = position_data[0]

model_init_key = random.PRNGKey(0)
constants = {'repulsive': (0.3165, 1., 0.5)}
idxs = {}
energy_fn_template, _, init_params, nbrs_init = \
    Initialization.select_model(model, R_init, displacement, box,
                                model_init_key, fractional=True,
                                prior_constants=constants, prior_idxs=idxs)

trainer = trainers.ForceMatching(init_params, energy_fn_template, nbrs_init,
                                 optimizer, position_data,
                                 force_data=force_data,
                                 batch_per_device=batch_per_device,
                                 box_tensor=box_tensor,
                                 batch_cache=batch_cache,
                                 train_ratio=train_ratio, val_ratio=val_ratio)

if checkpoint is not None:  # restart from a previous checkpoint
    trainer = util.load_trainer(checkpoint)

trainer.train(epochs, checkpoint_freq=check_freq)
trainer.save_trainer(save_path)
with open(save_params_path, 'wb') as pickle_file:
    pickle.dump(trainer.best_params, pickle_file)

with open(save_params_path2, 'wb') as pickle_file2:
    pickle.dump(trainer.params, pickle_file2)

plt.figure()
plt.plot(trainer.train_losses, label='Train')
plt.plot(trainer.val_losses, label='Val')
plt.legend()
plt.ylabel('MSE Loss')
plt.xlabel('Update step')
plt.savefig(save_plot)
