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

"""Example for active learning from experimental data via DiffTRe."""
import os
import sys

from jax_md import space
import optax

from chemtrain.jax_md_mod import io
from chemtrain import trainers, util
from chemtrain.trajectory import traj_util
from util import Postprocessing, Initialization

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = 1
    # controls on which gpu the program runs
os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)

# TODO do we need a smaller threshold for training, otherwise immediate
#  addition of state points? Or always training until decrease non-substantial?
al_loss_threshold = 1.e-3
thresh = 5


file = 'data/confs/Water_experimental_3nm.gro'  # 901 particles

# model = 'LJ'
# model = 'Tabulated'
# model = 'PairNN'
model = 'CGDimeNet'

kbt_dependent = True

# checkpoint = 'output/difftre/Checkpoints/epoch2.pkl'
checkpoint = None

mass = 18.0154
time_step = 0.002
total_time = 20.  # large system: 3ps minimum, 5 standard, 10 more than enough
t_equilib = 2.  # equilibration time before sampling RDF
print_every = 0.1
timings = traj_util.process_printouts(time_step, total_time, t_equilib,
                                      print_every)

# negative lr: step towards negative gradient direction
if model == 'Tabulated':
    initial_lr = -0.1
elif model == 'PairNN':
    initial_lr = -0.005
else:
    initial_lr = -0.002  # in npt ensemble no more than 0.002; maybe even 0.001

optimizer = optax.chain(
    optax.scale_by_adam(0.1, 0.4),
    optax.scale_by_schedule(optax.exponential_decay(initial_lr, 100, 0.01))
)

box, R, _, _ = io.load_box(file)  # initial configuration

# initial state point
# TODO maybe use future statepoint generation function?
system_temperature = 296.15  # Kelvin
boltzmann_constant = 0.0083145107  # in kJ / mol K
kbt = system_temperature * boltzmann_constant
kJ_mol_nm3_to_bar = 16.6054
pressure_target = 1.01325 / kJ_mol_nm3_to_bar  # 1 atm in kJ / mol nm^3
simulation_data = Initialization.InitializationClass(
    R_init=R, box=box, kbT=kbt, masses=mass, dt=time_step,
    ref_press=pressure_target, temperature=system_temperature
)

# loss function for initial NVT simulation
target_rdf = 'Water_Ox'
rdf_struct = Initialization.select_target_rdf(target_rdf)
nvt_target_dict = {'rdf': rdf_struct, 'pressure': pressure_target}

# Pretrain on NVT ensemble such that density does not decrease too much in NPT
load_init_trainer = False
init_trainer_path = f'output/difftre/{model}_init_state_trainer.pkl'
if load_init_trainer:
    init_trainer = util.load_trainer(init_trainer_path)
else:
    load_pretrained = False
    pre_trainer_path = f'output/difftre/{model}_nvt_pretrained.pkl'
    pretrain_epochs = 100

    pretrain_state, init_params, nvt_sim_fns, nvt_compute_fns, nvt_targets \
        = Initialization.initialize_simulation(simulation_data,
                                               model,
                                               nvt_target_dict,
                                               integrator='Nose_Hoover',
                                               kbt_dependent=kbt_dependent)
    # weigh pressure higher for pretraining
    nvt_targets['pressure']['gamma'] = 1.e-6
    nvt_sim_template, energy_fn_template, neighbor_fn = nvt_sim_fns
    pre_trainer = trainers.Difftre(init_params, optimizer,
                                   energy_fn_template=energy_fn_template)
    pre_trainer.add_statepoint(energy_fn_template, nvt_sim_template,
                               neighbor_fn, timings, kbt, nvt_compute_fns,
                               pretrain_state, nvt_targets)
    if load_pretrained:
        pre_trainer = util.load_trainer(pre_trainer_path)
    else:
        pre_trainer.train(pretrain_epochs)
        pre_trainer.save_trainer(pre_trainer_path)

    # Visualize pre-training results
    visualize_pretrained = True
    if visualize_pretrained:
        test_traj_fn = traj_util.trajectory_generator_init(
            nvt_sim_template, energy_fn_template, timings)
        test_traj = test_traj_fn(pre_trainer.best_params,
                                 pre_trainer.get_sim_state(0), kT=kbt)
        quantity_traj = traj_util.quantity_traj(test_traj, nvt_compute_fns,
                                                pre_trainer.best_params)
        predicted_quantities = traj_util.average_predictions(quantity_traj)

        initial_press = pre_trainer.predictions[0][0]['pressure']
        last_press_estimate = (pre_trainer.predictions[0]
                               [pretrain_epochs - 1]['pressure'])
        press_after_pretrain = predicted_quantities['pressure']
        print(f'Init guess pressure = {initial_press:.3f} '
              f'Last pressure estimate = {last_press_estimate:.3f} '
              f'After NVT pretrain: {press_after_pretrain:.3f}')
        Postprocessing.plot_initial_and_predicted_rdf(
            rdf_struct.rdf_bin_centers, predicted_quantities['rdf'],
            model, visible_device, rdf_struct.reference,
            pre_trainer.predictions[0][0]['rdf'], after_pretraining=True)
        # Postprocessing.plot_initial_and_predicted_adf(
        #     adf_struct.adf_bin_centers, predicted_quantities['adf'],
        #     model, visible_device, adf_struct.reference,
        #     pre_trainer.predictions[0][0]['adf'], after_pretraining=True)

    init_params = pre_trainer.params
    pre_trained_state = pre_trainer.get_sim_state(0)[0]
    nvt_position = space.transform(box, pre_trained_state.position)
    simulation_data = simulation_data.replace(R_init=nvt_position)
    # TODO improve this once automated setup function is available

    # train on NPT
    # TODO adf is not super reliable
    adf_struct = Initialization.select_target_adf('Water_Ox', 0.318)
    target_density = 997.87 / 1.66054  # experimental density in u / nm^3
    kappa = 45.25e-6 * kJ_mol_nm3_to_bar
    alpha = 2.573e-4  # 1/K
    cal_to_kJ = 4.184e-3
    cp = 18 * cal_to_kJ
    target_npt_dict = {'rdf': rdf_struct, 'density': target_density,
                       'kappa': kappa, 'alpha': alpha, 'cp': cp}
    # 'adf': adf_struct,
    reference_state, _, simulation_fns, compute_fns, targets = \
        Initialization.initialize_simulation(simulation_data,
                                             model,
                                             target_npt_dict,
                                             integrator='NPT',
                                             kbt_dependent=kbt_dependent)
    simulator_template, energy_fn_template, neighbor_fn = simulation_fns

    optimizer = optax.chain(  # reduce learning rate for npt ensemble
        optax.scale_by_adam(0.1, 0.4),
        optax.scale_by_schedule(optax.exponential_decay(initial_lr / 8.,
                                                        200, 0.1))
    )

    init_trainer = trainers.Difftre(init_params,
                                    optimizer,
                                    energy_fn_template=energy_fn_template,
                                    reweight_ratio=1.)  # no reweighting

    init_trainer.add_statepoint(energy_fn_template, simulator_template,
                                neighbor_fn, timings, kbt, compute_fns,
                                reference_state, targets, pressure_target)

    debugging_npt = True
    if debugging_npt:
        Postprocessing.debug_npt_ensemble(init_trainer.trajectory_states[0],
                                          pressure_target, kbt, target_density,
                                          mass)

    init_trainer.train(200)  # thresh= for early stopping later
    init_trainer.save_trainer(init_trainer_path)

    # Postprocess initial state point results
    predicted_quantities = init_trainer.predictions[0]

    from pathlib import Path
    Path('output/figures').mkdir(parents=True, exist_ok=True)
    Postprocessing.plot_loss_and_gradient_history(init_trainer.batch_losses,
                                                  visible_device)



    if model == 'Tabulated':
        # make sure this is the same as in the tabulated model definition
        x_vals = Initialization.default_x_vals(0.9, 0.1)
        Postprocessing.plot_and_save_tabulated_potential(
            x_vals, init_trainer.params, init_params, model, visible_device)

    if 'rdf' in predicted_quantities[0]:
        RDF_save_dict = {'reference': rdf_struct.reference,
                         'x_vals': rdf_struct.rdf_bin_centers}
        rdf_series = [predicted_quantities[epoch]['rdf']
                      for epoch in predicted_quantities]
        RDF_save_dict['series'] = rdf_series
        Postprocessing.plot_initial_and_predicted_rdf(
            rdf_struct.rdf_bin_centers, rdf_series[-1], model, visible_device,
            rdf_struct.reference, rdf_series[0])

        import pickle
        rdf_pickle_file_path = f'output/difftre/RDF_{model}{visible_device}.pkl'
        with open(rdf_pickle_file_path, 'wb') as f:
            pickle.dump(RDF_save_dict, f)
        Postprocessing.visualize_time_series(rdf_pickle_file_path)

    if 'adf' in predicted_quantities[0]:
        ADF_save_dict = {'reference': adf_struct.reference,
                         'x_vals': adf_struct.adf_bin_centers}
        adf_series = [predicted_quantities[epoch]['adf']
                      for epoch in predicted_quantities]
        ADF_save_dict['series'] = adf_series
        Postprocessing.plot_initial_and_predicted_adf(
            adf_struct.adf_bin_centers, adf_series[-1], model, visible_device,
            adf_struct.reference, adf_series[0])

    if 'pressure' in predicted_quantities[0]:
        pressure_series = [predicted_quantities[epoch]['pressure']
                           for epoch in predicted_quantities]
        Postprocessing.plot_pressure_history(pressure_series, model,
                                             visible_device,
                                             reference_pressure=pressure_target)

    if 'density' in predicted_quantities[0]:
        density_series = [predicted_quantities[epoch]['density'] for epoch in
                          predicted_quantities]
        Postprocessing.plot_density_history(density_series, model,
                                            visible_device,
                                            reference_density=target_density)

    if 'alpha' in predicted_quantities[0]:
        alpha_series = [predicted_quantities[epoch]['alpha'] for epoch in
                        predicted_quantities]
        Postprocessing.plot_alpha_history(alpha_series, model,
                                          visible_device, alpha)

    if 'kappa' in predicted_quantities[0]:
        kappa_series = [predicted_quantities[epoch]['kappa'] for epoch in
                        predicted_quantities]
        Postprocessing.plot_kappa_history(kappa_series, model,
                                          visible_device, kappa)

    if 'cp' in predicted_quantities[0]:
        cp_series = [predicted_quantities[epoch]['cp'] for epoch in
                     predicted_quantities]
        Postprocessing.plot_cp_history(cp_series, model, visible_device, kappa)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(init_trainer.update_times)
    plt.savefig('output/figures/difftre_time_per_epoche.png')

# start active learning
active_trainer = trainers.DifftreActive(init_trainer)
