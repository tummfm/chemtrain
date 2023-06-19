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

import jax.numpy as jnp
from jax import lax
import pickle
import numpy as onp

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

from scipy import interpolate as sci_interpolate
from chemtrain import traj_quantity


def box_density(R_snapshot, bin_edges, axis=0):
    # assumes all particles are wrapped into the same box
    profile, _ = jnp.histogram(R_snapshot[:, axis], bins=bin_edges)
    profile *= (profile.shape[0] / R_snapshot.shape[0]) # norm via n_bins and n_particles
    return profile


def get_bin_centers_from_edges(bin_edges):
    """To get centers from bin edges as generated from jnp.histogram"""
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    return bin_centers


def plot_density(file_name, n_bins=50):
    with open(file_name, 'rb') as f:
        R_traj_list, box = pickle.load(f)

    R_traj = R_traj_list[10]  # select one trajectory from all trajectories over optimization
    bin_edges = jnp.linspace(0., box[0], n_bins + 1)
    bin_centers = get_bin_centers_from_edges(bin_edges)
    compute_box_density = partial(box_density, bin_edges=bin_edges)
    density_snapshots = lax.map(compute_box_density, R_traj)
    density = jnp.mean(density_snapshots, axis=0)

    file_name = file_name[:-4]
    plt.figure()
    plt.plot(bin_centers, density)
    plt.ylabel('Normalizes Density')
    plt.xlabel('x')
    plt.savefig(file_name + '.png')


def visualize_time_series(file_name):
    with open(file_name, 'rb') as f:
        plot_dict = pickle.load(f)
    x_vals = plot_dict['x_vals']
    reference = plot_dict['reference']
    time_series = plot_dict['series']

    fig, ax = plt.subplots(figsize=(5, 3))
    series_line = ax.plot(x_vals, reference, label='Predicted')[0]
    ax.plot(x_vals, reference, label='Reference')
    ax.legend()
    # ax.set(xlim=(-3, 3), ylim=(-1, 1))

    def animate(i):
        series_line.set_ydata(time_series[i])
        ax.set_title('Epoche ' + str(i))

    file_name = file_name[:-4]
    anim = FuncAnimation(fig, animate, interval=200, frames=len(time_series) - 1)
    anim.save(file_name + '.gif', writer='imagemagick')


def plot_initial_and_predicted_rdf(rdf_bin_centers, g_average_final, model,
                                   visible_device, reference_rdf=None,
                                   g_average_init=None, after_pretraining=False,
                                   std=None, T=None):
    if after_pretraining:
        pretrain_str = '_after_pretrain'
    else:
        pretrain_str = ''

    plt.figure()
    plt.plot(rdf_bin_centers, g_average_final, label='predicted')
    if reference_rdf is not None:
        plt.plot(rdf_bin_centers, reference_rdf, label='reference')
    if g_average_init is not None:
        plt.plot(rdf_bin_centers, g_average_init, label='initial guess')
    if std is not None:
        plt.fill_between(rdf_bin_centers, g_average_final - std,
                         g_average_final + std, alpha=0.3,
                         facecolor='#3c5488ff', label='Uncertainty')
    plt.legend()
    plt.xlabel('r [$\mathrm{nm}$]')
    plt.savefig(f'output/figures/predicted_RDF_'
                f'{model}_{T or ""}_{visible_device}{pretrain_str}.png')
    return

def plot_initial_and_predicted_adf(adf_bin_centers, predicted_adf_final, model,
                                   visible_device, reference_adf=None,
                                   adf_init=None, after_pretraining=False,
                                   std=None, T=None):
    if after_pretraining:
        pretrain_str = '_after_pretrain'
    else:
        pretrain_str = ''

    plt.figure()
    plt.plot(adf_bin_centers, predicted_adf_final, label='predicted')
    if reference_adf is not None:
        plt.plot(adf_bin_centers, reference_adf, label='reference')
    if adf_init is not None:
        plt.plot(adf_bin_centers, adf_init, label='initial guess')
    if std is not None:
        plt.fill_between(adf_bin_centers, predicted_adf_final - std,
                         predicted_adf_final + std, alpha=0.3,
                         facecolor='#3c5488ff', label='Uncertainty')
    plt.legend()
    plt.xlabel('Angle [rad]')
    plt.savefig(f'output/figures/predicted_ADF_'
                f'{model}_{T or ""}_{visible_device}{pretrain_str}.png')
    return


def plot_initial_and_predicted_tcf(bin_centers, g_average_final, model,
                                   reference=None, g_average_init=None,
                                   transparent=False, labels=None,
                                   axis_label=None):
    if labels is None:
        labels = ['reference', 'predicted', 'initial guess']
    plt.figure()
    plt.plot(bin_centers, g_average_final, label=labels[1])
    if g_average_init is not None:
        plt.plot(bin_centers, g_average_init, label=labels[2])
    if reference is not None:
        plt.plot(bin_centers, reference, label=labels[0], dashes=(4, 3),
                 color='k', linestyle='--')

    plt.legend()
    if axis_label is not None:
        plt.ylabel(axis_label[0])
        plt.xlabel(axis_label[1])
    else:
        plt.xlabel('r in $\mathrm{nm}$')
    if transparent:
        plt.savefig('output/figures/predicted_TCF_' + model + '.png',
                    transparent=True)
    else:
        plt.savefig('output/figures/predicted_TCF_' + model + '.png')
    return


def plot_pressure_history(pressure_history, model, visible_device,
                          reference_pressure=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('Pressure in kJ/ (mol nm^3)')
    ax1.plot(pressure_history, label='Predicted')
    if reference_pressure is not None:
        ax1.axhline(y=reference_pressure, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_pressure_history_' + model + str(visible_device) + '.png')
    return

def plot_density_history(density_history, model, visible_device, reference_density=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('Density in g / l')
    ax1.plot(density_history, label='Predicted')
    if reference_density is not None:
        ax1.axhline(y=reference_density, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_density_history_' + model + str(visible_device) + '.png')
    return

def plot_alpha_history(prediction_history, model, visible_device, reference=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('alpha in 1 / K')
    ax1.plot(prediction_history, label='Predicted')
    if reference is not None:
        ax1.axhline(y=reference, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_alpha_history_' + model + str(visible_device) + '.png')
    return

def plot_kappa_history(prediction_history, model, visible_device, reference=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('kappa in 1 / (kJ / mol nm**3)')
    ax1.plot(prediction_history, label='Predicted')
    if reference is not None:
        ax1.axhline(y=reference, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_kappa_history_' + model + str(visible_device) + '.png')
    return

def plot_cp_history(prediction_history, model, visible_device, reference=None):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('update step')
    ax1.set_ylabel('c_p in kJ / mol K')
    ax1.plot(prediction_history, label='Predicted')
    if reference is not None:
        ax1.axhline(y=reference, linestyle='--', label='Reference', color='k')
    ax1.legend()
    plt.savefig('output/figures/difftre_cp_history_' + model + str(visible_device) + '.png')
    return

def plot_loss_and_gradient_history(loss_history, visible_device,
                                   gradient_history=None):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('update step')
    ax1.set_ylabel('Loss')
    ax1.plot(loss_history, color=color, label='Loss')
    if gradient_history is not None:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.semilogy(gradient_history, label='Gradient norm', color=color)
        ax2.set_ylabel('Gradient norm', color=color)  # we already handled the x-label with ax1
    plt.savefig('output/figures/difftre_train_history' + str(visible_device) + '.png')
    return


def plot_and_save_tabulated_potential(x_vals, params, init_params, model, visible_device):
    U_int = sci_interpolate.interp1d(x_vals, params, kind='cubic')
    spline_predicted = jnp.array(U_int(x_vals))
    result_array = onp.array([x_vals, spline_predicted]).T
    onp.savetxt('output/predicted_tabulated_potential.csv', result_array)

    plt.figure()
    plt.plot(x_vals, init_params, label='initial guess')
    plt.plot(x_vals, params, label='predicted table points')
    plt.plot(x_vals, spline_predicted, label='predicted spline')
    plt.ylim([-2.5, 5.])
    plt.legend()
    plt.savefig('output/figures/difftre_predicted_Potential_' + model + str(visible_device) + '.png')


def debug_npt_ensemble(init_traj_state, pressure_target, kbt, target_density,
                       mass):
    n_particles = init_traj_state.sim_state[0].position.shape[0]
    N = init_traj_state.aux['energy'].shape[0]
    plt.figure()
    plt.plot(init_traj_state.aux['energy'])
    plt.ylabel('Energy in kJ/mol')
    plt.savefig('NPT_energy_distribution.png')
    plt.figure()
    plt.plot(init_traj_state.aux['pressure'])
    plt.ylabel('Pressure in kJ/mol nm**3')
    plt.savefig('NPT_pressure_distribution.png')
    mean_pressure = jnp.mean(init_traj_state.aux['pressure'], axis=0)
    std_pressure = jnp.std(init_traj_state.aux['pressure'], axis=0)
    print(f'Mean pressure: {mean_pressure}. Target: {pressure_target}.'
          f'Statistical uncertainty = {std_pressure / jnp.sqrt(N)}')

    volume_traj = traj_quantity.volumes(init_traj_state)
    kappa = traj_quantity.isothermal_compressibility_npt(volume_traj, kbt)
    print(f'Isothermal compressibility {kappa} / kJ / mol nm**3  equals '
          f'{kappa * 16.6054} / bar')

    volumes = traj_quantity.volumes(init_traj_state)
    mean_volume = jnp.mean(volumes, axis=0)
    std_volume = jnp.std(volumes, axis=0)
    print(f'Mean volume = {mean_volume} nm**3. Statistical uncertainty = '
          f'{std_volume / jnp.sqrt(N)}\n '
          f'Mean density = {mass * n_particles / mean_volume}. '
          f'Target = {target_density}')

    plt.figure()
    plt.plot(volumes)
    plt.savefig('NPT_volume_distribution.png')


if __name__ == '__main__':

    # visualize_time_series('RDFs_Tabulated.pkl')
    plot_density('output/Trajectories/Traj_GNN3.pkl')
