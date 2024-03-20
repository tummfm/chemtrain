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

"""Plot functions to visualize free energy surface of alanine dipeptide."""
from jax import numpy as jnp
from matplotlib.animation import FuncAnimation
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as onp
from scipy.interpolate import interp1d
from util import Postprocessing


def plot_scatter_forces(predicted, reference, save_as, model, line=2000):
    """Scatter plot of predicted and reference forces."""
    plt.figure()
    hex_bin = plt.hexbin(predicted, reference, gridsize=50, mincnt=1,
                         vmax=200000)
    plt.plot([-line, line], [-line, line], 'r--')
    plt.ylabel('Reference Force Components [kJ $\mathrm{mol^{-1} \ nm^{-1}}$]')
    plt.xlabel('Predicted Force Components [kJ $\mathrm{mol^{-1} \ nm^{-1}}$]')
    plt.title(model)
    # plt.xlim([-line,line])
    # plt.ylim([-line,line])
    plt.tight_layout()
    cbar = plt.colorbar(hex_bin)
    cbar.ax.set_yticklabels(['{:,.0f}K'.format(i/1000)
                             for i in cbar.get_ticks()])
    cbar.set_label('Number of data points')
    plt.savefig('output/postprocessing/force_scatter_' + save_as + '.pdf')


def dihedral_map():
    mymap = onp.array([[0.9, 0.9, 0.9],
                       [0.85, 0.85, 0.85],
                       [0.8, 0.8, 0.8],
                       [0.75, 0.75, 0.75],
                       [0.7, 0.7, 0.7],
                       [0.65, 0.65, 0.65],
                       [0.6, 0.6, 0.6],
                       [0.55, 0.55, 0.55],
                       [0.5, 0.5, 0.5],
                       [0.45, 0.45, 0.45],
                       [0.4, 0.4, 0.4],
                       [0.35, 0.35, 0.35],
                       [0.3, 0.3, 0.3],
                       [0.25, 0.25, 0.25],
                       [0.2, 0.2, 0.2],
                       [0.15, 0.15, 0.15],
                       [0.1, 0.1, 0.1],
                       [0.05, 0.05, 0.05],
                       [0, 0, 0]])
    newcmp = clr.ListedColormap(mymap)
    return newcmp


def annotate_alanine_histrogram(axis=None):
    target = plt if axis is None else axis

    target.xlabel('$\phi$ in $\mathrm{deg}$')
    target.ylabel('$\psi$ in $\mathrm{deg}$')
    target.xlim([-180, 180])
    target.ylim([-180, 180])
    target.text(-155, 90, '$C5$', fontsize=18)
    target.text(-70, 90, '$C7eq$', fontsize=18)
    target.text(145, 90, '$C5$', fontsize=18)
    target.text(-155, -150, '$C5$', fontsize=18)
    target.text(-70, -150, '$C7eq$', fontsize=18)
    target.text(145, -150, '$C5$', fontsize=18)
    target.text(-170, -90, r'$\alpha_R$"', fontsize=18)
    target.text(140, -90, r'$\alpha_R$"', fontsize=18)
    target.text(-70, -90, r'$\alpha_R$', fontsize=18)
    target.text(70, 0, r'$\alpha_L$', fontsize=18)
    target.plot([-180, 13], [74, 74], 'k', linewidth=0.5)
    target.plot([128, 180], [74, 74], 'k', linewidth=0.5)
    target.plot([13, 13], [-180, 180], 'k', linewidth=0.5)
    target.plot([128, 128], [-180, 180], 'k', linewidth=0.5)
    target.plot([-180, 13], [-125, -125], 'k', linewidth=0.5)
    target.plot([128, 180], [-125, -125], 'k', linewidth=0.5)
    target.plot([-134, -134], [-125, 74], 'k', linewidth=0.5)
    target.plot([-110, -110], [-180, -125], 'k', linewidth=0.5)
    target.plot([-110, -110], [74, 180], 'k', linewidth=0.5)


def plot_histogram_dihedral(angles, saveas, reference=None, init_angles=None,
                            folder='', plot_format='pdf'):
    """Plot and save 2D density histogram for alanine dihedral angles."""
    newcmp = dihedral_map()

    plt.figure()
    plt.hist2d(angles[:, 0], angles[:, 1], bins=60, cmap=newcmp, cmin=1)
    plt.colorbar()
    if reference is not None:
        refcmp = plt.get_cmap('Reds')
        plt.hist2d(reference[:, 0], reference[:, 1], bins=60, cmap=refcmp,
                   cmin=1, alpha=0.6)
    if init_angles is not None:
        for idx, angle in enumerate(init_angles):
            plt.scatter(angle[0], angle[1], label=idx)
        plt.legend(loc='upper left', bbox_to_anchor=(1.22, 0.6))
    annotate_alanine_histrogram()
    plt.savefig(f'output/postprocessing/{folder}/histogram_'
                f'dihedral_{saveas}.{plot_format}')
    plt.close('all')


def plot_histogram_density(angles, saveas, folder=''):
    """Plot 2D histogram for alanine free energies from the dihedral angles."""
    newcmp = dihedral_map()
    h, x_edges, y_edges = jnp.histogram2d(angles[:, 0], angles[:, 1],
                                          bins=60, density=True)

    h_masked = onp.where(h == 0, onp.nan, h)
    x, y = onp.meshgrid(x_edges, y_edges)

    plt.figure()
    # vmin=1, vmax=5.25
    plt.pcolormesh(x, y, h_masked.T, cmap=newcmp, vmax=0.000225)
    # axs.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    # axs.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label('Density')
    annotate_alanine_histrogram()
    plt.savefig(f'output/postprocessing/{folder}/histogram_density'
                f'_{saveas}.pdf')
    plt.close('all')


def plot_compare_histogram_density(list_angles, saveas, titles=None, folder=''):
    """Plot 2D density histogram for alanine from the dihedral angles."""
    newcmp = dihedral_map()

    n_plots = len(list_angles)
    fig, axs = plt.subplots(ncols=n_plots, figsize=(6.4 * n_plots, 4.8),
                            constrained_layout=True)

    images = []
    for i in range(n_plots):
        h, x_edges, y_edges  = jnp.histogram2d(
            list_angles[i][:, 0], list_angles[i][:, 1], bins=60, density=True)
        h_masked = onp.where(h == 0, onp.nan, h)
        x, y = onp.meshgrid(x_edges, y_edges)
        images.append(axs[i].pcolormesh(x,y,h_masked.T, cmap=newcmp))
        if titles:
            axs[i].set_title(titles[i])
        annotate_alanine_histrogram(axs[i])

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    cbar = fig.colorbar(images[0], ax=axs)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label('Density')

    plt.savefig(f'output/postprocessing/{folder}/histogram_compare_'
                f'density_{saveas}.pdf')
    plt.close('all')


def visualize_dihedral_series(angles, saveas, dihedral_ref=None):
    """Visualize the initial points of parallel trajectories
    during training. Option to add a reference histogram as a background."""
    fig, ax = plt.subplots()
    if dihedral_ref is not None:
        newcmp = dihedral_map()
        plt.hist2d(dihedral_ref[:, 0], dihedral_ref[:, 1], bins=60, cmap=newcmp,
                   cmin=1)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel('$\phi$ in $\mathrm{deg}$')
    plt.ylabel('$\psi$ in $\mathrm{deg}$')
    #Plot initial points
    plt.scatter(angles[0, :, 0], angles[0, :, 1], color='tab:orange')
    graph = plt.scatter([], [], color='tab:blue')

    def update(i):
        if i % 10 == 0:
            label = f'Epoch {i}'
            print(label)
        # Plot next points dynamically
        graph.set_offsets(angles[1:i+1].reshape((i * angles.shape[1], 2)))
        ax.set_title('Epoche ' + str(i+1))
        return graph

    anim = FuncAnimation(fig, update, frames=angles.shape[0] - 1, interval=200,
                         repeat=False)
    anim.save(f'output/postprocessing/dihedral_series_{saveas}.gif', dpi=80,
              writer='imagemagick', extra_args=['-loop', '1'])


def plot_dihedral_diff(angles, saveas, dihedral_ref):
    """Plots the difference between the reference dihedrals and the predicted
    angles."""
    h_ref, x_ref, y_ref = jnp.histogram2d(
        dihedral_ref[:, 0], dihedral_ref[:, 1], bins=60, density=True)
    h_predicted, _, _ = jnp.histogram2d(angles[:, 0], angles[:, 1],
                                        bins=60, density=True)

    x, y = onp.meshgrid(x_ref, y_ref)
    h_diff = h_ref - h_predicted
    h_max = jnp.max(h_diff)
    vmin, vmax = -h_max, h_max
    norm = clr.Normalize(vmin, vmax)
    cmap = plt.get_cmap('PiYG')

    plt.figure()
    plt.pcolormesh(x, y, h_diff.T, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.savefig(f'output/postprocessing/diff_dihedral_{saveas}.pdf')


def plot_histogram_free_energy(angles, saveas, kbt, degrees=True, folder=''):
    """Plot 2D free energy histogram for alanine from the dihedral angles."""
    cmap = plt.get_cmap('magma')

    if degrees:
        angles = jnp.deg2rad(angles)

    h, x_edges, y_edges = jnp.histogram2d(angles[:, 0], angles[:, 1],
                                          bins=60, density=True)

    h = jnp.log(h) * -(kbt / 4.184)
    x, y = onp.meshgrid(x_edges, y_edges)

    plt.figure()
    plt.pcolormesh(x, y, h.T, cmap=cmap, vmax=5.25)
    cbar = plt.colorbar()
    cbar.set_label('Free Energy (kcal/mol)')
    plt.xlabel('$\phi$ in rad')
    plt.ylabel('$\psi$ in rad')
    plt.savefig(f'output/postprocessing/{folder}/histogram_free'
                f'_energy_{saveas}.pdf')
    plt.close('all')


def plot_compare_histogram_free_energy(list_angles, saveas, kbt, degrees=True,
                                       titles=None, folder=''):
    """Plot 2D free energy histogram for alanine from the dihedral angles.
    Comparison of multiple curves."""
    cmap = plt.get_cmap('magma')

    n_plots = len(list_angles)
    fig, axs = plt.subplots(ncols=n_plots, figsize=(6.4 * n_plots, 4.8),
                            constrained_layout=True)

    images = []
    for i in range(n_plots):
        if degrees:
            angles = jnp.deg2rad(list_angles[i])
        else:
            angles = list_angles[i]
        h, x_edges, y_edges = jnp.histogram2d(angles[:, 0], angles[:, 1],
                                              bins=60, density=True)
        h_masked = jnp.log(h) * -kbt / 4.184
        x, y = onp.meshgrid(x_edges, y_edges)
        images.append(axs[i].pcolormesh(x, y, h_masked.T, cmap=cmap))
        axs[i].set_xlabel('$\phi$ in rad')
        axs[i].set_ylabel('$\psi$ in rad')
        if titles:
            axs[i].set_title(titles[i])

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = clr.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    cbar = fig.colorbar(images[0], ax=axs)
    cbar.set_label('Free Energy (kcal/mol)')

    plt.savefig(f'output/postprocessing/{folder}/histogram_compare'
                f'_free_energy_{saveas}.pdf')
    plt.close('all')


def _spline_free_energy(angles, bins, kbt):
    h, x_bins = jnp.histogram(angles, bins=bins, density=True)
    h = jnp.log(h) * -kbt / 4.184
    h_spline = onp.where(h == jnp.inf, 15, h)
    width = x_bins[1] - x_bins[0]
    bin_center = x_bins + width / 2
    cubic_interploation_model = interp1d(bin_center[:-1], h_spline,
                                         kind='cubic')
    x_ = onp.linspace(bin_center[:-1].min(), bin_center[:-1].max(), 40)
    y_ = cubic_interploation_model(x_)
    return x_, y_


def plot_compare_1d_free_energy(angles, reference, saveas, labels, kbt, bins=60,
                                degrees=True, xlabel='$\phi$ in rad',
                                folder=''):
    """Plot and save spline interpolation of the 1D histogram for
    alanine dipeptide free energies from the dihedral angles.
    angles: angles in form of list of numpy arrays of [Nmodels, Nangles].
    Default for phi angles.
    Use xlabel='$\psi$' for psi angles.
    """
    if degrees:
        reference = jnp.deg2rad(reference)
    x_ref, y_ref = _spline_free_energy(reference, bins, kbt)
    plt.figure()
    plt.plot(x_ref, y_ref, color='k', linestyle='--', label='Reference',
             linewidth=2.5)
    n_models = len(angles)
    for i in range(n_models):
        angle = jnp.deg2rad(angles[i]) if degrees else angles[i]
        x, y = _spline_free_energy(angle, bins, kbt)
        plt.plot(x, y, linewidth=2.5, label=labels[i])

    plt.xlabel(xlabel)
    plt.ylabel('Free Energy (kcal/mol)')
    plt.xlim([-jnp.pi, jnp.pi])
    plt.ylim([0, 7])
    plt.title(saveas)
    plt.legend()
    plt.savefig(f'output/postprocessing/{folder}/free_energy_1D_{saveas}.pdf')
    plt.close('all')


def _evaluate_densities(angles, bins, hist_range):
    """Evaluates dihedral density for an [N_traj, N] array of dihedrals."""
    n_traj = angles.shape[0]
    histograms = onp.zeros((n_traj, bins))
    for j in range(n_traj):
        h, x_bins = jnp.histogram(angles[j, :], bins=bins,
                                  density=True, range=hist_range)
        width = x_bins[1] - x_bins[0]
        bin_center = x_bins + width / 2
        histograms[j] = h
    return histograms, bin_center


def plot_1d_dihedral_sigma(ref_angles, pred_angles, path, bins=60,
                           xlabel='$\phi$ in deg'):
    """Plot  1D histogram splines for alanine dipeptide dihedral angles with
    mean and standard deviation for different models.

    angles: angles in form of list of [Ntrajectory x Nangles] or
    numpy arrays of [Nmodels, Ntrajectory, Nangles].
    Default for phi angles. Use xlabel='$\psi$' for psi angles.
    """

    hist_range = [-180, 180]
    h_ref, bin_center = _evaluate_densities(ref_angles, bins, hist_range)
    ref_mean = onp.mean(h_ref, axis=0)

    h_pred, bin_center = _evaluate_densities(pred_angles, bins, hist_range)
    bin_center = bin_center[:-1]

    ref_density = {'coordinates': bin_center,
                   'values': ref_mean}

    Postprocessing.plot_sigmas(
        h_pred, bin_center, path, reference_data=ref_density, x_axis=xlabel,
        y_axis=r'$\mathrm{Density}$', yticks=[0., 0.005, 0.01, 0.015, 0.02],
        style='sci', y_lim=[-0.001, 0.02])


def plot_1d_dihedral(angles, saveas, labels, bins=60, degrees=True,
                     location=None, xlabel='$\phi$ in deg', folder='',
                     color=None, line=None, plot_format='pdf'):
    """Plot  1D histogram splines for alanine dipeptide dihedral angles with
    mean and standard deviation for different models.

    angles: angles in form of list of [Ntrajectory x Nangles] or
    numpy arrays of [Nmodels, Ntrajectory, Nangles].
    Default for phi angles. Use xlabel='$\psi$' for psi angles.
    """
    plt.figure()
    if color is None:
        color = ['k', '#00A087FF', '#3C5488FF']
    if line is None:
        line = ['--', '-', '-']
    n_models = len(angles)
    for i in range(n_models):
        if degrees:
            angles_conv = angles[i]
            hist_range = [-180, 180]
        else:
            angles_conv = onp.rad2deg(angles[i])
            hist_range = [-onp.pi, onp.pi]

        h_temp, bin_center = _evaluate_densities(angles_conv, bins, hist_range)
        h_mean = jnp.mean(h_temp, axis=0)
        h_std = jnp.std(h_temp, axis=0)
        plt.plot(bin_center[:-1], h_mean, label=labels[i], color=color[i],
                 linestyle=line[i], linewidth=2.0)
        plt.fill_between(bin_center[:-1], h_mean - 2. * h_std,
                         h_mean + 2. * h_std,
                         color=color[i], alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    if location is not None:
        plt.legend(loc=location)
    else:
        plt.legend()
    plt.savefig(f'output/postprocessing/{folder}/'
                f'dihedral_1D_{saveas}.{plot_format}')
    plt.close('all')


def plot_density_percentiles(dihedrals, phi_angles_ref, psi_angles_ref, folder,
                             bins=60):
    phi_histograms, bin_centers = _evaluate_densities(
        dihedrals[:, :, 0], bins, [-180., 180.])
    psi_histograms, _ = _evaluate_densities(
        dihedrals[:, :, 1], bins, [-180., 180.])
    phi_ref, _ = _evaluate_densities(
        phi_angles_ref, bins, [-180., 180.])
    psi_ref, _ = _evaluate_densities(
        psi_angles_ref, bins, [-180., 180.])
    bin_centers = bin_centers[:-1]
    reference_data_phi = {'coordinates': bin_centers,
                          'values': jnp.mean(phi_ref, axis=0)}
    reference_data_psi = {'coordinates': bin_centers,
                          'values': jnp.mean(psi_ref, axis=0)}
    Postprocessing.plot_percentile(
        phi_histograms, bin_centers, f'{folder}/phi_percentiles',
        reference_data=reference_data_phi, x_axis='$\phi$ in rad$',
        y_axis='Density')
    Postprocessing.plot_percentile(
        psi_histograms, bin_centers, f'{folder}/psi_percentiles',
        reference_data=reference_data_psi, x_axis='$\psi$ in rad$',
        y_axis='Density')
