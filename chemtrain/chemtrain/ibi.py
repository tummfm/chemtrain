import numpy as onp
from jax import numpy as np
from scipy import interpolate as sci_interpolate
import matplotlib.pyplot as plt


def initial_guess(RDF_target, kbT):
    """Compute IBI initial guess via potential of mean force (PMF)"""
    u_vals_initial = onp.asarray(-kbT) * onp.log(RDF_target)  # potential of mean force

    # to avoid inf where rdf is 0: keep force = potential derivative constant from last non-inf point onwards
    # we assume that infs are all grouped together at the left side of the PMF
    u_vals_initial[onp.isinf(u_vals_initial)] = onp.nan
    maximum_index = onp.nanargmax(u_vals_initial)  # last non-inf value
    max_difference = u_vals_initial[maximum_index] - u_vals_initial[maximum_index + 1]
    correction_array = onp.flip(onp.arange(maximum_index) + 1.)
    correction_array = correction_array * max_difference + u_vals_initial[maximum_index]
    u_vals_initial[:maximum_index] = correction_array

    return u_vals_initial


def update_potential(cur_RDF, target_RDF, u_vals, kbT):
    """IBI update step. To handle 0s, we cast them to a very small number to avoid inf/nan"""
    epsilon = 1.e-7
    target_safe = np.where(target_RDF < epsilon, epsilon, target_RDF)
    cur_safe = np.where(cur_RDF < epsilon, epsilon, cur_RDF)
    delta_u = kbT * np.log(cur_safe / target_safe)
    return u_vals + delta_u


def load_table(location, x_vals):
    """Assumes that x values are stored in first column and corresponding values in the second column"""
    tabulated_array = onp.loadtxt(location)
    y_interpolator = sci_interpolate.interp1d(tabulated_array[:, 0], tabulated_array[:, 1], kind='cubic')
    y_vals = onp.array(y_interpolator(x_vals))
    return y_vals


def plot_verification_PMF(x_vals, energy_PMF_comparison):
    initial_guess_table_loc = 'data/IBI_Initial_guess.csv'
    u_vals_reference = load_table(initial_guess_table_loc, x_vals)
    plt.figure()
    plt.plot(x_vals, energy_PMF_comparison, label='My_PMF')
    plt.plot(x_vals, u_vals_reference, label='Reference', linestyle='--')
    plt.legend()
    plt.ylim(-2.5, 10.)
    plt.savefig('Figures/PMF.png')
    return

def plot_potential_iterations(potentials):
    plt.figure()
    for i, u_vals in enumerate(potentials):
        plt.plot(u_vals, label=str(i))
    plt.ylim(-3., 3.)
    plt.legend()
    plt.savefig('Figures/IBI_Iterations.png')
    return
