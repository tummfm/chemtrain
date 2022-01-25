"""This module provides utilities for setting up probabilistic trainers,
 such as trainers.SGMC"""
from jax import scipy as jscipy, numpy as jnp
from scipy import stats
import tree_math

from chemtrain import force_matching


def uniform_prior(sample):
    """Uniform improper prior function.

    Args:
        sample: Dummy argument

    Returns:
        A uniform logpdf value of 0.
    """
    del sample
    return 0.


def init_elementwise_prior_fn(scale, distribution=jscipy.stats.norm.logpdf,
                              loc=0.):
    """Initializes a prior distribution that acts on all parameters
    independently.

    Args:
        scale: Scale of the prior distribution.
        distribution: Prior logpdf, defaults to Gaussian.
        loc: Location of prior mean. Usually default of 0.

    Returns:
        Prior function
    """
    def prior_fn(params):
        params = tree_math.Vector(params)
        log_pdf = tree_math.unwrap(distribution, vector_argnums=0)(params,
                                                                   loc=loc,
                                                                   scale=scale)
        return log_pdf.sum()
    return prior_fn


def init_likelihood(energy_fn_template, nbrs_init, virial_fn=None,
                    distribution=jscipy.stats.norm.logpdf):
    """Returns the likelihood function for Bayesian potential optimization.

    Args:
        energy_fn_template: Energy function template
        nbrs_init: Neighbor list to be used for initialization
        virial_fn: Virial function compatible ẃith virial data type,
                   e.g. initialized via force_matching.init_virial_fn.
        distribution: Likelihood distribution. Defaults to a Gaussian logpdf,
                      but any jax.scipy logpdf with the same signature can
                      be provided.
    """
    single_prediction = force_matching.init_single_prediction(
        nbrs_init, energy_fn_template, virial_fn)

    def sum_log_likelihood(predictions, targets, std):
        likelihoods = distribution(predictions, loc=targets, scale=std)
        return jnp.sum(likelihoods)

    def likelihood_fn(sample, observation):
        prediction = single_prediction(sample['params'], observation['R'])

        likelihood = 0.
        if 'U' in observation.keys():  # energy likelihood component
            likelihood += sum_log_likelihood(prediction['U'], observation['U'],
                                             sample['U_std'])
        if 'F' in observation.keys():  # forces likelihood component
            likelihood += sum_log_likelihood(prediction['F'], observation['F'],
                                             sample['F_std'])
        if 'p' in observation.keys():  # virial likelihood component
            likelihood += sum_log_likelihood(prediction['p'], observation['p'],
                                             sample['p_std'])
        return likelihood
    return likelihood_fn


def _set_or_draw_exponential_rv(rv_scale, draw=True):
    """Returns a random variable, either drawn from the exponential distribution
     or set to its mean value. Also ensures the mean is defined.
     """
    assert rv_scale is not None
    if draw:
        rv = stats.expon.rvs(scale=rv_scale)
    else:
        rv = rv_scale  # exponential distribution mean is scale = 1 / lambda
    return jnp.array(rv)


def dataloader_likelihood_fm(
        energy_param_prior, energy_fn_template, nbrs_init, init_params,
        position_data, energy_data=None, energy_scale=None, force_data=None,
        force_scale=None, virial_data=None, virial_scale=None, box_tensor=None,
        train_ratio=0.875, likelihood_distribution=jscipy.stats.norm.logpdf,
        draw_std=False):
    """
    Also multiple init samples possible, build list of samples following the
    dict layout in this function.

    Data scales are used for parametrization of the exponential prior
    distributions for the standard deviation of the likelihood distribution.
    Note that scale = 1 / lambda for the common parametrization of the
    exponential  distribution via the rate parameter lambda. See the
    scipy.stats.expon documentation for more details.
    This allows for accounting for the different scales of energies, forces and
    virial, similar to the los weights in standard force matching.

    Args:
        energy_param_prior:
        energy_fn_template:
        nbrs_init:
        init_params:
        position_data:
        energy_data:
        energy_scale: Scale of energy data.
        force_data:
        force_scale:
        virial_data:
        virial_scale:
        box_tensor:
        train_ratio:
        likelihood_distribution:
        draw_std: If True, draws initial likelihood std values from the
                  exponential prior distribution. If False, sets it to the
                  mean of the prior distribution.

    Returns:

    """
    train_loader, val_loader = force_matching.init_dataloaders(
        position_data, energy_data, force_data, virial_data, train_ratio)

    virial_fn = force_matching.init_virial_fn(virial_data, energy_fn_template,
                                              box_tensor)

    likelihood_fn = init_likelihood(energy_fn_template, nbrs_init, virial_fn,
                                    likelihood_distribution)

    def prior_fn(sample):
        prior = energy_param_prior(sample['params'])
        if 'U_std' in sample:
            prior += jscipy.stats.expon.logpdf(sample['U_std'],
                                               scale=energy_scale)
        if 'F_std' in sample:
            prior += jscipy.stats.expon.logpdf(sample['F_std'],
                                               scale=force_scale)
        if 'p_std' in sample:
            prior += jscipy.stats.expon.logpdf(sample['p_std'],
                                               scale=virial_scale)
        return prior

    if not isinstance(init_params, list):  # canonicalize init_params
        init_params = [init_params]

    init_samples = []
    for energy_params in init_params:
        sample = {'params': energy_params}
        if energy_data is not None:
            assert energy_scale is not None
            sample['U_std'] = _set_or_draw_exponential_rv(energy_scale,
                                                          draw_std)
        if force_data is not None:
            assert force_scale is not None
            sample['F_std'] = _set_or_draw_exponential_rv(force_scale,
                                                          draw_std)
        if virial_data is not None:
            assert virial_scale is not None
            sample['p_std'] = _set_or_draw_exponential_rv(virial_scale,
                                                          draw_std)
        init_samples.append(sample)

    return prior_fn, likelihood_fn, init_samples, train_loader, val_loader
