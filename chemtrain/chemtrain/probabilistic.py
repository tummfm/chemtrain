"""This module provides utilities for setting up probabilistic trainers,
 such as trainers.SGMC"""
import abc
import time

from jax import (lax, vmap, checkpoint, random, jit, device_count,
                 scipy as jscipy, numpy as jnp, tree_map)
from jax_md import quantity
from jax_sgmc import data, potential
from scipy import stats
import tree_math

from chemtrain import force_matching, util, traj_util, dropout, data_processing


# Modeling


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
    """Returns the likelihood function for Bayesian potential optimization
    based on a force-matching formulation.

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


def init_force_matching(
        energy_param_prior, energy_fn_template, nbrs_init, init_params,
        position_data, energy_data=None, energy_scale=None, force_data=None,
        force_scale=None, virial_data=None, virial_scale=None, box_tensor=None,
        train_ratio=0.7, val_ratio=0.1,
        likelihood_distribution=jscipy.stats.norm.logpdf, draw_std=False):
    """Initializes a compatible set of prior, likelihood, initial MCMC samples
    as well as train and validation loaders  for learning probabilistic
    potentials via force-matching.

    Data scales are used for parametrization of the exponential prior
    distributions for the standard deviations of the (energy, force and/or
    virial) likelihood distributions. Note that scale = 1 / lambda for the
    common parametrization of the exponential  distribution via the rate
    parameter lambda. See the scipy.stats.expon documentation for more details.
    This allows accounting for the different scales of energies, forces and
    virial, similar to the los weights in standard force matching.

    Args:
        energy_param_prior: Prior function for , e.g. as generated from
                            'init_elementwise_prior_fn'.
        energy_fn_template: Energy function template
        nbrs_init: Initial neighbor list
        init_params: Initial energy params
        position_data: (N_snapshots x N_particles x dim) array of particle
                       positions
        energy_data: (N_snapshots,) array of corresponding energy values,
                     if applicable
        energy_scale: Prior scale of energy data.
        force_data: (N_snapshots x N_particles x dim) array of corresponding
                    forces acting on particles, if applicable.
        force_scale: Prior scale of force components.
        virial_data: (N_snapshots,) or (N_snapshots, dim, dim) array of
                     corresponding virial (tensor) values (without kinetic
                     contribution), if applicable.
        virial_scale: Prior scale of virial components.
        box_tensor: Box tensor, only needed if virial_data used.
        train_ratio: Ratio of dataset to be used for training. The remaining
                     data can be used for validation.
        val_ratio: Ratio of dataset to be used for validation. The remaining
                   data will be used for testing.
        likelihood_distribution: Likelihood distribution, defaults to Gaussian.
        draw_std: If True, draws initial likelihood std values from the
                  exponential prior distribution. If False, sets it to the
                  mean of the prior distribution, i.e. the scale.

    Returns:
        A tuple (prior_fn, likelihood_fn, init_samples, train_loader,
        val_loader, test_loader, test_set). Prior and likelihood can be used to
        construct the potential function for Hamiltonian MCMC formulations.
        Init_samples is a list of initial values for multiple MCMC chains,
        e.g. for SGMCMCTrainer. The data loaders are jax-sgmc NumpyDataloaders
        used for training, validation and testing. The test_set can be used for
        further analyses of the trained model on unseen data.
    """
    dataset = force_matching.build_dataset(position_data, energy_data,
                                           force_data, virial_data)
    train_loader, val_loader, test_loader, test_set = \
        data_processing.init_dataloaders(dataset, train_ratio, val_ratio)

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

    return (prior_fn, likelihood_fn, init_samples, train_loader, val_loader,
            test_loader, test_set)


def init_log_posterior_fn(likelihood, prior, train_loader, batch_size,
                          batch_cache):
    """Initializes the log-posterior function.

    Initializes a function that computes the log-posterior value of a parameter
    sample. The full log-posterior is computed batch-wise to avoid out-of-memory
    errors during the forward and backward pass.

    Args:
        likelihood: Likelihood function
        prior: Prior function
        train_loader: Train data loader
        batch_size: Batch-size for batch-wise computation of the sub-likelihoods
        batch_cache: Number of batches to store in cache

    Returns:
        Log-posterior function
    """
    likelihood = checkpoint(likelihood)  # avoid OOM for grad over whole dataset
    full_potential_fn = potential.full_potential(prior, likelihood,
                                                 strategy='vmap')
    init_fun, fmap_fun = data.full_reference_data(train_loader,
                                                  batch_cache,
                                                  batch_size)
    data_state = init_fun()

    # TODO this is not valid with dataloader; possibly use masked
    #  epoch-wise loader
    def log_posterior_fn(sample):
        potential_val, _ = full_potential_fn(sample, data_state, fmap_fun)
        return -potential_val  # potential is negative posterior
    return log_posterior_fn

# Trainers


class ProbabilisticFMTrainerTemplate(util.TrainerInterface):
    """Trainer template for methods that result in multiple parameter sets for
    Monte-Carlo-style uncertainty quantification, based on a force-matching
    formulation.
    """
    def __init__(self, checkpoint_path, energy_fn_template,
                 val_dataloader=None):
        super().__init__(checkpoint_path, energy_fn_template)
        self.results = []

        # TODO use val_loader for some metrics that are interesting for MCMC
        #  and SG-MCMC

    def move_to_device(self):
        params = []
        for param_set in self.params:
            params.append(tree_map(jnp.array, param_set))  # move on device
        self.params = params

    @property
    @abc.abstractmethod
    def list_of_params(self):
        """ Returns a list containing n single model parameter sets, where n
        is the number of samples. This provides a more intuitive parameter
        interface that self.params, which returns a large set of parameters,
        where n is the leading axis of each leaf. Self.params is most useful,
        if parameter sets are mapped via map or vmap in a postprocessing step.
        """


class MCMCForceMatchingTemplate(ProbabilisticFMTrainerTemplate):
    """Initializes log_posterior function to be used for MCMC with blackjax,
    including batch-wise evaluation of the likelihood and re-materialization.
    """
    def __init__(self, init_state, kernel, checkpoint_path, val_loader=None,
                 ref_energy_fn_template=None):
        super().__init__(checkpoint_path, ref_energy_fn_template, val_loader)
        # re-materialization of the likelihood for each batch of data allows
        # circumventing the enormous memory requirements of backpropagating
        # through the full potential - at the expense of additional
        # computational cost.

        self.kernel = jit(kernel)
        self.state = init_state

    def train(self, num_samples, checkpoint_freq=None, init_samples=None,
              rng_key=random.PRNGKey(0)):
        if init_samples is not None:
            # TODO implement multiple chains
            raise NotImplementedError

        for i in range(num_samples):
            start_time = time.time()
            rng_key, consumed_key = random.split(rng_key)
            self.state, info = self.kernel(consumed_key, self.state)
            self.results.append(self.state)
            print(f'Time for sample {i}: {(time.time() - start_time) / 60.}'
                  f' min.', info)
            self._epoch += 1
            self._dump_checkpoint_occasionally(frequency=checkpoint_freq)

    @property
    def list_of_params(self):
        return [state.position['params'] for state in self.results]

    @property
    def params(self):
        return util.tree_stack(self.list_of_params)

    @params.setter
    def params(self, loaded_params):
        raise NotImplementedError('Setting params seems not meaningful for MCMC'
                                  ' samplers.')


# Uncertainty propagation

def init_force_uq(energy_fn_template, n_splits=16, vmap_batch_size=1):
    n_devies = device_count()
    util.assert_distributable(n_splits, n_devies, vmap_batch_size)

    @jit
    def forces(keys, energy_params, sim_state):

        def single_force(key):
            state, nbrs = sim_state  # assumes state and nbrs to be in sync
            dropout_params = dropout.build_dropout_params(energy_params, key)
            energy_fn = energy_fn_template(dropout_params)
            force_fn = quantity.canonicalize_force(energy_fn)
            return force_fn(state.position, neighbor=nbrs)

        # map in case not all necessary samples per device fit memory for vmap
        mapped_force = lax.map(single_force, keys)
        return mapped_force

    def force_uq(meta_params, sim_state):
        energy_params, key = dropout.split_dropout_params(meta_params)
        keys = random.split(key, n_splits)
        keys = keys.reshape((vmap_batch_size, -1, 2))  # 2 values per key
        # TODO add pmap
        # keys = keys.reshape((n_devies, vmap_batch_size, -1, 2))
        vmap_forces = vmap(forces, (0, None, None))
        batched_forces = vmap_forces(keys, energy_params, sim_state)
        shape = batched_forces.shape
        # reshape such that all sampled force predictions are along axis 0
        lined_forces = batched_forces.reshape((-1, shape[-2], shape[-1]))
        # TODO check that std and forces are correct
        f_std_per_atom = jnp.std(lined_forces, axis=0)
        mean_std = jnp.mean(f_std_per_atom)
        return mean_std

    return force_uq


def infer_output_uncertainty(param_sets, init_state, trajectory_generator,
                             quantities, total_samples, kt_schedule=None,
                             vmap_simulations_per_device=1):
    n_devies = device_count()

    # Check whether dropout was used or not
    dropout_active = dropout.dropout_is_used(param_sets)
    # TODO add vmap
    # TODO add pmap

    if dropout_active:  # map over keys
        energy_params, key = dropout.split_dropout_params(param_sets)
        param_sets = random.split(key, total_samples)

    def single_prediction(mapped_param):
        if dropout_active:  # param == key and we extract new parameter set
            param_set = dropout.build_dropout_params(energy_params,
                                                     mapped_param)
        else:
            param_set = mapped_param

        traj_state = trajectory_generator(param_set, init_state,
                                          kt_schedule=kt_schedule)
        quantity_traj = traj_util.quantity_traj(traj_state, quantities,
                                                param_set)
        # TODO replace with new interface of DiffTRe:
        predictions = {}
        for quantity_key in quantities:
            quantity_snapshots = quantity_traj[quantity_key]
            predictions[quantity_key] = jnp.mean(quantity_snapshots, axis=0)
        return predictions

    # accumulates predictions in axis 0 of leaves of prediction_dict
    predictions = lax.map(single_prediction, param_sets)
    return predictions


def mcmc_statistics(uq_predictions):
    statistics = {}
    for quantity_key in uq_predictions:
        quantity_samples = uq_predictions[quantity_key]
        statistics[quantity_key] = {'mean': jnp.mean(quantity_samples, axis=0),
                                    'std': jnp.std(quantity_samples, axis=0)}
    return statistics


def validation_mae_params_fm(params, val_loader, energy_fn_template, nbrs_init,
                             box_tensor=None, batch_size=1, batch_cache=1):
    """Evaluates the mean absolute error of a list of parameter sets generated
    via sampling-based methods, based on validation data and a force-matching
    likelihood.
    """

    # test if virial data is contained in val_loader and initialize virial_fn
    # according to the virial data type.
    test_batch = val_loader.initializer_batch(batch_size)
    if 'p' in test_batch:
        test_virial = test_batch['p']
    else:
        test_virial = None

    virial_fn = force_matching.init_virial_fn(test_virial, energy_fn_template,
                                              box_tensor)
    mae_fn, mae_data_state = force_matching.init_mae_fn(
        val_loader, nbrs_init, energy_fn_template,
        batch_size, batch_cache, virial_fn
    )

    maes = []
    for i, param_set in enumerate(params):
        mae, mae_data_state = mae_fn(param_set, mae_data_state)
        maes.append(mae)
        for key, mae_value in mae.items():
            print(f'Parameter set {i}: {key}: MAE = {mae_value:.4f}')
    return maes
