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

"""A collection of functions evaluating quantiities of trajectories.
For easiest integration into chemtain, functions should be compatible with
traj_util.quantity_traj. Functions provided to quantity_traj need to take the
state and additional kwargs.
"""
from functools import partial

from jax import grad, vmap, lax, jacrev, jacfwd, numpy as jnp
from jax.scipy.stats import norm
from jax_md import space, util, dataclasses, quantity, simulate

from chemtrain import sparse_graph
from chemtrain.pickle_jit import jit

Array = util.Array


def energy_wrapper(energy_fn_template):
    """Wrapper around energy_fn to allow energy computation via
    traj_util.quantity_traj.
    """
    def energy(state, neighbor, energy_params, **kwargs):
        energy_fn = energy_fn_template(energy_params)
        return energy_fn(state.position, neighbor=neighbor, **kwargs)
    return energy


def kinetic_energy_wrapper(state, **unused_kwargs):
    """Wrapper around kinetic_energy to allow kinetic energy computation via
    traj_util.quantity_traj.
    """
    return quantity.kinetic_energy(state.velocity, state.mass)


def total_energy_wrapper(energy_fn_template):
    """Wrapper around energy_fn to allow total energy computation via
    traj_util.quantity_traj.
    """
    def energy(state, neighbor, energy_params, **kwargs):
        energy_fn = energy_fn_template(energy_params)
        pot_energy = energy_fn(state.position, neighbor=neighbor, **kwargs)
        kinetic_energy = quantity.kinetic_energy(state.velocity, state.mass)
        return pot_energy + kinetic_energy
    return energy


def temperature(state, **unused_kwargs):
    """Temperature function that is consistent with quantity_traj interface."""
    return quantity.temperature(velocity=state.velocity, mass=state.mass)


def _dyn_box(reference_box, **kwargs):
    """Gets box dynamically from kwargs, if provided, otherwise defaults to
    reference. Ensures that a box is provided and deletes from kwargs.
    """
    box = kwargs.pop('box', reference_box)
    assert box is not None, ('If no reference box is given, needs to be '
                             'given as kwarg "box".')
    return box, kwargs


def volume_npt(state, **unused_kwargs):
    """Returns volume of a single snapshot in the NPT ensemble, e.g. for use in
     DiffTRe learning of thermodynamic fluctiations in chemtrain.traj_quantity.
     """
    dim = state.position.shape[-1]
    box = simulate.npt_box(state)
    volume = quantity.volume(dim, box)
    return volume


def _canonicalized_masses(state):
    if state.mass.ndim == 0:
        masses = jnp.ones(state.position.shape[0]) * state.mass
    else:
        masses = state.mass
    return masses


def density(state, **unused_kwargs):
    """Returns density of a single snapshot of the NPT ensemble."""
    masses = _canonicalized_masses(state)
    total_mass = jnp.sum(masses)
    volume = volume_npt(state)
    return total_mass / volume


# TODO distinct classes and discretization functions don't seem optimal
#  --> possible refactor


@dataclasses.dataclass
class RDFParams:
    """Hyperparameters to initialize the radial distribution function (RDF).

    Attributes:
    reference_rdf: The target rdf; initialize with None if no target available
    rdf_bin_centers: The radial positions of the centers of the rdf bins
    rdf_bin_boundaries: The radial positions of the edges of the rdf bins
    sigma_RDF: Standard deviation of smoothing Gaussian
    """
    reference: Array
    rdf_bin_centers: Array
    rdf_bin_boundaries: Array
    sigma: Array


def rdf_discretization(rdf_cut, nbins=300, rdf_start=0.):
    """Computes dicretization parameters for initialization of RDF function.

    Args:
        rdf_cut: Cut-off length inside which pairs of particles are considered
        nbins: Number of bins in radial direction
        rdf_start: Minimal distance after which particle pairs are considered

    Returns:
        Arrays with radial positions of bin centers, bin edges and the standard
        deviation of the Gaussian smoothing kernel.

    """
    dx_bin = (rdf_cut - rdf_start) / float(nbins)
    rdf_bin_centers = jnp.linspace(rdf_start + dx_bin / 2.,
                                   rdf_cut - dx_bin / 2.,
                                   nbins)
    rdf_bin_boundaries = jnp.linspace(rdf_start, rdf_cut, nbins + 1)
    sigma_rdf = jnp.array(dx_bin)
    return rdf_bin_centers, rdf_bin_boundaries, sigma_rdf


@dataclasses.dataclass
class ADFParams:
    """Hyperparameters to initialize a angular distribution function (ADF).

    Attributes:
    reference_adf: The target adf; initialize with None if no target available
    adf_bin_centers: The positions of the centers of the adf bins over theta
    sigma_ADF: Standard deviation of smoothing Gaussian
    r_outer: Outer radius beyond which particle triplets are not considered
    r_inner: Inner radius below which particle triplets are not considered
    """
    reference: Array
    adf_bin_centers: Array
    sigma: Array
    r_outer: Array
    r_inner: Array


def adf_discretization(nbins=200):
    """Computes dicretization parameters for initialization of ADF function.

    Args:
        nbins: Number of bins discretizing theta

    Returns:
        Arrays containing bin centers in theta direction and the standard
        deviation of the Gaussian smoothing kernel.
    """
    dtheta_bin = jnp.pi / float(nbins)
    adf_bin_centers = jnp.linspace(dtheta_bin / 2.,
                                   jnp.pi - dtheta_bin / 2.,
                                   nbins)
    sigma_adf = util.f32(dtheta_bin)
    return adf_bin_centers, sigma_adf


@dataclasses.dataclass
class TCFParams:
    """Hyperparameters to initialize a triplet correlation function (TFC)
    for a triplet with sides x, y, z. Implementation according to
    https://aip.scitation.org/doi/10.1063/1.4898755 and
    https://aip.scitation.org/doi/10.1063/5.0048450.

    Attributes:
    reference_tcf: The target tcf; initialize with None if no target available
    sigma_TCF: Standard deviation of smoothing Gaussian
    volume_TCF: Histogram volume element according to
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.42.849
    tcf_x_bin_centers: The radial positions of the centers of the tcf bins in
                       x direction
    tcf_x_bin_centers: The radial positions of the centers of the tcf bins in
                       y direction
    tcf_x_bin_centers: The radial positions of the centers of the tcf bins in
                       z direction
    """
    reference: Array
    sigma_tcf: Array
    volume: Array
    tcf_x_bin_centers: Array
    tcf_y_bin_centers: Array
    tcf_z_bin_centers: Array


def tcf_discretization(tcf_cut, nbins=30, tcf_start=0.1):
    """Computes dicretization parameters for initialization of TCF function.

    Args:
        tcf_cut: Cut-off length inside which pairs of particles are considered
        nbins: Number of bins in all three radial direction
        tcf_start: Minimal distance after which particle pairs are considered

    Returns:
        Tuple containing standard deviation of the Gaussian smoothing kernel,
        histogram volume array and arrays with radial positions of bin centers
        in x, y, z.

    """
    dx_bin = (tcf_cut - tcf_start) / float(nbins)
    tcf_bin_centers = jnp.linspace(tcf_start + dx_bin / 2.,
                                   tcf_cut - dx_bin / 2.,
                                   nbins)

    tcf_x_binx_centers, tcf_y_bin_centers, tcf_z_bin_centers = jnp.meshgrid(
        tcf_bin_centers, tcf_bin_centers, tcf_bin_centers, sparse=True)

    sigma_tcf = jnp.array(dx_bin)
    # volume for non-linear triplets (sigma / min(x,y,z)->0)
    volume_tcf = (8 * jnp.pi**2 * tcf_x_binx_centers * tcf_y_bin_centers
                  * tcf_z_bin_centers * sigma_tcf**3)

    return (sigma_tcf, volume_tcf, tcf_x_binx_centers, tcf_y_bin_centers,
            tcf_z_bin_centers)


def _ideal_gas_density(particle_density, bin_boundaries):
    """Returns bin densities that would correspond to an ideal gas."""
    r_small = bin_boundaries[:-1]
    r_large = bin_boundaries[1:]
    bin_volume = (4. / 3.) * jnp.pi * (r_large**3 - r_small**3)
    bin_weights = bin_volume * particle_density
    return bin_weights


def init_rdf(displacement_fn, rdf_params, reference_box=None):
    """Initializes a function that computes the radial distribution function
    (RDF) for a single state.

    Args:
        displacement_fn: Displacement function
        rdf_params: RDFParams defining the hyperparameters of the RDF
        reference_box: Simulation box. Can be provided here for constant boxes
                       or on-the-fly as kwarg 'box', e.g. for NPT ensemble

    Returns:
        A function taking a simulation state and returning the instantaneous RDF
    """
    _, bin_centers, bin_boundaries, sigma = dataclasses.astuple(rdf_params)
    distance_metric = space.canonicalize_displacement_or_metric(displacement_fn)
    bin_size = jnp.diff(bin_boundaries)

    def pair_corr_fun(position, box):
        """Returns instantaneous pair correlation function while ensuring
        each particle pair contributes exactly 1.
        """
        n_particles = position.shape[0]
        metric = partial(distance_metric, box=box)
        metric = space.map_product(metric)
        dr = metric(position, position)
        # neglect same particles i.e. distance = 0.
        dr = jnp.where(dr > util.f32(1.e-7), dr, util.f32(1.e7))

        #  Gaussian ensures that discrete integral over distribution is 1
        exp = jnp.exp(util.f32(-0.5) * (dr[:, :, jnp.newaxis] - bin_centers)**2
                      / sigma**2)
        gaussian_distances = exp * bin_size / jnp.sqrt(2 * jnp.pi * sigma**2)
        pair_corr_per_particle = util.high_precision_sum(gaussian_distances,
                                                         axis=1)  # sum nbrs
        mean_pair_corr = util.high_precision_sum(pair_corr_per_particle,
                                                 axis=0) / n_particles
        return mean_pair_corr

    def rdf_compute_fun(state, **kwargs):
        box, _ = _dyn_box(reference_box, **kwargs)
        # Note: we cannot use neighbor list since RDF cutoff and
        # neighbor list cut-off don't coincide in general
        n_particles, spatial_dim = state.position.shape
        total_vol = quantity.volume(spatial_dim, box)
        particle_density = n_particles / total_vol
        mean_pair_corr = pair_corr_fun(state.position, box)
        # RDF is defined to relate the particle densities to an ideal gas.
        rdf = mean_pair_corr / _ideal_gas_density(particle_density,
                                                  bin_boundaries)
        return rdf
    return rdf_compute_fun


def _triplet_pairwise_displacements(position, neighbor, displacement_fn):
    """For each triplet of particles ijk, computes the displacement vectors
    r_kj = R_k - R_j and r_ij, i.e. the vector pointing from the central
    particle j to the side particles i and k. Returns a tuple (r_kj, r_ij)
    that contains the dispacement vectors for all triplets.
    r_kj.shape = (N_triplets, 3). This sparse format simplifies capping
    non-existant triplets that only result from the overcapacity of a dense
    neighborlist, realiszing a computational speed-up.
    """
    # TODO unification with sparse neighborlist format would simplify this.
    neighbor_displacement_fn = space.map_neighbor(displacement_fn)
    n_particles, max_neighbors = neighbor.idx.shape
    r_neigh = position[neighbor.idx]
    neighbor_displacement = neighbor_displacement_fn(position, r_neigh)
    neighbor_displacement_flat = jnp.reshape(
        neighbor_displacement, (n_particles * max_neighbors, 3))
    r_kj = jnp.repeat(neighbor_displacement_flat, max_neighbors, axis=0)
    r_ij = jnp.tile(neighbor_displacement, (1, max_neighbors, 1)).reshape(
        [n_particles * max_neighbors**2, 3])
    return r_kj, r_ij


def _angle_neighbor_mask(neighbor):
    """Returns a boolean (N_triplets,) array. Each entry corresponds to a
    triplet stored in the sparse displacement vectors. For each triplet, masks
    the cases of non-existing neighbors and both neighbors being the same
    particle. The cases j=k and j=i is already excluded by the neighbor list
    construction. For more details on the sparse triplet structure, see
    _triplet_pairwise_displacements.
    """
    n_particles, max_neighbors = neighbor.idx.shape
    edge_idx_flat = jnp.ravel(neighbor.idx)
    idx_k = jnp.repeat(edge_idx_flat, max_neighbors)
    idx_i = jnp.tile(neighbor.idx, (1, max_neighbors)).ravel()
    neighbor_mask = neighbor.idx != n_particles
    neighbor_mask_flat = jnp.ravel(neighbor_mask)
    mask_k = jnp.repeat(neighbor_mask_flat, max_neighbors)
    mask_i = jnp.tile(neighbor_mask, (1, max_neighbors)).ravel()
    # Note: mask structure is known a priori: precompute likely more efficient
    mask_i_eq_k = idx_i != idx_k
    mask = mask_k * mask_i * mask_i_eq_k
    mask = jnp.expand_dims(mask, axis=-1)
    return mask


def init_adf_nbrs(displacement_fn, adf_params, smoothing_dr=0.01, r_init=None,
                  nbrs_init=None, max_weight_multiplier=2.):
    """Initializes a function that computes the angular distribution function
    (ADF) for a single state.

    Angles are smoothed in radial direction via a Gaussian kernel (compare RDF
    function). In radial direction, triplets are weighted according to a
    Gaussian cumulative distribution function, such that triplets with both
    radii inside the cut-off band are weighted approximately 1 and the weights
    of triplets towards the band edges are soomthly reduced to 0.
    For computational speed-up and reduced memory needs, r_init and nbrs_init
    can be provided to estmate the maximum number of triplets - similarly to
    the maximum capacity of neighbors in the neighbor list.

    Caution: currrently the user does not receive information whether overflow
    occured. This function assumes that r_outer is smaller than the neighbor
    list cut-off. If this is not the case, a function computing all pairwise
    distances is necessary.

    Args:
        displacement_fn: Displacement function
        adf_params: ADFParams defining the hyperparameters of the ADF
        smoothing_dr: Standard deviation of Gaussian smoothing in radial
                      direction
        r_init: Initial position to estimate maximum number of triplets
        nbrs_init: Initial neighborlist to estimate maximum number of triplets
        max_weight_multiplier: Multiplier for estimate of number of triplets

    Returns:
        A function that takes a simulation state with neighborlist and returns
        the instantaneous adf.
    """

    _, bin_centers, sigma_theta, r_outer, r_inner = dataclasses.astuple(
        adf_params)
    sigma_theta = util.f32(sigma_theta)
    bin_centers = util.f32(bin_centers)

    def cut_off_weights(r_kj, r_ij):
        """Smoothly constraints triplets to a radial band such that both
        distances are between r_inner and r_outer. The Gaussian cdf is used for
        smoothing. The smoothing width can be controlled by the gaussian
        standard deviation.
        """
        dist_kj = space.distance(r_kj)
        dist_ij = space.distance(r_ij)
        # get d_small and d_large for each triplet
        pair_dist = jnp.column_stack((dist_kj, dist_ij)).sort(axis=1)
        # get inner boundary weight from r_small and outer weight from r_large
        inner_weight = norm.cdf(pair_dist[:, 0], loc=r_inner,
                                scale=smoothing_dr**2)
        outer_weight = 1 - norm.cdf(pair_dist[:, 1], loc=r_outer,
                                    scale=smoothing_dr**2)
        weights = jnp.expand_dims(outer_weight * inner_weight, axis=-1)
        return weights

    def weighted_adf(angles, weights):
        """Compute weighted ADF contribution of each triplet. For
        differentiability, each triplet contribution is smoothed via a Gaussian.
        """
        exp = jnp.exp(util.f32(-0.5) * (angles[:, jnp.newaxis] - bin_centers)**2
                      / sigma_theta**2)
        gaussians = exp / jnp.sqrt(2 * jnp.pi * sigma_theta**2)
        gaussians *= weights
        unnormed_adf = util.high_precision_sum(gaussians, axis=0)
        adf = unnormed_adf / jnp.trapz(unnormed_adf, bin_centers)
        return adf

    # we use initial configuration to estimate the maximum number of non-zero
    # weights for speedup and reduced memory cost
    if r_init is not None:
        if nbrs_init is None:
            raise ValueError('If we estimate the maximum number of triplets, '
                             'the initial neighbor list is a necessary input.')
        mask = _angle_neighbor_mask(nbrs_init)
        r_kj, r_ij = _triplet_pairwise_displacements(r_init, nbrs_init,
                                                     displacement_fn)
        weights = cut_off_weights(r_kj, r_ij)
        weights *= mask  # combine radial cut-off with neighborlist mask
        max_weights = jnp.count_nonzero(weights > 1.e-6) * max_weight_multiplier

    def adf_fn(state, neighbor, **kwargs):
        """Returns ADF for a single snapshot. Allows changing the box
        on-the-fly via the 'box' kwarg.
        """
        dyn_displacement = partial(displacement_fn, **kwargs)  # box kwarg
        mask = _angle_neighbor_mask(neighbor)
        r_kj, r_ij = _triplet_pairwise_displacements(
            state.position, neighbor, dyn_displacement)
        weights = cut_off_weights(r_kj, r_ij)
        weights *= mask  # combine radial cut-off with neighborlist mask

        if r_init is not None:  # prune triplets
            non_zero_weights = weights > 1.e-6
            _, sorting_idxs = lax.top_k(non_zero_weights[:, 0], max_weights)
            weights = weights[sorting_idxs]
            r_ij = r_ij[sorting_idxs]
            r_kj = r_kj[sorting_idxs]
            mask = mask[sorting_idxs]
            num_non_zero = jnp.sum(non_zero_weights)
            del num_non_zero
            # TODO check if num_non_zero > max_weights. if yes, send an error
            #  message to the user to increase max_weights_multiplier

        # ensure differentiability of tanh
        r_ij_safe, r_kj_safe = sparse_graph.safe_angle_mask(r_ij, r_kj, mask)
        angles = vmap(sparse_graph.angle)(r_ij_safe, r_kj_safe)
        adf = weighted_adf(angles, weights)
        return adf
    return adf_fn


def init_tcf_nbrs(displacement_fn, tcf_params,  reference_box=None,
                  nbrs_init=None, batch_size=1000, max_weight_multiplier=1.2):
    """Initializes a function that computes the triplet correlation function
    (TCF) for a single state.

    This function assumes that the neighbor list curoff matches the tcf_cutoff.

    Args:
        displacement_fn: Displacement function
        tcf_params: TCFParams defining the hyperparameters of the TCF
        nbrs_init: Initial neighborlist to estimate maximum number of triplets
        max_weight_multiplier: Multiplier for estimate of number of triplets
        batch_size: Batch size for more efficient binning of triplets
        reference_box: Simulation box. Can be provided here for constant boxes
                       or on-the-fly as kwarg 'box', e.g. for NPT ensemble

    Returns:
        A function that takes a simulation state with neighborlist and returns
        the instantaneous tcf.
    """
    (_, sigma, volume, x_bin_centers, y_bin_centers,
     z_bin_centers) = dataclasses.astuple(tcf_params)
    nbins = x_bin_centers.shape[1]

    # we use initial configuration to estimate the maximum number of non-zero
    # weights for speedup and reduced memory cost
    if nbrs_init is None:
        raise NotImplementedError('nbrs_init currently needs to be provided.')
    mask = _angle_neighbor_mask(nbrs_init)

    max_triplets = int(jnp.count_nonzero(mask > 1.e-6) * max_weight_multiplier)

    # ensure triplets can be batched
    rem = jnp.remainder(max_triplets, batch_size)
    max_triplets = max_triplets + (batch_size - rem)

    def gaussian_3d_bins(exp, inputs):
        triplet_distances, triplet_mask = inputs
        batch_exp = jnp.exp(util.f32(-0.5) * (
                (triplet_distances[:, 0, jnp.newaxis, jnp.newaxis, jnp.newaxis]
                 - x_bin_centers)**2 / sigma**2
                + (triplet_distances[:, 1, jnp.newaxis, jnp.newaxis,
                   jnp.newaxis] - y_bin_centers)**2 / sigma**2
                + (triplet_distances[:, 2, jnp.newaxis, jnp.newaxis,
                   jnp.newaxis] - z_bin_centers)**2 / sigma**2
        ))
        batch_exp *= triplet_mask[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        batch_exp = jnp.sum(batch_exp, axis=0)
        exp += batch_exp
        return exp, 0

    def triplet_corr_fun(r_kj, r_ij, r_ki, triplet_mask):
        """Returns instantaneous triplet correlation function while ensuring
        each particle pair contributes exactly 1.
        """
        dist_kj = space.distance(r_kj)
        dist_ij = space.distance(r_ij)
        dist_ki = space.distance(r_ki)

        histogram = jnp.zeros((nbins, nbins, nbins))
        distances = jnp.stack((dist_kj, dist_ij, dist_ki), axis=1)
        distances = jnp.reshape(distances, (-1, batch_size, 3))
        triplet_mask = jnp.reshape(triplet_mask, (-1, batch_size))

        # scan over per-batch computations for computational efficiency
        histogram = lax.scan(gaussian_3d_bins, histogram,
                             (distances, triplet_mask))[0]
        return histogram / volume / jnp.sqrt((2 * jnp.pi)**3)

    def tcf_fn(state, neighbor, **kwargs):
        """Returns TCF for a single snapshot. Allows changing the box
        on-the-fly via the 'box' kwarg.
        """
        dyn_displacement = partial(displacement_fn, **kwargs)  # box kwarg
        triplet_mask = _angle_neighbor_mask(neighbor)

        r_kj, r_ij = _triplet_pairwise_displacements(
            state.position, neighbor, dyn_displacement)

        box, _ = _dyn_box(reference_box, **kwargs)
        n_particles, spatial_dim = state.position.shape
        total_vol = quantity.volume(spatial_dim, box)
        particle_density = n_particles / total_vol

        # cap at max weights
        non_zero_weights = triplet_mask > 1.e-6
        _, sorting_idxs = lax.top_k(non_zero_weights[:, 0], max_triplets)
        triplet_mask = triplet_mask[sorting_idxs]
        r_ij = r_ij[sorting_idxs]
        r_kj = r_kj[sorting_idxs]
        num_non_zero = jnp.sum(non_zero_weights)
        del num_non_zero

        r_ki = r_kj - r_ij
        tcf = triplet_corr_fun(r_kj, r_ij, r_ki, triplet_mask)
        return tcf / n_particles / particle_density**2
    return tcf_fn


def _nearest_tetrahedral_nbrs(displacement_fn, position, nbrs):
    """Returning displacement vectors r_ij of 4 nearest neighbors to a
    central particle.
    """
    neighbor_displacement = space.map_neighbor(displacement_fn)
    n_particles, _ = nbrs.idx.shape
    neighbor_mask = nbrs.idx != n_particles
    r_neigh = position[nbrs.idx]
    # R_ij = R_i - R_j; i = central atom
    displacements = neighbor_displacement(position, r_neigh)
    distances = space.distance(displacements)
    jnp.where(neighbor_mask, distances, 1.e7)  # mask non-existing neighbors
    _, nearest_idxs = lax.top_k(-1 * distances, 4)  # 4 nearest neighbor indices
    nearest_displ = jnp.take_along_axis(displacements,
                                        jnp.expand_dims(nearest_idxs, -1),
                                        axis=1)
    return nearest_displ


def init_tetrahedral_order_parameter(displacement):
    """Initializes a function that computes the tetrahedral order parameter q
    for a single state.

    Args:
        displacement: Displacemnet function

    Returns:
        A function that takes a simulation state with neighborlist and returns
         the instantaneous q value.
    """
    def q_fn(state, neighbor, **kwargs):
        dyn_displacement = partial(displacement, **kwargs)
        nearest_dispacements = _nearest_tetrahedral_nbrs(dyn_displacement,
                                                         state.position,
                                                         neighbor)
        # Note: for loop will be unrolled by jit.
        #   Is there a more elegant vectorization over nearest neighbors?
        summed_angles = jnp.zeros(state.position.shape[0])
        for j in range(3):
            r_ij = nearest_dispacements[:, j]
            for k in range(j + 1, 4):
                r_ik = nearest_dispacements[:, k]
                # cosine of angle for all central particles in box
                psi_ijk = vmap(quantity.cosine_angle_between_two_vectors)(
                    r_ij, r_ik)
                summand = jnp.square(psi_ijk + (1. / 3.))
                summed_angles += summand

        average_angle = jnp.mean(summed_angles)
        q = 1 - (3. / 8.) * average_angle
        return q
    return q_fn


def velocity_autocorrelation(traj_state, num_lags):
    """Returns the velocity autocorrelation function (VACF).

    Args:
        traj_state: TrajectoryState object containing the simulation trajectory
        num_lags: Number of time lags to compute VACF values for. The time lag
        is implicitly defined by the dime difference between two adjacent
        states in the trajectory.

    Returns:
        An array containing the value of the VACF for each considered time lag.
    """
    # TODO this quadratic-scaling implementation of autocorrelation is not
    #  optimal. Long-term this should be using FFT if efficiency is critical
    @jit
    def vel_correlation(arr, laged_array):
        # assuming spatial dimension is in last axis
        scalar_product = jnp.sum(arr * laged_array, axis=-1)
        average = jnp.mean(scalar_product)  # over particles and time
        return average

    velocity_traj = traj_state.trajectory.velocity
    vacf = [vel_correlation(velocity_traj, velocity_traj)]
    vacf += [vel_correlation(velocity_traj[i:], velocity_traj[:-i])
             for i in range(1, num_lags)]
    return jnp.array(vacf)


def self_diffusion_green_kubo(traj_state, time_step, t_cut):
    """Green-Kubo formulation to compute self-diffusion D via the velocity
    autocorrelation function (VACF).

    D = \frac{1}{dim} Int_0^{t_cut} VACF(\tau) d\tau

    Args:
        traj_state: TrajectoryState containing a finely resolved trajectory.
        time_step: Time lag between 2 adjacent states in the trajetcory. The
                   simulation time step, in the usual case where every state is
                   retained.
        t_cut: Cut-off time: Biggest time-difference to consider in the VACF.

    Returns:
        A tuple (D, VACF). Estimate of self-diffusion D and VACF that can be
        used for additional post-processing / analysis.
    """
    num_lags = int(t_cut / time_step)
    vel_autocorr = velocity_autocorrelation(traj_state, num_lags)
    dim = traj_state.trajectory.velocity.shape[-1]
    diffusion = jnp.trapz(vel_autocorr, dx=time_step) / dim
    return diffusion, vel_autocorr


def init_bond_length(displacement_fn, bonds, average=False):
    """Initializes a function that computes bond lengths for given atom pairs.

    Args:
        displacement_fn: Displacement function
        bonds: (n, 2) array defining IDs of bonded particles
        average: If False, returns per-pair bond lengths.
                 If True, returns scalar average over all pairs

    Returns:
        A function that takes a simulation state and returns bond lengths
    """
    metric = vmap(space.canonicalize_displacement_or_metric(displacement_fn))

    def bond_length(state, **kwargs):
        dyn_metric = partial(metric, **kwargs)
        r1 = state.position[bonds[:, 0]]
        r2 = state.position[bonds[:, 1]]
        distances = dyn_metric(r1, r2)
        if average:
            return jnp.mean(distances)
        else:
            return distances
    return bond_length


def _bond_length(bonds, positions, displacement_fn):
    """Computes bond lengths for given atom position vector and bonds."""

    def pairwise_bond_length(bond, pos):
        bond_displacement = displacement_fn(pos[bond[0]], pos[bond[1]])
        bond_distance = space.distance(bond_displacement)
        return bond_distance

    batched_pair_boond_length = vmap(pairwise_bond_length, (0, None))

    if positions.ndim == 3:
        distances = vmap(batched_pair_boond_length, (None, 0))(bonds, positions)
        return distances
    elif positions.ndim == 2:
        distances = vmap(pairwise_bond_length, (0, None))(bonds, positions)
        return distances
    else:
        raise ValueError('Positions must be either of shape Ntimestep x Natoms'
                         ' x spatial_dim or N_atoms x spatial_dim')


def estimate_bond_constants(positions, bonds, displacement_fn):
    """Calculates the equlibrium harmonic bond constants from given positions.

    Can be used to estimate the bond constants from an atomistic
    simulation to be used as a coarse-grained prior.

    Args:
        positions: Position vector of size [Ntimestep x Natoms x spatial_dim]
                   or [N_atoms x spatial_dim]
        bonds: (n_bonds, 2) array defining IDs of bonded particles
        displacement_fn: Displacement function

    Returns:
        Tuple (eq_distances, eq_variances) of harmonic bond coefficients.
    """
    distances = _bond_length(bonds, positions, displacement_fn)
    eq_distances = jnp.mean(distances, axis=0)
    eq_variances = jnp.var(distances, axis=0)
    return eq_distances, eq_variances


def dihedral_displacement(positions, displacement_fn, dihedral_idxs,
                          degrees=True):
    """Computes the dihedral angle for all quadruple of atoms given in idxs.

    Args:
        positions: Positions of atoms in box
        displacement_fn: Displacement function
        dihedral_idxs: (n, 4) array defining IDs of quadruple particles
        degrees: If False, returns angles in rads.
                 If True, returns angles in degrees.

    Returns:
        An array (n,) of the dihedral angles.
    """
    p0 = positions[dihedral_idxs[:, 0]]
    p1 = positions[dihedral_idxs[:, 1]]
    p2 = positions[dihedral_idxs[:, 2]]
    p3 = positions[dihedral_idxs[:, 3]]

    b0 = -1. * vmap(displacement_fn)(p1, p0)
    b1 = vmap(displacement_fn)(p2, p1)
    b2 = vmap(displacement_fn)(p3, p2)

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= jnp.linalg.norm(b1, axis=1)[:, None]

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - jnp.sum(b0 * b1, axis=1)[:, None] * b1
    w = b2 - jnp.sum(b2 * b1, axis=1)[:, None] * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = jnp.sum(v * w, axis=1)
    cross = vmap(jnp.cross)(b1, v)
    y = jnp.sum(cross * w, axis=1)

    if degrees:
        return jnp.degrees(jnp.arctan2(y, x))
    else:
        return jnp.arctan2(y, x)


def kinetic_energy_tensor(state):
    """Computes the kinetic energy tensor of a single snapshot.

    Args:
        state: Jax_md simulation state

    Returns:
        Kinetic energy tensor
    """
    average_velocity = jnp.mean(state.velocity, axis=0)
    thermal_excitation_velocity = state.velocity - average_velocity
    diadic_velocity_product = vmap(lambda v: jnp.outer(v, v))
    velocity_tensors = diadic_velocity_product(thermal_excitation_velocity)
    return util.high_precision_sum(state.mass * velocity_tensors, axis=0)


def virial_potential_part(energy_fn, state, nbrs, box_tensor, **kwargs):
    """Interaction part of the virial pressure tensor for a single snaphot
    based on the formulation of Chen at al. (2020). See
    init_virial_stress_tensor. for details."""
    energy_fn_ = lambda pos, neighbor, box: energy_fn(
        pos, neighbor=neighbor, box=box, **kwargs)  # for grad
    position = state.position  # in unit box if fractional coordinates used
    negative_forces, box_gradient = grad(energy_fn_, argnums=[0, 2])(
        position, nbrs, box_tensor)
    position = space.transform(box_tensor, position)  # back to real positions
    force_contribution = jnp.dot(negative_forces.T, position)
    box_contribution = jnp.dot(box_gradient.T, box_tensor)
    return force_contribution + box_contribution


def init_virial_stress_tensor(energy_fn_template, ref_box_tensor=None,
                              include_kinetic=True, pressure_tensor=False):
    """Initializes a function that computes the virial stress tensor for a
    single state.

    This function is applicable to arbitrary many-body interactions, even
    under periodic boundary conditions. This implementation is based on the
    formulation of Chen et al. (2020), which is well-suited for vectorized,
    differentiable MD libararies. This function requires that `energy_fn`
    takes a `box` keyword argument, usually alongside `periodic_general`
    boundary conditions.

    Chen et al. "TensorAlloy: An automatic atomistic neural network program
    for alloys". Computer Physics Communications 250 (2020): 107057

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        ref_box_tensor: The transformation T of general periodic boundary
                        conditions. If None, box_tensor needs to be provided as
                        'box' during function call, e.g. for the NPT ensemble.
        include_kinetic: Whether kinetic part of stress tensor should be added.
        pressure_tensor: If False (default), returns the stress tensor. If True,
                         returns the pressure tensor, i.e. the negative stress
                         tensor.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        virial stress tensor.
    """
    if pressure_tensor:
        pressure_sign = -1.
    else:
        pressure_sign = 1.

    def virial_stress_tensor_neighborlist(state, neighbor, energy_params,
                                          **kwargs):
        # Note: this workaround with the energy_template was needed to keep
        #       the function jitable when changing energy_params on-the-fly
        # TODO function to transform box to box-tensor
        box, kwargs = _dyn_box(ref_box_tensor, **kwargs)
        energy_fn = energy_fn_template(energy_params)
        virial_tensor = virial_potential_part(energy_fn, state, neighbor, box,
                                              **kwargs)
        spatial_dim = state.position.shape[-1]
        volume = quantity.volume(spatial_dim, box)
        if include_kinetic:
            kinetic_tensor = -1 * kinetic_energy_tensor(state)
            return pressure_sign * (kinetic_tensor + virial_tensor) / volume
        else:
            return pressure_sign * virial_tensor / volume

    return virial_stress_tensor_neighborlist


def init_pressure(energy_fn_template, ref_box_tensor=None,
                  include_kinetic=True):
    """Initializes a function that computes the pressure for a single state.

    This function is applicable to arbitrary many-body interactions, even
    under periodic boundary conditions. See `init_virial_stress_tensor`
    for details.

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        ref_box_tensor: The transformation T of general periodic boundary
                        conditions. If None, box_tensor needs to be provided as
                        'box' during function call, e.g. for NPT ensemble.
        include_kinetic: Whether kinetic part of stress tensor should be added.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        pressure.
    """
    # pressure is negative hydrostatic stress
    stress_tensor_fn = init_virial_stress_tensor(
        energy_fn_template, ref_box_tensor, include_kinetic=include_kinetic,
        pressure_tensor=True
    )

    def pressure_neighborlist(state, neighbor, energy_params, **kwargs):
        pressure_tensor = stress_tensor_fn(state, neighbor, energy_params,
                                           **kwargs)
        return jnp.trace(pressure_tensor) / 3.
    return pressure_neighborlist


def energy_under_strain(epsilon, energy_fn, box_tensor, state, neighbor,
                        **kwargs):
    """Potential energy of a state after applying linear strain epsilon."""
    # Note: When computing the gradient, we deal with infinitesimally
    #       small strains. Linear strain theory is therefore valid and
    #       additionally tan(gamma) = gamma. These assumptions are used
    #       computing the box after applying the stain.
    strained_box = jnp.dot(box_tensor, jnp.eye(box_tensor.shape[0]) + epsilon)
    energy = energy_fn(state.position, neighbor=neighbor, box=strained_box,
                       **kwargs)
    return energy


def init_sigma_born(energy_fn_template, ref_box_tensor=None):
    """Initialiizes a function that computes the Born contribution to the
    stress tensor.

    sigma^B_ij = d U / d epsilon_ij

    Can also be computed to compute the stress tensor at kbT = 0, when called
    on the state of minimum energy. This function requires that `energy_fn`
    takes a `box` keyword argument, usually alongside `periodic_general`
    boundary conditions.

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        ref_box_tensor: The transformation T of general periodic boundary
                        conditions. If None, box_tensor needs to be provided as
                        'box' during function call, e.g. for the NPT ensemble.

    Returns:
        A function that takes a simulation state with neighbor list,
        energy_params and box (if applicable) and returns the instantaneous
        Born contribution to the stress tensor.
    """
    def sigma_born(state, neighbor, energy_params, **kwargs):
        box, kwargs = _dyn_box(ref_box_tensor, **kwargs)
        spatial_dim = box.shape[-1]
        volume = quantity.volume(spatial_dim, box)
        epsilon0 = jnp.zeros((spatial_dim, spatial_dim))

        energy_fn = energy_fn_template(energy_params)
        sigma_b = jacrev(energy_under_strain)(
            epsilon0, energy_fn, box, state, neighbor, **kwargs)
        return sigma_b / volume
    return sigma_born


def init_stiffness_tensor_stress_fluctuation(energy_fn_template, box_tensor,
                                             kbt, n_particles):
    """Initializes all functions necessary to compute the elastic stiffness
    tensor via the stress fluctuation method in the NVT ensemble.

    The provided functions compute all necessary instantaneous properties
    necessary to compute the elastic stiffness tensor via the stress fluctuation
     method. However, for compatibility with DiffTRe, (weighted) ensemble
     averages need to be computed manually and given to the stiffness_tensor_fn
     for final computation of the stiffness tensor. For an example usage see
     the diamond notebook. The implementation follows the formulation derived by
     Van Workum et al., "Isothermal stress and elasticity tensors for ions and
     point dipoles using Ewald summations", PHYSICAL REVIEW E 71, 061102 (2005).

     # TODO provide sample usage

    Args:
        energy_fn_template: A function that takes energy parameters as input
                            and returns an energy function
        box_tensor: The transformation T of general periodic boundary
                    conditions. As the stress-fluctuation method is only
                    applicable to the NVT ensemble, the box_tensor needs to be
                    provided here as a constant, not on-the-fly.
        kbt: Temperature in units of the Boltzmann constant
        n_particles: Number of particles in the box

    Returns: A tuple of 3 functions:
        born_term_fn: A function computing the Born contribution to the
                      stiffness tensor for a single snapshot
        sigma_born: A function computing the Born contribution to the stress
                    tensor for a single snapshot
        sigma_tensor_prod: A function computing sigma^B_ij * sigma^B_kl given
                           a trajectory of sigma^B_ij
        stiffness_tensor_fn: A function taking ensemble averages of C^B_ijkl,
                             sigma^B_ij and sigma^B_ij * sigma^B_kl and
                             returning the resulting stiffness tensor.
    """
    # TODO this function simplifies a lot if split between per-snapshot
    #  and per-trajectory functions
    spatial_dim = box_tensor.shape[-1]
    volume = quantity.volume(spatial_dim, box_tensor)
    epsilon0 = jnp.zeros((spatial_dim, spatial_dim))

    def born_term_fn(state, neighbor, energy_params, **kwargs):
        """Born contribution to the stiffness tensor:
        C^B_ijkl = d^2 U / d epsilon_ij d epsilon_kl
        """
        energy_fn = energy_fn_template(energy_params)
        born_stiffness_contribution = jacfwd(jacrev(energy_under_strain))(
            epsilon0, energy_fn, box_tensor, state, neighbor, **kwargs)
        return born_stiffness_contribution / volume

    @vmap
    def sigma_tensor_prod(sigma):
        """A function that computes sigma_ij * sigma_kl for a whole trajectory
        to be averaged afterwards.
        """
        return jnp.einsum('ij,kl->ijkl', sigma, sigma)

    def stiffness_tensor_fn(mean_born, mean_sigma, mean_sig_ij_sig_kl):
        """Computes the stiffness tensor given ensemble averages of
        C^B_ijkl, sigma^B_ij and sigma^B_ij * sigma^B_kl.
        """
        sigma_prod = jnp.einsum('ij,kl->ijkl', mean_sigma, mean_sigma)
        delta_ij = jnp.eye(spatial_dim)
        delta_ik_delta_jl = jnp.einsum('ik,jl->ijkl', delta_ij, delta_ij)
        delta_il_delta_jk = jnp.einsum('il,jk->ijkl', delta_ij, delta_ij)
        # Note: maybe use real kinetic energy of trajectory rather than target
        #       kbt?
        kinetic_term = n_particles * kbt / volume * (
                delta_ik_delta_jl + delta_il_delta_jk)
        delta_sigma = mean_sig_ij_sig_kl - sigma_prod
        return mean_born - volume / kbt * delta_sigma + kinetic_term

    sigma_born = init_sigma_born(energy_fn_template, box_tensor)

    return born_term_fn, sigma_born, sigma_tensor_prod, stiffness_tensor_fn


def stiffness_tensor_components_cubic_crystal(stiffness_tensor):
    """Computes the 3 independent elastic stiffness components of a cubic
    crystal from the whole stiffness tensor.

    The number of independent components in a general stiffness tensor is 21
    for isotropic pressure. For a cubic crystal, these 21 parameters only take
    3 distinct values: c11, c12 and c44. We compute these values from averages
    using all 21 components for variance reduction purposes.

    Args:
        stiffness_tensor: The full (3, 3, 3, 3) elastic stiffness tensor

    Returns:
        A (3,) ndarray containing (c11, c12, c44)
    """
    # TODO likely there exists a better formulation via Einstein notation
    c = stiffness_tensor
    c11 = (c[0, 0, 0, 0] + c[1, 1, 1, 1] + c[2, 2, 2, 2]) / 3.
    c12 = (c[0, 0, 1, 1] + c[1, 1, 0, 0] + c[0, 0, 2, 2] + c[2, 2, 0, 0]
           + c[1, 1, 2, 2] + c[2, 2, 1, 1]) / 6.
    c44 = (c[0, 1, 0, 1] + c[1, 0, 0, 1] + c[0, 1, 1, 0] + c[1, 0, 1, 0] +
           c[0, 2, 0, 2] + c[2, 0, 0, 2] + c[0, 2, 2, 0] + c[2, 0, 2, 0] +
           c[2, 1, 2, 1] + c[1, 2, 2, 1] + c[2, 1, 1, 2] + c[1, 2, 1, 2]) / 12.
    return jnp.array([c11, c12, c44])
