import jax.numpy as jnp
from jax import jit, grad, vmap, lax, jacrev, jacfwd, ops
from functools import partial
from jax_md import space, util, dataclasses, quantity
from chemtrain.jax_md_mod import custom_nn
from jax.scipy.stats.norm import cdf as normal_cdf

Array = util.Array


def energy_wrapper(energy_fn_template):
    """Wrapper around energy_fn to allow energy computation via
    traj_util.quantity_traj.
    """
    def energy(state, neighbor, energy_params, **kwargs):
        energy_fn = energy_fn_template(energy_params)
        return energy_fn(state.position, neighbor=neighbor, **kwargs)
    return energy


@dataclasses.dataclass
class RDFParams:
    """
    A struct containing hyperparameters to initialize a radial distribution (RDF) compute function.

    Attributes:
    reference_rdf: The target rdf; initialize with None if no target available
    rdf_bin_centers: The radial positions of the centers of the rdf bins
    rdf_bin_boundaries: The radial positions of the edges of the rdf bins
    sigma_RDF: Standard deviation of smoothing Gaussian
    """
    reference_rdf: Array
    rdf_bin_centers: Array
    rdf_bin_boundaries: Array
    sigma_RDF: Array


def rdf_discretization(RDF_cut, nbins=300, RDF_start=0.):
    """
    Computes dicretization parameters for initialization of RDF compute function.

    Args:
        RDF_cut: Cut-off length inside which pairs of particles are considered
        nbins: Number of bins in radial direction
        RDF_start: Minimal radial distance after which pairs of particles are considered

    Returns:
        Arrays with radial positions of bin centers, and bin edges and the standard
        deviation of the Gaussian smoothing kernel.

    """
    dx_bin = (RDF_cut - RDF_start) / float(nbins)
    rdf_bin_centers = jnp.linspace(RDF_start + dx_bin / 2., RDF_cut - dx_bin / 2., nbins)
    rdf_bin_boundaries = jnp.linspace(RDF_start, RDF_cut, nbins + 1)
    sigma_RDF = jnp.array(dx_bin)
    return rdf_bin_centers, rdf_bin_boundaries, sigma_RDF


@dataclasses.dataclass
class ADFParams:
    """
    A struct containing hyperparameters to initialize a angular distribution (ADF) compute function.

    Attributes:
    reference_adf: The target adf; initialize with None if no target available
    adf_bin_centers: The positions of the centers of the adf bins over theta
    sigma_ADF: Standard deviation of smoothing Gaussian
    r_outer: Outer radius beyond which particle triplets are not considered
    r_inner: Inner radius below which particle triplets are not considered
    """
    reference_adf: Array
    adf_bin_centers: Array
    sigma_ADF: Array
    r_outer: Array
    r_inner: Array


def adf_discretization(nbins=200):
    """
    Computes dicretization parameters for initialization of ADF compute function.

    Args:
        nbins_theta: Number of bins discretizing theta

    Returns:
        Arrays containing bin centers in theta direction and the standard
        deviation of the Gaussian smoothing kernel.
    """
    dtheta_bin = jnp.pi / float(nbins)
    adf_bin_centers = jnp.linspace(dtheta_bin / 2., jnp.pi - dtheta_bin / 2., nbins)
    sigma_ADF = util.f32(dtheta_bin)
    return adf_bin_centers, sigma_ADF


def initialize_radial_distribution_fun(box, displacement_fn, rdf_params):
    """
    Initializes a function that computes the radial distribution function (RDF) for a single state.
    
    Args:
        box: Simulation box
        displacement_fn: Displacement function
        rdf_params: RDFParams defining the hyperparameters of the RDF

    Returns:
        A function that takes a simulation state and returns the instantaneous rdf
    """
    _, rdf_bin_centers, rdf_bin_boundaries, sigma = dataclasses.astuple(rdf_params)
    distance_metric = space.canonicalize_displacement_or_metric(displacement_fn)
    bin_size = jnp.diff(rdf_bin_boundaries)

    def pair_corr_fun(R, **dynamic_kwargs):
        # computes instantaneous pair correlation function ensuring each particle pair contributes exactly 1
        n_particles = R.shape[0]
        metric = partial(distance_metric, **dynamic_kwargs)
        metric = space.map_product(metric)
        dr = metric(R, R)
        dr = jnp.where(dr > util.f32(1.e-7), dr, util.f32(1.e7))  # neglect same particles i.e. distance = 0.

        #  Gaussian distribution ensures that discrete integral over distribution is 1
        exp = jnp.exp(-util.f32(0.5) * (dr[:, :, jnp.newaxis] - rdf_bin_centers) ** 2 / sigma ** 2)  # Gaussian exponent
        gaussian_distances = exp * bin_size / jnp.sqrt(2 * jnp.pi * sigma ** 2)
        pair_corr_per_particle = util.high_precision_sum(gaussian_distances, axis=1)  # sum over all neighbors
        mean_pair_corr = util.high_precision_sum(pair_corr_per_particle, axis=0) / n_particles
        return mean_pair_corr

    def norming_factors(particle_density, bin_boundaries):
        # RDF is defined to relate the particle densities to an ideal gas:
        # This function computes densities that would correspond to an ideal gas
        r_small = bin_boundaries[:-1]
        r_large = bin_boundaries[1:]
        bin_volume = (4. / 3.) * jnp.pi * (jnp.power(r_large, 3) - jnp.power(r_small, 3))
        bin_weights = bin_volume * particle_density
        return bin_weights

    def rdf_compute_fun(state, **unused_kwargs):
        # Note: we cannot use neighborlist as RDF cutoff and neighborlist cut-off don't coincide in general
        R = state.position
        n_particles, spatial_dim = R.shape
        total_vol = quantity.volume(spatial_dim, box)
        particle_density = n_particles / total_vol
        mean_pair_corr = pair_corr_fun(R)
        rdf = mean_pair_corr / norming_factors(particle_density, rdf_bin_boundaries)
        return rdf
    return rdf_compute_fun


def vectorized_angle_fn(R_ij, R_kj):
    return vmap(angle)(R_ij, R_kj)


def angle(R_ij, R_kj):
    """
    Computes the angle (kj, ij) from vectors R_kj and R_ij, correctly selecting the quadrant.

    Based on tan(theta) = |(R_ji x R_kj)| / (R_ji . R_kj). Beware non-differentability of arctan2(0,0).

    Args:
        R_ij: Vector pointing to i from j
        R_kj: Vector pointing to k from j

    Returns:
        Angle between vectors

    """
    cross = jnp.linalg.norm(jnp.cross(R_ij, R_kj))
    dot = jnp.dot(R_ij, R_kj)
    theta = jnp.arctan2(cross, dot)
    return theta


def initialize_angle_distribution_neighborlist(displacement_fn, adf_params, smoothing_dr=0.01,
                                               R_init=None, nbrs_init=None, max_weights_multiplier=2.):
    """
    Initializes a function that computes the angular distribution function (ADF) for a single state.

    Angles are smoothed in radial direction via a Gaussian kernel (compare RDF function). In radial
    direction, triplets are weighted according to a Gaussian cumulative distribution function, such that
    triplets with both radii inside the cut-off band are weighted approximately 1 and the weights of
    triplets towards the band edges are soomthly reduced to 0.
    For computational speed-up and reduced memory needs, R_init and nbrs_init can be provided
    to estmate the maximum number of triplets - similarly to the maximum capacity of neighbors
    in the neighbor list. Caution: currrently the user does not receive information if overflow occured.
    This function assumes that r_outer is smaller than the neighborlist cut-off. If this is not the case,
    a function computing all pairwise distances is necessary.

    Args:
        displacement_fn: Displacement function
        adf_params: ADFParams defining the hyperparameters of the RDF
        smoothing_dr: Standard deviation of Gaussian smoothing in radial direction
        R_init: Initial position to estimate maximum number of triplets
        nbrs_init: Initial neighborlist to estimate maximum number of triplets
        max_weights_multiplier: Multiplier for estimate of number of triplets

    Returns:
        A function that takes a simulation state with neighborlist and returns the instantaneous adf
    """

    _, adf_bin_centers, sigma_theta, r_outer, r_inner = dataclasses.astuple(adf_params)
    neighbor_displacement_fn = space.map_neighbor(displacement_fn)
    sigma_theta = util.f32(sigma_theta)
    adf_bin_centers = util.f32(adf_bin_centers)

    def adf_cutoff(r_outer, r_inner, smoothing_dr):
        """
        Smoothly constraints triplets to a radial band such that both distances are between
        r_inner and r_outer. The Gaussian cdf is used for smoothing. The smoothing width can
        be controlled by the gaussian standard deviation.
        """
        def smooth_cutoff(r_small, r_large):
            outer_weight = 1. - normal_cdf(r_large, loc=r_outer, scale=smoothing_dr**2)
            inner_weight = normal_cdf(r_small, loc=r_inner, scale=smoothing_dr**2)
            return outer_weight * inner_weight
        return smooth_cutoff

    smooth_cutoff_fn = adf_cutoff(r_outer, r_inner, smoothing_dr)

    def angle_neighbor_mask(neighbor):
        """
        Mask the cases of non-existing neighbors and both neighbors being the same particle.
        The cases j=k and j=i is already excluded by the neighborlist construction.
        """
        N, max_neighbors = neighbor.idx.shape
        edge_idx_flat = jnp.ravel(neighbor.idx)
        idx_k = jnp.repeat(edge_idx_flat, max_neighbors)
        idx_i = jnp.tile(neighbor.idx, (1, max_neighbors)).ravel()
        neighbor_mask = neighbor.idx != N
        neighbor_mask_flat = jnp.ravel(neighbor_mask)
        mask_k = jnp.repeat(neighbor_mask_flat, max_neighbors)
        mask_i = jnp.tile(neighbor_mask, (1, max_neighbors)).ravel()
        mask_i_eq_k = idx_i != idx_k  # this mask structure is known a priori, would be more efficient to precompute
        mask = mask_k * mask_i * mask_i_eq_k
        mask = jnp.expand_dims(mask, axis=-1)
        return mask

    def pairwise_distances_and_displacements(R, neighbor):
        """Build distance matrix and pairs of displacement vectors."""
        N, max_neighbors = neighbor.idx.shape
        R_neigh = R[neighbor.idx]
        neighbor_displacement = neighbor_displacement_fn(R, R_neigh)
        neighbor_distances = space.distance(neighbor_displacement)
        neighbor_displacement_flat = jnp.reshape(neighbor_displacement, (N * max_neighbors, 3))
        R_kj = jnp.repeat(neighbor_displacement_flat, max_neighbors, axis=0)
        R_ij = jnp.tile(neighbor_displacement, (1, max_neighbors, 1)).reshape([N * max_neighbors ** 2, 3])
        return neighbor_distances, R_kj, R_ij

    def cut_off_weights(pair_distances, max_neighbors):
        """Build differentiable cut-off weight vector using Gaussian cdf cut-off function."""
        distance_kj = jnp.repeat(jnp.ravel(pair_distances), max_neighbors, axis=0)
        distance_ij = jnp.tile(pair_distances, (1, max_neighbors)).ravel()
        pair_distances = jnp.column_stack((distance_kj, distance_ij)).sort(axis=1)  # get r_small and r_large
        weights = smooth_cutoff_fn(pair_distances[:, 0], pair_distances[:, 1])
        weights = jnp.expand_dims(weights, axis=-1)
        return weights

    def weighted_adf(angles, weights):
        """Compute weighted adf contribution of each triplet."""
        exponent = jnp.exp(-util.f32(0.5) * (angles[:, jnp.newaxis] - adf_bin_centers) ** 2 / sigma_theta ** 2)
        gaussians = exponent / jnp.sqrt(2 * jnp.pi * sigma_theta ** 2)
        gaussians *= weights
        unnormed_adf = util.high_precision_sum(gaussians, axis=0)
        adf = unnormed_adf / jnp.trapz(unnormed_adf, adf_bin_centers)
        return adf

    # we use initial configuration to estimate the maximum number of non-zero weights for speedup and reduced memory
    if R_init is not None:
        assert nbrs_init is not None, \
            "If we estimate the maximum number of triplets, the initial neighborlist is nneded."
        N, max_neighbors = nbrs_init.idx.shape
        mask = angle_neighbor_mask(nbrs_init)
        neighbor_distances, R_kj, R_ij = pairwise_distances_and_displacements(R_init, nbrs_init)
        weights = cut_off_weights(neighbor_distances, max_neighbors)
        weights *= mask  # combine radial cut-off with neighborlist mask
        max_weights = jnp.count_nonzero(weights > 1.e-6) * max_weights_multiplier

    def adf_fn(state, neighbor, **unused_kwargs):
        R = state.position
        N, max_neighbors = neighbor.idx.shape
        mask = angle_neighbor_mask(neighbor)
        neighbor_distances, R_kj, R_ij = pairwise_distances_and_displacements(R, neighbor)
        weights = cut_off_weights(neighbor_distances, max_neighbors)
        weights *= mask  # combine radial cut-off with neighborlist mask

        if R_init is not None:  # prune triplets to reduce effort in computation of angles and Gaussians
            non_zero_weights = weights > 1.e-6
            _, sorting_idxs = lax.top_k(non_zero_weights[:, 0], max_weights)
            weights = weights[sorting_idxs]
            R_ij = R_ij[sorting_idxs]
            R_kj = R_kj[sorting_idxs]
            mask = mask[sorting_idxs]
            num_non_zero = jnp.sum(non_zero_weights)
            # TODO check if num_non_zero > max_weights, if this is the case, send an error message
            #  to the user to increase max_weights_multiplier

        R_ij_safe, R_kj_safe = custom_nn.safe_angle_mask(R_ij, R_kj, mask)  # ensure differentiability
        angles = vectorized_angle_fn(R_ij_safe, R_kj_safe)
        adf = weighted_adf(angles, weights)
        return adf
    return adf_fn


def init_tetrahedral_order_parameter(displacement):
    """
    Initializes a function that computes the tetrahedral order parameter q for a single state.

    Args:
        displacement: Displacemnet function

    Returns:
        A function that takes a simulation state with neighborlist and returns the instantaneous q value
    """
    neighbor_displacement = space.map_neighbor(displacement)

    def nearest_neighbors(R, neighbor):
        """A function returning displacement vectors R_ij of 4 nearest neighbors to a central particle."""
        N, max_neighbors = neighbor.idx.shape
        neighbor_mask = neighbor.idx != N
        R_neigh = R[neighbor.idx]
        displacements = neighbor_displacement(R, R_neigh)  # R_ij = R_i - R_j; i = central atom
        distances = space.distance(displacements)
        jnp.where(neighbor_mask, distances, 1.e7)  # mask non-existing neighbors
        _, neighbor_idxs = lax.top_k(-distances, 4)  # 4 nearest neighbor indexes
        # nearest_neighbor_displacements = displacements[neighbor_idxs]
        nearest_neighbor_displacements = jnp.take_along_axis(displacements, jnp.expand_dims(neighbor_idxs, -1), axis=1)
        return nearest_neighbor_displacements

    def q_fn(state, neighbor, **unused_kwargs):
        R = state.position
        nearest_dispacements = nearest_neighbors(R, neighbor)

        # Note: for loop will be unrolled by jit; is there a more elegant version vectorizing over nearest neighbors?
        summed_angles = jnp.zeros(R.shape[0])
        for j in range(3):
            R_ij = nearest_dispacements[:, j]
            for k in range(j + 1, 4):
                R_ik = nearest_dispacements[:, k]
                # cosine of angle for all central particles in box
                psi_ijk = vmap(quantity.angle_between_two_vectors)(R_ij, R_ik)
                summand = jnp.square(psi_ijk + (1. / 3.))
                summed_angles += summand

        average_angle = jnp.mean(summed_angles)
        q = 1 - (3. / 8.) * average_angle
        return q

    return q_fn


def mean_squared_displacements(trajectory, num_lags, subsample=1):
    """
    Computes Mean Squared Displacement (MSD) values from a given trajectory.

    Note that the trajectory needs to be "unwrapped" such that periodic BCs
    do not artificially constrain diffusion. The series of time lags is defined
    implicitly by Delta lag = subsample * time per printout.

    Args:
        trajectory: Trajectory of states
        num_lags: Number of time lags to compute; determines the maximum lag
        subsample: Only consider every subsample state of the trajectory

    Returns:
        An array of MSD values for different time lags averaged over all particles in the box
    """
    # trajectory needs to be unwrapped
    positions = trajectory.position[::subsample]  # only use every subsample snapshot
    assert positions.shape[0] > num_lags, "The trajectory is not long enough for the specified number of " \
                                          "lags: " + str(positions.shape[0]) + " positions vs " + str(num_lags) + " lags."
    non_periodic_displacement = vmap(vmap(space.pairwise_displacement))
    lags = jnp.arange(1, num_lags + 1)

    @jit
    def msd_fn(R1, R2):
        displacements = non_periodic_displacement(R1, R2)
        squared_distances = space.square_distance(displacements)
        msd = jnp.mean(squared_distances)
        return msd

    msds = []
    for lag in lags:
        R1 = positions[:-lag]
        R2 = positions[lag:]
        msd = msd_fn(R1, R2)
        msds.append(msd)

    return jnp.array(msds)


def self_diffusion(msd_values, dt, dim, start_idx=0, end_idx=None):
    """
    Computes the self-diffusion coefficient D by linear regression on MSD values.

    start_idx and end_idx are useful to carve out the most representative part of
    MSD values to consider during regression. Most commonly, the error in MSD values
    is smallest in some "middle" range of lags.

    Args:
        msd_values: Mean Squared Displacement values for linearly increasing time lags
        dt: Difference in time lag between adjacent MSD values
        dim: Dimension of the simulation
        start_idx: Starting index for which MSD values are considered in the regression
        end_idx: End index for which MSD values are considered in the regression

    Returns:
        Self-diffusion coefficient
    """
    N_max = msd_values.shape[0]
    time_lags = jnp.arange(1, N_max + 1) * dt
    considered_msds = msd_values[start_idx:end_idx]
    considered_lags = time_lags[start_idx:end_idx]
    assert len(considered_lags) > 0, "No MSD values left for regression. Check start and end inices."

    # solve MSD(i * dt) = a + (i * dt) b
    system_matrix = jnp.ones((considered_lags.size, 2))  # intercept
    system_matrix = ops.index_update(system_matrix, ops.index[:, 1], considered_lags)
    solution, residuals, _, _ = jnp.linalg.lstsq(system_matrix, considered_msds)
    b = solution[1]

    D = b / (2 * dim)
    return D

def init_bond_length(displacement_fn, bonds, average=False):
    """
    Initializes a function that computes bond lengths for given atom pairs.

    Args:
        displacement_fn: Displacement function
        bonds: (n, 2) array defining IDs of bonded particles
        average: If False, returns per-pair bond lengths. If True, returns scalar average over all pairs

    Returns:
        A function that takes a simulation state and returns bond lengths

    """
    metric = vmap(space.canonicalize_displacement_or_metric(displacement_fn))
    def bond_length(state, **unused_kwargs):
        R = state.position
        R1 = R[bonds[:, 0]]
        R2 = R[bonds[:, 1]]
        distances = metric(R1, R2)
        if average:
            return jnp.mean(distances)
        else:
            return distances
    return bond_length


def kinetic_energy_tensor(mass, velocity):
    """Computes the kinetic energy componentwise, most commonly used for computing virial stress."""
    average_velocity = jnp.mean(velocity, axis=0)
    thermal_excitation_velocity = velocity - average_velocity
    diadic_velocity_product = vmap(lambda v: jnp.outer(v, v))
    velocity_tensors = diadic_velocity_product(thermal_excitation_velocity)
    return - util.high_precision_sum(mass * velocity_tensors, axis=0)


def init_virial_stress_tensor(energy_fn_template, box_tensor=None, include_kinetic=True):
    """
    Initializes a function that computes the virial stress tensor for a single state.

    This function is applicable to arbitrary many-body interactions, even under periodic
    boundary conditions. This implementation is based on the formulation of Chen et al. (2020),
    which is well-suited for vectorized, differentiable MD libararies.
    This function requires that `energy_fn` takes a `box` keyword argument,
    usually alongside `periodic_general` boundary conditions.
    Chen et al. "TensorAlloy: An automatic atomistic neural network program for alloys."
    Computer Physics Communications 250 (2020): 107057

    Args:
        energy_fn_template: A function that takes energy parameters as input and returns an energy function
        box_tensor: The transformation T of general periodic boundary
                    conditions. If None, box_tensor needs to be provided as
                    'box' during function call, e.g. for NPT ensemble.
        include_kinetic: Whether kinetic part of stress tensor should be included.
        fixed_box: If True: always uses same box tensor. If False: box_tensor can be provided on-the-fly

    Returns:
        A function that takes a simulation state with neighborlist and energy_params and
        returns the instantaneous virial stress tensor
    """

    def virial_potential_part(energy_fn, state, nbrs, box_tensor, **kwargs):
        energy_fn_without_kwargs = lambda R, nbrs, box_tensor: energy_fn(R, neighbor=nbrs, box=box_tensor, **kwargs)  # for grad
        R = state.position  # in unit box if fractional coordinaes used
        negative_forces, box_gradient = grad(energy_fn_without_kwargs, argnums=[0, 2])(R, nbrs, box_tensor)
        R = space.transform(box_tensor, R)  # transform back to real positions
        force_contribution = jnp.dot(negative_forces.T, R)
        box_contribution = jnp.dot(box_gradient.T, box_tensor)
        return force_contribution + box_contribution

    def virial_stress_tensor_neighborlist(state, neighbor, energy_params, box, **kwargs):
        # Note: this workaround with the energy_template was needed to keep the function jitable
        #       when changing energy_params on-the-fly
        # TODO function to transform box to box-tensor
        energy_fn = energy_fn_template(energy_params)
        virial_tensor = virial_potential_part(energy_fn, state, neighbor, box, **kwargs)
        spatial_dim = state.position.shape[-1]
        volume = quantity.volume(spatial_dim, box)
        if include_kinetic:
            kinetic_tensor = kinetic_energy_tensor(state.mass, state.velocity)
            return (kinetic_tensor + virial_tensor) / volume
        else:
            return virial_tensor / volume

    if box_tensor is None:
        return virial_stress_tensor_neighborlist
    else:
        return partial(virial_stress_tensor_neighborlist, box=box_tensor)


def init_pressure(energy_fn_template, box_tensor=None, include_kinetic=True):
    """
    Initializes a function that computes the pressure for a single state.

    This function is applicable to arbitrary many-body interactions, even under periodic
    boundary conditions. See `init_virial_stress_tensor` for details.

    Args:
        energy_fn_template: A function that takes energy parameters as input and returns an energy function
        box_tensor: The transformation T of general periodic boundary
                    conditions. If None, box_tensor needs to be provided as
                    'box' during function call, e.g. for NPT ensemble.
        include_kinetic: Whether kinetic part of stress tensor should be included.

    Returns:
        A function that takes a simulation state with neighborlist and energy_params and
        returns the instantaneous pressure
    """
    stress_tensor_fn = init_virial_stress_tensor(energy_fn_template,
                                                 box_tensor,
                                                 include_kinetic=include_kinetic)

    def pressure_neighborlist(state, neighbor, energy_params, **kwargs):
        stress_tensor = stress_tensor_fn(state, neighbor, energy_params,
                                         **kwargs)
        # pressure is negative hydrostatic stress
        return - jnp.trace(stress_tensor) / 3.
    return pressure_neighborlist


def init_stiffness_tensor_stress_fluctuation(energy_fn_template, box_tensor, kbT, n_particles):
    """
    Initializes all functions necessary to compute the elastic stiffness tensor via the stress fluctuation method.

    The provided functions compute all necessary instantaneous properties necessary to compute the elastic
    stiffness tensor via the stress fluctuation method. However, for compatibility with DiffTRe, (weighted)
    ensemble averages need to be computed manually and given to the stiffness_tensor_fn for final computation
    of the stiffness tensor. For an example usage see the diamond notebook. The implementation follows the
    formulation derived by Van Workum et al., "Isothermal stress and elasticity tensors for ions and point
    dipoles using Ewald summations", PHYSICAL REVIEW E 71, 061102 (2005).

    Args:
        energy_fn_template: A function that takes energy parameters as input and returns an energy function
        box_tensor: The transformation T of general periodic boundary conditions
        kbT: Temperature in units of the Boltzmann constant
        n_particles: Number of particles in the box

    Returns:
        born_term_fn: A function computing the Born contribution to the stiffness tensor for a single snapshot
        sigma_born: A function computing the Born contribution to the stress tensor for a single snapshot
        sigma_tensor_prod: A function computing sigma^B_ij * sigma^B_kl given a trajectory of sigma^B_ij
        stiffness_tensor_fn: A function taking ensemble averages of C^B_ijkl, sigma^B_ij and sigma^B_ij * sigma^B_kl
                             and returning the resulting stiffness tensor.
    """
    spatial_dim = box_tensor.shape[-1]
    volume = quantity.volume(spatial_dim, box_tensor)
    epsilon0 = jnp.zeros((spatial_dim, spatial_dim))

    def energy_under_strain(epsilon, energy_fn, state, neighbor):
        # Note: When computing the gradient, we deal with infinitesimally small strains.
        #       Linear strain theory is therefore valid and additionally tan(gamma) = gamma.
        #       These assumptions are used computing the box after applying the stain.
        strained_box = jnp.dot(box_tensor, jnp.eye(box_tensor.shape[0]) + epsilon)
        energy = energy_fn(state.position, neighbor=neighbor, box=strained_box)
        return energy

    def born_term_fn(state, neighbor, energy_params, **unused_kwargs):
        """Born contribution to the stiffness tensor: C^B_ijkl = d^2 U / d epsilon_ij d epsilon_kl"""
        energy_fn = energy_fn_template(energy_params)
        born_stiffness_contribution = jacfwd(jacrev(energy_under_strain))(epsilon0, energy_fn, state, neighbor)
        return born_stiffness_contribution / volume

    def sigma_born(state, neighbor, energy_params, **unused_kwargs):
        """Born contribution to the stress tensor: sigma^B_ij = d U / d epsilon_ij"""
        energy_fn = energy_fn_template(energy_params)
        sigma_born = jacrev(energy_under_strain)(epsilon0, energy_fn, state, neighbor)
        return sigma_born / volume

    @vmap
    def sigma_tensor_prod(sigma):
        """A function that computes sigma_ij * sigma_kl for a whole trajectory to be averaged afterwards."""
        return jnp.einsum('ij,kl->ijkl', sigma, sigma)

    def stiffness_tensor_fn(born_term_average, sigma_average, sigma_ij_sigma_kl_average):
        """Computes the stiffness tensor given ensemble averages of C^B_ijkl, sigma^B_ij and sigma^B_ij * sigma^B_kl"""
        sigma_prod = jnp.einsum('ij,kl->ijkl', sigma_average, sigma_average)
        delta_ik_delta_jl = jnp.einsum('ik,jl->ijkl', jnp.eye(spatial_dim), jnp.eye(spatial_dim))
        delta_il_delta_jk = jnp.einsum('il,jk->ijkl', jnp.eye(spatial_dim), jnp.eye(spatial_dim))
        kinetic_term = n_particles * kbT / volume * (delta_ik_delta_jl + delta_il_delta_jk)
        delta_sigma = sigma_ij_sigma_kl_average - sigma_prod
        return born_term_average - volume / kbT * delta_sigma + kinetic_term

    return born_term_fn, sigma_born, sigma_tensor_prod, stiffness_tensor_fn


def stiffness_tensor_components_cubic_crystal(stiffness_tensor):
    """
    Computes the 3 independent elastic stiffness components of a cubic crystal from the whole stiffness tensor.

    The number of independent components in a general stiffness tensor is 21 for isotropic pressure.
    For a cubic crystal, these 21 parameters only take 3 distinct values: c11, c12 and c44.
    We compute these values from averages using all 21 components for variance reduction purposes.

    Args:
        stiffness_tensor: The full (3, 3, 3, 3) elastic stiffness tensor

    Returns:
        A (3,) ndarray containing (c11, c12, c44)

    """
    C = stiffness_tensor
    c11 = (C[0, 0, 0, 0] + C[1, 1, 1, 1] + C[2, 2, 2, 2]) / 3.
    c12 = (C[0, 0, 1, 1] + C[1, 1, 0, 0] + C[0, 0, 2, 2] + C[2, 2, 0, 0] + C[1, 1, 2, 2] + C[2, 2, 1, 1]) / 6.
    c44 = (C[0, 1, 0, 1] + C[1, 0, 0, 1] + C[0, 1, 1, 0] + C[1, 0, 1, 0] +
           C[0, 2, 0, 2] + C[2, 0, 0, 2] + C[0, 2, 2, 0] + C[2, 0, 2, 0] +
           C[2, 1, 2, 1] + C[1, 2, 2, 1] + C[2, 1, 1, 2] + C[1, 2, 1, 2]) / 12.
    return jnp.array([c11, c12, c44])
