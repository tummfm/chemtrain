import jax
from jax import lax, vmap
import jax.numpy as jnp
import haiku as hk
from jax_md import space, partition, nn, util, energy, smap
from jax_md.energy import multiplicative_isotropic_cutoff, _sw_radial_interaction, _sw_angle_interaction
from chemtrain.jax_md_mod import custom_interpolate
from chemtrain.jax_md_mod import custom_nn
from functools import partial

from typing import Callable, Tuple, Dict, Any
# Types
f32 = util.f32
f64 = util.f64
Array = util.Array

PyTree = Any
Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList


def stillinger_weber_energy(dr,
                            dR,
                            mask=None,
                            A=7.049556277,
                            B=0.6022245584,
                            p=4,
                            lam=21.0,
                            epsilon=2.16826,
                            gamma=1.2,
                            sigma=2.0951,
                            cutoff=1.8*2.0951,
                            three_body_strength=1.0):
    """
    Stillinger-Weber (SW) potential [1] which is commonly used to model
    silicon and similar systems. This function uses the default SW parameters
    from the original paper. The SW potential was originally proposed to
    model diamond in the diamond crystal phase and the liquid phase, and is
    known to give unphysical amorphous configurations [2, 3]. For this reason,
    we provide a three_body_strength parameter. Changing this number to 1.5
    or 2.0 has been know to produce more physical amorphous phase, preventing
    most atoms from having more than four nearest neighbors. Note that this
    function currently assumes nearest-image-convention.

    [1] Stillinger, Frank H., and Thomas A. Weber. "Computer simulation of
    local order in condensed phases of silicon." Physical review B 31.8
    (1985): 5262.
    [2] Holender, J. M., and G. J. Morgan. "Generation of a large structure
    (105 atoms) of amorphous Si using molecular dynamics." Journal of
    Physics: Condensed Matter 3.38 (1991): 7241.
    [3] Barkema, G. T., and Normand Mousseau. "Event-based relaxation of
    continuous disordered systems." Physical review letters 77.21 (1996): 4358.

    Args:
        dr: A ndarray of pairwise distances between particles
        dR: An ndarray of pairwise displacements between particles
        A: A scalar that determines the scale of two-body term
        B: Factor for radial power term
        p: Power in radial interaction
        lam: A scalar that determines the scale of the three-body term
        epsilon: A scalar that sets the energy scale
        gamma: Exponential scale in three-body term
        sigma: A scalar that sets the length scale
        cutoff: Cut-off value defined as sigma * a
        three_body_strength: A scalar that determines the relative strength
                             of the angular interaction
        mask: ndarray of size dr masking non-existing neighbors in neighborlist (if applicable)
    Returns:
        The Stilinger-Weber energy for a snapshot.
    """

    # initialize
    if mask is None:
        N = dr.shape[0]
        mask = jnp.ones([N, N])
        angle_mask = jnp.ones([N, N, N])
    else:  # for neighborlist input
        max_neighbors = mask.shape[-1]
        angle_mask1 = jnp.tile(jnp.expand_dims(mask, 1), [1, max_neighbors, 1])  # first pair
        angle_mask2 = jnp.tile(jnp.expand_dims(mask, -1), [1, 1, max_neighbors])  # second pair
        angle_mask = angle_mask1 * angle_mask2
    sw_radial_interaction = partial(_sw_radial_interaction, sigma=sigma, p=p, B=B, cutoff=cutoff)
    sw_angle_interaction = partial(_sw_angle_interaction, gamma=gamma, sigma=sigma, cutoff=cutoff)
    sw_three_body_term = vmap(vmap(vmap(sw_angle_interaction, (0, None)), (None, 0)), 0)

    # compute SW energy
    radial_interactions = sw_radial_interaction(dr) * mask
    angular_interactions = sw_three_body_term(dR, dR) * angle_mask
    first_term = A * jnp.sum(radial_interactions) / 2.0
    second_term = lam * jnp.sum(angular_interactions) / 2.0
    return epsilon * (first_term + three_body_strength * second_term)


def stillinger_weber_pair(displacement,
                          A=7.049556277,
                          B=0.6022245584,
                          p=4,
                          lam=21.0,
                          epsilon=2.16826,
                          gamma=1.2,
                          sigma=2.0951,
                          cutoff=1.8*2.0951,
                          three_body_strength=1.0):
    """Convenience wrapper to compute stilinger-weber energy over a system with variable parameters."""

    def compute_fn(R, **dynamic_kwargs):
        d = partial(displacement, **dynamic_kwargs)
        dR = space.map_product(d)(R, R)  # N x N x3 displacement matrix
        dr = space.distance(dR)  # N x N distances
        return stillinger_weber_energy(dr, dR, None, A, B, p, lam, epsilon, gamma, sigma, cutoff, three_body_strength)

    return compute_fn


def stillinger_weber_neighborlist(displacement,
                                  box_size=None,
                                  A=7.049556277,
                                  B=0.6022245584,
                                  p=4,
                                  lam=21.0,
                                  epsilon=2.16826,
                                  gamma=1.2,
                                  sigma=2.0951,
                                  cutoff=1.8*2.0951,
                                  three_body_strength=1.0,
                                  dr_threshold=0.1,
                                  capacity_multiplier=1.25,
                                  initialize_neighbor_list=True):
    """Convenience wrapper to compute stilinger-weber energy using a neighbor list."""

    def energy_fn(R, neighbor, **dynamic_kwargs):
        d = partial(displacement, **dynamic_kwargs)
        N = R.shape[0]
        mask = neighbor.idx != N
        R_neigh = R[neighbor.idx]
        dR = space.map_neighbor(d)(R, R_neigh)
        dr = space.distance(dR)
        return stillinger_weber_energy(dr, dR, mask, A, B, p, lam, epsilon, gamma, sigma, cutoff, three_body_strength)

    if initialize_neighbor_list:
        assert box_size is not None
        neighbor_fn = partition.neighbor_list(displacement, box_size, cutoff, dr_threshold,
                                              capacity_multiplier=capacity_multiplier)
        return neighbor_fn, energy_fn

    return energy_fn


def generic_repulsion(dr: Array,
                      sigma: Array=1.,
                      epsilon: Array=1.,
                      exp: Array=12.,
                      **dynamic_kwargs) -> Array:
    """
    Repulsive interaction between soft sphere particles: U = epsilon * (sigma / r)**exp.

    Args:
      dr: An ndarray of pairwise distances between particles.
      sigma: Repulsion length scale
      epsilon: Interaction energy scale
      exp: Exponent specifying interaction stiffness

    Returns:
      Array of energies
    """

    dr = jnp.where(dr > 1.e-7, dr, 1.e7)  # save masks dividing by 0
    idr = (sigma / dr)
    U = epsilon * idr ** exp
    return U


def generic_repulsion_pair(displacement_or_metric: DisplacementOrMetricFn,
                     species: Array=None,
                     sigma: Array=1.0,
                     epsilon: Array=1.0,
                     exp: Array=12.,
                     r_onset: Array = 2.0,
                     r_cutoff: Array = 2.5,
                     per_particle: bool=False):
    """Convenience wrapper to compute generic repulsion energy over a system."""
    sigma = jnp.array(sigma, dtype=f32)
    epsilon = jnp.array(epsilon, dtype=f32)
    exp = jnp.array(exp, dtype=f32)
    r_onset = jnp.array(r_onset, dtype=f32)
    r_cutoff = jnp.array(r_cutoff, dtype=f32)

    return smap.pair(
        multiplicative_isotropic_cutoff(generic_repulsion, r_onset, r_cutoff),
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        species=species,
        sigma=sigma,
        epsilon=epsilon,
        exp=exp,
        reduce_axis=(1,) if per_particle else None)


def generic_repulsion_neighborlist(displacement_or_metric: DisplacementOrMetricFn,
                     box_size: Box=None,
                     species: Array=None,
                     sigma: Array=1.0,
                     epsilon: Array=1.0,
                     exp: Array=12.,
                     r_onset: Array = 0.9,
                     r_cutoff: Array = 1.,
                     dr_threshold: float=0.2,
                     per_particle: bool=False,
                     capacity_multiplier: float=1.25,
                     initialize_neighbor_list: bool=True):
    """
    Convenience wrapper to compute generic repulsion energy over a system with neighborlist.

    Provides option not to initialize neighborlist. This is useful if energy function needs
    to be initialized within a jitted function.
    """

    sigma = jnp.array(sigma, dtype=f32)
    epsilon = jnp.array(epsilon, dtype=f32)
    exp = jnp.array(exp, dtype=f32)
    r_onset = jnp.array(r_onset, dtype=f32)
    r_cutoff = jnp.array(r_cutoff, dtype=f32)

    energy_fn = smap.pair_neighbor_list(
      multiplicative_isotropic_cutoff(generic_repulsion, r_onset, r_cutoff),
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      sigma=sigma,
      epsilon=epsilon,
      exp=exp,
      reduce_axis=(1,) if per_particle else None)

    if initialize_neighbor_list:
        assert box_size is not None
        neighbor_fn = partition.neighbor_list(displacement_or_metric, box_size, r_cutoff, dr_threshold,
                                              capacity_multiplier=capacity_multiplier)
        return neighbor_fn, energy_fn

    return energy_fn


def customn_lennard_jones_neighbor_list(displacement_or_metric: DisplacementOrMetricFn,
                                box_size: Box,
                                species: Array=None,
                                sigma: Array=1.0,
                                epsilon: Array=1.0,
                                r_onset: float=2.0,
                                r_cutoff: float=2.5,
                                dr_threshold: float=0.5,
                                per_particle: bool=False,
                                capacity_multiplier: float=1.25,
                                initialize_neighbor_list: bool=True,
                                fractional=True,
                                ) -> Tuple[NeighborFn,
                                           Callable[[Array, NeighborList],
                                                    Array]]:
  """Convenience wrapper to compute lennard-jones using a neighbor list.
     Different implementation of the cutoff to disentable with energy_params.
     Option not to initialize neighbor list to allow jitable building of
     energy function for varying sigma and epsilon."""
  sigma = jnp.array(sigma, f32)
  epsilon = jnp.array(epsilon, f32)
  r_onset = jnp.array(r_onset, f32)
  r_cutoff = jnp.array(r_cutoff, f32)
  dr_threshold = jnp.array(dr_threshold, f32)

  energy_fn = smap.pair_neighbor_list(
    multiplicative_isotropic_cutoff(energy.lennard_jones, r_onset, r_cutoff),
    space.canonicalize_displacement_or_metric(displacement_or_metric),
    species=species,
    sigma=sigma,
    epsilon=epsilon,
    reduce_axis=(1,) if per_particle else None)

  if initialize_neighbor_list:
    neighbor_fn = partition.neighbor_list(displacement_or_metric, box_size, r_cutoff, dr_threshold,
                                          capacity_multiplier=capacity_multiplier, fractional_coordinates=fractional)
    return neighbor_fn, energy_fn
  return energy_fn


def tabulated(dr: Array, spline: Callable[[Array], Array], **unused_kwargs) -> Array:
    """
    Tabulated radial potential between particles given a spline function.

    Args:
        dr: An ndarray of pairwise distances between particles
        spline: A function computing the spline values at a given pairwise distance

    Returns:
        Array of energies
    """

    return spline(dr)


def tabulated_pair(displacement_or_metric: DisplacementOrMetricFn,
                   x_vals: Array,
                   y_vals: Array,
                   degree: int=3,
                   monotonic: bool=True,
                   r_onset: Array=0.9,
                   r_cutoff: Array=1.,
                   species: Array = None,
                   per_particle: bool=False) -> Callable[[Array], Array]:
    """Convenience wrapper to compute tabulated energy over a system."""
    x_vals = jnp.array(x_vals, f32)
    y_vals = jnp.array(y_vals, f32)
    r_onset = jnp.array(r_onset, f32)
    r_cutoff = jnp.array(r_cutoff, f32)

    if monotonic:
        spline = custom_interpolate.MonotonicInterpolate(x_vals, y_vals)
    else:
        spline = custom_interpolate.InterpolatedUnivariateSpline(x_vals, y_vals, k=degree)
    tabulated_partial = partial(tabulated, spline=spline)

    return smap.pair(
      multiplicative_isotropic_cutoff(tabulated_partial, r_onset, r_cutoff),
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      reduce_axis=(1,) if per_particle else None)


def tabulated_neighbor_list(displacement_or_metric: DisplacementOrMetricFn,
                            x_vals: Array,
                            y_vals: Array,
                            box_size: Box,
                            degree: int=3,
                            monotonic: bool=True,
                            r_onset: Array=0.9,
                            r_cutoff: Array=1.,
                            dr_threshold: Array=0.2,
                            species: Array = None,
                            capacity_multiplier: float=1.25,
                            initialize_neighbor_list: bool=True,
                            per_particle: bool=False,
                            fractional=True) -> Callable[[Array], Array]:
    """
    Convenience wrapper to compute tabulated energy using a neighbor list.

    Provides option not to initialize neighborlist. This is useful if energy function needs
    to be initialized within a jitted function.
    """

    x_vals = jnp.array(x_vals, f32)
    y_vals = jnp.array(y_vals, f32)
    box_size = jnp.array(box_size, f32)
    r_onset = jnp.array(r_onset, f32)
    r_cutoff = jnp.array(r_cutoff, f32)
    dr_threshold = jnp.array(dr_threshold, f32)

    # Note: cannot provide the spline parameters via kwargs because only per-perticle parameters are supported
    if monotonic:
        spline = custom_interpolate.MonotonicInterpolate(x_vals, y_vals)
    else:
        spline = custom_interpolate.InterpolatedUnivariateSpline(x_vals, y_vals, k=degree)
    tabulated_partial = partial(tabulated, spline=spline)

    energy_fn = smap.pair_neighbor_list(
      multiplicative_isotropic_cutoff(tabulated_partial, r_onset, r_cutoff),
      space.canonicalize_displacement_or_metric(displacement_or_metric),
      species=species,
      reduce_axis=(1,) if per_particle else None)

    if initialize_neighbor_list:
        neighbor_fn = partition.neighbor_list(displacement_or_metric, box_size, r_cutoff, dr_threshold,
                                              capacity_multiplier=capacity_multiplier, fractional_coordinates=fractional)
        return neighbor_fn, energy_fn
    return energy_fn


def pair_interaction_nn(displacement: DisplacementFn,
                        r_cutoff: float,
                        hidden_layers,
                        activation=jax.nn.swish,
                        num_rbf: int = 6,
                        envelope_p: int = 6,
                        init_kwargs: Dict = None):
    """An MLP acting on pairwise distances independently and
    summing the contributions.

    Embeds pairwise distances via radial Bessel functions (RBF). The
    RBF is also used to enforce a differentiable cut-off.

    Args:
        displacement: Displacement function
        r_cutoff: Radial cut-off of pairwise interactions and neighbor list
        hidden_layers: A list (or scalar in the case of a single hidden layer)
                       of number of neurons for each hidden layer in the MLP
        activation: Activation function
        num_rbf: Number of radial Bessel embedding functions
        envelope_p: Power of envelope polynomial
        init_kwargs: Kwargs for initializaion of MLP

    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters.
    """

    if init_kwargs is None:
        init_kwargs = {
          'w_init': custom_nn.OrthogonalVarianceScalingInit(scale=1.),
          'b_init': hk.initializers.Constant(0.),
        }

    if jnp.isscalar(hidden_layers):
        hidden_layers = [hidden_layers]
    hidden_layers.append(1)  # output layer is scalar energy

    @hk.without_apply_rng
    @hk.transform
    def model(position, neighbor, species=None, **dynamic_kwargs):
        n_particles, _ = neighbor.idx.shape
        if species is not None:
            smap._check_species_dtype(species)  # assert species are int
            raise NotImplementedError('Add species embedding to distance '
                                      'embedding.')

        dynamic_displacement = partial(displacement, **dynamic_kwargs)
        dyn_neighbor_displacement_fn = space.map_neighbor(dynamic_displacement)

        # compute pairwise distances
        neighbor_mask = neighbor.idx != n_particles
        r_neigh = position[neighbor.idx]
        pair_displacement = dyn_neighbor_displacement_fn(position, r_neigh)
        pair_distances = space.distance(pair_displacement)
        pair_distances = jnp.where(neighbor_mask, pair_distances,
                                   2. * r_cutoff)

        net = custom_nn.PairwiseNNEnergy(r_cutoff, hidden_layers, init_kwargs,
                                         activation, num_rbf, envelope_p)
        nn_energy = net(pair_distances, species, **dynamic_kwargs)
        return nn_energy
    return model.init, model.apply


def DimeNetPP_neighborlist(displacement: DisplacementFn,
                           R_test: Array,
                           neighbor_test,
                           r_cutoff: float,
                           kbt_dependent=False,
                           embed_size: int = 32,
                           n_interaction_blocks: int = 4,
                           num_residual_before_skip: int = 1,
                           num_residual_after_skip: int = 2,
                           out_embed_size=None,
                           type_embed_size=None,
                           angle_int_embed_size=None,
                           basis_int_embed_size: int = 8,
                           num_dense_out: int = 3,
                           num_RBF: int = 6,
                           num_SBF: int = 7,
                           activation=jax.nn.swish,
                           envelope_p: int = 6,
                           init_kwargs: Dict[str, Any] = None,
                           n_species: int = 10,
                           max_angle_multiplier: float = 1.25,
                           max_edge_multiplier: float = 1.25
                           ) -> Tuple[nn.InitFn, Callable[[PyTree, Array], Array]]:
    """
    Convenience wrapper around DimeNetPPEnergy model using a neighbor list.

    The defaults correspond to the orinal values of DimeNet++, except for
    embedding sizes that are reduced by a factor of 4 for efficiency reasons.
    Computes molecular inputs: pairwise distances and angular triplets.
    For computational efficiency, a sparse edge and angle representation is
    implemented. A sample neighbor list to estimate max_edges and max_angles
    is therefore necessary.

    Args:
        displacement: Displacement function
        R_test: Sample state.position for initialization of network and estimate of max_edges / max_angles
        neighbor_test: Sample neighborlist for estimate of max_edges / max_angles
        r_cutoff: Cut-off of DimeNetPP and neighbor list
        embed_size: Size of embeddings; will scale interaction and output embedding sizes accordingly
        n_interaction_blocks: Number of interaction blocks
        num_residual_before_skip: Number of residual blocks before the skip connection
        num_residual_after_skip:Number of residual blocks after the skip connection
        out_embed_size: Embedding size of output block; If None is 2 * embed_size
        type_embed_size: Embedding size of type embedding; If None is 0.5 * embed_size
        angle_int_embed_size: Embedding size of angle interactions; If None is 0.5 * embed_size
        basis_int_embed_size: Embedding size of Bessel basis embedding in interaction block
        num_dense_out: Number of fully-connected layers in output block
        num_RBF: Number of radial Bessel embedding functions
        num_SBF: Number of spherical Bessel embedding functions
        activation: Activation function
        envelope_p: Power of envelope polynomial
        init_kwargs: Kwargs for initializaion of MLPs
        n_species: Number of species
        max_angle_multiplier: Multiplier for initial estimate of maximum triplets
        max_edge_multiplier: Multiplier for initial estimate of maximum edges

    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters 
        and an energy function that computes the energy for a particular state 
        given model parameters.
    """
    r_cutoff = jnp.array(r_cutoff, dtype=f32)
    # Estimate maximum number of edges and angles for efficient computations during simulation:
    # neighbor.idx: an index j in row i encodes a directed edge from particle j to particle i. edge_idx[i, j]: j->i
    # if j == N: encodes no edge exists;  would index out-of-bounds, but in jax last element is returned in this case
    neighbor_displacement_fn = space.map_neighbor(displacement)
    R_neigh_test = R_test[neighbor_test.idx]
    test_displacement = neighbor_displacement_fn(R_test, R_neigh_test)
    pair_distances_test = space.distance(test_displacement)
    # adds all edges > cut-off to masked edges
    edge_idx_test = jnp.where(pair_distances_test < r_cutoff, neighbor_test.idx, R_test.shape[0])
    _, _, _, _, (n_edges_init, n_angles_init) = custom_nn.sparse_representation(pair_distances_test, edge_idx_test)
    max_angles = jnp.int32(jnp.ceil(n_angles_init * max_edge_multiplier))
    max_edges = jnp.int32(jnp.ceil(n_edges_init * max_angle_multiplier))

    if init_kwargs is None:
        init_kwargs = {
          'w_init': custom_nn.OrthogonalVarianceScalingInit(scale=1.),
          'b_init': hk.initializers.Constant(0.),
        }

    @hk.without_apply_rng
    @hk.transform
    def model(R: Array, neighbor, species=None, **dynamic_kwargs) -> Array:
        n_particles, _ = neighbor.idx.shape
        if species is None:  # dummy species to allow streamlined use of different species
            species = jnp.zeros(n_particles, dtype=jnp.int32)
        else:
            smap._check_species_dtype(species)  # assert species are int

        dynamic_displacement = partial(displacement, **dynamic_kwargs)  # necessary for pressure computation
        dyn_neighbor_displacement_fn = space.map_neighbor(dynamic_displacement)

        # compute pairwise distances
        R_neigh = R[neighbor.idx]
        pair_displacement = dyn_neighbor_displacement_fn(R, R_neigh)
        pair_distances = space.distance(pair_displacement)

        # compute adjacency matrix via neighbor_list, then build sparse representation to avoid part of padding overhead
        # TODO new sparse neighborlist format would simplify building the connectivity
        edge_idx_ji = jnp.where(pair_distances < r_cutoff, neighbor.idx, n_particles)  # adds all edges > cut-off to masked edges
        pair_distances_sparse, pair_connections, angle_idxs, angular_connectivity, (n_edges, n_angles) = \
            custom_nn.sparse_representation(pair_distances, edge_idx_ji, max_edges, max_angles)
        too_many_edges_error_code = lax.cond(jnp.bitwise_or(n_edges > max_edges, n_angles > max_angles),
                                             lambda _: True, lambda _: False, n_edges)
        # TODO: return too_many_edges_error_code

        idx_i, idx_j, pair_mask = pair_connections
        # cutoff all non existing edges: are encoded as 0 by rbf envelope; non-existing angles will also be encoded as 0
        pair_distances_sparse = jnp.where(pair_mask[:, 0], pair_distances_sparse, 2. * r_cutoff)
        angles = custom_nn.angles_triplets(R, dynamic_displacement, angle_idxs, angular_connectivity)
        net = custom_nn.DimeNetPPEnergy(r_cutoff,
                                        n_particles=n_particles,
                                        n_species=n_species,
                                        kbt_dependent=kbt_dependent,
                                        embed_size=embed_size,
                                        n_interaction_blocks=n_interaction_blocks,
                                        num_residual_before_skip=num_residual_before_skip,
                                        num_residual_after_skip=num_residual_after_skip,
                                        out_embed_size=out_embed_size,
                                        type_embed_size=type_embed_size,
                                        angle_int_embed_size=angle_int_embed_size,
                                        basis_int_embed_size=basis_int_embed_size,
                                        num_dense_out=num_dense_out,
                                        num_RBF=num_RBF,
                                        num_SBF=num_SBF,
                                        activation=activation,
                                        envelope_p=envelope_p,
                                        init_kwargs=init_kwargs)
        gnn_energy = net(pair_distances_sparse, angles, species, pair_connections, angular_connectivity, **dynamic_kwargs)
        gnn_energy = gnn_energy[0]  # the net returns a 1D array as output, but grad needs a scalar for differentiation
        return gnn_energy

    return model.init, model.apply
