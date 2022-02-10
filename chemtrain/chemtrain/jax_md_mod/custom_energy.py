from functools import partial
from typing import Callable, Tuple, Dict, Any

import jax
from jax import lax, vmap, nn as jax_nn
import jax.numpy as jnp
import haiku as hk
from jax_md import space, partition, nn, util, energy, smap
from jax_md.energy import multiplicative_isotropic_cutoff, _sw_radial_interaction, _sw_angle_interaction

from chemtrain.jax_md_mod import custom_interpolate, custom_nn, sparse_graph
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


def dimenetpp_neighborlist(displacement: DisplacementFn,
                           r_cutoff: float,
                           positions_test: jnp.ndarray = None,
                           neighbor_test: partition.NeighborList = None,
                           max_angle_multiplier: float = 1.25,
                           max_edge_multiplier: float = 1.25,
                           kbt_dependent: bool = False,
                           embed_size: int = 128,
                           n_interaction_blocks: int = 4,
                           num_residual_before_skip: int = 1,
                           num_residual_after_skip: int = 2,
                           out_embed_size=None,
                           type_embed_size=None,
                           angle_int_embed_size=None,
                           basis_int_embed_size: int = 8,
                           num_dense_out: int = 3,
                           num_rbf: int = 6,
                           num_sbf: int = 7,
                           activation=jax_nn.swish,
                           envelope_p: int = 6,
                           init_kwargs: Dict[str, Any] = None,
                           n_species: int = 10,
                           ) -> Tuple[nn.InitFn,
                                      Callable[[PyTree, Array], Array]]:
    """DimeNet++ energy function for Jax, M.D.

    The default values correspond to the orinal values of DimeNet++.

    This function provides an interface for the DimeNet++ haiku model to be used
    as a jax_md energy_fn. Analogous to jax_md energy_fns, the initialized
    DimeNet++ energy_fn requires particle positions and a dense neighbor list as
    input - plus an array for species or other dynamic kwargs, if applicable.

    From particle positions and neighbor list, the sparse graph representation
    with edges and angle triplets is computed. Due to the constant shape
    requirement of jit of the neighborlist in jax_md, the neighbor list contains
    many masked edges, i.e. pairwise interactions that only "fill" the neighbor
    list, but are set to 0 during computation. This translates to masked edges
    and triplets in the sparse graph representation.

    For improved computational efficiency during jax_md simulations, the
    maximum number of edges and triplets can be estimated during model
    initialization. Edges and triplets beyond this maximum estimate can be
    capped to reduce computational and memory requirements. Capping is enabled
    by providing sample inputs (positions_test and neighbor_test) at
    initialization time. However, beware that currently, an overflow of
    max_edges and max_angles is not caught, as this requires passing an error
    code throgh jax_md simulators - analogous to the overflow detection in
    jax_md neighbor lists. If in doubt, increase the max edges/angles
    multipliers or disable capping.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        positions_test: Sample positions to estimate max_edges / max_angles.
                        Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
                       Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_angle_multiplier: Multiplier for initial estimate of maximum angles.
        kbt_dependent: True, if potential explicitly depends on temperature.
                       In this case 'kT' needs to be provided as a kwarg during
                       the call to the energy_fn. Default False results in a
                       potential function independent of temperature.
        embed_size: Size of message embeddings. Scale interaction and output
                    embedding sizes accordingly, if not specified explicitly.
        n_interaction_blocks: Number of interaction blocks
        num_residual_before_skip: Number of residual blocks before the skip
                                  connection in the Interaction block.
        num_residual_after_skip: Number of residual blocks after the skip
                                 connection in the Interaction block.
        out_embed_size: Embedding size of output block.
                        If None is set to 2 * embed_size.
        type_embed_size: Embedding size of atom type embeddings.
                         If None is set to 0.5 * embed_size.
        angle_int_embed_size: Embedding size of Linear layers for down-projected
                              triplet interation. If None is 0.5 * embed_size.
        basis_int_embed_size: Embedding size of Linear layers for interation
                              of RBS/ SBF basis in interaction block
        num_dense_out: Number of final Linear layers in output block
        num_rbf: Number of radial Bessel embedding functions
        num_sbf: Number of spherical Bessel embedding functions
        activation: Activation function
        envelope_p: Power of envelope polynomial
        init_kwargs: Kwargs for initializaion of Linear layers
        n_species: Number of different atom species the network is supposed
                   to process.

    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """
    if init_kwargs is None:
        init_kwargs = {
          'w_init': custom_nn.OrthogonalVarianceScalingInit(scale=1.),
          'b_init': hk.initializers.Constant(0.),
        }
    r_cutoff = jnp.array(r_cutoff, dtype=f32)

    if positions_test is not None and neighbor_test is not None:
        print('Capping edges and triplets. Beware of overflow, which is'
              ' currently not being detected.')

        # neighbor.idx: an index j in row i encodes a directed edge from
        #               particle j to particle i.
        # edge_idx[i, j]: j->i. if j == N: encodes masked edge.
        # Index N would index out-of-bounds, but in jax the last element is
        # returned instead
        neighbor_displacement_fn = space.map_neighbor(displacement)
        pos_neigh_test = positions_test[neighbor_test.idx]
        test_displacement = neighbor_displacement_fn(positions_test,
                                                     pos_neigh_test)
        pair_distances_test = space.distance(test_displacement)
        # add all edges > cut-off to masked edges
        edge_idx_test = jnp.where(pair_distances_test < r_cutoff,
                                  neighbor_test.idx, positions_test.shape[0])
        _, _, _, _, (n_edges_init, n_angles_init) = sparse_graph.sparse_graph(
            pair_distances_test, edge_idx_test)
        max_angles = jnp.int32(jnp.ceil(n_angles_init * max_edge_multiplier))
        max_edges = jnp.int32(jnp.ceil(n_edges_init * max_angle_multiplier))
    else:
        max_angles = None
        max_edges = None

    @hk.without_apply_rng
    @hk.transform
    def model(positions: jnp.ndarray,
              neighbor: partition.NeighborList,
              species: jnp.ndarray = None,
              **dynamic_kwargs) -> jnp.ndarray:
        """Evalues the DimeNet++ model and predicts the potential energy.

        Args:
            positions: Jax_md state-position. (N_particles x dim) array of
                       particle positions
            neighbor: Jax_md dense neighbor list corresponding to positions
            species: (N_particles,) Array encoding atom types. If None, assumes
                     all particles to belong to the same species
            **dynamic_kwargs: Dynamic kwargs, such as 'box' or 'kT'.

        Returns:
            Potential energy value of state
        """
        assert neighbor.format.name == 'Dense', ('Currently only dense neighbor'
                                                 ' lists supported.')
        n_particles, _ = neighbor.idx.shape
        if species is None:  # build dummy species
            species = jnp.zeros(n_particles, dtype=jnp.int32)
        else:
            smap._check_species_dtype(species)  # assert species are int

        # dynamic box necessary for pressure computation
        dynamic_displacement = partial(displacement, **dynamic_kwargs)
        dyn_neighbor_displacement_fn = space.map_neighbor(dynamic_displacement)

        # compute pairwise distances
        pos_neigh = positions[neighbor.idx]
        pair_displacement = dyn_neighbor_displacement_fn(positions, pos_neigh)
        pair_distances = space.distance(pair_displacement)

        # compute adjacency matrix via neighbor_list, then build sparse graph
        # representation to avoid part of padding overhead in dense neighborlist
        # adds all edges > cut-off to masked edges
        edge_idx_ji = jnp.where(pair_distances < r_cutoff, neighbor.idx,
                                n_particles)
        sparse_rep = sparse_graph.sparse_graph(pair_distances, edge_idx_ji,
                                               max_edges, max_angles)

        (pair_distances_sparse, pair_connections, angle_idxs,
         angular_connectivity, (n_edges, n_angles)) = sparse_rep
        idx_i, idx_j, pair_mask = pair_connections
        angle_mask, _, _ = angular_connectivity

        too_many_edges_error_code = lax.cond(
            jnp.bitwise_or(n_edges > max_edges, n_angles > max_angles),
            lambda _: True, lambda _: False, n_edges
        )
        # TODO: return too_many_edges_error_code to detect possible overflow
        del too_many_edges_error_code

        # cutoff all non existing edges: are encoded as 0 by rbf envelope
        pair_distances_sparse = jnp.where(pair_mask[:, 0],
                                          pair_distances_sparse, 2. * r_cutoff)
        angles = sparse_graph.angle_triplets(positions, dynamic_displacement,
                                             angle_idxs, angle_mask)
        # non-existing angles will also be masked explicitly in DimeNet++
        net = custom_nn.DimeNetPP(
            r_cutoff,
            n_species,
            num_targets=1,
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
            num_rbf=num_rbf,
            num_sbf=num_sbf,
            activation=activation,
            envelope_p=envelope_p,
            init_kwargs=init_kwargs
        )

        per_atom_energies = net(pair_distances_sparse, angles, species,
                                (idx_i, idx_j), angular_connectivity,
                                **dynamic_kwargs)
        gnn_energy = util.high_precision_sum(per_atom_energies)
        return gnn_energy

    return model.init, model.apply
