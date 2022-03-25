from functools import partial
from typing import Callable, Tuple, Any

from jax import vmap
import jax.numpy as jnp
from jax_md import space, partition, util, energy, smap
from jax_md.energy import multiplicative_isotropic_cutoff, _sw_radial_interaction, _sw_angle_interaction

from chemtrain.jax_md_mod import custom_interpolate

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
                                disable_cell_list=False,
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
                                          capacity_multiplier=capacity_multiplier, fractional_coordinates=fractional, disable_cell_list=disable_cell_list)
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
