"""
Customn implementation of the outstanding DimeNet++ architecture in haiku.
https://github.com/klicperajo/dimenet.
Includes functions computing a sparse edge / angle representation from neighbor lists.
"""

import jax.nn
from sympy import symbols  # to setup spherical basis functions
from sympy.utilities.lambdify import lambdify
import jax.scipy as jsp
from jax import vmap, lax, ops
import jax.numpy as jnp
from jax_md import util
import haiku as hk
from chemtrain.jax_md_mod import custom_quantity, dimenet_basis_util, \
    custom_util
from typing import Dict, Any

# sparse representation utility functions


def safe_angle_mask(R_ji, R_kj, angle_mask):
    """Masking angles to ensure fifferentiablility."""
    safe_ji = jnp.array([1., 0., 0.], dtype=jnp.float32)
    safe_kj = jnp.array([0., 1., 0.], dtype=jnp.float32)
    R_ji_safe = jnp.where(angle_mask, R_ji, safe_ji)
    R_kj_safe = jnp.where(angle_mask, R_kj, safe_kj)
    return R_ji_safe, R_kj_safe


def angles_triplets(R, displacement_fn, angle_idxs, angular_connectivity):
    """Computes the angle for all triplets between 0 and Pi. Masked angles are set to pi/2."""
    angle_mask, _, _ = angular_connectivity
    R_i = R[angle_idxs[:, 0]]
    R_j = R[angle_idxs[:, 1]]
    R_k = R[angle_idxs[:, 2]]

    # Note: The DimeNet implementation uses R_ji, however R_ij is the correct vector to get the angle between
    #       both vectors. With R_ji, Pi - Theta is computed. This is a known issue in DimeNet.
    #       We apply the proper definition of the angle.
    R_ij = vmap(displacement_fn)(R_i, R_j)  # R_i - R_j respecting periodic BCs
    R_kj = vmap(displacement_fn)(R_k, R_j)
    # we need to mask as the case R_ji is co-linear with R_kj would otherwise generate NaNs on the backward pass
    R_ij_safe, R_kj_safe = safe_angle_mask(R_ij, R_kj, angle_mask)
    angles = vmap(custom_quantity.angle)(R_ij_safe, R_kj_safe)
    return angles


def flatten_sort_and_capp(matrix, sorting_args, cap_size):
    """
    Takes a 2D array, flattens it, sorts it using the args (usually provided via argsort) and
    capps the end of the resulting vect. Used to delete non-existing edges. Returns the capped vector.
    """
    vect = jnp.ravel(matrix)
    sorted_vect = vect[sorting_args]
    capped_vect = sorted_vect[0:cap_size]
    return capped_vect


def sparse_representation(pair_distances, edge_idx_ji, max_edges=None, max_angles=None):
    """
    Constructs a sparse representation of graph edges and angles to save memory and computations over neighbor list.

    To allow for a representation of constant size required by jit, we pad the resulting vectors.

    Args:
        pair_distances: Pairwise distances of particles
        cutoff: Cutoff value obove which edges are not included in graph
        max_edges: Maximum number of edges storable in the graph
        max_angles: Maximum number of edges storable in the graph

    Returns:
        Arrays defining sparse graph connectivity
    """

    # conservative estimates for initialization run
    # use guess from initialization for tighter bound to save memory and computations during production runs
    N, max_neighbors = edge_idx_ji.shape
    if max_edges is None:
        max_edges = N * max_neighbors
    if max_angles is None:
        max_angles = max_edges * max_neighbors

    # sparse edge representation: construct vectors from adjacency matrix and only keep existing edges
    # Target node (i) and source (j) of edges
    pair_mask = edge_idx_ji != N  # non-existing neighbors are encoded as N
    n_edges = jnp.count_nonzero(pair_mask)  # due to undirectedness, each edge is included twice
    pair_mask_flat = jnp.ravel(pair_mask)
    sorting_idxs = jnp.argsort(~pair_mask_flat)  # non-existing edges are sorted to end
    _, yy = jnp.meshgrid(jnp.arange(max_neighbors), jnp.arange(N))
    idx_i = flatten_sort_and_capp(yy, sorting_idxs, max_edges)
    idx_j = flatten_sort_and_capp(edge_idx_ji, sorting_idxs, max_edges)
    d_ij = flatten_sort_and_capp(pair_distances, sorting_idxs, max_edges)
    sparse_pair_mask = flatten_sort_and_capp(pair_mask_flat, sorting_idxs, max_edges)
    pair_indicies = (idx_i, idx_j, jnp.expand_dims(sparse_pair_mask, -1))  # edge connectivity

    # build sparse angle combinations from adjacency matrix:
    # angle defined for 3 particles with connections k->j and j->i
    # directional message passing accumulates all k->j to update each m_ji
    idx3_i = jnp.repeat(idx_i, max_neighbors)
    idx3_j = jnp.repeat(idx_j, max_neighbors)
    idx3_k_mat = edge_idx_ji[idx_j]  # retrieves for each j in idx_j its neighbors k: stored in 2nd axis
    idx3_k = idx3_k_mat.ravel()
    angle_idxs = jnp.column_stack([idx3_i, idx3_j, idx3_k])
    # masking:
    # k and j are different particles, by edge_idx_ji construction. The same applies to j - i, except for masked ones
    mask_i_eq_k = idx3_i != idx3_k
    mask_ij = jnp.repeat(sparse_pair_mask, max_neighbors)  # mask for ij known a priori
    mask_k = idx3_k != N
    angle_mask = mask_ij * mask_k * mask_i_eq_k  # union of masks
    angle_mask, sorting_idx3 = lax.top_k(angle_mask, max_angles)
    angle_idxs = angle_idxs[sorting_idx3]
    n_angles = jnp.count_nonzero(angle_mask)

    # retrieving edge_id m_ji from nodes i and j:
    # idx_i < N by construction, but idx_j can be N: will override lookup[i, N-1],
    # which is problematic if [i, N-1] is an existing edge
    edge_id_lookup = jnp.zeros([N, N + 1], dtype=jnp.int32)
    edge_id_lookup_direct = ops.index_update(edge_id_lookup, (idx_i, idx_j), jnp.arange(max_edges))

    # stores for each angle kji edge index j->i to aggregate messages via a segment_sum:
    # each m_ji is a distinct segment containing all incoming m_kj
    reduce_to_ji = edge_id_lookup_direct[(angle_idxs[:, 0], angle_idxs[:, 1])]
    # stores for each angle kji edge index k->j to gather all incoming edges for message passing
    expand_to_kj = edge_id_lookup_direct[(angle_idxs[:, 1], angle_idxs[:, 2])]
    angle_connectivity = (jnp.expand_dims(angle_mask, -1), reduce_to_ji, expand_to_kj)

    return d_ij, pair_indicies, angle_idxs, angle_connectivity, (n_edges, n_angles)

# Initializers


class OrthogonalVarianceScalingInit(hk.initializers.Initializer):
    """
    Scales Variance of uniform orthogonal matrix distribution.
    Approach adopted from original DimeNet and implementation inspired by Haiku Variance scaling.
    """
    def __init__(self, scale=2.):
        super().__init__()
        self.scale = scale
        self.orth_init = hk.initializers.Orthogonal()

    def __call__(self, shape, dtype=jnp.float32):
        assert len(shape) == 2
        fan_in, fan_out = shape
        w_init = self.orth_init(shape, dtype)  # uniformly distributed orthogonal weight matrix
        w_init *= jnp.sqrt(self.scale / (max(1., (fan_in + fan_out)) * jnp.var(w_init)))
        return w_init

# DimeNet++ Layers


class SmoothingEnvelope(hk.Module):
    """
    A function that is 1 at 0 and has root of multiplicity 3 at 1 as defined in DimeNet.
    Apply on d/c [0, 1] to enable a smooth cut-off.

    The implementation corresponds to the definition in the DimeNet paper. It is different from
    the implementation of DimeNet / DimeNet++ that define incorrect spherical basis layers (a known issue).
    """

    def __init__(self, p=6, name='Envelope'):
        super().__init__(name=name)
        self.p = p
        self.a = -(p + 1.) * (p + 2.) / 2.
        self.b = p * (p + 2.)
        self.c = -p * (p + 1.) / 2.

    def __call__(self, inputs):
        envelope_val = 1. + self.a * inputs**self.p + self.b * inputs**(self.p + 1.) + self.c * inputs**(self.p + 2.)
        return jnp.where(inputs < 1., envelope_val, 0.)


class RBFFrequencyInitializer(hk.initializers.Initializer):
    """Initializes the frequencies of the Radial Bessel Function to its canonical values."""
    def __call__(self, shape, dtype):
        return jnp.pi * jnp.arange(1, shape[0] + 1, dtype=dtype)


class RadialBesselLayer(hk.Module):
    """A Layer that computes the Radial Bessel Function representation of pairwise distances."""

    def __init__(self, cutoff, num_radial=16, envelope_p=6, name='BesselRadial'):
        super().__init__(name=name)
        self.inv_cutoff = 1. / cutoff
        self.envelope = SmoothingEnvelope(p=envelope_p)
        self.num_radial = [num_radial]
        self.RBF_scale = jnp.sqrt(2. / cutoff)
        self.freq_init = RBFFrequencyInitializer()

    def __call__(self, distances):
        distances = jnp.expand_dims(distances, -1)  # to broadcast to num_radial
        scaled_distances = distances * self.inv_cutoff
        envelope_vals = self.envelope(scaled_distances)
        frequencies = hk.get_parameter("RBF_Frequencies", shape=self.num_radial, dtype=jnp.float32,
                                       init=self.freq_init)
        RBF_vals = self.RBF_scale * jnp.sin(frequencies * scaled_distances) / distances
        return envelope_vals * RBF_vals


class SphericalBesselLayer(hk.Module):
    """A Layer that computes the Spherical Bessel Function representation of triplets."""
    def __init__(self, r_cutoff, num_spherical, num_radial, envelope_p=6, name='BesselSpherical'):
        super().__init__(name=name)

        assert num_spherical > 1
        self.envelope = SmoothingEnvelope(p=envelope_p)
        self.inv_cutoff = 1. / r_cutoff
        self.num_radial = num_radial

        bessel_formulars = dimenet_basis_util.bessel_basis(num_spherical, num_radial)
        sph_harmonic_formulas = dimenet_basis_util.real_sph_harm(num_spherical)

        self.sph_funcs = []
        self.radual_bessel_funcs = []
        # convert sympy functions: modules overrides sympy functions by jax.numpy functions
        x = symbols('x')
        theta = symbols('theta')
        for i in range(num_spherical):
            if i == 0:
                first_sph = lambdify([theta], sph_harmonic_formulas[i][0], modules=[jnp, jsp.special])(0)
                self.sph_funcs.append(lambda input: jnp.zeros_like(input) + first_sph)
            else:
                self.sph_funcs.append(lambdify([theta], sph_harmonic_formulas[i][0], modules=[jnp, jsp.special]))
            for j in range(num_radial):
                self.radual_bessel_funcs.append(lambdify([x], bessel_formulars[i][j], modules=[jnp, jsp.special]))

    def __call__(self, pair_distances, angles, angular_connectivity):
        angle_mask, _, expand_to_kj = angular_connectivity

        # initialize distances and envelope values
        scaled_distances = pair_distances * self.inv_cutoff
        envelope_vals = self.envelope(scaled_distances)
        envelope_vals = jnp.expand_dims(envelope_vals, -1)  # to allow broadcast to rbf

        # compute radial bessel envelope for distances kj
        rbf = [radial_bessel(scaled_distances) for radial_bessel in self.radual_bessel_funcs]
        rbf = jnp.stack(rbf, axis=1)
        rbf_envelope = rbf * envelope_vals
        rbf_env_expanded = rbf_envelope[expand_to_kj]

        # compute spherical bessel embedding
        sbf = [spherical_bessel(angles) for spherical_bessel in self.sph_funcs]
        sbf = jnp.stack(sbf, axis=1)
        sbf = jnp.repeat(sbf, self.num_radial, axis=1)

        sbf *= angle_mask  # mask non-existing triplets

        return rbf_env_expanded * sbf  # combine radial, spherical and envelope


class ResidualLayer(hk.Module):
    def __init__(self, layer_size, activation=jax.nn.swish, init_kwargs=None, name='ResLayer'):
        super().__init__(name=name)
        self.residual = hk.Sequential([
            hk.Linear(layer_size, name='ResidualFirstLinear', **init_kwargs), activation,
            hk.Linear(layer_size, name='ResidualSecondLinear', **init_kwargs), activation
        ])

    def __call__(self, inputs):
        out = inputs + self.residual(inputs)
        return out


class EmbeddingBlock(hk.Module):
    def __init__(self, embed_size, n_species, type_embed_size=None, activation=jax.nn.swish,
                 init_kwargs=None, kbt_dependent=False, name='Embedding'):
        super().__init__(name=name,)

        if type_embed_size is None:
            type_embed_size = int(embed_size / 2)

        embed_init = hk.initializers.RandomUniform(minval=-jnp.sqrt(3), maxval=jnp.sqrt(3))
        self.embedding_vect = hk.get_parameter('Embedding_vect', [n_species, type_embed_size],
                                               init=embed_init, dtype=jnp.float32)
        self.kbt_dependent = kbt_dependent
        if kbt_dependent:
            self.kbt_embedding = hk.get_parameter('Embedding_kbt', [1, type_embed_size],
                                                  init=embed_init, dtype=jnp.float32)

        # unlike the original DimeNet implementation, there is no activation and bias in RBF_Dense as shown in the
        # network sketch. This is consistent with other Layers processing rbf values throughout the network
        self.rbf_dense = hk.Linear(embed_size, name='RBF_Dense', with_bias=False, **init_kwargs)
        self.dense_after_concat = hk.Sequential([hk.Linear(embed_size, name='Concat_Dense', **init_kwargs), activation])

    def __call__(self, rbf, species, pair_connectivity, **kwargs):
        idx_i, idx_j, _ = pair_connectivity
        transformed_rbf = self.rbf_dense(rbf)

        type_i = species[idx_i]
        type_j = species[idx_j]

        h_i = self.embedding_vect[type_i]
        h_j = self.embedding_vect[type_j]

        edge_embedding = jnp.concatenate([h_i, h_j, transformed_rbf], axis=-1)
        if self.kbt_dependent:
            assert 'kT' in kwargs, ('If potential should be kbt-dependent. '
                                    '"kT" needs to be provided as kwarg.')
            kbt_embeds = jnp.tile(self.kbt_embedding, (h_j.shape[0], 1))
            kbt_embeds *= kwargs['kT']
            edge_embedding = jnp.concatenate([edge_embedding, kbt_embeds],
                                             axis=-1)
        embedded_messages = self.dense_after_concat(edge_embedding)
        return embedded_messages


class OutputBlock(hk.Module):
    def __init__(self, embed_size, n_particles, out_embed_size=None, num_dense=2, num_targets=1,
                 activation=jax.nn.swish, init_kwargs=None, name='Output'):
        super().__init__(name=name)

        if out_embed_size is None:
            out_embed_size = int(2 * embed_size)

        self.n_particles = n_particles
        self.rbf_dense = hk.Linear(embed_size, with_bias=False, name='RBF_Dense', **init_kwargs)
        self.upprojection = hk.Linear(out_embed_size, with_bias=False, name='Upprojection', **init_kwargs)

        # transform summed messages over multiple dense layers before predicting output
        self.dense_layers = []
        for _ in range(num_dense):
            self.dense_layers.append(hk.Sequential([hk.Linear(
                out_embed_size, with_bias=True, name='Dense_Series', **init_kwargs), activation]))

        self.dense_final = hk.Linear(num_targets, with_bias=False, name='Final_output', **init_kwargs)

    def __call__(self, messages, rbf, connectivity):
        idx_i, _, _ = connectivity
        transformed_rbf = self.rbf_dense(rbf)
        messages *= transformed_rbf  # rbf is masked correctly, transformation only via weights --> rbf acts as mask

        # sum incoming messages for each atom: becomes a per-atom quantity
        summed_messages = custom_util.high_precision_segment_sum(messages, idx_i, num_segments=self.n_particles)

        upsampled_messages = self.upprojection(summed_messages)
        for dense_layer in self.dense_layers:
            upsampled_messages = dense_layer(upsampled_messages)

        per_atom_targets = self.dense_final(upsampled_messages)
        return per_atom_targets


class InteractionBlock(hk.Module):
    def __init__(self, embed_size, num_res_before_skip, num_res_after_skip, activation=jax.nn.swish,
                 init_kwargs=None, angle_int_embed_size=None, basis_int_embed_size=8, name='Interaction'):
        super().__init__(name=name)

        if angle_int_embed_size is None:
            angle_int_embed_size = int(embed_size / 2)

        # directional message passing block
        self.rbf1 = hk.Linear(basis_int_embed_size, name='rbf1', with_bias=False,  **init_kwargs)
        self.rbf2 = hk.Linear(embed_size, name='rbf2', with_bias=False, **init_kwargs)
        self.sbf1 = hk.Linear(basis_int_embed_size, name='sbf1', with_bias=False,  **init_kwargs)
        self.sbf2 = hk.Linear(angle_int_embed_size, name='sbf2', with_bias=False, **init_kwargs)

        self.dense_kj = hk.Sequential([hk.Linear(embed_size, name='Dense_kj', **init_kwargs), activation])
        self.down_projection = hk.Sequential([
            hk.Linear(angle_int_embed_size, name='Downprojection', with_bias=False, **init_kwargs), activation])
        self.up_projection = hk.Sequential([
            hk.Linear(embed_size, name='Upprojection', with_bias=False, **init_kwargs), activation])

        # propagation block:
        self.dense_ji = hk.Sequential([hk.Linear(embed_size, name='Dense_ji', **init_kwargs), activation])

        self.res_before_skip = []
        for _ in range(num_res_before_skip):
            self.res_before_skip.append(ResidualLayer(embed_size, activation, init_kwargs, name='ResLayerBeforeSkip'))
        self.final_before_skip = hk.Sequential([hk.Linear(
            embed_size, name='FinalBeforeSkip', **init_kwargs), activation])

        self.res_after_skip = []
        for _ in range(num_res_after_skip):
            self.res_after_skip.append(ResidualLayer(embed_size, activation, init_kwargs, name='ResLayerAfterSkip'))

    def __call__(self, m_input, rbf, sbf, angular_connectivity):
        # directional message passing block:
        _, reduce_to_ji, expand_to_kj = angular_connectivity
        m_ji_angular = self.dense_kj(m_input)  # transformed messages for expansion to k -> j
        rbf = self.rbf1(rbf)
        rbf = self.rbf2(rbf)
        m_ji_angular *= rbf

        m_ji_angular = self.down_projection(m_ji_angular)
        m_kj = m_ji_angular[expand_to_kj]  # expand to nodes k connecting to j

        sbf = self.sbf1(sbf)
        sbf = self.sbf2(sbf)
        m_kj *= sbf  # automatic mask: sbf was masked during initial computation. Sbf1 and 2 only weights, no biases

        aggregated_m_ji = custom_util.high_precision_segment_sum(m_kj, reduce_to_ji, num_segments=m_input.shape[0])
        propagated_messages = self.up_projection(aggregated_m_ji)

        # add directional messages to original ones; afterwards only independent edge transformations
        # masking is lost, but rbf masks in output layer before aggregation
        m_ji = self.dense_ji(m_input)
        m_combined = m_ji + propagated_messages

        for layer in self.res_before_skip:
            m_combined = layer(m_combined)
        m_combined = self.final_before_skip(m_combined)

        m_ji_with_skip = m_combined + m_input

        for layer in self.res_after_skip:
            m_ji_with_skip = layer(m_ji_with_skip)
        return m_ji_with_skip


class DimeNetPPEnergy(hk.Module):
    """Implements DimeNet++ for predicting energies with sparse graph representation and masked edges / angles.

    This customn implementation follows the original DimeNet / DimeNet++, while correcting for known issues.
    https://arxiv.org/abs/2011.14115 ; https://github.com/klicperajo/dimenet
    This model takes sparse representation of molecular graph (pairwise distances and angular triplets) as input
    and predicts energy.
    Non-existing edges from fixed array size requirement are masked indirectly via RBF envelope function and
    non-existing triplets are masked explicitly in SBF embedding layer.
    """

    def __init__(self,
                 r_cutoff: float,
                 n_species: int,
                 n_particles: int,
                 kbt_dependent=False,
                 embed_size: int = 32,
                 n_interaction_blocks: int = 4,
                 num_residual_before_skip: int = 1,
                 num_residual_after_skip: int = 3,
                 out_embed_size=None,
                 type_embed_size=None,
                 angle_int_embed_size=None,
                 basis_int_embed_size = 8,
                 num_dense_out: int = 3,
                 num_RBF: int = 6,
                 num_SBF: int = 7,
                 activation=jax.nn.swish,
                 envelope_p: int = 6,
                 init_kwargs: Dict[str, Any] = None,
                 name: str = 'Energy'):
        super(DimeNetPPEnergy, self).__init__(name=name)

        # input representation:
        self.rbf_layer = RadialBesselLayer(r_cutoff, num_radial=num_RBF, envelope_p=envelope_p)
        self.sbf_layer = SphericalBesselLayer(r_cutoff, num_radial=num_RBF, num_spherical=num_SBF, envelope_p=envelope_p)

        # build GNN structure
        self.n_interactions = n_interaction_blocks
        self.output_blocks = []
        self.int_blocks = []
        self.embedding_layer = EmbeddingBlock(embed_size, n_species, type_embed_size=type_embed_size,
                                              activation=activation, init_kwargs=init_kwargs, kbt_dependent=kbt_dependent)
        self.output_blocks.append(OutputBlock(embed_size, n_particles, out_embed_size, num_dense=num_dense_out,
                                              num_targets=1, activation=activation, init_kwargs=init_kwargs))

        for _ in range(n_interaction_blocks):
            self.int_blocks.append(InteractionBlock(embed_size, num_residual_before_skip, num_residual_after_skip,
                                                    activation=activation, angle_int_embed_size=angle_int_embed_size,
                                                    basis_int_embed_size=basis_int_embed_size, init_kwargs=init_kwargs))
            self.output_blocks.append(OutputBlock(embed_size, n_particles, out_embed_size, num_dense=num_dense_out,
                                                  num_targets=1, activation=activation, init_kwargs=init_kwargs))

    def __call__(self, distances, angles, species, pair_connections, angular_connections, **dyn_kwargs) -> jnp.ndarray:
        rbf = self.rbf_layer(distances)  # correctly masked by construction: masked distances are 2 * cut-off --> rbf=0
        sbf = self.sbf_layer(distances, angles, angular_connections)  # is masked too

        messages = self.embedding_layer(rbf, species, pair_connections, **dyn_kwargs)
        per_atom_quantities = self.output_blocks[0](messages, rbf, pair_connections)

        for i in range(self.n_interactions):
            messages = self.int_blocks[i](messages, rbf, sbf, angular_connections)
            per_atom_quantities += self.output_blocks[i + 1](messages, rbf, pair_connections)

        predicted_quantities = util.high_precision_sum(per_atom_quantities, axis=0)  # sum over all atoms
        return predicted_quantities
