"""Jax / haiku implementation of the outstanding DimeNet++ architecture.
https://github.com/klicperajo/dimenet.
"""
from typing import Dict, Any, Callable, Tuple

import haiku as hk
from jax import nn, numpy as jnp, scipy as jsp
from jax_md import util as jax_md_util
from sympy import symbols, utilities

from chemtrain.jax_md_mod import dimenet_basis_util, util


# Initializers

class OrthogonalVarianceScalingInit(hk.initializers.Initializer):
    """Initializer scaling variance of uniform orthogonal matrix distribution.

    Generates a weight matrix with variance according to Glorot initialization.
    Based on a random (semi-)orthogonal matrix. Neural networks are expected to
    learn better when features are decorrelated e.g. stated by
    "Reducing overfitting in deep networks by decorrelating representations".

    The approach is adopted from the original DimeNet and the implementation
    is inspired by Haiku's variance scaling initializer.

    Attributes:
        scale: Variance scaling factor
        orth_init: Haiku Orthogonal initializer
    """
    def __init__(self, scale=2.):
        """Constructs the OrthogonalVarianceScaling Initializer.

        Args:
            scale: Variance scaling factor
        """
        super().__init__()
        self.scale = scale
        self.orth_init = hk.initializers.Orthogonal()

    def __call__(self, shape, dtype=jnp.float32):
        assert len(shape) == 2
        fan_in, fan_out = shape
        # uniformly distributed orthogonal weight matrix
        w_init = self.orth_init(shape, dtype)
        w_init *= jnp.sqrt(self.scale / (max(1., (fan_in + fan_out))
                                         * jnp.var(w_init)))
        return w_init


class RBFFrequencyInitializer(hk.initializers.Initializer):
    """Initializer of the frequencies of the RadialBesselLayer.

     Initializes the  frequency values to its canonical values.
     """

    def __call__(self, shape, dtype):
        return jnp.pi * jnp.arange(1, shape[0] + 1, dtype=dtype)


# DimeNet++ Layers

class SmoothingEnvelope(hk.Module):
    """Smoothing envelope function for radial edge embeddings.

    Smoothing the cut-off enables twice continuous differentiability of the
    model output, including the potential energy.
    The envelope function is 1 at 0 and has a root of multiplicity of 3 at 1 as
    defined in DimeNet. It is applied to scaled radial edge distances
    d_ij / cut_off [0, 1].

    The implementation corresponds to the definition in the DimeNet paper.
    It is different from the original implementation of DimeNet / DimeNet++ that
    define incorrect spherical basis layers as a result (a known issue).
    """

    def __init__(self, p=6, name='Envelope'):
        """Initializes the SmoothingEnvelope layer.

        Args:
            p: Power of the smoothing polynomial
            name: Name of the layer
        """
        super().__init__(name=name)
        self._p = p
        self._a = -(p + 1.) * (p + 2.) / 2.
        self._b = p * (p + 2.)
        self._c = -p * (p + 1.) / 2.

    def __call__(self, inputs):
        """Returns the envelope values."""
        envelope_val = (1. + self._a * inputs**self._p
                        + self._b * inputs**(self._p + 1.)
                        + self._c * inputs**(self._p + 2.))
        return jnp.where(inputs < 1., envelope_val, 0.)


class RadialBesselLayer(hk.Module):
    """Radial Bessel Function (RBF) representation of pairwise distances.

    Attributes:
        freq_init: RBFFrequencyInitializer
    """
    def __init__(self, cutoff, num_radial=16, envelope_p=6,
                 name='BesselRadial'):
        """Initializes the RBF layer.

        Args:
            cutoff: Radial cut-off distance
            num_radial: Number of radial Bessel embedding functions
            envelope_p: Power of envelope polynomial
            name: Name of RBF layer
        """
        super().__init__(name=name)
        self._inv_cutoff = 1. / cutoff
        self._envelope = SmoothingEnvelope(p=envelope_p)
        self._num_radial = [num_radial]
        self._rbf_scale = jnp.sqrt(2. / cutoff)
        self.freq_init = RBFFrequencyInitializer()

    def __call__(self, distances):
        """Returns the RBF embeddings of edge distances."""
        distances = jnp.expand_dims(distances, -1)  # to broadcast to num_radial
        scaled_distances = distances * self._inv_cutoff
        envelope_vals = self._envelope(scaled_distances)
        frequencies = hk.get_parameter('RBF_Frequencies',
                                       shape=self._num_radial,
                                       init=self.freq_init)
        rbf_vals = (self._rbf_scale * jnp.sin(frequencies * scaled_distances)
                    / distances)
        return envelope_vals * rbf_vals


class SphericalBesselLayer(hk.Module):
    """Spherical Bessel Function (SBF) representation of angular triplets."""
    def __init__(self, r_cutoff, num_spherical, num_radial, envelope_p=6,
                 name='BesselSpherical'):
        """Initializes the SBF layer.

        Args:
            r_cutoff: Radial cut-off
            num_spherical: Number of spherical Bessel embedding functions
            num_radial: Number of radial Bessel embedding functions
            envelope_p: Power of envelope polynomial
            name: Name of SBF layer
        """
        super().__init__(name=name)

        assert num_spherical > 1
        self._envelope = SmoothingEnvelope(p=envelope_p)
        self._inv_cutoff = 1. / r_cutoff
        self._num_radial = num_radial

        bessel_formulars = dimenet_basis_util.bessel_basis(num_spherical,
                                                           num_radial)
        sph_harmonic_formulas = dimenet_basis_util.real_sph_harm(num_spherical)
        self._sph_funcs = []
        self._radual_bessel_funcs = []
        # convert sympy functions: modules overrides them by jax.numpy functions
        x = symbols('x')
        theta = symbols('theta')
        first_sph = utilities.lambdify([theta],  # pyling-ignore
                                       sph_harmonic_formulas[0][0],
                                       modules=[jnp, jsp.special])(0)
        for i in range(num_spherical):
            if i == 0:
                self._sph_funcs.append(lambda a: jnp.zeros_like(a) + first_sph)
            else:
                self._sph_funcs.append(
                    utilities.lambdify([theta], sph_harmonic_formulas[i][0],
                                       modules=[jnp, jsp.special])
                )
            for j in range(num_radial):
                self._radual_bessel_funcs.append(
                    utilities.lambdify([x], bessel_formulars[i][j],
                                       modules=[jnp, jsp.special])
                )

    def __call__(self, pair_distances, angles, angular_connectivity):
        """Returns the SBF embeddings of angular triplets."""
        angle_mask, _, expand_to_kj = angular_connectivity

        # initialize distances and envelope values
        scaled_distances = pair_distances * self._inv_cutoff
        envelope_vals = self._envelope(scaled_distances)
        envelope_vals = jnp.expand_dims(envelope_vals, -1)  # broadcast to rbf

        # compute radial bessel envelope for distances kj
        rbf = [radial_bessel(scaled_distances)
               for radial_bessel in self._radual_bessel_funcs]
        rbf = jnp.stack(rbf, axis=1)
        rbf_envelope = rbf * envelope_vals
        rbf_env_expanded = rbf_envelope[expand_to_kj]

        # compute spherical bessel embedding
        sbf = [spherical_bessel(angles) for spherical_bessel in self._sph_funcs]
        sbf = jnp.stack(sbf, axis=1)
        sbf = jnp.repeat(sbf, self._num_radial, axis=1)
        angle_mask = jnp.expand_dims(angle_mask, -1)
        sbf *= angle_mask  # mask non-existing triplets

        return rbf_env_expanded * sbf  # combine radial, spherical and envelope


class ResidualLayer(hk.Module):
    """Residual Layer: 2 activated Linear layers and a skip connection."""
    def __init__(self, layer_size, activation=nn.swish, init_kwargs=None,
                 name='ResLayer'):
        """Initializes the Residual layer.

        Args:
            layer_size: Output size of the Linear layers
            activation: Activation function
            init_kwargs: Dict of initialization kwargs for Linear layers
            name: Name of the Residual layer
        """
        super().__init__(name=name)
        self._residual = hk.Sequential([
            hk.Linear(layer_size, name='ResidualFirstLinear', **init_kwargs),
            activation,
            hk.Linear(layer_size, name='ResidualSecondLinear', **init_kwargs),
            activation
        ])

    def __call__(self, inputs):
        """Returns the ouput of the Residual layer."""
        out = inputs + self._residual(inputs)
        return out


class EmbeddingBlock(hk.Module):
    """Embeddimg block of DimeNet.

    Embeds edges by concattenating RBF embeddings with atom type embeddings of
    both connected atoms. If the network is defined to be kbT-dependent,
    adds a temperature embedding.
    """
    def __init__(self, embed_size, n_species, type_embed_size=None,
                 activation=nn.swish, init_kwargs=None, kbt_dependent=False,
                 name='Embedding'):
        """Initializes an Embedding block.

        Args:
            embed_size: Size of the edge embedding.
            n_species: Number of different atom species the network is supposed
                       to process.
            type_embed_size: Embedding size of atom type embedding. Default None
                             results in embed_size / 2.
            activation: Activation function
            init_kwargs: Dict of initialization kwargs for Linear layers
            kbt_dependent: Boolean, whether network prediction should depend on
                           temperature.
            name: Name of Embedding block
        """
        super().__init__(name=name)

        if type_embed_size is None:
            type_embed_size = int(embed_size / 2)

        embed_init = hk.initializers.RandomUniform(minval=-jnp.sqrt(3),
                                                   maxval=jnp.sqrt(3))
        self._embedding_vect = hk.get_parameter(
            'Embedding_vect', [n_species, type_embed_size], init=embed_init)
        self._kbt_dependent = kbt_dependent
        if kbt_dependent:
            self._kbt_embedding = hk.get_parameter(
                'Embedding_kbt', [1, type_embed_size], init=embed_init)

        # unlike the original DimeNet implementation, there is no activation
        # and bias in RBF_Dense as shown in the network sketch. This is
        # consistent with other Layers processing rbf values in the network
        self._rbf_dense = hk.Linear(embed_size, name='RBF_Dense',
                                    with_bias=False, **init_kwargs)
        self._dense_after_concat = hk.Sequential(
            [hk.Linear(embed_size, name='Concat_Dense', **init_kwargs),
             activation]
        )

    def __call__(self, rbf, species, pair_connectivity, **kwargs):
        """Returns output of the Embedding block."""
        idx_i, idx_j = pair_connectivity
        transformed_rbf = self._rbf_dense(rbf)

        type_i = species[idx_i]
        type_j = species[idx_j]

        h_i = self._embedding_vect[type_i]
        h_j = self._embedding_vect[type_j]

        edge_embedding = jnp.concatenate([h_i, h_j, transformed_rbf], axis=-1)
        if self._kbt_dependent:
            assert 'kT' in kwargs, ('If potential should be kbt-dependent. '
                                    '"kT" needs to be provided as kwarg.')
            kbt_embeds = jnp.tile(self._kbt_embedding, (h_j.shape[0], 1))
            kbt_embeds /= kwargs['kT']
            edge_embedding = jnp.concatenate([edge_embedding, kbt_embeds],
                                             axis=-1)
        embedded_messages = self._dense_after_concat(edge_embedding)
        return embedded_messages


class OutputBlock(hk.Module):
    """DimeNet++ Output block.

    Predicts per-atom quantities given RBF embeddings and messages.
    """
    def __init__(self, embed_size, n_particles, out_embed_size=None,
                 num_dense=3, num_targets=1, activation=nn.swish,
                 init_kwargs=None, name='Output'):
        """Initializes an Output block.

        Args:
            embed_size: Size of the edge embedding.
            n_particles: Number of particles in the graph
            out_embed_size: Output size of Linear layers after upsampling
            num_dense: Number of dense layers
            num_targets: Number of target quantities to be predicted
            activation: Activation function
            init_kwargs: Dict of initialization kwargs for Linear layers
            name: Name of Output block
        """
        super().__init__(name=name)
        if out_embed_size is None:
            out_embed_size = int(2 * embed_size)

        self._n_particles = n_particles
        self._rbf_dense = hk.Linear(embed_size, with_bias=False,
                                    name='RBF_Dense', **init_kwargs)
        self._upprojection = hk.Linear(out_embed_size, with_bias=False,
                                       name='Upprojection', **init_kwargs)

        # transform summed messages via multiple dense layers before predicting
        # target quantities
        self._dense_layers = []
        for _ in range(num_dense):
            self._dense_layers.append(hk.Sequential([
                hk.Linear(out_embed_size, with_bias=True, name='Dense_Series',
                          **init_kwargs), activation])
            )

        self._dense_final = hk.Linear(num_targets, with_bias=False,
                                      name='Final_output', **init_kwargs)

    def __call__(self, messages, rbf, pair_connectivity):
        """Returns predicted per-atom quantities."""
        idx_i, _ = pair_connectivity
        transformed_rbf = self._rbf_dense(rbf)
        # rbf is masked correctly and transformation only via weights
        # Hence rbf acts as mask
        messages *= transformed_rbf

        # sum incoming messages for each atom: becomes a per-atom quantity
        summed_messages = util.high_precision_segment_sum(
            messages, idx_i, num_segments=self._n_particles)

        upsampled_messages = self._upprojection(summed_messages)
        for dense_layer in self._dense_layers:
            upsampled_messages = dense_layer(upsampled_messages)

        per_atom_targets = self._dense_final(upsampled_messages)
        return per_atom_targets


class InteractionBlock(hk.Module):
    """DimeNet++ Interaction block.

    Performs directional message-passing based on RBF and SBF embeddings as well
    as messages from the previous message-passing iteration. Updated messages
    are used in the subsequent Output block.
    """
    def __init__(self, embed_size, num_res_before_skip, num_res_after_skip,
                 activation=nn.swish, init_kwargs=None,
                 angle_int_embed_size=None, basis_int_embed_size=8,
                 name='Interaction'):
        """Initializes an Interaction block.

        Args:
            embed_size: Size of the edge embedding.
            num_res_before_skip: Number of Residual blocks before skip
            num_res_after_skip: Number of Residual blocks after skip
            activation: Activation function
            init_kwargs: Dict of initialization kwargs for Linear layers
            angle_int_embed_size: Embedding size of Linear layers for
                                  down-projected triplet interation
            basis_int_embed_size: Embedding size of Linear layers for interation
                                  of RBS/ SBF basis
            name: Name of Interaction block
        """
        super().__init__(name=name)
        if angle_int_embed_size is None:
            angle_int_embed_size = int(embed_size / 2)

        # directional message passing block
        self._rbf1 = hk.Linear(basis_int_embed_size, name='rbf1',
                               with_bias=False, **init_kwargs)
        self._rbf2 = hk.Linear(embed_size, name='rbf2',
                               with_bias=False, **init_kwargs)
        self._sbf1 = hk.Linear(basis_int_embed_size, name='sbf1',
                               with_bias=False, **init_kwargs)
        self._sbf2 = hk.Linear(angle_int_embed_size, name='sbf2',
                               with_bias=False, **init_kwargs)

        self._dense_kj = hk.Sequential([
            hk.Linear(embed_size, name='Dense_kj', **init_kwargs), activation]
        )
        self._down_projection = hk.Sequential([
            hk.Linear(angle_int_embed_size, name='Downprojection',
                      with_bias=False, **init_kwargs), activation]
        )
        self._up_projection = hk.Sequential([
            hk.Linear(embed_size, name='Upprojection', with_bias=False,
                      **init_kwargs), activation]
        )

        # propagation block:
        self._dense_ji = hk.Sequential(
            [hk.Linear(embed_size, name='Dense_ji', **init_kwargs), activation]
        )

        self._res_before_skip = []
        for _ in range(num_res_before_skip):
            self._res_before_skip.append(ResidualLayer(
                embed_size, activation, init_kwargs, name='ResLayerBeforeSkip')
            )
        self._final_before_skip = hk.Sequential([hk.Linear(
            embed_size, name='FinalBeforeSkip', **init_kwargs), activation]
        )

        self._res_after_skip = []
        for _ in range(num_res_after_skip):
            self._res_after_skip.append(ResidualLayer(
                embed_size, activation, init_kwargs, name='ResLayerAfterSkip')
            )

    def __call__(self, m_input, rbf, sbf, angular_connectivity):
        # directional message passing block:
        _, reduce_to_ji, expand_to_kj = angular_connectivity
        m_ji_angular = self._dense_kj(m_input)  # messages for expansion to k->j
        rbf = self._rbf1(rbf)
        rbf = self._rbf2(rbf)
        m_ji_angular *= rbf

        m_ji_angular = self._down_projection(m_ji_angular)
        m_kj = m_ji_angular[expand_to_kj]  # expand to nodes k connecting to j

        sbf = self._sbf1(sbf)
        sbf = self._sbf2(sbf)
        # automatic mask: sbf was masked during initial computation.
        # Sbf1 and 2 only weights, no biases
        m_kj *= sbf

        aggregated_m_ji = util.high_precision_segment_sum(
            m_kj, reduce_to_ji, num_segments=m_input.shape[0])
        propagated_messages = self._up_projection(aggregated_m_ji)

        # add directional messages to original ones:
        # afterwards only independent edge transformations.
        # masking is lost, but rbf masks in output layer before aggregation
        m_ji = self._dense_ji(m_input)
        m_combined = m_ji + propagated_messages

        for layer in self._res_before_skip:
            m_combined = layer(m_combined)
        m_combined = self._final_before_skip(m_combined)

        m_ji_with_skip = m_combined + m_input

        for layer in self._res_after_skip:
            m_ji_with_skip = layer(m_ji_with_skip)
        return m_ji_with_skip


class DimeNetPP(hk.Module):
    """DimeNet++ for molecular property prediction.

    This model takes as input a sparse representation of a molecular graph
    - consisting of pairwise distances and angular triplets - and predicts
    per-atom properties. Global properties can be obtained by summing over
    per-atom predictions.

    Non-existing edges from fixed array size requirement are masked implicitly
    via the RBF envelope function. Hence, masked edges are assumed to be set to
    a distance > cut-off. Non-existing triplets are masked explicitly in the SBF
    embedding layer.

    This custom implementation follows the original DimeNet / DimeNet++
    (https://arxiv.org/abs/2011.14115), while correcting for known issues
    (see https://github.com/klicperajo/dimenet).
    """
    def __init__(self,
                 r_cutoff: float,
                 n_species: int,
                 n_particles: int,
                 num_targets: int = 1,
                 kbt_dependent: bool = False,
                 embed_size: int = 128,
                 n_interaction_blocks: int = 4,
                 num_residual_before_skip: int = 1,
                 num_residual_after_skip: int = 2,
                 out_embed_size: int = None,
                 type_embed_size: int = None,
                 angle_int_embed_size: int = None,
                 basis_int_embed_size: int = 8,
                 num_dense_out: int = 3,
                 num_rbf: int = 6,
                 num_sbf: int = 7,
                 activation: Callable = nn.swish,
                 envelope_p: int = 6,
                 init_kwargs: Dict[str, Any] = None,
                 name: str = 'DimeNetPP'):
        """Initializes the DimeNet++ model

        The default values correspond to the orinal values of DimeNet++.

        Args:
            r_cutoff: Radial cut-off distance of edges
            n_species: Number of different atom species the network is supposed
                       to process.
            n_particles: TODO
            num_targets: Number of different atomic properties to predict
            kbt_dependent: True, if DimeNet explicitly depends on temperature.
                           In this case 'kT' needs to be provided as a kwarg
                           during the model call to the energy_fn. Default False
                           results in a model independent of temperature.
            embed_size: Size of message embeddings. Scale interaction and output
                        embedding sizes accordingly, if not specified
                        explicitly.
            n_interaction_blocks: Number of interaction blocks
            num_residual_before_skip: Number of residual blocks before the skip
                                      connection in the Interaction block.
            num_residual_after_skip: Number of residual blocks after the skip
                                     connection in the Interaction block.
            out_embed_size: Embedding size of output block.
                            If None is set to 2 * embed_size.
            type_embed_size: Embedding size of atom type embeddings.
                             If None is set to 0.5 * embed_size.
            angle_int_embed_size: Embedding size of Linear layers for
                                  down-projected triplet interation.
                                  If None is 0.5 * embed_size.
            basis_int_embed_size: Embedding size of Linear layers for interation
                                  of RBS/ SBF basis in interaction block
            num_dense_out: Number of final Linear layers in output block
            num_rbf: Number of radial Bessel embedding functions
            num_sbf: Number of spherical Bessel embedding functions
            activation: Activation function
            envelope_p: Power of envelope polynomial
            init_kwargs: Kwargs for initializaion of Linear layers
            name: Name of DimeNet++ model
        """
        super().__init__(name=name)
        # input representation:
        self._rbf_layer = RadialBesselLayer(r_cutoff, num_rbf, envelope_p)
        self._sbf_layer = SphericalBesselLayer(r_cutoff, num_sbf, num_rbf,
                                               envelope_p)

        # build GNN structure
        self._n_interactions = n_interaction_blocks
        self._output_blocks = []
        self._int_blocks = []
        self._embedding_layer = EmbeddingBlock(
            embed_size, n_species, type_embed_size, activation, init_kwargs,
            kbt_dependent)
        self._output_blocks.append(OutputBlock(
            embed_size, n_particles, out_embed_size, num_dense_out,
            num_targets, activation, init_kwargs)
        )

        for _ in range(n_interaction_blocks):
            self._int_blocks.append(InteractionBlock(
                embed_size, num_residual_before_skip, num_residual_after_skip,
                activation, init_kwargs, angle_int_embed_size,
                basis_int_embed_size)
            )
            self._output_blocks.append(OutputBlock(
                embed_size, n_particles, out_embed_size, num_dense_out,
                num_targets, activation, init_kwargs)
            )

    def __call__(self,
                 distances: jnp.ndarray,
                 angles: jnp.ndarray,
                 species: jnp.ndarray,
                 pair_connections: Tuple[jnp.ndarray, jnp.ndarray],
                 angular_connections: Tuple[jnp.ndarray, jnp.ndarray,
                                            jnp.ndarray],
                 **dyn_kwargs) -> jnp.ndarray:
        """Predicts per-atom quantities for a given molecular graph.

        If edges and triplets are supposted to be masked, beware the different
        masking conventions: Edges are masked implicitly. This implementation
        assumes that a masked edge has a distance > cut-off, such that the RBF
        layer automatically masks the edge. By contrast, as triplets cannot be
        masked in an analogous way by the SBF layer, an explicit triplet mask
        array needs to be provided as part of 'angular_connections'.

        Args:
            distances: A (n_edges,) array storing for each edge the
                       corresponding distance between 2 particles
            angles: A (n_angles,) array storing for each triplet the
                    corresponding angle between the 3 particles
            species: A (n_particles) array storing the atom type of each paticle
            pair_connections: A tuple (idx_i, idx_j) of (n_edges,) arrays
                              storing for each edge the particle ID if connected
                              particles i and j.
            angular_connections: A tuple (angle_mask, reduce_to_ji,
                                 expand_to_kj) of (n_angles,) arrays. angle_mask
                                 stores for each triplet if it is real (True) or
                                 not (False). reduce_to_ji stores for each
                                 triplet kji edge index j->i to aggregate
                                 messages via a segment_sum. expand_to_kj stores
                                 for all triplets kji edge index k->j to gather
                                 all incoming edges for message passing.
            **dyn_kwargs: Kwargs supplied on-the-fly, uch as 'kT' for
                          temperature-dependent models.

        Returns:
            An (n_partciles, num_targets) array of predicted per-atom quantities
        """
        # TODO replace num_particles input by size of species
        # correctly masked (rbf=0) by construction if edge distance > cut-off:
        rbf = self._rbf_layer(distances)
        # explicitly masked via mask array in angular_connections
        sbf = self._sbf_layer(distances, angles, angular_connections)

        messages = self._embedding_layer(rbf, species, pair_connections,
                                         **dyn_kwargs)
        per_atom_quantities = self._output_blocks[0](messages, rbf,
                                                     pair_connections)

        for i in range(self._n_interactions):
            messages = self._int_blocks[i](messages, rbf, sbf,
                                           angular_connections)
            per_atom_quantities += self._output_blocks[i + 1](messages, rbf,
                                                              pair_connections)
        return per_atom_quantities


class PairwiseNNEnergy(hk.Module):
    """A neural network predicting the potential energy from pairwise
     interactions.
     """
    def __init__(self,
                 r_cutoff: float,
                 hidden_layers,
                 init_kwargs,
                 activation=nn.swish,
                 num_rbf: int = 6,
                 envelope_p: int = 6,
                 name: str = 'PairNN'):
        super().__init__(name=name)
        self.embedding = RadialBesselLayer(r_cutoff, num_radial=num_rbf,
                                           envelope_p=envelope_p)
        self.pair_nn = hk.nets.MLP(hidden_layers, activation=activation,
                                   **init_kwargs)
        self.rbf_transform = hk.Linear(1, with_bias=False, name='RBF_Transform',
                                       **init_kwargs)

    def __call__(self, distances, species=None, **kwargs):
        # ensure differentiability construction: rbf=0 for r > r_cut
        rbf = self.embedding(distances)
        predicted_energies = self.pair_nn(rbf)

        # rbf_transform has no bias: masked pairs remain 0 and counteract
        # possible non-zero contribution from biases in MPL (in a continuously
        # differentiable manner)
        per_pair_energy = predicted_energies * self.rbf_transform(rbf)
        # pairs are counted twice
        total_energy = jax_md_util.high_precision_sum(per_pair_energy) / 2.
        return total_energy
