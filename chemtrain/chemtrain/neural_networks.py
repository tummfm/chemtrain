"""Some neural network models for potential energy and molecular property
 prediction.
 """
from functools import partial, wraps
from typing import Callable, Dict, Any, Tuple

import haiku as hk
from jax import lax, numpy as jnp, nn as jax_nn
from jax_md import smap, space, partition, nn, util

from chemtrain import layers, sparse_graph


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
                 num_targets: int,
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
                 activation: Callable = jax_nn.swish,
                 envelope_p: int = 6,
                 init_kwargs: Dict[str, Any] = None,
                 name: str = 'DimeNetPP'):
        """Initializes the DimeNet++ model

        The default values correspond to the orinal values of DimeNet++.

        Args:
            r_cutoff: Radial cut-off distance of edges
            n_species: Number of different atom species the network is supposed
                       to process.
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
        self._rbf_layer = layers.RadialBesselLayer(r_cutoff, num_rbf,
                                                   envelope_p)
        self._sbf_layer = layers.SphericalBesselLayer(r_cutoff, num_sbf,
                                                      num_rbf, envelope_p)

        # build GNN structure
        self._n_interactions = n_interaction_blocks
        self._output_blocks = []
        self._int_blocks = []
        self._embedding_layer = layers.EmbeddingBlock(
            embed_size, n_species, type_embed_size, activation, init_kwargs,
            kbt_dependent)
        self._output_blocks.append(layers.OutputBlock(
            embed_size, out_embed_size, num_dense_out, num_targets, activation,
            init_kwargs)
        )

        for _ in range(n_interaction_blocks):
            self._int_blocks.append(layers.InteractionBlock(
                embed_size, num_residual_before_skip, num_residual_after_skip,
                activation, init_kwargs, angle_int_embed_size,
                basis_int_embed_size)
            )
            self._output_blocks.append(layers.OutputBlock(
                embed_size, out_embed_size, num_dense_out, num_targets,
                activation, init_kwargs)
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
            species: A (n_particles,) array storing the atom type of each
                     particle
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
        n_particles = species.size
        # correctly masked (rbf=0) by construction if edge distance > cut-off:
        rbf = self._rbf_layer(distances)
        # explicitly masked via mask array in angular_connections
        sbf = self._sbf_layer(distances, angles, angular_connections)

        messages = self._embedding_layer(rbf, species, pair_connections,
                                         **dyn_kwargs)
        per_atom_quantities = self._output_blocks[0](messages, rbf,
                                                     pair_connections,
                                                     n_particles)

        for i in range(self._n_interactions):
            messages = self._int_blocks[i](messages, rbf, sbf,
                                           angular_connections)
            per_atom_quantities += self._output_blocks[i + 1](messages, rbf,
                                                              pair_connections,
                                                              n_particles)
        return per_atom_quantities


def dimenetpp_neighborlist(displacement: space.DisplacementFn,
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
                           ) -> Tuple[nn.InitFn, Callable[[Any, util.Array],
                                                          util.Array]]:
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
          'w_init': layers.OrthogonalVarianceScalingInit(scale=1.),
          'b_init': hk.initializers.Constant(0.),
        }
    r_cutoff = jnp.array(r_cutoff, dtype=util.f32)

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
        net = DimeNetPP(r_cutoff,
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


class PairwiseNNEnergy(hk.Module):
    """A neural network predicting the potential energy from pairwise
     interactions.
     """
    def __init__(self,
                 r_cutoff: float,
                 hidden_layers,
                 init_kwargs,
                 activation=jax_nn.swish,
                 num_rbf: int = 6,
                 envelope_p: int = 6,
                 name: str = 'PairNN'):
        super().__init__(name=name)
        self.embedding = layers.RadialBesselLayer(r_cutoff, num_radial=num_rbf,
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
        return per_pair_energy


def pair_interaction_nn(displacement: space.DisplacementFn,
                        r_cutoff: float,
                        hidden_layers,
                        activation=jax_nn.swish,
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
          'w_init': layers.OrthogonalVarianceScalingInit(scale=1.),
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

        net = PairwiseNNEnergy(r_cutoff, hidden_layers, init_kwargs, activation,
                               num_rbf, envelope_p)
        per_pair_energy = net(pair_distances, species, **dynamic_kwargs)
        # pairs are counted twice
        pot_energy = util.high_precision_sum(per_pair_energy) / 2.
        return pot_energy

    return model.init, model.apply


def molecular_property_predictor(model, n_per_atom=None):
    # TODO test and document

    @wraps(model)
    def property_wrapper(*args, **kwargs):
        per_atom_quantities = model(*args, **kwargs)

        if n_per_atom is None:  # all properties global by default
            return jnp.sum(per_atom_quantities, axis=0)
        else:
            n_predicted = per_atom_quantities.shape[1]
            n_global = n_predicted - n_per_atom
            per_atom_props = per_atom_quantities[:, n_global:]
            global_properties = jnp.sum(per_atom_quantities[:, :n_global],
                                        axis=0)
            return global_properties, per_atom_props
    return property_wrapper
