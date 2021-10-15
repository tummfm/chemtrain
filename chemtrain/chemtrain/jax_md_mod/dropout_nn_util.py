"""Customn function for enabling dropout applications in haiku."""

# TODO: merge with custom_nn as soon as good implementation found

from typing import Dict, Any

import haiku as hk
import jax.nn
from jax import random, numpy as jnp
import jax.numpy as jnp
from jax_md import util

from chemtrain.jax_md_mod import custom_util
from chemtrain.jax_md_mod.custom_nn import RadialBesselLayer, \
    SphericalBesselLayer


# Note: This implementation currently stores the RNG key as float32
#  rather than uint32. This way, both energy_params and the RNG key
#  can be treated analogously in grad and optimizer of difftre.
#  Therefore, the RNG key is updated twice per parameter update:
#  By optimizer via (arbitrary) gradient and by the standard RNG update.
#  This allows a more unified treatment in difftre, without frequent
#  branching if dropout is used or not.
# TODO implement properly, where branching is done at optimizer init
#  and update as well as gradient computation

def split_dropout_params(meta_params):
    """Splitting up meta params, built up by energy_params and
    dropout_key.
    """
    return meta_params['energy_params'], jnp.uint32(meta_params['Dropout_RNG_key'])


def next_dropout_params(meta_params):
    """Splits dropout_key and re-packes it in meta_params."""
    _, old_dropout_key = split_dropout_params(meta_params)
    new_dropout_key, _ = random.split(old_dropout_key, 2)
    return build_dropout_params(meta_params['energy_params'], new_dropout_key)


def dropout_is_used(meta_params):
    """A function that returns whether dropout is used by
    checking if the 'Dropout_RNG_key' is set.
    """
    is_used = 'Dropout_RNG_key' in meta_params.keys()
    return is_used


def build_dropout_params(energy_params, dropout_key):
    """Combines meta_params, built up by energy_params and
    dropout_key.
    """

    return {'energy_params': energy_params,
            'Dropout_RNG_key': jnp.float32(dropout_key)}


def construct_dropout_dict(dropout_key, dropout_setup):
    """Splits and distributes the random key for all dropout_nn_util.Linear
    layers.
    """
    dropout_key_dict = {}
    if dropout_key is not None and len(dropout_setup) != 0:  # dropout
        n_keys = len(dropout_setup)
        split = random.split(dropout_key, n_keys)
        for i, (layer_name, do_rate) in enumerate(dropout_setup.items()):
            dropout_key_dict[layer_name] = {'key': split[i], 'do_rate': do_rate}

    return dropout_key_dict


class Linear(hk.Module):
    """Wrapper function for hk.Linear that applies dropout if
    name exists as key in dropout_dict during call.
    """
    def __init__(self, embed_size, name, with_bias=True, **init_kwargs):
        super().__init__(name=name)
        self.linear = hk.Linear(embed_size, with_bias=with_bias, **init_kwargs)
        # False: independent dropout of output features
        # True: same parts of feature vector are dropped for all edge embeddings
        self.shared = True  # TODO check which is most common mode

    def __call__(self, input, dropout_dict=None):
        linear_output = self.linear(input)
        if dropout_dict is None or self.module_name not in dropout_dict.keys():
            return linear_output
        else:  # apply dropout
            dropout_key = dropout_dict[self.module_name]['key']
            if self.shared:
                vectorized_dropout = jax.vmap(hk.dropout,
                                              in_axes=(None, None, 0))
                dropped_array = vectorized_dropout(dropout_key, dropout_dict[
                    self.module_name]['do_rate'], linear_output)
            else:
                dropped_array = hk.dropout(dropout_key, dropout_dict[
                    self.module_name]['do_rate'], linear_output)
            return dropped_array


def dimenetpp_dropout_setup(setup_dict,
                            num_dense_out,
                            n_interaction_blocks,
                            num_res_before_skip,
                            num_res_after_skip,
                            overall_dropout_rate=None):
    """
    Function that builds dropout_setup Dict containing Dropout
    hyperparameters for DimeNet++.

    Args:
        setup_dict: Dict containing the block name to be dropouted
                    alongside the target dropout rate for each block.
                    Available blocks are 'output', 'interaction' and
                    'embedding'.
        num_dense_out: Number of fully-connected layers in output block
        n_interaction_blocks: Number of interaction blocks
        num_res_before_skip: Number of residual blocks before the skip connection
        num_res_after_skip: Number of residual blocks after the skip connection
        overall_dropout_rate: If given, will override all droput rates
                              with this global rate.

    Returns: dropout_setup: Dict encoding dropout structure of DimeNet++
    """
    def index_to_layer_name(i):
        if i == 0:
            return ''
        else:
            return '_' + str(i)

    # These names are hardcoded to naming of layers in DimeNetPP
    # If you change layer names, make sure to adjust these
    delimiter = '/~/'
    net_prefix = 'Energy'
    output_prefix = 'Output'
    interaction_prefix = 'Interaction'
    embedding_prefix = 'Embedding'
    residual_prefix = 'ResLayer'
    output_layers = ['Dense_Series' + index_to_layer_name(i)
                     for i in range(num_dense_out)]
    output_layers.extend(['RBF_Dense', 'Upprojection'])
    interaction_layers = ['Dense_ji', 'Dense_kj', 'Downprojection',
                          'Upprojection', 'FinalBeforeSkip', 'rbf1', 'rbf2',
                          'sbf1', 'sbf2']
    residual_layers = ['ResidualSubLayer' + index_to_layer_name(i)
                       for i in range(2)]
    embedding_layers = ['Concat_Dense', 'RBF_Dense']

    if setup_dict is None:  # no dropout
        setup_dict = {}

    dropout_setup = {}
    if 'output' in setup_dict.keys():
        layer_prefix = net_prefix + delimiter + output_prefix
        droprate = setup_dict['output']

        for i_outblock in range(n_interaction_blocks + 1):
            i_interaction_prefix = layer_prefix + \
                                   index_to_layer_name(i_outblock) + delimiter
            for layer_name in output_layers:
                name = i_interaction_prefix + layer_name
                dropout_setup[name] = droprate

    if 'interaction' in setup_dict.keys():
        layer_prefix = net_prefix + delimiter + interaction_prefix
        droprate = setup_dict['interaction']

        for i_interaction in range(n_interaction_blocks):
            i_interaction_prefix = layer_prefix + \
                                   index_to_layer_name(i_interaction) \
                                   + delimiter
            for layer_name in interaction_layers:
                name = i_interaction_prefix + layer_name
                dropout_setup[name] = droprate
            for i_res_block in range(num_res_before_skip + num_res_after_skip):
                res_block_prefix = i_interaction_prefix + residual_prefix + \
                                   index_to_layer_name(i_res_block) + delimiter
                for res_layer_name in residual_layers:
                    name = res_block_prefix + res_layer_name
                    dropout_setup[name] = droprate

    if 'embedding' in setup_dict.keys():
        layer_prefix = net_prefix + delimiter + embedding_prefix
        droprate = setup_dict['embedding']
        for layer_name in embedding_layers:
            name = layer_prefix + delimiter + layer_name
            dropout_setup[name] = droprate

    if overall_dropout_rate is not None:  # override dropout rate
        for layer in dropout_setup.keys():
            dropout_setup[layer] = overall_dropout_rate

    return dropout_setup


class ResidualLayer_Dropout(hk.Module):
    def __init__(self, layer_size, activation=jax.nn.swish, init_kwargs=None, name='ResLayer'):
        super().__init__(name=name)

        self.residual = []
        for _ in range(2):
            self.residual.append(hk.Sequential([Linear(
                layer_size, name='ResidualSubLayer', **init_kwargs), activation]))

    def __call__(self, inputs, dropout_dict):
        out = inputs

        for residual_sub_layer in self.residual:
            out = residual_sub_layer(out, dropout_dict)

        out = inputs + out
        return out


class EmbeddingBlock_Dropout(hk.Module):
    def __init__(self, embed_size, n_species, type_embed_size=None, activation=jax.nn.swish,
                 init_kwargs=None, name='Embedding'):
        super().__init__(name=name,)

        if type_embed_size is None:
            type_embed_size = int(embed_size / 2)

        embed_init = hk.initializers.RandomUniform(minval=-jnp.sqrt(3), maxval=jnp.sqrt(3))
        self.embedding_vect = hk.get_parameter('Embedding_vect', [n_species, type_embed_size],
                                               init=embed_init, dtype=jnp.float32)

        # unlike the original DimeNet implementation, there is no activation and bias in RBF_Dense as shown in the
        # network sketch. This is consistent with other Layers processing rbf values throughout the network
        self.rbf_dense = Linear(embed_size, name='RBF_Dense', with_bias=False, **init_kwargs)
        self.dense_after_concat = hk.Sequential([Linear(embed_size, name='Concat_Dense', **init_kwargs), activation])

    def __call__(self, rbf, species, pair_connectivity, dropout_dict):
        idx_i, idx_j, _ = pair_connectivity
        transformed_rbf = self.rbf_dense(rbf, dropout_dict)

        type_i = species[idx_i]
        type_j = species[idx_j]

        h_i = self.embedding_vect[type_i]
        h_j = self.embedding_vect[type_j]

        edge_embedding = jnp.concatenate([h_i, h_j, transformed_rbf], axis=-1)

        embedded_messages = self.dense_after_concat(edge_embedding, dropout_dict)
        return embedded_messages


class OutputBlock_Dropout(hk.Module):
    def __init__(self, embed_size, n_particles, out_embed_size=None, num_dense=2, num_targets=1,
                 activation=jax.nn.swish, init_kwargs=None, name='Output'):
        super().__init__(name=name)

        if out_embed_size is None:
            out_embed_size = int(2 * embed_size)

        self.n_particles = n_particles
        self.rbf_dense = Linear(embed_size, with_bias=False, name='RBF_Dense', **init_kwargs)
        self.upprojection = Linear(out_embed_size, with_bias=False, name='Upprojection', **init_kwargs)

        # transform summed messages over multiple dense layers before predicting output
        self.dense_layers = []
        for _ in range(num_dense):
            self.dense_layers.append(hk.Sequential([Linear(
                out_embed_size, with_bias=True, name='Dense_Series', **init_kwargs), activation]))

        self.dense_final = hk.Linear(num_targets, with_bias=False, name='Final_output', **init_kwargs)

    def __call__(self, messages, rbf, connectivity, dropout_dict):
        idx_i, _, _ = connectivity
        transformed_rbf = self.rbf_dense(rbf, dropout_dict)
        messages *= transformed_rbf  # rbf is masked correctly, transformation only via weights --> rbf acts as mask

        # sum incoming messages for each atom: becomes a per-atom quantity
        summed_messages = custom_util.high_precision_segment_sum(messages, idx_i, num_segments=self.n_particles)

        upsampled_messages = self.upprojection(summed_messages, dropout_dict)

        for dense_layer in self.dense_layers:
            upsampled_messages = dense_layer(upsampled_messages, dropout_dict)

        per_atom_targets = self.dense_final(upsampled_messages)
        return per_atom_targets


class InteractionBlock_Dropout(hk.Module):
    def __init__(self, embed_size, num_res_before_skip, num_res_after_skip, activation=jax.nn.swish,
                 init_kwargs=None, angle_int_embed_size=None, basis_int_embed_size=8, name='Interaction'):
        super().__init__(name=name)

        if angle_int_embed_size is None:
            angle_int_embed_size = int(embed_size / 2)

        # directional message passing block
        self.rbf1 = Linear(basis_int_embed_size, name='rbf1', with_bias=False,  **init_kwargs)
        self.rbf2 = Linear(embed_size, name='rbf2', with_bias=False, **init_kwargs)
        self.sbf1 = Linear(basis_int_embed_size, name='sbf1', with_bias=False,  **init_kwargs)
        self.sbf2 = Linear(angle_int_embed_size, name='sbf2', with_bias=False, **init_kwargs)

        self.dense_kj = hk.Sequential([Linear(embed_size, name='Dense_kj', **init_kwargs), activation])
        self.down_projection = hk.Sequential([
            Linear(angle_int_embed_size, name='Downprojection', with_bias=False, **init_kwargs), activation])
        self.up_projection = hk.Sequential([
            Linear(embed_size, name='Upprojection', with_bias=False, **init_kwargs), activation])

        # propagation block:
        self.dense_ji = hk.Sequential([Linear(embed_size, name='Dense_ji', **init_kwargs), activation])

        self.res_before_skip = []
        for _ in range(num_res_before_skip):
            self.res_before_skip.append(ResidualLayer_Dropout(embed_size, activation, init_kwargs, name='ResLayer'))
        self.final_before_skip = hk.Sequential([Linear(
            embed_size, name='FinalBeforeSkip', **init_kwargs), activation])

        self.res_after_skip = []
        for _ in range(num_res_after_skip):
            self.res_after_skip.append(ResidualLayer_Dropout(embed_size, activation, init_kwargs, name='ResLayer'))

    def __call__(self, m_input, rbf, sbf, angular_connectivity, dropout_dict):
        # directional message passing block:
        _, reduce_to_ji, expand_to_kj = angular_connectivity

        m_ji_angular = self.dense_kj(m_input, dropout_dict) # transformed messages for expansion to k -> j

        rbf = self.rbf1(rbf, dropout_dict)
        rbf = self.rbf2(rbf, dropout_dict)
        m_ji_angular *= rbf

        m_ji_angular = self.down_projection(m_ji_angular, dropout_dict)
        m_kj = m_ji_angular[expand_to_kj]  # expand to nodes k connecting to j

        sbf = self.sbf1(sbf, dropout_dict)
        sbf = self.sbf2(sbf, dropout_dict)
        m_kj *= sbf  # automatic mask: sbf was masked during initial computation. Sbf1 and 2 only weights, no biases

        aggregated_m_ji = custom_util.high_precision_segment_sum(m_kj, reduce_to_ji, num_segments=m_input.shape[0])

        propagated_messages = self.up_projection(aggregated_m_ji, dropout_dict)

        # add directional messages to original ones; afterwards only independent edge transformations
        # masking is lost, but rbf masks in output layer before aggregation

        m_ji = self.dense_ji(m_input, dropout_dict) # transformed messages j -> i

        m_combined = m_ji + propagated_messages

        for layer in self.res_before_skip:
            m_combined = layer(m_combined, dropout_dict) # dropout happening in residual blocks, contrasting klicpera paper

        m_combined = self.final_before_skip(m_combined, dropout_dict)

        m_ji_with_skip = m_combined + m_input

        for layer in self.res_after_skip:
            m_ji_with_skip = layer(m_ji_with_skip, dropout_dict)
        return m_ji_with_skip


class DimeNetPPEnergy_Dropout(hk.Module):
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
                 dropout_setup = None,
                 name: str = 'Energy'):
        super(DimeNetPPEnergy_Dropout, self).__init__(name=name)

        self.dropout_setup = dropout_setup
        # input representation:
        self.rbf_layer = RadialBesselLayer(r_cutoff, num_radial=num_RBF, envelope_p=envelope_p)
        self.sbf_layer = SphericalBesselLayer(r_cutoff, num_radial=num_RBF, num_spherical=num_SBF, envelope_p=envelope_p)

        # build GNN structure
        self.n_interactions = n_interaction_blocks
        self.output_blocks = []
        self.int_blocks = []
        self.embedding_layer = EmbeddingBlock_Dropout(embed_size, n_species, type_embed_size=type_embed_size,
                                              activation=activation, init_kwargs=init_kwargs)
        self.output_blocks.append(OutputBlock_Dropout(embed_size, n_particles, out_embed_size, num_dense=num_dense_out,
                                              num_targets=1, activation=activation, init_kwargs=init_kwargs))

        for _ in range(n_interaction_blocks):
            self.int_blocks.append(InteractionBlock_Dropout(embed_size, num_residual_before_skip, num_residual_after_skip,
                                                    activation=activation, angle_int_embed_size=angle_int_embed_size,
                                                    basis_int_embed_size=basis_int_embed_size, init_kwargs=init_kwargs))
            self.output_blocks.append(OutputBlock_Dropout(embed_size, n_particles, out_embed_size, num_dense=num_dense_out,
                                                  num_targets=1, activation=activation, init_kwargs=init_kwargs))

    def __call__(self, distances, angles, species, pair_connections, angular_connections,
                 dropout_key) -> jnp.ndarray:
        rbf = self.rbf_layer(distances)  # correctly masked by construction: masked distances are 2 * cut-off --> rbf=0
        sbf = self.sbf_layer(distances, angles, angular_connections)  # is masked too

        dropout_key_dict = construct_dropout_dict(dropout_key,
                                                  self.dropout_setup)

        messages = self.embedding_layer(rbf, species, pair_connections, dropout_key_dict)
        per_atom_quantities = self.output_blocks[0](messages, rbf, pair_connections, dropout_key_dict)

        for i in range(self.n_interactions):
            messages = self.int_blocks[i](messages, rbf, sbf, angular_connections, dropout_key_dict)
            per_atom_quantities += self.output_blocks[i + 1](messages, rbf, pair_connections, dropout_key_dict)

        predicted_quantities = util.high_precision_sum(per_atom_quantities, axis=0)  # sum over all atoms
        return predicted_quantities