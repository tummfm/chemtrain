"""Customn function for enabling dropout applications in haiku."""

# TODO: merge with custom_nn as soon as good implementation found
# TODO: simplify hyperparameter dict
# TODO: implement that no dropout is done when dropout_key=None

from typing import Dict, Any

import haiku as hk
import jax.nn
from jax import random, numpy as jnp
import jax.numpy as jnp
from jax_md import util

from chemtrain.jax_md_mod import custom_util
from chemtrain.jax_md_mod.custom_nn import RadialBesselLayer, \
    SphericalBesselLayer


def split_dropout_params(meta_params):
    """Splitting up meta params, built up by energy_params and
    dropout_key. Built for costum_energy.py level, where the
    energy_params and dropout_key needs to be handled seperatly.

    Args:
        meta_params: Combining energy_params and dropout_key.

    Returns:
        energy_params: Model/net parameters upgrading energy_fn_template
                       to energy_fn.

        dropout_key: random.PRNGKey defining the dropout pattern for
                     one epoch.
    """
    dict_params = hk.data_structures.to_mutable_dict(meta_params)
    dropout_key = dict_params['Dropout_RNG_key']
    dict_params.pop('Dropout_RNG_key')
    energy_params = hk.data_structures.to_immutable_dict(dict_params)
    return energy_params, dropout_key


def next_dropout_params(meta_params):
    """Splitting dropout_key and re-packing it in meta params.

    Args:
        meta_params: Combining energy_params and dropout_key.

    Returns:
        params: Updated meta params, combining energy_params and
                updated dropout_key.
    """

    dict_params = hk.data_structures.to_mutable_dict(meta_params)
    old_dropout_key = dict_params['Dropout_RNG_key']
    old_dropout_key = jnp.uint32(old_dropout_key)

    new_dropout_key, subkey = random.split(old_dropout_key, 2)
    dict_params['Dropout_RNG_key'] = jnp.float32(new_dropout_key)

    new_params = hk.data_structures.to_immutable_dict(dict_params)
    return new_params


def dropout_is_used(meta_params):
    """A function that returns whether dropout is used by
    checking if the 'Dropout_RNG_key' is set.
    """
    dict_params = hk.data_structures.to_mutable_dict(meta_params)
    is_used = 'Dropout_RNG_key' in dict_params.keys()
    return is_used


def set_dropout_params(meta_params, dropout_seed):
    """Sets dropout_key of meta params to specific seed

    Args:
        meta_params: Combining energy_params and dropout_key.
        dropout_seed: Integer that will directly coverted into the PRNGKey.

    Returns:
        new_params: Set meta params, combining energy_params and the
                    set dropout_key.
    """

    dict_params = hk.data_structures.to_mutable_dict(meta_params)
    new_dropout_key = random.PRNGKey(dropout_seed)
    dict_params['Dropout_RNG_key'] = jnp.float32(new_dropout_key)
    new_params = hk.data_structures.to_immutable_dict(dict_params)
    return new_params


def fuse_dropout_params(energy_params, dropout_key):
    """Combining meta params, built up by energy_params and
    dropout_key.

    Args:
        energy_params: Model/net parameters as input to energy_fn_template.
        dropout_key: random.PRNGKey defining the dropout pattern for
                     one epoch.

    Returns:
        meta_params: Combining energy_params and dropout_key.
    """
    
    dict_params = hk.data_structures.to_mutable_dict(energy_params)
    dict_params['Dropout_RNG_key'] = dropout_key
    meta_params = hk.data_structures.to_immutable_dict(dict_params)
    return meta_params


def multikey(dropout_key, option, dropout_hp, block_type=None):
    """Helper function providing the dropout_key split on block
    and layer level.

    Distributing the key over different blocks, option 'blockwise' derives
    a dropout_multikey that suits the dropout hyperparameter.

    Args:
        dropout_key: Single key on the higher level that needs to be
                     distributed to a lower level.
        option: 'blockwise' key split on model (DimeNet) level, where for
                    every block there has to be a unique key
                'layerwise' key split on block (output/interaction/aso) level, 
                    where the unique block key needs to be distributed to
                    layers, e.g. Dense_Series, Dense_Series_1, Dense_ji...
        dropout_hp: Dropout hyperparameter containing block and layer
                    architecture of the structures to be dropped out on.
        block_type: In case option 'layerwise is selected' the detection
                    needs the corresponding block_type multikey beeing
                    used in. E.g. in case the single key inside the
                    interaction-call has to be split up, block_type has to
                    be 'interaction', so the dropout_hyperparameter is
                    processed correctly. 'None' is the standard config,
                    usable for splitting on the block level.

    Returns:
        dropout_multikey:   Dictionary of keys corresponding to the
                            layers/blocks mentioned in the dropout
                            hyperparameter.
    """
    
    if option == 'blockwise':
        dropout_multikey = {}
        for block_type in ['embedding', 'interaction', 'output', 'residual']:
            if block_type in dropout_hp.keys():
                num_blocks = len(dropout_hp[block_type]['layer_mapping']
                                 ['block_list'])
                dropout_key = jnp.uint32(dropout_key)
                split = random.split(dropout_key, num_blocks)
                dropout_multikey[block_type] = jnp.float32(split)
            else: 
                dropout_multikey[block_type] = None
        
    elif option == 'layerwise':
        dropout_multikey = {}
        if block_type in dropout_hp.keys():
            num = len(dropout_hp[block_type]['layer_mapping']['layer_list'])
            dropout_key = jnp.uint32(dropout_key)
            split = random.split(dropout_key, num)
            split = jnp.float32(split)
            for index, layer_type in enumerate(dropout_hp[block_type]
                                               ['layer_mapping']
                                               ['layer_list']):
                dropout_multikey[layer_type] = split[index]
        else: 
            dropout_multikey = None

    else:
        raise ValueError("Wrong multikey option", block_type)

    return dropout_multikey


def controlled_dropout(layer, inputs, dropout_hp, dropout_key, block_type):

    """Wrapper function for layer outputs combined with a targetted dropout.

    The for loop is only kept for potential multilayer execution.
    Right now, in the modus of single layer processing, the loop only
    walks through once with i_layer=0.
    This funtion is usually located in the call function of certain layers.
    
    Warning: Validity given only for single layer processing!!

    Args:
        layer:  Layer function initialized before in __init__
        inputs:  Input array that usually is inputted to the layer.
        dropout_hp: Dropout hyperparameter containing block and layer
                    architecture of the structures to be dropped out on.
        dropout_key:    Unique dropout key.
        block_type: Block type corresponding to the dropout hyperparameter
                    entry, enabling dropout in the specific block and
                    checking for the concrete layer.

    Returns:
        out:    In case dropout is triggered, the output is the dropped
                out and scaled layer output.
        info:   Debug feedback. 
                'True' if layer output is dropped out. 
                'False' if layer output is not dropped out.
    """
    
    # validity only for single layer processing!!
    raw_out = layer(inputs)

    # loop only for potential multilayer execution. Not yet implemented.
    # TODO is this really helpful, even if implemented?
    for i_layer in range(0, len(layer.layers), 2):

        # TODO should we check here or is this too much?
        # TODO where do we check for training is true?
        if block_type in dropout_hp.keys() and layer.layers[i_layer].name in \
                dropout_hp[block_type]['layer_mapping']['layer_list']:
            
            if i_layer > 0:
                raise NotImplementedError("There is a sequantial layer with "
                                          "the length > 2, more than one "
                                          "linear layer. This is not "
                                          "implemented yet. See: ", block_type)

            dropout_key = jnp.uint32(dropout_key)

            # TODO does this work without the dummy?
            num_out = raw_out.shape[1]
            dummy_array = jnp.ones((1, num_out), dtype=jnp.float32)
            do_array = hk.dropout(dropout_key, dropout_hp[block_type]
                                                         ['dropout_rate'],
                                  dummy_array)
            
            out = do_array * raw_out
            info = True
            assert out.shape == raw_out.shape

        else:
            out = raw_out
            info = False

    return out, info


def dimenetpp_dropout_setup(mode, overall_dropout_rate=None):
    """Setup function creating dropout preferences in different
    parts of the model.

    TODO Define different dropout settings for each model
    Different layers can have independent dropout rates. The
    layers and blocks where dropout should not be enabled
    have to be commented out or deleted.
    Currrently only defined only for DimeNet.
    The second key block_list in layer_mapping is not used right now.

    Args:
        mode:  Model string, e.g. tabulated or CGDimeNet
        overall_dropout_rate:   Overriding layer specific dropout rates
                                to a uniform one if not None.

    Returns:
        do_hp: Dictionary specifying dropout procedure per block and layer
    """
    if mode is None:
        mode = {}

    do_hp = {}
    if 'output' in mode:
        do_hp['output'] = {}
        do_hp['output']['dropout_rate'] = 0.2
        do_hp['output']['layer_mapping'] = {}

        do_hp['output']['layer_mapping']['net_prefix'] = 'Energy/~/'
        do_hp['output']['layer_mapping']['block_list'] = ['Output',
                                                          'Output_1',
                                                          'Output_2',
                                                          'Output_3',
                                                          'Output_4']
        do_hp['output']['layer_mapping']['layer_list'] = ['Dense_Series',
                                                          'Dense_Series_1',
                                                          'Dense_Series_2']
    if 'interaction' in mode:
        do_hp['interaction'] = {}
        do_hp['interaction']['dropout_rate'] = 0.2
        do_hp['interaction']['layer_mapping'] = {}

        do_hp['interaction']['layer_mapping']['net_prefix'] = 'Energy/~/'
        do_hp['interaction']['layer_mapping']['block_list'] = ['Interaction',
                                                               'Interaction_1',
                                                               'Interaction_2',
                                                               'Interaction_3']
        do_hp['interaction']['layer_mapping']['layer_list'] = ['Dense_ji',
                                                               'Dense_kj',
                                                               'Downprojection',
                                                               'Upprojection',
                                                               'FinalBeforeSkip']

    if 'embedding' in mode:
        do_hp['embedding'] = {}
        do_hp['embedding']['dropout_rate'] = 0.2
        do_hp['embedding']['layer_mapping'] = {}

        do_hp['embedding']['layer_mapping']['net_prefix'] = 'Energy/~/'
        do_hp['embedding']['layer_mapping']['block_list'] = ['Embedding/~/']
        do_hp['embedding']['layer_mapping']['layer_list'] = ['Concat_Dense']

    if 'residual' in mode:
        do_hp['residual'] = {}
        do_hp['residual']['dropout_rate'] = 0.3
        do_hp['residual']['layer_mapping'] = {}

        do_hp['residual']['layer_mapping']['net_prefix'] = 'Energy/~/'
        do_hp['residual']['layer_mapping']['block_list'] = ['','','','']
        do_hp['residual']['layer_mapping']['layer_list'] = ['ResidualSubLayer',
                                                            'ResidualSubLayer_1']

    # override dropout rate
    if overall_dropout_rate is not None:
        for block_type in do_hp.keys():
            do_hp[block_type]['dropout_rate'] = overall_dropout_rate

    return do_hp


class ResidualLayer_Dropout(hk.Module):
    def __init__(self, layer_size, activation=jax.nn.swish, init_kwargs=None, name='ResLayer'):
        super().__init__(name=name)

        self.residual = []
        for _ in range(2): # hardcoded number of layers
            self.residual.append(hk.Sequential([hk.Linear(
                layer_size, name='ResidualSubLayer', **init_kwargs), activation]))

    def __call__(self, inputs, dropout_hp, dropout_key_block):
        out = inputs

        dropout_multikey_layer = multikey(dropout_key_block,'layerwise',dropout_hp,block_type='residual')

        for residual_sub_layer in self.residual:
            dropout_key = None if dropout_multikey_layer is None else dropout_multikey_layer[residual_sub_layer.layers[0].name]
            out, info = controlled_dropout(residual_sub_layer,out,dropout_hp,dropout_key,'residual')

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
        self.rbf_dense = hk.Linear(embed_size, name='RBF_Dense', with_bias=False, **init_kwargs)
        self.dense_after_concat = hk.Sequential([hk.Linear(embed_size, name='Concat_Dense', **init_kwargs), activation])

    def __call__(self, rbf, species, pair_connectivity, dropout_hp, dropout_key_block):
        idx_i, idx_j, _ = pair_connectivity
        transformed_rbf = self.rbf_dense(rbf)

        type_i = species[idx_i]
        type_j = species[idx_j]

        h_i = self.embedding_vect[type_i]
        h_j = self.embedding_vect[type_j]

        edge_embedding = jnp.concatenate([h_i, h_j, transformed_rbf], axis=-1)

        embedded_messages, info = controlled_dropout(self.dense_after_concat,edge_embedding,dropout_hp,dropout_key_block,'embedding')
        return embedded_messages


class OutputBlock_Dropout(hk.Module):
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

    def __call__(self, messages, rbf, connectivity, dropout_hp, dropout_key_block):
        idx_i, _, _ = connectivity
        transformed_rbf = self.rbf_dense(rbf)
        messages *= transformed_rbf  # rbf is masked correctly, transformation only via weights --> rbf acts as mask

        # sum incoming messages for each atom: becomes a per-atom quantity
        summed_messages = custom_util.high_precision_segment_sum(messages, idx_i, num_segments=self.n_particles)

        upsampled_messages = self.upprojection(summed_messages)

        dropout_multikey_layer = multikey(dropout_key_block,'layerwise',dropout_hp,block_type='output')

        for dense_layer in self.dense_layers:
            dropout_key = None if dropout_multikey_layer is None else dropout_multikey_layer[dense_layer.layers[0].name]
            upsampled_messages, info = controlled_dropout(dense_layer,upsampled_messages,dropout_hp,dropout_key,'output')

        per_atom_targets = self.dense_final(upsampled_messages)
        return per_atom_targets


class InteractionBlock_Dropout(hk.Module):
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
            self.res_before_skip.append(ResidualLayer_Dropout(embed_size, activation, init_kwargs, name='ResLayerBeforeSkip'))
        self.final_before_skip = hk.Sequential([hk.Linear(
            embed_size, name='FinalBeforeSkip', **init_kwargs), activation])

        self.res_after_skip = []
        for _ in range(num_res_after_skip):
            self.res_after_skip.append(ResidualLayer_Dropout(embed_size, activation, init_kwargs, name='ResLayerAfterSkip'))

    def __call__(self, m_input, rbf, sbf, angular_connectivity, dropout_hp, dropout_key_block):
        # directional message passing block:
        _, reduce_to_ji, expand_to_kj = angular_connectivity

        dropout_multikey_layer = multikey(dropout_key_block,'layerwise',dropout_hp,block_type='interaction')

        dropout_key = None if dropout_multikey_layer is None else dropout_multikey_layer[self.dense_kj.layers[0].name]
        m_ji_angular, info = controlled_dropout(self.dense_kj,m_input,dropout_hp,dropout_key,'interaction') # transformed messages for expansion to k -> j

        rbf = self.rbf1(rbf)
        rbf = self.rbf2(rbf)
        m_ji_angular *= rbf

        dropout_key = None if dropout_multikey_layer is None else dropout_multikey_layer[self.down_projection.layers[0].name]
        m_ji_angular, info = controlled_dropout(self.down_projection,m_ji_angular,dropout_hp,dropout_key,'interaction')
        m_kj = m_ji_angular[expand_to_kj]  # expand to nodes k connecting to j

        sbf = self.sbf1(sbf)
        sbf = self.sbf2(sbf)
        m_kj *= sbf  # automatic mask: sbf was masked during initial computation. Sbf1 and 2 only weights, no biases

        aggregated_m_ji = custom_util.high_precision_segment_sum(m_kj, reduce_to_ji, num_segments=m_input.shape[0])

        dropout_key = None if dropout_multikey_layer is None else dropout_multikey_layer[self.up_projection.layers[0].name]
        propagated_messages, info = controlled_dropout(self.up_projection,aggregated_m_ji,dropout_hp,dropout_key,'interaction')

        # add directional messages to original ones; afterwards only independent edge transformations
        # masking is lost, but rbf masks in output layer before aggregation

        dropout_key = None if dropout_multikey_layer is None else dropout_multikey_layer[self.dense_ji.layers[0].name]
        m_ji, info = controlled_dropout(self.dense_ji,m_input,dropout_hp,dropout_key,'interaction') # transformed messages j -> i

        m_combined = m_ji + propagated_messages

        for layer in self.res_before_skip:
            m_combined = layer(m_combined,dropout_hp,dropout_key) # dropout happening in residual blocks, contrasting klicpera paper

        dropout_key = None if dropout_multikey_layer is None else dropout_multikey_layer[self.final_before_skip.layers[0].name]
        m_combined, info = controlled_dropout(self.final_before_skip,m_combined,dropout_hp,dropout_key,'interaction')

        m_ji_with_skip = m_combined + m_input

        for layer in self.res_after_skip:
            m_ji_with_skip = layer(m_ji_with_skip,dropout_hp,dropout_key)
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
                 dropout_hp = None,
                 name: str = 'Energy'):
        super(DimeNetPPEnergy_Dropout, self).__init__(name=name)

        # TODO only use self.dropout_hp to construct dict of keys!
        #  Choice for dropout can then be taken if layer name is in dict
        #  if not scip, otherwise this is place where correspoding
        #  sub-key is stored
        self.dropout_hp = dropout_hp
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

        if dropout_key is None:  # no dropout
            # TODO implement this!
            dropout_key_dict = {}
        else:
            dropout_multikey_blocks = multikey(dropout_key, 'blockwise', self.dropout_hp)

        messages = self.embedding_layer(rbf, species, pair_connections, self.dropout_hp, dropout_key)

        # TODO can we create key dict that is None if no dropout and otherwise
        #  the key? --> could create easy dropout distinguishing
        dropout_key = dropout_multikey_blocks['output'][0] if 'output' in self.dropout_hp.keys() else None
        per_atom_quantities = self.output_blocks[0](messages, rbf, pair_connections, self.dropout_hp, dropout_key)

        for i in range(self.n_interactions):
            dropout_key = dropout_multikey_blocks['interaction'][i] if 'interaction' in self.dropout_hp.keys() else None
            messages = self.int_blocks[i](messages, rbf, sbf, angular_connections, self.dropout_hp, dropout_key)

            dropout_key = dropout_multikey_blocks['output'][1 + i] if 'output' in self.dropout_hp.keys() else None
            per_atom_quantities += self.output_blocks[i + 1](messages, rbf, pair_connections, self.dropout_hp, dropout_key)

        predicted_quantities = util.high_precision_sum(per_atom_quantities, axis=0)  # sum over all atoms
        return predicted_quantities