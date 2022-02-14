"""Customn function for enabling dropout applications in haiku."""

import haiku as hk
import jax.nn
from jax import random, vmap, jit, lax, numpy as jnp
from jax_md import quantity
from chemtrain import difftre, util

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
    return (meta_params['energy_params'],
            jnp.uint32(meta_params['Dropout_RNG_key']))


def next_dropout_params(meta_params):
    """Splits dropout_key and re-packes it in meta_params."""
    _, old_dropout_key = split_dropout_params(meta_params)
    new_dropout_key, _ = random.split(old_dropout_key, 2)
    return build_dropout_params(meta_params['energy_params'], new_dropout_key)


def dropout_is_used(meta_params):
    """A function that returns whether dropout is used by
    checking if the 'Dropout_RNG_key' is set or exists at all.
    """
    return 'Dropout_RNG_key' in meta_params.keys()


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
        self.shared = False  # TODO check which is most common mode

    def __call__(self, inputs, dropout_dict=None):
        linear_output = self.linear(inputs)
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
        num_res_before_skip: Number of residual blocks before the skip
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

        for layer in dropout_setup:
            dropout_setup[layer] = overall_dropout_rate

    return dropout_setup


def init_force_uq(energy_fn_template, n_splits=16, vmap_batch_size=1):
    n_devies = jax.device_count()
    util.assert_distribuatable(n_splits, n_devies, vmap_batch_size)

    @jit
    def forces(keys, energy_params, sim_state):

        def single_force(key):
            state, nbrs = sim_state  # assumes state and nbrs to be in sync
            dropout_params = build_dropout_params(energy_params, key)
            energy_fn = energy_fn_template(dropout_params)
            force_fn = quantity.canonicalize_force(energy_fn)
            return force_fn(state.position, neighbor=nbrs)

        # map in case not all necessary samples per device fit memory for vmap
        mapped_force = lax.map(single_force, keys)
        return mapped_force

    def force_uq(meta_params, sim_state):
        energy_params, key = split_dropout_params(meta_params)
        keys = random.split(key, n_splits)
        keys = keys.reshape((vmap_batch_size, -1, 2))  # 2 values per key
        # TODO add pmap
        # keys = keys.reshape((n_devies, vmap_batch_size, -1, 2))
        vmap_forces = vmap(forces, (0, None, None))
        batched_forces = vmap_forces(keys, energy_params, sim_state)
        shape = batched_forces.shape
        # reshape such that all sampled force predictions are along axis 0
        lined_forces = batched_forces.reshape((-1, shape[-2], shape[-1]))
        # TODO check that std and forces are correct
        f_std_per_atom = jnp.std(lined_forces, axis=0)
        mean_std = jnp.mean(f_std_per_atom)
        return mean_std

    return force_uq


def infer_output_uncertainty(meta_params, init_state, trajectory_generator,
                             quantities, neighbor_fn, total_samples,
                             kt_schedule=None, vmap_simulations_per_device=1):
    n_devies = jax.device_count()

    # Check whether dropout was used or not
    dropout_active = dropout_is_used(meta_params)
    # TODO add vmap
    # TODO add pmap

    if dropout_active:
        # If dropout is active (i.e. we need keys to map over)
        energy_params, key = split_dropout_params(meta_params)
        keys = random.split(key, total_samples)

        def single_prediction(key):
            params = build_dropout_params(energy_params, key)
            traj_state = trajectory_generator(params,
                                            init_state,
                                            kt_schedule=kt_schedule)
            quantity_traj = difftre.quantity_traj(traj_state, quantities,
                                                neighbor_fn, params)
            predictions = {}
            for quantity_key in quantities:
                quantity_snapshots = quantity_traj[quantity_key]
                predictions[quantity_key] = jnp.mean(quantity_snapshots, axis=0)
            return predictions
    else:
         # If dropout is not active (mapp over energy_params directly)
        def single_prediction(params):
            traj_state = trajectory_generator(params,
                                            init_state,
                                            kt_schedule=kt_schedule)
            quantity_traj = difftre.quantity_traj(traj_state, quantities,
                                                neighbor_fn, params)
            predictions = {}
            for quantity_key in quantities:
                quantity_snapshots = quantity_traj[quantity_key]
                predictions[quantity_key] = jnp.mean(quantity_snapshots, axis=0)
            return predictions

    # accumulates predictions in axis 0 of leaves of prediction_dict
    if dropout_active:
        predictions = lax.map(single_prediction, keys)
    else:
        predictions = lax.map(single_prediction, meta_params)
    return predictions


def mcmc_statistics(uq_predictions):
    statistics = {}
    for quantity_key in uq_predictions:
        quantity_samples = uq_predictions[quantity_key]
        statistics[quantity_key] = {'mean': jnp.mean(quantity_samples, axis=0),
                                    'std': jnp.std(quantity_samples, axis=0)}
    return statistics
