"""Molecular property prediction."""
from functools import wraps
from typing import ClassVar, Tuple, Callable, Any

import haiku as hk
from jax import vmap, numpy as jnp
from jax_md import nn, util as jax_md_util

from chemtrain import sparse_graph, util, neural_networks


def build_dataset(targets, graph_dataset):
    """Builds dataset in format that is used for dataloading and throughout
    property predictions.

    Args:
        targets: Dict containing all targets to be predicted. Can be retrieved
                 in error_fn under the respective key.
        graph_dataset: Dataset of graphs, e.g. as obtained from
                       sparse_graph.convert_dataset_to_graphs.

    Returns:
        A dictionary containing the combined dataset.
    """
    return {**targets, **graph_dataset.to_dict()}


def init_loss_fn(model, error_fn):
    """Returns a pmapped update function to optimize model parameters.

    Signature of error_fn:
    error = error_fn(predictions, batch, mask), where mask is a (batchsize,)
    boolean array.

    Args:
        model: Molecular property prediction model (Haiku apply_fn).
        error_fn: Error model quantifying the discrepancy between preditions
                  and respective targets.
    """
    def loss_fn(params, batch, mask=None):
        # for masking whole snapshots, e.g. when mapping over whole dataset
        if mask is None:
            mask = jnp.ones(util.tree_multiplicity(batch))
        graph = sparse_graph.SparseDirectionalGraph.from_dict(batch)
        predictions = vmap(model, in_axes=(None, 0))(params, graph)
        return error_fn(predictions, batch, mask)

    return loss_fn


def molecular_property_predictor(model, n_per_atom=0):
    """Wraps models that predict per-atom quantities to predict both global and
    per-atom quantities.

    Args:
        model: Initialized model predicting per-atom quantities, e.g. DimeNetPP.
        n_per_atom: Number of per-atom quantities to predict. Remaining
                    predictions are assumed to be global.

    Returns:
        A tuple (global_properties, per_atom_properties) being a (n_globals,)
        and a (n_particles, n_per_atom) array of predictions.
    """
    @wraps(model)
    def property_wrapper(*args, **kwargs):
        per_atom_quantities = model(*args, **kwargs)
        n_predicted = per_atom_quantities.shape[1]
        n_global = n_predicted - n_per_atom
        per_atom_properties = per_atom_quantities[:, n_global:]
        global_properties = jnp.sum(per_atom_quantities[:, :n_global], axis=0)
        return global_properties, per_atom_properties
    return property_wrapper


def partial_charge_prediction(
        r_cutoff: float,
        n_species: int = 10,
        model_class: ClassVar = neural_networks.DimeNetPP,
        **model_kwargs) -> Tuple[nn.InitFn, Callable[[Any, jax_md_util.Array],
                                                     jax_md_util.Array]]:
    """Initializes a model that predicts partial charges.

    Args:
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
                   to process.
        model_class: Haiku model class that predicts per-atom quantities.
                     By default, DimeNetPP.
        **model_kwargs: Kwargs to change the default structure of model_class.

    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an apply_function that predictions partial charges.
    """
    @hk.without_apply_rng
    @hk.transform
    def property_predictor(
            mol_graph: sparse_graph.SparseDirectionalGraph,
            **dynamic_kwargs):
        model = model_class(r_cutoff, n_species, 1, **model_kwargs)
        per_atom_predictions = model(mol_graph, **dynamic_kwargs)
        charges = per_atom_predictions[:, 0]
        # ensure sum of partial charges is neutral
        charge_correction = jnp.sum(charges) / mol_graph.n_particles
        partial_charges = (charges - charge_correction) * mol_graph.species_mask
        return partial_charges
    return property_predictor.init, property_predictor.apply
