# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Molecular property prediction."""
from functools import wraps
from typing import ClassVar, Tuple, Callable, Any

import haiku as hk
from jax import vmap, numpy as jnp
from jax_md import nn, util as jax_md_util

from chemtrain import sparse_graph, neural_networks, dropout


def build_dataset(targets, graph_dataset):
    """Builds dataset in format that is used for dataloading and throughout
    property predictions.

    Args:
        targets: Dict containing all targets to be predicted. Can be retrieved
                 in error_fn under the respective key.
        graph_dataset: Dataset of graphs, e.g. as obtained from
                       sparse_graph.convert_dataset_to_graphs.

    Returns:
        A dictionary containing the combined dataset and a list of target keys
    """
    target_keys = list(targets)
    target_keys.append('species_mask')
    return {**targets, **graph_dataset.to_dict()}, target_keys


def init_model(prediction_model):
    """Initializes a model that returns predictions for a single observation."""
    def mol_prediction_model(params, observation):
        graph = sparse_graph.SparseDirectionalGraph.from_dict(observation)
        predictions = prediction_model(params, graph)
        return predictions
    return mol_prediction_model


def init_loss_fn(error_fn):
    """Returns a loss function to optimize model parameters.

    Signature of error_fn:
    error = error_fn(predictions, batch, mask) ,
    where mask has the same shape as species to mask padded particles.

    Args:
        model: Molecular property prediction model (Haiku apply_fn).
        error_fn: Error model quantifying the discrepancy between preditions
                  and respective targets.
    """
    def loss_fn(predictions, batch):
        mask = jnp.ones_like(predictions) * batch['species_mask']
        return error_fn(predictions, batch, mask)
    return loss_fn


def per_species_results(species, per_atom_quantities, species_idxs):
    """Sorts per-atom results by species and returns a per-species mean.

    Only real (non-masked) particles should be input.

    Args:
        species: An array storing for each atom the corresponding species.
        per_atom_quantities: An array with the same shape as species, storing
                             per-atom quantities to be evaluated per-species.
        species_idxs: A (species,) array storing species-types for evaluation,
                      e.g. jnp.unique(species).

    Returns:
        A (species,) array of per-species quantities.
    """
    @vmap
    def process_single_species(species_idx):
        species_mask = (species == species_idx)
        species_members = jnp.count_nonzero(species_mask)
        screened_results = jnp.where(species_mask, per_atom_quantities, 0.)
        mean_if_species_exists = jnp.sum(screened_results) / species_members
        return jnp.where(species_members == 0, 0., mean_if_species_exists)
    return process_single_species(species_idxs)


def per_species_box_errors(dataset, per_atom_errors):
    """Computes for each snapshot in the provided graph dataset,
    the per-species error.

    Args:
        dataset: Graph dataset cantaining the snapshots of interest.
        per_atom_errors: Per-atom error for each atom in the dataset.
         Has same shape as dataset['species'].

    Returns:
        Mean per-species error for each snapshot in the dataset.
    """
    mask = dataset['species_mask']
    species = dataset['species']
    real_species = species[mask]
    unique_species = jnp.unique(real_species)
    species_masked = jnp.where(mask, species, 1000)  # species 1000 nonexistant
    per_box_and_species_fn = vmap(per_species_results, in_axes=(0, 0, None))
    per_box_species_errors = per_box_and_species_fn(
        species_masked, per_atom_errors, unique_species)
    distinct_per_box_species = jnp.sum(per_box_species_errors > 0., axis=1)
    mean_per_box_species_errors = (jnp.sum(per_box_species_errors, axis=1)
                                   / distinct_per_box_species)
    return mean_per_box_species_errors


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
        n_species: int = 100,
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
    return dropout.model_init_apply(property_predictor, model_kwargs)
