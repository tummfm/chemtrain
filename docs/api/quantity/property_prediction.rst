``chemtrain.quantity.property_prediction``
==========================================

.. currentmodule:: chemtrain.quantity.property_prediction

.. automodule:: chemtrain.quantity.property_prediction

This module contains methods for molecular property prediction,
which build on neural network architectures used for potential
energy prediction.

Molecular property predictor
-------------------------------

This wrapper function is at the core of the molecular property prediction
module by transforming models for (atom-wise) potential energy prediction to
molecular property predictors for atom and molecule level properties.

.. autofunction:: molecular_property_predictor

Examples
-------------------------------

We provide an example to transform DimeNet++ to a partial charge predictor,
which enforces charge neutrality of its predictions. For a real-world
application of this partial charge predictor in an active learning context,
see this `code <https://github.com/tummfm/mof-al>`_ of
`Thaler et al. (2024) <https://www.nature.com/articles/s41524-024-01277-8>`_.

.. autofunction:: partial_charge_prediction

Pre-processing
-------------------------------
Utility functions for initialization of the dataset and the model as well as
a masked loss function for masking virtual atoms that are added to conserve
array shapes.

.. autofunction:: build_dataset

.. autofunction:: init_model

.. autofunction:: init_loss_fn

Post-processing
-------------------------------
Property prediction at the atom level is often highly class-imbalanced.
To account for this imbalance, the following functions evaluate prediction
accuracy for each atom species:

.. autofunction:: per_species_results

.. autofunction:: per_species_box_errors
