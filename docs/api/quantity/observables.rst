chemtrain.quantity.observables
===============================

This module contains methods to compute ensemble quantities given a trajectory
of instantaneous quantities.

Ensemble Averages
------------------

Ensemble averages of instantaneous quantities, optionally via a reweighting
approach.

.. autofunction:: chemtrain.quantity.observables.init_traj_mean_fn

.. autofunction:: chemtrain.quantity.observables.init_linear_traj_mean_fn

Fluctuation Quantities
-----------------------

Comming soon...


State-Space Quantities
-----------------------

Quantities which are no ensemble averages.

.. autofunction:: chemtrain.quantity.observables.init_relative_entropy_traj_fn

Utility
--------

.. autofunction::chemtrain.quantity.observables.init_identity_fn
