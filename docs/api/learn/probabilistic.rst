``chemtrain.learn.probabilistic``
=================================

.. currentmodule:: chemtrain.learn.probabilistic

.. automodule:: chemtrain.learn.probabilistic

Bayesian UQ
-------------------------------

Defines the Bayesian UQ problem by building prior and likelihood functions as well as
combining them to prepare the problem in order to initialize an SGMCMC force matching
trainer.

.. autofunction:: uniform_prior

.. autofunction:: init_elementwise_prior_fn

.. autofunction:: init_likelihood

.. autofunction:: init_log_posterior_fn

.. autofunction:: init_force_matching

Dropout Monte Carlo
-------------------------------
Utility functions to perform forward UQ of a model trained via Dropout.


.. autofunction:: init_dropout_uq_fwd

.. autofunction:: dropout_uq_predictions

Propagate Uncertainty
-------------------------------

Propagate the uncertainty of probabilistic potentials to obtain UQ of MD observables.

.. autofunction:: uq_trajectories


UQ Postprocessing
-------------------------------

Utility functions to assess statistics of Markov chains and compute error metrics.

.. autofunction:: mcmc_statistics

.. autofunction:: validation_mae_params_fm

.. autofunction:: test_rmse_params_fm
