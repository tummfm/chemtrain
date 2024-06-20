Chemtrain --- Machine-Learning Molecular Dynamics Potentials in JAX
====================================================================

**chemtrain** is a library to simplify machine learning of MD potentials.
It builds on and extends the JAXMD_ end-to-end differentiable MD framework.
These extensions are gathered in the
:doc:`chemtrain.jax_md_mod <jax_md_mod/index>` sub-module.

.. _JAXMD: https://github.com/jax-md/jax-md


**Features**

- `Differentiable Trajectory Reweighting <https://www.nature.com/articles/s41467-021-27241-4>`_
  (DiffTRe) to train on experimental data

- `Force Matching and Relative Entropy Minimization <https://doi.org/10.1063/5.0124538>`_
  for coarse-grained systems

**Future Extensions**

- Uncertainty Quantification via
  `Deep Ensembles and Stochastic gradient MCMC <https://doi.org/10.1021/acs.jctc.2c01267>`_
- Active Learning
- Hybrid Trainers to combine different training methods
- Molecular Property Prediction


.. toctree::
   :maxdepth: 2
   :caption: Algorithms

   algorithms/getting_started
   algorithms/difftre
   algorithms/force_matching
   algorithms/relative_entropy
   algorithms/prior_simulation


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Examples

   examples/CG_water_difftre
   examples/CG_alanine_dipeptide


.. toctree::
   :hidden:
   :caption: Chemtrain
   :maxdepth: 2

   api/index


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Jax, M.D. Extensions

   jax_md_mod/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
