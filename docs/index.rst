Documentation Overview
=======================================

Neural Networks are promising models for enhancing the accuracy of classical molecular
simulations. However, the training of accurate models is challenging.
**chemtrain** is a framework for learning sophisticated Neural Network potential
models by combining customizable training routines with advanced training
algorithms.
This combination enables the inclusion of high-quality reference data from
simulations and experiments and the lowering computational demands of training
through complementing algorithms with different advantages.

**chemtrain** is written in JAX, integrating with the differentiable MD engine
`JAX, M.D. <https://github.com/jax-md/jax-md>`_
Therefore, **chemtrain** leverages end-to-end differentiable
physics and hardware acceleration through GPUs to provide flexibility at scale.


Installation
=============

**chemtrain** can be installed with pip:

.. code-block:: shell

   pip install chemtrain --upgrade

The above command installs **JAX for CPU**.
Running **chemtrain on the GPU** requires the installation of a particular JAX
version.
Please follow the `JAX Installation Instructions <https://github.com/google/jax#installation>`_.

.. note::

   Chemtrain installs ``jax == 0.4.30`` which is, in principle, incompatible
   with ``jax_md <= 0.1.29`` but resolves an
   `XLA issue <https://github.com/google/jax/issues/17730>`_ which can prevent
   training.
   By important ``chemtrain`` or the ``jax_md_mod`` module
   **before importing** ``jax_md``, the compatibility is restored by a simple
   patch.


Advanced Installation
-----------------------


Additional Packages
____________________

Some parts of **chemtrain** require additional packages.
To install these, provide the `all` option.

.. code-block:: shell

   pip install 'chemtrain[all]' --upgrade


Installation from Source
_________________________

The lines below install **chemtrain** from source for development purposes.

.. code-block:: shell

   git clone git@github.com:tummfm/chemtrain.git
   pip install -e '.[all,docs,test]'

This command additionally installs the requirements to run the tests

.. code-block:: shell

   pytest ./tests

and to build the documentation (e.g., in html)

.. code-block:: shell

   make -C docs html

Getting Started
================

To get started with chemtrain and with the most important algorithms,
we provide simple toy examples.
These examples are simple to run on the CPU and sufficient to illustrate the basic
concepts of the algorithms:

- :doc:`algorithms/force_matching`
- :doc:`algorithms/relative_entropy`
- :doc:`algorithms/difftre`
- :doc:`algorithms/prior_simulation`

To see the usage of chemtrain in real examples, we implemented the training
procedures of some recent papers:

- :doc:`examples/CG_water_difftre`
- :doc:`examples/CG_alanine_dipeptide`
- :doc:`examples/CG_titanium_difftre`

.. toctree::
   :maxdepth: 2
   :caption: Algorithms
   :hidden:

   algorithms/difftre
   algorithms/force_matching
   algorithms/relative_entropy
   algorithms/prior_simulation


.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Examples
   :hidden:

   examples/CG_water_difftre
   examples/CG_alanine_dipeptide


API Documentation
==================

.. toctree::
   :caption: Chemtrain
   :maxdepth: 2
   :titlesonly:

   api/index


.. toctree::
   :titlesonly:
   :caption: Jax, M.D. Extensions
   :maxdepth: 2

   jax_md_mod/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`