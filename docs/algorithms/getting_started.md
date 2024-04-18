# Getting Started

## Installation

**chemtrain** can be installed with pip:

```shell
pip install chemtrain --upgrade
```

The above command installs **JAX for CPU**.

Running **chemtrain on the GPU** requires the installation of a special JAX
version.
Please follow the
[JAX Installation Instructions](https://github.com/google/jax#installation).

(additional_requirements)=
### Additional Packages

Some parts of **chemtrain** require additional packages.
To install these, just provide the `all` option.

```shell
pip install 'chemtrain[all]' --upgrade
```


### Installation from Source

The lines below install **chemtrain** from source for development purposes.

```shell
git clone git@github.com:tummfm/chemtrain.git
pip install -e '.[all,docs]'
```

This command additionally installs the requirements to run the tests

```shell
pytest tests
```

and to build the documentation (e.g. in html):

```shell
make -C docs html
```

## Overview of Chemtrain

Chemtrain is an extensive library to train machine-learning models for
molecular dynamics.
Its modular design tries to enable the creation of new training strategies.
The functionalities of these modules are:

### ``trainers`` -- **High-Level API to Algorithms**

This subpackage bundles high-level entry points to 
pre-implemented algorithms. For a list of currently supported algorithms,
see section [Getting Started with Trainers](#trainers-reference).

### ``learn`` --- **(Deep) Learning Algorithms**

This subpackage provides a lower-level entry to algorithms.
Divided into the respective submodules, it provides all the necessary
ingredients to, e.g., setup force matching or differentiable trajectory
reweighting.

### ``potential`` --- **(Deep) Potential Models**

Besides algorithms, **chemtrain** provides some deep potential 
models, e.g., the DimeNet model.


### ``trajectory`` --- **Simulating and Sampling from Ensembles**

While JAX, M. D. implements multiple simulators to sample
from different ensembles, **chemtrain** provides utilities to efficiently
execute these simulators. Additionally, this module enables the efficient
computation of single-snapshot quantities. Moreover, this module provides
schemes to reuse trajectories via re-weighting based on perturbation theory.

### ``quantity`` --- **Computing Ensemble Quantities**

In addition to computing single-snapshot quantities,
**chemtrain** enables to computation of macroscopic ensemble-based
quantities. Examples of such quantities are simple ensemble averages, but
also thermodynamic quantities like the free energy or heat capacity.

### ``data`` --- **Data Loading and Processing**

This module provides simple utilities to load and process reference
data, mainly atomistic data, e.g., for force matching and relative entropy
maximization.

### ``jax_md_mod`` --- **Extensions and Modifications to JAX, M. D.**

This module contains extensions to the JAX, M. D. library that
are, or were, not yet implemented. 

(trainers-reference)=
## Getting Started with Trainers

To get started with chemtrain and with the most important algorithms,
we provide simple toy examples.
These examples are simple to run on the CPU and explain the basic ideas behind
the algorithms:

- [Force Matching](./force_matching.md)
- [DiffTre](./difftre.md)
- [Prior Simulation](./prior_simulation.md)
- [Relative Entropy Maximization](./relative_entropy.md)

For a more extensive list of available trainers, please refer to the [API
documentation of the ``trainers`` module](../api/trainers.rst).

Additionally, we provide some real examples in which chemtrain was used:

- [Difftre: CG Water on Structural Data](../examples/CG_water_difftre.ipynb)

