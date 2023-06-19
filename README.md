# Training Molecular Dynamics Potentials in JAX
[![Documentation Status](https://readthedocs.org/projects/chemtrain/badge/?version=latest)](https://chemtrain.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/chemtrain.svg)](https://badge.fury.io/py/chemtrain)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

*chemtrain* is a library for training MD potentials and integrates with
[JAX, M.D.](https://github.com/jax-md/jax-md) as its differentiable MD engine.
Note that this is the first alpha release of *chemtrain*, expect breaking changes.
Over the course of the next weeks, several updates will extend its
functionalities and documentation.

## Features

* [Differentiable Trajectory Reweighting](https://www.nature.com/articles/s41467-021-27241-4) (DiffTRe)
to train on experimental data
* [Force Matching and Relative Entropy Minimization](https://doi.org/10.1063/5.0124538) for coarse-grained systems

Incoming features:

* Uncertainty Quantification via [Deep Ensembles and Stochastic gradient MCMC](https://doi.org/10.1021/acs.jctc.2c01267)
* Active Learning
* Hybrid Trainers to combine different training methods
* Molecular Property Prediction

## Getting started

To get started with training MD potentials, using existing trainers that
implement standard training schemes is the most straightforward apprach.
Please refer to the provided [examples](examples) as a refernece on how to set
up the corresponding training environment and dataset.
```python
trainer = trainers.Difftre(init_params,
                           optimizer)

trainer.add_statepoint(energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, compute_fns, reference_state,
                       targets)
trainer.train(num_updates)
```

More advanced users may want to extend existing trainers or combine different
trainers to implement custom training pipelines.

## Installation
*chemtrain* can be installed via pip:
```
pip install chemtrain
```

## Requirements
The repository uses the following packages:
```
    'jax',
    'jax-md',
    'jax-sgmc',
    'optax',
    'dm-haiku',
    'sympy',
    'tree_math',
    'cloudpickle',
    'chex',
```
The code runs with Python >=3.8.

## Contribution
Contributions are always welcome! Please open a pull request to discuss the code
additions.

Since this is a very early alpha release, do not hesitate to reach out if some
feature or example is failing. If the specific feature is a priority for you,
its support can be accelerated.

## Contact
For questions, please contact stephan.thaler@tum.de or open an Issue on GitHub.



