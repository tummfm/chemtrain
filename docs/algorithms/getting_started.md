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

## Chemtrain and Jax, M. D.

TODO...

## Potential Model

TODO...

## Getting Started with Trainers

TODO...

