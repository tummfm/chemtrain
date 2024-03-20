---
jupytext:
  formats: md:myst,
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3
  name: python3
---

```{code-cell} ipython3
:tags: [hide-cell]

from functools import partial

import numpy as onp

import jax.numpy as jnp
import jax

import optax

import matplotlib.pyplot as plt

from chemtrain import quantity, trainers, traj_util
```

# Differentiable Trajectory Reweighting (DiffTRe)

## Concepts

The key concept that enables DiffTRe is the interchangeability of ensemble and
trajectory averages for ergodic systems

```{math}
\langle a \rangle_T = \lim_{T\rightarrow \infty}\frac{1}{T}\int_0^T a(x(t))dt = \int a(x)p(x)dx = \langle a \rangle_{p(x)}.
```

This view does not treat the simulation as a dynamic process but merely an
efficient method to sample from the distribution $p(x)$.

However, to compute gradients of the ensemble averages, an additional step is
necessary.
Umbrella sampling is a method that uses non-physical potentials to compute
ensemble averages of target potentials. 
The empowering idea is to re-weight the collected samples by the relative
probability between the non-physical and target distribution.
Hence, this re-weighting step corrects the bias introduced by the non-physical
potential.

Given a trainable potential $U_\theta$ and samples $x^{(i)}$ from a reference
potential $\tilde U$, the discretized reweighting reads

```{math}
\langle a \rangle_{U_\theta} \approx \sum_{i=1}^N w^{(i)}a\left(x^{(i)}\right), \quad w^{(i)} = \frac{\exp\left(-\beta(U_\theta(x^{(i)}) - \tilde U(x^{(i)})\right)}{\sum_{j=1}^N\exp\left(-\beta(U_\theta(x^{(j)}) - \tilde U(x^{(j)})\right)}
```

The reference potential $\tilde U$ does not depend on the learnable parameters.
Hence, the only contribution to the gradients arises through the weights 
$w^{(i)}$ and the instantaneous states $a(x)$ of the observables.
This enables DiffTRe to compute gradients of the loss function without
differentiating through the costly molecular dynamics simulation.

Unfortunately, the statistical error of the approximation grows exponentially
fast with the difference between the target and the reference potential.
If the number of effective samples

```{math}
N_{eff} = e^{-\sum_{i=1}^N w^{(i)}\log w^{(i)}}
```

decreases below a threshold, DiffTRe replaces the reference potential by the
current potential $\tilde U \leftarrow U_\theta$ and resamples the conformations
$x^{(i)}$.


## Toy Example

For a canonical system of two-particles connected by a spring,
the Boltzmann factor is

```{math}
  \rho(\mathbf{r}) \propto e^{-\frac{1}{2}b(||\mathbf{r}_1 - \mathbf{r}_2|| - r_0)^2}
```

Hence, the probability of finding the two identical
particles in a distance of $r$ is
```{math}
  p(r) = \sqrt{\frac{2b}{\pi}}\left(\frac{b}{1 + br_0^2}\right)r^{2}e^{-\frac{1}{2}b(r - r_0)^2}.
```
The term $r^2$ in front of the exponential factor emerges in the transformation
from cartesian to spherical coordinates.
With this probability distribution, we can directly compute the radial
distribution function.

```{math}
  g(r) = \frac{V}{4 \pi r^2 N^2}p(r).
```

```{code-cell} ipython3
box = 1.0

def radial_distribution(r, r_0=0.35, b=200.0):
    norm = onp.sqrt(onp.pi / (2 * b)) * (1 + b * r_0 ** 2) / b
    g_r = box ** 3 / (16 * onp.pi) * onp.exp(-0.5 * b * (r - r_0) ** 2) / norm
    return g_r
```

We now want to learn the parameters of this harmonic bond based on a reference
radial distribution function.

```{code-cell} ipython3
r = onp.linspace(0.0, box, 100)
target = onp.vstack((r, radial_distribution(r))).T
```

Although we could find an analytic relation to the potential parameters,
this is not possible for more complex systems.
Thus, we need to simulate this relation and set up a model of the system.

Thus, we first need to define an appropriate potential model.

```{code-cell} ipython3
from jax_md import energy, space, simulate, partition

def energy_fn_template(params):
    energy_fn = energy.simple_spring_bond(
        displacement_fn,
        jnp.asarray([[0, 1]]),
        length=params["r_0"],
        epsilon=100 * params["b"],
        alpha=2.0
    )
    return energy_fn

init_params = {"r_0": 0.3, "b": 1.5}

```

Secondly, we need a routine to simulate the positions of the particles.

```{code-cell} ipython3
r_init = jnp.asarray([[0.0, 0.0, 0.0], [0.11, 0.09, 0.12]])
displacement_fn, shift_fn = space.periodic_general(box)

dt = 0.01
timings = traj_util.process_printouts(dt, 1100, 100, 1.0)

simulator_template = partial(
    simulate.nvt_langevin, shift_fn=shift_fn,
    dt=dt, kT=2.56, gamma=0.5, mass=10.0)

neighbor_fn = partition.neighbor_list(displacement_fn, box, 0.5)

simulator_init, _ = simulator_template(energy_fn_template(init_params))
simulator_init_state = simulator_init(jax.random.PRNGKey(0), r_init)
nbrs_init = neighbor_fn.allocate(r_init)

system = {
    'displacement_fn': displacement_fn,
    'reference_box': box
}
```

There are multiple classical approaches that enable the inversion of a 
radial distribution function into a pair-potential.
However, they are not applicable to general models, e.g., neural networks.
Thus, DiffTRe enables gradient based training, which we are going to set up in
the next step.

```{code-cell} ipython3
import optax

lr_schedule = optax.exponential_decay(-0.05, 300, 0.1)
optimizer = optax.chain(
    optax.scale_by_rms(0.9),
    optax.scale_by_schedule(lr_schedule)
)
```

Finally, we have to specify the training targets, which is in our case the
radial distribution function.
Since we only have two particles in a box, we approximate the distribution 
with slightly coarser bins.

```{code-cell} ipython3
target_builder = quantity.TargetBuilder()

target_builder['rdf'] = quantity.structure.init_radial_distribution_target(
    target, rdf_start=0.00, rdf_cut=1.0, nbins=50)

targets, compute_fns = target_builder.build(system)
```

We now created a numerical representation of the system and can run the trainer.

```{code-cell} ipython3

trainer = trainers.Difftre(
    init_params, optimizer, reweight_ratio=0.99
)

trainer.add_statepoint(
  energy_fn_template, simulator_template, neighbor_fn, timings, 2.56,
  compute_fns, (simulator_init_state, nbrs_init), targets=targets)

```

```{code-cell} ipython3
:tags: [hide-output]
trainer.train(300)
```

### Results

```{code-cell} ipython3
plt.plot(trainer.epoch_losses)
plt.title("Loss History")
plt.xlabel("Iterations")
plt.ylabel("Loss")
```

```{code-cell} ipython3
plt.plot(onp.linspace(0.00, 1.0, 50), trainer.predictions[0][0]['rdf'], label="Initial")
plt.plot(onp.linspace(0.00, 1.0, 50), trainer.predictions[0][149]['rdf'], label="Final")
plt.plot(r, radial_distribution(r), label="Reference")
plt.legend()
plt.title("Radial Distribution Function")
plt.show()
```

Let's also take a look at the inferred parameters.

```{code-cell} ipython3
print(trainer.params)
```


## Further Reading

### Examples

- [CG Water from Structural Data](../examples/CG_water_difftre.ipynb)

### Publications

1. Thaler, S., Zavadlav, J. *Learning neural network potentials from experimental data via Differentiable Trajectory Reweighting*. Nat Commun 12, 6884 (2021). <https://doi.org/10.1038/s41467-021-27241-4>
2. Carles Navarro and Maciej Majewski and Gianni de Fabritiis *Top-down machine learning of coarse-grained protein force-fields*. arXiv (2023). <https://arxiv.org/abs/2306.11375>
