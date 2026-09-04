# chemtrain-deploy Lennard-Jones Export

`export_lennard_jones.py` generates deterministic sparse and dense example
model bundles for the LAMMPS chemtrain-deploy package. The analytic model uses
Lennard-Jones reduced units with $\epsilon = \sigma = 1$. A C1 switching
function smoothly truncates the energy between 2.0 and 2.5. It matches LAMMPS
`pair_style lj/charmm/coul/charmm 2.0 2.5` with zero particle charges.

Run:

```bash
python examples/chemtrain-deploy/export_lennard_jones.py OUTPUT_DIRECTORY
```

Each `.ptb` stores CPU and CUDA implementations in its Newton-off and Newton-on
variants without model communication. The example implements the energy
directly in JAX and does not use a JAX-MD potential helper.
