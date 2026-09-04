# Getting Started

The steps below export a per-particle potential and evaluate the potential in
LAMMPS. chemtrain and the {doc}`chemtrain-deploy installation <installation>`
must be available.

## Export a Model

Potential models are exported by subclassing
{class}`chemtrain.deploy.exporter.Exporter`. The model must return one energy
value per particle. The exporter generates forces by differentiating the
selected energy sum.

```python
from pathlib import Path

from jax_md import space
from jax_md_mod import custom_energy

from chemtrain.deploy import exporter, graphs


class LennardJonesModel(exporter.Exporter):
    graph_type = graphs.SimpleSparseNeighborList
    r_cutoff = 5.0
    nbr_order = [1, 2]
    unit_style = "metal"

    def __init__(self):
        displacement, _ = space.free()
        self.apply = custom_energy.customn_lennard_jones_neighbor_list(
            displacement,
            box_size=None,
            species=None,
            sigma=1.0,
            epsilon=1.0,
            r_onset=4.0,
            r_cutoff=self.r_cutoff,
            initialize_neighbor_list=False,
            per_particle=True,
        )

    def energy_fn(self, position, particle_data, graph, comm=None):
        neighbor = graph.to_neighborlist()
        return self.apply(
            position,
            neighbor,
            species=particle_data["species"],
        )


model = LennardJonesModel()
model.export()
model.save(Path("lennard_jones.ptb"))
```

`unit_style` must match the LAMMPS `units` command. The example exports
separate comm-off executables for Newton pair on and off. Models that call the
communication interface must also export the comm-on, Newton-on executable
with `model.export(communication=True)`.

(chemtrain-deploy-platforms)=
## Platforms and Model Variants

A model bundle can contain executable code for more than one PJRT backend. The
`platforms` argument selects the backends stored in every generated variant.
CUDA is the default:

```python
model.export(platforms=("cuda",))
```

Export both currently supported backends when the same bundle should run with
either the CPU or CUDA PJRT backend:

```python
model.export(platforms=("cpu", "cuda"))
```

The exporter uses `float32` positions and differentiation by default. To use
`float64`, enable JAX x64 support before export and declare the position dtype:

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class Float64Model(LennardJonesModel):
    position_dtype = jnp.float64
```

The force and virial derivatives follow the position dtype. The energy
function still controls the precision of its parameters, constants,
calculations, and returned energy. To evaluate the complete model in
`float64`, use `float64` throughout the energy function and avoid explicit
downcasts. StableHLO retains the dtypes that JAX traces for each output.

An engine adapter may use a different floating-point ABI. The connector
inserts explicit conversions at the model boundary, so a model traced with
`float64` positions can, for example, run behind a `float32` device adapter.
Avoid that conversion when the complete input and output path must retain
`float64` precision.

Backend executables and model variants describe different choices. Every
bundle contains `comm_off_newton_off` and `comm_off_newton_on` variants.
`communication=True` also adds `comm_on_newton_on`. Communication cannot be
combined with Newton pair off. Each variant can contain CPU, CUDA, or both
backend implementations. CPU communication requires host callbacks from the
embedding application. CUDA communication uses the CUDA runtime path.

Every variant computes and returns the extensive rank-local strain virial.
Newton and virial behavior are fixed in the executable rather than selected by
dynamic StableHLO inputs.

## Run the Model in LAMMPS

The following input reads an existing atomic configuration and uses the model
for energy and force evaluation:

```text
units          metal
atom_style     atomic
boundary       p p p

read_data      system.data
mass           1 1.0

pair_style     chemtrain cuda
pair_coeff     * * lennard_jones.ptb

neighbor       2.0 bin
neigh_modify   every 1 delay 0 check yes

thermo         10
thermo_style   custom step atoms temp pe ke etotal press

run            0
```

The backend name is required. An optional second argument sets the PJRT memory
fraction, which defaults to `0.75`.

For one CUDA device with the Kokkos pair style:

```bash
CUDA_VISIBLE_DEVICES=0 \
lmp -k on g 1 -sf kk \
  -pk kokkos neigh half newton on comm device gpu/aware on \
  -in input.lmp
```

The `-sf kk` suffix selects `chemtrain/kk`. Use one MPI rank per GPU
for multi-GPU simulations and export a communication variant when the model
exchanges intermediate per-atom values between domains.

> **Warning**
>
> Kokkos `/kk/device` communication strictly requires CUDA-aware MPI. Kokkos
> `/kk/host` and host-staged Kokkos communication are unsupported.

See {doc}`lammps` for multiple models, named inputs, model outputs, topology,
MPI communication, and Kokkos restrictions. See {doc}`model_inputs` for the
export-side input schema.
