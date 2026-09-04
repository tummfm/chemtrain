# MACE Foundation Models

The example exports the
[MACE-MH-1](https://github.com/ACEsuit/mace-foundations) foundation model and
runs the exported model with the chemtrain-deploy LAMMPS package.
The simulation uses a 10,000-molecule, 30,000-atom water box on two GPUs.

## Prerequisites

- A CUDA {doc}`chemtrain-deploy installation <installation>` built with the
  optional OpenEquivariance extension.
- `chemtrain`
- MACE 0.3.16 and
  [MACE-JAX PR #21 at commit 594563b](https://github.com/ACEsuit/mace-jax/commit/594563b322d6127f9b8903eec534dcde51fed83d)
  with OpenEquivariance support
- ASE for writing the LAMMPS data file
- A PDB file for the initial configuration. The example below uses
  `water_10k.pdb` from `docs/_data/`.

Keep the downloaded MACE-MH-1 model intact and select its `omol` head during
conversion. In particular, do not rebuild it with `remove_pt_head` when using
MACE 0.3.16. The rebuilt model stores activation-normalization constants in a
form that the current
[MACE-JAX pull request](https://github.com/ACEsuit/mace-jax/pull/21) does not
fully import. Selecting the head on the original model avoids that conversion
error. Compare a few energies and forces with the PyTorch model whenever the
MACE or MACE-JAX version changes.

## Overview

The workflow has three steps:

1. **Export** — convert the PyTorch MACE-MH-1 weights to a chemtrain-deploy
   model bundle (`.ptb`).
2. **Prepare** — convert the PDB file to a LAMMPS data file.
3. **Simulate** — run the LAMMPS script against the exported model.

## Step 1: Export the Model

The export script loads MACE-MH-1 with the `omol` head and wraps the model in an
{class}`chemtrain.deploy.exporter.Exporter` subclass. The bundle contains both
comm-off Newton variants and a communication-enabled Newton-on variant.

`export_mace_mh1.py`:
```python
import argparse
import functools
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference_output", type=Path, required=True)
    parser.add_argument("--family", default="mp")
    parser.add_argument("--version", default="mh-1")
    args = parser.parse_args()

    # Must precede JAX / MACE / OpenEquivariance imports.
    os.environ["OEQ_NOTORCH"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["JAX_PLATFORMS"] = "cuda"

    import jax.numpy as jnp
    import torch
    from chemtrain.compose import mace_jax as mace_jax_compose
    from chemtrain.deploy import exporter, graphs as export_graphs
    from jax_md import space
    from mace_jax.modules.wrapper_ops import EquivarianceConfig

    class MaceWaterExporter(exporter.Exporter):
        graph_type = export_graphs.SimpleSparseNeighborList
        unit_style = "metal"

        def __init__(self, model_apply, model_config, displacement):
            self.model = model_apply
            self.r_cutoff = model_config["r_max"]
            self.nbr_order = [
                model_config["num_interactions"],
                2 * model_config["num_interactions"],
            ]
            self.displacement = displacement
            super().__init__()

        def energy_fn(self, position, particle_data, graph, comm=None):
            neighbor = graph.to_neighborlist()
            # The data file uses type 1 for H and type 2 for O.
            atomic_numbers = jnp.asarray((1, 8), dtype=jnp.int32)
            return self.model(
                position,
                neighbor,
                species=atomic_numbers[particle_data["species"]],
                comm=comm,
            )

    equivariance = EquivarianceConfig(
        backend="openeq",
        layout="mul_ir",
        group="O3_e3nn",
        optimize_channelwise=True,
        conv_fusion=True,
    )
    torch_model, model_config = mace_jax_compose.load_foundational_model(
        family=args.family,
        version=args.version,
    )
    head = "omol"
    if not hasattr(torch_model, "heads") or head not in torch_model.heads:
        raise ValueError(
            f"Head {head!r} is unavailable. Found "
            f"{tuple(getattr(torch_model, 'heads', ()))}"
        )
    displacement, _ = space.free()
    variables, apply_fn = mace_jax_compose.mace_jax_neighborlist_from_torch(
        model_config,
        torch_model,
        displacement,
        max_edge_multiplier=None,
        per_particle=True,
        species_mapping=mace_jax_compose.AtomicNumberMapping(max_number=100),
        scale_pot=1.0,
        scale_pos=1.0,
        equivariance_config=equivariance,
        head=head,
    )

    model = MaceWaterExporter(
        functools.partial(apply_fn, variables),
        model_config,
        displacement,
    )
    model.export(
        communication=True,
        custom_calls=exporter.OPENEQUIVARIANCE_CUSTOM_CALLS,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    args.reference_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_model, args.reference_output)


if __name__ == "__main__":
    main()
```

A few things to note:

- `head="omol"` selects the Open Molecules head while preserving the original
  multi-head model. The model ships with several heads trained on different
  datasets.
- `scale_pot=1.0` and `scale_pos=1.0` keep the model's native units
  (eV and Å). `unit_style = "metal"` tells LAMMPS to expect the same.
- `max_edge_multiplier=None` disables the static edge-count cap, which is
  required for the deployment path because the graph size is managed by the
  connector's dynamic padding instead.
- `per_particle=True` returns per-atom energies, which is required by
  {class}`chemtrain.deploy.exporter.Exporter`.
- `communication=True` instructs the exporter to trace and bundle a second
  variant that performs halo exchanges between message-passing layers,
  enabling message passing across ranks.

Run the export (the model weights are downloaded automatically on first use):

```bash
python export_mace_mh1.py \
    --output results/mace_mh1_water10k.ptb \
    --reference_output results/mace_mh1_reference.model
```

## Step 2: Prepare the LAMMPS Data File

Convert the PDB file with ASE. `specorder` fixes the LAMMPS type mapping used
by the exporter: type 1 is hydrogen and type 2 is oxygen.

```python
from ase.io import read, write

atoms = read("water_10k.pdb")
atoms.set_cell([67.1, 67.1, 67.1])
atoms.set_pbc(True)
write(
    "results/water_10k.lmpdat",
    atoms,
    format="lammps-data",
    atom_style="atomic",
    specorder=["H", "O"],
    masses=True,
    units="metal",
)
```

The box length comes from the PDB remark. For another structure, set its
periodic cell explicitly and update both `specorder` and the atomic-number
lookup in the exporter.

## Step 3: Run the Simulation

`simulation.lmp`:
```text
variable model      index results/mace_mh1_water10k.ptb
variable data_file  index results/water_10k.lmpdat
variable traj_dump  index results/water10k.lammpstrj
variable proc_x     index 2
variable proc_y     index 1
variable proc_z     index 1

units metal
atom_style atomic
boundary p p p
newton on

processors ${proc_x} ${proc_y} ${proc_z}
read_data ${data_file}

# Select the communication-enabled variant, which requires Newton on and also
# works with one MPI rank.
pair_style chemtrain cuda 0.85 comm on \
  capacity/atom 1.2 capacity/edge 1.5
pair_coeff * * ${model}

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

thermo 10
thermo_style custom step atoms temp pe ke etotal press

dump trajectory all custom 50 ${traj_dump} id type x y z fx fy fz
dump_modify trajectory sort id first yes

# Validate neighbor construction and force evaluation.
run 0

min_style quickmin
minimize 1.0e-4 1.0e-6 20 200

velocity all create 300.0 20260626 mom yes rot no dist gaussian
fix thermostat all nvt temp 300.0 300.0 0.1
timestep 0.0005

run 1000
```

Key settings:

- `pair_style chemtrain cuda 0.85 comm on` selects the CUDA backend
  with 85 % memory allocation and enables halo-exchange communication.
- `capacity/atom` and `capacity/edge` control how buffers grow after an
  overflow. Capacity options belong to `pair_style`, not `pair_coeff`.
- `processors 2 1 1` decomposes the box into two domains along x, one per GPU.

## Running on 2 GPUs

Expose exactly two GPUs via `CUDA_VISIBLE_DEVICES` and
launch LAMMPS with 2 MPI ranks:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/path/to/chemtrain-deploy/lib:${LD_LIBRARY_PATH:-}

mpirun -np 2 lmp -k on g 2 -sf kk \
  -pk kokkos neigh half newton on comm device gpu/aware on \
  -in simulation.lmp
```

The packaged installation finds its runtime backend and extensions
automatically. Set `JCN_PJRT_PATH` only when the runtime backend is installed
elsewhere. Set `JCN_FFI_PATH` only when extensions are installed elsewhere. It
accepts an ordered, colon-separated list of directories.

The runtime selects one visible GPU per local MPI rank. `OMP_NUM_THREADS=1`
avoids CPU thread oversubscription when MPI and OpenMP are both active.

> **Warning**
>
> Kokkos `/kk/device` communication strictly requires CUDA-aware MPI. Kokkos
> `/kk/host` and host-staged Kokkos communication are unsupported.
