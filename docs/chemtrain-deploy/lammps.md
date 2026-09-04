(chemtrain-deploy-lammps)=
# LAMMPS Package

The optional `chemtrain-deploy` LAMMPS package evaluates chemtrain model
bundles through `libconnector.so`. The package provides the host pair style
`chemtrain`, the Kokkos pair style `chemtrain/kk`, and matching
computes for particle and configuration-level auxiliary outputs.

## Pair Style

### Syntax

For one model:

```text
pair_style chemtrain BACKEND [MEMORY_FRACTION] [keywords ...]
pair_coeff * * MODEL
```

For multiple models:

```text
pair_style chemtrain BACKEND [MEMORY_FRACTION] \
  models NAME SCALE [NAME SCALE ...] [keywords ...]
pair_coeff * * MODEL [MODEL ...]
```

`BACKEND` selects the PJRT backend. The documented runtime uses `cuda`.
`MEMORY_FRACTION` is optional and defaults to `0.75`. The value limits CUDA
memory and is accepted but ignored by the CPU backend. The selected model
variant must contain an executable for `BACKEND`. See
{ref}`chemtrain-deploy-platforms` for the distinction between executable
platforms and model variants. Each model declared by `models` has a unique
context name and a force/energy scale. Model files in `pair_coeff` follow
declaration order. Their scaled force and energy contributions are added by the
pair style.

`pair_coeff` requires the complete `* *` type range and accepts only model
files. Trailing numeric capacity or scale arguments are not supported.

### Keywords

| Keyword | Meaning | Default |
|---|---|---|
| `comm on\|off` | Enable the communication variant. With `comm off`, the Newton setting selects the matching non-communication variant. | `off` |
| `capacity/atom F` | Growth factor for atom buffers after overflow. | `1.20` |
| `capacity/edge F` | Growth factor for sparse raw-edge buffers. | `1.50` |
| `capacity/dense-neighbor F` | Growth factor for dense neighbor rows. | `1.25` |
| `models NAME SCALE ...` | Declare named model contexts and their scales. | one unnamed model |
| `atom/input [MODEL] FIELD SOURCE` | Map a particle field to a LAMMPS custom integer property. | `i_FIELD` |
| `global/input [MODEL] FIELD SOURCE` | Map a global field to a literal or equal-style variable. | no implicit mapping |
| `output [MODEL] NAME` | Retain an exported auxiliary output for a compute. | none |
| `topology` | Install zero styles for missing bonded interaction classes. | disabled |

All capacity growth factors must be greater than one. The `models` keyword must
appear before `atom/input`, `global/input`, or `output`, because those mappings
are resolved against a model context.

The pair style no longer accepts a `device` keyword. Select devices with
`CUDA_VISIBLE_DEVICES`, the MPI rank mapping, and the LAMMPS Kokkos options.

### Single-Model Example

```text
units          metal
atom_style     atomic
newton         on
read_data      system.data

pair_style     chemtrain cuda 0.80 comm off \
                 capacity/atom 1.20 capacity/edge 1.50
pair_coeff     * * potential.ptb
```

## Model Contexts

The following combines a potential with a separately exported order parameter.
The order model contributes no energy or force because its scale is zero, but
its `steinhardt_l4` output remains available:

```text
variable       order_r0 equal 3.2
variable       order_d0 equal 0.2

pair_style     chemtrain cuda comm on \
                 models potential 1.0 order 0.0 \
                 global/input order steinhardt_r0 v_order_r0 \
                 global/input order steinhardt_d0 v_order_d0 \
                 output order steinhardt_l4
pair_coeff     * * potential.ptb steinhardt_order.ptb
```

All active models use the same communication setting and the same
neighbor-list format. The pair style requests the largest cutoff and
communication distance declared by any model.

## Supplying Model Inputs

Export-side declarations, supported dtypes, and communication variants are
described in {doc}`model_inputs`. This section explains how LAMMPS supplies
the declared fields.

### Particle Fields

LAMMPS maps its one-based atom type to the model's zero-based `species` field.
`species` is always present and cannot be remapped.

For every additional particle field, LAMMPS reads an existing custom integer
property. By default, field `FIELD` uses the property named `i_FIELD`:

```text
fix             fields all property/atom i_residue_id ghost yes
set             atom * i_residue_id 0

pair_style      chemtrain cuda
pair_coeff      * * residue_model.ptb
```

Define the property before `pair_coeff`, use `ghost yes`, and keep the fix
active while the pair style is in use. The adapter reads current custom-field
values before every model evaluation, so updates are visible on the following
force call. Use `atom/input` to override the default property name:

```text
pair_style chemtrain cuda \
  atom/input residue_id i_residue_order
```

With named model contexts, prefix the field with its model name:

```text
pair_style chemtrain cuda \
  models potential 1.0 correction 1.0 \
  atom/input correction region_id i_region
```

### Global Fields

Every declared global field needs a `global/input` mapping. Its source is
either a numeric literal or an equal-style variable:

```text
variable        switching_radius equal 3.2
pair_style      chemtrain cuda \
                  global/input r0 v_switching_radius \
                  global/input width 0.2
pair_coeff      * * parameterized_model.ptb
```

Equal-style variables are evaluated for every force call. LAMMPS converts the
result to the dtype declared by the model.

The ordinary `chemtrain` pair style uses LAMMPS host-side `float64` positions
and outputs with either the CPU or CUDA backend. The Kokkos device pair style
uses a `float32` device interface. Use the ordinary pair style with CUDA when
the complete LAMMPS-to-model path must remain `float64`. See
{doc}`getting_started` for export-side precision choices.

## Auxiliary Outputs

Request an exported auxiliary output with `pair_style` and expose the output
through a compute:

```text
pair_style     chemtrain cuda output order_parameter
pair_coeff     * * model.ptb

compute        q all chemtrain/output order_parameter
compute        Q all reduce sum c_q
```

For multiple models, the compute also takes the model context:

```text
compute q all chemtrain/output order steinhardt_l4
```

Define `compute chemtrain/output` after `pair_coeff`. LAMMPS reads the output
scope and shape from the model when the compute is defined. Particle outputs
become per-atom values. Local outputs are summed once across MPI ranks, while
global outputs are already complete and are not reduced again. The exporter
defines each output's scope and whether it is extensive. See {doc}`model_inputs`.

All configuration outputs require group `all`.

The compute reads values cached by the pair evaluation at the current timestep
without executing the model again. `F`, `U`, and `V` are built-in
force, energy, and virial outputs and cannot be requested as auxiliary outputs.

When Kokkos suffixing is active, LAMMPS selects
`chemtrain/output/kk`. The Kokkos compute reads the pair style's output cache
and exposes the same scope-specific interface.

## Communication and MPI

`comm off` selects a non-communication model variant. `comm on` selects the
communication variant created by `model.export(communication=True)`. It is
required when the model exchanges intermediate values between MPI ranks.
Models that call `comm.reduce` always require `comm on`. See
{ref}`chemtrain-deploy-model-inputs-communication` for the export-side API.

Communication-aware runs require Newton pair forces:

```text
newton on
pair_style chemtrain cuda comm on
```

Use one MPI rank per GPU. The local rank determines the selected visible PJRT
device:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
mpirun -np 2 lmp -in input.lmp
```

The exported communication distance determines the required ghost
halo. LAMMPS adds the current neighbor skin to that distance.

## Virial

Exported models provide the configuration virial in LAMMPS order
`(xx, yy, zz, xy, xz, yz)`. The pair style adds it only when LAMMPS requests
the global virial. Per-atom stress is not available. See {doc}`model_inputs`
for the export-side definition.

## Kokkos CUDA

Configure Kokkos with its half-list neighbor builder and device communication:

```bash
CUDA_VISIBLE_DEVICES=0 \
lmp -k on g 1 -sf kk \
  -pk kokkos neigh half newton on comm device gpu/aware on \
  -in input.lmp
```

For multiple GPUs, launch one MPI rank per visible device:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
mpirun -np 2 lmp -k on g 2 -sf kk \
  -pk kokkos neigh half newton on comm device gpu/aware on \
  -in input.lmp
```

The Kokkos device path currently requires:

- `SimpleSparseNeighborList` exports.
- The `package kokkos neigh half` setting.
- Newton pair forces when `comm on` is selected.
- `CommKokkos` device pair communication and CUDA-aware MPI for multi-rank
  communication models.

> **Warning**
>
> `/kk/device` communication strictly requires CUDA-aware MPI. `/kk/host`
> and host-staged Kokkos communication are unsupported.

The Kokkos package setting controls which neighbor builder is available. The
pair style still requests the full or ghost rows required by the exported
graph and the current Newton mode. Dense and device-built sparse exported
graphs are not supported by `chemtrain/kk`. Do not configure Kokkos
with `neigh full`.

## Topology

Models that declare pair-topology input receive topology categories for their
geometric neighbor edges. See {ref}`chemtrain-deploy-model-inputs-topology`
for the export-side declaration and category values.

The `topology` pair-style keyword installs `zero` styles for bonded interaction
classes present in the system but not otherwise configured. Existing styles
are left unchanged. If a data file contains coefficients owned entirely by the
model, use `read_data ... nocoeff`.
