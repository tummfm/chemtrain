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

## Model Inputs

### Particle Fields

LAMMPS maps its one-based atom type to the model's zero-based `species` field.
`species` is always present and cannot be remapped.

Every additional particle field is a scalar `int32` array. By default, model
field `FIELD` maps to an existing LAMMPS custom property named `i_FIELD`:

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

Every exported global field requires an explicit `global/input` mapping. A
source is either a numeric literal or an equal-style variable:

```text
variable        switching_radius equal 3.2
pair_style      chemtrain cuda \
                  global/input r0 v_switching_radius \
                  global/input width 0.2
pair_coeff      * * parameterized_model.ptb
```

Equal-style variables are evaluated for every force call. The resulting value
is converted to the `float32`, `float64`, or `int32` type declared by the
exported model.

The ordinary `chemtrain` pair style uses LAMMPS host-side `float64` positions
and outputs with either the CPU or CUDA PJRT backend. The Kokkos device pair
style currently uses a `float32` device ABI. A model exported with
`position_dtype = jnp.float64` differentiates with respect to `float64`
positions behind that Kokkos interface, but the connector converts positions
and built-in outputs at the adapter boundary. The energy function must also
use `float64` parameters, constants, and calculations when the entire model is
intended to run in double precision. Use the ordinary pair style with the CUDA
backend when the complete LAMMPS-to-model path must remain `float64`.

See {doc}`model_inputs` for the corresponding exporter declarations.

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

Define `compute chemtrain/output` after `pair_coeff`. LAMMPS obtains the output
scope and shape from model metadata when the command is parsed. `PARTICLE`
outputs use a per-atom vector or array. `LOCAL` outputs are summed once across MPI
ranks and exposed as a global scalar or vector. `GLOBAL` outputs are complete
configuration values and are exposed without a LAMMPS reduction. For example,
an interface-pinning bias computed after `comm.reduce` is `GLOBAL`. LAMMPS must
not reduce the already complete bias again. For configuration outputs, the
exported `extensive` flag tells LAMMPS whether the value scales with system
size. It affects LAMMPS normalization but does not change the `LOCAL` or
`GLOBAL` reduction behavior.

All configuration outputs require group `all`.

The compute reads values cached by the pair evaluation at the current timestep
without executing the model again. `F`, `U`, and `V` are built-in
force, energy, and virial outputs and cannot be requested as auxiliary outputs.

When Kokkos suffixing is active, LAMMPS selects
`chemtrain/output/kk`. The Kokkos compute reads the pair style's output cache
and exposes the same scope-specific interface.

## Communication and MPI

`comm off` selects `comm_off_newton_off` or `comm_off_newton_on` from the
LAMMPS Newton setting. Both variants can run on one or multiple ranks without
exchanging intermediate model values.

`comm on` selects `comm_on_newton_on` from a model exported with
`Exporter.export(communication=True)`. It allows model code to call
`comm.gather(...)` between message-passing layers and `comm.reduce(...)` for
scalar or vector sums across ranks. The communication structure must be fixed
during export.

Calling `comm.reduce` marks the bundle as communication-required. LAMMPS then
rejects `comm off` even for a single MPI rank. Gather-only models may still use
`comm off` with their exported expanded neighbor halo.

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

Format-5 bundles always provide

$$
V = -\frac{d U_{\mathrm{local}}}{d(e_{xx}, e_{yy}, e_{zz}, e_{xy}, e_{xz}, e_{yz})}
$$

for both Newton settings. The exporter applies the lower-triangular deformation

$$
\mathbf{r}' = \bigl((1 + e_{xx})x,\ e_{xy}x + (1 + e_{yy})y,\ e_{xz}x + e_{yz}y + (1 + e_{zz})z\bigr)
$$

while graph connectivity remains fixed. `V` has the model's energy units and
uses LAMMPS order `(xx, yy, zz, xy, xz, yz)`. The adapter binds `V` for every
evaluation and accumulates it only when LAMMPS requests the global virial.
Bundles without this output are rejected. Per-atom stress is not provided.

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

Set `include_pair_type = True` on the exporter when a model consumes topology
categories aligned with geometric neighbor edges. LAMMPS supplies:

| Value | Category |
|---:|---|
| `0` | unclassified or padding |
| `1` | 1-2 pair |
| `2` | 1-3 pair |
| `3` | 1-4 pair |
| `4` | 1-5 pair |

Topology labels annotate geometric edges inside the model cutoff. The labels do
not add excluded or out-of-cutoff edges.

The `topology` pair-style keyword installs `zero` styles for bonded interaction
classes present in the system but not otherwise configured. Existing styles
are left unchanged. If a data file contains coefficients owned entirely by the
model, use `read_data ... nocoeff`.

## Troubleshooting

`Model does not contain the requested 'comm' variant`
: Re-export the model with `communication=True`, or use `comm off`.

`global field ... requires an explicit global/input mapping`
: Add one `global/input` entry for every global field declared by the selected
  model.

`Compute chemtrain/output must be defined after pair_coeff`
: Move the compute command below the command that loads the model bundle.

`chemtrain/kk communication models require newton pair on`
: Set `newton on` before `pair_coeff`.

`chemtrain/kk multi-rank device communication requires CUDA-aware MPI`
: Use a CUDA-aware MPI build and
  `-pk kokkos ... comm device gpu/aware on`, or select the host pair style.

`Incorrect number of arguments for pair_coeff chemtrain`
: Supply exactly one model file for the unnamed form, or one file per declared
  model context. Configure scales and capacities on `pair_style`.
