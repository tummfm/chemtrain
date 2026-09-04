# Model Inputs

Exported models receive engine-independent particle data, global parameters,
and a graph representation. The exporter records the input requirements in the
model bundle so an engine adapter can validate every input before execution.

## Energy Function

An exporter implements `energy_fn` and returns one energy value per particle:

```python
def energy_fn(
    self,
    position,
    particle_data,
    graph,
    comm=None,
    global_data=None,
):
    ...
    return per_particle_energy
```

`position` includes owned particles, real ghost particles, and padded rows.
The exporter supplies two masks through `particle_data`:

- `local_mask` selects particles owned by the current rank.
- `valid_mask` selects owned and real ghost particles while excluding padding.

The exporter uses `local_mask` and `valid_mask` when constructing total
energies and forces. Model code may also use both masks for global observables.

When `has_aux = True`, declare every auxiliary output and return
`(per_particle_energy, outputs)`:

```python
output_fields = (
    exporter.OutputField("order_parameter"),
    exporter.OutputField(
        "local_bias", exporter.OutputScope.LOCAL, extensive=True
    ),
    exporter.OutputField("global_bias", exporter.OutputScope.GLOBAL),
)

return per_particle_energy, {
    "order_parameter": per_particle_order,
    "local_bias": local_bias,
    "global_bias": global_bias,
}
```

`PARTICLE` outputs use the particle dimension as their leading dimension.
`LOCAL` outputs are additive rank-local configuration values that an adapter
reduces once. `GLOBAL` outputs are complete configuration values that the
adapter must not reduce. For example, an interface-pinning model can combine
rank-local contributions to its order parameter with `comm.reduce`, compute
the bias from that total, and return the bias as `GLOBAL`. LAMMPS then exposes
the completed bias without a second reduction. Output shapes and floating-point
dtypes are checked from the traced arrays. Names, scopes, logical dimensions,
flattened component counts, and configuration-output extensivity are stored in
the model bundle. Set ``extensive=True`` when a ``LOCAL`` or ``GLOBAL`` value
scales with system size. This flag controls engine-side normalization and does
not change whether an MPI reduction occurs. Output dtypes are determined by the
executable and aligned with the engine ABI by the connector's compiler wrapper.

Energy (`U`), force (`F`), and virial (`V`) are reserved built-in outputs. The
exporter computes

$$
V = -\frac{d U_{\mathrm{local}}}{d(e_{xx}, e_{yy}, e_{zz}, e_{xy}, e_{xz}, e_{yz})}
$$

by applying the lower-triangular deformation

$$
\mathbf{r}' = \bigl((1 + e_{xx})x,\ e_{xy}x + (1 + e_{yy})y,\ e_{xz}x + e_{yz}y + (1 + e_{zz})z\bigr)
$$

while keeping graph connectivity fixed. The result uses LAMMPS order
`(xx, yy, zz, xy, xz, yz)` and the model's energy units. Model code only
returns its per-particle energy and declared auxiliary outputs.

## Particle Fields

The zero-based scalar `int32` field `species` is always present. Declare
additional scalar `int32` fields with
{class}`chemtrain.deploy.exporter.ParticleField`:

```python
from chemtrain.deploy import exporter


class ResidueModel(exporter.Exporter):
    particle_fields = (
        exporter.ParticleField("residue_id"),
        exporter.ParticleField("region_id"),
    )

    def energy_fn(self, position, particle_data, graph, comm=None):
        species = particle_data["species"]
        residue_id = particle_data["residue_id"]
        region_id = particle_data["region_id"]
        ...
```

Field names are case-sensitive identifiers. Do not declare `species`
explicitly. Floating-point particle fields are not supported by the current
model format.

In LAMMPS, additional fields map to scalar integer properties created with
`fix property/atom ... ghost yes`. The default source for `FIELD` is
`i_FIELD`. Use `atom/input` to select another property. See
{ref}`chemtrain-deploy-lammps`.

## Global Fields

Global fields are scalar values shared by the configuration. Declare them with
{class}`chemtrain.deploy.exporter.GlobalField`:

```python
import jax
import jax.numpy as jnp

from chemtrain.deploy import exporter

jax.config.update("jax_enable_x64", True)


class ParameterizedModel(exporter.Exporter):
    global_fields = (
        exporter.GlobalField("cutoff_center", jnp.float32),
        exporter.GlobalField("bias_strength", jnp.float64),
        exporter.GlobalField("mode", jnp.int32),
    )

    def energy_fn(
        self,
        position,
        particle_data,
        graph,
        comm=None,
        global_data=None,
    ):
        cutoff_center = global_data["cutoff_center"]
        bias_strength = global_data["bias_strength"]
        ...
```

Supported global dtypes are `float32`, `float64`, and `int32`. Values may
change between force evaluations without re-exporting the model. Engine
adapters must map every declared field explicitly. LAMMPS accepts a literal or
an equal-style variable through `global/input`. A model that declares a
`float64` field must enable `jax_enable_x64` before export. The exporter rejects
the declaration otherwise because JAX would trace a `float32` input while the
model metadata still requested `float64`.

## Communication

A communication-enabled model receives an
{class}`chemtrain.deploy.comm.ExportCommunication` object when the `comm`
variant is exported:

```python
features = comm.gather(features)
total = comm.reduce(local_value)
```

`gather` exchanges an atom-leading floating pytree whose leaves share the same
particle dimension and dtype.
`reduce` sums a floating scalar or vector across ranks. Both operations are
differentiable and their reverse-mode communication is emitted automatically.

Communication calls and packed array shapes must remain fixed while tracing
the model. Export the additional variant with:

```python
model.export(communication=True)
```

The `comm_off_newton_off` and `comm_off_newton_on` variants remain in the same
bundle and are selected with `comm off`. Calling `comm.reduce` marks the bundle
as communication-required, so an adapter must reject both non-communication
variants. Set
`communication_required = True` for another model-specific reason that cannot
be represented by an expanded halo alone.

## Pair Topology

Set `include_pair_type = True` when a model requires a topology category for
each geometric neighbor edge:

```python
class TopologicalModel(exporter.Exporter):
    include_pair_type = True

    def energy_fn(self, position, particle_data, graph, comm=None):
        pair_type = graph.pair_type
        ...
```

Sparse graphs store one category per sender/receiver entry. Dense graphs use
the same matrix shape as their neighbor indices. Padding and unclassified
pairs have value zero.

| Value | Category |
|---:|---|
| `0` | unclassified or padding |
| `1` | 1-2 pair |
| `2` | 1-3 pair |
| `3` | 1-4 pair |
| `4` | 1-5 pair |

`DeviceSparseNeighborList` is experimental and cannot currently be exported
for the chemtrain-deploy connector. Use `SimpleSparseNeighborList` or
`SimpleDenseNeighborList` instead.
