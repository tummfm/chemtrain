# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exporting potential models to serialized StableHLO bundles."""

import abc
import enum
import inspect
import math
import re
from typing import Any, Callable, Dict, List, NamedTuple, NoReturn, Tuple

import jax
from jax import numpy as jnp, export, lax
import jax_md_mod
from jax_md import space

from . import comm, graphs, util
from ._protobuf import model_pb2 as model_proto


OPENEQUIVARIANCE_CUSTOM_CALLS = (
    "conv_forward",
    "conv_backward",
    "conv_double_backward",
)

COMMUNICATION_CUSTOM_CALLS = comm.CUSTOM_CALL_TARGETS

MODEL_FORMAT_VERSION = 5
_SUPPORTED_PLATFORMS = ("cpu", "cuda")


class ParticleField(NamedTuple):
    """Descriptor for an additional scalar per-particle model input.

    ``species`` is implicit and must not be registered here. Additional field
    semantics are defined by the exported model and its accompanying
    documentation.
    """

    name: str
    dtype: Any = jnp.int32


class GlobalField(NamedTuple):
    """Descriptor for a scalar model input shared by the full configuration."""

    name: str
    dtype: Any = jnp.float32


class OutputScope(enum.Enum):
    """Declares the leading shape and engine reduction of an output.

    ``PARTICLE`` outputs have a leading particle axis. ``LOCAL`` outputs are
    additive rank-local configuration values that the engine adapter reduces
    once. ``GLOBAL`` outputs are complete configuration values and must not be
    reduced again.
    """

    PARTICLE = "particle"
    LOCAL = "local"
    GLOBAL = "global"


class OutputField(NamedTuple):
    """Descriptor for an auxiliary model output.

    ``scope`` controls the particle axis and MPI reduction. ``extensive``
    records whether a configuration value scales with system size so an engine
    can normalize it correctly. Extensivity does not change reduction scope
    and is not valid for ``PARTICLE`` outputs.
    """

    name: str
    scope: OutputScope = OutputScope.PARTICLE
    extensive: bool = False


class _ExportComputation:
    """Creates fresh communication state for every JAX trace.

    The callable may be retraced, so each invocation replaces the recorded
    packed widths instead of appending metadata from an earlier trace.
    """

    def __init__(self, owner, neighbor_order, newton_pair, enabled=False):
        self.owner = owner
        self.neighbor_order = neighbor_order
        self.newton_pair = newton_pair
        self.enabled = enabled
        self.gather_widths = None
        self.reduce_widths = None

    def __call__(self, *args):
        communication = None
        if self.enabled:
            communication = comm.ExportCommunication(enabled=True)
        result = self.owner._energy_fn(
            *args,
            communication=communication,
            neighbor_order=self.neighbor_order,
            newton_pair=self.newton_pair,
        )
        if communication is not None:
            self.gather_widths = tuple(communication.gather_widths)
            self.reduce_widths = tuple(communication.reduce_widths)
        return result


class Exporter(metaclass=abc.ABCMeta):
    """Exports a potential model to a serialized StableHLO bundle.

    Subclasses select a graph representation and implement :meth:`energy_fn`.
    The deployment documentation contains complete export examples.

    Attributes:
        graph_type: Graph representation used for engine neighbor data.
        nbr_order: Two neighbor orders used with Newton on and Newton off,
            respectively.
        r_cutoff: Model cutoff radius.
        unit_style: Unit style used for positions and energies. Force units
            follow from the length and energy units.
        position_dtype: Floating-point dtype of the position input and its
            force and virial derivatives. The energy function determines the
            dtypes of its calculations and returned values. Engine adapters
            may use a different floating-point dtype because the connector
            converts at the engine ABI boundary.
        has_aux: If ``True``, ``energy_fn`` returns
            ``(particle_energy, outputs)``.
        particle_fields: Ordered descriptors for additional scalar ``int32``
            arrays supplied through ``particle_data``. Zero-based ``species``
            is always provided and must not be registered explicitly.
        global_fields: Ordered descriptors for scalar inputs supplied through
            ``global_data``. Global field values may vary between force calls.
        output_fields: Ordered descriptors for auxiliary outputs and their
            ``PARTICLE``, ``LOCAL``, or ``GLOBAL`` scope. Configuration
            outputs also declare whether they are extensive. Shapes are
            inferred during tracing.
        communication_required: If ``True``, callers must select the
            communication-enabled variant.
        include_pair_type: If ``True``, exposes engine topology categories as
            ``graph.pair_type``.

    """

    # Default engine graph representation.
    graph_type: graphs.NeighborList = graphs.SimpleSparseNeighborList

    # Required graph depth for Newton on and Newton off, in that order.
    nbr_order: List[int] = [1, 1]

    r_cutoff: float
    unit_style: str = "real"
    position_dtype: Any = jnp.float32
    has_aux: bool = False
    particle_fields: Tuple[ParticleField, ...] = ()
    global_fields: Tuple[GlobalField, ...] = ()
    output_fields: Tuple[OutputField, ...] = ()
    communication_required: bool = False
    include_pair_type: bool = False

    _symbols: List[str] = None
    _constraints: List[str] = None
    _init_fns: List[Callable] = None
    _proto: model_proto.Model = None

    @abc.abstractmethod
    def energy_fn(self, position, particle_data, graph, comm=None):
        """Computes particle energies for positions and a graph.

        Args:
            position: (N, dim) Array of particle positions, including ghost
                atoms that are not within the local domain.
            particle_data: Mapping containing the built-in zero-based scalar
                ``int32`` ``species`` array and every registered additional
                scalar array, with one value per particle.
            graph: Graph representation of the neighborhood around atoms. If
                ``include_pair_type`` is enabled, ``graph.pair_type`` contains
                aligned topology categories, with zero used for unclassified
                and padding entries.
            comm: Optional communication interface. Models that use
                communication call ``comm.gather`` or ``comm.reduce`` at fixed
                locations in the traced computation.

        Returns:
            An energy contribution for each particle. If ``has_aux`` is
            ``True``, return ``(energy, outputs)``. Output keys must match
            ``output_fields``. ``PARTICLE`` outputs start with the particle
            axis, while ``LOCAL`` and ``GLOBAL`` outputs describe one
            configuration.

        """
        pass

    def _define_input_shapes(self):
        particle_fields = self._effective_particle_fields
        global_fields = self._effective_global_fields

        @util.define_symbols("n_atoms")
        def define(n_atoms, **kwargs):
            shape_defs = (
                jax.ShapeDtypeStruct(
                    (n_atoms, 3), self._effective_position_dtype
                ),
                *(jax.ShapeDtypeStruct(
                    (n_atoms,), field.dtype
                ) for field in particle_fields),
                *(jax.ShapeDtypeStruct((), field.dtype)
                  for field in global_fields),
                jax.ShapeDtypeStruct((), jnp.int32),  # n_local
                jax.ShapeDtypeStruct((), jnp.int32),  # n_ghost
            )
            return shape_defs

        return define

    def _add_shapes(self, init_fn, **kwargs):
        init_fn(self._symbols, self._constraints, self._init_fns, **kwargs)

    def _create_shapes(self):
        all_symbols = ",".join(self._symbols)
        symbols = {
            key: symb for key, symb in zip(
                self._symbols,
                export.symbolic_shape(all_symbols, constraints=self._constraints),
            )
        }
        shapes = []
        for init_fn in self._init_fns:
            shapes.extend(init_fn(**symbols))

        self._symbols, self._constraints, self._init_fns = [], [], []
        return shapes


    def _energy_fn(
        self, position, *args,
        communication=None, neighbor_order=None, newton_pair=None,
    ):
        """Evaluates the model and forms its engine-facing outputs.

        Forces use ``F = -dE/dR``. The virial uses
        ``V = -dE_local/d(exx, eyy, ezz, exy, exz, eyz)`` under a
        lower-triangular affine deformation. Energy, force, and virial units
        follow ``unit_style``. With Newton off, all valid particle energies
        contribute to owned forces. In both Newton modes, only owned particle
        energies contribute to the rank-local virial.
        """
        # Decode engine inputs.
        n_particle_fields = len(self._effective_particle_fields)
        n_global_fields = len(self._effective_global_fields)
        field_values = args[:n_particle_fields]
        global_values = args[n_particle_fields:n_particle_fields + n_global_fields]
        scalar_start = n_particle_fields + n_global_fields
        n_local, n_ghost = args[scalar_start:scalar_start + 2]
        graph_args = args[scalar_start + 2:]
        particle_data = {
            field.name: value
            for field, value in zip(
                self._effective_particle_fields, field_values
            )
        }
        global_data = {
            field.name: value
            for field, value in zip(
                self._effective_global_fields, global_values
            )
        }

        # Engine inputs place owned atoms first, followed by ghosts and padding.
        # Graph pruning relies on this ordering, so construct both masks first.
        valid_mask = jnp.arange(position.shape[0]) < (n_local + n_ghost)
        local_mask = jnp.arange(position.shape[0]) < n_local

        graph, build_statistics = self.graph_type.create_from_args(
            self.r_cutoff, neighbor_order, position,
            local_mask, valid_mask, newton_pair, *graph_args)
        graph = lax.stop_gradient(graph)
        particle_data["local_mask"] = local_mask
        particle_data["valid_mask"] = valid_mask

        # Evaluate energy and auxiliary outputs under the affine deformation
        # used for the virial derivative. Graph connectivity remains fixed
        # during differentiation.
        def model_outputs(pos, strain):
            # The lower-triangular deformation produces xx, yy, zz, xy, xz,
            # and yz in LAMMPS order. The one-sided shear placement avoids the
            # factor of two introduced by a symmetric shear tensor.
            exx, eyy, ezz, exy, exz, eyz = strain
            strained_position = jnp.stack(
                (
                    (1.0 + exx) * pos[:, 0],
                    exy * pos[:, 0] + (1.0 + eyy) * pos[:, 1],
                    exz * pos[:, 0] + eyz * pos[:, 1]
                    + (1.0 + ezz) * pos[:, 2],
                ),
                axis=1,
            )
            kwargs = {}
            if communication is not None:
                kwargs["comm"] = communication
            if self._effective_global_fields:
                kwargs["global_data"] = global_data
            model_result = self.energy_fn(
                strained_position, particle_data, graph, **kwargs
            )
            if self.has_aux:
                particle_energy, auxiliary_outputs = model_result
            else:
                particle_energy = model_result
                auxiliary_outputs = {}

            assert particle_energy.shape == local_mask.shape, (
                f"Per particle energies have shape {particle_energy.shape}, "
                f"but should have shape {local_mask.shape}."
            )

            return particle_energy, auxiliary_outputs

        # Differentiate the vector of particle energies once. The VJP seed
        # selects which particle energies contribute to each derivative. With
        # Newton on, owned energies determine forces and virial because the
        # engine returns ghost force contributions to their owners. With Newton
        # off, all valid energies determine owned forces because no reverse
        # force communication follows. The local virial still uses owned energy.
        strain = jnp.zeros((6,), dtype=position.dtype)
        energy, pullback, aux = jax.vjp(
            model_outputs, position, strain, has_aux=True
        )
        local_seed = -jnp.where(
            local_mask, jnp.ones_like(energy), jnp.zeros_like(energy)
        )
        total_seed = -jnp.where(
            valid_mask, jnp.ones_like(energy), jnp.zeros_like(energy)
        )
        if newton_pair:
            force, virial = pullback(local_seed)
        else:
            force, _ = pullback(total_seed)
            _, virial = pullback(local_seed)

        # PARTICLE outputs keep their particle axis. LOCAL outputs are
        # rank-local contributions that the adapter reduces once. GLOBAL
        # outputs are already complete and must not be reduced again.
        declared_scopes = {
            field.name: field.scope for field in self._effective_output_fields
        }
        particle_outputs = dict(U=energy, F=force)
        configuration_outputs = {"V": virial}
        for key, value in aux.items():
            if key not in declared_scopes:
                raise ValueError(
                    f"Auxiliary output '{key}' is not declared in output_fields"
                )
            if declared_scopes[key] is OutputScope.PARTICLE:
                particle_outputs[key] = value
            else:
                configuration_outputs[key] = value

        return particle_outputs, configuration_outputs, build_statistics

    def _validate_position_dtype(self):
        """Validates the dtype used for positions and differentiation."""
        dtype = jnp.dtype(self.position_dtype)
        if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise ValueError("position_dtype must be float32 or float64")
        if dtype == jnp.dtype(jnp.float64) and not jax.config.x64_enabled:
            raise ValueError(
                "position_dtype uses float64, but JAX x64 support is "
                "disabled. Enable jax_enable_x64 before export"
            )
        return dtype

    def _validate_particle_fields(self):
        """Validates particle input declarations for one export."""
        fields = tuple(self.particle_fields)
        if not all(isinstance(field, ParticleField) for field in fields):
            raise TypeError(
                "particle_fields entries must be ParticleField descriptors"
            )

        names = [field.name for field in fields]
        invalid = [
            name
            for name in names
            if not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
        ]
        if invalid:
            raise ValueError(
                "Invalid particle field names: "
                + ", ".join(repr(name) for name in invalid)
            )
        if "species" in names:
            raise ValueError("species is implicit and must not be registered")
        if len(set(names)) != len(names):
            raise ValueError("particle_fields must not contain duplicates")

        for field in fields:
            if jnp.dtype(field.dtype) != jnp.dtype(jnp.int32):
                raise ValueError(
                    f"Particle field '{field.name}' must have dtype int32"
                )
        return (ParticleField("species", jnp.int32), *fields)

    def _validate_global_fields(self):
        """Validates global input declarations for one export."""
        fields = tuple(self.global_fields)
        if not all(isinstance(field, GlobalField) for field in fields):
            raise TypeError(
                "global_fields entries must be GlobalField descriptors"
            )

        names = [field.name for field in fields]
        invalid = [
            name
            for name in names
            if not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
        ]
        if invalid:
            raise ValueError(
                "Invalid global field names: "
                + ", ".join(repr(name) for name in invalid)
            )
        if len(set(names)) != len(names):
            raise ValueError("global_fields must not contain duplicates")

        for field in fields:
            dtype = jnp.dtype(field.dtype)
            if dtype not in (
                jnp.dtype(jnp.float32),
                jnp.dtype(jnp.float64),
                jnp.dtype(jnp.int32),
            ):
                raise ValueError(
                    f"Global field '{field.name}' must have dtype float32, "
                    "float64, or int32"
                )
            if (
                dtype == jnp.dtype(jnp.float64)
                and not jax.config.x64_enabled
            ):
                raise ValueError(
                    f"Global field '{field.name}' uses float64, but JAX x64 "
                    "support is disabled. Enable jax_enable_x64 before export"
                )
        return fields

    def _validate_output_fields(self):
        """Validates auxiliary output declarations for one export."""
        fields = tuple(self.output_fields)
        if not all(isinstance(field, OutputField) for field in fields):
            raise TypeError(
                "output_fields entries must be OutputField descriptors"
            )

        names = [field.name for field in fields]
        invalid = [
            name
            for name in names
            if not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
        ]
        if invalid:
            raise ValueError(
                "Invalid output field names: "
                + ", ".join(repr(name) for name in invalid)
            )

        reserved = sorted(set(names).intersection(("U", "F", "V")))
        if reserved:
            raise ValueError(
                "Output fields use reserved names: " + ", ".join(reserved)
            )
        if len(set(names)) != len(names):
            raise ValueError("output_fields must not contain duplicates")
        for field in fields:
            if not isinstance(field.scope, OutputScope):
                raise TypeError(
                    f"Output field '{field.name}' has an invalid scope"
                )
            if not isinstance(field.extensive, bool):
                raise TypeError(
                    f"Output field '{field.name}' extensive flag must be bool"
                )
            if field.scope is OutputScope.PARTICLE and field.extensive:
                raise ValueError(
                    f"Particle output '{field.name}' cannot be extensive"
                )
        if bool(fields) != bool(self.has_aux):
            raise ValueError(
                "has_aux must be true exactly when output_fields are declared"
            )
        return fields

    def _validate_neighbor_metadata(self):
        """Validates graph metadata used by every exported variant."""
        neighbor_orders = tuple(self.nbr_order)
        if len(neighbor_orders) != 2 or any(
            not isinstance(order, int) or isinstance(order, bool) or order < 1
            for order in neighbor_orders
        ):
            raise ValueError(
                "nbr_order must contain two positive integer orders for "
                "Newton on and Newton off"
            )
        if (
            not isinstance(self.r_cutoff, (int, float))
            or isinstance(self.r_cutoff, bool)
            or not math.isfinite(self.r_cutoff)
            or self.r_cutoff <= 0
        ):
            raise ValueError("r_cutoff must be finite and greater than zero")
        if (
            isinstance(self.graph_type, type)
            and issubclass(self.graph_type, graphs.DeviceSparseNeighborList)
        ):
            raise ValueError(
                "DeviceSparseNeighborList is experimental and is not "
                "supported by the connector"
            )
        return neighbor_orders

    def _validate_export_options(self, communication, custom_calls, platforms):
        """Validates and normalizes platform and custom-call options."""
        custom_calls = tuple(custom_calls)
        if isinstance(platforms, str):
            raise TypeError("platforms must be an iterable of platform names")

        platforms = tuple(platforms)
        if not platforms:
            raise ValueError("platforms must not be empty")
        if len(set(platforms)) != len(platforms):
            raise ValueError("platforms must not contain duplicates")

        unknown = set(platforms).difference(_SUPPORTED_PLATFORMS)
        if unknown:
            raise ValueError(
                "Unsupported export platform(s): "
                + ", ".join(sorted(unknown))
            )
        platforms = tuple(
            platform
            for platform in _SUPPORTED_PLATFORMS
            if platform in platforms
        )

        if "cpu" in platforms and any(
            target in OPENEQUIVARIANCE_CUSTOM_CALLS
            for target in custom_calls
        ):
            raise ValueError(
                "CPU export is unavailable for OpenEquivariance custom calls"
            )

        signature = inspect.signature(self.energy_fn)
        supports_comm = "comm" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if communication and not supports_comm:
            raise TypeError(
                "communication=True requires energy_fn(..., comm=None)"
            )
        if self.communication_required and not communication:
            raise ValueError(
                "communication_required=True requires communication=True"
            )
        return custom_calls, platforms

    def export(
        self,
        *,
        communication=False,
        custom_calls=(),
        platforms=("cuda",),
    ) -> None:
        """Exports model variants specialized by communication and Newton mode.

        Every bundle contains the two variants without model communication,
        one for each Newton mode. ``communication=True`` also adds the
        communication-enabled Newton-on variant. ``custom_calls`` lists
        additional model-specific FFI targets, while ``platforms`` selects
        CPU, CUDA, or both implementations in each variant.
        """
        # Clear metadata inferred by earlier traces before starting an export.
        # JAX may trace each variant more than once, and callers may export the
        # same instance again. Compare variants only with metadata inferred
        # during the current export.
        self._export_output_fields = None
        self._effective_position_dtype = self._validate_position_dtype()
        self._effective_particle_fields = self._validate_particle_fields()
        self._effective_global_fields = self._validate_global_fields()
        self._effective_output_fields = self._validate_output_fields()
        neighbor_orders = self._validate_neighbor_metadata()
        custom_calls, platforms = self._validate_export_options(
            communication, custom_calls, platforms
        )

        # Export the two variants that do not use model-feature communication.
        proto = model_proto.Model()
        proto.format_version = MODEL_FORMAT_VERSION
        proto.unit_style = self.unit_style

        newton_off, _ = self._export_variant(
            name="comm_off_newton_off",
            neighbor_order=neighbor_orders[1],
            newton_pair=False,
            communication_enabled=False,
            custom_calls=custom_calls,
            platforms=platforms,
        )
        proto.variants.add().CopyFrom(newton_off)

        newton_on, _ = self._export_variant(
            name="comm_off_newton_on",
            neighbor_order=neighbor_orders[0],
            newton_pair=True,
            communication_enabled=False,
            custom_calls=custom_calls,
            platforms=platforms,
        )
        proto.variants.add().CopyFrom(newton_on)

        # Model loading exposes provisional properties before the adapter
        # selects a variant. Mirror the conventional Newton-on configuration.
        proto.mlir_module_serialized = newton_on.mlir_module_serialized
        proto.neighbor_list.CopyFrom(newton_on.neighbor_list)
        proto.output_fields.extend(self._export_output_fields)
        proto.quantities.extend(
            field.name for field in self._export_output_fields
        )
        proto.quantity_components.extend(
            field.components for field in self._export_output_fields
        )
        proto.uses_communication = False
        proto.communication_buffer_width = 0
        proto.calling_convention_version = newton_on.calling_convention_version
        proto.particle_fields.extend(newton_on.particle_fields)
        proto.global_fields.extend(newton_on.global_fields)
        proto.platforms.extend(newton_on.platforms)

        # Export the communication variant.
        if communication:
            communicating, trace_state = self._export_variant(
                name="comm_on_newton_on",
                neighbor_order=1,
                newton_pair=True,
                communication_enabled=True,
                custom_calls=custom_calls + COMMUNICATION_CUSTOM_CALLS,
                platforms=platforms,
            )
            communicating.uses_communication = True
            proto.variants.add().CopyFrom(communicating)
            proto.requires_communication = bool(
                self.communication_required or trace_state.reduce_widths
            )

        # Publish the bundle after all variants succeed.
        self._proto = proto

    def _add_particle_metadata(self, target):
        for descriptor in self._effective_particle_fields:
            field = target.particle_fields.add()
            field.name = descriptor.name
            field.dtype = model_proto.Model.ParticleDtype.INT32
            field.components = 1

    def _add_global_metadata(self, target):
        for descriptor in self._effective_global_fields:
            field = target.global_fields.add()
            field.name = descriptor.name
            dtype = jnp.dtype(descriptor.dtype)
            if dtype == jnp.dtype(jnp.float32):
                field.dtype = model_proto.Model.GlobalDtype.GLOBAL_FLOAT32
            elif dtype == jnp.dtype(jnp.float64):
                field.dtype = model_proto.Model.GlobalDtype.GLOBAL_FLOAT64
            elif dtype == jnp.dtype(jnp.int32):
                field.dtype = model_proto.Model.GlobalDtype.GLOBAL_INT32
            else:
                raise AssertionError("validated global field dtype changed")

    def _variant_inputs(self, neighbor_order, newton_pair):
        """Builds graph metadata and symbolic inputs for one variant."""
        self._symbols: List[str] = []
        self._constraints: List[str] = []
        self._init_fns: List[Callable] = []
        metadata = model_proto.Model()
        metadata.format_version = MODEL_FORMAT_VERSION
        metadata.neighbor_list.cutoff = self.r_cutoff
        metadata.neighbor_list.nbr_order.append(neighbor_order)
        self.graph_type.set_properties(
            metadata,
            include_pair_type=self.include_pair_type,
            newton_pair=newton_pair,
        )
        self._add_particle_metadata(metadata)
        self._add_global_metadata(metadata)

        # Ghost energies stay in the differentiated model so their position
        # dependence contributes to owned forces. The final energy seed masks
        # ghost and padding rows to avoid counting non-owned energy twice.
        self._add_shapes(self._define_input_shapes())
        self._add_shapes(
            self.graph_type.create_symbolic_input_format,
            include_pair_type=self.include_pair_type,
        )
        return metadata, self._create_shapes()

    def _export_variant(
        self, *, name, neighbor_order, newton_pair, communication_enabled,
        custom_calls, platforms,
    ):
        """Traces one model variant and returns its metadata."""
        metadata, shapes = self._variant_inputs(neighbor_order, newton_pair)

        # Trace the variant.
        trace_state = _ExportComputation(
            self,
            neighbor_order,
            newton_pair,
            enabled=communication_enabled,
        )
        export_fn = jax.jit(trace_state)

        exported: export.Exported = export.export(
            export_fn,
            platforms=platforms,
            disabled_checks=tuple(
                export.DisabledSafetyCheck.custom_call(target)
                for target in custom_calls
            ),
        )(*shapes)

        # Recover output shapes and graph statistics inferred during tracing.
        particle_outputs, configuration_outputs, statistics = (
            exported.out_tree.unflatten(exported.out_avals)
        )

        metadata.neighbor_list.statistics_keys.extend(statistics.keys())
        declared_scopes = {
            field.name: field.scope
            for field in self._effective_output_fields
        }
        actual_aux = (
            set(particle_outputs).union(configuration_outputs)
            .difference(("U", "F", "V"))
        )
        if actual_aux != set(declared_scopes):
            raise ValueError(
                "Auxiliary output keys do not match output_fields: "
                f"declared={sorted(declared_scopes)}, "
                f"returned={sorted(actual_aux)}"
            )

        # Preserve the traced result order in the protobuf. The connector uses
        # the recorded order to match executable buffers with descriptors before
        # exposing named outputs.
        output_fields = []
        ordered_outputs = (
            tuple(particle_outputs.items())
            + tuple(configuration_outputs.items())
        )
        for output_name, aval in ordered_outputs:
            if output_name in ("U", "F"):
                scope = OutputScope.PARTICLE
            elif output_name == "V":
                scope = OutputScope.LOCAL
            else:
                scope = declared_scopes[output_name]

            shape = tuple(aval.shape)
            if scope is OutputScope.PARTICLE:
                if not shape or shape[0] != shapes[0].shape[0]:
                    raise ValueError(
                        f"Particle output '{output_name}' must use the "
                        "particle dimension as its first axis"
                    )
                value_shape = shape[1:]
            else:
                value_shape = shape
            if any(not isinstance(dim, int) for dim in value_shape):
                raise ValueError(
                    f"Output '{output_name}' has dynamic value dimensions"
                )

            dtype = jnp.dtype(aval.dtype)
            if dtype not in (
                    jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
                raise TypeError(
                    f"Output '{output_name}' must use float32 or float64"
                )

            proto_scope = {
                OutputScope.PARTICLE:
                    model_proto.Model.OutputScope.PARTICLE,
                OutputScope.LOCAL:
                    model_proto.Model.OutputScope.LOCAL,
                OutputScope.GLOBAL:
                    model_proto.Model.OutputScope.GLOBAL,
            }[scope]
            field = model_proto.Model.OutputField()
            field.name = output_name
            field.scope = proto_scope
            field.dimensions.extend(value_shape)
            components = 1
            for dim in value_shape:
                if dim <= 0:
                    raise ValueError(
                        f"Output '{output_name}' dimensions must be positive"
                    )
                components *= dim
            if components > 2**31 - 1:
                raise ValueError(
                    f"Output '{output_name}' has too many components"
                )
            field.components = components
            if output_name == "V":
                field.extensive = True
            elif output_name not in ("U", "F"):
                field.extensive = next(
                    descriptor.extensive
                    for descriptor in self._effective_output_fields
                    if descriptor.name == output_name
                )
            output_fields.append(field)

        serialized_fields = [
            field.SerializeToString() for field in output_fields
        ]
        if self._export_output_fields is not None:
            previous_fields = [
                field.SerializeToString()
                for field in self._export_output_fields
            ]
            if serialized_fields != previous_fields:
                raise ValueError(
                    "All exported variants must return identical output fields"
                )
        else:
            self._export_output_fields = output_fields

        # Build self-contained variant metadata.
        variant = model_proto.Model.ModelVariant()
        variant.name = name
        variant.newton_pair = newton_pair
        # JAX structurally checks every StableHLO custom call against the
        # target-specific DisabledSafetyCheck entries supplied above. Avoid
        # duplicating that check with version-sensitive MLIR text parsing.
        variant.mlir_module_serialized = exported.mlir_module_serialized
        variant.calling_convention_version = exported.calling_convention_version
        variant.platforms.extend(platforms)
        if communication_enabled:
            # Gather and reduce share one runtime width limit. Store the largest
            # number of scalar values packed by either call. Keeping the lists
            # separate also shows whether reduction makes communication
            # required.
            communication_widths = (
                *trace_state.gather_widths,
                *trace_state.reduce_widths,
            )
            if not communication_widths:
                raise ValueError(
                    "communication=True was requested, but the exported "
                    "model did not call a communication collective"
                )
            variant.communication_buffer_width = max(communication_widths)
        variant.neighbor_list.CopyFrom(metadata.neighbor_list)
        variant.particle_fields.extend(metadata.particle_fields)
        variant.global_fields.extend(metadata.global_fields)
        return variant, trace_state

    def __str__(self):
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        return str(self._proto)

    def save(self, file: str) -> None:
        """Saves the exported protocol buffer to ``file``."""
        assert self._proto is not None, (
            "Model has not been exported yet. Please call `export()` first."
        )

        with open(file, "wb") as f:
            f.write(self._proto.SerializeToString())
