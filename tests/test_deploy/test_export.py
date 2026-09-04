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
import itertools

import jax
from jax import numpy as jnp, Array

from chemtrain.deploy import comm, graphs, exporter

from jax_md_mod import custom_partition
from jax_md import space, partition, util as md_util

import numpy as onp

import pytest


def model_neighborlist_pp(displacement: space.DisplacementFn,
                          r_cutoff: float,
                          positions_test: jnp.ndarray = None,
                          neighbor_test: partition.NeighborList = None,
                          max_edge_multiplier: float = 1.25,
                          max_edges=None,
                          avg_num_neighbors: float = None,
                          ):
    """Export test model."""
    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        _avg_num_neighbors, _ = custom_partition.test_graph_statistics(
            displacement, positions_test, neighbor_test,
            r_cutoff, max_edge_multiplier=max_edge_multiplier
        )

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors

    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        vectors, senders, receivers = custom_partition.readout_vectors(
            displacement, r_cutoff, position, neighbor, species,
            mask, max_edges=max_edges, **dynamic_kwargs
        )

        vectors /= r_cutoff

        pot = (jnp.linalg.norm(vectors, axis=-1) - 1.0) ** 2
        return jax.ops.segment_sum(pot, senders, num_segments=position.shape[0])

    return jax.jit(model)





class TestExport:

    def test_export_platform_validation(self, setup_export):
        model = setup_export(max_edges=None)
        with pytest.raises(ValueError, match="must not be empty"):
            model.export(platforms=())
        with pytest.raises(ValueError, match="duplicates"):
            model.export(platforms=("cuda", "cuda"))
        with pytest.raises(ValueError, match="Unsupported"):
            model.export(platforms=("rocm",))

    def test_float64_global_field_requires_x64(self, setup_export):
        class GlobalModel(setup_export):
            global_fields = (
                exporter.GlobalField("bias_strength", jnp.float64),
            )

        with jax.enable_x64(False):
            with pytest.raises(ValueError, match="JAX x64 support is disabled"):
                GlobalModel(max_edges=None).export(platforms=("cpu",))

    def test_float64_positions_require_x64(self, setup_export):
        class Float64Model(setup_export):
            position_dtype = jnp.float64

        with jax.enable_x64(False):
            with pytest.raises(ValueError, match="JAX x64 support is disabled"):
                Float64Model(max_edges=None).export(platforms=("cpu",))

    @pytest.mark.parametrize("dtype", (jnp.int32, jnp.float16))
    def test_position_dtype_must_be_float32_or_float64(
        self, setup_export, dtype
    ):
        class InvalidModel(setup_export):
            position_dtype = dtype

        with pytest.raises(ValueError, match="float32 or float64"):
            InvalidModel(max_edges=None).export(platforms=("cpu",))

    def test_float64_global_field_exports_with_x64(self, setup_export):
        class GlobalModel(setup_export):
            global_fields = (
                exporter.GlobalField("bias_strength", jnp.float64),
            )

            def energy_fn(
                self, pos, particle_data, graph, global_data=None
            ):
                energy = super().energy_fn(pos, particle_data, graph)
                return energy * global_data["bias_strength"]

        with jax.enable_x64():
            model = GlobalModel(max_edges=None)
            model.export(platforms=("cpu",))

        [field] = model._proto.global_fields
        assert field.name == "bias_strength"
        assert field.dtype == model._proto.GLOBAL_FLOAT64

    def test_cpu_cuda_share_one_default_variant(self, setup_export):
        model = setup_export(max_edges=None)
        model.export(platforms=("cuda", "cpu"))
        assert model._proto.format_version == 5
        assert list(model._proto.platforms) == ["cpu", "cuda"]
        assert len(model._proto.variants) == 2
        assert list(model._proto.variants[0].platforms) == ["cpu", "cuda"]

    def test_communication_variant_uses_requested_platforms(
        self, setup_export
    ):
        class CommunicatingModel(setup_export):
            def energy_fn(self, pos, particle_data, graph, comm=None):
                energy = super().energy_fn(pos, particle_data, graph)
                if comm is not None:
                    energy = comm.gather(energy)
                return energy

        model = CommunicatingModel(max_edges=None)
        model.export(communication=True, platforms=("cpu", "cuda"))
        assert [variant.name for variant in model._proto.variants] == [
            "comm_off_newton_off",
            "comm_off_newton_on",
            "comm_on_newton_on",
        ]
        for variant in model._proto.variants:
            assert list(variant.platforms) == ["cpu", "cuda"]

    def test_communication_variant_supports_cpu_only(self, setup_export):
        class CommunicatingModel(setup_export):
            def energy_fn(self, pos, particle_data, graph, comm=None):
                energy = super().energy_fn(pos, particle_data, graph)
                if comm is not None:
                    energy = comm.gather(energy)
                return energy

        model = CommunicatingModel(max_edges=None)
        model.export(communication=True, platforms=("cpu",))

        assert [variant.name for variant in model._proto.variants] == [
            "comm_off_newton_off",
            "comm_off_newton_on",
            "comm_on_newton_on",
        ]
        assert list(model._proto.variants[2].platforms) == ["cpu"]

    @pytest.fixture(scope="function")
    def setup_export(self):
        class ExportedModel(exporter.Exporter):

            graph_type = graphs.SimpleSparseNeighborList
            r_cutoff = 5.0
            unit_style = "real"
            nbr_order = [1, 1]

            def __init__(self, max_edge_multiplier=None, max_edges=None):
                self.max_edge_multiplier = max_edge_multiplier
                self.max_edges = max_edges

            def energy_fn(self, pos, particle_data,
                          graph: graphs.SimpleSparseNeighborList):
                neighbors = graph.to_neighborlist()
                displacement_fn, _ = space.free()

                model = model_neighborlist_pp(
                    displacement_fn, self.r_cutoff,
                    max_edges=self.max_edges, avg_num_neighbors=20.0
                )

                pot = model(
                    pos, neighbors, species=particle_data["species"])
                return pot

        yield ExportedModel

    def test_no_max_edges(self, tmp_path, setup_export):
        model = setup_export(max_edges=None)

        model.export()
        model.save(tmp_path / "exported_no_max_edges.ptb")

    def test_export_custom_call_allowlist(self, setup_export):
        model = setup_export(max_edges=None)
        model.export(custom_calls=exporter.OPENEQUIVARIANCE_CUSTOM_CALLS)
        assert [variant.name for variant in model._proto.variants] == [
            "comm_off_newton_off", "comm_off_newton_on"
        ]

    def test_export_resets_output_metadata(self, setup_export):
        model = setup_export(max_edges=None)
        model._export_output_fields = ["stale_output"]

        model.export()

        assert [field.name for field in model._proto.output_fields] == [
            "F", "U", "V"
        ]
        assert [field.scope for field in model._proto.output_fields] == [
            model._proto.PARTICLE, model._proto.PARTICLE, model._proto.LOCAL
        ]
        assert [list(field.dimensions) for field in model._proto.output_fields] == [
            [3], [], [6]
        ]
        assert [field.extensive for field in model._proto.output_fields] == [
            False, False, True
        ]
        assert "dtype" not in (
            model._proto.output_fields[0].DESCRIPTOR.fields_by_name
        )
        assert list(model._proto.quantities) == ["F", "U", "V"]

    def test_symbolic_max_edges(self, tmp_path, setup_export):
        class ExportedModelSymbolic(setup_export):

            def __init__(self):
                super().__init__(max_edge_multiplier=0.5, max_edges=None)

        model = ExportedModelSymbolic()

        model.export()
        model.save(tmp_path / "exported_no_max_edges.ptb")

    def test_static_max_edges(self, tmp_path, setup_export):
        # A static edge capacity would freeze the runtime edge count. The
        # exporter must reject it instead of producing a fixed-shape model.

        class ExportedModelStatic(setup_export):
            def __init__(self):
                super().__init__(max_edge_multiplier=None, max_edges=10)

        model = ExportedModelStatic()

        with pytest.raises(TypeError, match="max_edges must be symbolic"):
            model.export()

        with pytest.raises(AssertionError, match="has not been exported yet"):
            model.save(tmp_path / "exported_no_max_edges.ptb")

    def test_communication_export_records_variants_and_widths(self, setup_export):
        class CommunicatingModel(setup_export):
            nbr_order = [2, 3]

            def energy_fn(self, pos, particle_data, graph, comm=None):
                energy = super().energy_fn(pos, particle_data, graph)
                if comm is not None:
                    features = jnp.stack([energy, energy], axis=1)
                    features = comm.gather(features)
                    energy = comm.gather(features[:, 0])
                return energy

        model = CommunicatingModel(max_edges=None)
        model.export(communication=True)

        assert [variant.name for variant in model._proto.variants] == [
            "comm_off_newton_off",
            "comm_off_newton_on",
            "comm_on_newton_on",
        ]
        newton_off, newton_on, communicating = model._proto.variants
        assert list(newton_off.neighbor_list.nbr_order) == [3]
        assert list(newton_on.neighbor_list.nbr_order) == [2]
        assert list(communicating.neighbor_list.nbr_order) == [1]
        assert not newton_off.newton_pair
        assert newton_on.newton_pair
        assert communicating.newton_pair
        assert newton_off.neighbor_list.half_list
        assert not newton_on.neighbor_list.half_list
        assert newton_off.mlir_module_serialized
        assert newton_on.mlir_module_serialized
        assert communicating.mlir_module_serialized
        assert newton_off.calling_convention_version > 0
        assert newton_on.calling_convention_version > 0
        assert communicating.calling_convention_version > 0
        assert communicating.uses_communication
        assert communicating.communication_buffer_width == 2
        assert not model._proto.uses_communication
        assert list(model._proto.neighbor_list.nbr_order) == [2]
        assert (
            model._proto.mlir_module_serialized
            == newton_on.mlir_module_serialized
        )
        assert (
            model._proto.calling_convention_version
            == newton_on.calling_convention_version
        )

        # Re-exporting one instance must replace tracing metadata. Appending a
        # second trace would retain or inflate stale communication capacity.
        model.export(communication=True)
        communicating = model._proto.variants[2]
        assert communicating.communication_buffer_width == 2

    def test_reduce_only_variant_is_communication_required(self, setup_export):
        class ReducedModel(setup_export):
            def energy_fn(self, pos, particle_data, graph, comm=None):
                energy = super().energy_fn(pos, particle_data, graph)
                if comm is not None:
                    reduced = comm.reduce(jnp.sum(energy))
                    energy = energy + jnp.float32(0) * reduced
                return energy

        model = ReducedModel(max_edges=None)
        model.export(communication=True)

        assert model._proto.requires_communication
        assert model._proto.variants[2].communication_buffer_width == 1

    def test_communication_width_includes_gather_and_reduce(self, setup_export):
        class MixedModel(setup_export):
            def energy_fn(self, pos, particle_data, graph, comm=None):
                energy = super().energy_fn(pos, particle_data, graph)
                if comm is not None:
                    gathered = comm.gather(jnp.stack((energy, energy), axis=1))
                    reduced = comm.reduce(jnp.arange(5, dtype=energy.dtype))
                    energy = gathered[:, 0] + jnp.float32(0) * jnp.sum(reduced)
                return energy

        model = MixedModel(max_edges=None)
        model.export(communication=True)

        assert model._proto.requires_communication
        assert model._proto.variants[2].communication_buffer_width == 5

    def test_communication_variant_requires_a_collective(self, setup_export):
        class NoCollectiveModel(setup_export):
            def energy_fn(self, pos, particle_data, graph, comm=None):
                del comm
                return super().energy_fn(pos, particle_data, graph)

        with pytest.raises(ValueError, match="did not call a communication"):
            NoCollectiveModel(max_edges=None).export(communication=True)

    def test_explicit_communication_requirement_needs_comm_variant(
            self, setup_export):
        class RequiredModel(setup_export):
            communication_required = True

        with pytest.raises(ValueError, match="requires communication=True"):
            RequiredModel(max_edges=None).export()

    def test_auxiliary_output_scopes_are_exported(self, setup_export):
        class AuxiliaryModel(setup_export):
            has_aux = True
            output_fields = (
                exporter.OutputField("particle_value"),
                exporter.OutputField(
                    "local_value", exporter.OutputScope.LOCAL, extensive=True
                ),
                exporter.OutputField("global_value", exporter.OutputScope.GLOBAL),
            )

            def energy_fn(self, pos, particle_data, graph):
                energy = super().energy_fn(pos, particle_data, graph)
                return energy, {
                    "particle_value": energy,
                    "local_value": jnp.sum(energy),
                    "global_value": jnp.sum(energy),
                }

        model = AuxiliaryModel(max_edges=None)
        model.export()
        fields = {field.name: field for field in model._proto.output_fields}
        assert fields["particle_value"].scope == model._proto.PARTICLE
        assert fields["local_value"].scope == model._proto.LOCAL
        assert fields["global_value"].scope == model._proto.GLOBAL
        assert list(fields["particle_value"].dimensions) == []
        assert list(fields["local_value"].dimensions) == []
        assert fields["local_value"].extensive
        assert not fields["global_value"].extensive

    def test_particle_output_cannot_be_extensive(self, setup_export):
        class InvalidOutputModel(setup_export):
            has_aux = True
            output_fields = (
                exporter.OutputField("value", extensive=True),
            )

        with pytest.raises(ValueError, match="cannot be extensive"):
            InvalidOutputModel(max_edges=None).export()

    def test_undeclared_auxiliary_output_is_rejected(self, setup_export):
        class AuxiliaryModel(setup_export):
            has_aux = True
            output_fields = (exporter.OutputField("declared"),)

            def energy_fn(self, pos, particle_data, graph):
                energy = super().energy_fn(pos, particle_data, graph)
                return energy, {"undeclared": energy}

        with pytest.raises(ValueError, match="is not declared"):
            AuxiliaryModel(max_edges=None).export()

    def test_zero_length_auxiliary_output_is_rejected(self, setup_export):
        class AuxiliaryModel(setup_export):
            has_aux = True
            output_fields = (
                exporter.OutputField("empty", exporter.OutputScope.LOCAL),
            )

            def energy_fn(self, pos, particle_data, graph):
                energy = super().energy_fn(pos, particle_data, graph)
                return energy, {"empty": jnp.zeros((0,), dtype=energy.dtype)}

        with pytest.raises(ValueError, match="dimensions must be positive"):
            AuxiliaryModel(max_edges=None).export()

    def test_three_argument_model_rejects_communication(self, setup_export):
        model = setup_export(max_edges=None)
        with pytest.raises(TypeError, match="comm=None"):
            model.export(communication=True)

    def test_named_particle_fields_are_ordered_in_all_variants(self, setup_export):
        class ParticleDataModel(setup_export):
            particle_fields = (exporter.ParticleField("residue_id"),)

            def energy_fn(self, pos, particle_data, graph, comm=None):
                energy = super().energy_fn(pos, particle_data, graph)
                if comm is not None:
                    energy = comm.gather(energy)
                return energy + jnp.float32(0) * particle_data["residue_id"]

        model = ParticleDataModel(max_edges=None)
        model.export(communication=True)

        assert model._proto.format_version == 5
        assert model._proto.mlir_module_serialized
        assert model._proto.calling_convention_version > 0
        assert [field.name for field in model._proto.particle_fields] == [
            "species", "residue_id"
        ]
        for variant in model._proto.variants:
            assert [field.name for field in variant.particle_fields] == [
                "species", "residue_id"
            ]
            assert all(field.dtype == 1 for field in variant.particle_fields)
            assert all(field.components == 1 for field in variant.particle_fields)

    def test_species_is_implicit_when_no_extras_are_registered(
            self, setup_export):
        model = setup_export(max_edges=None)
        model.export()
        assert [field.name for field in model._proto.particle_fields] == [
            "species"
        ]

    @pytest.mark.parametrize(
        "fields, error, message",
        [
            (("residue_id",), TypeError, "ParticleField descriptors"),
            ((exporter.ParticleField("species"),), ValueError, "implicit"),
            ((exporter.ParticleField("residue_id"),
              exporter.ParticleField("residue_id")), ValueError, "duplicates"),
            ((exporter.ParticleField("residue-id"),), ValueError,
             "Invalid particle field"),
            ((exporter.ParticleField([]),), ValueError,
             "Invalid particle field"),
            ((exporter.ParticleField("charge", jnp.float32),), ValueError,
             "must have dtype int32"),
        ],
    )
    def test_invalid_particle_field_registry(
            self, setup_export, fields, error, message):
        class InvalidParticleDataModel(setup_export):
            particle_fields = fields

        with pytest.raises(error, match=message):
            InvalidParticleDataModel(max_edges=None).export()

    def test_arbitrary_portable_integer_field_is_supported(self, setup_export):
        class ChargeCategoryModel(setup_export):
            particle_fields = (exporter.ParticleField("charge_category"),)

            def energy_fn(self, pos, particle_data, graph):
                energy = super().energy_fn(pos, particle_data, graph)
                return energy + jnp.float32(0) * particle_data["charge_category"]

        model = ChargeCategoryModel(max_edges=None)
        model.export()
        assert [field.name for field in model._proto.particle_fields] == [
            "species", "charge_category"
        ]

    def test_pair_type_capability_is_exported(self, setup_export):
        class PairTypeModel(setup_export):
            include_pair_type = True

            def energy_fn(self, pos, particle_data, graph):
                energy = super().energy_fn(pos, particle_data, graph)
                return energy + jnp.float32(0) * jax.ops.segment_sum(
                    graph.pair_type, graph.senders, num_segments=pos.shape[0]
                )

        model = PairTypeModel(max_edges=None)
        model.export()

        assert model._proto.neighbor_list.include_pair_type
        assert model._proto.variants[0].neighbor_list.include_pair_type

    def test_dense_pair_type_payload_is_exported(self):
        class DensePairTypeModel(exporter.Exporter):
            graph_type = graphs.SimpleDenseNeighborList
            include_pair_type = True
            r_cutoff = 5.0
            nbr_order = [1, 1]

            def energy_fn(self, pos, particle_data, graph):
                del particle_data
                category_sum = jnp.sum(graph.pair_type, axis=1)
                return jnp.zeros(pos.shape[0], dtype=pos.dtype) + (
                    jnp.float32(0.0) * category_sum
                )

        model = DensePairTypeModel()
        model.export()

        neighbor_list = model._proto.variants[0].neighbor_list
        assert neighbor_list.type == model._proto.NeighborListType.SIMPLE_DENSE
        assert neighbor_list.include_pair_type

    def test_device_sparse_is_rejected_by_exporter(self, setup_export):
        class DeviceSparseModel(setup_export):
            graph_type = graphs.DeviceSparseNeighborList

        with pytest.raises(ValueError, match="not supported by the connector"):
            DeviceSparseModel(max_edges=None).export()

    def test_device_sparse_subclass_is_rejected_by_exporter(self, setup_export):
        class CustomDeviceSparse(graphs.DeviceSparseNeighborList):
            pass

        class DeviceSparseModel(setup_export):
            graph_type = CustomDeviceSparse

        with pytest.raises(ValueError, match="not supported by the connector"):
            DeviceSparseModel(max_edges=None).export()

    @pytest.mark.parametrize(
        "neighbor_orders", ([1], [1, 1, 1], [0, 1], [True, 1])
    )
    def test_invalid_neighbor_orders_are_rejected(
        self, setup_export, neighbor_orders
    ):
        class InvalidNeighborModel(setup_export):
            nbr_order = neighbor_orders

        with pytest.raises(ValueError, match="two positive integer orders"):
            InvalidNeighborModel(max_edges=None).export()

    @pytest.mark.parametrize("cutoff", (0.0, -1.0, float("inf"), float("nan")))
    def test_invalid_cutoffs_are_rejected(self, setup_export, cutoff):
        class InvalidCutoffModel(setup_export):
            r_cutoff = cutoff

        with pytest.raises(ValueError, match="finite and greater than zero"):
            InvalidCutoffModel(max_edges=None).export()
