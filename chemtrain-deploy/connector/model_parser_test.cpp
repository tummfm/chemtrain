/*
Copyright 2025 Multiscale Modeling of Fluid Materials, TU Munich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "connector/model_parser.h"

#include <string>
#include <utility>

#include "gtest/gtest.h"

namespace jcn {
namespace {

void AddCanonicalOutputs(Model* model) {
  auto* energy = model->add_output_fields();
  energy->set_name("U");
  energy->set_scope(Model::PARTICLE);
  energy->set_components(1);

  auto* force = model->add_output_fields();
  force->set_name("F");
  force->set_scope(Model::PARTICLE);
  force->add_dimensions(3);
  force->set_components(3);

  auto* virial = model->add_output_fields();
  virial->set_name("V");
  virial->set_scope(Model::LOCAL);
  virial->add_dimensions(6);
  virial->set_components(6);
  virial->set_extensive(true);

  model->add_quantities("U");
  model->add_quantities("F");
  model->add_quantities("V");
  model->add_quantity_components(1);
  model->add_quantity_components(3);
  model->add_quantity_components(6);
}

Model::ModelVariant* AddCommOffVariant(Model* model, const char* name,
                                       bool newton_pair) {
  auto* variant = model->add_variants();
  variant->set_name(name);
  variant->set_newton_pair(newton_pair);
  variant->set_mlir_module_serialized(name);
  variant->set_calling_convention_version(10);
  variant->add_platforms("cpu");
  variant->add_platforms("cuda");
  variant->mutable_neighbor_list()->CopyFrom(model->neighbor_list());
  variant->mutable_neighbor_list()->set_half_list(!newton_pair);
  variant->add_particle_fields()->CopyFrom(model->particle_fields(0));
  return variant;
}

Model ValidModel() {
  Model model;
  model.set_format_version(5);

  auto* field = model.add_particle_fields();
  field->set_name("species");
  field->set_dtype(Model::INT32);
  field->set_components(1);

  model.mutable_neighbor_list()->add_nbr_order(1);
  model.mutable_neighbor_list()->set_cutoff(5.0);
  model.mutable_neighbor_list()->set_type(Model::SIMPLE_SPARSE);
  model.mutable_neighbor_list()->set_half_list(false);

  AddCanonicalOutputs(&model);
  AddCommOffVariant(&model, "comm_off_newton_off", false);
  const auto* newton_on =
      AddCommOffVariant(&model, "comm_off_newton_on", true);
  model.set_mlir_module_serialized(newton_on->mlir_module_serialized());
  model.set_calling_convention_version(
      newton_on->calling_convention_version());
  model.add_platforms("cpu");
  model.add_platforms("cuda");
  return model;
}

TEST(ModelParserTest, RejectsCorruptWireData) {
  const std::string corrupt("\xff\xff\xff", 3);
  EXPECT_THROW(ParseModelProtobuf(corrupt.data(), corrupt.size()),
               std::runtime_error);
}

TEST(ModelParserTest, RejectsUnsupportedVersion) {
  Model model = ValidModel();
  model.set_format_version(1);
  const std::string bytes = model.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(bytes.data(), bytes.size()),
               std::runtime_error);
}

TEST(ModelParserTest, RejectsPreviousModelFormats) {
  Model model = ValidModel();
  model.set_format_version(4);
  const std::string bytes = model.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(bytes.data(), bytes.size()),
               std::runtime_error);
}

TEST(ModelParserTest, RejectsInvalidPlatformMetadata) {
  Model missing = ValidModel();
  missing.mutable_variants(0)->clear_platforms();
  EXPECT_THROW(SelectModelVariant(missing, false, false), std::runtime_error);

  Model duplicate = ValidModel();
  duplicate.mutable_variants(0)->add_platforms("cuda");
  EXPECT_THROW(SelectModelVariant(duplicate, false, false), std::runtime_error);

  Model unknown = ValidModel();
  unknown.mutable_variants(0)->set_platforms(0, "rocm");
  EXPECT_THROW(SelectModelVariant(unknown, false, false), std::runtime_error);
}

TEST(ModelParserTest, SelectsAndNormalizesCommOffNewtonOffVariant) {
  Model source = ValidModel();
  source.mutable_variants(0)->mutable_neighbor_list()->set_include_pair_type(
      true);
  const std::string bytes = source.SerializeAsString();
  Model parsed = ParseModelProtobuf(bytes.data(), bytes.size());
  SelectedModel selected = SelectModelVariant(parsed, false, false);
  EXPECT_EQ(selected.model.mlir_module_serialized(),
            "comm_off_newton_off");
  EXPECT_EQ(selected.model.calling_convention_version(), 10);
  EXPECT_TRUE(selected.model.neighbor_list().include_pair_type());
  EXPECT_FALSE(selected.model.uses_communication());
}

TEST(ModelParserTest, EveryLegalSelectionRetainsRequiredVirial) {
  Model source = ValidModel();
  auto* communication = source.add_variants();
  communication->CopyFrom(source.variants(1));
  communication->set_name("comm_on_newton_on");
  communication->set_uses_communication(true);
  communication->set_communication_buffer_width(1);

  const std::pair<bool, bool> selections[] = {
      {false, false}, {false, true}, {true, true}};
  for (const auto& selection : selections) {
    const SelectedModel selected =
        SelectModelVariant(source, selection.first, selection.second);
    const auto& virial = selected.model.output_fields(2);
    EXPECT_EQ(virial.name(), "V");
    EXPECT_EQ(virial.scope(), Model::LOCAL);
    ASSERT_EQ(virial.dimensions_size(), 1);
    EXPECT_EQ(virial.dimensions(0), 6);
    EXPECT_EQ(virial.components(), 6);
    EXPECT_TRUE(virial.extensive());
  }
}

TEST(ModelParserTest, RejectsParticleMetadataMismatch) {
  Model model = ValidModel();
  model.mutable_variants(0)->mutable_particle_fields(0)->set_name("residue_id");
  EXPECT_THROW(SelectModelVariant(model, false, false), std::runtime_error);
}

TEST(ModelParserTest, RejectsMissingAndDuplicateVariants) {
  Model missing = ValidModel();
  missing.clear_variants();
  const std::string bytes = missing.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(bytes.data(), bytes.size()),
               std::runtime_error);

  Model duplicate = ValidModel();
  duplicate.add_variants()->CopyFrom(duplicate.variants(0));
  EXPECT_THROW(SelectModelVariant(duplicate, false, false),
               std::runtime_error);
  EXPECT_THROW(SelectModelVariant(ValidModel(), true, true),
               std::runtime_error);
}

TEST(ModelParserTest, RejectsInvalidAndDuplicateParticleDescriptors) {
  Model invalid = ValidModel();
  invalid.mutable_particle_fields(0)->set_components(2);
  const std::string invalid_bytes = invalid.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(invalid_bytes.data(), invalid_bytes.size()),
               std::runtime_error);

  Model duplicate = ValidModel();
  duplicate.add_particle_fields()->CopyFrom(duplicate.particle_fields(0));
  const std::string duplicate_bytes = duplicate.SerializeAsString();
  EXPECT_THROW(
      ParseModelProtobuf(duplicate_bytes.data(), duplicate_bytes.size()),
      std::runtime_error);
}

TEST(ModelParserTest, ValidatesTypedOutputScopesAndShapes) {
  Model invalid_scope = ValidModel();
  invalid_scope.mutable_output_fields(2)->set_scope(
      Model::OUTPUT_SCOPE_UNSPECIFIED);
  const std::string invalid_scope_bytes = invalid_scope.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(invalid_scope_bytes.data(),
                                  invalid_scope_bytes.size()),
               std::runtime_error);

  Model invalid_shape = ValidModel();
  invalid_shape.mutable_output_fields(1)->set_dimensions(0, 2);
  invalid_shape.mutable_output_fields(1)->set_components(2);
  const std::string invalid_shape_bytes = invalid_shape.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(invalid_shape_bytes.data(),
                                  invalid_shape_bytes.size()),
               std::runtime_error);

  Model invalid_virial_scope = ValidModel();
  invalid_virial_scope.mutable_output_fields(2)->set_scope(Model::GLOBAL);
  const std::string invalid_virial_scope_bytes =
      invalid_virial_scope.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(invalid_virial_scope_bytes.data(),
                                  invalid_virial_scope_bytes.size()),
               std::runtime_error);

  Model missing_virial = ValidModel();
  missing_virial.mutable_output_fields()->RemoveLast();
  const std::string missing_virial_bytes = missing_virial.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(missing_virial_bytes.data(),
                                  missing_virial_bytes.size()),
               std::runtime_error);

  Model auxiliary = ValidModel();
  auto* output = auxiliary.add_output_fields();
  output->set_name("bias");
  output->set_scope(Model::GLOBAL);
  output->set_components(1);
  auxiliary.add_quantities("bias");
  auxiliary.add_quantity_components(1);
  const std::string auxiliary_bytes = auxiliary.SerializeAsString();
  EXPECT_NO_THROW(
      ParseModelProtobuf(auxiliary_bytes.data(), auxiliary_bytes.size()));
}

TEST(ModelParserTest, ValidatesOutputExtensivity) {
  Model particle = ValidModel();
  particle.mutable_output_fields(0)->set_extensive(true);
  const std::string particle_bytes = particle.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(particle_bytes.data(), particle_bytes.size()),
               std::runtime_error);

  Model virial = ValidModel();
  virial.mutable_output_fields(2)->set_extensive(false);
  const std::string virial_bytes = virial.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(virial_bytes.data(), virial_bytes.size()),
               std::runtime_error);
}

TEST(ModelParserTest, ValidatesNeighborMetadata) {
  Model unsupported = ValidModel();
  unsupported.mutable_neighbor_list()->set_type(Model::DEVICE_SPARSE);
  const std::string unsupported_bytes = unsupported.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(unsupported_bytes.data(),
                                  unsupported_bytes.size()),
               std::runtime_error);

  Model missing_order = ValidModel();
  missing_order.mutable_neighbor_list()->mutable_nbr_order()->RemoveLast();
  const std::string missing_order_bytes = missing_order.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(missing_order_bytes.data(),
                                  missing_order_bytes.size()),
               std::runtime_error);

  Model invalid_order = ValidModel();
  invalid_order.mutable_neighbor_list()->set_nbr_order(0, 0);
  const std::string invalid_order_bytes = invalid_order.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(invalid_order_bytes.data(),
                                  invalid_order_bytes.size()),
               std::runtime_error);

  Model invalid_cutoff = ValidModel();
  invalid_cutoff.mutable_neighbor_list()->set_cutoff(0.0);
  const std::string invalid_cutoff_bytes = invalid_cutoff.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(invalid_cutoff_bytes.data(),
                                  invalid_cutoff_bytes.size()),
               std::runtime_error);
}

TEST(ModelParserTest, RequiredCommunicationRejectsCommOffSelection) {
  Model model = ValidModel();
  auto* variant = model.add_variants();
  variant->set_name("comm_on_newton_on");
  variant->set_newton_pair(true);
  variant->set_mlir_module_serialized("comm module");
  variant->set_calling_convention_version(10);
  variant->set_uses_communication(true);
  variant->set_communication_buffer_width(1);
  variant->add_platforms("cuda");
  variant->mutable_neighbor_list()->CopyFrom(model.neighbor_list());
  variant->add_particle_fields()->CopyFrom(model.particle_fields(0));
  model.set_requires_communication(true);
  EXPECT_THROW(SelectModelVariant(model, false, true), std::runtime_error);
  EXPECT_NO_THROW(SelectModelVariant(model, true, true));
  EXPECT_THROW(SelectModelVariant(model, true, false), std::runtime_error);
}

TEST(ModelParserTest, RequiresSpeciesFirst) {
  Model missing = ValidModel();
  missing.mutable_particle_fields(0)->set_name("residue_id");
  const std::string missing_bytes = missing.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(missing_bytes.data(), missing_bytes.size()),
               std::runtime_error);

  Model misplaced = ValidModel();
  auto* extra = misplaced.add_particle_fields();
  extra->set_name("residue_id");
  extra->set_dtype(Model::INT32);
  extra->set_components(1);
  misplaced.mutable_particle_fields()->SwapElements(0, 1);
  const std::string misplaced_bytes = misplaced.SerializeAsString();
  EXPECT_THROW(
      ParseModelProtobuf(misplaced_bytes.data(), misplaced_bytes.size()),
      std::runtime_error);
}

TEST(ModelParserTest, AcceptsPortableNamedExtraFields) {
  Model model = ValidModel();
  auto* extra = model.add_particle_fields();
  extra->set_name("custom_residue_42");
  extra->set_dtype(Model::INT32);
  extra->set_components(1);
  for (auto& variant : *model.mutable_variants())
    variant.add_particle_fields()->CopyFrom(*extra);
  const std::string bytes = model.SerializeAsString();
  EXPECT_NO_THROW(ParseModelProtobuf(bytes.data(), bytes.size()));

  model.mutable_particle_fields(1)->set_name("not-portable");
  const std::string invalid_bytes = model.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(invalid_bytes.data(), invalid_bytes.size()),
               std::runtime_error);
}

TEST(ModelParserTest, ValidatesCommunicationBufferWidth) {
  Model model = ValidModel();
  auto* variant = model.add_variants();
  variant->set_name("comm_on_newton_on");
  variant->set_newton_pair(true);
  variant->set_mlir_module_serialized("comm module");
  variant->set_calling_convention_version(10);
  variant->set_uses_communication(true);
  variant->set_communication_buffer_width(4);
  variant->add_platforms("cpu");
  variant->add_platforms("cuda");
  variant->mutable_neighbor_list()->CopyFrom(model.neighbor_list());
  variant->add_particle_fields()->CopyFrom(model.particle_fields(0));
  SelectedModel selected = SelectModelVariant(model, true, true);
  EXPECT_TRUE(selected.model.uses_communication());
  EXPECT_EQ(selected.model.communication_buffer_width(), 4);
}

TEST(ModelParserTest, RejectsInvalidVariantTupleMetadata) {
  Model wrong_name = ValidModel();
  wrong_name.mutable_variants(0)->set_name("comm_off_newton_on");
  const std::string wrong_name_bytes = wrong_name.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(wrong_name_bytes.data(),
                                  wrong_name_bytes.size()),
               std::runtime_error);

  Model wrong_half_list = ValidModel();
  wrong_half_list.mutable_variants(0)->mutable_neighbor_list()->set_half_list(
      false);
  const std::string wrong_half_list_bytes =
      wrong_half_list.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(wrong_half_list_bytes.data(),
                                  wrong_half_list_bytes.size()),
               std::runtime_error);

  Model illegal_communication = ValidModel();
  auto* variant = illegal_communication.add_variants();
  variant->CopyFrom(illegal_communication.variants(0));
  variant->set_name("comm_on_newton_on");
  variant->set_uses_communication(true);
  variant->set_communication_buffer_width(1);
  const std::string illegal_bytes = illegal_communication.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(illegal_bytes.data(), illegal_bytes.size()),
               std::runtime_error);
}

TEST(ModelParserTest, RejectsBrokenTopLevelMirror) {
  Model executable = ValidModel();
  executable.set_mlir_module_serialized("different module");
  const std::string executable_bytes = executable.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(executable_bytes.data(),
                                  executable_bytes.size()),
               std::runtime_error);

  Model calling_convention = ValidModel();
  calling_convention.set_calling_convention_version(9);
  const std::string calling_convention_bytes =
      calling_convention.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(calling_convention_bytes.data(),
                                  calling_convention_bytes.size()),
               std::runtime_error);

  Model outputs = ValidModel();
  outputs.set_quantities(0, "energy");
  const std::string outputs_bytes = outputs.SerializeAsString();
  EXPECT_THROW(ParseModelProtobuf(outputs_bytes.data(), outputs_bytes.size()),
               std::runtime_error);
}

TEST(ModelParserTest, RejectsCommunicationMetadataOnCommOffVariant) {
  Model model = ValidModel();
  model.mutable_variants(0)->set_communication_buffer_width(4);
  EXPECT_THROW(SelectModelVariant(model, false, false), std::runtime_error);

  model.mutable_variants(0)->set_communication_buffer_width(0);
  EXPECT_NO_THROW(SelectModelVariant(model, false, false));
}

}  // namespace
}  // namespace jcn
