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

#include "connector/compiler.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "connector/utils.h"
#include "connector/xla_call_module_loader.h"

#include "xla/xla_data.pb.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"

#include "absl/types/span.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"


namespace jcn {

namespace {

constexpr llvm::StringRef kGatherForwardTarget =
    "chemtrain_deploy.gather_forward";
constexpr llvm::StringRef kGatherReverseTarget =
    "chemtrain_deploy.gather_reverse";
constexpr llvm::StringRef kReduceTarget = "chemtrain_deploy.reduce";
constexpr llvm::StringRef kReduceTransposeTarget =
    "chemtrain_deploy.reduce_transpose";

bool IsCommunicationTarget(mlir::stablehlo::CustomCallOp op) {
  llvm::StringRef target = op.getCallTargetName();
  return target == kGatherForwardTarget || target == kGatherReverseTarget ||
         target == kReduceTarget || target == kReduceTransposeTarget;
}

bool IsReduceTarget(mlir::stablehlo::CustomCallOp op) {
  llvm::StringRef target = op.getCallTargetName();
  return target == kReduceTarget || target == kReduceTransposeTarget;
}

int CommunicationBufferWidth(mlir::stablehlo::CustomCallOp op) {
  auto feature_input =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  if (!feature_input) {
    throw std::runtime_error(
        "communication call operand must be a ranked tensor");
  }
  int64_t width = 1;
  const int start_dim = IsReduceTarget(op) ? 0 : 1;
  if (!IsReduceTarget(op) && feature_input.getRank() < 2) {
    throw std::runtime_error(
        "communication exchange operand must be atom-leading");
  }
  for (int dim = start_dim; dim < feature_input.getRank(); ++dim) {
    if (feature_input.isDynamicDim(dim)) {
      throw std::runtime_error(
          "communication call width must be static");
    }
    width *= feature_input.getDimSize(dim);
    if (width > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          "communication call width exceeds integer range");
    }
  }
  return static_cast<int>(width);
}

void ValidateCommunicationBufferWidth(mlir::ModuleOp module,
                                      int communication_buffer_width) {
  if (communication_buffer_width <= 0) return;
  module.walk([&](mlir::stablehlo::CustomCallOp op) {
    if (!IsCommunicationTarget(op)) return;
    const int width = CommunicationBufferWidth(op);
    if (width <= 0 || width > communication_buffer_width) {
      throw std::runtime_error(
          "communication call width " +
          std::to_string(width) +
          " exceeds exported communication buffer width " +
          std::to_string(communication_buffer_width));
    }
  });
}

}  // namespace

Compiler::Compiler(const std::string& mlir_module_serialized,
                   int calling_convention_version,
                   int communication_buffer_width,
                   std::vector<std::string> platforms, std::string backend)
    : mlir_module_serialized(mlir_module_serialized),
      calling_convention_version(calling_convention_version),
      communication_buffer_width(communication_buffer_width),
      platforms(std::move(platforms)),
      backend(std::move(backend)) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::ml_program::MLProgramDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::sdy::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

void Compiler::compile(
    int n_atoms, std::vector<GraphInputDescriptor> graph_inputs,
    std::vector<xla::PrimitiveType> particle_types,
    std::vector<std::string> particle_names,
    std::vector<xla::PrimitiveType> global_types,
    std::vector<std::string> /*global_names*/, const EngineAbiSpec& engine_abi,
    std::vector<ModelProperties::OutputField> output_fields) {
  Logger logger = Logger::getlogger();

  // Count the model inputs before loading the serialized module. Shape
  // refinement receives the concrete shapes later, after platform selection
  // exposes the canonical model signature.
  const xla::Shape count_shape =
      xla::ShapeUtil::MakeShape(xla::S32, absl::Span<const int64_t>{});
  int next_input_index = 1 + static_cast<int>(particle_types.size()) +
                         static_cast<int>(global_types.size()) + 2;
  std::vector<int> abstract_argument_indices;
  for (const GraphInputDescriptor& input : graph_inputs) {
    if (input.kind == GraphInputKind::ABSTRACT) {
      abstract_argument_indices.push_back(next_input_index);
    }
    ++next_input_index;
  }

  // Load the exported StableHLO module.
  std::vector<std::string> disabled_checks;
  auto loader_status = XlaCallModuleLoader::Create(
      &context, calling_convention_version, mlir_module_serialized,
      disabled_checks, platforms, next_input_index, false, false);
  if (!loader_status.ok()) {
    throw std::runtime_error("Failed to load StableHLO module: " +
                             std::string(loader_status.status().message()));
  }
  std::unique_ptr<XlaCallModuleLoader> module_loader =
      std::move(loader_status).value();

  // Run the preparation steps in their required order. Platform selection and
  // shape refinement operate on the exported signature. Abstract capacity
  // arguments disappear only after refinement. The engine wrapper then
  // converts dtypes and rewrites communication calls. Static-shape validation
  // and lowering must operate on the final wrapper.
  absl::Status status = module_loader->ValidateXlaCallModuleInvariants();
  if (!status.ok()) {
    throw std::runtime_error("Failed to validate dialect: " +
                             std::string(status.message()));
  }

  ValidateCommunicationBufferWidth(module_loader->module(),
                                   communication_buffer_width);

  status = module_loader->SetPlatformIndex(backend);
  if (!status.ok()) {
    throw std::runtime_error("Failed to set platform index: " +
                             std::string(status.message()));
  }

  // Refine against the dtype exported by the model. The engine dtype is
  // deliberately applied later by WrapMainForEngineAbi, which inserts an
  // explicit conversion when the adapter and model precisions differ.
  mlir::RankedTensorType canonical_position;
  for (mlir::Type input_type : module_loader->InputTypes()) {
    if (IsTokenType(input_type)) continue;
    canonical_position = mlir::dyn_cast<mlir::RankedTensorType>(input_type);
    break;
  }
  if (!canonical_position || canonical_position.getRank() != 2) {
    throw std::runtime_error(
        "Exported model must begin with a ranked position tensor");
  }
  xla::PrimitiveType canonical_position_type;
  if (canonical_position.getElementType().isF32()) {
    canonical_position_type = xla::F32;
  } else if (canonical_position.getElementType().isF64()) {
    canonical_position_type = xla::F64;
  } else {
    throw std::runtime_error(
        "Exported model position tensor must use float32 or float64");
  }

  std::vector<xla::Shape> input_shapes = {xla::ShapeUtil::MakeShape(
      canonical_position_type, absl::Span<const int64_t>{n_atoms, 3})};
  for (xla::PrimitiveType particle_type : particle_types) {
    input_shapes.push_back(xla::ShapeUtil::MakeShape(
        particle_type, absl::Span<const int64_t>{n_atoms}));
  }
  for (xla::PrimitiveType global_type : global_types) {
    input_shapes.push_back(xla::ShapeUtil::MakeShape(
        global_type, absl::Span<const int64_t>{}));
  }
  input_shapes.push_back(count_shape);
  input_shapes.push_back(count_shape);
  for (const GraphInputDescriptor& input : graph_inputs) {
    input_shapes.push_back(xla::ShapeUtil::MakeShape(input.type, input.shape));
  }

  if (logger.log(LogLevel::INFO)) {
    logger.log(LogLevel::INFO, "Input shapes for the XLA computation:");
    for (const auto& shape : input_shapes) {
      logger.log(LogLevel::INFO,
                 "Input: " + xla::PrimitiveType_Name(shape.element_type()) +
                     ":" + shape.ToString());
    }
  }

  status = module_loader->RefineDynamicShapes(input_shapes);
  if (!status.ok()) {
    throw std::runtime_error("Failed to refine dynamic shapes: " +
                             std::string(status.message()));
  }

  status = module_loader->RemoveAbstractArguments(abstract_argument_indices);
  if (!status.ok()) {
    throw std::runtime_error("Failed to remove graph shape carriers: " +
                             std::string(status.message()));
  }

  status = module_loader->WrapMainForEngineAbi(
      engine_abi, particle_names, output_fields);
  if (!status.ok()) {
    throw std::runtime_error("Failed to wrap model for engine ABI: " +
                             std::string(status.message()));
  }

  status = module_loader->ValidateStaticShapes();
  if (!status.ok()) {
    throw std::runtime_error("Failed to validate static shapes: " +
                             std::string(status.message()));
  }

  status = module_loader->PrepareStablehloForLowering();
  if (!status.ok()) {
    throw std::runtime_error("Failed to lower to XLA: " +
                             std::string(status.message()));
  }

  // PJRT compiles the transformed module after an HLO-to-StableHLO round trip.
  // The conversion must follow all refinement, ABI, communication, and
  // lowering passes.
  auto computation_status = module_loader->ToXlaComputation();
  if (!computation_status.ok()) {
    throw std::runtime_error(
        "Failed to convert the module to XLA computation: " +
        std::string(computation_status.status().message()));
  }
  xla::XlaComputation computation = std::move(computation_status).value();

  auto module_status =
      xla::ConvertHloToStablehlo(export_context, &computation.proto());
  if (!module_status.ok()) {
    throw std::runtime_error("Failed to convert the computation to MLIR: " +
                             std::string(module_status.status().message()));
  }
  module_ref = std::move(module_status).value();
}

}  // namespace jcn
