/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Modifications Copyright 2025 Multiscale Modeling of Fluid Materials,
   TU Munich.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
===============================================================================

Derived from TensorFlow's XlaCallModule loader:
https://github.com/tensorflow/tensorflow

chemtrain-deploy modifications add engine ABI wrapping, dtype
canonicalization, and communication custom-call adaptation.

*/

#include "xla_call_module_loader.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "shardy/dialect/sdy/ir/dialect.h"  // from @shardy
#include "shardy/dialect/sdy/transforms/import/passes.h"  // from @shardy
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "stablehlo/transforms/StablehloRefineShapes.h"  // from @stablehlo
// #include "tensorflow/compiler/jit/flags.h"
// #include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
// #include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/python/refine_polymorphic_shapes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace jcn {

namespace {

// When adding a new version, write when it was added. Also change the default
// version in the constructor in xla.py.
// See
// https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions
// for a description of the different versions.

constexpr int kVersionStartStableHloCompatibility = 4;
constexpr int kVersionStartSupportCallTFGraph = 5;
constexpr int kVersionStartSupportDisabledChecks = 6;
constexpr int kVersionStartSupportShapeAssertions = 7;
constexpr int kVersionStartSupportUsesShapePolymorphismAttr = 8;
constexpr int kVersionStartSupportEffects = 9;
constexpr int kVersionStartSupportShardyPartitioner = 10;
constexpr int kVersionMinimumSupported = kVersionStartStableHloCompatibility;

// This should match xla.py:call_module_maximum_supported_version
constexpr int kVersionMaximumSupported = kVersionStartSupportShardyPartitioner;

constexpr llvm::StringRef kDisabledCheckPlatform = "platform";

bool IsPlatformCheckDisabled(absl::Span<const std::string> disabled_checks) {
  return llvm::is_contained(disabled_checks, kDisabledCheckPlatform);
}

constexpr llvm::StringRef kDisabledCheckShapeAssertions = "shape_assertions";

bool IsShapeAssertionsCheckDisabled(
    absl::Span<const std::string> loading_disabled_checks) {
  return llvm::is_contained(loading_disabled_checks,
                            kDisabledCheckShapeAssertions);
}

constexpr llvm::StringRef kUsesShapePolymorphismAttr =
    "jax.uses_shape_polymorphism";

constexpr llvm::StringRef kWrappedModelMainName = "__jcn_model_main";
constexpr llvm::StringRef kGatherForwardTarget =
    "chemtrain_deploy.gather_forward";
constexpr llvm::StringRef kGatherReverseTarget =
    "chemtrain_deploy.gather_reverse";
constexpr llvm::StringRef kReduceTarget = "chemtrain_deploy.reduce";
constexpr llvm::StringRef kReduceTransposeTarget =
    "chemtrain_deploy.reduce_transpose";

mlir::Type ElementTypeForDtype(mlir::Builder& builder, TensorDtype dtype,
                               mlir::Type model_default) {
  switch (dtype) {
    case TensorDtype::ModelDefault:
      return model_default;
    case TensorDtype::F32:
      return builder.getF32Type();
    case TensorDtype::F64:
      return builder.getF64Type();
    case TensorDtype::S32:
      return builder.getI32Type();
  }
  return model_default;
}

mlir::RankedTensorType WithElementType(mlir::Type type,
                                       mlir::Type element_type) {
  auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  if (!ranked) return {};
  return mlir::RankedTensorType::get(ranked.getShape(), element_type,
                                     ranked.getEncoding());
}

bool IsFloatTensor(mlir::Type type) {
  auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return ranked && mlir::isa<mlir::FloatType>(ranked.getElementType());
}

bool IsCommunicationTarget(mlir::stablehlo::CustomCallOp op) {
  llvm::StringRef target = op.getCallTargetName();
  return target == kGatherForwardTarget || target == kGatherReverseTarget ||
         target == kReduceTarget || target == kReduceTransposeTarget;
}

absl::StatusOr<mlir::Value> ConvertTensor(mlir::OpBuilder& builder,
                                          mlir::Location loc,
                                          mlir::Value value,
                                          mlir::Type dst_type) {
  if (value.getType() == dst_type) return value;
  if (!mlir::isa<mlir::RankedTensorType>(value.getType()) ||
      !mlir::isa<mlir::RankedTensorType>(dst_type)) {
    return absl::InvalidArgumentError(
        "engine ABI wrapper can only convert ranked tensor values");
  }
  return builder.create<mlir::stablehlo::ConvertOp>(loc, dst_type, value)
      .getResult();
}

absl::Status RewriteCommunicationCustomCalls(mlir::ModuleOp module,
                                             const EngineAbiSpec& spec) {
  if (spec.communication_dtype == TensorDtype::ModelDefault) {
    return absl::OkStatus();
  }
  mlir::Builder type_builder(module.getContext());
  mlir::Type communication_element_type =
      ElementTypeForDtype(type_builder, spec.communication_dtype,
                          type_builder.getF32Type());
  if (!mlir::isa<mlir::FloatType>(communication_element_type)) {
    return absl::InvalidArgumentError(
        "communication dtype must be f32, f64, or model default");
  }

  llvm::SmallVector<mlir::stablehlo::CustomCallOp> calls;
  module.walk([&](mlir::stablehlo::CustomCallOp op) {
    if (IsCommunicationTarget(op)) calls.push_back(op);
  });

  for (mlir::stablehlo::CustomCallOp op : calls) {
    if (op->getNumOperands() != 2 || op->getNumResults() != 2) {
      return absl::InvalidArgumentError(
          "communication custom call must have two operands and two results");
    }
    auto feature_input =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto token_input =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto feature_result =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto token_result =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
    if (!feature_input || !feature_result ||
        !mlir::isa<mlir::FloatType>(feature_input.getElementType()) ||
        !mlir::isa<mlir::FloatType>(feature_result.getElementType())) {
      return absl::InvalidArgumentError(
          "communication custom call feature input/result must be ranked float tensors");
    }
    const bool is_reduce = op.getCallTargetName() == kReduceTarget ||
                           op.getCallTargetName() == kReduceTransposeTarget;
    const int expected_rank = is_reduce ? 1 : 2;
    if (feature_input.getRank() != expected_rank ||
        feature_result.getRank() != expected_rank) {
      return absl::InvalidArgumentError(absl::StrCat(
          is_reduce ? "reduce" : "gather",
          " communication custom call feature input/result must have rank ",
          expected_rank));
    }
    if (!token_input || !token_result ||
        token_input != token_result ||
        token_input.getRank() != 1 ||
        token_input.getDimSize(0) != 1 ||
        !token_input.getElementType().isF32()) {
      return absl::InvalidArgumentError(
          "communication custom call token must be tensor<1xf32>");
    }

    mlir::RankedTensorType communication_feature_type =
        WithElementType(feature_input, communication_element_type);
    mlir::RankedTensorType communication_result_type =
        WithElementType(feature_result, communication_element_type);
    mlir::OpBuilder builder(op);
    TF_ASSIGN_OR_RETURN(
        mlir::Value converted_feature_input,
        ConvertTensor(builder, op.getLoc(), op->getOperand(0),
                      communication_feature_type));

    llvm::SmallVector<mlir::Value> operands = {
        converted_feature_input, op->getOperand(1)};
    llvm::SmallVector<mlir::Type> result_types = {
        communication_result_type, token_result};
    mlir::OperationState state(op.getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(result_types);
    state.addAttributes(op->getAttrs());
    mlir::Operation* replacement = builder.create(state);

    builder.setInsertionPointAfter(replacement);
    TF_ASSIGN_OR_RETURN(
        mlir::Value restored_feature_result,
        ConvertTensor(builder, op.getLoc(), replacement->getResult(0),
                      feature_result));
    op->getResult(0).replaceAllUsesWith(restored_feature_result);
    op->getResult(1).replaceAllUsesWith(replacement->getResult(1));
    op->erase();
  }

  return absl::OkStatus();
}

}  // namespace

bool IsTokenType(mlir::Type type) {
  return mlir::isa<mlir::stablehlo::TokenType>(type);
}

absl::StatusOr<std::unique_ptr<XlaCallModuleLoader>>
XlaCallModuleLoader::Create(mlir::MLIRContext *context, int version,
                            mlir::StringRef module_str,
                            std::vector<std::string> disabled_checks,
                            std::vector<std::string> platforms,
                            int num_invocation_args,
                            bool main_has_token_input_output,
                            bool use_shardy_partitioner) {
  std::unique_ptr<XlaCallModuleLoader> loader(new XlaCallModuleLoader);
  TF_RETURN_IF_ERROR(loader->LoadModule(
      context, version, module_str, std::move(disabled_checks),
      std::move(platforms), num_invocation_args, main_has_token_input_output,
      use_shardy_partitioner));
  return loader;
}

absl::Status XlaCallModuleLoader::SetPlatformIndex(
    absl::string_view compilation_platform) {
  int platform_index = -1;
  if (!platforms_.empty()) {
    auto found_platform =
        std::find(platforms_.begin(), platforms_.end(), compilation_platform);
    if (found_platform == platforms_.end()) {
      if (!IsPlatformCheckDisabled(loading_disabled_checks_)) {
        return absl::NotFoundError(absl::StrCat(
            "The current platform ", compilation_platform,
            " is not among the platforms required by the module: [",
            absl::StrJoin(platforms_, ", "), "]"));
      } else {
        if (platforms_.size() > 1) {
          platform_index = 0;
        }
      }
    } else {
      // We only use a platform index argument if we support at least 2
      // platforms.
      if (platforms_.size() > 1) {
        platform_index = found_platform - platforms_.begin();
      }
    }
  }

  if (platform_index < 0) return absl::OkStatus();
  VLOG(3) << "XlaCallModule setting the platform_index to " << platform_index
          << " for platform " << compilation_platform << ".";
  mlir::Block &main_body = main_.front();

  if (main_.getNumArguments() < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The module should have a platform index argument but it has no ",
        "arguments"));
  }
  mlir::OpBuilder op_builder(main_);
  op_builder.setInsertionPointToStart(&main_body);
  mlir::BlockArgument platform_index_arg = main_body.getArgument(0);
  mlir::RankedTensorType arg_ranked_type =
      mlir::dyn_cast<mlir::RankedTensorType>(platform_index_arg.getType());
  if (!arg_ranked_type || arg_ranked_type.getRank() != 0 ||
      !(arg_ranked_type.getElementType().isSignlessInteger(32) ||
        arg_ranked_type.getElementType().isSignlessInteger(64))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module argument at index 0 should be a 0-dimensional "
                     "32-bit or 64-bit integer-tensor platform index argument "
                     "but has type ",
                     mlir::debugString(platform_index_arg.getType())));
  }
  bool is_32_bit = arg_ranked_type.getElementType().isSignlessInteger(32);
  auto const_attr = is_32_bit ? op_builder.getI32IntegerAttr(platform_index)
                              : op_builder.getI64IntegerAttr(platform_index);
  auto platform_index_op = op_builder.create<mlir::stablehlo::ConstantOp>(
      platform_index_arg.getLoc(), const_attr);
  platform_index_arg.replaceAllUsesWith(platform_index_op);

  CHECK(llvm::succeeded(main_.eraseArgument(0)));
  platform_index_arg_set_ = true;
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::RefineDynamicShapes(
    llvm::ArrayRef<xla::Shape> input_shapes) {
  // Skip shape refinement for new versions if USES_SHAPE_POLYMORPHISM_ATTR=1
  if (version_ >= kVersionStartSupportUsesShapePolymorphismAttr) {
    if (mlir::Attribute uses_shape_poly_attr =
            (*module_)->getAttr(kUsesShapePolymorphismAttr)) {
      mlir::BoolAttr uses_shape_poly_bool_attr =
          llvm::dyn_cast<mlir::BoolAttr>(uses_shape_poly_attr);

      if (!uses_shape_poly_bool_attr) {
        return absl::InvalidArgumentError(absl::StrCat(
            "jax.uses_shape_polymorphism is not a boolean attribute: ",
            mlir::debugString(uses_shape_poly_attr)));
      }
      if (!uses_shape_poly_bool_attr.getValue()) {
        VLOG(3) << "XlaCallModule skipping shape refinement due to module "
                << " attribute " << kUsesShapePolymorphismAttr.str() << "="
                << mlir::debugString(uses_shape_poly_attr);
        return absl::OkStatus();
      }
    } else {
      VLOG(3) << "XlaCallModule skipping shape refinement due to module "
              << " attribute " << kUsesShapePolymorphismAttr.str()
              << " missing";
      return absl::OkStatus();
    }
  }
  // Add the tokens to the input_shapes. Starting with version 9, the main
  // function may take token arguments that do not correspond with op inputs.
  int nr_inputs = NrInputs();
  int nr_expected_tokens = llvm::count_if(InputTypes(), IsTokenType);
  bool has_platform_index_arg =
      platforms_.size() > 1 && !platform_index_arg_set_;
  int nr_expected_platform_index_args = has_platform_index_arg ? 1 : 0;
  if (input_shapes.size() !=
      nr_inputs - nr_expected_tokens - nr_expected_platform_index_args) {
    return absl::InvalidArgumentError(absl::StrCat(
        "XlaCallModule RefineDynamicShapes called with ", input_shapes.size(),
        " input shapes, but the main function takes ",
        nr_inputs - nr_expected_tokens - nr_expected_platform_index_args,
        " non-token and non-platform-index arguments. The input ",
        "shapes are (",
        absl::StrJoin(input_shapes, ", ",
                      [](std::string *out, const xla::Shape &s) {
                        absl::StrAppend(out, s.ToString());
                      }),
        ") and the main function argument types are ",
        absl::StrJoin(InputTypes(), ", ",
                      [](std::string *out, const mlir::Type &t) {
                        absl::StrAppend(out, mlir::debugString(t));
                      }),
        ")"));
  }

  // Derive static input types to use for main.
  mlir::Block &main_body = main_.front();
  mlir::Builder builder(module_->getContext());
  std::vector<mlir::Type> static_array_input_types(nr_inputs);
  int next_actual_input = 0;
  for (int i = 0, end = nr_inputs; i < end; ++i) {
    mlir::Type arg_type = main_body.getArgument(i).getType();
    if (i == 0 && has_platform_index_arg) {
      static_array_input_types[i] = arg_type;
      continue;
    }
    if (IsTokenType(arg_type)) {
      static_array_input_types[i] = arg_type;
      VLOG(3) << "XlaCallModule static array input type #" << i << ": "
              << mlir::debugString(static_array_input_types[i])
              << " for argument type " << mlir::debugString(arg_type);
      continue;
    }

    // Get static MLIR Type from xla Shape.
    const xla::Shape &xla_shape = input_shapes[next_actual_input++];
    std::vector<int64_t> xla_dimensions;
    if (xla_shape.IsArray()) {
      xla_dimensions = std::vector<int64_t>(xla_shape.dimensions().begin(),
                                            xla_shape.dimensions().end());
    }
    TF_ASSIGN_OR_RETURN(
        mlir::Type element_type,
        ConvertPrimitiveTypeToMlirType(xla_shape.element_type(), builder));
    mlir::RankedTensorType type =
        mlir::RankedTensorType::get(xla_dimensions, element_type);

    VLOG(3) << "XlaCallModule static array input type #" << i << ": "
            << mlir::debugString(type) << " for argument type "
            << mlir::debugString(arg_type);
    static_array_input_types[i] = type;
  }

  // Insert custom_call ops as shims to maintain the validity of the module when
  // main's input types are changed later. This is a workaround to allow shape
  // refinement to be applied; the custom_calls are removed before returning.
  // Arguments to main may occur as return values, or as inputs to called
  // functions, and changing their types may invalidate the module due to type
  // mismatches. To prevent this, for each argument that is a dynamically-shaped
  // tensor, we insert a custom_call op that takes the argument as an input and
  // replace uses of the argument with the custom_call's result. custom_call
  // is used as it allows its inputs and outputs to be unranked.
  //
  // Example:
  //
  // The below main function returns its argument directly:
  //
  // func.func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  //   return %arg0 : tensor<*xf32>
  // }
  //
  // Changing the argument's type invalidates the IR (type mismatch):
  //
  // func.func @main(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  //   return %arg0 : tensor<*xf32>
  // }
  //
  // Inserting a custom_call allows the IR to remain valid:
  //
  // func.func @main(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  //   %0 = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
  //   %1 = stablehlo.custom_call
  //   @stablehlo.shape_refinement_operand_wrapper(%arg0, %0)
  //   {indices_of_shape_operands = dense<1> : tensor<1xi64>} :
  //   (tensor<2x3xf32>, tensor<2xi64>) -> tensor<*xf32>
  //   return %1 : tensor<*xf32>
  // }
  //
  // After shapes are refined and the custom_calls are removed, we get:
  //
  // func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  //   return %arg0 : tensor<2x3xf32>
  // }
  //
  {
    if (failed(mlir::stablehlo::refineArguments(main_,
                                                static_array_input_types))) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error refining argument shapes."));
    }
  }

  bool enable_shape_assertions =
      (version_ >= kVersionStartSupportShapeAssertions &&
       !IsShapeAssertionsCheckDisabled(loading_disabled_checks_));

  // Store the original output types before shape refinement.
  mlir::TypeRange original_output_types = OutputTypes();

  // RefinePolymorphicShapes will refine using the new static types and clean up
  // the shape_refinement_operand_wrapper custom calls.
  TF_RETURN_IF_ERROR(
      xla::RefinePolymorphicShapes(*module_, enable_shape_assertions));

  // Mark the output types as refined if they are different from the original
  // output types.
  if (OutputTypes() != original_output_types) {
    output_types_refined_ = true;
  }

  return absl::OkStatus();
}

absl::Status RemoveAbstractArgumentsFromMain(
    mlir::ModuleOp module, mlir::func::FuncOp main,
    llvm::ArrayRef<int> argument_indices) {
  if (!main) {
    return absl::InvalidArgumentError(
        "Cannot remove abstract arguments from missing main function");
  }

  // Shape refinement can leave pure operations whose results are dead. Remove
  // those before deciding whether a carrier still has semantic value uses.
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  if (mlir::failed(pm.run(module))) {
    return absl::InternalError(
        "Failed to canonicalize abstract argument uses");
  }

  llvm::SmallVector<int> sorted(argument_indices.begin(),
                                argument_indices.end());
  llvm::sort(sorted, std::greater<int>());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    return absl::InvalidArgumentError(
        "Abstract argument indices must be unique");
  }
  for (int index : sorted) {
    if (index < 0 || index >= static_cast<int>(main.getNumArguments())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Abstract argument index ", index, " is out of range"));
    }
    mlir::BlockArgument argument = main.getArgument(index);
    if (!argument.use_empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Abstract argument ", index,
          " still has value-semantic uses after shape refinement"));
    }
  }
  for (int index : sorted) {
    if (mlir::failed(main.eraseArgument(index))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to erase abstract argument ", index));
    }
  }
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::RemoveAbstractArguments(
    llvm::ArrayRef<int> argument_indices) {
  if (!main_) {
    return absl::InvalidArgumentError(
        "Cannot remove abstract arguments from missing main function");
  }
  return RemoveAbstractArgumentsFromMain(*module_, main_, argument_indices);
}

absl::Status XlaCallModuleLoader::WrapMainForEngineAbi(
    const EngineAbiSpec& spec,
    const std::vector<std::string>& particle_names,
    const std::vector<ModelProperties::OutputField>& output_fields) {
  // Keep the exported function as the canonical model and build a public
  // wrapper for the engine ABI. The wrapper converts engine input dtypes and
  // species numbering before the call, then converts results to requested
  // engine dtypes. Output descriptors carry scope and shape but no dtype.
  if (!main_) {
    return absl::InvalidArgumentError("Cannot wrap missing main function");
  }
  if (main_.getName() == kWrappedModelMainName) {
    return absl::InvalidArgumentError("Model main is already wrapped");
  }
  const size_t fixed_model_arguments =
      particle_names.size() + 3;
  if (main_.getNumArguments() < fixed_model_arguments) {
    return absl::InvalidArgumentError(
        "Engine ABI wrapper has fewer canonical inputs than particle metadata requires");
  }

  mlir::OpBuilder builder(main_);
  mlir::Location loc = main_.getLoc();
  mlir::Builder type_builder(module_->getContext());

  llvm::SmallVector<mlir::Type> canonical_arg_types(main_.getArgumentTypes());
  llvm::SmallVector<mlir::Type> canonical_result_types(main_.getResultTypes());
  mlir::ArrayAttr canonical_arg_attrs = main_.getAllArgAttrs();
  mlir::ArrayAttr canonical_result_attrs = main_.getAllResultAttrs();
  llvm::SmallVector<mlir::Type> engine_arg_types(
      canonical_arg_types.begin(), canonical_arg_types.end());

  // Derive the engine input types.

  auto position_type =
      mlir::dyn_cast<mlir::RankedTensorType>(canonical_arg_types[0]);
  auto species_it =
      std::find(particle_names.begin(), particle_names.end(), "species");
  const int species_index = species_it == particle_names.end()
                                ? -1
                                : 1 + std::distance(particle_names.begin(), species_it);
  mlir::RankedTensorType species_type;
  if (species_index >= 0) {
    species_type =
        mlir::dyn_cast<mlir::RankedTensorType>(canonical_arg_types[species_index]);
  }
  if (!position_type || (species_index >= 0 && !species_type)) {
    return absl::InvalidArgumentError(
        "Engine ABI wrapper expects ranked position and species tensors");
  }

  engine_arg_types[0] =
      WithElementType(canonical_arg_types[0],
                      ElementTypeForDtype(type_builder, spec.position_dtype,
                                          position_type.getElementType()));
  if (species_index >= 0) {
    engine_arg_types[species_index] = WithElementType(
        canonical_arg_types[species_index],
        ElementTypeForDtype(type_builder, spec.species_dtype,
                            species_type.getElementType()));
  }
  if (!engine_arg_types[0] ||
      (species_index >= 0 && !engine_arg_types[species_index])) {
    return absl::InvalidArgumentError(
        "Engine ABI wrapper failed to derive engine input tensor types");
  }

  // Model results may end with internal neighbor statistics that do not
  // have public output descriptors.
  if (output_fields.size() > canonical_result_types.size()) {
    return absl::InvalidArgumentError(
        "Engine ABI wrapper received too many output descriptors");
  }

  // Output descriptors follow canonical result order. PARTICLE results have
  // the refined particle axis followed by declared value dimensions. LOCAL and
  // GLOBAL results contain only their configuration dimensions.
  for (int i = 0; i < static_cast<int>(output_fields.size()); ++i) {
    auto ranked =
        mlir::dyn_cast<mlir::RankedTensorType>(canonical_result_types[i]);
    if (!ranked || !IsFloatTensor(canonical_result_types[i])) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Model output '", output_fields[i].name,
          "' must be a ranked floating-point tensor"));
    }

    const auto& field = output_fields[i];
    const int particle_axis =
        field.scope == ModelProperties::OutputScope::PARTICLE ? 1 : 0;
    const int expected_rank =
        static_cast<int>(field.dimensions.size()) + particle_axis;
    if (ranked.getRank() != expected_rank) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Model output '", field.name, "' has rank ", ranked.getRank(),
          ", expected ", expected_rank));
    }
    if (particle_axis == 1 &&
        ranked.getDimSize(0) != position_type.getDimSize(0)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Particle output '", field.name,
          "' does not use the refined particle dimension"));
    }
    for (int dim = 0; dim < static_cast<int>(field.dimensions.size()); ++dim) {
      if (ranked.getDimSize(dim + particle_axis) != field.dimensions[dim]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Model output '", field.name,
            "' does not match its declared dimensions"));
      }
    }
  }

  // Derive the engine result dtypes.
  llvm::SmallVector<mlir::Type> engine_result_types(
      canonical_result_types.begin(), canonical_result_types.end());
  auto output_dtype = [&](const std::string& name) {
    for (const auto& entry : spec.output_dtypes) {
      if (entry.first == name) return entry.second;
    }
    return spec.default_output_dtype;
  };
  auto convert_result_type = [&](int index) {
    if (index >= static_cast<int>(engine_result_types.size()) ||
        !IsFloatTensor(engine_result_types[index])) {
      return;
    }
    auto ranked =
        mlir::cast<mlir::RankedTensorType>(engine_result_types[index]);
    engine_result_types[index] =
        WithElementType(engine_result_types[index],
                        ElementTypeForDtype(type_builder,
                                            output_dtype(output_fields[index].name),
                                            ranked.getElementType()));
  };
  for (int i = 0; i < static_cast<int>(output_fields.size()); ++i) {
    convert_result_type(i);
  }

  // Create the engine wrapper and convert its inputs.
  main_.setName(kWrappedModelMainName);
  auto wrapper_type =
      builder.getFunctionType(engine_arg_types, engine_result_types);
  auto wrapper = mlir::func::FuncOp::create(loc, "main", wrapper_type);
  if (canonical_arg_attrs) wrapper.setAllArgAttrs(canonical_arg_attrs);
  if (canonical_result_attrs) wrapper.setAllResultAttrs(canonical_result_attrs);
  wrapper.setPublic();
  mlir::Block* body = wrapper.addEntryBlock();
  builder.setInsertionPointToStart(body);

  llvm::SmallVector<mlir::Value> canonical_args;
  canonical_args.reserve(canonical_arg_types.size());
  for (int i = 0; i < static_cast<int>(canonical_arg_types.size()); ++i) {
    mlir::Value value = body->getArgument(i);
    TF_ASSIGN_OR_RETURN(mlir::Value converted,
                        ConvertTensor(builder, loc, value,
                                      canonical_arg_types[i]));
    if (i == species_index &&
        spec.species_encoding == SpeciesEncoding::OneBased) {
      auto canonical_species =
          mlir::dyn_cast<mlir::RankedTensorType>(canonical_arg_types[i]);
      if (!canonical_species ||
          !canonical_species.getElementType().isSignlessInteger(32)) {
        return absl::InvalidArgumentError(
            "One-based species conversion requires canonical s32 species");
      }
      auto one_attr = mlir::DenseElementsAttr::get(
          canonical_species,
          type_builder.getIntegerAttr(canonical_species.getElementType(), 1));
      auto one =
          builder.create<mlir::stablehlo::ConstantOp>(loc, one_attr);
      converted =
          builder.create<mlir::stablehlo::SubtractOp>(
              loc, canonical_species, converted, one.getResult())
              .getResult();
    }
    canonical_args.push_back(converted);
  }

  // Call the canonical model and convert its results.
  auto call = builder.create<mlir::func::CallOp>(
      loc, kWrappedModelMainName, canonical_result_types, canonical_args);
  llvm::SmallVector<mlir::Value> engine_results;
  engine_results.reserve(engine_result_types.size());
  for (int i = 0; i < static_cast<int>(engine_result_types.size()); ++i) {
    TF_ASSIGN_OR_RETURN(mlir::Value converted,
                        ConvertTensor(builder, loc, call.getResult(i),
                                      engine_result_types[i]));
    engine_results.push_back(converted);
  }
  builder.create<mlir::func::ReturnOp>(loc, engine_results);

  // Apply communication rewriting after the wrapper is complete because the
  // rewrite uses the selected engine dtypes. Then verify the full transformed
  // module.
  module_->push_back(wrapper);
  main_ = wrapper;
  TF_RETURN_IF_ERROR(RewriteCommunicationCustomCalls(*module_, spec));
  if (mlir::failed(mlir::verify(*module_))) {
    return absl::InvalidArgumentError(
        "Engine ABI wrapper generated an invalid StableHLO module");
  }
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::LoadModule(
    mlir::MLIRContext *context, int version, mlir::StringRef module_str,
    std::vector<std::string> disabled_checks,
    std::vector<std::string> platforms, int num_invocation_args,
    bool main_has_token_input_output, bool use_shardy_partitioner) {
  context_ = context;
  version_ = version;
  platforms_ = platforms;
  loading_disabled_checks_ = disabled_checks;
  use_shardy_partitioner_ = use_shardy_partitioner;

  // Load a superset of dialects; we should check at serialization time that
  // we only include allowable dialects.
  context_->loadDialect<mlir::func::FuncDialect>();
  context_->loadDialect<mlir::stablehlo::StablehloDialect>();
  context_->loadDialect<mlir::chlo::ChloDialect>();
  context_->loadDialect<mlir::vhlo::VhloDialect>();
  context_->loadDialect<mlir::sdy::SdyDialect>();

  if (version < kVersionMinimumSupported) {
    return absl::InvalidArgumentError(absl::StrCat(
        "XlaCallModuleOp with version ", version,
        " is not supported anymore. Must be >= ", kVersionMinimumSupported));
  }
  if (version > kVersionMaximumSupported) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModuleOp with version ", version,
                     " is not supported by this build. Must be <= ",
                     kVersionMaximumSupported));
  }
  if (version >= kVersionStartSupportDisabledChecks && platforms.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModuleOp with version ", version,
                     " must have non-empty platforms."));
  }

  // Parse the StableHLO/VHLO bytecode
  {
    module_ =
        mlir::stablehlo::deserializePortableArtifact(module_str, context_);
    if (!module_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot deserialize computation."));
    }
  }

  if (use_shardy_partitioner) {
    // We need to inline `sdy.mesh` symbols because otherwise they are going
    // to be discarded or their names might collide with `sdy.mesh` symbols in
    // another XlaCallModuleOp.
    mlir::PassManager pm(module_->getContext());
    // TODO(b/422690222): Remove `addSdyRoundTripImportPipeline` 6 months
    // after mixed serialization will be supported by Shardy+StableHLO in JAX
    xla::sdy::addSdyRoundTripImportPipeline(pm, /*enableConstantImport=*/false);
    pm.addPass(mlir::sdy::createInlineMeshesPass());
    if (failed(pm.run(*module_))) {
      return absl::InternalError(
          absl::StrCat("Shardy inline meshes pass failed. "));
    }
  }

  {
    if (mlir::failed(mlir::verify(*module_))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Error verifying module."));
    }
  }
  main_ = module_->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_) {
    return absl::InvalidArgumentError("Cannot find 'main' in module");
  }

  mlir::Block &main_body = main_.front();

  int nr_token_arguments = llvm::count_if(InputTypes(), IsTokenType);
  if (version < kVersionStartSupportEffects) {
    bool has_token_at_start = (nr_token_arguments == 1 &&
                               IsTokenType(main_.getArgument(0).getType()));
    if (main_has_token_input_output != has_token_at_start) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected a token at start iff main_has_token_input_output. ",
          "Found main function type ",
          mlir::debugString(main_.getFunctionType()),
          " and main_has_token_input_output = ", main_has_token_input_output));
    }
  }
  int nr_platform_args = (platforms.size() > 1 ? 1 : 0);
  if (num_invocation_args !=
      main_body.getNumArguments() - nr_platform_args - nr_token_arguments) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Incorrect number of arguments passed to XlaCallModule = ",
        num_invocation_args, ". It must be called with ",
        main_body.getNumArguments() - nr_platform_args - nr_token_arguments,
        " because the module main function takes ", main_body.getNumArguments(),
        " arguments of which ", nr_platform_args, " platform index arguments, ",
        "and ", nr_token_arguments, " token arguments."));
  }
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::ValidateXlaCallModuleInvariants() {
  bool moduleValidationFailed = false;

  module_->walk([&](mlir::Operation *op) {
    // StableHLO programs created by jax2tf only contain operations
    // from Builtin, Func, StableHLO, Shardy dialects.
    if (!llvm::isa<mlir::BuiltinDialect, mlir::chlo::ChloDialect,
                   mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect,
                   mlir::sdy::SdyDialect>(op->getDialect())) {
      op->emitOpError() << "is an op from an unsupported dialect";
      moduleValidationFailed = true;
    }
    // `shape_assertion` custom calls must have side effects. We check this here
    // because a pure `shape_assertion` is likely to be removed by MLIR's
    // dead-code elimination, preventing us from detecting the issue later.
    if (auto customCallOp = llvm::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
      if (!customCallOp.getHasSideEffect() &&
          customCallOp.getCallTargetName() == "shape_assertion") {
        op->emitOpError() << "`shape_assertion` custom calls must set "
                             "`has_side_effect = true`.";
        moduleValidationFailed = true;
      }
    }
  });

  if (moduleValidationFailed) {
    return absl::InvalidArgumentError(
        absl::StrCat("XlaCallModule failed validation."));
  }
  return absl::OkStatus();
}

absl::Status XlaCallModuleLoader::ValidateStaticShapes() {
  return xla::ValidateStaticShapes(*module_);
}

absl::Status XlaCallModuleLoader::PrepareStablehloForLowering() {

  // TODO (b/410057228): Replace MHLO canonicalization with StableHLO.
  // This code requires MHLO CaseOp canonicalization to remove unreachable
  // branches, else `tf.call_tf_function` inlining can fail.
  mlir::PassManager pm(module_->getContext());
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (use_shardy_partitioner_) {
    // We need to export shardings because the lowering path go directly to
    // HLO but not the MLIR to HLO path that invokes SdyRoundTripExport.
    // We keep meshes inlined to avoid naming collisions when multiple
    // XlaCallModules are combined.
    xla::sdy::addSdyRoundTripExportPipeline(pm);
  }

  if (failed(pm.run(*module_))) {
    return absl::InternalError(
        absl::StrCat("MHLO->HLO lowering passes failed."));
  }

  return absl::OkStatus();
}

absl::StatusOr<xla::XlaComputation> XlaCallModuleLoader::ToXlaComputation() {
  xla::HloProto proto;
  TF_RETURN_IF_ERROR(xla::ConvertStablehloToHloProto(*module_, &proto));
  return xla::XlaComputation(std::move(*proto.mutable_hlo_module()));
}

}  // namespace jcn
