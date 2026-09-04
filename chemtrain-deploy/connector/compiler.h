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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_COMPILER_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_COMPILER_H_

#include <string>
#include <vector>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"

#include "xla/xla_data.pb.h"

#include "absl/types/span.h"
#include "connector/model_shape.h"
#include "connector/runtime_types.h"


namespace jcn {

class Compiler {
 public:
  Compiler(const std::string& mlir_module_serialized,
           int calling_convention_version, int communication_buffer_width,
           std::vector<std::string> platforms, std::string backend);
  ~Compiler() = default;

  /**
   * Refines a serialized StableHLO module for the engine ABI.
   *
   * @param n_atoms Compiled capacity for owned, ghost, and padding atoms. The
   *     value determines every particle-leading input shape.
   * @param graph_inputs Runtime graph buffers and shape-only capacity inputs.
   *     Shape-only inputs participate in refinement and are removed from the
   *     compiled ABI afterward.
   * @param particle_types Canonical dtypes of the named particle fields.
   * @param particle_names Particle field names in exported argument order.
   * @param global_types Canonical dtypes of the named global fields.
   * @param global_names Global field names in exported argument order.
   * @param engine_abi Dtype and species-numbering policy expected by the
   *     simulation-engine adapter.
   * @param output_fields Output descriptors in executable result order.
   */
  void compile(
      int n_atoms, std::vector<GraphInputDescriptor> graph_inputs,
      std::vector<xla::PrimitiveType> particle_types,
      std::vector<std::string> particle_names,
      std::vector<xla::PrimitiveType> global_types,
      std::vector<std::string> global_names, const EngineAbiSpec& engine_abi,
      std::vector<ModelProperties::OutputField> output_fields);

  mlir::ModuleOp module() const { return module_ref.get(); }

 private:
  mlir::MLIRContext context;
  mlir::MLIRContext export_context;

  std::string mlir_module_serialized;
  int calling_convention_version;
  int communication_buffer_width;
  std::vector<std::string> platforms;
  std::string backend;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref;
};

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_COMPILER_H_
