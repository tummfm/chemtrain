/*
Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich

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

#include "connector/xla_call_module_loader.h"

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/Version.h"

namespace jcn {
namespace {

class AbstractArgumentTest : public ::testing::Test {
 protected:
  AbstractArgumentTest() {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    mlir::stablehlo::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> Parse(const std::string& source) {
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }

  mlir::MLIRContext context;
};

TEST_F(AbstractArgumentTest, RemovesMultipleArgumentsAndPreservesOrder) {
  auto module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<3xi1>,
                      %arg2: tensor<4xf32>, %arg3: tensor<5xi1>)
          -> tensor<2xi32> {
        return %arg0 : tensor<2xi32>
      }
    })mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<mlir::func::FuncOp>("main");

  EXPECT_TRUE(RemoveAbstractArgumentsFromMain(*module, main, {1, 3}).ok());
  ASSERT_EQ(main.getNumArguments(), 2);
  EXPECT_EQ(main.getArgument(0).getType(),
            mlir::RankedTensorType::get({2}, mlir::IntegerType::get(&context, 32)));
  EXPECT_EQ(main.getArgument(1).getType(),
            mlir::RankedTensorType::get({4}, mlir::Float32Type::get(&context)));
}

TEST_F(AbstractArgumentTest, AcceptsUseEliminatedByCanonicalization) {
  auto module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<3xi1>)
          -> tensor<2xi32> {
        %unused = stablehlo.convert %arg1 : (tensor<3xi1>) -> tensor<3xi32>
        return %arg0 : tensor<2xi32>
      }
    })mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<mlir::func::FuncOp>("main");

  EXPECT_TRUE(RemoveAbstractArgumentsFromMain(*module, main, {1}).ok());
  EXPECT_EQ(main.getNumArguments(), 1);
}

TEST_F(AbstractArgumentTest, RejectsLiveSemanticUseWithoutPartialMutation) {
  auto module = Parse(R"mlir(
    module {
      func.func @main(%arg0: tensor<2xi32>, %arg1: tensor<3xi1>,
                      %arg2: tensor<4xf32>) -> tensor<3xi1> {
        return %arg1 : tensor<3xi1>
      }
    })mlir");
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<mlir::func::FuncOp>("main");

  const absl::Status status =
      RemoveAbstractArgumentsFromMain(*module, main, {1, 2});
  EXPECT_FALSE(status.ok());
  EXPECT_NE(std::string(status.message()).find("value-semantic"),
            std::string::npos);
  EXPECT_EQ(main.getNumArguments(), 3);
}

TEST_F(AbstractArgumentTest, ValidatesOutputShapesAndRewritesOutputDtypes) {
  auto module = Parse(R"mlir(
    module {
      func.func @main(%position: tensor<4x3xf32>, %species: tensor<4xi32>,
                      %nlocal: tensor<i32>, %nghost: tensor<i32>)
          -> (tensor<4x3xf32>, tensor<4xf32>, tensor<6xf32>) {
        %energy = stablehlo.constant dense<0.0> : tensor<4xf32>
        %stress = stablehlo.constant dense<0.0> : tensor<6xf32>
        return %position, %energy, %stress :
            tensor<4x3xf32>, tensor<4xf32>, tensor<6xf32>
      }
    })mlir");
  ASSERT_TRUE(module);

  std::string artifact;
  llvm::raw_string_ostream stream(artifact);
  ASSERT_TRUE(mlir::succeeded(mlir::stablehlo::serializePortableArtifact(
      *module, mlir::vhlo::Version::getCurrentVersion().toString(), stream)));
  stream.flush();

  auto loader_or = XlaCallModuleLoader::Create(
      &context, 10, artifact, {}, {"cpu"}, 4, false, false);
  ASSERT_TRUE(loader_or.ok()) << loader_or.status();
  auto loader = std::move(loader_or).value();

  EngineAbiSpec abi;
  abi.position_dtype = TensorDtype::F64;
  abi.default_output_dtype = TensorDtype::F64;
  abi.output_dtypes = {{"U", TensorDtype::F32}};
  const std::vector<ModelProperties::OutputField> outputs = {
      {"F", ModelProperties::OutputScope::PARTICLE, {3}, 3},
      {"U", ModelProperties::OutputScope::PARTICLE, {}, 1},
      {"V", ModelProperties::OutputScope::LOCAL, {6}, 6},
  };
  ASSERT_TRUE(
      loader->WrapMainForEngineAbi(abi, {"species"}, outputs).ok());

  auto position = mlir::cast<mlir::RankedTensorType>(loader->InputTypes()[0]);
  auto force = mlir::cast<mlir::RankedTensorType>(loader->OutputTypes()[0]);
  auto energy = mlir::cast<mlir::RankedTensorType>(loader->OutputTypes()[1]);
  auto virial = mlir::cast<mlir::RankedTensorType>(loader->OutputTypes()[2]);
  EXPECT_TRUE(position.getElementType().isF64());
  EXPECT_TRUE(force.getElementType().isF64());
  EXPECT_TRUE(energy.getElementType().isF32());
  EXPECT_TRUE(virial.getElementType().isF64());
}

}  // namespace
}  // namespace jcn
