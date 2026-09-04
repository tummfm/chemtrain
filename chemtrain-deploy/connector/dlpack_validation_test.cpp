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

#include "connector/dlpack_validation.h"

#include <cstdint>

#include "connector/jcn_buffer_internal.h"
#include "gtest/gtest.h"

namespace jcn {
namespace {

constexpr uint8_t kDlInt = 0;
constexpr uint8_t kDlUInt = 1;
constexpr uint8_t kDlFloat = 2;

constexpr DLDataType kS32{kDlInt, 32, 1};
constexpr DLDataType kF32{kDlFloat, 32, 1};
constexpr DLDataType kF64{kDlFloat, 64, 1};

struct TensorFixture {
  int data = 0;
  int64_t shape[2] = {4, 3};
  int64_t strides[2] = {3, 1};
  DLManagedTensor managed{};
  jcn_buffer buffer{};

  TensorFixture() {
    managed.dl_tensor.data = &data;
    managed.dl_tensor.device = {kDLCPU, 0};
    managed.dl_tensor.ndim = 2;
    managed.dl_tensor.dtype = {kDlInt, 32, 1};
    managed.dl_tensor.shape = shape;
    buffer.tensor = &managed;
  }
};

TEST(DlpackValidationTest, AcceptsExactContiguousS32Tensor) {
  TensorFixture fixture;
  EXPECT_NO_THROW(ValidateDlpackInput(
      &fixture.buffer, {4, 3}, kS32, "neighbors"));
}

TEST(DlpackValidationTest, AcceptsExactFloatingTensors) {
  TensorFixture fixture;
  fixture.managed.dl_tensor.dtype = kF32;
  EXPECT_NO_THROW(ValidateDlpackInput(
      &fixture.buffer, {4, 3}, kF32, "positions"));
  fixture.managed.dl_tensor.dtype = kF64;
  EXPECT_NO_THROW(ValidateDlpackInput(
      &fixture.buffer, {4, 3}, kF64, "positions"));
}

TEST(DlpackValidationTest, RejectsStridedTensor) {
  TensorFixture fixture;
  fixture.managed.dl_tensor.strides = fixture.strides;
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
}

TEST(DlpackValidationTest, RequiresExactShape) {
  TensorFixture fixture;
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 2}, kS32, "neighbors"),
               std::runtime_error);
  fixture.shape[0] = 5;
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
}

TEST(DlpackValidationTest, RequiresExactRank) {
  TensorFixture fixture;
  fixture.managed.dl_tensor.ndim = 1;
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
}

TEST(DlpackValidationTest, RequiresExactDtype) {
  TensorFixture fixture;
  fixture.managed.dl_tensor.dtype = {kDlUInt, 32, 1};
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
  fixture.managed.dl_tensor.dtype = {kDlInt, 64, 1};
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
  fixture.managed.dl_tensor.dtype = {kDlInt, 32, 2};
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
}

TEST(DlpackValidationTest, RejectsMissingTensorMetadata) {
  TensorFixture fixture;
  fixture.managed.dl_tensor.data = nullptr;
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
  fixture.managed.dl_tensor.data = &fixture.data;
  fixture.managed.dl_tensor.shape = nullptr;
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
  fixture.buffer.tensor = nullptr;
  EXPECT_THROW(ValidateDlpackInput(
                   &fixture.buffer, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
  EXPECT_THROW(ValidateDlpackInput(
                   nullptr, {4, 3}, kS32, "neighbors"),
               std::runtime_error);
}

}  // namespace
}  // namespace jcn
