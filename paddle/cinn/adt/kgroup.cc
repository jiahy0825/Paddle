// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/adt/kgroup.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/igroup.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/m_ir.h"
#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/adt/schedule_dim.h"
#include "paddle/cinn/hlir/framework/graph.h"

namespace cinn::adt {

using AnchorTensor = Variable;

namespace {

List<LoopSize> GetDefaultScheduleSizesFromTensorImpl(
    const adapter::Tensor& tensor) {
  List<LoopSize> ret{};
  for (int32_t dim : tensor.GetShape()) {
    ret->emplace_back(LoopSize{dim});
  }
  return ret;
}

namespace {

LoopSize MakeLoopSizeImpl(const SymbolicDim& symbolic_dim) {
  return LoopSize{symbolic_dim};
}

LoopSize MakeLoopSizeImpl(const std::int64_t dim) { return LoopSize{dim}; }

LoopSize MakeLoopSize(const Union<SymbolicDim, std::int64_t>& dim) {
  return std::visit([&](const auto& impl) { return MakeLoopSizeImpl(impl); },
                    dim.variant());
}

}  // namespace

List<LoopSize> GetDefaultScheduleSizesFromTensorImpl(
    const adapter::DynamicTensor& tensor) {
  List<LoopSize> ret{};
  for (const Union<SymbolicDim, std::int64_t>& dim : tensor.GetShape()) {
    ret->emplace_back(MakeLoopSize(dim));
  }
  return ret;
}

List<LoopSize> GetDefaultScheduleSizesFromTensorImpl(
    const SSAShadowTensor& tensor) {
  ADT_TODO();
}

List<LoopSize> GetDefaultScheduleSizesFromTensorImpl(
    const TempStorage& tensor) {
  ADT_TODO();
}

List<LoopSize> GetDefaultScheduleSizesFromTensor(const Tensor& tensor) {
  return std::visit(
      [&](const auto& impl) {
        return GetDefaultScheduleSizesFromTensorImpl(impl);
      },
      tensor.variant());
}

}  // namespace

List<LoopSize> KGroup::GetDefaultScheduleSizes(
    const std::shared_ptr<IGroup>& igroup) const {
  const Tensor& tensor = igroup->anchor_tensor();
  return GetDefaultScheduleSizesFromTensor(tensor);
}

}  // namespace cinn::adt
