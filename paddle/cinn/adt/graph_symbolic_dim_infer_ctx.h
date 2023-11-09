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

#pragma once

#include <optional>
#include <unordered_map>
#include <vector>

#include "paddle/cinn/adt/dim_expr.h"
#include "paddle/cinn/utils/type_defs.h"
#include "paddle/pir/core/value.h"
namespace pir {
class Operation;
}

namespace cinn::hlir::framework::pir {
struct Group;
}  // namespace cinn::hlir::framework::pir

namespace cinn::adt::config {

class GraphSymbolicDimInferCtx {
 public:
  GraphSymbolicDimInferCtx(const GraphSymbolicDimInferCtx&) = delete;
  GraphSymbolicDimInferCtx(GraphSymbolicDimInferCtx&&) = delete;

  explicit GraphSymbolicDimInferCtx(
      const cinn::hlir::framework::pir::Group* group)
      : group_(group) {
    InitOp2TensorRanks();
    InitGraphInputDimExpr();
  }

  const cinn::hlir::framework::pir::Group* group() const { return group_; }

  const std::vector<std::uint64_t>& GetInTensorsRanks(
      const ::pir::Operation* node) const;

  std::uint64_t GetNumOutTensors(const ::pir::Operation* node) const;

  const DimExpr& GetInputDimExpr(const ::pir::Operation* node,
                                 std::size_t arg_idx,
                                 std::size_t dim_idx) const;

  const std::vector<std::optional<DimExpr>>& GetTensorDimExprs(
      const ::pir::Value tensor) const {
    const auto& iter = tensor2dim_exprs_.find(tensor);
    CHECK(iter != tensor2dim_exprs_.end());
    return iter->second;
  }

  void SetOutputDimExpr(const ::pir::Operation* node,
                        std::size_t arg_idx,
                        std::size_t dim_idx,
                        const DimExpr& value);

  cinn::utils::AttributeMap GetAttributeMap(const ::pir::Operation* node) const;

 private:
  void InitOp2TensorRanks();
  void InitGraphInputDimExpr();

  const cinn::hlir::framework::pir::Group* group_;
  std::unordered_map<::pir::Value, std::vector<std::optional<DimExpr>>>
      tensor2dim_exprs_;
  std::unordered_map<const ::pir::Operation*, std::vector<std::uint64_t>>
      op2input_ranks_;
};

}  // namespace cinn::adt::config
