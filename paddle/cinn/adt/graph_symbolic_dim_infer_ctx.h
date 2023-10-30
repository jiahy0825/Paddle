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

#include <unordered_map>
#include <vector>

#include "paddle/cinn/adt/symbolic_dim_expr.h"

namespace cinn::hlir::framework {
class Graph;
class NodeData;
class Node;
}  // namespace cinn::hlir::framework

namespace cinn::adt::config {

class GraphSymbolicDimInferCtx {
 public:
  GraphSymbolicDimInferCtx(const GraphSymbolicDimInferCtx&) = delete;
  GraphSymbolicDimInferCtx(GraphSymbolicDimInferCtx&&) = delete;

  explicit GraphSymbolicDimInferCtx(const hlir::framework::Graph* graph)
      : graph_(graph) {}

  const std::vector<std::uint64_t>& GetInTensorsRanks(
      const hlir::framework::Node* node) const;

  const std::vector<std::uint64_t>& GetOutTensorsRanks(
      const hlir::framework::Node* node) const;

  const SymbolicDimExpr& GetInputDimExpr(const hlir::framework::Node* node,
                                         std::size_t arg_idx,
                                         std::size_t dim_idx) const;

  SymbolicDimExpr* MutOutputDimExpr(const hlir::framework::Node* node,
                                    std::size_t arg_idx,
                                    std::size_t dim_idx);

  const framework::AttrMapType& GetAttributeMap(
      const hlir::framework::Node* node) const;

 private:
  std::size_t GetTensorRank(const hlir::framework::NodeData* tensor) const;

  const hlir::framework::Graph* graph_;
  std::unordered_map<const hlir::framework::NodeData*,
                     std::vector<SymbolicDimExpr>>
      tensor2symbolic_dim_exprs_;
  std::unordered_map<const hlir::framework::Node*, std::vector<std::uint64_t>>
      op2input_ranks_;
  std::unordered_map<const hlir::framework::Node*, std::vector<std::uint64_t>>
      op2output_ranks_;
};

}  // namespace cinn::adt::config
