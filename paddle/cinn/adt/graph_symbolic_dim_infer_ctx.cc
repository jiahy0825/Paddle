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

#include "paddle/cinn/adt/graph_symbolic_dim_infer_ctx.h"

#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn::adt::config {

const std::vector<std::uint64_t>& GraphSymbolicDimInferCtx::GetInTensorsRanks(
    const hlir::framework::Node* node) const {
  const auto& iter = op2input_ranks_.find(node);
  if (iter != op2input_ranks_.end()) {
    return iter->second;
  }
  std::vector<std::uint64_t> ret{};
  for (const auto& graph_edge : node->inlinks_in_order()) {
    const hlir::framework::NodeData* tensor =
        graph_edge->source()->safe_as<hlir::framework::NodeData>();
    ret.emplace_back(GetTensorRank(tensor));
  }
  CHECK(op2input_ranks_.emplace_back(node, ret).second);
  return op2input_ranks_.at(node);
}

const std::vector<std::uint64_t>& GraphSymbolicDimInferCtx::GetOutTensorsRanks(
    const hlir::framework::Node* node) const {
  const auto& iter = op2output_ranks_.find(node);
  if (iter != op2output_ranks_.end()) {
    return iter->second;
  }
  std::vector<std::uint64_t> ret{};
  for (const auto& graph_edge : node->outlinks_in_order()) {
    const hlir::framework::NodeData* tensor =
        graph_edge->sink()->safe_as<hlir::framework::NodeData>();
    ret.emplace_back(GetTensorRank(tensor));
  }
  CHECK(op2output_ranks_.emplace_back(node, ret).second);
  return op2output_ranks_.at(node);
}

const SymbolicDimExpr& GraphSymbolicDimInferCtx::GetInputDimExpr(
    const hlir::framework::Node* node,
    std::size_t arg_idx,
    std::size_t dim_idx) const {
  const auto& edges = node->inlinks_in_order();
  CHECK_LT(arg_idx, edges.size());
  const hlir::framework::NodeData* tensor =
      edges.at(arg_idx)->source()->safe_as<hlir::framework::NodeData>();
  const auto& iter = tensor2symbolic_dim_exprs_.find(tensor);
  CHECK(iter != tensor2symbolic_dim_exprs_.end());
  CHECK_LT(dim_idx, iter->second.size());
  return iter->second.at(dim_idx);
}

SymbolicDimExpr* GraphSymbolicDimInferCtx::MutOutputDimExpr(
    const hlir::framework::Node* node,
    std::size_t arg_idx,
    std::size_t dim_idx) {
  const auto& edges = node->outlinks_in_order();
  CHECK_LT(arg_idx, edges.size());
  const hlir::framework::NodeData* tensor =
      edges.at(arg_idx)->sink()->safe_as<hlir::framework::NodeData>();
  std::size_t rank = GetTensorRank(tensor);
  CHECK_LT(dim_idx, rank);
  const auto& iter = tensor2symbolic_dim_exprs_.find(tensor);
  CHECK(iter == tensor2symbolic_dim_exprs_.end() ||
        iter->second.size() <= dim_idx);
  if (iter == tensor2symbolic_dim_exprs_.end()) {
    CHECK(tensor2symbolic_dim_exprs_
              .emplace_back(tensor, std::vector<SymbolicDimExpr>{})
              .second);
  }
  const auto& symbolic_exprs_iter = tensor2symbolic_dim_exprs_.find(tensor);
  while (symbolic_exprs_iter->second.size() < dim_idx) {
    symbolic_exprs_iter->second.emplace_back(SymbolicDimExpr{});
  }
  return &symbolic_exprs_iter->second.at(dim_idx);
}

const framework::AttrMapType& GraphSymbolicDimInferCtx::GetAttributeMap(
    const hlir::framework::Node* node) const {
  return op_node->attrs.attr_store;
}

std::size_t GraphSymbolicDimInferCtx::GetTensorRank(
    const hlir::framework::NodeData* tensor) const {
  const auto& shape_dict =
      graph_->GetAttrs<absl::flat_hash_map<std::string, utils::ShapeType>>(
          "infershape");
  CHECK(shape_dict.count(tensor->id()))
      << "Can't find " << tensor->id() << " 's shape!";
  return shape_dict.at(tensor->id()).size();
}

}  // namespace cinn::adt::config
