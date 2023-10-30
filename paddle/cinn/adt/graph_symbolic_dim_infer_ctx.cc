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

namespace {

std::size_t GraphSymbolicDimInferCtx::GetTensorRank(
    const hlir::framework::Graph* graph,
    const hlir::framework::NodeData* tensor) const {
  const auto& shape_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, utils::ShapeType>>(
          "infershape");
  CHECK(shape_dict.count(tensor->id()))
      << "Can't find " << tensor->id() << " 's shape!";
  return shape_dict.at(tensor->id()).size();
}

std::vector<std::uint64_t> GetOpInputRanks(const hlir::framework::Graph* graph,
                                           const hlir::framework::Node* node) {
  std::vector<std::uint64_t> ret{};
  for (const auto& graph_edge : node->inlinks_in_order()) {
    const hlir::framework::NodeData* tensor =
        graph_edge->source()->safe_as<hlir::framework::NodeData>();
    ret.emplace_back(GetTensorRank(graph, tensor));
  }
  return ret;
}

}  // namespace

void GraphSymbolicDimInferCtx::InitOp2TensorRanks() {
  std::vector<hlir::framework::GraphNode*> topo_nodes =
      std::get<0>(graph_->topological_order());
  for (const hlir::framework::GraphNode* graph_node : topo_nodes) {
    const hlir::framework::Node* op_node = graph_node->safe_as<Node>();
    CHECK(op_node != nullptr && op_node->op() != nullptr);

    const auto& input_ranks = GetOpInputRanks(graph_, op_node);
    if (op2input_ranks_.find(op_node) == op2input_ranks_.end()) {
      op2input_ranks_.emplace_back(op_node, input_ranks);
    } else {
      CHECK_EQ(input_ranks, op2input_ranks_.at(op_node));
    }
  }
}

const std::vector<std::uint64_t>& GraphSymbolicDimInferCtx::GetInTensorsRanks(
    const hlir::framework::Node* node) const {
  const auto& iter = op2input_ranks_.find(node);
  CHECK(iter != op2input_ranks_.end());
  return iter->second;
}

std::uint64_t GraphSymbolicDimInferCtx::GetNumOutTensors(
    const hlir::framework::Node* node) const {
  return node->outlinks_in_order().size();
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
  const auto& opt_symbolic_dim_expr = iter->second.at(dim_idx);
  CHECK(opt_symbolic_dim_expr.has_value());
  return opt_symbolic_dim_expr.value();
}

void GraphSymbolicDimInferCtx::SetOutputDimExpr(
    const hlir::framework::Node* node,
    std::size_t arg_idx,
    std::size_t dim_idx,
    const SymbolicDimExpr& value) {
  const auto& edges = node->outlinks_in_order();
  CHECK_LT(arg_idx, edges.size());
  const hlir::framework::NodeData* tensor =
      edges.at(arg_idx)->sink()->safe_as<hlir::framework::NodeData>();
  std::size_t rank = GetTensorRank(tensor);
  CHECK_LT(dim_idx, rank);
  auto* opt_symbolic_dims = &tensor2symbolic_dim_exprs_[tensor];
  if (dim_idx >= opt_symbolic_dims->size()) {
    opt_symbolic_dims->resize(dim_idx + 1);
  }
  opt_symbolic_dims->at(dim_idx) = value;
}

const framework::AttrMapType& GraphSymbolicDimInferCtx::GetAttributeMap(
    const hlir::framework::Node* node) const {
  return op_node->attrs.attr_store;
}

}  // namespace cinn::adt::config
