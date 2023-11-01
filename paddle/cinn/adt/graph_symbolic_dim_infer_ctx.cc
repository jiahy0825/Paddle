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

#include "paddle/cinn/adt/symbolic_dim_expr_simplifier.h"
#include "paddle/cinn/adt/unique_id.h"
#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn::adt::config {

namespace {

const std::vector<int32_t>& GetShape(const hlir::framework::Graph* graph,
                                     const hlir::framework::NodeData* tensor) {
  const auto& shape_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, utils::ShapeType>>(
          "infershape");
  CHECK(shape_dict.count(tensor->id()))
      << "Can't find " << tensor->id() << " 's shape!";
  return shape_dict.at(tensor->id());
}

std::size_t GetTensorRank(const hlir::framework::Graph* graph,
                          const hlir::framework::NodeData* tensor) {
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

std::vector<const hlir::framework::Node*> GetTopoOrderOpNodes(
    const hlir::framework::Graph* graph) {
  std::vector<const hlir::framework::Node*> ret{};
  std::vector<common::GraphNode*> topo_nodes =
      std::get<0>(graph->topological_order());
  for (const common::GraphNode* graph_node : topo_nodes) {
    const hlir::framework::Node* op_node =
        graph_node->safe_as<hlir::framework::Node>();
    // if node is NodeData or not op, continue.
    if (!op_node || op_node->op() == nullptr) {
      continue;
    }
    ret.emplace_back(op_node);
  }
  return ret;
}

}  // namespace

void GraphSymbolicDimInferCtx::InitOp2TensorRanks() {
  for (const hlir::framework::Node* op_node : GetTopoOrderOpNodes(graph_)) {
    const auto& input_ranks = GetOpInputRanks(graph_, op_node);
    if (op2input_ranks_.find(op_node) == op2input_ranks_.end()) {
      op2input_ranks_.emplace(op_node, input_ranks);
    } else {
      CHECK(input_ranks == op2input_ranks_.at(op_node));
    }
  }
}

namespace {

std::unordered_set<std::string> GetAllOutputNames(
    const std::vector<const hlir::framework::Node*>& nodes) {
  std::unordered_set<std::string> output_names;
  for (const auto* node : nodes) {
    for (const auto& link : node->outlinks()) {
      const auto* out_node = link->sink()->safe_as<hlir::framework::NodeData>();
      output_names.emplace(out_node->id());
    }
  }
  return output_names;
}

std::vector<const hlir::framework::NodeData*> GetFeedList(
    const std::vector<const hlir::framework::Node*>& nodes,
    const std::unordered_set<std::string>& out_names) {
  std::vector<const hlir::framework::NodeData*> ret{};
  // if the op's input var name cannot found in out_names, it is the group's
  // feed var
  std::unordered_set<std::string> feed_names;
  for (const auto* node : nodes) {
    for (const auto& link : node->inlinks()) {
      const auto* in_node =
          link->source()->safe_as<hlir::framework::NodeData>();
      if (!out_names.count(in_node->id()) && !feed_names.count(in_node->id())) {
        feed_names.emplace(in_node->id());
        ret.emplace_back(in_node);
      }
    }
  }
  return ret;
}

std::vector<std::optional<SymbolicDimExpr>> MakeSymbolicDimExprForTensor(
    const hlir::framework::Graph* graph,
    const hlir::framework::NodeData* node_data) {
  std::vector<std::optional<SymbolicDimExpr>> ret{};

  const std::vector<int32_t>& shape = GetShape(graph, node_data);
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i == 0) {
      static SymbolicDimExpr temp_elementwise_dim_expr{
          SymbolicDim{UniqueId::New()}};
      ret.emplace_back(temp_elementwise_dim_expr);
    } else {
      ret.emplace_back(SymbolicDimExpr{shape.at(i)});
    }
  }
  return ret;
}

}  // namespace

void GraphSymbolicDimInferCtx::InitGraphInputSymbolicDimExpr() {
  std::vector<const hlir::framework::Node*> topo_op_nodes =
      GetTopoOrderOpNodes(graph_);
  std::vector<const hlir::framework::NodeData*> feed_list =
      GetFeedList(topo_op_nodes, GetAllOutputNames(topo_op_nodes));
  for (const hlir::framework::NodeData* node_data : feed_list) {
    CHECK(
        tensor2symbolic_dim_exprs_
            .emplace(node_data, MakeSymbolicDimExprForTensor(graph_, node_data))
            .second);
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
  std::size_t rank = GetTensorRank(graph_, tensor);
  CHECK_LT(dim_idx, rank);
  auto* opt_symbolic_dims = &tensor2symbolic_dim_exprs_[tensor];
  if (dim_idx >= opt_symbolic_dims->size()) {
    opt_symbolic_dims->resize(dim_idx + 1);
  }
  opt_symbolic_dims->at(dim_idx) = SimplifySymbolicDimExpr(value);
}

const hlir::framework::AttrMapType& GraphSymbolicDimInferCtx::GetAttributeMap(
    const hlir::framework::Node* op_node) const {
  return op_node->attrs.attr_store;
}

}  // namespace cinn::adt::config
