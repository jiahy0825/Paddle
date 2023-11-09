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

#include "paddle/cinn/adt/dim_expr_simplifier.h"
#include "paddle/cinn/adt/unique_id.h"
#include "paddle/cinn/hlir/framework/pir/group.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"

namespace cinn::adt::config {

namespace {

std::vector<int32_t> GetShape(const ::pir::Value& tensor) {
  std::vector<int> tensor_shape =
      hlir::framework::pir::CompatibleInfo::ValueShape(tensor);
  std::vector<int32_t> ret{};
  for (int32_t dim : tensor_shape) {
    ret.push_back(dim);
  }
  return ret;
}

std::size_t GetTensorRank(const ::pir::Value& tensor) {
  return hlir::framework::pir::CompatibleInfo::ValueShape(tensor).size();
}

std::vector<std::uint64_t> GetOpInputRanks(const ::pir::Operation* node) {
  std::vector<std::uint64_t> ret{};
  for (const ::pir::Value& tensor : node->operands_source()) {
    ret.emplace_back(GetTensorRank(tensor));
  }
  return ret;
}

std::vector<const ::pir::Operation*> GetTopoOrderOpNodes(
    const hlir::framework::pir::Group* group) {
  std::vector<const ::pir::Operation*> ret{};
  for (const ::pir::Operation* op_node : group->ops) {
    ret.emplace_back(op_node);
  }
  return ret;
}

}  // namespace

void GraphSymbolicDimInferCtx::InitOp2TensorRanks() {
  for (const ::pir::Operation* op_node : GetTopoOrderOpNodes(group_)) {
    std::vector<std::uint64_t> input_ranks = GetOpInputRanks(op_node);
    if (op2input_ranks_.find(op_node) == op2input_ranks_.end()) {
      op2input_ranks_.emplace(op_node, input_ranks);
    } else {
      CHECK(input_ranks == op2input_ranks_.at(op_node));
    }
  }
}

namespace {

std::unordered_set<std::string> GetAllOutputNames(
    const std::vector<const ::pir::Operation*>& nodes) {
  std::unordered_set<std::string> output_names;
  for (const auto* op_node : nodes) {
    for (const ::pir::Value& out_node :
         const_cast<::pir::Operation*>(op_node)->results()) {
      output_names.emplace(
          hlir::framework::pir::CompatibleInfo::ValueName(out_node));
    }
  }
  return output_names;
}

std::vector<::pir::Value> GetFeedList(
    const std::vector<const ::pir::Operation*>& op_nodes,
    const std::unordered_set<std::string>& out_names) {
  std::vector<::pir::Value> ret{};
  // if the op's input var name cannot found in out_names, it is the group's
  // feed var
  std::unordered_set<std::string> feed_names;
  for (const auto* op_node : op_nodes) {
    for (const ::pir::Value in_node : op_node->operands_source()) {
      const auto& node_id =
          hlir::framework::pir::CompatibleInfo::ValueName(in_node);
      if (!out_names.count(node_id) && !feed_names.count(node_id)) {
        feed_names.emplace(node_id);
        ret.emplace_back(in_node);
      }
    }
  }
  return ret;
}

std::vector<std::optional<DimExpr>> MakeDimExprForTensor(
    const ::pir::Value& node_data) {
  std::vector<std::optional<DimExpr>> ret{};

  std::vector<int32_t> shape = GetShape(node_data);
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i == 0) {
      static DimExpr temp_elementwise_dim_expr{SymbolicDim{UniqueId::New()}};
      ret.emplace_back(temp_elementwise_dim_expr);
    } else {
      ret.emplace_back(DimExpr{shape.at(i)});
    }
  }
  return ret;
}

}  // namespace

void GraphSymbolicDimInferCtx::InitGraphInputDimExpr() {
  std::vector<const ::pir::Operation*> topo_op_nodes =
      GetTopoOrderOpNodes(group_);
  std::vector<::pir::Value> feed_list =
      GetFeedList(topo_op_nodes, GetAllOutputNames(topo_op_nodes));
  for (const ::pir::Value node_data : feed_list) {
    CHECK(tensor2dim_exprs_.emplace(node_data, MakeDimExprForTensor(node_data))
              .second);
  }
}

const std::vector<std::uint64_t>& GraphSymbolicDimInferCtx::GetInTensorsRanks(
    const ::pir::Operation* node) const {
  const auto& iter = op2input_ranks_.find(node);
  CHECK(iter != op2input_ranks_.end());
  return iter->second;
}

std::uint64_t GraphSymbolicDimInferCtx::GetNumOutTensors(
    const ::pir::Operation* node) const {
  return node->num_results();
}

const DimExpr& GraphSymbolicDimInferCtx::GetInputDimExpr(
    const ::pir::Operation* node,
    std::size_t arg_idx,
    std::size_t dim_idx) const {
  CHECK_LT(arg_idx, node->num_operands());
  const ::pir::Value tensor = node->operand_source(arg_idx);
  const auto& iter = tensor2dim_exprs_.find(tensor);
  CHECK(iter != tensor2dim_exprs_.end());
  CHECK_LT(dim_idx, iter->second.size());
  const auto& opt_dim_expr = iter->second.at(dim_idx);
  CHECK(opt_dim_expr.has_value());
  return opt_dim_expr.value();
}

void GraphSymbolicDimInferCtx::SetOutputDimExpr(const ::pir::Operation* node,
                                                std::size_t arg_idx,
                                                std::size_t dim_idx,
                                                const DimExpr& value) {
  CHECK_LT(arg_idx, node->num_results());
  const ::pir::Value tensor =
      const_cast<::pir::Operation*>(node)->result(arg_idx);
  std::size_t rank = GetTensorRank(tensor);
  CHECK_LT(dim_idx, rank);
  auto* opt_symbolic_dims = &tensor2dim_exprs_[tensor];
  if (dim_idx >= opt_symbolic_dims->size()) {
    opt_symbolic_dims->resize(dim_idx + 1);
  }
  opt_symbolic_dims->at(dim_idx) = SimplifyDimExpr(value);
}

cinn::utils::AttributeMap GraphSymbolicDimInferCtx::GetAttributeMap(
    const ::pir::Operation* op_node) const {
  return hlir::framework::pir::CompatibleInfo::ConvertAttributes(*op_node);
}

}  // namespace cinn::adt::config
