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

void GraphSymbolicDimInferCtx::InitTensorDimExpr() {
  ShapeDialectConstraints constraints =
      BuildShapeDialectConstraints(group_, symbolic_dim_mgr_);

  const auto& equation_start = MakeEquationStartExpr(group_, symbolic_dim_mgr_);

  tensor2dim_exprs_ = SolveShapeDialectConstraints(constraints, equation_start);
}

}  // namespace cinn::adt::config
