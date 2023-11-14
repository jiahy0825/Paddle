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
#include "paddle/cinn/adt/arithmetic.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/hlir/framework/pir/group.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"
#include "paddle/pir/dialect/shape/utils/shape_optimization_utils.h"

namespace cinn::adt::config {

namespace {

// clang-format off
// Dim equations' configuration:
//
//     ShapeDialectConstraints = [ShapeDialectConstraint]
//     ShapeDialectConstraint = Equal ShapeDialectDimExpr ShapeDialectDimExpr
//
//     ShapeDialectDimExpr = ShapeDialectAtomicDim
//                         | Product ShapeDialectAtomicDim
//
//     ShapeDialectAtomicDim = int64_t | ShapeDialectSymbolicDim
//     ShapeDialectSymbolicDim = (::pir::Value, tAxis int)
//
//
// Dim equations' variables:
//
//     ShapeDialectSymbolicDim
//
// Dim equations' functions:
// DimFunction = DimIdentity (tOut ShapeDialectSymbolicDim) (tIn ShapeDialectSymbolicDim)
//             | DimProduct (tOut ShapeDialectSymbolicDim) [tIn ShapeDialectSymbolicDim]
//             | DimReciprocal (tOut ShapeDialectSymbolicDim) (tIn ShapeDialectSymbolicDim)
//
// Dim equations' solutions:
//
//     DimExpr
// clang-format on

// ShapeDialectSymbolicDim = (::pir::Value, tAxis int)
struct ShapeDialectSymbolicDim {
 ::pir::Value tensor;
 int axis;

 bool operator==(const ShapeDialectSymbolicDim& other) const {
   return this->tensor == other.tensor && this->axis == other.tensor;
 }
};
// ShapeDialectAtomicDim = int64_t | ShapeDialectSymbolicDim
DEFINE_ADT_UNION(ShapeDialectAtomicDim, std::int64_t, ShapeDialectSymbolicDim);
// ShapeDialectDimExpr = ShapeDialectAtomicDim
//                     | Product ShapeDialectAtomicDim
DEFINE_ADT_UNION(ShapeDialectDimExpr,
                 ShapeDialectAtomicDim,
                 Product<ShapeDialectAtomicDim>);
// ShapeDialectConstraint = Equal ShapeDialectDimExpr ShapeDialectDimExpr
using ShapeDialectConstraint = Equal<ShapeDialectDimExpr, ShapeDialectDimExpr>;
// ShapeDialectConstraints = [ShapeDialectConstraint]
using ShapeDialectConstraints = List<ShapeDialectConstraint>;

template<typename T0, typename T1>
struct DimIdentity;

// DimIdentity (tOut ShapeDialectSymbolicDim) (tIn ShapeDialectSymbolicDim)
template<>
struct DimIdentity<tOut<ShapeDialectSymbolicDim>, tIn<ShapeDialectSymbolicDim>>
    : public Tuple<tOut<ShapeDialectSymbolicDim>, tIn<ShapeDialectSymbolicDim>> {
  using Tuple<tOut<ShapeDialectSymbolicDim>, tIn<ShapeDialectSymbolicDim>>::Tuple;
};

template<typename T0, typename T1>
struct DimProduct;

// DimProduct (tOut ShapeDialectSymbolicDim) [tIn ShapeDialectSymbolicDim]
template<>
struct DimProduct<tOut<ShapeDialectSymbolicDim>, List<tIn<ShapeDialectSymbolicDim>>>
    : public Tuple<tOut<ShapeDialectSymbolicDim>, List<tIn<ShapeDialectSymbolicDim>>> {
  using Tuple<tOut<ShapeDialectSymbolicDim>, List<tIn<ShapeDialectSymbolicDim>>>::Tuple;
};

// DimReciprocal (tOut ShapeDialectSymbolicDim) (tIn ShapeDialectSymbolicDim)
template<>
struct DimReciprocal<tOut<ShapeDialectSymbolicDim>, tIn<ShapeDialectSymbolicDim>>
    : public Tuple<tOut<ShapeDialectSymbolicDim>, tIn<ShapeDialectSymbolicDim>> {
  using Tuple<tOut<ShapeDialectSymbolicDim>, tIn<ShapeDialectSymbolicDim>>::Tuple;
};

// DimFunction = DimIdentity (tOut ShapeDialectSymbolicDim) (tIn ShapeDialectSymbolicDim)
//             | DimProduct (tOut ShapeDialectSymbolicDim) [tIn ShapeDialectSymbolicDim]
//             | DimReciprocal (tOut ShapeDialectSymbolicDim) (tIn ShapeDialectSymbolicDim)

DEFINE_ADT_UNION(DimFunction,
                 DimIdentity<tOut<ShapeDialectSymbolicDim>, tIn<ShapeDialectSymbolicDim>>,
                 DimProduct<tOut<ShapeDialectSymbolicDim>, List<tIn<ShapeDialectSymbolicDim>>>,
                 DimReciprocal<tOut<ShapeDialectSymbolicDim>, tIn<ShapeDialectSymbolicDim>>);
}

}

namespace std {

template<>
struct hash<cinn::adt::config::ShapeDialectSymbolicDim> final {
  using namespace cinn::adt::config;
  std::size_t operator()(const ShapeDialectSymbolicDim& dim) const {
    return hash_combine(std::hash<::pir::Value>()(dim.tensor), dim.axis);
  }
};

}

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

namespace {

template <typename DoEachT>
void VisitEachTensorPair(const ::pir::Operation* op_node,
                         const DoEachT& DoEach) {
  std::vector<::pir::Value> all_tensors{};
  for (const ::pir::Value tensor : op_node->operands_source()) {
    all_tensors.emplace_back(tensor);
  }
  for (const ::pir::Value tensor : op_node->results()) {
    all_tensors.emplace_back(tensor);
  }
  for (std::size_t i = 0; i < all_tensors.size(); ++i) {
    for (std::size_t j = i + 1; j < all_tensors.size(); ++j) {
      DoEach(all_tensors.at(i), all_tensors.at(j));
    }
  }
}

void BuildTensorShapeDialectConstraints(
    const ::pir::Value& lhs,
    const ::pir::Value& rhs,
    const ::pir::SymbolicDimMgr* symbolic_dim_mgr,
    ShapeDialectConstraints* ret) {
  const auto& lhs_symbolic_dim_ops =
      symbolic_dim_mgr->CreateSymbolicDimsForRankedValue(lhs);
  ADT_TODO();
}

void BuildOpShapeDialectConstraints(
    const ::pir::Operation* op_node,
    const ::pir::SymbolicDimMgr* symbolic_dim_mgr,
    ShapeDialectConstraints* ret) {
  VisitEachTensorPair(
      op_node, [&](const ::pir::Value& lhs, const ::pir::Value& rhs) {
        BuildTensorShapeDialectConstraints(lhs, rhs, symbolic_dim_mgr, ret);
      });
}

ShapeDialectConstraints BuildGraphShapeDialectConstraints(
    const cinn::hlir::framework::pir::Group* group,
    const ::pir::SymbolicDimMgr* symbolic_dim_mgr) {
  ShapeDialectConstraints ret{};
  for (const ::pir::Operation* op_node : group->ops) {
    BuildOpShapeDialectConstraints(op_node, symbolic_dim_mgr, &ret);
  }
  return ret;
}

// ADT_TODO();
using GraphView =
    EquationGraphTopoWalker<ShapeDialectSymbolicDim, const DimFunction*>;

GraphView MakeEquationGraphView(const ShapeDialectConstraints& constraints,
                                const cinn::hlir::framework::pir::Group* group,
                                const ::pir::SymbolicDimMgr* symbolic_dim_mgr) {
  ADT_TODO();
}

std::unordered_map<ShapeDialectSymbolicDim, DimExpr> MakeEquationStartExpr(
    const GraphView& graph_view,
    const cinn::hlir::framework::pir::Group* group,
    const ::pir::SymbolicDimMgr* symbolic_dim_mgr) {
  ADT_TODO();
}

std::unordered_map<::pir::Value, std::vector<std::optional<DimExpr>>>
SolveShapeDialectConstraints(
    const GraphView& graph_view,
    const std::unordered_map<ShapeDialectSymbolicDim, DimExpr>&
        equation_start) {
  ADT_TODO();
}

}  // namespace

void GraphSymbolicDimInferCtx::InitTensorDimExpr() {
  ShapeDialectConstraints constraints =
      BuildGraphShapeDialectConstraints(group_, symbolic_dim_mgr_);

  const auto& graph_view =
      MakeEquationGraphView(constraints, group_, symbolic_dim_mgr_);

  const auto& equation_start =
      MakeEquationStartExpr(graph_view, group_, symbolic_dim_mgr_);

  tensor2dim_exprs_ = SolveShapeDialectConstraints(graph_view, equation_start);
}

}  // namespace cinn::adt::config
