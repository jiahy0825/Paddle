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

#include "paddle/cinn/adt/dim_expr.h"
#include <type_traits>

namespace cinn::adt {

namespace {

template <typename T0, typename T1>
bool DimExprEqualImpl(const T0&, const T1&) {
  LOG(FATAL) << "Dead code";
}

bool DimExprEqualImpl(std::int64_t lhs, std::int64_t rhs) {
  return lhs == rhs;
}

bool DimExprEqualImpl(const SymbolicDim& lhs, const SymbolicDim& rhs) {
  return lhs == rhs;
}

bool DimExprEqualImpl(const Negative<DimExpr>& lhs,
                              const Negative<DimExpr>& rhs) {
  const auto& [lhs_arg0] = lhs.tuple();
  const auto& [rhs_arg0] = rhs.tuple();
  return lhs_arg0 == rhs_arg0;
}

bool DimExprEqualImpl(const Reciprocal<DimExpr>& lhs,
                              const Reciprocal<DimExpr>& rhs) {
  const auto& [lhs_arg0] = lhs.tuple();
  const auto& [rhs_arg0] = rhs.tuple();
  return lhs_arg0 == rhs_arg0;
}

bool DimExprEqualImpl(
    const Add<DimExpr, DimExpr>& lhs,
    const Add<DimExpr, DimExpr>& rhs) {
  const auto& [lhs_arg0, lhs_arg1] = lhs.tuple();
  const auto& [rhs_arg0, rhs_arg1] = rhs.tuple();
  return lhs_arg0 == rhs_arg0 && lhs_arg1 == rhs_arg1;
}

bool DimExprEqualImpl(
    const Mul<DimExpr, DimExpr>& lhs,
    const Mul<DimExpr, DimExpr>& rhs) {
  const auto& [lhs_arg0, lhs_arg1] = lhs.tuple();
  const auto& [rhs_arg0, rhs_arg1] = rhs.tuple();
  return lhs_arg0 == rhs_arg0 && lhs_arg1 == rhs_arg1;
}

bool DimExprEqualImpl(
    const BroadcastedDim<DimExpr, DimExpr>& lhs,
    const BroadcastedDim<DimExpr, DimExpr>& rhs) {
  const auto& [lhs_arg0, lhs_arg1] = lhs.tuple();
  const auto& [rhs_arg0, rhs_arg1] = rhs.tuple();
  return lhs_arg0 == rhs_arg0 && lhs_arg1 == rhs_arg1;
}

}  // namespace

bool operator==(const DimExpr& lhs, const DimExpr& rhs) {
  return std::visit(
      [](const auto& lhs, const auto& rhs) {
        if (std::is_same_v<std::decay_t<decltype(lhs)>,
                           std::decay_t<decltype(rhs)>>) {
          return DimExprEqualImpl(lhs, rhs);
        } else {
          return false;
        }
      },
      lhs.variant(),
      rhs.variant());
}

namespace {

std::size_t GetHashValueImpl(std::int64_t expr) { return expr; }

std::size_t GetHashValueImpl(const SymbolicDim& expr) {
  return expr.value().unique_id();
}

std::size_t GetHashValueImpl(const Negative<DimExpr>& expr) {
  const auto& [item] = expr.tuple();
  return -GetHashValue(item);
}

std::size_t GetHashValueImpl(const Reciprocal<DimExpr>& expr) {
  const auto& [item] = expr.tuple();
  return -GetHashValue(item);
}

std::size_t GetHashValueImpl(
    const Add<DimExpr, DimExpr>& expr) {
  const auto& [lhs, rhs] = expr.tuple();
  return hash_combine(GetHashValue(lhs), GetHashValue(rhs));
}

std::size_t GetHashValueImpl(
    const Mul<DimExpr, DimExpr>& expr) {
  const auto& [lhs, rhs] = expr.tuple();
  return hash_combine(GetHashValue(lhs), GetHashValue(rhs));
}

std::size_t GetHashValueImpl(
    const BroadcastedDim<DimExpr, DimExpr>& expr) {
  const auto& [lhs, rhs] = expr.tuple();
  return hash_combine(GetHashValue(lhs), GetHashValue(rhs));
}

}  // namespace

std::size_t GetHashValue(const DimExpr& expr) {
  return std::visit([&](const auto& impl) { return GetHashValueImpl(impl); },
                    expr.variant());
}

}  // namespace cinn::adt
