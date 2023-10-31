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

#include "paddle/cinn/adt/symbolic_dim_expr.h"
#include <type_traits>

namespace cinn::adt {

namespace {

bool SymbolicDimExprEqualImpl(std::int64_t lhs, std::int64_t rhs) {
  return lhs == rhs;
}

bool SymbolicDimExprEqualImpl(const SymbolicDim& lhs, const SymbolicDim& rhs) {
  return lhs == rhs;
}

bool SymbolicDimExprEqualImpl(const Negtive<SymbolicDimExpr>& lhs,
                              const Negtive<SymbolicDimExpr>& rhs) {
  const auto& [lhs_arg0] = lhs.tuple();
  const auto& [rhs_arg0] = lhs.tuple();
  return lhs_arg0 == rhs_arg0;
}

bool SymbolicDimExprEqualImpl(const Reciprocal<SymbolicDimExpr>& lhs,
                              const Reciprocal<SymbolicDimExpr>& rhs) {
  const auto& [lhs_arg0] = lhs.tuple();
  const auto& [rhs_arg0] = lhs.tuple();
  return lhs_arg0 == rhs_arg0;
}

bool SymbolicDimExprEqualImpl(const Add<SymbolicDimExpr, SymbolicDimExpr>& lhs,
                              const Add<SymbolicDimExpr, SymbolicDimExpr>& rhs) {
  const auto& [lhs_arg0, lhs_arg1] = lhs.tuple();
  const auto& [rhs_arg0, rhs_arg1] = lhs.tuple();
  return lhs_arg0 == rhs_arg0 && lhs_arg1 == rhs_arg1;
}

bool SymbolicDimExprEqualImpl(const Mul<SymbolicDimExpr, SymbolicDimExpr>& lhs,
                              const Mul<SymbolicDimExpr, SymbolicDimExpr>& rhs) {
  const auto& [lhs_arg0, lhs_arg1] = lhs.tuple();
  const auto& [rhs_arg0, rhs_arg1] = lhs.tuple();
  return lhs_arg0 == rhs_arg0 && lhs_arg1 == rhs_arg1;
}

}  // namespace

bool operator==(const SymbolicDimExpr& lhs, const SymbolicDimExpr& rhs) {
  return std::visit(
      [](const auto& lhs, const auto& rhs) {
        if (std::is_same_v<std::decay_t<decltype(lhs)>,
                           std::decay_t<decltype(rhs)>>) {
          return SymbolicDimExprEqualImpl(lhs, rhs);
        } else {
          return false;
        }
      },
      lhs.variant(),
      rhs.variant());
}

}  // namespace cinn::adt
