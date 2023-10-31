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

template <template <typename, typename> class Op>
bool SymbolicDimExprEqual(const Op<SymbolicDimExpr, SymbolicDimExpr>& lhs,
                          const Op<SymbolicDimExpr, SymbolicDimExpr>& rhs) {
  const auto& [lhs_arg0, lhs_arg1] = lhs.tuple();
  const auto& [rhs_arg0, rhs_arg1] = lhs.tuple();
  return lhs_arg0 == rhs_arg0 && lhs_arg1 == rhs_arg1;
}

#define SPECIALIZE_SYMBOLIC_DIM_EXPR(Op)                 \
  bool SymbolicDimExprEqualImpl(                         \
      const Op<SymbolicDimExpr, SymbolicDimExpr>& lhs,   \
      const Op<SymbolicDimExpr, SymbolicDimExpr>& rhs) { \
    return SymbolicDimExprEqual<Op>(lhs, rhs);           \
  }
SPECIALIZE_SYMBOLIC_DIM_EXPR(Add);
SPECIALIZE_SYMBOLIC_DIM_EXPR(Sub);
SPECIALIZE_SYMBOLIC_DIM_EXPR(Mul);
SPECIALIZE_SYMBOLIC_DIM_EXPR(Div);
SPECIALIZE_SYMBOLIC_DIM_EXPR(BroadcastedDim);
#undef SPECIALIZE_SYMBOLIC_DIM_EXPR;

bool SymbolicDimExprEqualImpl(const OneOf<SymbolicDimExpr>& lhs,
                              const OneOf<SymbolicDimExpr>& rhs) {
  return lhs.list() == rhs.list();
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
