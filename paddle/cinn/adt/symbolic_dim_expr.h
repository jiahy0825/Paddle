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

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/arithmetic.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/symbolic_dim.h"

namespace cinn::adt {

DEFINE_ADT_BINARY(BroadcastedDim);

// SymbolicDimExpr = std::int64_t
//                 | SymbolicDim
//                 | Negative SymbolicDimExpr
//                 | Reciprocal SymbolicDimExpr
//                 | Add SymbolicDimExpr SymbolicDimExpr
//                 | Mul SymbolicDimExpr SymbolicDimExpr
//                 | BroadcastedDim SymbolicDimExpr SymbolicDimExpr
DEFINE_ADT_UNION(SymbolicDimExpr,
                 std::int64_t,
                 SymbolicDim,
                 Negative<SymbolicDimExpr>,
                 Reciprocal<SymbolicDimExpr>,
                 Add<SymbolicDimExpr, SymbolicDimExpr>,
                 Mul<SymbolicDimExpr, SymbolicDimExpr>,
                 BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>);

inline SymbolicDimExpr operator+(const SymbolicDimExpr& lhs,
                                 const SymbolicDimExpr& rhs) {
  return Add<SymbolicDimExpr, SymbolicDimExpr>{lhs, rhs};
}

inline SymbolicDimExpr operator-(const SymbolicDimExpr& lhs,
                                 const SymbolicDimExpr& rhs) {
  return Add<SymbolicDimExpr, SymbolicDimExpr>{lhs,
                                               Negative<SymbolicDimExpr>{rhs}};
}

inline SymbolicDimExpr operator*(const SymbolicDimExpr& lhs,
                                 const SymbolicDimExpr& rhs) {
  return Mul<SymbolicDimExpr, SymbolicDimExpr>{lhs, rhs};
}

inline SymbolicDimExpr operator/(const SymbolicDimExpr& lhs,
                                 const SymbolicDimExpr& rhs) {
  return Mul<SymbolicDimExpr, SymbolicDimExpr>{
      lhs, Reciprocal<SymbolicDimExpr>{rhs}};
}

inline SymbolicDimExpr MakeBroadcastedDim(const SymbolicDimExpr& lhs,
                                          const SymbolicDimExpr& rhs) {
  return BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>{lhs, rhs};
}

bool operator==(const SymbolicDimExpr& lhs, const SymbolicDimExpr& rhs);

inline bool operator!=(const SymbolicDimExpr& lhs, const SymbolicDimExpr& rhs) {
  return !(lhs == rhs);
}

}  // namespace cinn::adt
