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

template <typename T>
class OneOf final {
 public:
  explicit OneOf(const List<T>& list) : list_(list) {}

  const List<T>& list() const { return list_; }

 private:
  List<T> list_;
};

// SymbolicDimExpr = std::int64_t
//                 | SymbolicDim
//                 | Add SymbolicDimExpr SymbolicDimExpr
//                 | Sub SymbolicDimExpr SymbolicDimExpr
//                 | Mul SymbolicDimExpr SymbolicDimExpr
//                 | Div SymbolicDimExpr SymbolicDimExpr
//                 | BroadcastedDim SymbolicDimExpr SymbolicDimExpr
//                 | OneOf SymbolicDimExpr
DEFINE_ADT_UNION(SymbolicDimExpr,
                 std::int64_t,
                 SymbolicDim,
                 Add<SymbolicDimExpr, SymbolicDimExpr>,
                 Sub<SymbolicDimExpr, SymbolicDimExpr>,
                 Mul<SymbolicDimExpr, SymbolicDimExpr>,
                 Div<SymbolicDimExpr, SymbolicDimExpr>,
                 BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>,
                 OneOf<SymbolicDimExpr>);

}  // namespace cinn::adt
