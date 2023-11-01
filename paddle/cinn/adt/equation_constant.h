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
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/symbolic_dim.h"
#include "paddle/cinn/adt/symbolic_dim_expr.h"
#include "paddle/cinn/adt/tags.h"
#include "paddle/cinn/adt/unique_id.h"

namespace cinn::adt {

// EquationDim = tEquationDim UniqueId
using EquationDim = tEquationDim<UniqueId>;
// DimTuple = [EquationDim]
using DimTuple = List<EquationDim>;

DEFINE_ADT_UNION(
    Constant, std::int64_t, EquationDim, List<Constant>, SymbolicDimExpr);

OVERLOAD_OPERATOR_EQ_NE(Constant, UnionEqual);

inline std::size_t GetHashValue(const Constant& c);

inline std::size_t GetHashValueImpl(const std::int64_t& c) { return c; }
inline std::size_t GetHashValueImpl(const EquationDim& c) {
  return c.value().unique_id();
}
inline std::size_t GetHashValueImpl(const SymbolicDimExpr& c) {
  return GetHashValue(c);
}
inline std::size_t GetHashValueImpl(const List<Constant>& c) {
  std::size_t ret = 0;
  for (const auto& c_item : *c) {
    ret = hash_combine(ret, GetHashValue(c_item));
  }
  return ret;
}

OVERRIDE_UNION_GET_HASH_VALUE(Constant);

}  // namespace cinn::adt

namespace std {

template <>
struct hash<::cinn::adt::EquationDim> final {
  std::size_t operator()(const ::cinn::adt::EquationDim& dim) const {
    return dim.value().unique_id();
  }
};

template <>
struct hash<cinn::adt::Constant> {
  std::size_t operator()(const cinn::adt::Constant& constant) const {
    return GetHashValue(constant);
  }
};

}  // namespace std
