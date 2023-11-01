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

#include "paddle/cinn/adt/print_symbolic_dim_expr.h"

namespace cinn::adt {

std::string ToTxtString(const SymbolicDimExpr& loop_size);

namespace {

std::string ToTxtStringImpl(std::int64_t symbolic_dim_expr) {
  return std::to_string(symbolic_dim_expr);
}

std::string ToTxtStringImpl(const SymbolicDim& symbolic_dim_expr) {
  return std::string("sym_") +
         std::to_string(symbolic_dim_expr.value().unique_id());
}

std::string ToTxtStringImpl(
    const Negative<SymbolicDimExpr>& symbolic_dim_expr) {
  const auto& [item] = symbolic_dim_expr.tuple();
  return std::string("-") + ToTxtString(item);
}

std::string ToTxtStringImpl(
    const Reciprocal<SymbolicDimExpr>& symbolic_dim_expr) {
  const auto& [item] = symbolic_dim_expr.tuple();
  return std::string("1 / (") + ToTxtString(item) + std::string(")");
}

std::string ToTxtStringImpl(
    const Add<SymbolicDimExpr, SymbolicDimExpr>& symbolic_dim_expr) {
  const auto& [lhs, rhs] = symbolic_dim_expr.tuple();
  return std::string("(") + ToTxtString(lhs) + std::string(" + ") +
         ToTxtString(rhs) + std::string(")");
}

std::string ToTxtStringImpl(
    const Mul<SymbolicDimExpr, SymbolicDimExpr>& symbolic_dim_expr) {
  const auto& [lhs, rhs] = symbolic_dim_expr.tuple();
  return std::string("(") + ToTxtString(lhs) + std::string(" * ") +
         ToTxtString(rhs) + std::string(")");
}

std::string ToTxtStringImpl(
    const BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>& symbolic_dim_expr) {
  const auto& [lhs, rhs] = symbolic_dim_expr.tuple();
  return std::string("BD(") + ToTxtString(lhs) + std::string(", ") +
         ToTxtString(rhs) + std::string(")");
}

}  // namespace

std::string ToTxtString(const SymbolicDimExpr& loop_size) {
  return std::visit([&](const auto& impl) { return ToTxtStringImpl(impl); },
                    loop_size.variant());
}

std::string ToTxtString(const List<SymbolicDimExpr>& loop_sizes) {
  std::string ret;
  ret += "[";
  for (std::size_t idx = 0; idx < loop_sizes->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(loop_sizes.Get(idx));
  }
  ret += "]";
  return ret;
}

}  // namespace cinn::adt
