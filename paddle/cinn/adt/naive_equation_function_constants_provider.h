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

#include <unordered_map>

#include "paddle/cinn/adt/equation_function_constants_provider.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"

namespace cinn::adt {

class NaiveEquationFunctionConstantsProvider final
    : public EquationFunctionConstantsProvider {
 public:
  using EquationCtx4OpStmtT =
      std::function<std::shared_ptr<config::NaiveOpEquationContext>(
          const OpStmt&)>;

  NaiveEquationFunctionConstantsProvider(
      const List<OpStmt>& op_stmts,
      const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
    Init(op_stmts, EquationCtx4OpStmt);
  }

  NaiveEquationFunctionConstantsProvider(
      const NaiveEquationFunctionConstantsProvider&) = delete;
  NaiveEquationFunctionConstantsProvider(
      NaiveEquationFunctionConstantsProvider&&) = delete;

  std::optional<std::int64_t> GetStaticDimSize(
      const EquationDim& dim) const override {
    const auto& iter = dim2constant_.find(dim);
    CHECK(iter != dim2constant_.end());
    if (iter->second.Has<std::int64_t>()) {
      return iter->second.Get<std::int64_t>();
    } else {
      return std::nullopt;
    }
    LOG(FATAL) << "Dead code";
  }

  std::optional<SymbolicDim> GetSymbolicDimSize(
      const EquationDim& dim) const override {
    const auto& iter = dim2constant_.find(dim);
    CHECK(iter != dim2constant_.end());
    if (iter->second.Has<SymbolicDim>()) {
      return iter->second.Get<SymbolicDim>();
    } else {
      return std::nullopt;
    }
    LOG(FATAL) << "Dead code";
  }

  Constant GetDimSize(const EquationDim& dim) const override {
    const auto& iter = dim2constant_.find(dim);
    CHECK(iter != dim2constant_.end());
    CHECK(iter->second.Has<std::int64_t>() || iter->second.Has<SymbolicDim>());
    return iter->second;
  }

  bool AddDim(const EquationDim& dim, const Constant& dim_value) override {
    return dim2constant_.emplace(dim, dim_value).second;
  }

 private:
  void Init(const List<OpStmt>& op_stmts,
            const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
    const auto& GetConstantValue = [&](const auto& ctx,
                                       bool is_out,
                                       std::size_t arg_idx,
                                       std::size_t axis) -> Constant {
      std::optional<std::int64_t> static_dim_size =
          ctx->GetStaticDimSize(is_out, arg_idx, axis);
      if (static_dim_size.has_value()) {
        return static_dim_size.value();
      }
      std::optional<SymbolicDim> symbolic_dim_size =
          ctx->GetSymbolicDimSize(is_out, arg_idx, axis);
      if (symbolic_dim_size.has_value()) {
        return symbolic_dim_size.value();
      }
      LOG(FATAL) << "Dead code, cannot get StaticDim or SymbolicDim";
    };

    for (const auto& op_stmt : *op_stmts) {
      const auto& ctx = EquationCtx4OpStmt(op_stmt);
      ctx->VisitEachArgPos(
          [&](bool is_out, std::size_t arg_idx, std::size_t axis) {
            const EquationDim& dim = ctx->GetDim(is_out, arg_idx, axis);
            const Constant& constant =
                GetConstantValue(ctx, is_out, arg_idx, axis);
            CHECK(dim2constant_.emplace(dim, constant).second);
          });
    }
  }

  std::unordered_map<EquationDim, const Constant> dim2constant_;
};

}  // namespace cinn::adt
