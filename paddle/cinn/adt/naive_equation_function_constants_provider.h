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
    const auto& symbolic_dim_expr = GetDimSize(dim);
    if (symbolic_dim_expr.Has<std::int64_t>()) {
      return symbolic_dim_expr.Get<std::int64_t>();
    }
    return std::nullopt;
  }

  SymbolicDimExpr GetDimSize(const EquationDim& dim) const override {
    const auto& iter = equation_dim2symbolic_dim_expr_.find(dim);
    CHECK(iter != equation_dim2symbolic_dim_expr_.end());
    return iter->second;
  }

  bool AddDim(const EquationDim& dim,
              const SymbolicDimExpr& symbolic_dim_expr) override {
    return equation_dim2symbolic_dim_expr_.emplace(dim, symbolic_dim_expr)
        .second;
  }

 private:
  void Init(const List<OpStmt>& op_stmts,
            const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
    const auto& GetSymbolicDimExpr = [&](const auto& ctx,
                                         bool is_out,
                                         std::size_t arg_idx,
                                         std::size_t axis) -> SymbolicDimExpr {
      std::optional<std::int64_t> static_dim_size =
          ctx->GetStaticDimSize(is_out, arg_idx, axis);
      if (static_dim_size.has_value()) {
        return SymbolicDimExpr{static_dim_size.value()};
      }
      std::optional<SymbolicDimExpr> symbolic_dim_expr =
          ctx->GetSymbolicDimSize(is_out, arg_idx, axis);
      if (symbolic_dim_expr.has_value()) {
        return symbolic_dim_expr.value();
      }
      LOG(FATAL) << "Dead code, cannot get StaticDim or SymbolicDim";
    };

    for (const auto& op_stmt : *op_stmts) {
      const auto& ctx = EquationCtx4OpStmt(op_stmt);
      ctx->VisitEachArgPos([&](bool is_out,
                               std::size_t arg_idx,
                               std::size_t axis) {
        const EquationDim& dim = ctx->GetDim(is_out, arg_idx, axis);
        const SymbolicDimExpr& symbolic_dim_expr =
            GetSymbolicDimExpr(ctx, is_out, arg_idx, axis);
        CHECK(equation_dim2symbolic_dim_expr_.emplace(dim, symbolic_dim_expr)
                  .second);
      });
    }
  }

  std::unordered_map<EquationDim, const SymbolicDimExpr>
      equation_dim2symbolic_dim_expr_;
};

}  // namespace cinn::adt
