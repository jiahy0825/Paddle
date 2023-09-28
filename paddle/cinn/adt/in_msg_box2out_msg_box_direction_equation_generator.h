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

#include "paddle/cinn/adt/direction_equation_generator.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"

namespace cinn::adt {

class InMsgBox2OutMsgBoxDirectionEquationGenerator final
    : public DirectionEquationGenerator {
 public:
  using EquationCtx4OpStmtT =
      std::function<std::shared_ptr<config::NaiveOpEquationContext>(
          const OpStmt&)>;

  InMsgBox2OutMsgBoxDirectionEquationGenerator(
      const InMsgBox2OutMsgBoxDirectionEquationGenerator&) = delete;
  InMsgBox2OutMsgBoxDirectionEquationGenerator(
      InMsgBox2OutMsgBoxDirectionEquationGenerator&&) = delete;

  InMsgBox2OutMsgBoxDirectionEquationGenerator(
      const List<OpStmt>& op_stmts,
      const EquationCtx4OpStmtT& EquationCtx4OpStmt)
      : op_stmts_(op_stmts), EquationCtx4OpStmt_(EquationCtx4OpStmt) {}

  Equations generate_direction_equations() const override {}

  void EraseWriteBroadcastOutMsgBoxes();

 private:
  List<OpStmt> op_stmts_;
  EquationCtx4OpStmtT EquationCtx4OpStmt_;
};

}  // namespace cinn::adt
