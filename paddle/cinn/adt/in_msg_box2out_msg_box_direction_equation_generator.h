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

#include "paddle/cinn/adt/direction_equation_generator.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"
#include "paddle/cinn/adt/equation_function.h"

namespace cinn::adt {

class EquationFunctionConstantsProvider;
namespace config {
class NaiveOpEquationContext;
}

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
      : op_stmts_(op_stmts), EquationCtx4OpStmt_(EquationCtx4OpStmt) {
    Init();
  }

  Equations GetDirectionEquations() const override { return equations_; }

  std::function<const OpStmt*(const FakeOpPlaceHolder&)> MakeGetterOpStmt4OpPlaceHolder() const override;

  std::optional<Index> OutMsgBoxIndex4InMsgBoxIndex(
      const Index& index) const override {
    const auto& iter = in_msg_box_index2out_msg_box_index_.find(index);
    if (iter == in_msg_box_index2out_msg_box_index_.end()) {
      return std::nullopt;
    } else {
      return iter->second;
    }
  }

  void EraseWriteBroadcastOutMsgBoxes();

 private:
  void InitInMsgBoxIndex2OutMsgBoxIndex();
  void InitEquations();

  void Init() {
    InitInMsgBoxIndex2OutMsgBoxIndex();
    InitEquations();
  }

  std::vector<Index> GenerateWriteBroadcastTensorIndexs(
      const std::shared_ptr<config::NaiveOpEquationContext>& ctx,
      const std::shared_ptr<const EquationFunctionConstantsProvider>&
          constants_provider);

  List<OpStmt> op_stmts_;
  EquationCtx4OpStmtT EquationCtx4OpStmt_;
  Equations equations_;
  List<FakeOpPlaceHolder> fake_op_placeholders_;
  std::unordered_map<Index, Index> in_msg_box_index2out_msg_box_index_;
};

}  // namespace cinn::adt
