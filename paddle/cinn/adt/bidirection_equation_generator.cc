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

#include "paddle/cinn/adt/bidirection_equation_generator.h"

#include "paddle/cinn/adt/equation_graph.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/naive_equation_function_constants_provider.h"

namespace cinn::adt {

namespace {

using EquationCtx4OpStmtT =
    std::function<std::shared_ptr<config::NaiveOpEquationContext>(
        const OpStmt&)>;

template <
    typename DoEachT /*: void(&)(std::size_t, OpStmt, OpEquationContext)*/>
void VisitEachOpStmtAndEquationCtx(
    const List<OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const DoEachT& DoEach) {
  for (std::size_t i = 0; i < op_stmts->size(); ++i) {
    const auto& ctx = EquationCtx4OpStmt(op_stmts->at(i));
    DoEach(i, op_stmts->at(i), ctx);
  }
}

List<Index> MakeArgIndexes(std::size_t num_args) {
  List<Index> ret{};
  for (std::size_t i = 0; i < num_args; ++i) {
    Index index{UniqueId::New()};
    ret->emplace_back(index);
  }
  return ret;
}

OpArgIndexes<std::optional<Index>> MakeOutMsgBoxOpArgIndexes(
    const List<std::optional<Index>>& opt_out_msg_box_in_indexes,
    const List<std::optional<Index>>& opt_out_msg_box_out_indexes) {
  List<Index> out_msg_box_in_indexes{};
  for (const auto& out_msg_box_in_index : *opt_out_msg_box_in_indexes) {
    CHECK(out_msg_box_in_index.has_value());
    out_msg_box_in_indexes->emplace_back(out_msg_box_in_index.value());
  }
  return OpArgIndexes<std::optional<Index>>{out_msg_box_in_indexes,
                                            opt_out_msg_box_out_indexes};
}

OpArgIndexes<Index> MakeInMsgBoxOpArgIndexes(
    const List<Index>& in_msg_box_in_indexes,
    const List<Index>& in_msg_box_out_indexes) {
  return OpArgIndexes<Index>{in_msg_box_in_indexes, in_msg_box_out_indexes};
}

template <typename DoEachT>
void VisitEachInMsgOutMsgPair(const List<Index>& in_msg_box,
                              const List<Index>& out_msg_box,
                              const DoEachT& DoEach) {
  CHECK_EQ(in_msg_box->size(), out_msg_box->size());
  for (std::size_t i = 0; i < in_msg_box->size(); ++i) {
    DoEach(in_msg_box->at(i), out_msg_box->at(i));
  }
}

List<std::optional<Index>> GetOutMsgBoxIndexes(
    const List<Index>& in_indexes,
    const BidirectionEquationGenerator& generator) {
  List<std::optional<Index>> ret{};
  for (const auto& index : *in_indexes) {
    ret->emplace_back(generator.OutMsgBoxIndex4InMsgBoxIndex(index));
  }
  return ret;
}

using InBox2OutBox =
    InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                       tOut<OpArgIndexes<std::optional<Index>>>,
                       tIn<OpArgIndexes<Index>>>;

}  // namespace

void BidirectionEquationGenerator::InitInMsgBoxIndex2OutMsgBoxIndex() {
  const auto& InitEachOpInMsgBoxIndex2OutMsgBoxIndex =
      [&](const std::shared_ptr<config::NaiveOpEquationContext>& ctx,
          bool is_output) {
        List<Index> in_msg_box_indexes =
            is_output ? ctx->out_indexes() : ctx->in_indexes();
        std::size_t out_msg_box_index_size =
            is_output ? ctx->GetOutTensorsRanks().size()
                      : ctx->GetInTensorsRanks().size();
        List<Index> out_msg_box_indexes =
            MakeArgIndexes(out_msg_box_index_size);
        VisitEachInMsgOutMsgPair(
            in_msg_box_indexes,
            out_msg_box_indexes,
            [&](const Index& in_index, const Index& out_index) {
              CHECK(this->in_msg_box_index2out_msg_box_index_
                        .emplace(in_index, out_index)
                        .second);
            });
      };

  VisitEachOpStmtAndEquationCtx(
      this->op_stmts_,
      this->EquationCtx4OpStmt_,
      [&](std::size_t idx,
          const OpStmt& op_stmt,
          const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
        InitEachOpInMsgBoxIndex2OutMsgBoxIndex(ctx, /*is_output=*/false);
        InitEachOpInMsgBoxIndex2OutMsgBoxIndex(ctx, /*is_output=*/true);
      });
}

void BidirectionEquationGenerator::InitEquations() {
  VisitEachOpStmtAndEquationCtx(
      this->op_stmts_,
      this->EquationCtx4OpStmt_,
      [&](std::size_t idx,
          const OpStmt& op_stmt,
          const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
        List<Index> in_msg_box_in_indexes = ctx->in_indexes();
        List<Index> in_msg_box_out_indexes = ctx->out_indexes();
        List<std::optional<Index>> out_msg_box_in_indexes =
            GetOutMsgBoxIndexes(in_msg_box_in_indexes, *this);
        List<std::optional<Index>> out_msg_box_out_indexes =
            GetOutMsgBoxIndexes(in_msg_box_out_indexes, *this);

        Equation equation =
            InBox2OutBox{ctx->fake_op_placeholder(),
                         MakeOutMsgBoxOpArgIndexes(out_msg_box_in_indexes,
                                                   out_msg_box_out_indexes),
                         MakeInMsgBoxOpArgIndexes(in_msg_box_in_indexes,
                                                  in_msg_box_out_indexes)};

        this->fake_op_placeholders_->emplace_back(ctx->fake_op_placeholder());
        this->equations_->emplace_back(equation);
      });
}

std::function<const OpStmt*(const FakeOpPlaceHolder&)>
BidirectionEquationGenerator::MakeGetterOpStmt4OpPlaceHolder() const {
  using FakeOpPlaceHolder2OpStmt =
      std::unordered_map<FakeOpPlaceHolder, OpStmt>;
  const auto& fake_op_placeholder2op_stmt =
      std::make_shared<FakeOpPlaceHolder2OpStmt>();

  for (std::size_t i = 0; i < fake_op_placeholders_->size(); ++i) {
    CHECK(fake_op_placeholder2op_stmt
              ->emplace(fake_op_placeholders_->at(i), op_stmts_->at(i))
              .second);
  }

  return [fake_op_placeholder2op_stmt](
             const FakeOpPlaceHolder& fake_op_placeholder) {
    return &fake_op_placeholder2op_stmt->at(fake_op_placeholder);
  };
}

}  // namespace cinn::adt
