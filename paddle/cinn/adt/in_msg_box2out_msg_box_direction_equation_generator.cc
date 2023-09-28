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

#include "paddle/cinn/adt/in_msg_box2out_msg_box_direction_equation_generator.h"

#include "paddle/cinn/adt/naive_equation_function_constants_provider.h"

namespace cinn::adt {

namespace {

using EquationCtx4OpStmtT =
    std::function<std::shared_ptr<config::NaiveOpEquationContext>(
        const OpStmt&)>;

template <typename DoEachT>
void VisitEachOpStmt(const List<OpStmt>& op_stmts,
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
    const List<Index>& out_msg_box_in_indexes,
    const List<Index>& out_msg_box_out_indexes) {
  List<std::optional<Index>> opt_out_msg_box_out_indexes{};
  for (const auto& out_index : *out_msg_box_out_indexes) {
    opt_out_msg_box_out_indexes->emplace_back(out_index);
  }
  return OpArgIndexes<std::optional<Index>>{out_msg_box_in_indexes,
                                            opt_out_msg_box_out_indexes};
}

OpArgIndexes<Index> MakeInMsgBoxOpArgIndexes(
    const List<Index>& in_msg_box_in_indexes,
    const List<Index>& in_msg_box_out_indexes) {
  return OpArgIndexes<Index>{in_msg_box_in_indexes, in_msg_box_out_indexes};
}

Equation GenerateInMsgBox2OutMsgBoxEquation(
    const OpStmt& op_stmt,
    const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
  List<Index> in_msg_box_in_indexes = ctx->in_indexes();
  List<Index> in_msg_box_out_indexes = ctx->out_indexes();
  List<Index> out_msg_box_in_indexes =
      MakeArgIndexes(ctx->GetInTensorsRanks().size());
  List<Index> out_msg_box_out_indexes =
      MakeArgIndexes(ctx->GetOutTensorsRanks().size());

  return InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                            tOut<OpArgIndexes<std::optional<Index>>>,
                            tIn<OpArgIndexes<Index>>>{
      ctx->fake_op_placeholder(),
      MakeOutMsgBoxOpArgIndexes(out_msg_box_in_indexes,
                                out_msg_box_out_indexes),
      MakeInMsgBoxOpArgIndexes(in_msg_box_in_indexes, in_msg_box_out_indexes)};
}

template <typename DoEachT>
void VisitEachInMsgOutMsgPair(
    const OpArgIndexes<Index>& in_msg_box,
    const OpArgIndexes<std::optional<Index>>& out_msg_box,
    const DoEachT& DoEach) {
  const auto& [in_msg_box_in_indexes, in_msg_box_out_indexes] =
      in_msg_box.tuple();
  const auto& [out_msg_box_in_indexes, out_msg_box_out_indexes] =
      out_msg_box.tuple();
  CHECK_EQ(in_msg_box_in_indexes.value()->size(),
           out_msg_box_in_indexes.value()->size());
  for (std::size_t i = 0; i < in_msg_box_in_indexes.value()->size(); ++i) {
    DoEach(in_msg_box_in_indexes.value()->at(i),
           out_msg_box_in_indexes.value()->at(i));
  }
  CHECK_EQ(in_msg_box_out_indexes.value()->size(),
           out_msg_box_out_indexes.value()->size());
  for (std::size_t i = 0; i < in_msg_box_out_indexes.value()->size(); ++i) {
    CHECK(out_msg_box_out_indexes.value()->at(i).has_value());
    DoEach(in_msg_box_out_indexes.value()->at(i),
           out_msg_box_out_indexes.value()->at(i).value());
  }
}

bool LocalEquationsSolvable(
    const GraphView& graph_view,
    const Index& anchor_index,
    const FakeOpPlaceHolder& fake_op_placeholder,
    const std::shared_ptr<const EquationFunctionConstantsProvider>&
        constants_provider) {
  const auto& init_var2value = MakeAnchorIndex2Ok(anchor_index);
  IndexExprInferContext ctx{init_var2value, constants_provider};
  bool has_no_conflict_value =
      TrySolveEquations(graph_view, anchor_index, &ctx).value();
  return has_no_conflict_value && ctx.HasValue(fake_op_placeholder);
}

using InBox2OutBox =
    InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                       tOut<OpArgIndexes<std::optional<Index>>>,
                       tIn<OpArgIndexes<Index>>>;

List<std::optional<Index>> GetMaskedOutIndexes(
    const List<Index>& in_box_out_indexes,
    const List<std::optional<Index>>& out_box_out_indexes,
    const std::vector<Index>& erased_in_msg_box_out_tensor_indexes) {
  List<std::optional<Index>> ret{};
  const auto& erased = erased_in_msg_box_out_tensor_indexes;
  CHECK_EQ(in_box_out_indexes->size(), out_box_out_indexes->size());
  for (std::size_t i = 0; i < in_box_out_indexes->size(); ++i) {
    const auto& in_box_index = in_box_out_indexes->at(i);
    if (std::find(erased.begin(), erased.end(), in_box_index) == erased.end()) {
      ret->emplace_back(out_box_out_indexes->at(i));
    } else {
      ret->emplace_back(std::nullopt);
    }
  }
  return ret;
}

Equation EraseIndexes(
    const Equation& equation,
    const std::vector<Index>& erased_in_msg_box_out_tensor_indexes) {
  const auto& in_msg_box2out_msg_box = equation.Get<InBox2OutBox>();
  const auto& [op_placeholder, out_box_indexes, in_box_indexes] =
      in_msg_box2out_msg_box.tuple();

  const auto& [_, in_box_out_indexes] = in_box_indexes.value().tuple();
  const auto& [out_box_in_indexes, out_box_out_indexes] =
      out_box_indexes.value().tuple();
  const auto& masked_out_indexes =
      GetMaskedOutIndexes(in_box_out_indexes.value(),
                          out_box_out_indexes.value(),
                          erased_in_msg_box_out_tensor_indexes);

  OpArgIndexes<std::optional<Index>> out_box{out_box_in_indexes,
                                             masked_out_indexes};

  Equation ret_equation = InBox2OutBox{op_placeholder, out_box, in_box_indexes};

  return ret_equation;
}

void NaiveOpEquationContext::EraseOutMsgBoxIndexes(
    const std::vector<Index>& erased_output_tensor_indexes,
    const Equation& equation) {
  return EraseIndexes(equation, erased_output_tensor_indexes);
}

}  // namespace

void InMsgBox2OutMsgBoxDirectionEquationGenerator::Init() {
  VisitEachOpStmt(
      this->op_stmts_,
      this->EquationCtx4OpStmt_,
      [&](std::size_t idx,
          const OpStmt& op_stmt,
          const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
        Equation equation = GenerateInMsgBox2OutMsgBoxEquation(op_stmt, ctx);
        const auto& [fake_op_placeholder, out_msg_box, in_msg_box] =
            equation
                .Get<
                    InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                                       tOut<OpArgIndexes<std::optional<Index>>>,
                                       tIn<OpArgIndexes<Index>>>>()
                .tuple();
        this->fake_op_placeholders_->emplace_back(fake_op_placeholder.value());
        this->equations_->emplace_back(equation);
        VisitEachInMsgOutMsgPair(
            in_msg_box.value(),
            out_msg_box.value(),
            [&](const Index& in_index, const Index& out_index) {
              CHECK(this->in_msg_box_index2out_msg_box_index_
                        .emplace(in_index, out_index)
                        .second);
            });
      });
}

std::vector<Index> InMsgBox2OutMsgBoxDirectionEquationGenerator::
    GenerateWriteBroadcastTensorIndexs(
        const std::shared_ptr<config::NaiveOpEquationContext>& ctx,
        const std::shared_ptr<const EquationFunctionConstantsProvider>&
            constants_provider) {
  const auto& eqaution_graph_view =
      Graph::New(ctx->equations())->GetGraphView();
  GraphView graph_view = eqaution_graph_view.Merge(
      Graph::New(this->GetDirectionEquations())->GetGraphView());
  std::vector<Index> ret{};
  const auto& fake_op_placeholder = ctx->fake_op_placeholder();
  ctx->VisitEachOutputTensorIndex([&](const auto& out_index) {
    if (!LocalEquationsSolvable(
            graph_view, out_index, fake_op_placeholder, constants_provider)) {
      ret.emplace_back(out_index);
    }
  });
  return ret;
}

void InMsgBox2OutMsgBoxDirectionEquationGenerator::
    EraseWriteBroadcastOutMsgBoxes() {
  std::shared_ptr<const EquationFunctionConstantsProvider> constants_provider{
      new NaiveEquationFunctionConstantsProvider{this->op_stmts_,
                                                 this->EquationCtx4OpStmt_}};
  VisitEachOpStmt(
      this->op_stmts_,
      this->EquationCtx4OpStmt_,
      [&](std::size_t idx,
          const OpStmt& op_stmt,
          const std::shared_ptr<config::NaiveOpEquationContext>& ctx) {
        const auto& truncated_output_tensor_idxes =
            GenerateWriteBroadcastTensorIndexs(ctx, constants_provider);
        this->equstions_->at(idx) = EraseWriteBroadcastOutMsgBox(
            truncated_output_tensor_idxes, this->equations_->at(i));
      });
}

}  // namespace cinn::adt
