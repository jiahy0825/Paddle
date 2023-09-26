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

#include "paddle/cinn/adt/print_equations.h"

#include <sstream>
#include <string>

namespace cinn::adt {

std::string ToTxtString(const Iterator& iterator) {
  std::size_t iterator_unique_id = iterator.value().unique_id();
  return "i_" + std::to_string(iterator_unique_id);
}

std::string ToTxtString(const Index& index) {
  std::size_t index_unique_id = index.value().unique_id();
  return "idx_" + std::to_string(index_unique_id);
}

std::string ToTxtString(const Stride& stride) {
  std::size_t stride_unique_id = stride.value().unique_id();
  return "stride_" + std::to_string(stride_unique_id);
}

std::string ToTxtString(const FakeOpPlaceHolder& op) {
  std::size_t op_unique_id = op.value().unique_id();
  return "op_" + std::to_string(op_unique_id);
}

std::string ToTxtString(const List<Index>& index_list) {
  std::string ret;
  ret += "(";

  for (std::size_t idx = 0; idx < index_list->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(index_list.Get(idx));
  }

  ret += ")";
  return ret;
}

std::string ToTxtString(const List<std::optional<Index>>& index_list) {
  std::string ret;
  ret += "(";

  for (std::size_t idx = 0; idx < index_list->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    if (index_list->at(idx).has_value()) {
      ret += ToTxtString(index_list.Get(idx).value());
    }
  }

  ret += ")";
  return ret;
}

std::string ToTxtString(const List<Iterator>& iterator_list) {
  std::string ret;
  ret += "(";
  for (std::size_t idx = 0; idx < iterator_list->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(iterator_list.Get(idx));
  }
  ret += ")";
  return ret;
}

std::string ToTxtString(const List<Stride>& stride_list) {
  std::string ret;
  ret += "(";
  for (std::size_t idx = 0; idx < stride_list->size(); ++idx) {
    if (idx != 0) {
      ret += ", ";
    }
    ret += ToTxtString(stride_list.Get(idx));
  }
  ret += ")";
  return ret;
}

std::string ToTxtString(const tInMsgBox<List<Index>>& in_msg_box_indexes) {
  std::string ret;
  const List<Index>& index_list = in_msg_box_indexes.value();
  ret += ToTxtString(index_list);
  return ret;
}

std::string ToTxtString(const tOutMsgBox<List<Index>>& out_msg_box_indexes) {
  std::string ret;
  const List<Index>& index_list = out_msg_box_indexes.value();
  ret += ToTxtString(index_list);
  return ret;
}

struct ToTxtStringStruct {
  std::string operator()(
      const Identity<tOut<Iterator>, tIn<Iterator>>& id) const {
    std::string ret;
    const auto& [out_iter_tag, in_iter_tag] = id.tuple();
    const Iterator& out_iter = out_iter_tag.value();
    const Iterator& in_iter = in_iter_tag.value();
    ret += ToTxtString(out_iter) + " = " + ToTxtString(in_iter);
    return ret;
  }

  std::string operator()(const Identity<tOut<Index>, tIn<Index>>& id) const {
    std::string ret;
    const auto& [out_index_tag, in_index_tag] = id.tuple();
    const Index& out_index = out_index_tag.value();
    const Index& in_index = in_index_tag.value();
    ret += ToTxtString(out_index) + " = " + ToTxtString(in_index);
    return ret;
  }

  std::string operator()(
      const Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>& dot) const {
    std::string ret;
    const auto& [stride_list, out_index_tag, in_iterator_list_tag] =
        dot.tuple();
    const Index& out_index = out_index_tag.value();
    const List<Iterator>& in_iterator_list = in_iterator_list_tag.value();
    ret += ToTxtString(out_index) + " = Dot(" + ToTxtString(in_iterator_list) +
           ")";
    return ret;
  }

  std::string operator()(
      const UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>& undot)
      const {
    std::string ret;
    const auto& [stride_list, out_iterator_list_tag, in_index_tag] =
        undot.tuple();
    const List<Iterator>& out_iterator_list = out_iterator_list_tag.value();
    const Index& in_index = in_index_tag.value();
    ret += ToTxtString(out_iterator_list) + " = UnDot(" +
           ToTxtString(in_index) + ")";
    return ret;
  }

  std::string operator()(
      const InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                               tOut<OpArgIndexes<std::optional<Index>>>,
                               tIn<OpArgIndexes<Index>>>& box) const {
    std::string ret;
    const auto& [out_op_tag, out_index_list_tag, in_index_list_tag] =
        box.tuple();
    const FakeOpPlaceHolder& op = out_op_tag.value();
    const auto& out_index_tuple = out_index_list_tag.value();
    const auto& in_index_tuple = in_index_list_tag.value();
    const auto& [out_index_list_inbox, out_index_list_outbox] =
        out_index_tuple.tuple();
    const auto& [in_index_list_inbox, in_index_list_outbox] =
        in_index_tuple.tuple();
    ret += ToTxtString(op) + ", ";
    ret += "(" + ToTxtString(out_index_list_inbox.value()) + ", " +
           ToTxtString(out_index_list_outbox.value()) +
           ") = InMsgBox2OutMsgBox(";
    ret += ToTxtString(in_index_list_inbox.value()) + ", " +
           ToTxtString(in_index_list_outbox.value()) + ")";
    return ret;
  }
};

struct ToDotStringStruct {
  std::string operator()(
      const Identity<tOut<Iterator>, tIn<Iterator>>& id) const {
    std::string ret;
    const auto& [out_iter_tag, in_iter_tag] = id.tuple();
    const Iterator& out_iter = out_iter_tag.value();
    const Iterator& in_iter = in_iter_tag.value();
    ret += ToTxtString(out_iter) + " = " + ToTxtString(in_iter);
    return ret;
  }

  std::string operator()(const Identity<tOut<Index>, tIn<Index>>& id) const {
    std::string ret;
    const auto& [out_index_tag, in_index_tag] = id.tuple();
    const Index& out_index = out_index_tag.value();
    const Index& in_index = in_index_tag.value();
    ret += ToTxtString(out_index) + " = " + ToTxtString(in_index);
    return ret;
  }

  std::string operator()(
      const Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>& dot) const {
    std::string ret;
    const auto& [stride_list, out_index_tag, in_iterator_list_tag] =
        dot.tuple();
    const Index& out_index = out_index_tag.value();
    const List<Iterator>& in_iterator_list = in_iterator_list_tag.value();
    ret += ToTxtString(out_index) + " = Dot(" + ToTxtString(in_iterator_list) +
           ")";
    return ret;
  }

  std::string operator()(
      const UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>& undot)
      const {
    std::string ret;
    const auto& [stride_list, out_iterator_list_tag, in_index_tag] =
        undot.tuple();
    const List<Iterator>& out_iterator_list = out_iterator_list_tag.value();
    const Index& in_index = in_index_tag.value();
    ret += ToTxtString(out_iterator_list) + " = UnDot(" +
           ToTxtString(in_index) + ")";
    return ret;
  }

  std::string operator()(
      const InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                               tOut<OpArgIndexes<std::optional<Index>>>,
                               tIn<OpArgIndexes<Index>>>& box) const {
    std::string ret;
    const auto& [out_op_tag, out_index_list_tag, in_index_list_tag] =
        box.tuple();
    const FakeOpPlaceHolder& op = out_op_tag.value();
    const auto& out_index_tuple = out_index_list_tag.value();
    const auto& in_index_tuple = in_index_list_tag.value();
    const auto& [out_index_list_inbox, out_index_list_outbox] =
        out_index_tuple.tuple();
    const auto& [in_index_list_inbox, in_index_list_outbox] =
        in_index_tuple.tuple();
    ret += ToTxtString(op) + ", ";
    ret += "(" + ToTxtString(out_index_list_inbox.value()) + ", " +
           ToTxtString(out_index_list_outbox.value()) +
           ") = InMsgBox2OutMsgBox(";
    ret += ToTxtString(in_index_list_inbox.value()) + ", " +
           ToTxtString(in_index_list_outbox.value()) + ")";
    return ret;
  }
};

std::string ToTxtString(const Equation& equation) {
  return std::visit(ToTxtStringStruct{}, equation.variant());
}

std::string ToTxtString(const Equations& equations,
                        const std::string& separator) {
  std::stringstream ret;
  std::size_t count = 0;

  for (const auto& equation : *equations) {
    if (count++ > 0) {
      ret << separator;
    }
    ret << &equation << ": ";
    ret << ToTxtString(equation);
  }
  return ret.str();
}

void PrintOpStmtsEquations(const List<OpStmt>& op_stmts,
                           const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
  for (const auto& op_stmt : *op_stmts) {
    const auto& ctx = EquationCtx4OpStmt(op_stmt);
    ctx->Print();
  }
}

void PrintIndexVector(const std::vector<Index>& indexes) {
  VLOG(3) << "tensor_indexes.size():" << indexes.size();
  for (const auto& index : indexes) {
    VLOG(3) << ToTxtString(index);
  }
}

}  // namespace cinn::adt
