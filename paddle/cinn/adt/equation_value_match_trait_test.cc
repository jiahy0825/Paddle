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

#include "paddle/cinn/adt/equation_value_match_trait.h"
#include "gtest/gtest.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/match.h"

namespace cinn::adt::test {

TEST(Match, Union) {
  using Pattern = Union<IndexUnDotValue<Value, List<SymbolicDimExpr>>, IndexDotValue<Value, List<SymbolicDimExpr>>>;
  {
    Value expr =
        IndexUnDotValue<Value, List<SymbolicDimExpr>>{Value{Ok()}, List<SymbolicDimExpr>{std::int64_t(1)}};

    bool ret = cinn::adt::Match<Pattern>(expr);
    ASSERT_TRUE(ret);
  }
  {
    Value expr =
        IndexDotValue<Value, List<SymbolicDimExpr>>{Value{Ok()}, List<SymbolicDimExpr>{std::int64_t(1)}};

    bool ret = cinn::adt::Match<Pattern>(expr);
    ASSERT_TRUE(ret);
  }
  {
    Value expr = List<Value>{Value{Ok()}, Value{Ok()}, Value{Ok()}};

    bool ret = cinn::adt::Match<Pattern>(expr);
    ASSERT_FALSE(ret);
  }
}

TEST(Match, index_undot) {
  Value expr =
      IndexUnDotValue<Value, List<SymbolicDimExpr>>{Value{Ok()}, List<SymbolicDimExpr>{std::int64_t(1)}};

  bool ret = cinn::adt::Match<IndexUnDotValue<Value, List<SymbolicDimExpr>>>(expr);
  ASSERT_TRUE(ret);
}

TEST(Match, index_dot) {
  Value expr =
      IndexDotValue<Value, List<SymbolicDimExpr>>{Value{Ok()}, List<SymbolicDimExpr>{std::int64_t(1)}};

  bool ret = cinn::adt::Match<IndexDotValue<Value, List<SymbolicDimExpr>>>(expr);
  ASSERT_TRUE(ret);
}

TEST(Match, list) {
  Value expr = List<Value>{Value{Ok()}, Value{Ok()}, Value{Ok()}};

  bool ret = cinn::adt::Match<List<Value>>(expr);
  ASSERT_TRUE(ret);
}

TEST(Match, list_get_item) {
  Value list = List<Value>{Value{Ok()}, Value{Ok()}, Value{Ok()}};
  Value expr = ListGetItem<Value, SymbolicDimExpr>{list, SymbolicDimExpr{std::int64_t(1)}};

  bool ret = cinn::adt::Match<ListGetItem<Value, std::int64_t>>(expr);
  ASSERT_TRUE(ret);
}

TEST(Match, list_get_item_index_undot) {
  Value undot1 =
      IndexUnDotValue<Value, List<SymbolicDimExpr>>{Value{Ok()}, List<SymbolicDimExpr>{std::int64_t(1)}};
  ASSERT_TRUE((cinn::adt::Match<IndexUnDotValue<Value, List<SymbolicDimExpr>>>(undot1)));

  Value expr = ListGetItem<Value, SymbolicDimExpr>{undot1, SymbolicDimExpr{std::int64_t(1)}};
  ASSERT_TRUE(
      (cinn::adt::Match<
          ListGetItem<IndexUnDotValue<Value, List<SymbolicDimExpr>>, std::int64_t>>(expr)));
}

// List<ListGetItem<IndexUnDotValue<Value>, std::int64_t>>
TEST(Match, list_list_get_item_index_undot) {
  Value undot =
      IndexUnDotValue<Value, List<SymbolicDimExpr>>{Value{Ok()}, List<SymbolicDimExpr>{std::int64_t(1)}};
  ASSERT_TRUE((cinn::adt::Match<IndexUnDotValue<Value, List<SymbolicDimExpr>>>(undot)));
  Value expr1 = ListGetItem<Value, SymbolicDimExpr>{undot, SymbolicDimExpr{std::int64_t(0)}};
  ASSERT_TRUE(
      (cinn::adt::Match<
          ListGetItem<IndexUnDotValue<Value, List<SymbolicDimExpr>>, std::int64_t>>(expr1)));
  Value expr2 = ListGetItem<Value, SymbolicDimExpr>{undot, SymbolicDimExpr{std::int64_t(1)}};
  ASSERT_TRUE(
      (cinn::adt::Match<
          ListGetItem<IndexUnDotValue<Value, List<SymbolicDimExpr>>, std::int64_t>>(expr2)));
  Value list = List<Value>{expr1, expr2};
  ASSERT_TRUE(
      (cinn::adt::Match<
          List<ListGetItem<IndexUnDotValue<Value, List<SymbolicDimExpr>>, std::int64_t>>>(
          list)));
}

// IndexDotValue<List<ListGetItem<IndexUnDotValue<Value>, std::int64_t>>>
TEST(Match, index_dot_list_list_get_item_index_undot) {
  Value undot1 =
      IndexUnDotValue<Value, List<SymbolicDimExpr>>{Value{Ok()}, List<SymbolicDimExpr>{std::int64_t(1)}};
  ASSERT_TRUE((cinn::adt::Match<IndexUnDotValue<Value, List<SymbolicDimExpr>>>(undot1)));
  Value expr1 = ListGetItem<Value, SymbolicDimExpr>{undot1, SymbolicDimExpr{std::int64_t(0)}};
  ASSERT_TRUE(
      (cinn::adt::Match<
          ListGetItem<IndexUnDotValue<Value, List<SymbolicDimExpr>>, std::int64_t>>(expr1)));
  Value expr2 = ListGetItem<Value, SymbolicDimExpr>{undot1, SymbolicDimExpr{std::int64_t(1)}};
  ASSERT_TRUE(
      (cinn::adt::Match<
          ListGetItem<IndexUnDotValue<Value, List<SymbolicDimExpr>>, std::int64_t>>(expr2)));
  Value list = List<Value>{expr1, expr2};
  ASSERT_TRUE(
      (cinn::adt::Match<
          List<ListGetItem<IndexUnDotValue<Value, List<SymbolicDimExpr>>, std::int64_t>>>(
          list)));
  Value dot = IndexDotValue<Value, List<SymbolicDimExpr>>{list, List<SymbolicDimExpr>{std::int64_t(1)}};
  ASSERT_TRUE(
      (cinn::adt::Match<IndexDotValue<
           List<ListGetItem<IndexUnDotValue<Value, List<SymbolicDimExpr>>, std::int64_t>>,
           List<SymbolicDimExpr>>>(dot)));
}

}  // namespace cinn::adt::test
