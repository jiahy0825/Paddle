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

#include <memory>
#include <unordered_map>

#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/utils/ir_copy.h"

namespace cinn::adt {

class Kernel;
using MapExpr = Kernel;

class MapExprGuard final {
 public:
  using Node2LoweredFuncs =
      std::unordered_map<hlir::framework::Node*, std::vector<ir::LoweredFunc>>;

  MapExprGuard(const MapExprGuard&) = delete;
  MapExprGuard(MapExprGuard&&) = delete;

  explicit MapExprGuard(const std::shared_ptr<MapExpr>& map_expr)
      : map_expr_(map_expr) {
    CHECK(!HasMapExprGuard());
    MutThreadLocalGuard() = this;
  }

  ~MapExprGuard() { MutThreadLocalGuard() = nullptr; }

  static bool HasMapExprGuard() {
    if (MutThreadLocalGuard()) {
      return true;
    }
    return false;
  }

  static const MapExpr& GetMapExpr() {
    CHECK(HasMapExprGuard());
    return *(MutThreadLocalGuard()->map_expr_);
  }

  static void UpdateOpLoweredFuncKey(
      hlir::framework::Node* node,
      const std::vector<ir::LoweredFunc>& lowered_funcs) {
    CHECK(HasMapExprGuard());
    Node2LoweredFuncs* map = &MutThreadLocalGuard()->node2lowered_funcs_;
    CHECK(map->emplace(node, ir::ir_utils::IRCopy(lowered_funcs)).second);
  }

  static const Node2LoweredFuncs& GetNode2LoweredFuncs() {
    CHECK(HasMapExprGuard());
    return MutThreadLocalGuard()->node2lowered_funcs_;
  }

 private:
  static MapExprGuard*& MutThreadLocalGuard() {
    static thread_local MapExprGuard* map_expr_guard = nullptr;
    return map_expr_guard;
  }

  std::shared_ptr<MapExpr> map_expr_;
  Node2LoweredFuncs node2lowered_funcs_;
};

}  // namespace cinn::adt
