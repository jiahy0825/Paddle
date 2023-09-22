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

#include "paddle/cinn/adt/print_constant.h"

namespace cinn::adt {

namespace {
struct ToTxtStringStruct {
  std::string operator()(const std::int64_t constant) {
    return std::to_string(constant);
  }

  std::string operator()(const tDim<UniqueId>& constant) {
    std::size_t constant_unique_id = constant.value().unique_id();
    return "dim_" + std::to_string(constant_unique_id);
  }

  std::string operator()(const List<Constant>& constant_list) {
    std::string ret;
    ret += "(";

    for (std::size_t idx = 0; idx < constant_list->size(); ++idx) {
      if (idx != 0) {
        ret += ", ";
      }
      ret += ToTxtString(constant_list.Get(idx));
    }

    ret += ")";
    return ret;
  }

  std::string operator()(const Neg<Constant>& constant) {
    std::string ret;
    const auto& [constant_] = constant.tuple();
    ret += ToTxtString(constant_);
    return ret;
  }

  std::string operator()(const Add<Constant, Constant>& constant) {
    std::string ret;
    const auto& [left, right] = constant.tuple();
    ret += "add(" + ToTxtString(left) + ", " + ToTxtString(right) + ")";
    return ret;
  }

  std::string operator()(const Mul<Constant, Constant>& constant) {
    std::string ret;
    const auto& [left, right] = constant.tuple();
    ret += "mul(" + ToTxtString(left) + ", " + ToTxtString(right) + ")";
    return ret;
  }
};
}  // namespace

std::string ToTxtString(const Constant& constant) {
  return std::visit(ToTxtStringStruct{}, constant.variant());
}

}  // namespace cinn::adt