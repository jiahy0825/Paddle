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

#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/equation_function_constants_provider.h"

namespace cinn::adt {

std::optional<std::int64_t> IndexExprInferContext::GetStaticDimSize(
    const EquationDim& dim) const {
  return constants_provider_->GetStaticDimSize(dim);
}

SymbolicDimExpr IndexExprInferContext::GetDimSize(
    const EquationDim& dim) const {
  return constants_provider_->GetDimSize(dim);
}

bool IndexExprInferContext::DimsEqual(const List<Constant>& lhs,
                                      const List<Constant>& rhs) const {
  const auto& GetSymbolicDimExpr =
      [&](const Constant& constant) -> SymbolicDimExpr {
    if (constant.Has<std::int64_t>()) {
      return SymbolicDimExpr{constant.Get<std::int64_t>()};
    } else if (constant.Has<EquationDim>()) {
      return GetDimSize(constant.Get<EquationDim>());
    } else {
      LOG(FATAL) << "Not supported";
    }
  };
  if (lhs == rhs) {
    return true;
  }
  if (lhs->size() != rhs->size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs->size(); ++i) {
    if (GetSymbolicDimExpr(lhs->at(i)) != GetSymbolicDimExpr(rhs->at(i))) {
      return false;
    }
  }
  return true;
}

}  // namespace cinn::adt
