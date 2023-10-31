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

#include "paddle/cinn/adt/graph_symbolic_dim_infer_ctx.h"
#include "paddle/cinn/adt/symbolic_dim_expr.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn::adt::config {

class SymbolicDimInferCtx {
 public:
  SymbolicDimInferCtx(const SymbolicDimInferCtx&) = delete;
  SymbolicDimInferCtx(SymbolicDimInferCtx&&) = delete;

  SymbolicDimInferCtx(const hlir::framework::Node* node,
                      GraphSymbolicDimInferCtx* graph_ctx)
      : node_(node), graph_ctx_(graph_ctx) {}

  const std::vector<std::uint64_t>& GetInTensorsRanks() const {
    return graph_ctx_->GetInTensorsRanks(node_);
  }

  std::uint64_t GetNumOutTensors() const {
    return graph_ctx_->GetNumOutTensors(node_);
  }

  const SymbolicDimExpr& GetInputDimExpr(std::size_t arg_idx,
                                         std::size_t dim_idx) const {
    return graph_ctx_->GetInputDimExpr(node_, arg_idx, dim_idx);
  }

  void SetOutputDimExpr(std::size_t arg_idx,
                        std::size_t dim_idx,
                        const SymbolicDimExpr& value) {
    return graph_ctx_->SetOutputDimExpr(node_, arg_idx, dim_idx, value);
  }

  const framework::AttrMapType& GetAttributeMap() const {
    return graph_ctx_->GetAttributeMap(node_);
  }

 private:
  const hlir::framework::Node* node_;
  GraphSymbolicDimInferCtx* graph_ctx_;
};

}  // namespace cinn::adt::config
