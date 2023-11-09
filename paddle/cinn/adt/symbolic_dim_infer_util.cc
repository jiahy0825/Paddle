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

#include "paddle/cinn/adt/symbolic_dim_infer_util.h"

#include "paddle/cinn/adt/symbolic_dim_infer_ctx.h"
#include "paddle/cinn/hlir/framework/pir/group.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"

namespace cinn::adt {

// ADT_TODO : Replace Group with AnalysisManager
std::unique_ptr<config::GraphSymbolicDimInferCtx> InferSymbolicDim(
    const cinn::hlir::framework::pir::Group* group) {
  using InferSymbolicDimFunc =
      std::function<void(adt::config::SymbolicDimInferCtx * ctx)>;
  auto infer_ctx_ptr =
      std::make_unique<config::GraphSymbolicDimInferCtx>(group);

  for (const ::pir::Operation* op_node : group->ops) {
    VLOG(1) << "op_name : "
            << hlir::framework::pir::CompatibleInfo::OpName(*op_node);
    const auto& infer_symbolic_dim =
        hlir::framework::Operator::GetAttrs<InferSymbolicDimFunc>(
            "infer_symbolic_dim");

    const hlir::framework::Operator* cinn_op = hlir::framework::Operator::Get(
        hlir::framework::pir::CompatibleInfo::OpName(*op_node));
    CHECK(infer_symbolic_dim.Find(cinn_op));

    adt::config::SymbolicDimInferCtx ctx{op_node, infer_ctx_ptr.get()};
    infer_symbolic_dim[cinn_op](&ctx);
  }
  return infer_ctx_ptr;
}

}  // namespace cinn::adt
