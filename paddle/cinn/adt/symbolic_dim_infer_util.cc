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
#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"

namespace cinn::adt {

std::unique_ptr<config::GraphSymbolicDimInferCtx> InferSymbolicDim(
    const hlir::framework::Graph* graph) {
  using InferSymbolicDimFunc =
      std::function<void(adt::config::SymbolicDimInferCtx * ctx)>;
  auto infer_ctx_ptr =
      std::make_unique<config::GraphSymbolicDimInferCtx>(graph);

  std::vector<common::GraphNode*> topo_nodes =
      std::get<0>(graph->topological_order());
  for (const common::GraphNode* graph_node : topo_nodes) {
    const hlir::framework::Node* op_node =
        graph_node->safe_as<hlir::framework::Node>();
    // if node is NodeData or not op, continue.
    if (!op_node || op_node->op() == nullptr) {
      continue;
    }

    VLOG(1) << "op_name : " << op_node->op()->name;
    const auto& infer_symbolic_dim =
        hlir::framework::Operator::GetAttrs<InferSymbolicDimFunc>(
            "infer_symbolic_dim");
    CHECK(infer_symbolic_dim.Find(op_node->op()));

    adt::config::SymbolicDimInferCtx ctx{op_node, infer_ctx_ptr.get()};
    infer_symbolic_dim[op_node->op()](&ctx);
  }
  return infer_ctx_ptr;
}

}  // namespace cinn::adt
