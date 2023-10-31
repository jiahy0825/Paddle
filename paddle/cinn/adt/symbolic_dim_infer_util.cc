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

void InferSymbolicDim(const hlir::framework::Graph* graph) {
  using InferSymbolicDimFunc =
      std::function<void(adt::config::SymbolicDimInferCtx * ctx)>;

  graph.set_graph_ctx(std::make_unique<GraphSymbolicDimInferCtx>(graph));

  std::vector<GraphNode*> topo_nodes = std::get<0>(topological_order());
  for (const hlir::framework::GraphNode* graph_node : topo_nodes) {
    const Node* op_node = graph_node->safe_as<Node>();
    VLOG(1) << "op_name : " << op_node->op()->name;

    CHECK(op_node != nullptr && op_node->op() != nullptr);

    const auto& infer_symbolic_dim =
        hlir::framework::Operator::GetAttrs<InferSymbolicDimFunc>(
            "infer_symbolic_dim");
    CHECK(infer_symbolic_dim.Find(op_node->op()));

    adt::config::SymbolicDimInferCtx ctx{op_node, graph.mut_graph_ctx()};
    infer_symbolic_dim[op_node->op()](&ctx);
  }
}

}  // namespace cinn::adt
