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

#include "paddle/cinn/adt/schedule_descriptor.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/igroup.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/kgroup.h"
#include "paddle/cinn/adt/schedule_dim.h"

namespace cinn::adt {

LoopDescriptors CreateScheduleDescriptor(const ScheduleMesh& sched_mesh,
                                         const List<LoopType>& loop_types) {
  const auto& sched_dims = GetOutputDimValues(sched_mesh);
  CHECK_EQ(sched_dims->size(), loop_types->size());
  LoopDescriptors ret{};
  for (std::size_t i = 0; i < sched_dims->size(); ++i) {
    const auto& sched_dim = sched_dims->at(i);
    const auto& opt_loop_size = ConstantToLoopSize(sched_dim);
    CHECK(opt_loop_size.has_value());
    ret->emplace_back(LoopDescriptor{loop_types->at(i), opt_loop_size.value()});
  }
  return ret;
}

List<LoopSize> GenerateLoopSizeFromSd(const LoopDescriptors& sd) {
  List<LoopSize> sd_sizes{};
  for (const auto& loop_descriptor : *sd) {
    const auto& [loop_type, loop_size] = loop_descriptor.tuple();
    sd_sizes->emplace_back(loop_size);
  }
  return sd_sizes;
}

std::optional<LoopSize> ConstantToLoopSizeImpl(const std::int64_t constant) {
  return LoopSize{constant};
}

std::optional<LoopSize> ConstantToLoopSizeImpl(const EquationDim& constant) {
  return std::nullopt;
}

std::optional<LoopSize> ConstantToLoopSizeImpl(const SymbolicDim& constant) {
  return LoopSize{constant};
}

std::optional<LoopSize> ConstantToLoopSizeImpl(const List<Constant>& constant) {
  return std::nullopt;
}

std::optional<LoopSize> ConstantToLoopSize(const Constant& constant) {
  return std::visit(
      [&](const auto& impl) { return ConstantToLoopSizeImpl(impl); },
      constant.variant());
}

Constant LoopSize2ConstantImpl(const std::int64_t loop_size) {
  return Constant{loop_size};
}

Constant LoopSize2ConstantImpl(const SymbolicDim& loop_size) {
  return Constant{loop_size};
}

Constant LoopSize2Constant(const LoopSize& loop_size) {
  return std::visit(
      [&](const auto& impl) { return LoopSize2ConstantImpl(impl); },
      loop_size.variant());
}

}  // namespace cinn::adt
