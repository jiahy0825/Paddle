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

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"

#include <vector>
#include "glog/logging.h"
#include "paddle/common/enforce.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/op_base.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_tensor.h"
#include "paddle/fluid/pir/dialect/operator/ir/ir_meta_tensor.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"

namespace cinn {
namespace dialect {

const char *GroupOp::attributes_name[GroupOp::attributes_num] = {"group_info"};
const char *ConcatOp::attributes_name[ConcatOp::attributes_num] = {"axis"};
const char *SplitOp::attributes_name[SplitOp::attributes_num] = {
    "num_or_sections", "axis"};

void GroupOp::Build(pir::Builder &builder,
                    pir::OperationArgument &argument,
                    const std::vector<pir::Type> &output_types) {
  argument.AddRegion(nullptr);
  argument.output_types = output_types;
}

void GroupOp::Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    std::unique_ptr<pir::Block> &&block) {
  VLOG(4) << "Start build GroupOp";
  if (block && !block->empty()) {
    IR_ENFORCE(block->back().isa<pir::YieldOp>());
    auto &op = block->back();
    for (size_t i = 0; i < op.num_operands(); ++i) {
      argument.AddOutput(op.operand(i).type());
    }
  }
  argument.AddRegion().push_back(block.release());
}

pir::Block *GroupOp::block() {
  pir::Region &region = (*this)->region(0);
  if (region.empty()) region.emplace_back();
  return &region.front();
}

std::vector<pir::Operation *> GroupOp::ops() {
  std::vector<pir::Operation *> rt_ops;
  for (auto &op : *block()) {
    rt_ops.push_back(&op);
  }
  return rt_ops;
}

void GroupOp::VerifySig() {}

void GroupOp::Print(pir::IrPrinter &printer) {
  auto &os = printer.os;
  auto op = operation();
  printer.PrintOpResult(op);
  os << " = " << name();
  printer.PrintOpOperands(op);
  os << " -> ";
  printer.PrintOpReturnType(op);
  os << " {";
  for (auto &sub_op : ops()) {
    os << "\n";
    printer.PrintOperation(sub_op);
  }
  os << " \n }";
}

void ConcatOp::Build(pir::Builder &builder,             // NOLINT
                     pir::OperationArgument &argument,  // NOLINT
                     const std::vector<pir::Value> &inputs,
                     int axis) {
  VLOG(4) << "Start build ConcatOp";

  argument.inputs = inputs;
  std::vector<pir::Type> inputs_type(inputs.size());

  IR_ENFORCE(inputs.size() > 0);

  auto first_ele =
      inputs[0].type().dyn_cast<paddle::dialect::DenseTensorType>();
  phi::DDim out_dims = first_ele.dims();

  if (axis < 0) {
    axis += out_dims.size();
  }

  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    inputs_type[idx] = inputs[idx].type();

    if (idx > 0) {
      auto dim_i = inputs[idx]
                       .type()
                       .dyn_cast<paddle::dialect::DenseTensorType>()
                       .dims();

      out_dims[axis] += dim_i[axis];
    }
  }

  auto out_type =
      paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(),
                                            first_ele.dtype(),
                                            out_dims,
                                            first_ele.data_layout(),
                                            first_ele.lod(),
                                            first_ele.offset());

  argument.output_types.emplace_back(out_type);

  PassStopGradientsDefaultly(argument);

  argument.AddAttribute(
      "axis", pir::Int32Attribute::get(pir::IrContext::Instance(), axis));
}

void SplitOp::Build(pir::Builder &builder,             // NOLINT
                    pir::OperationArgument &argument,  // NOLINT
                    pir::Value input,
                    const std::vector<int> &sections,
                    int axis) {
  VLOG(4) << "Start build ConcatOp";

  argument.inputs.push_back(input);

  std::vector<pir::Type> output_type(sections.size());

  auto input_ele = input.type().dyn_cast<paddle::dialect::DenseTensorType>();

  if (axis < 0) {
    axis += input_ele.dims().size();
  }
  std::vector<pir::Attribute> section_attrs;
  for (size_t idx = 0; idx < sections.size(); ++idx) {
    auto out_dims = input_ele.dims();
    out_dims[axis] = sections[idx];
    auto out_type =
        paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(),
                                              input_ele.dtype(),
                                              out_dims,
                                              input_ele.data_layout(),
                                              input_ele.lod(),
                                              input_ele.offset());

    argument.output_types.emplace_back(out_type);

    pir::Attribute attr_axis =
        pir::Int32Attribute::get(pir::IrContext::Instance(), sections[idx]);

    section_attrs.push_back(attr_axis);
  }

  PassStopGradientsDefaultly(argument);

  argument.AddAttribute(
      "num_or_sections",
      pir::ArrayAttribute::get(pir::IrContext::Instance(), section_attrs));

  argument.AddAttribute(
      "axis", pir::Int32Attribute::get(pir::IrContext::Instance(), axis));
}

const char *GenerateShapeOp::attributes_name[attributes_num] = {
  "output_dim_exprs",
  "symbol_bindings"};

void GenerateShapeOp::Build(pir::Builder &builder,
                            pir::OperationArgument& argument,
                            const std::vector<pir::Value>& inputs,
                            const std::vector<pir::Attribute>& output_dim_exprs,
                            const GenerateShapeOp::SymbolBindings& symbol_bindings) {
  CHECK(!inputs.empty());
  argument.AddInputs(inputs);
  argument.AddAttribute("output_dim_exprs", builder.array_attr(output_dim_exprs));
  argument.AddAttribute("symbol_bindings", ConvertSymbolBindingsToAttribute(builder, symbol_bindings));
  argument.AddOutputs({[&](){
    auto* ctx = pir::IrContext::Instance();
    auto type = pir::Int64Type::get(ctx);
    auto dim = ::common::make_ddim({static_cast<int64_t>(output_dim_exprs.size())});
    return paddle::dialect::DenseTensorType::get(ctx, type, dim);
  }()});
  ::pir::PassStopGradientsDefaultly(argument);
}

pir::Attribute GenerateShapeOp::ConvertSymbolBindingsToAttribute(pir::Builder &builder, const GenerateShapeOp::SymbolBindings& symbol_bindings) {
  const auto& ConvertSymbolBindingToAttr = [&](const SymbolBinding& binding) {
    const auto& [symbol_name, input_tensor_idx, input_tensor_dim_idx] = binding;
    return builder.array_attr({
      builder.str_attr(symbol_name),
      builder.int64_attr(input_tensor_idx),
      builder.int64_attr(input_tensor_dim_idx),
    });
  };
  std::vector<pir::Attribute> bindings_attr{};
  for (const auto& symbol_binding : symbol_bindings) {
    bindings_attr.push_back(ConvertSymbolBindingToAttr(symbol_binding));
  }
  return builder.array_attr(bindings_attr);
}

std::optional<GenerateShapeOp::SymbolBindings> GenerateShapeOp::ConvertAttributeToSymbolBindings(const pir::Attribute& symbol_bindings) {
  if (!symbol_bindings.isa<pir::ArrayAttribute>()) {
    return std::nullopt;
  }
  const auto& symbol_bindings_array_attr = symbol_bindings.dyn_cast<pir::ArrayAttribute>();
  GenerateShapeOp::SymbolBindings ret{GenerateShapeOp::SymbolBindings{}};
  for (int i = 0; i < symbol_bindings_array_attr.size(); ++i) {
    const auto& symbol_binding = symbol_bindings_array_attr.at(i);
    if (!symbol_binding.isa<pir::ArrayAttribute>()) {
      return std::nullopt;
    }
    const auto& symbol_binding_array_attr = symbol_binding.dyn_cast<pir::ArrayAttribute>();
    if (symbol_binding_array_attr.size() != 3) {
      return std::nullopt;
    }
    if (!symbol_binding_array_attr.at(0).isa<pir::StrAttribute>()) {
      return std::nullopt;
    }
    if (!symbol_binding_array_attr.at(1).isa<pir::Int64Attribute>()) {
      return std::nullopt;
    }
    if (!symbol_binding_array_attr.at(2).isa<pir::Int64Attribute>()) {
      return std::nullopt;
    }
    ret.emplace_back(std::tuple{
      symbol_binding_array_attr.at(0).dyn_cast<pir::StrAttribute>().AsString(),
      symbol_binding_array_attr.at(1).dyn_cast<pir::Int64Attribute>().data(),
      symbol_binding_array_attr.at(2).dyn_cast<pir::Int64Attribute>().data(),
    });
  }
  return std::move(ret);
}

}  // namespace dialect
}  // namespace cinn

IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::GroupOp)
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::ConcatOp)
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::SplitOp)
IR_DEFINE_EXPLICIT_TYPE_ID(cinn::dialect::GenerateShapeOp);