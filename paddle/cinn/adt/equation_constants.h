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

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/logical.h"
#include "paddle/cinn/adt/tags.h"
#include "paddle/cinn/adt/unique_id.h"

namespace cinn::adt {

// Dim = tDim UniqueId
using Dim = tDim<UniqueId>;
// DimTuple = [Dim]
using DimTuple = List<Dim>;
// Stride = tStride UniqueId
using Stride = tStride<UniqueId>;
// StrideTuple = [Stride]
using StrideTuple = List<Stride>;

// EquationStaticValue = Dim | Stride | std::int64_t
DEFINE_ADT_UNION(EquationStaticValue, Dim, Stride, std::int64_t);
OVERLOAD_OPERATOR_EQ_NE(EquationStaticValue, UnionEqual);

using EquationStaticLogical = Logical<EquationStaticValue>;

}  // namespace cinn::adt
