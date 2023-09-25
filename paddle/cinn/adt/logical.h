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

namespace cinn::adt {

DEFINE_ADT_BINARY(EQ);
DEFINE_ADT_BINARY(LT);
DEFINE_ADT_BINARY(GT);
DEFINE_ADT_BINARY(NE);
DEFINE_ADT_BINARY(GE);
DEFINE_ADT_BINARY(LE);
DEFINE_ADT_BINARY(And);
DEFINE_ADT_BINARY(Or);
DEFINE_ADT_UNARY(Not);

// Logical T = EQ T T
//           | LT T T
//           | GT T T
//           | NE T T
//           | GE T T
//           | LE T T
//           | And (Logical T) (Logical T)
//           | Or (Logical T) (Logical T)
//           | Not (Logical T)
template <typename EquationValue>
DEFINE_ADT_UNION(Logical,
                 EQ<EquationValue, EquationValue>,
                 LT<EquationValue, EquationValue>,
                 GT<EquationValue, EquationValue>,
                 NE<EquationValue, EquationValue>,
                 GE<EquationValue, EquationValue>,
                 LE<EquationValue, EquationValue>,
                 And<Logical<EquationValue>, Logical<EquationValue>>,
                 Or<Logical<EquationValue>, Logical<EquationValue>>,
                 Not<Logical<EquationValue>>);
}  // namespace cinn::adt
