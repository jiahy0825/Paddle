#pragma once

#include <optional>
#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/arithmetic.h"

namespace cinn::adt {

template<typename VarT, typename ValueT, typename BodyT>
struct Let final {
  List<std::pair<VarT, ValueT>> var2value;
  BodyT body;
};

template<typename ConditionT, typename TrueValueT, typename FalseValueT>
struct If final {
  ConditionT condition;
  TrueValueT true_value;
  std::optional<FalseValueT> false_value;
};

}
