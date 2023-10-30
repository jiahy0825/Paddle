#pragma once

#include "paddle/cinn/adt/symbolic_dim_expr.h"
#include "paddle/cinn/adt/match.h"

namespace cinn::adt {

template <>
struct MatchTrait<SymbolicDimExpr, std::int64_t> final {
  static constexpr int is_template = false;
};

template <>
struct MatchTrait<SymbolicDimExpr, SymbolicDim> final {
  static constexpr int is_template = false;
};

#define DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(name, type0, type1)          \
  template <typename T0, typename T1>                                         \
  struct MatchTrait<SymbolicDimExpr, name<T0, T1>> final {                              \
    using base_type = name<type0, type1>;                                     \
                                                                              \
    static constexpr int is_template = true;                                  \
                                                                              \
    template <template <typename> class Matcher>                              \
    static bool MatchChildren(const base_type& value) {                       \
      return Matcher<type0>::template Call<T0>(std::get<0>(value.tuple())) && \
             Matcher<type1>::template Call<T1>(std::get<1>(value.tuple()));   \
    }                                                                         \
  };

DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(Add, SymbolicDimExpr, SymbolicDimExpr);
DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(Sub, SymbolicDimExpr, SymbolicDimExpr);
DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(Mul, SymbolicDimExpr, SymbolicDimExpr);
DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(Div, SymbolicDimExpr, SymbolicDimExpr);
DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(BroadcastedDim, SymbolicDimExpr, SymbolicDimExpr);
#undef DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2

template <typename T>
struct MatchTrait<SymbolicDimExpr, OneOf<T>> final {
  using base_type = OneOf<SymbolicDimExpr>;

  static constexpr int is_template = true;

  template <template <typename> class Matcher>
  static bool MatchChildren(const base_type& oneof) {
    for (const auto& value : *oneof->list()) {
      if (!Matcher<Constant>::template Call<T>(value)) {
        return false;
      }
    }
    return true;
  }
};

}  // namespace cinn::adt
