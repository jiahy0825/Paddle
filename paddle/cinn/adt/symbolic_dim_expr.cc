#include "paddle/cinn/adt/symbolic_dim_expr.h"
#include <type_traits>


namespace cinn::adt {

namespace {
  
bool SymbolicDimExprEqualImpl(std::int64_t lhs, std::int64_t rhs) {
  return lhs == rhs;
}

bool SymbolicDimExprEqualImpl(const SymbolicDim& lhs, const SymbolicDim& rhs) {
  return lhs == rhs;
}

template<template<typename,typename> class Op>
bool SymbolicDimExprEqual(
  const Op<SymbolicDimExpr, SymbolicDimExpr>& lhs, const Op<SymbolicDimExpr, SymbolicDimExpr>& rhs) {
  const auto& [lhs_arg0, lhs_arg1] = lhs.tuple();
  const auto& [rhs_arg0, rhs_arg1] = lhs.tuple();
  return lhs_arg0 == rhs_arg0 and lhs_arg1 == rhs_arg1;
}

#define SPECIALIZE_SYMBOLIC_DIM_EXPR(Op)                                                                \
bool SymbolicDimExprEqualImpl( \
    const Op<SymbolicDimExpr, SymbolicDimExpr>& lhs, const Op<SymbolicDimExpr, SymbolicDimExpr>& rhs) { \
  return SymbolicDimExprEqual<Op>(lhs, rhs); \
}
SPECIALIZE_SYMBOLIC_DIM_EXPR(Add);
SPECIALIZE_SYMBOLIC_DIM_EXPR(Sub);
SPECIALIZE_SYMBOLIC_DIM_EXPR(Mul);
SPECIALIZE_SYMBOLIC_DIM_EXPR(Div);
SPECIALIZE_SYMBOLIC_DIM_EXPR(BroadcastedDim);
#undef SPECIALIZE_SYMBOLIC_DIM_EXPR;

}  // namespace

bool operator==(const SymbolicDimExpr& lhs, const SymbolicDimExpr& rhs) {
  return std::visit([](const auto& lhs, const auto& rhs) {
    if (std::is_same_v<std::decay_t<decltype(lhs)>, std::decay_t<decltype(rhs)>>) {
      return SymbolicDimExprEqualImpl(lhs, rhs);
    } else {
      return false;
    }
  }, lhs.variant(), rhs.variant());
}

}  // namespace cinn::adt
