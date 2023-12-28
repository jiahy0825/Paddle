#include "paddle/cinn/common/broadcast_tree.h"
#include "paddle/cinn/common/dim_expr_util.h"

#include <optional>
#include <unordered_map>

namespace cinn::common {

namespace {

template <typename DoEachT>
bool SearchBroadcast(const symbol::DimExpr& dim_expr, const DoEachT& DoEach);

template <typename DoEachT>
bool SearchBroadcastImpl(int64_t, const DoEachT& DoEach) {
  return false;
}

template <typename DoEachT>
bool SearchBroadcastImpl(const std::string&, const DoEachT& DoEach) {
  return false;
}

template <typename T, typename DoEachT>
bool SearchBroadcastImplForUnary(const T& unary, const DoEachT& DoEach) {
  const auto& [operand] = *unary;
  return SearchBroadcast(operand, DoEach);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Negative<symbol::DimExpr>& unary, const DoEachT& DoEach) {
  return SearchBroadcastImplForUnary(unary);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Reciprocal<symbol::DimExpr>& unary, const DoEachT& DoEach) {
  return SearchBroadcastImplForUnary(unary);
}

template <typename T, typename DoEachT>
bool SearchBroadcastImplForVariadic(const T& variadic, const DoEachT& DoEach) {
  const auto& [operands] = *variadic;
  for (const auto& operand : operands) {
    if (SearchBroadcast(operand, DoEach)) return true;
  }
  return false;
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Add<symbol::DimExpr>& variadic, const DoEachT& DoEach) {
  return SearchBroadcastImplForVariadic(variadic);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Mul<symbol::DimExpr>& variadic, const DoEachT& DoEach) {
  return SearchBroadcastImplForVariadic(variadic);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Max<symbol::DimExpr>& variadic, const DoEachT& DoEach) {
  return SearchBroadcastImplForVariadic(variadic);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Min<symbol::DimExpr>& variadic, const DoEachT& DoEach) {
  return SearchBroadcastImplForVariadic(variadic);
}

template <typename DoEachT>
bool SearchBroadcastImpl(const symbol::Broadcast<symbol::DimExpr>& variadic, const DoEachT& DoEach) {
  const auto& [operands] = *variadic;
  for (const auto& operand : operands) {
    CHECK(!operand.isa<int64_t>());
    if (SearchBroadcast(operand, DoEach)) return true;
  }
  return DoEach(variadic);
}

template <typename DoEachT>
bool SearchBroadcast(const symbol::DimExpr& dim_expr, const DoEachT& DoEach) {
  return std::visit([&](const auto& impl) {
    return SearchBroadcastImpl(impl, DoEach);
  }, dim_expr.variant());
}

template <typename DoEachT>
void ForEachBroadcastDimExpr(const BroadcastLeaf& leaves, const DoEachT& DoEach) {
  for (const auto& dim_exprs : *leaves) {
    for (const auto& dim_expr : dim_exprs) {
      if (SearchBroadcast(dim_expr, DoEach)) return;
    }
  }
}

std::optional<symbol::Broadcastable<symbol::DimExpr>> GetFirstCstrBroadcastable(const BroadcastLeaf& leaves) {
  std::optional<symbol::Broadcastable<symbol::DimExpr>> ret;
  ForEachBroadcastDimExpr(leaves, [&](const auto& broadcast) -> bool {
    const auto& [operands] = *broadcast;
    std::optional<symbol::DimExpr> lhs_symbol;
    std::optional<symbol::DimExpr> rhs_symbol;
    size_t i = 0;
    for (; i < operands->size(); ++i) {
      if (operands->at(i).isa<std::string>()) {
        lhs_symbol = operands->at(i);
        break;
      }
    }
    for (; i < operands->size(); ++i) {
      if (operands->at(i).isa<std::string>()) {
        rhs_symbol = operands->at(i);
        break;
      }
    }
    if (lhs_symbol.has_value() && rhs_symbol.has_value()) {
      CHECK(lhs_symbol != rhs_symbol);
      ret = symbol::Broadcastable{lhs_symbol.value(), rhs_symbol.value()};
      return true;
    }
    return false;
  });
  if (ret.has_value()) return ret.value();
  ForEachBroadcastDimExpr(leaves, [&](const auto& broadcast) -> bool {
    const auto& [operands] = *broadcast;
    std::optional<symbol::DimExpr> lhs_symbol;
    std::optional<symbol::DimExpr> rhs;
    for (const auto& operand : *operands) {
      if (operand.isa<std::string>()) {
        lhs_symbol = operand;
        break;
      }
    }
    for (const auto& operand : *operands) {
      if (operand != lhs_symbol) {
        rhs = operand;
        break;
      }
    }
    if (lhs_symbol.has_value() && rhs.has_value()) {
      ret = symbol::Broadcastable{lhs_symbol.value(), rhs.value()};
      return true;
    }
    return false;
  });
  if (ret.has_value()) return ret.value();
  ForEachBroadcastDimExpr(leaves, [&](const auto& broadcast) -> bool {
    const auto& [operands] = *broadcast;
    CHECK_GE(operands->size(), 2);
    CHECK(operands->at(0) != operands->at(1));
    ret = symbol::Broadcastable<symbol::DimExpr>{operands->at(0), operands->at(1)};
    return true;
  });
  return ret;
}

using Pattern2Placement = std::unordered_map<symbol::DimExpr, symbol::DimExpr>;

Pattern2Placement
ConstructCstrLhsEqRhsReplacement(const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition) {
  auto [lhs, rhs] = *broadcastable_condition;
  if (lhs.isa<std::string>()) return Pattern2Placement{{lhs, rhs}};
  if (rhs.isa<std::string>()) return Pattern2Placement{{rhs, lhs}};
  return Pattern2Placement{{lhs, rhs}};
}

Pattern2Placement
ConstructCstrLhsEqOneReplacement(const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition) {
  const auto& [lhs, rhs] = *broadcastable_condition;
  return Pattern2Placement{{lhs, symbol::DimExpr{1}}};
}

Pattern2Placement
ConstructCstrRhsEqOneReplacement(const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition) {
  const auto& [lhs, rhs] = *broadcastable_condition;
  return Pattern2Placement{{rhs, symbol::DimExpr{1}}};
}

symbol::DimExpr GetCstrLhsEqRhsDimExpr(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr) {
  const auto& pattern2replacement = ConstructCstrLhsEqRhsReplacement(broadcastable_condition);
  return SimplifyDimExpr(SubstituteDimExpr(dim_expr, pattern2replacement));
}

symbol::DimExpr GetCstrLhsEqOneDimExpr(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr) {
  const auto& pattern2replacement = ConstructCstrLhsEqOneReplacement(broadcastable_condition);
  return SimplifyDimExpr(SubstituteDimExpr(dim_expr, pattern2replacement));
}

symbol::DimExpr GetCstrRhsEqOneDimExpr(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr) {
  const auto& pattern2replacement = ConstructCstrRhsEqOneReplacement(broadcastable_condition);
  return SimplifyDimExpr(SubstituteDimExpr(dim_expr, pattern2replacement));
}

typedef symbol::DimExpr (*ConvertDimExprT)(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const symbol::DimExpr& dim_expr);

template <ConvertDimExprT ConvertDimExpr>
BroadcastLeaf ConvertBroadcastLeaf(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  BroadcastLeaf ret{};
  for (const auto& dim_exprs : *leaves) {
    std::vector<symbol::DimExpr> converted{};
    converted.reserve(dim_exprs.size());
    for (const auto& dim_expr : dim_exprs) {
      converted.push_back(ConvertDimExpr(broadcastable_condition, dim_expr));
    }
    ret->emplace_back(std::move(converted));
  }
  return ret;
}

BroadcastLeaf GetCstrLhsEqRhsLeaves(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  return ConvertBroadcastLeaf<&GetCstrLhsEqRhsDimExpr>(broadcastable_condition, leaves);
}

BroadcastLeaf GetCstrLhsEqOneLeaves(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  return ConvertBroadcastLeaf<&GetCstrLhsEqOneDimExpr>(broadcastable_condition, leaves);
}

BroadcastLeaf GetCstrRhsEqOneLeaves(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  return ConvertBroadcastLeaf<&GetCstrRhsEqOneDimExpr>(broadcastable_condition, leaves);
}

BroadcastBranch<BroadcastTree> ConstructBroadcastBranch(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const BroadcastLeaf& leaves) {
  BroadcastLeaf cstr_lhs_eq_rhs_leaves = GetCstrLhsEqRhsLeaves(broadcastable_condition, leaves);
  BroadcastLeaf cstr_lhs_eq_one_leaves = GetCstrLhsEqOneLeaves(broadcastable_condition, leaves);
  BroadcastLeaf cstr_rhs_eq_one_leaves = GetCstrRhsEqOneLeaves(broadcastable_condition, leaves);
  return BroadcastBranch<BroadcastTree>{
    .broadcastable_condition=broadcastable_condition,
    .cstr_lhs_eq_rhs_branch=ConstructBroadcastTree(cstr_lhs_eq_rhs_leaves),
    .cstr_lhs_eq_one_branch=ConstructBroadcastTree(cstr_lhs_eq_one_leaves),
    .cstr_rhs_eq_one_branch=ConstructBroadcastTree(cstr_rhs_eq_one_leaves),
  };
}

}

BroadcastTree ConstructBroadcastTree(const BroadcastLeaf& leaves) {
  std::optional<symbol::Broadcastable<symbol::DimExpr>> broadcastable_condition =
      GetFirstCstrBroadcastable(leaves);
  if (!broadcastable_condition.has_value()) return leaves;
  return ConstructBroadcastBranch(broadcastable_condition.value(), leaves);
}

}