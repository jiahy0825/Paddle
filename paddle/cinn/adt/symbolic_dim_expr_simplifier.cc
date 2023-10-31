#include "paddle/cinn/adt/symbolic_dim_expr_simplifier.h"

namespace cinn::adt {

namespace {

SymbolicDimExpr Simplify(SymbolicDimExpr&& expr);

template <typename T>
SymbolicDimExpr TrySimplifyPass(SymbolicDimExpr&& expr) {
  if (cinn::adt::Match<typename T::source_pattern_type>(expr)) {
    return T().MatchAndRewrite(std::move(expr));
  } else {
    return expr;
  }
}

struct FoldConstantAdd {
  using source_pattern_type = Add<std::int64_t, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    const auto&[lhs, rhs] = expr.Get<Add<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    return lhs.Get<std::int64_t>() + rhs.Get<std::int64_t>();
  }
};

struct FoldConstantMul {
  using source_pattern_type = Mul<std::int64_t, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    const auto&[lhs, rhs] = expr.Get<Mul<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    return lhs.Get<std::int64_t>() * rhs.Get<std::int64_t>();
  }
};

struct FoldConstantBroadcastedDim {
  using source_pattern_type = BroadcastedDim<std::int64_t, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    const auto&[lhs, rhs] = expr.Get<BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>>();
    const std::int64_t int64_lhs = lhs.Get<std::int64_t>();
    const std::int64_t int64_rhs = rhs.Get<std::int64_t>();
    if (int64_lhs == 1) {
      return rhs;
    }
    if (int64_rhs == 1) {
      return lhs;
    }
    CHECK_EQ(int64_lhs, int64_rhs);
    return rhs;
  }
};

template <template<typename, typename> class Op>
struct SwapConstantLhs {
  using source_pattern_type = Op<std::int64_t, SymbolicDimExpr>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [lhs, rhs] = *expr.Mut<Op<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    std::swap(lhs, rhs);
    return expr;
  }
};

template <template<typename, typename> class Op, int unit>
struct FoldConstantUnit {
  using source_pattern_type = Op<SymbolicDimExpr, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    const auto& [lhs, rhs] = expr.Get<Op<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    if (rhs.Get<std::int64_t>() == unit) {
      return lhs;
    } else {
      return expr;
    }
    LOG(FATAL) << "Dead code.";
  }
};

using ConstIntegerPattern = Union<std::int64_t, Negtive<std::int64_t>>;

std::int64_t GetInteger(const SymbolicDimExpr& expr) {
  if (expr.Has<Negtive<SymbolicDimExpr>>()) {
    const auto& [integer] = expr.Get<Negtive<SymbolicDimExpr>>();
    CHECK(integer.Has<std::int64_t>());
    return -integer.Get<std::int64_t>();
  }
  CHECK(expr.Has<std::int64_t>());
  return expr.Get<std::int64_t>();
}

struct RemoveRedundantConstantLhs_Add_Add {
  using source_pattern_type = Add<Add<SymbolicDimExpr, ConstIntegerPattern>, ConstIntegerPattern>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Add<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Add<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    inner_lhs = Simplify(std::Move(inner_lhs));
    inner_rhs = SymbolicDimExpr(GetInteger(outter_rhs) + GetInteger(inner_rhs));
    return outter_lhs;
  }
};

using ConstRationalPattern = Union<std::int64_t,
                                   Reciprocal<std::int64_t>,
                                   Mul<std::int64_t, Reciprocal<std::int64_t>>>;

using ConstRational = std::pair<std::int64_t, std::int64_t>;

template<typename T>
ConstRational GetConstRationalImpl(const T& expr) {
  LOG(FATAL) << "not supported.";
}

ConstRational GetConstRationalImpl(std::int64_t value) {
  return ConstRational{value, 1};
}

ConstRational GetConstRationalImpl(const Reciprocal<SymbolicDimExpr>& value) {
  const auto& [denominator] = value.tuple();
  return ConstRational{1, denominator.Get<std::int64_t>()};
}

ConstRational SimplifiedConstRational(int64_t num, int64_t dem) {
  std::int64_t gcd = std::gcd(num, dem);
  return ConstRational{num / gcd, dem / gdc};
}

ConstRational GetConstRationalImpl(const Mul<SymbolicDimExpr, SymbolicDimExpr>& value) {
  const auto& [numerator, reciprocal] = value.tuple();
  const auto& [denominator] = reciprocal.tuple();
  return SimplifiedConstRational(numerator.Get<std::int64_t>(), denominator.Get<std::int64_t>());
}

ConstRational GetConstRational(const SymbolicDimExpr& expr) {
  return std::visit([&](const auto& impl) {
    return GetConstRationalImpl(impl);
  }, expr.variant());
}

ConstRational MulConstRational(const ConstRational& lhs, const ConstRational& rhs) {
 const auto [lhs_num, lhs_dem] = lhs;
 const auto [rhs_num, rhs_dem] = rhs;
 // Crossing is correct.
 const auto [simplifed_lhs_num, simplifed_rhs_dem] = SimplifiedConstRational(lhs_num, rhs_dem);
 const auto [simplifed_rhs_num, simplifed_lhs_dem] = SimplifiedConstRational(rhs_num, lhs_dem);
 return ConstRational{simplifed_lhs_num * simplifed_rhs_num, simplifed_lhs_dem * simplifed_rhs_dem};
}

struct RemoveRedundantConstantLhs_Mul_Mul {
  using source_pattern_type = Mul<Mul<SymbolicDimExpr, ConstIntegerPattern>, ConstIntegerPattern>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Mul<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Mul<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    inner_lhs = Simplify(std::Move(inner_lhs));
    const auto [num, dem] = MulConstRational(GetConstRational(outter_rhs), GetConstRational(inner_rhs));
    if (num == 1) {
      if (dem == 1) {
        return inner_lhs;
      } else {
        inner_rhs = Reciprocal<SymbolicDimExpr>{SymbolicDimExpr{dem}};
        return outter_lhs;
      }
    } else {
      if (dem == 1) {
        inner_rhs = SymbolicDimExpr{num};
        return outter_lhs;
      } else {
        inner_rhs = SymbolicDimExpr{num} / SymbolicDimExpr{dem};
        return outter_lhs;
      }
    }
    LOG(FATAL) << "Dead code";
  }
};


SymbolicDimExpr Simplify(SymbolicDimExpr&& expr) {
  expr = TrySimplifyPass<FoldConstantAdd>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantMul>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantBroadcastedDim>(std::move(expr));
  expr = TrySimplifyPass<SwapConstantLhs<Add>>(std::move(expr));
  expr = TrySimplifyPass<SwapConstantLhs<Mul>>(std::move(expr));
  expr = TrySimplifyPass<SwapConstantLhs<BroadcastedDim>>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantUnit<Add, 0>>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantUnit<Mul, 1>>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantUnit<BroadcastedDim, 1>>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Add_Add>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Mul_Mul>(std::move(expr));
  return expr;
}

}  // namespace

SymbolicDimExpr SimplifySymbolicDimExpr(SymbolicDimExpr&& expr) {
  return Simplify(expr);
}

}  // namespace cinn::adt
