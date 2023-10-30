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

struct FoldConstantSub {
  using source_pattern_type = Sub<std::int64_t, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    const auto&[lhs, rhs] = expr.Get<Sub<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    CHECK_GE(lhs.Get<std::int64_t>(), rhs.Get<std::int64_t>());
    return lhs.Get<std::int64_t>() - rhs.Get<std::int64_t>();
  }
};

struct FoldConstantMul {
  using source_pattern_type = Mul<std::int64_t, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    const auto&[lhs, rhs] = expr.Get<Mul<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    return lhs.Get<std::int64_t>() * rhs.Get<std::int64_t>();
  }
};

struct FoldConstantDiv {
  using source_pattern_type = Div<std::int64_t, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    const auto&[lhs, rhs] = expr.Get<Div<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    CHECK_EQ(lhs.Get<std::int64_t>() % rhs.Get<std::int64_t>(), 0);
    return lhs.Get<std::int64_t>() / rhs.Get<std::int64_t>();
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

struct RemoveRedundantConstantLhs_Add_Add {
  using source_pattern_type = Add<Add<SymbolicDimExpr, std::int64_t>, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Add<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Add<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    inner_lhs = Simplify(std::Move(inner_lhs));
    inner_rhs = SymbolicDimExpr(outter_rhs.Get<std::int64_t>() + inner_rhs.Get<std::int64_t>());
    return outter_lhs;
  }
};

struct RemoveRedundantConstantLhs_Add_Sub {
  using source_pattern_type = Add<Sub<SymbolicDimExpr, std::int64_t>, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Add<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Sub<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    if (outter_rhs.Get<std::int64_t>() == inner_rhs.Get<std::int64_t>()) {
      return Simplify(std::Move(inner_lhs));
    } else if (outter_rhs.Get<std::int64_t>() > inner_rhs.Get<std::int64_t>()) {
      std::int64_t ret_rhs = outter_rhs.Get<std::int64_t>() - inner_rhs.Get<std::int64_t>();
      outter_lhs = Simplify(std::Move(inner_lhs));
      outter_rhs = ret_rhs;
      return expr;
    } else if (outter_rhs.Get<std::int64_t>() < inner_rhs.Get<std::int64_t>()) {
      std::int64_t ret_rhs = inner_rhs.Get<std::int64_t>() - outter_rhs.Get<std::int64_t>();
      inner_lhs = Simplify(std::Move(inner_lhs));
      inner_rhs = SimplicDimExpr{ret_rhs};
      return outter_lhs;
    }
    LOG(FATAL) << "Dead code.";
  }
};

struct RemoveRedundantConstantLhs_Sub_Sub {
  using source_pattern_type = Sub<Sub<SymbolicDimExpr, std::int64_t>, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Sub<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Sub<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    inner_lhs = Simplify(std::Move(inner_lhs));
    inner_rhs = SymbolicDimExpr(outter_rhs.Get<std::int64_t>() + inner_rhs.Get<std::int64_t>());
    return outter_lhs;
  }
};

struct RemoveRedundantConstantLhs_Sub_Add {
  using source_pattern_type = Sub<Add<SymbolicDimExpr, std::int64_t>, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Sub<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Add<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    if (outter_rhs.Get<std::int64_t>() == inner_rhs.Get<std::int64_t>()) {
      return Simplify(std::Move(inner_lhs));
    } else if (outter_rhs.Get<std::int64_t>() > inner_rhs.Get<std::int64_t>()) {
      std::int64_t ret_rhs = outter_rhs.Get<std::int64_t>() - inner_rhs.Get<std::int64_t>();
      outter_lhs = Simplify(std::Move(inner_lhs));
      outter_rhs = ret_rhs;
      return expr;
    } else if (outter_rhs.Get<std::int64_t>() < inner_rhs.Get<std::int64_t>()) {
      std::int64_t ret_rhs = inner_rhs.Get<std::int64_t>() - outter_rhs.Get<std::int64_t>();
      inner_lhs = Simplify(std::Move(inner_lhs));
      inner_rhs = SimplicDimExpr{ret_rhs};
      return outter_lhs;
    }
    LOG(FATAL) << "Dead code.";
  }
};

struct RemoveRedundantConstantLhs_Mul_Mul {
  using source_pattern_type = Mul<Mul<SymbolicDimExpr, std::int64_t>, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Mul<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Mul<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    inner_lhs = Simplify(std::Move(inner_lhs));
    inner_rhs = SymbolicDimExpr(outter_rhs.Get<std::int64_t>() * inner_rhs.Get<std::int64_t>());
    return outter_lhs;
  }
};

struct RemoveRedundantConstantLhs_Mul_Div {
  using source_pattern_type = Mul<Div<SymbolicDimExpr, std::int64_t>, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Mul<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Div<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    if (outter_rhs.Get<std::int64_t>() == inner_rhs.Get<std::int64_t>()) {
      return Simplify(std::Move(inner_lhs));
    } else if (outter_rhs.Get<std::int64_t>() % inner_rhs.Get<std::int64_t>() == 0) {
      std::int64_t ret_rhs = outter_rhs.Get<std::int64_t>() / inner_rhs.Get<std::int64_t>();
      outter_lhs = Simplify(std::Move(inner_lhs));
      outter_rhs = ret_rhs;
      return expr;
    } else if (inner_rhs.Get<std::int64_t>() % outter_rhs.Get<std::int64_t>() == 0) {
      std::int64_t ret_rhs = inner_rhs.Get<std::int64_t>() / outter_rhs.Get<std::int64_t>();
      inner_lhs = Simplify(std::Move(inner_lhs));
      inner_rhs = SimplicDimExpr{ret_rhs};
      return outter_lhs;
    } else {
      return expr;
    }
    LOG(FATAL) << "Dead code.";
  }
};

struct RemoveRedundantConstantLhs_Div_Div {
  using source_pattern_type = Div<Div<SymbolicDimExpr, std::int64_t>, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Div<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Div<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    inner_lhs = Simplify(std::Move(inner_lhs));
    inner_rhs = SymbolicDimExpr(outter_rhs.Get<std::int64_t>() * inner_rhs.Get<std::int64_t>());
    return outter_lhs;
  }
};

struct RemoveRedundantConstantLhs_Div_Mul {
  using source_pattern_type = Div<Mul<SymbolicDimExpr, std::int64_t>, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(SymbolicDimExpr&& expr) {
    auto& [outter_lhs, outter_rhs] = *expr.Mut<Div<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    auto& [inner_lhs, inner_rhs] = *outter_lhs.Mut<Mul<SymbolicDimExpr, SymbolicDimExpr>>()->mut_tuple();
    if (outter_rhs.Get<std::int64_t>() == inner_rhs.Get<std::int64_t>()) {
      return Simplify(std::Move(inner_lhs));
    } else if (outter_rhs.Get<std::int64_t>() % inner_rhs.Get<std::int64_t>() == 0) {
      std::int64_t ret_rhs = outter_rhs.Get<std::int64_t>() / inner_rhs.Get<std::int64_t>();
      outter_lhs = Simplify(std::Move(inner_lhs));
      outter_rhs = ret_rhs;
      return expr;
    } else if (inner_rhs.Get<std::int64_t>() % outter_rhs.Get<std::int64_t>() == 0) {
      std::int64_t ret_rhs = inner_rhs.Get<std::int64_t>() / outter_rhs.Get<std::int64_t>();
      inner_lhs = Simplify(std::Move(inner_lhs));
      inner_rhs = SimplicDimExpr{ret_rhs};
      return outter_lhs;
    } else {
      return expr;
    }
    LOG(FATAL) << "Dead code.";
  }
};

SymbolicDimExpr Simplify(SymbolicDimExpr&& expr) {
  expr = TrySimplifyPass<FoldConstantAdd>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantSub>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantMul>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantDiv>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantBroadcastedDim>(std::move(expr));
  expr = TrySimplifyPass<SwapConstantLhs<Add>>(std::move(expr));
  expr = TrySimplifyPass<SwapConstantLhs<Sub>>(std::move(expr));
  expr = TrySimplifyPass<SwapConstantLhs<Mul>>(std::move(expr));
  expr = TrySimplifyPass<SwapConstantLhs<Div>>(std::move(expr));
  expr = TrySimplifyPass<SwapConstantLhs<BroadcastedDim>>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantUnit<Add, 0>>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantUnit<Sub, 0>>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantUnit<Mul, 1>>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantUnit<Div, 1>>(std::move(expr));
  expr = TrySimplifyPass<FoldConstantUnit<BroadcastedDim, 1>>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Add_Add>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Add_Sub>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Sub_Add>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Sub_Sub>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Mul_Mul>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Mul_Div>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Div_Mul>(std::move(expr));
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Div_Div>(std::move(expr));
  return expr;
}

}  // namespace

SymbolicDimExpr SimplifySymbolicDimExpr(SymbolicDimExpr&& expr) {
  return Simplify(expr);
}

}  // namespace cinn::adt
