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

#include "paddle/cinn/adt/symbolic_dim_expr_simplifier.h"

#include <numeric>

namespace cinn::adt {

namespace {

SymbolicDimExpr Simplify(SymbolicDimExpr expr);

template <typename T>
SymbolicDimExpr TrySimplifyPass(const SymbolicDimExpr& expr) {
  if (cinn::adt::Match<typename T::source_pattern_type>(expr)) {
    return T().MatchAndRewrite(expr);
  } else {
    return expr;
  }
}

struct FoldConstantAdd {
  using source_pattern_type = Add<std::int64_t, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [lhs, rhs] =
        expr.Get<Add<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    return lhs.Get<std::int64_t>() + rhs.Get<std::int64_t>();
  }
};

struct FoldConstantMul {
  using source_pattern_type = Mul<std::int64_t, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [lhs, rhs] =
        expr.Get<Mul<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    return lhs.Get<std::int64_t>() * rhs.Get<std::int64_t>();
  }
};

struct FoldConstantBroadcastedDim {
  using source_pattern_type = BroadcastedDim<std::int64_t, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [lhs, rhs] =
        expr.Get<BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
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

struct EraseRedundantBroadcastedDim {
  using source_pattern_type = BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [lhs, rhs] =
        expr.Get<BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    if (lhs == rhs) {
      return lhs;
    } else {
      return expr;
    }
  }
};

template <template <typename, typename> class Op>
struct SwapConstantLhs {
  using source_pattern_type = Op<std::int64_t, SymbolicDimExpr>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [lhs, rhs] =
        expr.Get<Op<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    return Op<SymbolicDimExpr, SymbolicDimExpr>{rhs, lhs};
  }
};

template <template <typename, typename> class Op, int unit>
struct FoldConstantUnit {
  using source_pattern_type = Op<SymbolicDimExpr, std::int64_t>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [lhs, rhs] =
        expr.Get<Op<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    if (rhs.template Get<std::int64_t>() == unit) {
      return lhs;
    } else {
      return expr;
    }
    LOG(FATAL) << "Dead code.";
  }
};

using ConstIntegerPattern = Union<std::int64_t, Negative<std::int64_t>>;

std::int64_t GetInteger(const SymbolicDimExpr& expr) {
  if (expr.Has<Negative<SymbolicDimExpr>>()) {
    const auto& [integer] = expr.Get<Negative<SymbolicDimExpr>>().tuple();
    CHECK(integer.Has<std::int64_t>());
    return -integer.Get<std::int64_t>();
  }
  CHECK(expr.Has<std::int64_t>());
  return expr.Get<std::int64_t>();
}

struct RemoveRedundantConstantLhs_Add_Add {
  using source_pattern_type =
      Add<Add<SymbolicDimExpr, ConstIntegerPattern>, ConstIntegerPattern>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [outter_lhs, outter_rhs] =
        expr.Get<Add<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    const auto& [inner_lhs, inner_rhs] =
        outter_lhs.Get<Add<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    const auto& new_inner_lhs = Simplify(inner_lhs);
    const auto& new_inner_rhs =
        SymbolicDimExpr(GetInteger(outter_rhs) + GetInteger(inner_rhs));
    return Add<SymbolicDimExpr, SymbolicDimExpr>{new_inner_lhs, new_inner_rhs};
  }
};

using ConstRationalPattern = Union<std::int64_t,
                                   Reciprocal<std::int64_t>,
                                   Mul<std::int64_t, Reciprocal<std::int64_t>>>;

using ConstRational = std::pair<std::int64_t, std::int64_t>;

template <typename T>
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
  return ConstRational{num / gcd, dem / gcd};
}

ConstRational GetConstRationalImpl(
    const Mul<SymbolicDimExpr, SymbolicDimExpr>& value) {
  const auto& [numerator, reciprocal] = value.tuple();
  const auto& [denominator] =
      reciprocal.Get<Reciprocal<SymbolicDimExpr>>().tuple();
  return SimplifiedConstRational(numerator.Get<std::int64_t>(),
                                 denominator.Get<std::int64_t>());
}

ConstRational GetConstRational(const SymbolicDimExpr& expr) {
  return std::visit(
      [&](const auto& impl) { return GetConstRationalImpl(impl); },
      expr.variant());
}

ConstRational MulConstRational(const ConstRational& lhs,
                               const ConstRational& rhs) {
  const auto [lhs_num, lhs_dem] = lhs;
  const auto [rhs_num, rhs_dem] = rhs;
  // Crossing is correct.
  const auto [simplifed_lhs_num, simplifed_rhs_dem] =
      SimplifiedConstRational(lhs_num, rhs_dem);
  const auto [simplifed_rhs_num, simplifed_lhs_dem] =
      SimplifiedConstRational(rhs_num, lhs_dem);
  return ConstRational{simplifed_lhs_num * simplifed_rhs_num,
                       simplifed_lhs_dem * simplifed_rhs_dem};
}

struct RemoveRedundantConstantLhs_Mul_Mul {
  using source_pattern_type =
      Mul<Mul<SymbolicDimExpr, ConstIntegerPattern>, ConstIntegerPattern>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [outter_lhs, outter_rhs] =
        expr.Get<Mul<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    const auto& [inner_lhs, inner_rhs] =
        outter_lhs.Get<Mul<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    const auto& new_inner_lhs = Simplify(inner_lhs);
    const auto [num, dem] = MulConstRational(GetConstRational(outter_rhs),
                                             GetConstRational(inner_rhs));
    if (num == 1) {
      if (dem == 1) {
        return inner_lhs;
      } else {
        const auto& new_inner_rhs =
            Reciprocal<SymbolicDimExpr>{SymbolicDimExpr{dem}};
        return Mul<SymbolicDimExpr, SymbolicDimExpr>{new_inner_lhs,
                                                     new_inner_rhs};
      }
    } else {
      if (dem == 1) {
        const auto& new_inner_rhs = SymbolicDimExpr{num};
        return Mul<SymbolicDimExpr, SymbolicDimExpr>{new_inner_lhs,
                                                     new_inner_rhs};
      } else {
        const auto& new_inner_rhs = SymbolicDimExpr{num} / SymbolicDimExpr{dem};
        return Mul<SymbolicDimExpr, SymbolicDimExpr>{new_inner_lhs,
                                                     new_inner_rhs};
      }
    }
    LOG(FATAL) << "Dead code";
  }
};

struct RemoveRedundantConstantLhs_BD_LeftBD {
  using source_pattern_type =
      BroadcastedDim<BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>,
                     SymbolicDimExpr>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [outter_lhs, outter_rhs] =
        expr.Get<BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    const auto& [inner_lhs, inner_rhs] =
        outter_lhs.Get<BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>>()
            .tuple();
    if (outter_rhs == inner_lhs) {
      return Simplify(outter_lhs);
    } else if (outter_rhs == inner_rhs) {
      return Simplify(outter_lhs);
    } else {
      return expr;
    }
    LOG(FATAL) << "Dead code";
  }
};

struct RemoveRedundantConstantLhs_BD_RightBD {
  using source_pattern_type =
      BroadcastedDim<SymbolicDimExpr,
                     BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>>;

  SymbolicDimExpr MatchAndRewrite(const SymbolicDimExpr& expr) {
    const auto& [outter_lhs, outter_rhs] =
        expr.Get<BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>>().tuple();
    const auto& [inner_lhs, inner_rhs] =
        outter_rhs.Get<BroadcastedDim<SymbolicDimExpr, SymbolicDimExpr>>()
            .tuple();
    if (outter_lhs == inner_lhs) {
      return Simplify(outter_rhs);
    } else if (outter_lhs == inner_rhs) {
      return Simplify(outter_rhs);
    } else {
      return expr;
    }
    LOG(FATAL) << "Dead code";
  }
};

SymbolicDimExpr Simplify(SymbolicDimExpr expr) {
  expr = TrySimplifyPass<FoldConstantAdd>(expr);
  expr = TrySimplifyPass<FoldConstantMul>(expr);
  expr = TrySimplifyPass<FoldConstantBroadcastedDim>(expr);
  expr = TrySimplifyPass<EraseRedundantBroadcastedDim>(expr);
  expr = TrySimplifyPass<SwapConstantLhs<Add>>(expr);
  expr = TrySimplifyPass<SwapConstantLhs<Mul>>(expr);
  expr = TrySimplifyPass<SwapConstantLhs<BroadcastedDim>>(expr);
  expr = TrySimplifyPass<FoldConstantUnit<Add, 0>>(expr);
  expr = TrySimplifyPass<FoldConstantUnit<Mul, 1>>(expr);
  expr = TrySimplifyPass<FoldConstantUnit<BroadcastedDim, 1>>(expr);
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Add_Add>(expr);
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_Mul_Mul>(expr);
  expr = TrySimplifyPass<RemoveRedundantConstantLhs_BD_LeftBD>(expr);
  return expr;
}

}  // namespace

SymbolicDimExpr SimplifySymbolicDimExpr(const SymbolicDimExpr& expr) {
  return Simplify(expr);
}

}  // namespace cinn::adt
