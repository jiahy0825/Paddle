#pragma once

#include "paddle/cinn/adt/symbolic_dim_expr_match_trait.h"

namespace cinn::adt {

SymbolicDimExpr SimplifySymbolicDimExpr(SymbolicDimExpr&& SymbolicDimExpr);

}
