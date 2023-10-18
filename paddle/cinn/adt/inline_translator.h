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
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/tree.h"

namespace cinn::adt {

template <typename OutT, typename InT>
class Store final : public Tuple<OutT, InT> {
 public:
  using Tuple<OutT, InT>::Tuple;
};

template <typename T>
class Load final : public Tuple<T> {
 public:
  using Tuple<T>::Tuple;
};

template <template <typename> class MapT>
struct InlineTranslatorTrait;

template <>
struct InlineTranslatorTrait<MapStmt> final {
  template <typename T>
  static List<T> GetTreeInnerNodeChildren(const MapStmt<T>& map_stmt) {
    const auto& [iterators, stmts] = map_stmt.tuple();
    return stmts;
  }

  template <typename SrcTreeT, typename DstTreeT>
  static MapStmt<DstTreeT> ConvertMap(const MapStmt<SrcTreeT>& src_map,
                                      const List<DstTreeT>& dst_children) {
    const auto& [iterators, src_children] = src_map.tuple();
    return MapStmt<DstTreeT>{iterators, dst_children};
  }
};

template <template <typename> class MapT,
          template <typename>
          class OpCallT,
          typename TensorT>
struct InlineTranslator final {
  using SrcLeaf = Store<TensorT, OpCallT<Load<TensorT>>>;
  using DstLeaf = Store<TensorT, Tree<OpCallT, Load<TensorT>>>;
  using SrcTree = Tree<MapT, SrcLeaf>;
  using DstTree = Tree<MapT, DstLeaf>;

  static DstTree Call(const SrcTree& src_tree) {
    CHECK(src_tree.Has<MapT<SrcTree>>());
    const MapT<DstTree> dst_tree = Call(src_tree.Get<MapT<SrcTree>>());

    return DstTree{dst_tree};
  }

 private:
  static MapT<DstTree> Call(const MapT<SrcTree>& src_map) {
    const List<SrcTree> src_children =
        InlineTranslatorTrait<MapT>::GetTreeInnerNodeChildren(src_map);
    const List<DstTree> dst_children = Call(src_children);
    return InlineTranslatorTrait<MapT>::ConvertMap(src_map, dst_children);
  }

  static List<DstTree> Call(const List<SrcTree>& src_children) {
    List<DstTree> ret{};

    VisitEachContiguousSegment(
        src_children, [&](int start, int end, bool is_leaf) {
          if (is_leaf) {
            for (int i = start; i < end; ++i) {
              ret->emplace_back(Call(src_children->at(i)));
            }
          } else {
            const auto& converted = TranslateContiguousLeaves(
                std::next(src_children->begin(), start),
                std::next(src_children->begin(), end));
            ret->insert(ret->end(), converted->begin(), converted->end());
          }
        });

    return ret;
  }

  struct ConsumerPos {
    int leaf_index;
    int arg_index;
  };

  // using DstLeaf = Store<TensorT, Tree<OpCallT, Load<TensorT>>>;
  static DstLeaf UpdateConsumerArg(const DstLeaf& consumer,
                                   int arg_index,
                                   const DstLeaf& producer) {
    const auto& [consumer_tensor, consumer_tree] = consumer.tuple();
    CheckConsumerPosIsLoadTensor(consumer, arg_index);
    const auto& op_call =
        consumer_tree.Get<OpCallT<Tree<OpCallT, Load<TensorT>>>>();
    const auto& op_call_children =
        InlineTranslatorTrait<OpCallT>::GetTreeInnerNodeChildren(op_call);
    const auto& ret_op_call_children =
        UpdateConsumerArg(op_call_children, arg_index, producer);
    const auto& ret_op_call = InlineTranslatorTrait<OpCallT>::ConvertMap(
        op_call, ret_op_call_children);
    Tree<OpCallT, Load<TensorT>> ret_op_call_tree = ret_op_call;
    return DstLeaf{consumer_tensor, ret_op_call_tree};
  }

  static List<DstTree> UpdateConsumerArg(const List<DstTree>& op_call_children,
                                         int arg_index,
                                         const DstLeaf& producer) {
    const auto& [producer_tensor, producer_tree] = producer.tuple();
    const auto& arg = op_call_children.at(arg_index);
    const auto& arg_leaf = arg.Get<Load<TensorT>>();
    const auto& [arg_tensor] = arg_leaf.tuple();
    CHECK(producer_tensor == arg_tensor);
    List<DstTree> ret{};
    ret->assign(op_call_children->begin(), op_call_children->end());
    ret->at(arg_index) = producer_tree;
    return ret;
  }

  static void CheckConsumerPosIsLoadTensor(const DstLeaf& consumer,
                                           int arg_index) {
    ADT_TODO();
  }

  template <typename SrcTreeIterT>
  static std::vector<std::vector<ConsumerPos>> MakeProducerIndex2ConsumerPos(
      SrcTreeIterT begin, SrcTreeIterT end) {
    ADT_TODO();
  }

  template <typename SrcTreeIterT>
  static List<DstTree> TranslateContiguousLeaves(SrcTreeIterT begin,
                                                 SrcTreeIterT end) {
    int size = end - begin;
    const auto producer_idx2consumer_pos =
        MakeProducerIndex2ConsumerPos(begin, end);
    const auto& GetConsumerPos4ProducerIndex =
        [&](int index) -> std::vector<ConsumerPos> {
      return producer_idx2consumer_pos.at(index);
    };
    std::unordered_map<int, DstLeaf> index2dst_leaf{};
    // Init dst leaves
    for (int i = 0; i < size; ++i) {
      CHECK(index2dst_leaf.emplace(i, NaiveTranslateLeaf(*std::next(begin, i)))
                .second);
    }
    // Inline dst leaves
    for (int producer_i = 0; producer_i < size; ++producer_i) {
      const auto& consumer_positions = GetConsumerPos4ProducerIndex(producer_i);
      if (consumer_positions.empty()) {
        // Do nothing
      } else {
        DstLeaf producer = index2dst_leaf.at(producer_i);
        for (const auto& consumer_pos : consumer_positions) {
          DstLeaf consumer = index2dst_leaf.at(consumer_pos.leaf_index);
          index2dst_leaf.at(consumer_pos.leaf_index) =
              UpdateConsumerArg(consumer, consumer_pos.arg_index, producer);
        }
        index2dst_leaf.erase(producer_i);
      }
    }
    // Collect inlined leaves
    List<DstTree> ret{};
    for (int i = 0; i < size; ++i) {
      const auto& iter = index2dst_leaf.find(i);
      if (iter != index2dst_leaf.end()) {
        ret->emplace_back(iter->second);
      }
    }
    return ret;
  }

  // using SrcLeaf = Store<TensorT, OpCallT<Load<TensorT>>>;
  // using DstLeaf = Store<TensorT, Tree<OpCallT, Load<TensorT>>>;
  static DstLeaf NaiveTranslateLeaf(const SrcLeaf& src_leaf) {
    const auto& [tensor, op_call] = src_leaf.tuple();
    const List < Load < TensorT >>> & src_loads =
        InlineTranslatorTrait<OpCallT>::GetTreeInnerNodeChildren(op_call);
    List<Tree<OpCallT, Load<TensorT>>> dst_loads{};
    for (const auto& src_load : *src_loads) {
      dst_loads->emplace_back(src_load);
    }
    OpCallT<Tree<OpCallT, Load<TensorT>>> dst_op_call =
        InlineTranslatorTrait<OpCallT>::ConvertMap(op_call, dst_loads);
    Tree<OpCallT, Load<TensorT>> dst_op_call_tree = dst_op_call;
    return DstLeaf{tensor, dst_op_call_tree};
  }

  template <typename DoEachT /*void(&)(int start, int end, bool is_leaf)*/>
  static void VisitEachContiguousSegment(const List<SrcTree>& src_children,
                                         const DoEachT& DoEach) {
    std::vector<int> child_index2is_leaf(src_children->size(), 0);
    for (int i = 0; i < src_children->size(); ++i) {
      child_index2is_leaf.at(i) = src_children->at(i).template Has<SrcLeaf>();
    }
    int start = 0;
    for (int i = 1; i < child_index2is_leaf.size(); ++i) {
      if (child_index2is_leaf.at(i - 1) == child_index2is_leaf.at(i)) {
        DoEach(start, i, child_index2is_leaf.at(i - 1));
        start = i;
      } else {
        // Do nothing
      }
    }
    if (start != child_index2is_leaf.size()) {
      DoEach(start, child_index2is_leaf.size(), child_index2is_leaf.back());
    }
  }
};

}  // namespace cinn::adt
