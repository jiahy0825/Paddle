#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..utils.hybrid_parallel_util import (
    broadcast_dp_parameters,
    broadcast_input_data,
    broadcast_mp_parameters,
    broadcast_sep_parameters,
    broadcast_sharding_parameters,
)
from ..utils.log_util import logger
from .meta_parallel_base import MetaParallelBase

__all__ = []


class TensorParallel(MetaParallelBase):
    def __init__(self, layers, hcg, **kwargs):
        super().__init__(layers, hcg, **kwargs)

    def _prepare_for_model(self):
        logger.info("start broadcast mp parameters")
        broadcast_mp_parameters(self._layers, self._hcg)

        if self._hcg.get_sep_parallel_world_size() > 1:
            logger.info("start broadcast sep parameters")
            broadcast_sep_parameters(self._layers, self._hcg)

        if self._hcg.get_sharding_parallel_world_size() > 1:
            logger.info("start broadcast sharding parameters")
            broadcast_sharding_parameters(self._layers, self._hcg)

        if self._hcg.get_data_parallel_world_size() > 1:
            logger.info("start broadcast dp parameters")
            broadcast_dp_parameters(self._layers, self._hcg)

        logger.info("mp's parameters is ready")

    def _pre_forward(self, *inputs, **kwargs):
        mp_configs = self._strategy.hybrid_configs["mp_configs"]
        need_broadcast_data = mp_configs.need_broadcast_data
        if need_broadcast_data:
            logger.debug("mp start broadcast input data")
            return broadcast_input_data(self._hcg, *inputs, **kwargs)
