# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import utils

import paddle
from paddle import nn


class CINNRMSNormSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.variance_epsilon = 1e-6
        self.hidden_size = 4096
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(self, hidden_states):
        variance = (hidden_states * hidden_states).sum(
            -1, keepdim=True
        ) / self.hidden_size
        hidden_states = (
            paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        )
        return hidden_states * self.weight


class PHIRMSNormSubGraphNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        return paddle.incubate.nn.functional.fused_rms_norm(
            x=hidden_states,
            norm_weight=self.weight,
            norm_bias=None,
            epsilon=self.variance_epsilon,
            begin_norm_axis=2,
        )


class TestCinnRMSNorm(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [1, 1, 4096]
        self.x = paddle.uniform(self.shape, dtype="float32", min=-0.5, max=0.5)
        self.x.stop_gradient = False

    def train(self, use_cinn):
        if use_cinn:
            net = CINNRMSNormSubGraphNet()
        else:
            net = PHIRMSNormSubGraphNet()
        # input_spec = [
        #     InputSpec(shape=[None, None, 4096], dtype="float32"),
        # ]
        # net = utils.apply_to_static(net, use_cinn, input_spec)
        net = utils.apply_to_static(net, use_cinn)
        net.eval()
        self.x.stop_gradient = False
        for i in range(10000):
            out = net(self.x)
        return out

    def test_train(self):
        cinn_out = self.train(use_cinn=True)

        dy_out = self.train(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == '__main__':
    unittest.main()
