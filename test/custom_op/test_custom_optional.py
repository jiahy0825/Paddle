# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from utils import extra_cc_args, extra_nvcc_args, paddle_includes

import paddle
from paddle import static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_optional\\custom_optional.pyd'.format(get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
custom_optional = load(
    name='custom_optional',
    sources=['custom_optional.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cflags
    extra_cuda_cflags=extra_nvcc_args,  # test for cflags
    verbose=True,
)


def optional_dynamic_add(phi_func, device, dtype, np_x, np_y):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    if np_y is not None:
        y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=False)
    else:
        y = x
    if phi_func:
        out = custom_optional.custom_add(x, y if np_y is not None else None)
    else:
        out = paddle.add(x, y)

    out.backward()
    return x.numpy(), out.numpy(), x.grad.numpy()


def optional_static_add(phi_func, device, dtype, np_x, np_y):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            x.stop_gradient = False
            if np_y is not None:
                y = static.data(
                    name="y", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y.stop_gradient = False
                feed_dict = {
                    "x": np_x.astype(dtype),
                    "y": np_y.astype(dtype),
                }
            else:
                y = x
                feed_dict = {
                    "x": np_x.astype(dtype),
                }
            if phi_func:
                out = custom_optional.custom_add(
                    x, y if np_y is not None else None
                )
            else:
                out = paddle.add(x, y)

            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            x_v, out_v, x_grad_v = exe.run(
                static.default_main_program(),
                feed=feed_dict,
                fetch_list=[
                    x.name,
                    out.name,
                    x.name + "@GRAD",
                ],
            )
    paddle.disable_static()
    return x_v, out_v, x_grad_v


'''
if (y) {
  outX = 2 * x + y;
  outY = x + y;
} else {
  outX = 2 * x;
  outY = None;
}
'''


def optional_inplace_dynamic_add(phi_func, device, dtype, np_x, np_y):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    if np_y is not None:
        y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=True)
        if phi_func:
            outx, outy = custom_optional.custom_optional_inplace_add(x, y)
        else:
            # We need to accumulate y's grad here.
            y.stop_gradient = False
            outx = 2 * x + y
            # Inplace leaf Tensor's stop_gradient should be True
            y.stop_gradient = True
            outy = y.add_(x)
    else:
        y = None
        if phi_func:
            outx, outy = custom_optional.custom_optional_inplace_add(x, y)
        else:
            outx = 2 * x
            outy = None
        assert (
            outy is None
        ), "The output `outy` of optional_inplace_dynamic_add should be None"

    out = outx + outy if outy is not None else outx
    out.backward()
    return (
        x.numpy(),
        outx.numpy(),
        y.numpy() if y is not None else None,
        outy.numpy() if outy is not None else None,
        out.numpy(),
        x.grad.numpy(),
        y.grad.numpy() if y is not None and y.grad is not None else None,
    )


def optional_inplace_static_add(phi_func, device, dtype, np_x, np_y):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            x.stop_gradient = False
            if np_y is not None:
                y = static.data(
                    name="y", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y.stop_gradient = False
                feed_dict = {
                    "x": np_x.astype(dtype),
                    "y": np_y.astype(dtype),
                }
                if phi_func:
                    outx, outy = custom_optional.custom_optional_inplace_add(
                        x, y
                    )
                else:
                    outx = 2 * x + y
                    outy = x + y
            else:
                feed_dict = {
                    "x": np_x.astype(dtype),
                }
                if phi_func:
                    outx, outy = custom_optional.custom_optional_inplace_add(
                        x, None
                    )
                else:
                    outx = 2 * x
                    outy = None
            out = outx + outy if outy is not None else outx
            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            if np_y is not None:
                x_v, out_v, x_grad_v, y_grad_v = exe.run(
                    static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=[
                        x.name,
                        out.name,
                        x.name + "@GRAD",
                        y.name + "@GRAD",
                    ],
                )
                paddle.disable_static()
                return [x_v, out_v, x_grad_v, y_grad_v]
            else:
                x_v, out_v, x_grad_v = exe.run(
                    static.default_main_program(),
                    feed=feed_dict,
                    fetch_list=[
                        x.name,
                        out.name,
                        x.name + "@GRAD",
                    ],
                )
                paddle.disable_static()
                return [x_v, out_v, x_grad_v]


def optional_vector_dynamic_add(phi_func, device, dtype, np_x, np_inputs):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    if np_inputs is not None:
        inputs = [
            paddle.to_tensor(np_input, dtype=dtype, stop_gradient=False)
            for np_input in np_inputs
        ]
        if phi_func:
            out = custom_optional.custom_add_vec(x, inputs)
        else:
            out = paddle.add(x, inputs[0])
            for input in inputs[1:]:
                out = paddle.add(out, input)
    else:
        if phi_func:
            out = custom_optional.custom_add_vec(x, None)
        else:
            out = paddle.add(x, x)

    out.backward()
    return x.numpy(), out.numpy(), x.grad.numpy()


def optional_vector_static_add(phi_func, device, dtype, np_x, np_inputs):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            x.stop_gradient = False
            feed_dict = {"x": np_x.astype(dtype)}
            if np_inputs is not None:
                y1 = static.data(
                    name="y1", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y1.stop_gradient = False
                y2 = static.data(
                    name="y2", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y2.stop_gradient = False
                feed_dict.update(
                    {
                        "y1": np_inputs[0].astype(dtype),
                        "y2": np_inputs[1].astype(dtype),
                    }
                )
                if phi_func:
                    out = custom_optional.custom_add_vec(x, [y1, y2])
                else:
                    out = paddle.add(x, y1)
                    out = paddle.add(out, y2)
            else:
                if phi_func:
                    out = custom_optional.custom_add_vec(x, None)
                else:
                    out = paddle.add(x, x)

            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            x_v, out_v, x_grad_v = exe.run(
                static.default_main_program(),
                feed=feed_dict,
                fetch_list=[
                    x.name,
                    out.name,
                    x.name + "@GRAD",
                ],
            )
    paddle.disable_static()
    return x_v, out_v, x_grad_v


class TestCustomOptionalJit(unittest.TestCase):
    def setUp(self):
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        self.np_x = np.random.random((3, 2)).astype("float32")
        self.np_y = np.random.random((3, 2)).astype("float32")
        self.np_inputs = [
            np.random.random((3, 2)).astype("float32"),
            np.random.random((3, 2)).astype("float32"),
        ]

    def check_output(self, out, pd_out, name):
        if out is None and pd_out is None:
            return
        assert out is not None, "out value of " + name + " is None"
        assert pd_out is not None, "pd_out value of " + name + " is None"
        np.testing.assert_array_equal(
            out,
            pd_out,
            err_msg='custom op {}: {},\n paddle api {}: {}'.format(
                name, out, name, pd_out
            ),
        )

    def check_output_allclose(self, out, pd_out, name):
        if out is None and pd_out is None:
            return
        assert out is not None, "out value of " + name + " is None"
        assert pd_out is not None, "pd_out value of " + name + " is None"
        np.testing.assert_allclose(
            out,
            pd_out,
            rtol=5e-5,
            atol=1e-2,
            err_msg='custom op {}: {},\n paddle api {}: {}'.format(
                name, out, name, pd_out
            ),
        )

    def test_optional_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_y]:
                    (pd_x, pd_out, pd_x_grad,) = optional_static_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (phi_x, phi_out, phi_x_grad,) = optional_static_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    self.check_output(phi_x, pd_x, "x")
                    self.check_output(phi_out, pd_out, "out")
                    self.check_output(phi_x_grad, pd_x_grad, "x_grad")

    def test_optional_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_y]:
                    (pd_x, pd_out, pd_x_grad,) = optional_dynamic_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (phi_x, phi_out, phi_x_grad,) = optional_dynamic_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    self.check_output(phi_x, pd_x, "x")
                    self.check_output(phi_out, pd_out, "out")
                    self.check_output(phi_x_grad, pd_x_grad, "x_grad")

    def test_optional_inplace_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_y]:
                    pd_tuple = optional_inplace_static_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    phi_tuple = optional_inplace_static_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    self.check_output(phi_tuple[0], pd_tuple[0], "x")
                    self.check_output(phi_tuple[1], pd_tuple[1], "out")
                    self.check_output(phi_tuple[2], pd_tuple[2], "x_grad")
                    if len(phi_tuple) > 3:
                        self.check_output(phi_tuple[3], pd_tuple[3], "y_grad")

    def test_optional_inplace_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_y]:
                    (
                        pd_x,
                        pd_outx,
                        pd_y,
                        pd_outy,
                        pd_out,
                        pd_x_grad,
                        pd_y_grad,
                    ) = optional_inplace_dynamic_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (
                        phi_x,
                        phi_outx,
                        phi_y,
                        phi_outy,
                        phi_out,
                        phi_x_grad,
                        phi_y_grad,
                    ) = optional_inplace_dynamic_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    self.check_output(pd_y, pd_outy, "inplace_pd_y")
                    self.check_output(phi_y, phi_outy, "inplace_phi_y")

                    self.check_output(phi_x, pd_x, "x")
                    self.check_output(phi_outx, pd_outx, "outx")
                    self.check_output(phi_y, pd_y, "y")
                    self.check_output(phi_outy, pd_outy, "outy")
                    self.check_output(phi_out, pd_out, "out")
                    self.check_output(phi_x_grad, pd_x_grad, "x_grad")
                    self.check_output(phi_y_grad, pd_y_grad, "y_grad")

    def test_optional_vector_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_inputs]:
                    (phi_x, phi_out, phi_x_grad,) = optional_vector_static_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (pd_x, pd_out, pd_x_grad,) = optional_vector_static_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    self.check_output(phi_x, pd_x, "x")
                    self.check_output(phi_out, pd_out, "out")
                    self.check_output(phi_x_grad, pd_x_grad, "x_grad")

    def test_optional_vector_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                for np_y in [None, self.np_inputs]:
                    (phi_x, phi_out, phi_x_grad,) = optional_vector_dynamic_add(
                        True,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )
                    (pd_x, pd_out, pd_x_grad,) = optional_vector_dynamic_add(
                        False,
                        device,
                        dtype,
                        self.np_x,
                        np_y,
                    )

                    self.check_output(phi_x, pd_x, "x")
                    self.check_output(phi_out, pd_out, "out")
                    self.check_output(phi_x_grad, pd_x_grad, "x_grad")


if __name__ == "__main__":
    unittest.main()
