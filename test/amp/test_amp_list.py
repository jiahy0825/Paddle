# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import core
from paddle.static.amp import fp16_lists
from paddle.static.amp.fp16_lists import AutoMixedPrecisionLists


class TestAMPList(unittest.TestCase):
    def test_main(self):
        custom_white_list = [
            'lookup_table',
            'lookup_table_v2',
        ]
        amp_list = AutoMixedPrecisionLists(custom_white_list=custom_white_list)
        for op in custom_white_list:
            self.assertTrue(op in amp_list.white_list)
            self.assertTrue(op not in amp_list.black_list)
            self.assertTrue(op not in amp_list.unsupported_list)

        default_black_list = [
            'linear_interp_v2',
            'nearest_interp_v2',
            'bilinear_interp_v2',
            'bicubic_interp_v2',
            'trilinear_interp_v2',
        ]
        for op in default_black_list:
            self.assertTrue(op in amp_list.black_list)

    def test_apis(self):
        def _run_check_dtype():
            fp16_lists.check_amp_dtype(dtype="int64")

        self.assertRaises(ValueError, _run_check_dtype)

        for vartype in [core.VarDesc.VarType.FP16, core.VarDesc.VarType.BF16]:
            self.assertEqual(
                fp16_lists.get_low_precision_vartype(vartype), vartype
            )
        self.assertEqual(
            fp16_lists.get_low_precision_vartype("float16"),
            core.VarDesc.VarType.FP16,
        )
        self.assertEqual(
            fp16_lists.get_low_precision_vartype("bfloat16"),
            core.VarDesc.VarType.BF16,
        )

        def _run_get_vartype():
            fp16_lists.get_low_precision_vartype(dtype="int64")

        self.assertRaises(ValueError, _run_get_vartype)

        for dtype in ["float16", "bfloat16"]:
            self.assertEqual(
                fp16_lists.get_low_precision_dtypestr(dtype), dtype
            )
        self.assertEqual(
            fp16_lists.get_low_precision_dtypestr(core.VarDesc.VarType.FP16),
            "float16",
        )
        self.assertEqual(
            fp16_lists.get_low_precision_dtypestr(core.VarDesc.VarType.BF16),
            "bfloat16",
        )

        def _run_get_dtypestr():
            fp16_lists.get_low_precision_dtypestr(dtype="int64")

        self.assertRaises(ValueError, _run_get_dtypestr)


if __name__ == "__main__":
    unittest.main()
