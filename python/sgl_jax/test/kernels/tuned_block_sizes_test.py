# Copyright 2026 Google LLC
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

from absl.testing import absltest, parameterized

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.tuned_block_sizes import (
    get_default_gmm_block_sizes,
    round_up_to_multiple_of_128_within_limit,
)


class RoundUpHeuristicTest(parameterized.TestCase):
    """Pure-Python tests for the GMM tile-size heuristic.

    The heuristic must (a) preserve legacy behaviour when `divides` is None,
    and (b) only ever return a value that divides the target when `divides`
    is set. The latter is required by `_calculate_num_tiles` in gmm.py
    (`m % tm == 0`); a regression there crashed MiMo V2 Pro EP-MoE on
    v6e/v7x with m=4096, num_current_groups=24 -> tm=384.
    """

    def test_divides_none_preserves_legacy(self):
        # x <= 128 -> 128
        self.assertEqual(round_up_to_multiple_of_128_within_limit(64, 512), 128)
        self.assertEqual(round_up_to_multiple_of_128_within_limit(128, 512), 128)
        # x < limit -> ceil to nearest 128
        self.assertEqual(round_up_to_multiple_of_128_within_limit(341, 512), 384)
        self.assertEqual(round_up_to_multiple_of_128_within_limit(129, 512), 256)
        # x >= limit -> largest 128-multiple >= 512 dividing x
        self.assertEqual(round_up_to_multiple_of_128_within_limit(2048, 2048), 2048)
        self.assertEqual(round_up_to_multiple_of_128_within_limit(8192, 2048), 2048)

    def test_divides_walks_down_to_valid_tile(self):
        # The MiMo V2 Pro regression case: 2*m/g = 341, naive ceil -> 384,
        # but 4096 % 384 != 0. With divides=4096 we must walk down to 256.
        self.assertEqual(
            round_up_to_multiple_of_128_within_limit(341, 512, divides=4096),
            256,
        )

    def test_divides_no_op_when_already_divisor(self):
        # 256 already divides 4096; no decrement should happen.
        self.assertEqual(
            round_up_to_multiple_of_128_within_limit(200, 512, divides=4096),
            256,
        )

    def test_divides_floor_at_128(self):
        # 128 always divides any sane m (powers of 2). The decrement loop
        # must stop at 128 even if the target somehow forbade larger tiles.
        result = round_up_to_multiple_of_128_within_limit(341, 512, divides=128)
        self.assertEqual(result, 128)

    @parameterized.named_parameters(
        # (m, g) -> heuristic must return tm s.t. m % tm == 0
        ("mimo_v2_pro_v7x", 4096, 24),  # the actual regression
        ("m4k_g32", 4096, 32),
        ("m8k_g24", 8192, 24),
        ("m16k_g32", 16384, 32),
        ("m1k_g8", 1024, 8),
        ("m512_g16", 512, 16),
        ("m4k_g384", 4096, 384),  # all experts on one device (degenerate)
    )
    def test_get_default_gmm_block_sizes_tm_divides_m(self, m, g):
        tm, _, _ = get_default_gmm_block_sizes(m=m, k=4096, n=4096, g=g)
        self.assertEqual(
            m % tm,
            0,
            msg=f"tm={tm} does not divide m={m} (g={g})",
        )
        self.assertGreaterEqual(tm, 128)
        self.assertEqual(tm % 128, 0, msg=f"tm={tm} is not a multiple of 128")

    def test_mimo_v2_pro_no_longer_picks_384(self):
        # Tight regression assertion: this exact shape used to return tm=384.
        tm, _, _ = get_default_gmm_block_sizes(m=4096, k=4096, n=4096, g=24)
        self.assertNotEqual(tm, 384, msg="heuristic regressed to broken tm=384")
        self.assertEqual(tm, 256)


if __name__ == "__main__":
    absltest.main()
