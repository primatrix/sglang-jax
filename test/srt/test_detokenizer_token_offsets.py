"""Regression tests for incremental output IDs; no TPU/server is required."""

import unittest
from types import SimpleNamespace

from sgl_jax.srt.managers.detokenizer_manager import DetokenizerManager


class ByteTokenizer:
    all_special_ids = [1000]

    def batch_decode(self, rows, skip_special_tokens, **kwargs):
        return [bytes(x for x in row if x != 1000).decode("utf-8", errors="replace")
                for row in rows]


class Batch(SimpleNamespace):
    def __getattr__(self, name):
        # Unused logprob, hidden-state and routing metadata.
        return None


class TestDetokenizerTokenOffsets(unittest.TestCase):
    def setUp(self):
        self.manager = object.__new__(DetokenizerManager)
        self.manager.tokenizer = ByteTokenizer()
        self.manager.decode_status = {}

    def send(self, ids, *, rid="r", prefix=(), finished=None, skip=True, no_trim=False):
        batch = Batch(rids=[rid], decoded_texts=[""],
                      decode_ids=[list(prefix) + list(ids)], read_offsets=[len(prefix)],
                      finished_reasons=[finished], no_stop_trim=[no_trim],
                      skip_special_tokens=[skip], spaces_between_special_tokens=[False])
        return self.manager.handle_batch_token_id_out(batch)

    def test_long_incomplete_utf8_suffix_is_not_resent(self):
        total = 0
        for _ in range(3000):
            result = self.send([0xE4])
            self.assertEqual(result.output_ids, [[0xE4]])
            total += len(result.output_ids[0])
        self.assertEqual(total, 3000)

    def test_utf8_completion_does_not_repeat_ids(self):
        output = [self.send([x]) for x in (0xE4, 0xB8, 0xAD)]
        self.assertEqual([r.output_ids[0] for r in output], [[0xE4], [0xB8], [0xAD]])

    def test_printable_text_path_is_unchanged(self):
        self.assertEqual(self.send([65]).output_strs, ["A"])
        self.assertEqual(self.send([0xE4, 0xB8, 0xAD]).output_strs, ["中"])

    def test_prompt_surrounding_tokens_are_not_output(self):
        self.assertEqual(self.send([65], prefix=[80, 81]).output_ids, [[65]])
        self.assertEqual(self.send([66]).output_ids, [[66]])

    def test_empty_update_does_not_repeat_pending_ids(self):
        self.send([0xE4])
        self.assertEqual(self.send([]).output_ids, [[]])

    def test_final_flush_does_not_repeat_pending_ids(self):
        self.send([0xE4])
        result = self.send([0xB8, 0xAD], finished={"type": "length"})
        self.assertEqual(result.output_ids, [[0xB8, 0xAD]])

    def test_special_token_filtering_still_advances_offset(self):
        self.assertEqual(self.send([1000]).output_ids, [[]])
        self.assertEqual(self.send([65]).output_ids, [[65]])

    def test_unfiltered_empty_text_token_is_sent_once(self):
        self.assertEqual(self.send([1000], skip=False).output_ids, [[1000]])
        self.assertEqual(self.send([65], skip=False).output_ids, [[65]])

    def test_stop_token_is_trimmed(self):
        self.send([65])
        result = self.send([66, 33], finished={"matched": 33})
        self.assertEqual(result.output_ids, [[66]])

    def test_no_stop_trim_preserves_stop_token(self):
        self.send([65])
        result = self.send([66, 33], finished={"matched": 33}, no_trim=True)
        self.assertEqual(result.output_ids, [[66, 33]])

    def test_interleaved_requests_have_independent_offsets(self):
        self.send([0xE4], rid="a")
        self.send([65], rid="b")
        self.assertEqual(self.send([0xB8], rid="a").output_ids, [[0xB8]])
        self.assertEqual(self.send([66], rid="b").output_ids, [[66]])


if __name__ == "__main__":
    unittest.main()
