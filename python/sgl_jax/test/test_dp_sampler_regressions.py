import base64
import importlib.util
import sys
import types
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

if importlib.util.find_spec("llguidance") is None:
    llguidance_stub = types.ModuleType("llguidance")
    llguidance_stub.LLMatcher = object
    llguidance_stub.LLInterpreter = object
    llguidance_stub.LLTokenizer = object
    llguidance_stub.StructTag = object
    llguidance_stub.grammar_from = lambda *args, **kwargs: None
    sys.modules["llguidance"] = llguidance_stub

if importlib.util.find_spec("pybase64") is None:
    pybase64_stub = types.ModuleType("pybase64")
    pybase64_stub.b64decode = base64.b64decode
    sys.modules["pybase64"] = pybase64_stub

if importlib.util.find_spec("pathwaysutils") is None:
    sys.modules["pathwaysutils"] = types.ModuleType("pathwaysutils")

if importlib.util.find_spec("setproctitle") is None:
    setproctitle_stub = types.ModuleType("setproctitle")
    setproctitle_stub.setproctitle = lambda *args, **kwargs: None
    sys.modules["setproctitle"] = setproctitle_stub

from sgl_jax.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.managers.schedule_batch import (
    FINISH_ABORT,
    ModelWorkerSamplingInfo,
    ScheduleBatch,
    ScheduleReqsInfo,
)
from sgl_jax.srt.managers.scheduler import GenerationBatchResult, Scheduler
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


class TestDPSamplerRegressions(unittest.TestCase):
    def _sampling_info(
        self,
        *,
        batch_size: int,
        need_min_p_sampling: bool = False,
        min_p: float = 0.0,
        vocab_size: int = 16,
        linear_penalty: np.ndarray | None = None,
    ) -> SamplingBatchInfo:
        return SamplingBatchInfo(
            temperatures=np.ones((batch_size, 1), dtype=np.float32),
            top_ps=np.ones(batch_size, dtype=np.float32),
            top_ks=np.ones(batch_size, dtype=np.int32),
            min_ps=np.full(batch_size, min_p, dtype=np.float32),
            vocab_size=vocab_size,
            is_all_greedy=False,
            need_min_p_sampling=need_min_p_sampling,
            linear_penalty=linear_penalty,
        )

    def test_merge_sampling_info_preserves_min_p_flag(self):
        batch = ScheduleBatch(
            dp_size=2,
            has_grammar=False,
            reqs_info=[
                ScheduleReqsInfo(
                    reqs=[],
                    seq_lens=np.array([], dtype=np.int32),
                    sampling_info=None,
                ),
                ScheduleReqsInfo(
                    reqs=[],
                    seq_lens=np.array([3], dtype=np.int32),
                    sampling_info=self._sampling_info(
                        batch_size=1,
                        need_min_p_sampling=True,
                        min_p=0.2,
                    ),
                ),
            ],
        )

        merged = batch._merge_sampling_info(per_dp_bs_size=2, total_bs=4)

        self.assertTrue(merged.need_min_p_sampling)
        np.testing.assert_array_equal(
            merged.min_ps,
            np.array([0.0, 0.0, 0.2, 0.0], dtype=np.float32),
        )

    def test_merge_sampling_info_preserves_linear_penalty_layout(self):
        batch = ScheduleBatch(
            dp_size=2,
            has_grammar=False,
            reqs_info=[
                ScheduleReqsInfo(
                    reqs=[],
                    seq_lens=np.array([3, 2], dtype=np.int32),
                    sampling_info=self._sampling_info(
                        batch_size=2,
                        vocab_size=3,
                        linear_penalty=np.array(
                            [
                                [-1.0, 0.0, 0.0],
                                [0.0, -2.0, 0.0],
                            ],
                            dtype=np.float32,
                        ),
                    ),
                ),
                ScheduleReqsInfo(
                    reqs=[],
                    seq_lens=np.array([4], dtype=np.int32),
                    sampling_info=self._sampling_info(
                        batch_size=1,
                        vocab_size=3,
                        linear_penalty=np.array([[0.0, 0.0, -3.0]], dtype=np.float32),
                    ),
                ),
            ],
        )

        merged = batch._merge_sampling_info(per_dp_bs_size=3, total_bs=6)

        expected = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -3.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(merged.linear_penalty, expected)

    def test_merge_sampling_info_materializes_penalty_orchestrator(self):
        class FakePenalizerOrchestrator:
            is_required = True

            def apply(self):
                return np.array([[0.0, -4.0, 0.0]], dtype=np.float32)

        batch = ScheduleBatch(
            dp_size=1,
            has_grammar=False,
            reqs_info=[
                ScheduleReqsInfo(
                    reqs=[],
                    seq_lens=np.array([3], dtype=np.int32),
                    sampling_info=SamplingBatchInfo(
                        temperatures=np.ones((1, 1), dtype=np.float32),
                        top_ps=np.ones(1, dtype=np.float32),
                        top_ks=np.ones(1, dtype=np.int32),
                        min_ps=np.zeros(1, dtype=np.float32),
                        vocab_size=3,
                        is_all_greedy=False,
                        penalizer_orchestrator=FakePenalizerOrchestrator(),
                    ),
                ),
            ],
        )

        merged = batch._merge_sampling_info(per_dp_bs_size=2, total_bs=2)

        np.testing.assert_array_equal(
            merged.linear_penalty,
            np.array(
                [
                    [0.0, -4.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_model_worker_sampling_info_update_penalties_preserves_linear_penalty(self):
        linear_penalty = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        sampling_info = ModelWorkerSamplingInfo(
            temperatures=np.ones((1, 1), dtype=np.float32),
            top_ps=np.ones(1, dtype=np.float32),
            top_ks=np.ones(1, dtype=np.int32),
            min_ps=np.zeros(1, dtype=np.float32),
            vocab_size=3,
            linear_penalty=linear_penalty,
        )

        sampling_info.update_penalties()

        np.testing.assert_array_equal(sampling_info.linear_penalty, linear_penalty)

    def test_retract_decode_aborts_single_oom_request(self):
        req = SimpleNamespace(
            rid="oom-request",
            output_ids=[7],
            origin_input_ids=[1, 2, 3],
            sampling_params=SimpleNamespace(max_new_tokens=4),
            to_finish=None,
        )
        allocator = SimpleNamespace(
            page_size=1,
            available_size=lambda dp_rank=0: 0,
        )
        batch = ScheduleBatch(
            dp_size=1,
            reqs_info=[
                ScheduleReqsInfo(
                    reqs=[req],
                    seq_lens=np.array([3], dtype=np.int32),
                )
            ],
            token_to_kv_pool_allocator=allocator,
            tree_cache=None,
            is_hybrid=False,
        )
        released = []

        def release_req(idx, dp_rank, remaining_req_count, server_args):
            released.append((idx, dp_rank, remaining_req_count))

        batch.release_req = release_req

        retracted, new_ratio, aborted = batch.retract_decode(SimpleNamespace(page_size=1))

        self.assertEqual(retracted, [])
        self.assertEqual(aborted, [req])
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(req.to_finish.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertEqual(req.to_finish.err_type, "InternalServerError")
        self.assertEqual(released, [(0, 0, 0)])
        self.assertEqual(batch.batch_size(), 0)
        self.assertEqual(new_ratio, 0.0)

    def test_scheduler_update_running_batch_sends_abort_req(self):
        req = SimpleNamespace(rid="oom-request")

        class FakeBatch:
            def __init__(self):
                self.prepared = False

            def batch_size(self):
                return 1

            def filter_batch(self):
                return None

            def is_empty(self):
                return False

            def check_decode_mem(self, buf_multiplier):
                return False

            def retract_decode(self, server_args):
                return [], 0.25, [req]

            def prepare_for_decode(self):
                self.prepared = True

        sender = SimpleNamespace(send_pyobj=MagicMock())
        scheduler = SimpleNamespace(
            decode_mem_cache_buf_multiplier=1,
            new_token_ratio=0.75,
            server_args=SimpleNamespace(),
            _comm_backend=None,
            send_to_tokenizer=sender,
            _extend_requests_to_queue=MagicMock(),
            new_token_ratio_decay=0.1,
            min_new_token_ratio=0.1,
        )
        batch = FakeBatch()

        result = Scheduler.update_running_batch(scheduler, batch)

        self.assertIs(result, batch)
        self.assertTrue(batch.prepared)
        self.assertEqual(scheduler.new_token_ratio, 0.25)
        scheduler._extend_requests_to_queue.assert_called_once_with([], is_retracted=True)
        abort_out = sender.send_pyobj.call_args.args[0]
        self.assertIsInstance(abort_out, AbortReq)
        self.assertEqual(abort_out.rid, "oom-request")

    def test_scheduler_run_batch_spec_decode_uses_single_dp_layout(self):
        req0 = SimpleNamespace(lora_id="0", mm_inputs=None)
        req1 = SimpleNamespace(lora_id="0", mm_inputs=None)
        initial_spec_info = EagleDraftInput(
            allocate_lens=np.array([4, 5], dtype=np.int32),
        )
        batch = ScheduleBatch(
            dp_size=1,
            reqs_info=[
                ScheduleReqsInfo(
                    reqs=[req0, req1],
                    input_ids=np.array([11, 22], dtype=np.int32),
                    req_pool_indices=np.array([0, 1], dtype=np.int32),
                    seq_lens=np.array([4, 5], dtype=np.int32),
                    out_cache_loc=np.array([8, 9], dtype=np.int32),
                    sampling_info=self._sampling_info(batch_size=2),
                    spec_info=initial_spec_info,
                )
            ],
            req_to_token_pool=SimpleNamespace(
                req_to_token=np.array(
                    [
                        [100, 101, 102, 103, 104, 0],
                        [200, 201, 202, 203, 204, 205],
                    ],
                    dtype=np.int32,
                )
            ),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            tree_cache=None,
            forward_mode=ForwardMode.DECODE,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
        )
        next_draft_input = EagleDraftInput(
            verified_id=np.array([101, 201], dtype=np.int32),
            allocate_lens=np.array([6, 7], dtype=np.int32),
        )

        class FakeTPWorker:
            def get_precompile_paddings(self):
                return [1, 2, 4], [1, 2, 4], [4, 8, 16]

        class FakeDraftWorker:
            def __init__(self):
                self.seen_batch = None

            def forward_batch_speculative_generation(self, model_worker_batch):
                self.seen_batch = model_worker_batch
                return GenerationBatchResult(
                    logits_output=SimpleNamespace(),
                    next_token_ids=np.array([10, 11, 12, 20, 0, 0], dtype=np.int32),
                    extend_input_len_per_req=None,
                    extend_logprob_start_len_per_req=None,
                    bid=model_worker_batch.bid,
                    cache_miss_count=0,
                    next_draft_input=next_draft_input,
                    accept_lens=np.array([3, 1], dtype=np.int32),
                    allocate_lens=np.array([6, 7], dtype=np.int32),
                )

        draft_worker = FakeDraftWorker()

        class FakeScheduler:
            _extract_dp_output_ids = Scheduler._extract_dp_output_ids

            def __init__(self):
                self.forward_ct = 0
                self.is_generation = True
                self.spec_algorithm = SpeculativeAlgorithm.EAGLE
                self.dp_size = 1
                self.tp_worker = FakeTPWorker()
                self.draft_worker = draft_worker
                self.page_size = 1
                self.server_args = SimpleNamespace(enable_static_lora=False)

            def _profile_batch_predicate(self, batch):
                return None

        scheduler = FakeScheduler()

        previous_alloc_len = EagleDraftInput.ALLOC_LEN_PER_DECODE
        EagleDraftInput.ALLOC_LEN_PER_DECODE = 2
        try:
            result = Scheduler.run_batch(scheduler, batch)
        finally:
            EagleDraftInput.ALLOC_LEN_PER_DECODE = previous_alloc_len

        self.assertIs(result.next_draft_input, next_draft_input)
        self.assertEqual(result.next_token_ids, [10, 11, 12, 20, 0, 0])
        np.testing.assert_array_equal(result.accept_lens, np.array([3, 1], dtype=np.int32))
        np.testing.assert_array_equal(batch.reqs_info[0].seq_lens, np.array([7, 6], dtype=np.int32))
        self.assertIs(batch.reqs_info[0].spec_info, next_draft_input)
        self.assertEqual(draft_worker.seen_batch.dp_size, 1)
        self.assertEqual(draft_worker.seen_batch.real_bs_per_dp, [2])

    def test_get_top_logprobs_handles_padding_rows(self):
        logprobs = jnp.array(
            [
                [0.1, 0.9, 0.2],
                [0.3, 0.4, 0.5],
            ],
            dtype=jnp.float32,
        )

        values, indices = get_top_logprobs(logprobs, [2, 0])

        self.assertEqual(values.shape, (2, 2))
        self.assertEqual(indices.shape, (2, 2))
        np.testing.assert_allclose(np.asarray(values[0]), np.array([0.9, 0.2], dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(indices[0]), np.array([1, 2], dtype=np.int32))

    def test_get_token_ids_logprobs_handles_padding_rows(self):
        logprobs = jnp.array(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6, 0.7],
                [0.8, 0.9, 1.0, 1.1],
            ],
            dtype=jnp.float32,
        )
        mesh = Mesh(np.array(jax.devices()[:1]), ("data",))

        values, indices = get_token_ids_logprobs(logprobs, [[1, 3], None, [0]], mesh)

        self.assertEqual(values.shape, (3, 2))
        self.assertEqual(indices.shape, (3, 2))
        np.testing.assert_allclose(
            np.asarray(values),
            np.array(
                [
                    [0.1, 0.3],
                    [0.0, 0.0],
                    [0.8, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            np.asarray(indices),
            np.array(
                [
                    [1, 3],
                    [0, 0],
                    [0, 0],
                ],
                dtype=np.int32,
            ),
        )

    def test_decode_logprob_uses_padded_dp_row_index(self):
        class FakeReq:
            def __init__(self):
                self.is_retracted = False
                self.output_ids = []
                self.return_output_logprob_only = False
                self.return_logprob = True
                self.output_token_logprobs_val = []
                self.output_token_logprobs_idx = []
                self.output_top_logprobs_val = []
                self.output_top_logprobs_idx = []
                self.output_token_ids_logprobs_val = []
                self.output_token_ids_logprobs_idx = []
                self.top_logprobs_num = 1
                self.token_ids_logprob = [1]
                self.grammar = None
                self.return_hidden_states = False

            def check_finished(self, new_accepted_len=1):
                return None

            def finished(self):
                return False

        req = FakeReq()
        batch = SimpleNamespace(
            dp_size=2,
            per_dp_bs_size=2,
            return_logprob=True,
            return_output_logprob_only=False,
            spec_algorithm=None,
            cache_miss_count=0,
            reqs_info=[
                ScheduleReqsInfo(reqs=[]),
                ScheduleReqsInfo(reqs=[req]),
            ],
            batch_size=lambda: 1,
        )
        result = SimpleNamespace(
            bid=3,
            next_token_ids=[0, 0, 42, 0],
            cache_miss_count=0,
            logits_output=SimpleNamespace(
                next_token_logprobs=jnp.array([0.1, 0.0, 0.7, 0.0], dtype=jnp.float32),
                next_token_top_logprobs_val=np.array(
                    [[1.0], [0.0], [7.0], [0.0]], dtype=np.float32
                ),
                next_token_top_logprobs_idx=np.array([[10], [0], [70], [0]], dtype=np.int32),
                next_token_token_ids_logprobs_val=np.array(
                    [[1.1], [0.0], [7.7], [0.0]], dtype=np.float32
                ),
                next_token_token_ids_logprobs_idx=np.array([[11], [0], [77], [0]], dtype=np.int32),
                hidden_states=None,
            ),
        )
        scheduler = SimpleNamespace(
            spec_algorithm=None,
            num_generated_tokens=0,
            enable_overlap=False,
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=MagicMock(),
            forward_ct_decode=0,
            server_args=SimpleNamespace(decode_log_interval=100),
            set_next_batch_sampling_info_done=lambda batch: None,
            stream_output=lambda *args, **kwargs: None,
            log_decode_stats=lambda *args, **kwargs: None,
        )

        SchedulerOutputProcessorMixin.process_batch_result_decode(scheduler, batch, result)

        np.testing.assert_allclose(req.output_token_logprobs_val, [0.7])
        self.assertEqual(req.output_token_logprobs_idx, [42])
        np.testing.assert_allclose(req.output_top_logprobs_val[0], np.array([7.0]))
        np.testing.assert_array_equal(req.output_top_logprobs_idx[0], np.array([70]))
        np.testing.assert_allclose(req.output_token_ids_logprobs_val[0], np.array([7.7]))
        np.testing.assert_array_equal(req.output_token_ids_logprobs_idx[0], np.array([77]))

    def test_prefill_logprob_uses_padded_dp_row_index(self):
        class FakeReq:
            def __init__(self):
                self.is_retracted = False
                self.is_chunked = 0
                self.output_ids = []
                self.origin_input_ids = [5]
                self.logprob_start_len = 0
                self.return_output_logprob_only = False
                self.return_logprob = True
                self.output_token_logprobs_val = []
                self.output_token_logprobs_idx = []
                self.output_top_logprobs_val = []
                self.output_top_logprobs_idx = []
                self.output_token_ids_logprobs_val = []
                self.output_token_ids_logprobs_idx = []
                self.input_token_logprobs = []
                self.input_token_logprobs_val = None
                self.input_token_logprobs_idx = None
                self.input_top_logprobs_val = None
                self.input_top_logprobs_idx = None
                self.input_token_ids_logprobs_val = None
                self.input_token_ids_logprobs_idx = None
                self.temp_input_top_logprobs_val = None
                self.temp_input_top_logprobs_idx = None
                self.temp_input_token_ids_logprobs_val = None
                self.temp_input_token_ids_logprobs_idx = None
                self.top_logprobs_num = 1
                self.token_ids_logprob = [1]
                self.grammar = None
                self.return_hidden_states = False

            def check_finished(self, new_accepted_len=1):
                return None

            def finished(self):
                return False

        req = FakeReq()
        batch = SimpleNamespace(
            dp_size=2,
            per_dp_bs_size=2,
            return_logprob=True,
            return_output_logprob_only=False,
            spec_algorithm=None,
            cache_miss_count=0,
            reqs_info=[
                ScheduleReqsInfo(reqs=[]),
                ScheduleReqsInfo(reqs=[req], decoding_reqs=[]),
            ],
        )
        result = SimpleNamespace(
            bid=4,
            next_token_ids=[0, 0, 42, 0],
            extend_input_len_per_req=[1],
            extend_logprob_start_len_per_req=[0],
            cache_miss_count=0,
            next_draft_input=None,
            logits_output=SimpleNamespace(
                next_token_logprobs=np.array([0.1, 0.0, 0.7, 0.0], dtype=np.float32),
                next_token_top_logprobs_val=np.array(
                    [[1.0], [0.0], [7.0], [0.0]], dtype=np.float32
                ),
                next_token_top_logprobs_idx=np.array([[10], [0], [70], [0]], dtype=np.int32),
                next_token_token_ids_logprobs_val=np.array(
                    [[1.1], [0.0], [7.7], [0.0]], dtype=np.float32
                ),
                next_token_token_ids_logprobs_idx=np.array([[11], [0], [77], [0]], dtype=np.int32),
                input_token_logprobs=np.array([0.55], dtype=np.float32),
                input_top_logprobs_val=np.array(
                    [[["row0"]], [["pad1"]], [["row2"]], [["pad3"]]], dtype=object
                ),
                input_top_logprobs_idx=np.array([[[10]], [[0]], [[70]], [[0]]], dtype=object),
                input_token_ids_logprobs_val=np.array(
                    [[["tid0"]], [["pad1"]], [["tid2"]], [["pad3"]]], dtype=object
                ),
                input_token_ids_logprobs_idx=np.array([[[11]], [[0]], [[77]], [[0]]], dtype=object),
                hidden_states=None,
            ),
        )

        class FakeScheduler(SchedulerOutputProcessorMixin):
            pass

        scheduler = FakeScheduler()
        scheduler.is_generation = True
        scheduler.enable_overlap = False
        scheduler.is_mixed_chunk = False
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.tree_cache = MagicMock()
        scheduler.model_config = SimpleNamespace(vocab_size=100)
        scheduler.set_next_batch_sampling_info_done = lambda batch: None
        scheduler.stream_output = lambda *args, **kwargs: None

        SchedulerOutputProcessorMixin.process_batch_result_prefill(scheduler, batch, result)

        np.testing.assert_allclose(req.output_token_logprobs_val, [0.7])
        self.assertEqual(req.output_token_logprobs_idx, [42])
        np.testing.assert_array_equal(req.output_top_logprobs_val[0], np.array([7.0]))
        np.testing.assert_array_equal(req.output_top_logprobs_idx[0], np.array([70]))
        np.testing.assert_allclose(req.output_token_ids_logprobs_val[0], np.array([7.7]))
        np.testing.assert_array_equal(req.output_token_ids_logprobs_idx[0], np.array([77]))
        self.assertEqual(req.temp_input_top_logprobs_val, None)
        self.assertEqual(req.input_top_logprobs_val, [None])


if __name__ == "__main__":
    unittest.main()
