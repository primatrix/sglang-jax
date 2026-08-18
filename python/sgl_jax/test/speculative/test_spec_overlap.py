from types import SimpleNamespace
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.draft_extend_fused import (
    _prepare_eagle_overlap_verify,
    _prepare_mtp_overlap_verify,
    _spec_decode_fused_chain_overlap,
)
from sgl_jax.srt.speculative.multi_layer_draft_worker import MultiLayerDraftWorker


def test_fused_chain_overlap_uses_common_relay_envelope():
    published_new_seq_lens = Mock()
    next_draft_input = SimpleNamespace(
        future_indices=None,
        new_seq_lens=np.array([7, 8, 9], dtype=np.int32),
    )
    batch_output = SimpleNamespace(
        next_draft_input=next_draft_input,
        published_new_seq_lens=published_new_seq_lens,
    )
    draft_worker = object()
    spec_worker = SimpleNamespace(
        draft_worker=draft_worker,
        spec_relay_buffers="old-buffers",
    )
    model_worker_batch = SimpleNamespace(
        req_pool_indices=np.array([4, 5, 0, 8, 0, 0], dtype=np.int32),
        logits_indices_selector=np.array([0, 1, 3], dtype=np.int32),
        real_bs_per_dp=[2, 1],
        per_dp_bs_size=3,
    )
    prepare_verify = Mock(return_value=("token-map", True))
    launch_draft = Mock(return_value=SimpleNamespace(updated_relay_buffers="new-buffers"))

    with patch(
        "sgl_jax.srt.speculative.draft_extend_fused.spec_decode_verify",
        return_value=batch_output,
    ) as verify:
        result, published = _spec_decode_fused_chain_overlap(
            spec_worker,
            model_worker_batch,
            np.array([16, 24, 32], dtype=np.int32),
            prepare_verify=prepare_verify,
            launch_draft=launch_draft,
        )

    assert result is batch_output
    assert published is published_new_seq_lens
    published_new_seq_lens.copy_to_host_async.assert_called_once_with()
    prepare_verify.assert_called_once_with(draft_worker, model_worker_batch)
    verify.assert_called_once()
    assert verify.call_args.kwargs["draft_to_target_token_ids"] == "token-map"
    assert verify.call_args.kwargs["draft_padding_prepared"] is True
    launch_draft.assert_called_once()
    np.testing.assert_array_equal(
        launch_draft.call_args.kwargs["relay_future_indices"],
        np.array([4, 5, 0, 8, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        launch_draft.call_args.kwargs["relay_valid_mask"],
        np.array([True, True, False, True, False, False]),
    )
    np.testing.assert_array_equal(
        next_draft_input.future_indices,
        np.array([4, 5, 8], dtype=np.int32),
    )
    assert next_draft_input.new_seq_lens is None
    assert spec_worker.spec_relay_buffers == "new-buffers"


def test_overlap_verify_strategies_keep_algorithm_specific_bootstrap():
    eagle_worker = SimpleNamespace(
        hot_token_ids="hot-token-map",
        prepare_for_fused_verify=Mock(return_value="bootstrap-token-map"),
    )
    relay_batch = SimpleNamespace(
        spec_info_padded=SimpleNamespace(
            future_indices=np.array([3], dtype=np.int32),
            topk_index=None,
        )
    )
    bootstrap_batch = SimpleNamespace(
        spec_info_padded=SimpleNamespace(
            future_indices=None,
            topk_index=np.array([[11]], dtype=np.int32),
        )
    )

    assert _prepare_eagle_overlap_verify(eagle_worker, relay_batch) == (
        "hot-token-map",
        False,
    )
    assert _prepare_eagle_overlap_verify(eagle_worker, bootstrap_batch) == (
        "bootstrap-token-map",
        True,
    )

    mtp_worker = SimpleNamespace(prepare_for_fused_verify=Mock())
    assert _prepare_mtp_overlap_verify(mtp_worker, relay_batch) == (None, False)
    assert _prepare_mtp_overlap_verify(mtp_worker, bootstrap_batch) == (None, True)
    mtp_worker.prepare_for_fused_verify.assert_called_once_with(bootstrap_batch)


def _make_mtp_worker_for_prefill():
    worker = object.__new__(MultiLayerDraftWorker)
    worker.speculative_num_steps = 3
    worker.speculative_num_draft_tokens = 4
    worker._workers = [object()] * 3
    return worker


def _make_prefill_mwb():
    return SimpleNamespace(
        real_bs=2,
        dp_size=1,
        per_dp_bs_size=2,
        seq_lens=np.array([2, 2], dtype=np.int32),
        input_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        extend_seq_lens=np.array([2, 2], dtype=np.int32),
        req_pool_indices=np.array([5, 6], dtype=np.int32),
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        spec_algorithm=None,
    )


def test_mtp_prefill_relay_publish_uses_relay_form_state():
    worker = _make_mtp_worker_for_prefill()
    mwb = _make_prefill_mwb()
    next_token_ids = np.array([10, 20], dtype=np.int32)
    relay_future_indices = np.array([5, 6], dtype=np.int32)
    relay_valid_mask = np.array([True, True])

    with patch(
        "sgl_jax.srt.speculative.draft_extend_fused.mtp_prefill_draft_extend",
        return_value=(None, None, "updated-relay"),
    ) as mtp_prefill:
        result = worker.draft_extend_for_prefill(
            mwb,
            hidden_states="target-hidden",
            next_token_ids=next_token_ids,
            relay_buffers="relay",
            relay_future_indices=relay_future_indices,
            relay_valid_mask=relay_valid_mask,
        )

    assert result == "updated-relay"
    mtp_prefill.assert_called_once()
    assert mtp_prefill.call_args.kwargs["relay_buffers"] == "relay"
    np.testing.assert_array_equal(
        mtp_prefill.call_args.kwargs["relay_future_indices"], relay_future_indices
    )
    np.testing.assert_array_equal(
        mtp_prefill.call_args.kwargs["relay_valid_mask"], relay_valid_mask
    )

    spec_info = mwb.spec_info_padded
    # Relay form: only req indices + allocation lengths cross the scheduler.
    np.testing.assert_array_equal(spec_info.future_indices, np.array([5, 6]))
    assert spec_info.topk_index is None
    assert spec_info.hidden_states is None
    np.testing.assert_array_equal(spec_info.allocate_lens, np.array([2, 2]))
    # The prefill input rotation still ran before the relay publish: each
    # request's extend segment shifts left by one and appends its verified id.
    np.testing.assert_array_equal(mwb.input_ids, np.array([2, 10, 4, 20], dtype=np.int32))


def test_mtp_prefill_without_relay_keeps_direct_state():
    worker = _make_mtp_worker_for_prefill()
    mwb = _make_prefill_mwb()
    next_token_ids = np.array([10, 20], dtype=np.int32)

    with patch(
        "sgl_jax.srt.speculative.draft_extend_fused.mtp_prefill_draft_extend",
        return_value=(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[11, 12, 13], [21, 22, 23]]),
            None,
        ),
    ) as mtp_prefill:
        result = worker.draft_extend_for_prefill(
            mwb,
            hidden_states="target-hidden",
            next_token_ids=next_token_ids,
        )

    assert result is None
    mtp_prefill.assert_called_once()
    assert "relay_buffers" not in mtp_prefill.call_args.kwargs
    spec_info = mwb.spec_info_padded
    np.testing.assert_array_equal(spec_info.topk_index, np.array([[11, 12, 13], [21, 22, 23]]))
    np.testing.assert_array_equal(spec_info.verified_id, np.array([10, 20]))
    assert spec_info.future_indices is None


def test_mtp_prefill_extend_branch_publishes_relay_state():
    draft_worker = Mock()
    draft_worker.draft_extend_for_prefill.return_value = "updated-relay"
    init_relay = Mock()
    spec_worker = SimpleNamespace(
        server_args=SimpleNamespace(disable_overlap_schedule=False),
        spec_relay_buffers="relay",
        _can_use_fused_mtp_verify=True,
        init_spec_relay_buffers=init_relay,
        draft_worker=draft_worker,
        mesh=object(),
        target_worker=SimpleNamespace(model_config=SimpleNamespace(vocab_size=100)),
        _prepare_overlap_sampling_info=Mock(),
        forward_target_extend=Mock(
            return_value=(
                SimpleNamespace(
                    next_token_logits=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
                    hidden_states="target-hidden",
                ),
                None,
                0,
                7,
                np.array([2, 2]),
            )
        ),
    )
    mwb = _make_prefill_mwb()
    mwb.forward_mode = ForwardMode.EXTEND
    mwb.sampling_info = SimpleNamespace(
        temperatures=np.array([0.0, 0.0]),
        is_all_greedy=True,
    )
    mwb.spec_info_padded = object()
    mwb.real_bs_per_dp = [2]

    from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

    with patch.object(SamplingMetadata, "from_model_worker_batch", return_value=None):
        result = _call_spec_generation(spec_worker, mwb)

    init_relay.assert_called_once()
    draft_worker.draft_extend_for_prefill.assert_called_once()
    call_kwargs = draft_worker.draft_extend_for_prefill.call_args.kwargs
    assert call_kwargs["relay_buffers"] == "relay"
    np.testing.assert_array_equal(call_kwargs["relay_future_indices"], np.array([5, 6]))
    np.testing.assert_array_equal(call_kwargs["relay_valid_mask"], np.array([True, True]))
    assert spec_worker.spec_relay_buffers == "updated-relay"
    assert result.next_draft_input is mwb.spec_info_padded


def test_mtp_prefill_extend_branch_without_relay_keeps_plain_call():
    draft_worker = Mock()
    spec_worker = SimpleNamespace(
        server_args=SimpleNamespace(disable_overlap_schedule=True),
        spec_relay_buffers=None,
        _can_use_fused_mtp_verify=True,
        init_spec_relay_buffers=Mock(),
        draft_worker=draft_worker,
        mesh=object(),
        target_worker=SimpleNamespace(model_config=SimpleNamespace(vocab_size=100)),
        _prepare_overlap_sampling_info=Mock(),
        forward_target_extend=Mock(
            return_value=(
                SimpleNamespace(
                    next_token_logits=jnp.array([[0.0, 1.0], [1.0, 0.0]]),
                    hidden_states="target-hidden",
                ),
                None,
                0,
                7,
                np.array([2, 2]),
            )
        ),
    )
    mwb = _make_prefill_mwb()
    mwb.forward_mode = ForwardMode.EXTEND
    mwb.sampling_info = SimpleNamespace(
        temperatures=np.array([0.0, 0.0]),
        is_all_greedy=True,
    )
    mwb.spec_info_padded = object()
    mwb.real_bs_per_dp = [2]

    from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

    with patch.object(SamplingMetadata, "from_model_worker_batch", return_value=None):
        _call_spec_generation(spec_worker, mwb, launch_done=Mock())

    spec_worker.init_spec_relay_buffers.assert_not_called()
    draft_worker.draft_extend_for_prefill.assert_called_once()
    assert "relay_buffers" not in draft_worker.draft_extend_for_prefill.call_args.kwargs


def _call_spec_generation(spec_worker, mwb, launch_done=None):
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    return BaseSpecWorker.forward_batch_speculative_generation(
        spec_worker, mwb, launch_done=launch_done
    )
