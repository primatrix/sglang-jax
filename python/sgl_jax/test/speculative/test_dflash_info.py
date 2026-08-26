import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.speculative.dflash_info import (
    DFlashDraftInput,
    DFlashVerifyInput,
    build_dflash_draft_block,
    build_dflash_flashback_feedback,
    dflash_greedy_verify,
    select_dflash_flashback_tokens,
)
from sgl_jax.srt.speculative.overlap_utils import (
    can_merge_spec_non_overlap_prefill,
    use_legacy_eagle3_non_overlap,
)
from sgl_jax.srt.speculative.relay_buffer import (
    create_dflash_relay_buffers,
    gather_dflash_relay_buffers,
    update_dflash_relay_buffers,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def test_dflash_verify_input_pytree_round_trip():
    vi = DFlashVerifyInput(
        draft_token=jnp.arange(8, dtype=jnp.int32),
        draft_token_num=4,
    )

    leaves, treedef = jax.tree_util.tree_flatten(vi)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, DFlashVerifyInput)
    assert restored.draft_token_num == 4


def test_dflash_greedy_verify_from_logits():
    # bs=2, block_size=4, vocab=100. draft_token[:,0] is the seed.
    draft_token = jnp.array(
        [10, 11, 12, 13, 20, 21, 22, 23],
        dtype=jnp.int32,
    )
    # Build target logits whose argmax reproduces a chosen target_predict.
    # req0 target_predict = [11, 12, 13, 99] -> accept all 3 drafts, bonus 99
    # req1 target_predict = [21, 77, 88, 99] -> accept 1 draft, bonus 77
    target_predict = np.array([[11, 12, 13, 99], [21, 77, 88, 99]], dtype=np.int32)
    logits = np.full((8, 100), -1.0, dtype=np.float32)
    for i, row in enumerate(target_predict.reshape(-1)):
        logits[i, row] = 10.0
    logits = jnp.asarray(logits)

    accept_lens_out, next_token_ids_flat, new_verified_id, accept_len_draft = dflash_greedy_verify(
        draft_token, logits, draft_token_num=4
    )

    np.testing.assert_array_equal(np.asarray(accept_lens_out), np.array([4, 2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(accept_len_draft), np.array([3, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(new_verified_id), np.array([99, 77], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(next_token_ids_flat).reshape(2, 4), target_predict)


def test_dflash_greedy_verify_keeps_outputs_data_sharded():
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = np.asarray(jax.devices())
    data_size = 2 if devices.size >= 2 and devices.size % 2 == 0 else 1
    tensor_size = devices.size // data_size
    mesh = Mesh(
        devices.reshape(data_size, tensor_size),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    bs = data_size * 2
    block_size = 4
    vocab_size = max(128, tensor_size * 128)
    candidates = np.tile(np.array([[10, 11, 12, 13]], dtype=np.int32), (bs, 1))
    target_predict = np.tile(np.array([[11, 12, 13, 99]], dtype=np.int32), (bs, 1))
    logits = np.full((bs * block_size, vocab_size), -1.0, dtype=np.float32)
    logits[np.arange(bs * block_size), target_predict.reshape(-1)] = 10.0

    draft_token = jax.device_put(candidates.reshape(-1), NamedSharding(mesh, P("data")))
    target_logits = jax.device_put(logits, NamedSharding(mesh, P("data", "tensor")))
    accept_lens, next_tokens, verified_id, accept_draft = dflash_greedy_verify(
        draft_token,
        target_logits,
        draft_token_num=block_size,
    )

    for output in (accept_lens, next_tokens, verified_id, accept_draft):
        assert output.sharding.spec == P("data")
    np.testing.assert_array_equal(np.asarray(accept_lens), np.full(bs, 4, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(verified_id), np.full(bs, 99, dtype=np.int32))


def test_build_dflash_flashback_feedback_skips_first_rejection_and_aligns_suffix():
    # Three accepted proposals: d4 is rejected/replaced, so d5..d8 align with
    # the first four proposal positions of the next round.
    candidates = jnp.asarray([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=jnp.int32)
    vocab_size = 32
    logits = np.full((1, 8, vocab_size), -10.0, dtype=np.float32)
    target_top1 = np.array([11, 12, 13, 19, 15, 21, 17, 22], dtype=np.int32)
    for row, token_id in enumerate(target_top1):
        logits[0, row, token_id] = 5.0
    # d5 and d7 are target top-1 on the stale path. d6 is two logits behind.
    logits[0, 5, 16] = 3.0

    stale_ids, target_margins, valid = build_dflash_flashback_feedback(
        candidates.reshape(-1),
        jnp.asarray(logits.reshape(-1, vocab_size)),
        jnp.asarray(target_top1.reshape(-1)),
        jnp.asarray([3], dtype=jnp.int32),
        draft_token_num=8,
    )

    np.testing.assert_array_equal(np.asarray(stale_ids[0, :4]), np.array([15, 16, 17, 0]))
    np.testing.assert_array_equal(
        np.asarray(valid),
        np.array([[True, True, True, False, False, False, False]]),
    )
    np.testing.assert_allclose(np.asarray(target_margins[0, :3]), np.array([0.0, -2.0, 0.0]))


def test_select_dflash_flashback_tokens_uses_sparse_target_discounted_bonus():
    logits = np.array(
        [
            [
                [0.0, 5.0, 4.4, 0.0],  # stale 2 trails by 0.6: recycled
                [0.0, 5.0, 4.4, 0.0],  # target margin discounts bonus: rejected
                [0.0, 5.0, 4.4, 0.0],  # position decay leaves 0.25: rejected
            ]
        ],
        dtype=np.float32,
    )
    selected = select_dflash_flashback_tokens(
        jnp.asarray(logits),
        jnp.asarray([[2, 2, 2]], dtype=jnp.int32),
        jnp.asarray([[0.0, -0.6, 0.0]], dtype=jnp.float32),
        jnp.asarray([[True, True, True]], dtype=jnp.bool_),
        bonus=1.0,
        target_margin_weight=1.0,
        position_decay=0.5,
    )
    np.testing.assert_array_equal(np.asarray(selected), np.array([[2, 1, 1]], dtype=np.int32))


def test_select_dflash_flashback_tokens_gathers_across_tensor_shards():
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = np.asarray(jax.devices())
    tensor_size = devices.size
    mesh = Mesh(
        devices.reshape(1, tensor_size),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    vocab_size = tensor_size * 8
    logits = np.zeros((1, 1, vocab_size), dtype=np.float32)
    logits[0, 0, vocab_size - 1] = 5.0
    logits[0, 0, 1] = 4.5
    logits = jax.device_put(logits, NamedSharding(mesh, P("data", None, "tensor")))
    feedback_sharding = NamedSharding(mesh, P("data", None))

    with jax.set_mesh(mesh):
        selected = select_dflash_flashback_tokens(
            logits,
            jax.device_put(np.array([[1]], dtype=np.int32), feedback_sharding),
            jax.device_put(np.array([[0.0]], dtype=np.float32), feedback_sharding),
            jax.device_put(np.array([[True]], dtype=np.bool_), feedback_sharding),
            bonus=1.0,
            target_margin_weight=1.0,
            position_decay=0.5,
        )

    assert selected.sharding.spec == P("data", None)
    np.testing.assert_array_equal(np.asarray(selected), np.array([[1]], dtype=np.int32))


def test_dflash_relay_round_trips_flashback_feedback():
    from types import SimpleNamespace

    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = np.asarray(jax.devices())
    mesh = Mesh(
        devices.reshape(1, -1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    req_pool = SimpleNamespace(req_to_token=np.zeros((4, 1), dtype=np.int32))
    buffers = create_dflash_relay_buffers(
        mesh,
        req_pool,
        dp_size=1,
        feedback_width=3,
    )
    vector_sharding = NamedSharding(mesh, P("data"))
    feedback_sharding = NamedSharding(mesh, P("data", None))
    with jax.set_mesh(mesh):
        buffers = update_dflash_relay_buffers(
            buffers,
            jax.device_put(np.array([2, 0], dtype=np.int32), vector_sharding),
            jax.device_put(np.array([True, False]), vector_sharding),
            jax.device_put(np.array([9, 7], dtype=np.int32), vector_sharding),
            jax.device_put(np.array([21, 19], dtype=np.int32), vector_sharding),
            jax.device_put(np.array([[4, 5, 0], [8, 8, 8]], dtype=np.int32), feedback_sharding),
            jax.device_put(
                np.array([[0.0, -0.5, 0.0], [0, 0, 0]], dtype=np.float32),
                feedback_sharding,
            ),
            jax.device_put(np.array([[True, True, False], [True, True, True]]), feedback_sharding),
            dp_size=1,
        )
        gathered = gather_dflash_relay_buffers(
            buffers,
            jax.device_put(np.array([2, 1], dtype=np.int32), vector_sharding),
            dp_size=1,
        )

    verified_id, seq_lens, stale_ids, margins, valid = map(np.asarray, gathered)
    np.testing.assert_array_equal(verified_id, np.array([9, 0], dtype=np.int32))
    np.testing.assert_array_equal(seq_lens, np.array([21, 0], dtype=np.int32))
    np.testing.assert_array_equal(stale_ids[0], np.array([4, 5, 0], dtype=np.int32))
    np.testing.assert_allclose(margins[0], np.array([0.0, -0.5, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(valid[0], np.array([True, True, False]))


def test_dflash_draft_input_filter_batch():
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 30], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([1, 2, 3], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7], dtype=np.int32),
        flashback_token_ids=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        flashback_target_margins=np.array([[0, -1], [-2, 0], [0, 0]], dtype=np.float32),
        flashback_valid_mask=np.array([[1, 1], [1, 1], [1, 0]], dtype=np.bool_),
        block_size=3,
    )

    di.filter_batch(np.array([2, 0], dtype=np.int32), has_been_filtered=False)

    np.testing.assert_array_equal(di.verified_id, np.array([30, 10], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([3, 1], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([7, 5], dtype=np.int32))
    np.testing.assert_array_equal(di.flashback_token_ids, np.array([[5, 6], [1, 2]]))
    np.testing.assert_array_equal(di.flashback_valid_mask, np.array([[True, False], [True, True]]))


def test_dflash_draft_input_new_tokens_required_next_decode_page_aligned():
    class Req:
        def __init__(self, committed, allocated):
            self.kv_committed_len = committed
            self.kv_allocated_len = allocated

    di = DFlashDraftInput(
        verified_id=np.array([0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0], dtype=np.int32),
        draft_seq_lens=np.array([0, 0], dtype=np.int32),
        block_size=16,
    )

    requests = [
        Req(committed=120, allocated=120),  # needs slots through 136 -> one new page
        Req(committed=16, allocated=128),  # already has enough page capacity
    ]

    assert di.new_tokens_required_next_decode(requests, page_size=128) == 128


def test_dflash_draft_input_align_to_reqs_appends_merged_request_state():
    class Req:
        def __init__(self, origin_input_ids, output_ids):
            self.origin_input_ids = origin_input_ids
            self.output_ids = output_ids

    di = DFlashDraftInput(
        verified_id=np.array([10, 20], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6], dtype=np.int32),
        block_size=16,
    )
    reqs = [
        Req([1, 10], []),
        Req([1, 20], []),
        Req([1, 2, 3], [30]),
    ]

    di._align_to_reqs(reqs, np.array([5, 6, 7], dtype=np.int32))

    np.testing.assert_array_equal(di.verified_id, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([5, 6, 7], dtype=np.int32))


def test_dflash_draft_input_aligns_dp_ranks_without_cross_rank_truncation():
    class Req:
        def __init__(self, token, committed):
            self.origin_input_ids = [token]
            self.output_ids = []
            self.kv_committed_len = committed

    rank0 = DFlashDraftInput(
        verified_id=np.array([10], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=np.array([5], dtype=np.int32),
        block_size=16,
    )
    rank1 = DFlashDraftInput(
        verified_id=np.array([20], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=np.array([7], dtype=np.int32),
        block_size=16,
    )
    flat = DFlashDraftInput(
        verified_id=np.array([10, 20], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 7], dtype=np.int32),
        block_size=16,
    )
    schedule_batch = type(
        "Batch",
        (),
        {
            "reqs_info": [
                type("Info", (), {"reqs": [Req(10, 5)], "spec_info": rank0})(),
                type("Info", (), {"reqs": [Req(20, 7)], "spec_info": rank1})(),
            ]
        },
    )()

    flat._align_dp_state_to_reqs(schedule_batch)

    np.testing.assert_array_equal(flat.verified_id, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(flat.ctx_lens, np.array([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(flat.draft_seq_lens, np.array([5, 7], dtype=np.int32))


def test_dflash_dp_scatter_rejects_incomplete_state():
    incomplete = DFlashDraftInput(
        verified_id=np.array([10], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=None,
    )

    with np.testing.assert_raises_regex(ValueError, "draft_seq_lens.*missing"):
        ScheduleBatch._scatter_spec_info_to_dp_slots(
            incomplete,
            selector=np.array([0], dtype=np.int32),
            total_bs=2,
        )


def test_dflash_concat_normalizes_empty_and_none_target_hidden():
    rank0 = DFlashDraftInput(
        verified_id=np.array([10], dtype=np.int32),
        target_hidden=jnp.zeros((0, 8), dtype=jnp.bfloat16),
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=np.array([5], dtype=np.int32),
    )
    rank1 = DFlashDraftInput(
        verified_id=np.array([20], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=np.array([7], dtype=np.int32),
    )

    flat = ScheduleBatch._concat_spec_info_per_rank([rank0, rank1])

    assert flat.target_hidden is None
    np.testing.assert_array_equal(flat.verified_id, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(flat.draft_seq_lens, np.array([5, 7], dtype=np.int32))


def test_dflash_draft_input_scatter_pads_to_spec_decode_bucket():
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 30], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7], dtype=np.int32),
        block_size=16,
    )

    padded = ScheduleBatch._scatter_spec_info_to_dp_slots(
        di,
        selector=np.array([0, 1, 2], dtype=np.int32),
        total_bs=4,
    )

    np.testing.assert_array_equal(padded.verified_id, np.array([10, 20, 30, 0], dtype=np.int32))
    np.testing.assert_array_equal(padded.ctx_lens, np.array([0, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(padded.draft_seq_lens, np.array([5, 6, 7, 0], dtype=np.int32))

    [unpadded] = ScheduleBatch._split_spec_info_per_rank(padded, [3])
    np.testing.assert_array_equal(unpadded.verified_id, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(unpadded.draft_seq_lens, np.array([5, 6, 7], dtype=np.int32))


def test_dflash_draft_input_dp_scatter_and_compact_split_round_trip():
    compact = DFlashDraftInput(
        verified_id=np.array([10, 20, 30], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7], dtype=np.int32),
        block_size=4,
    )
    selector = np.array([0, 1, 3], dtype=np.int32)  # rank0: 2/3, rank1: 1/3

    padded = ScheduleBatch._scatter_spec_info_to_dp_slots(
        compact,
        selector=selector,
        total_bs=6,
    )
    np.testing.assert_array_equal(
        padded.verified_id,
        np.array([10, 20, 0, 30, 0, 0], dtype=np.int32),
    )

    # The worker compacts verify output with the same selector before the
    # scheduler stores per-rank cross-round state.
    compact_again = DFlashDraftInput(
        verified_id=np.asarray(padded.verified_id)[selector],
        target_hidden=None,
        ctx_lens=np.asarray(padded.ctx_lens)[selector],
        draft_seq_lens=np.asarray(padded.draft_seq_lens)[selector],
        block_size=4,
    )
    rank0, rank1 = ScheduleBatch._split_spec_info_per_rank(compact_again, [2, 1])
    np.testing.assert_array_equal(rank0.verified_id, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(rank1.verified_id, np.array([30], dtype=np.int32))


def test_dflash_non_overlap_can_merge_without_legacy_eagle3_accounting():
    assert can_merge_spec_non_overlap_prefill(False, SpeculativeAlgorithm.DFLASH)
    assert not use_legacy_eagle3_non_overlap(False, SpeculativeAlgorithm.DFLASH)


def test_build_dflash_draft_block():
    verified_id = np.array([7, 8], dtype=np.int32)
    target_prefix_lens = np.array([5, 3], dtype=np.int32)

    block_ids, positions = build_dflash_draft_block(
        verified_id=verified_id,
        mask_token_id=99,
        target_prefix_lens=target_prefix_lens,
        block_size=4,
    )

    np.testing.assert_array_equal(
        np.asarray(block_ids),
        np.array([[7, 99, 99, 99], [8, 99, 99, 99]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(positions),
        np.array([[5, 6, 7, 8], [3, 4, 5, 6]], dtype=np.int32),
    )
