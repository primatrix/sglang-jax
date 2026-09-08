import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4CacheSpec, DeepseekV4TokenToKVPool


@pytest.fixture(autouse=True)
def isolated_mesh_context():
    # Legacy test modules set a global mesh at import time. Restore it after
    # each test so tests using different device subsets are order-independent.
    with jax.set_mesh(None):
        yield


def make_allocator(p=128, history=4, swa=4, dp=1):
    if jax.device_count() < dp:
        pytest.skip("requires multiple CPU devices")
    mesh = Mesh(np.array(jax.devices()[:dp]).reshape(dp, 1), ("data", "tensor"))
    pool = DeepseekV4TokenToKVPool(
        p * history * dp, p * swa * dp, p, DeepseekV4CacheSpec((0, 4, 128), 8, 4), mesh, dp
    )
    return DeepseekV4TokenToKVPoolAllocator(pool)


def assert_snapshot_equal(a, b):
    leaves_a, _ = jax.tree_util.tree_flatten(a)
    leaves_b, _ = jax.tree_util.tree_flatten(b)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("p", [128, 256])
def test_127_128_129_chunks_history_roundtrip_and_full_release(p):
    a = make_allocator(p)
    locs = a.alloc_extend([0], [127], [-1], 127)
    for end in (128, 129, 133, p + 7):
        demand = a.estimate_extend([len(locs)], [end], [int(locs[-1])], end - len(locs))
        before = (a.full_available_size(), a.swa_available_size())
        new = a.alloc_extend([len(locs)], [end], [int(locs[-1])], end - len(locs))
        assert (before[0] - a.full_available_size(), before[1] - a.swa_available_size()) == (
            demand.history_tokens,
            demand.swa_tokens,
        )
        locs = np.concatenate([locs, new])
    kv = a.get_kvcache()
    kv.write(
        "c4", 1, jnp.array([locs[3] // 4]), jnp.full((1, 8), 7, jnp.bfloat16), jnp.array([True])
    )
    kv.write(
        "c128",
        2,
        jnp.array([locs[127] // 128]),
        jnp.full((1, 8), 9, jnp.bfloat16),
        jnp.array([True]),
    )
    a.free_swa(locs[:p])
    assert a.full_available_size() == 2 * p
    assert a.swa_available_size() == 3 * p
    assert np.all(a.full_to_swa_index_mapping[locs[:p]] == 0)
    np.testing.assert_array_equal(kv.get_buffer("c4", 1).reshape(-1, 8)[locs[3] // 4], 7)
    np.testing.assert_array_equal(kv.get_buffer("c128", 2).reshape(-1, 8)[locs[127] // 128], 9)
    a.free(locs)
    a.free(locs)  # repeated cleanup before reuse is harmless
    assert a.full_available_size() == a.swa_available_size() == 4 * p


@pytest.mark.parametrize("history,swa", [(1, 3), (3, 1)])
def test_one_sided_exhaustion_preserves_old_partial_page_and_mapping(history, swa):
    a = make_allocator(history=history, swa=swa)
    first = a.alloc_extend([0], [127], [-1], 127)
    before = a.backup_state()
    demand = a.estimate_extend([127], [129], [first[-1]], 2)
    assert demand.history_pages == demand.swa_pages == 1
    assert not a.can_allocate(demand)
    assert a.alloc_extend([127], [129], [first[-1]], 2) is None
    assert_snapshot_equal(before, a.backup_state())
    # The old tail remains writable despite the failed cross-page extension.
    tail = a.alloc_decode([128], [first[-1]])
    assert tail.tolist() == [255]
    assert a.full_to_swa_index_mapping[first[0]] > 0
    a.free(np.concatenate([first, tail]))
    assert a.full_available_size() == 128 * history
    assert a.swa_available_size() == 128 * swa


def test_swa_tail_reallocation_and_snapshot_restore_keep_mapping_reference():
    a = make_allocator(history=2, swa=1)
    first = a.alloc_extend([0], [3], [-1], 3)
    a.free_swa(first)
    a.free_swa(first)
    snapshot = a.backup_state()
    mapping = a.full_to_swa_index_mapping
    demand = a.estimate_decode([4], [first[-1]])
    assert (demand.history_pages, demand.swa_pages) == (0, 1)
    tail = a.alloc_decode([4], [first[-1]])
    assert mapping[tail[0]] == 131
    a.restore_state(snapshot)
    assert a.full_to_swa_index_mapping is mapping
    assert mapping[tail[0]] == 0
    assert_snapshot_equal(snapshot, a.backup_state())
    tail = a.alloc_decode([4], [first[-1]])
    a.free(np.concatenate([first, tail]))
    assert a.swa_available_size() == 128


def test_batch_reordering_different_progress_and_partial_release_rejected():
    a = make_allocator(history=5, swa=5)
    batch = a.alloc_extend([0, 0], [3, 128], [-1, -1], 131)
    left, right = batch[:3], batch[3:]
    out = a.alloc_decode([129, 4], [right[-1], left[-1]])
    assert out[0] // 128 != right[-1] // 128
    assert out[1] == left[-1] + 1
    before = a.backup_state()
    with pytest.raises(ValueError, match="partial page"):
        a.free_swa(left[:1])
    with pytest.raises(ValueError, match="partial page"):
        a.free(left)
    assert_snapshot_equal(before, a.backup_state())
    a.free_group_begin()
    a.free(np.concatenate([right, out[:1]]))
    a.free(np.concatenate([left, out[1:]]))
    a.free_group_end()
    assert a.available_size() == 640


@pytest.mark.parametrize(
    "pre,seq,last,total",
    [
        ([0], [1], [-1], 2),
        ([3], [2], [130], -1),
        ([1], [2], [0], 1),
        ([0], [1], [7], 1),
        ([0.5], [1], [-1], 1),
    ],
)
def test_invalid_allocation_is_atomic(pre, seq, last, total):
    a = make_allocator()
    snapshot = a.backup_state()
    with pytest.raises(ValueError):
        a.alloc_extend(pre, seq, last, total)
    assert_snapshot_equal(snapshot, a.backup_state())


def test_padding_cannot_be_freed_and_duplicate_batch_tails_are_rejected():
    a = make_allocator()
    x = a.alloc_extend([0], [1], [-1], 1)
    before = a.backup_state()
    with pytest.raises(ValueError):
        a.free(np.array([0]))
    with pytest.raises(ValueError):
        a.alloc_extend([1, 1], [2, 2], [x[-1], x[-1]], 2)
    assert_snapshot_equal(before, a.backup_state())


def test_dp_allocator_independence():
    a = make_allocator(history=1, swa=1, dp=2)
    x = a.alloc(128, 0)
    y = a.alloc(128, 1)
    np.testing.assert_array_equal(x, y)
    assert a.alloc(128, 0) is None
    a.free_swa(x, 0)
    assert a.swa_available_size(0) == 128
    assert a.swa_available_size(1) == 0
    assert np.all(a.full_to_swa_index_mapping[1][y] > 0)
    a.free(x, 0)
    a.free(y, 1)
    assert a.available_size(0) == a.available_size(1) == 128


def test_randomized_request_lifecycles_conserve_both_resources():
    rng = np.random.default_rng(93)
    a = make_allocator(history=8, swa=5)
    requests = {}
    for _ in range(250):
        key = int(rng.integers(0, 4))
        locs = requests.get(key, np.empty(0, np.int32))
        op = int(rng.integers(0, 4))
        if op == 0 and locs.size:
            a.free(locs)
            del requests[key]
        elif op == 1 and locs.size:
            # Reclaim only complete old pages, retaining the current tail.
            n = max(0, (len(locs) // 128 - 1) * 128)
            a.free_swa(locs[:n])
        else:
            n = int(rng.integers(1, 160))
            before = a.backup_state()
            demand = a.estimate_extend(
                [len(locs)], [len(locs) + n], [locs[-1] if locs.size else -1], n
            )
            can = a.can_allocate(demand)
            out = a.alloc_extend([len(locs)], [len(locs) + n], [locs[-1] if locs.size else -1], n)
            assert (out is not None) == can
            if out is None:
                assert_snapshot_equal(before, a.backup_state())
            else:
                requests[key] = np.concatenate([locs, out])
        all_locs = np.concatenate(list(requests.values())) if requests else np.empty(0, np.int32)
        used_history = len(np.unique(all_locs // 128))
        mapped = a.full_to_swa_index_mapping[all_locs]
        used_swa = len(np.unique(mapped[mapped > 0] // 128))
        assert a.full_available_size() + 128 * used_history == 8 * 128
        assert a.swa_available_size() + 128 * used_swa == 5 * 128
    for locs in requests.values():
        a.free(locs)
    assert a.available_size() == 5 * 128
    assert a.full_available_size() == 8 * 128


def test_empty_operations_are_noops():
    a = make_allocator()
    snapshot = a.backup_state()
    a.free([])
    a.free_swa([])
    assert a.alloc_extend([], [], [], 0).dtype == np.int32
    assert a.alloc(0).size == 0
    assert_snapshot_equal(snapshot, a.backup_state())
