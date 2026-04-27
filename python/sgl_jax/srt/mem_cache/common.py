import logging

from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)

# TODO @pc we should separate all mem cache from schedule batch to support more flexible operations


def alloc_token_slots(
    tree_cache: BasePrefixCache,
    num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator

    evict_from_tree_cache(tree_cache, num_tokens)
    if backup_state:
        state = allocator.backup_state()
    out_cache_loc = allocator.alloc(num_tokens)
    if out_cache_loc is None:
        error_msg = (
            f"Out of memory. Try to lower your batch size.\n"
            f"Try to allocate {num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache=tree_cache)}"
        )
        logger.error(error_msg)
        if tree_cache is not None:
            tree_cache.pretty_print()
        raise RuntimeError(error_msg)
    if backup_state:
        return out_cache_loc, state
    else:
        return out_cache_loc


def alloc_paged_token_slots_extend(
    tree_cache: BasePrefixCache,
    prefix_lens: list[int],
    seq_lens: list[int],
    last_loc: list[int],
    extend_num_tokens: int,
    backup_state: bool = False,
):
    allocator = tree_cache.token_to_kv_pool_allocator
    num_tokens = extend_num_tokens + len(seq_lens) * allocator.page_size
    evict_from_tree_cache(tree_cache, num_tokens)
    if backup_state:
        state = allocator.backup_state()
    out_cache_loc = allocator.alloc_extend(prefix_lens, seq_lens, last_loc, extend_num_tokens)
    if out_cache_loc is None:
        error_msg = (
            f"Prefill out of memory. Try to lower your batch size.\n"
            f"Try to allocate {extend_num_tokens} tokens.\n"
            f"{available_and_evictable_str(tree_cache=tree_cache)}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    if backup_state:
        return out_cache_loc, state
    else:
        return out_cache_loc


def evict_from_tree_cache(tree_cache: BasePrefixCache | None, num_tokens: int):
    if tree_cache is None:
        return

    allocator = tree_cache.token_to_kv_pool_allocator

    # Check if this is a hybrid allocator
    if hasattr(allocator, "full_available_size"):
        # Hybrid allocator
        full_available_size = allocator.full_available_size()
        swa_available_size = allocator.swa_available_size()

        if full_available_size < num_tokens or swa_available_size < num_tokens:
            full_num_tokens = max(0, num_tokens - full_available_size)
            swa_num_tokens = max(0, num_tokens - swa_available_size)
            tree_cache.evict(full_num_tokens, swa_num_tokens)
    else:
        # Standard allocator
        if allocator.available_size() < num_tokens:
            tree_cache.evict(num_tokens)


def available_and_evictable_str(tree_cache) -> str:
    token_to_kv_pool_allocator = tree_cache.token_to_kv_pool_allocator
    if isinstance(token_to_kv_pool_allocator, SWATokenToKVPoolAllocator):
        full_available_size = token_to_kv_pool_allocator.full_available_size()
        swa_available_size = token_to_kv_pool_allocator.swa_available_size()
        full_evictable_size = tree_cache.full_evictable_size()
        swa_evictable_size = tree_cache.swa_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Available swa tokens: {swa_available_size + swa_evictable_size} ({swa_available_size=} + {swa_evictable_size=})\n"
        )
    else:
        available_size = token_to_kv_pool_allocator.available_size()
        evictable_size = tree_cache.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"


def release_kv_cache(req, tree_cache, is_insert: bool = True) -> None:
    """Single entry point for releasing all KV cache held by a finished req.

    Replaces scattered ``tree_cache.cache_finished_req(req)`` plus ad-hoc
    ``out_cache_loc[i:i+1]`` frees that previously caused double-frees.

    Steps:
      1. Delegate committed-range free to the tree cache's cache_finished_req.
      2. Free over-allocated range [committed_len, allocated_len) (page-aligned).
      3. Free the req_to_token_pool slot.

    Args:
        req: the finished Req.
        tree_cache: a RadixCache, ChunkCache, or compatible BasePrefixCache.
        is_insert: kept for upstream-API compatibility; unused in sgl-jax today.
    """
    from sgl_jax.srt.utils.common_utils import cdiv

    if req.req_pool_idx is None:
        # Already released (e.g., retract path, or streaming session early-free).
        return

    # Step 1: free committed KV via tree cache (radix evictable insert OR direct free).
    tree_cache.cache_finished_req(req)
    if req.req_pool_idx is None:
        return

    # Step 2: free over-allocated KV in [start_p, end_p).
    start_p, end_p = req.pop_overallocated_kv_cache()
    page_size = tree_cache.page_size
    if page_size > 1:
        # Align start_p UP to next page boundary; the partial page before start_p
        # is part of the committed range and was handled by cache_finished_req.
        start_p = cdiv(start_p, page_size) * page_size

    if start_p < end_p:
        indices = tree_cache.req_to_token_pool.req_to_token[req.req_pool_idx, start_p:end_p]
        # Filter zeros (uninitialized slots); same defensive pattern as cache_finished_req.
        indices = indices[indices != 0]
        if indices.size > 0:
            tree_cache.token_to_kv_pool_allocator.free(indices)

    # Step 3: release the req_to_token_pool slot.
    tree_cache.req_to_token_pool.free(req.req_pool_idx)
