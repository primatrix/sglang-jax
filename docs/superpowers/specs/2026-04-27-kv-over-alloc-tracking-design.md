# Design Spec: KV Cache Over-Allocation Tracking (sgl-jax)

**Date:** 2026-04-27
**Branch:** `worktree-fix-radix-cache` (worktree of `merge/dp`)
**Final spec destination:** `docs/superpowers/specs/2026-04-27-kv-over-alloc-tracking-design.md` (move + commit after plan approval)

---

## Context

`test/srt/test_engine_determine_generation.py` (introduced by PR #515) fails on `merge/dp` with `token_to_kv_pool_allocator memory leak detected!`. Root cause is **not** the abort/retract path — even a single `engine.async_generate` with no abort triggers the same hard assertion.

**Diagnosis (verified via 4-config matrix on Qwen3-8B + tp=4 + chunked_prefill=4 + single 13-token prompt + max_new_tokens=200, brian-deepseek-test pod v6e 2x2):**

| Config | Result | over-count |
|---|---|---|
| page=4 + radix on  | LEAK    | +4 |
| page=4 + radix off | NO_LEAK | 0  |
| page=1 + radix on  | LEAK    | +1 |
| page=1 + radix off | LEAK    | +1 |

A fixed 1-slot double-free; `PagedAllocator.setdiff1d` defense masks it for `page=4 + radix off`; `page=4 + radix on` amplifies to +4 because the page-granularity formula multiplies the leaked sub-page slot by `page_size`.

**The double-free is structural:** sgl-jax has KV release scattered across multiple paths:
1. `RadixCache.cache_finished_req` (`mem_cache/radix_cache.py:257-303`) infers committed length via `len(input) + max(len(output) - 1, 0)`, frees that range.
2. `process_batch_result_decode` (`scheduler_output_processor_mixin.py:284-293`) ad-hoc frees `out_cache_loc[i:i+1]` in the overlap+finished branch, then `continue`s.

The decode-step-N slot is included in BOTH paths → freed twice → allocator's `available + evict + protected` exceeds `max_total_num_tokens` → `scheduler.py:1060-1089` raises.

**Upstream sglang fix pattern:** `kv_committed_len` / `kv_allocated_len` fields on `Req` + a single `release_kv_cache(req, tree_cache)` entry point that orchestrates `cache_finished_req` → over-alloc range free → `req_to_token_pool.free`. sgl-jax has none of this.

**Intended outcome:** All 4 (page_size, radix on/off) baseline configurations of `test_engine_determine_generation.py::test_1` pass with no leak. PR #515's abort/retract path is a separate bug being fixed in another worktree; this spec covers only the baseline KV leak.

---

## Approach

Port upstream's `kv_committed_len` / `kv_allocated_len` tracking model and `release_kv_cache` single entry point, adapted for sgl-jax. Scope: **minimal** — fix only the baseline 4-config leak. EAGLE finished free path (`mixin:305-321`) and abort/retract paths are out of scope.

### Architecture changes

```
Before:                           After:
─────────                         ──────
mixin.prefill finished:           mixin.prefill finished:
  cache_finished_req(req)           release_kv_cache(req, tree_cache)
                                      ├─ tree_cache.cache_finished_req(req)
mixin.decode finished:                │    └─ frees committed KV [protected, committed)
  cache_finished_req(req)             ├─ pop_overallocated_kv_cache → (start_p, end_p)
                                      ├─ frees over-alloc KV [start_p, end_p)
mixin.prefill overlap:                └─ req_to_token_pool.free(req.req_pool_idx)
  free out_cache_loc[j:j+1]
  continue                          mixin.prefill overlap: REMOVED
                                    mixin.decode overlap:  REMOVED
mixin.decode overlap:               (over-alloc tracked by kv_allocated_len > kv_committed_len,
  free out_cache_loc[i:i+1]          freed via release_kv_cache)
  continue
```

### Components

**1. `Req` class — 5 new fields** (`python/sgl_jax/srt/managers/schedule_batch.py`, ~line 312):
```python
self.kv_committed_len: int = 0
self.kv_allocated_len: int = 0
self.cache_protected_len: int = 0
self.kv_committed_freed: bool = False
self.kv_overallocated_freed: bool = False
```

**2. `Req` — 2 new pop methods** (after existing methods, ~line 500):
```python
def pop_committed_kv_cache(self) -> int:
    assert not self.kv_committed_freed, "double pop_committed_kv_cache"
    self.kv_committed_freed = True
    return self.kv_committed_len

def pop_overallocated_kv_cache(self) -> Tuple[int, int]:
    assert not self.kv_overallocated_freed, "double pop_overallocated_kv_cache"
    self.kv_overallocated_freed = True
    return self.kv_committed_len, self.kv_allocated_len
```

**3. Field maintenance** (4 hook points):
- `prepare_for_extend` (~`schedule_batch.py:851`): `req.kv_committed_len = seq_len; req.kv_allocated_len = seq_len` after the existing `req.already_computed = seq_len`
- `prepare_for_decode` (~`schedule_batch.py:1152`): in the `for req in self.reqs` loop, `req.kv_committed_len += 1; req.kv_allocated_len += 1`
- `reset_for_retract` (`schedule_batch.py:501-519`): reset all 5 fields to `0` / `False`
- `init_next_round_input` equivalent — see §Open question below

**4. New `release_kv_cache` function** (`python/sgl_jax/srt/mem_cache/common.py`, append after `available_and_evictable_str`):
```python
def release_kv_cache(req: Req, tree_cache, is_insert: bool = True) -> None:
    """Single entry point for releasing all KV cache held by a finished req."""
    if req.req_pool_idx is None:
        return  # already freed (e.g., retract path)

    # Step 1: commit + free committed KV via tree cache
    tree_cache.cache_finished_req(req)
    if req.req_pool_idx is None:
        return

    # Step 2: free over-allocated KV range [start_p, end_p)
    start_p, end_p = req.pop_overallocated_kv_cache()
    page_size = tree_cache.page_size
    if page_size > 1:
        start_p = ceil_align(start_p, page_size)
    if start_p < end_p:
        indices = tree_cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, start_p:end_p
        ]
        indices = indices[indices != 0]
        tree_cache.token_to_kv_pool_allocator.free(indices)

    # Step 3: release req_to_token slot
    tree_cache.req_to_token_pool.free(req.req_pool_idx)
```

(`is_insert` accepted for upstream-compat; threaded through to `cache_finished_req` if/when needed — currently sgl-jax `cache_finished_req` doesn't take it; leave as no-op param for now.)

**5. `RadixCache.cache_finished_req`** (`python/sgl_jax/srt/mem_cache/radix_cache.py:257-303`) — 4 changes:
- Replace line 259 (`all_token_len = ...`) with `committed_len = req.pop_committed_kv_cache()`; use `committed_len` everywhere `all_token_len` appears
- Replace `old_prefix_len = len(req.prefix_indices)` (line 286) with `old_prefix_len = req.cache_protected_len`
- Remove `self.req_to_token_pool.free(req.req_pool_idx)` (line 302) — handed off to `release_kv_cache`
- **Fix existing page-tail double-free bug:** lines 280 + 299 both free `kv_indices[page_aligned_len:]` when `page_size > 1`. Remove the line-280 free; keep line-299 (which runs unconditionally). Sanity-check: when `page_size == 1`, `page_aligned_len == actual_kv_len == len(kv_indices)`, so line-299 frees an empty slice — fine.

**6. `ChunkCache.cache_finished_req`** (`python/sgl_jax/srt/mem_cache/chunk_cache.py:36-43`) — 2 changes:
- Replace `len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)` with `req.pop_committed_kv_cache()`
- Remove `self.req_to_token_pool.free(req.req_pool_idx)` — handed off to `release_kv_cache`

**7. `scheduler_output_processor_mixin.py`** — 4 changes:
- Line 125: `self.tree_cache.cache_finished_req(req)` → `release_kv_cache(req, self.tree_cache)`
- Line 338: same substitution
- Lines 98-101: **delete** entire prefill-overlap-finished ad-hoc free block (over-alloc now tracked & freed via `release_kv_cache`)
- Lines 284-293: **delete** entire decode-overlap-finished ad-hoc free block (same reason)
- Add `from sgl_jax.srt.mem_cache.common import release_kv_cache` import

**8. EAGLE finished free** (`scheduler_output_processor_mixin.py:305-321`) — **NOT touched** in this spec. EAGLE allocates `cur_allocate_len > all_token_len` and frees the diff inline before `cache_finished_req`. Once `release_kv_cache` is in, this path will free the EAGLE over-alloc, then `release_kv_cache` will free committed + (now-zero) over-alloc + req_pool. As long as EAGLE doesn't update `kv_allocated_len`, `pop_overallocated_kv_cache` returns `(committed_len, committed_len)` → empty range → no-op. This is correct but means EAGLE keeps its ad-hoc pattern; migrating it is a future PR.

### Critical files to modify

| File | Lines (approx) | Change |
|---|---|---|
| `python/sgl_jax/srt/managers/schedule_batch.py` | 312, 501-519, ~840-851, ~1140-1153, +new methods | 5 fields, reset, set in extend, increment in decode, 2 pop methods, `cache_protected_len` set point |
| `python/sgl_jax/srt/mem_cache/common.py` | append at end | new `release_kv_cache` |
| `python/sgl_jax/srt/mem_cache/radix_cache.py` | 257-303 | use `pop_committed_kv_cache`, use `cache_protected_len`, remove `req_to_token_pool.free`, fix page-tail double free |
| `python/sgl_jax/srt/mem_cache/chunk_cache.py` | 36-43 | use `pop_committed_kv_cache`, remove `req_to_token_pool.free` |
| `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py` | 98-101 (delete), 125, 284-293 (delete), 338, +import | 2 delete blocks, 2 call replacements |

### Reused existing utilities

- `ReqToTokenPool.free` (`memory_pool.py:133-138`) — accepts `int` or `list[int]`; we pass `req.req_pool_idx` (int).
- `TokenToKVPoolAllocator.free` / `PagedTokenToKVPoolAllocator.free` (`allocator.py:116-123`, `316-332`) — already handle empty slice and grouped frees correctly.
- `free_group_begin`/`free_group_end` (`mixin:274/386`) — `release_kv_cache` runs inside this group; all internal `free` calls go to the group buffer. **Do not** add nested `free_group_begin/end` in `release_kv_cache`.
- `ceil_align` — needs to exist somewhere in sgl-jax; if not, write inline as `((x + page_size - 1) // page_size) * page_size`.

---

## Open question (must resolve before coding)

**Where to set `req.cache_protected_len`?**

Upstream sets it in `Req.init_next_round_input` (`schedule_batch.py:1038-1041`) after prefix matching. sgl-jax does prefix matching in `Scheduler.add_request` and possibly in `cache_unfinished_req`. The protected length is `len(req.prefix_indices)` at the moment prefill begins.

Resolution plan (during writing-plans phase):
1. `grep -rn "prefix_indices = " python/sgl_jax/srt/` to find all assignment sites.
2. Identify the LAST assignment before `prepare_for_extend` runs.
3. Add `req.cache_protected_len = len(req.prefix_indices)` immediately after that assignment.

**Risk if mis-placed:** prefix tokens get double-freed (placed too early, before lock is taken) or under-freed (placed too late, after some prefix already released). The 4-config test will catch both.

---

## Risks

1. **`cache_protected_len` placement** (above) — highest-risk single decision. Resolve via grep + verification before writing the change.
2. **Existing page-tail double-free** (radix_cache.py:280 + 299) — must be fixed in this PR, otherwise page>1 + radix on still leaks even with new tracking.
3. **DP grouping in `prepare_for_decode`** — confirm the `for req in self.reqs` loop iterates ALL reqs across DP ranks (not just one rank's slice). If sgl-jax splits decode by DP, the increment must run for every rank's reqs.
4. **`free_group` nesting** — `release_kv_cache` is called inside `free_group_begin/end`. Do not call `free_group_begin/end` inside `release_kv_cache`.
5. **EAGLE path correctness** — left untouched but verify by inspection: `release_kv_cache` running after EAGLE's inline free should be a no-op for the over-alloc range (EAGLE doesn't touch `kv_allocated_len`, so `kv_committed_len == kv_allocated_len` → empty range).

---

## Verification

End-to-end test on TPU pod (e.g., `brian-deepseek-test`, v6e 2x2):

```bash
cd /Users/ramezes/job/sgl-project/sgl-jax/.claude/worktrees/fix-radix-cache

# 4-config matrix — single prompt baseline must NOT trigger memory leak detection
for page in 1 4; do
  for radix_flag in "" "--disable-radix-cache"; do
    echo "=== page=${page} radix_flag='${radix_flag}' ==="
    python -m pytest test/srt/test_engine_determine_generation.py::TestEngineDetermineGeneration::test_1 \
      -v --tb=short
    # (test launches engine internally; pass page/radix via test fixture or env if test supports it,
    # otherwise run minimal repro script directly with engine.async_generate + 13-token prompt + max_new_tokens=200)
  done
done
```

**Pass criteria:**
- All 4 configs: `scheduler.py:1060-1089` `check_memory` invariant `available + evictable + protected == max_total_num_tokens` holds throughout the run; no `ValueError: token_to_kv_pool_allocator memory leak detected!`.
- `test_1` (single retract baseline) passes — confirms baseline generation is unaffected.

**Optional regression spot-check:**
- `test/srt/test_srt_engine.py::test_smoke` — 1 run on default config to confirm no functional regression in the happy path.

**Out of scope:**
- `test_2/test_3/test_4` (abort/multi-req retract) — those require the abort path fix from the sibling worktree.
- EAGLE / speculative decode tests.
