# KV Cache Over-Allocation Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the structural KV double-free in sgl-jax that triggers `token_to_kv_pool_allocator memory leak detected!` on every plain `engine.async_generate` call. Port upstream sglang's `kv_committed_len` / `kv_allocated_len` tracking + single `release_kv_cache` entry point.

**Architecture:** Each `Req` tracks how many KV slots are committed vs. allocated. All finished-req KV release goes through one `release_kv_cache(req, tree_cache)` function in `mem_cache/common.py`. The two ad-hoc `out_cache_loc[i:i+1]` frees in `scheduler_output_processor_mixin.py` (the actual double-free source) are deleted. EAGLE/abort/retract paths are out of scope.

**Tech Stack:** Python, JAX, sgl-jax (worktree at `.claude/worktrees/fix-radix-cache`, branch `worktree-fix-radix-cache`).

**Spec:** `docs/superpowers/specs/2026-04-27-kv-over-alloc-tracking-design.md`

---

## File Structure

| File | Responsibility | Change type |
|---|---|---|
| `python/sgl_jax/srt/managers/schedule_batch.py` | `Req` data model + batch lifecycle (extend/decode/retract) | 5 new fields, 2 new methods, 4 maintenance hooks |
| `python/sgl_jax/srt/mem_cache/common.py` | Allocator helpers | Add `release_kv_cache` function |
| `python/sgl_jax/srt/mem_cache/radix_cache.py` | Radix tree KV cache | Refactor `cache_finished_req` to use new fields, fix page-tail double free, stop freeing `req_to_token_pool` slot |
| `python/sgl_jax/srt/mem_cache/chunk_cache.py` | Disabled-radix KV cache | Refactor `cache_finished_req` to use new fields, stop freeing `req_to_token_pool` slot |
| `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py` | Per-step batch result handling | Delete 2 ad-hoc free blocks, replace 2 `cache_finished_req` calls with `release_kv_cache` |

---

## Task 1: Add 5 new fields and 2 pop methods to `Req`

**Files:**
- Modify: `python/sgl_jax/srt/managers/schedule_batch.py:251` (add fields after `last_matched_prefix_len`)
- Modify: `python/sgl_jax/srt/managers/schedule_batch.py:~410` (add pop methods after `init_next_round_input` block, before `check_finished`)

- [ ] **Step 1.1: Add 5 fields in `Req.__init__`**

In `python/sgl_jax/srt/managers/schedule_batch.py`, find:

```python
        # The prefix length of the last prefix matching
        self.last_matched_prefix_len: int = 0
```

Add immediately after:

```python
        # KV cache lifecycle tracking (matches upstream sglang Req fields).
        # committed_len: number of tokens whose KV is safe to commit to radix evictable.
        # allocated_len: number of token slots currently allocated for this req.
        # cache_protected_len: prefix length locked into radix at prefill start; not freed by cache_finished_req.
        # *_freed flags guard against double pop_*_kv_cache calls.
        self.kv_committed_len: int = 0
        self.kv_allocated_len: int = 0
        self.cache_protected_len: int = 0
        self.kv_committed_freed: bool = False
        self.kv_overallocated_freed: bool = False
```

- [ ] **Step 1.2: Add `pop_committed_kv_cache` and `pop_overallocated_kv_cache`**

Find the end of `init_next_round_input` (around line 387, ends with `self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)`). Insert these two methods before `def adjust_max_prefix_ids` (around line 389):

```python
    def pop_committed_kv_cache(self) -> int:
        """Return committed KV length and mark as released. Single use per finished req."""
        assert (
            not self.kv_committed_freed
        ), f"double pop_committed_kv_cache for req {self.rid}"
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self) -> tuple[int, int]:
        """Return [committed_len, allocated_len) over-alloc range. Single use per finished req."""
        assert (
            not self.kv_overallocated_freed
        ), f"double pop_overallocated_kv_cache for req {self.rid}"
        self.kv_overallocated_freed = True
        return self.kv_committed_len, self.kv_allocated_len
```

- [ ] **Step 1.3: Run a syntax / import sanity check**

```bash
cd /Users/ramezes/job/sgl-project/sgl-jax/.claude/worktrees/fix-radix-cache
python -c "from sgl_jax.srt.managers.schedule_batch import Req; r = Req('rid', 'p', [1,2,3], None); print(r.kv_committed_len, r.kv_allocated_len, r.cache_protected_len, r.kv_committed_freed, r.kv_overallocated_freed)"
```

Expected: `0 0 0 False False`

- [ ] **Step 1.4: Verify pop methods**

```bash
python -c "
from sgl_jax.srt.managers.schedule_batch import Req
r = Req('rid', 'p', [1,2,3], None)
r.kv_committed_len = 10
r.kv_allocated_len = 12
print('committed:', r.pop_committed_kv_cache())
print('over:', r.pop_overallocated_kv_cache())
try:
    r.pop_committed_kv_cache()
    print('FAIL: double-pop should have asserted')
except AssertionError as e:
    print('OK assertion:', e)
"
```

Expected: `committed: 10`, `over: (10, 12)`, `OK assertion: double pop_committed_kv_cache for req rid`

- [ ] **Step 1.5: Commit**

```bash
git add python/sgl_jax/srt/managers/schedule_batch.py
git commit -m "feat(req): add kv_committed_len/kv_allocated_len tracking fields and pop methods"
```

---

## Task 2: Maintain new fields in extend / decode / retract / init_next_round_input

**Files:**
- Modify: `python/sgl_jax/srt/managers/schedule_batch.py:386-387` (init_next_round_input — set `cache_protected_len`)
- Modify: `python/sgl_jax/srt/managers/schedule_batch.py:501-519` (`reset_for_retract` — reset 5 fields)
- Modify: `python/sgl_jax/srt/managers/schedule_batch.py:851` (`prepare_for_extend` — set committed/allocated to seq_len)
- Modify: `python/sgl_jax/srt/managers/schedule_batch.py:1151-1153` (`prepare_for_decode` — increment committed/allocated)

- [ ] **Step 2.1: Set `cache_protected_len` in `init_next_round_input`**

Find lines 385-387:

```python
            self.last_matched_prefix_len = len(self.prefix_indices)
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)
```

Change to:

```python
            self.last_matched_prefix_len = len(self.prefix_indices)
        self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)
        # cache_protected_len is the prefix range already in radix (locked); cache_finished_req
        # must NOT free this range, since dec_lock_ref handles that side.
        self.cache_protected_len = len(self.prefix_indices)
```

- [ ] **Step 2.2: Reset 5 fields in `reset_for_retract`**

Find lines 501-519. The current method ends with:

```python
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0
```

Add immediately after:

```python
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.cache_protected_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
```

- [ ] **Step 2.3: Set committed/allocated in `prepare_for_extend`**

Find line 851 (inside the `for i, (req, seq_len, pre_len) in enumerate(zip(...))` loop):

```python
            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False
            req.extend_batch_idx += 1
```

Change to:

```python
            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False
            req.extend_batch_idx += 1
            # After prefill, all `seq_len` tokens have committed KV and are allocated.
            req.kv_committed_len = seq_len
            req.kv_allocated_len = seq_len
```

- [ ] **Step 2.4: Increment committed/allocated in `prepare_for_decode`**

Find lines 1151-1153:

```python
        for req in self.reqs:
            req.decode_batch_idx += 1
```

Change to:

```python
        for req in self.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1
```

(Note: this loop is NOT reached for `spec_algorithm.is_eagle()` due to early return at line 1109. EAGLE keeps its own ad-hoc free path — out of scope.)

- [ ] **Step 2.5: Sanity test — fields propagate through prefill+decode**

```bash
cd /Users/ramezes/job/sgl-project/sgl-jax/.claude/worktrees/fix-radix-cache
python -c "
from sgl_jax.srt.managers.schedule_batch import Req
r = Req('rid', 'p', [1,2,3], None)
# simulate post-extend
r.kv_committed_len = 13
r.kv_allocated_len = 13
# simulate 5 decode steps
for _ in range(5):
    r.kv_committed_len += 1
    r.kv_allocated_len += 1
assert r.kv_committed_len == 18 and r.kv_allocated_len == 18, (r.kv_committed_len, r.kv_allocated_len)
# simulate retract
r.reset_for_retract()
assert r.kv_committed_len == 0 and r.kv_allocated_len == 0 and not r.kv_committed_freed
print('OK')
"
```

Expected: `OK`

- [ ] **Step 2.6: Commit**

```bash
git add python/sgl_jax/srt/managers/schedule_batch.py
git commit -m "feat(req): maintain kv_committed_len/kv_allocated_len across extend/decode/retract"
```

---

## Task 3: Add `release_kv_cache` to `mem_cache/common.py`

**Files:**
- Modify: `python/sgl_jax/srt/mem_cache/common.py` (append at end)

- [ ] **Step 3.1: Add the function**

Open `python/sgl_jax/srt/mem_cache/common.py`. The file currently ends with `available_and_evictable_str` around line 102. Append at the end of the file:

```python


def release_kv_cache(req, tree_cache, is_insert: bool = True) -> None:
    """Single entry point for releasing all KV cache held by a finished req.

    Replaces scattered `tree_cache.cache_finished_req(req)` + ad-hoc
    `out_cache_loc[i:i+1]` frees that previously caused double-frees.

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
        indices = tree_cache.req_to_token_pool.req_to_token[
            req.req_pool_idx, start_p:end_p
        ]
        # Filter zeros (uninitialized slots); same defensive pattern as cache_finished_req.
        indices = indices[indices != 0]
        if indices.size > 0:
            tree_cache.token_to_kv_pool_allocator.free(indices)

    # Step 3: release the req_to_token_pool slot.
    tree_cache.req_to_token_pool.free(req.req_pool_idx)
```

- [ ] **Step 3.2: Sanity import check**

```bash
cd /Users/ramezes/job/sgl-project/sgl-jax/.claude/worktrees/fix-radix-cache
python -c "from sgl_jax.srt.mem_cache.common import release_kv_cache; print(release_kv_cache.__doc__.split(chr(10))[0])"
```

Expected: `Single entry point for releasing all KV cache held by a finished req.`

- [ ] **Step 3.3: Commit**

```bash
git add python/sgl_jax/srt/mem_cache/common.py
git commit -m "feat(mem_cache): add release_kv_cache as single KV release entry point"
```

---

## Task 4: Refactor `RadixCache.cache_finished_req`

**Files:**
- Modify: `python/sgl_jax/srt/mem_cache/radix_cache.py:257-303`

This task fixes 4 things at once:
- Use `req.pop_committed_kv_cache()` (not the `len(input)+max(len(output)-1,0)` formula)
- Use `req.cache_protected_len` (not `len(req.prefix_indices)`)
- Stop calling `req_to_token_pool.free` (handed off to `release_kv_cache`)
- Fix the existing page-tail double-free (line 280 + line 299 both freeing `kv_indices[page_aligned_len:]` when `page_size > 1`)

- [ ] **Step 4.1: Replace the entire method**

Find the current `cache_finished_req` at `python/sgl_jax/srt/mem_cache/radix_cache.py:257-303`:

```python
    def cache_finished_req(self, req: Req):
        """Cache completed requests"""
        all_token_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        if self.disable:
            kv_indices = self.req_to_token_pool.read(
                req.req_pool_idx,
                all_token_len,
            )
            kv_indices = kv_indices[kv_indices != 0]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:all_token_len]
        # For EAGLE radix cache, we will convert the key to bigram key, e.g. [1,2,3,4] -> [(1,2), (2,3), (3,4)], the length will -1. ((len([(1,2), (2,3), (3,4)]) = len([1,2,3,4]) - 1))
        # So for the corresponding kv length should also -1. Then we get the actual_kv_len, and use it to do later calculation and slicing.
        actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
        kv_indices = self.req_to_token_pool.read(req.req_pool_idx, all_token_len)
        kv_indices = kv_indices[kv_indices != 0]

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()

        page_aligned_token_len = page_aligned_len + 1 if self.is_eagle else page_aligned_len
        old_prefix_len = len(req.prefix_indices)
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes over one reference from memory pool
        new_prefix_len = self.insert(
            RadixKey(token_ids[:page_aligned_token_len], req.extra_key), page_aligned_kv_indices
        )

        self.token_to_kv_pool_allocator.free(kv_indices[old_prefix_len:new_prefix_len])
        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])

        # Remove request slot and release cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)
```

Replace with:

```python
    def cache_finished_req(self, req: Req):
        """Cache completed requests.

        Frees the committed KV range [cache_protected_len, committed_len). Does NOT
        free the req_to_token_pool slot — that's owned by release_kv_cache.
        """
        committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.read(
                req.req_pool_idx,
                committed_len,
            )
            kv_indices = kv_indices[kv_indices != 0]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:committed_len]
        # For EAGLE radix cache, the key is bigram, so kv length is -1.
        actual_kv_len = committed_len - 1 if self.is_eagle else committed_len
        kv_indices = self.req_to_token_pool.read(req.req_pool_idx, committed_len)
        kv_indices = kv_indices[kv_indices != 0]

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()
            # NOTE: the page-tail free for kv_indices[page_aligned_len:] is done
            # unconditionally below (after radix insert). Do NOT free it here too —
            # that was the historical double-free (page>1 + radix on amplifies x page_size).
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()

        page_aligned_token_len = page_aligned_len + 1 if self.is_eagle else page_aligned_len
        old_prefix_len = req.cache_protected_len
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes over one reference from memory pool
        new_prefix_len = self.insert(
            RadixKey(token_ids[:page_aligned_token_len], req.extra_key), page_aligned_kv_indices
        )

        self.token_to_kv_pool_allocator.free(kv_indices[old_prefix_len:new_prefix_len])
        # free the unaligned tail (single source — see NOTE above)
        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])

        # Release cache lock. req_to_token_pool slot is freed by release_kv_cache.
        self.dec_lock_ref(req.last_node)
```

- [ ] **Step 4.2: Sanity import**

```bash
cd /Users/ramezes/job/sgl-project/sgl-jax/.claude/worktrees/fix-radix-cache
python -c "from sgl_jax.srt.mem_cache.radix_cache import RadixCache; print('ok')"
```

Expected: `ok`

- [ ] **Step 4.3: Commit**

```bash
git add python/sgl_jax/srt/mem_cache/radix_cache.py
git commit -m "fix(radix_cache): use pop_committed_kv_cache + cache_protected_len, fix page-tail double free

- Replace indirect committed-length formula with req.pop_committed_kv_cache()
- Replace len(req.prefix_indices) with req.cache_protected_len (stable)
- Remove duplicate page-tail free (line 280 was duplicated by line 299)
- Stop freeing req_to_token_pool slot (now owned by release_kv_cache)"
```

---

## Task 5: Refactor `ChunkCache.cache_finished_req`

**Files:**
- Modify: `python/sgl_jax/srt/mem_cache/chunk_cache.py:36-43`

- [ ] **Step 5.1: Replace the method**

Find:

```python
    def cache_finished_req(self, req: Req):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            : len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
        ]
        self.req_to_token_pool.free(req.req_pool_idx)
        self.token_to_kv_pool_allocator.free(kv_indices)
```

Replace with:

```python
    def cache_finished_req(self, req: Req):
        committed_len = req.pop_committed_kv_cache()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx,
            :committed_len,
        ]
        # req_to_token_pool slot is freed by release_kv_cache.
        self.token_to_kv_pool_allocator.free(kv_indices)
```

- [ ] **Step 5.2: Sanity import**

```bash
cd /Users/ramezes/job/sgl-project/sgl-jax/.claude/worktrees/fix-radix-cache
python -c "from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache; print('ok')"
```

Expected: `ok`

- [ ] **Step 5.3: Commit**

```bash
git add python/sgl_jax/srt/mem_cache/chunk_cache.py
git commit -m "fix(chunk_cache): use pop_committed_kv_cache, defer req_pool free to release_kv_cache"
```

---

## Task 6: Replace mixin's scattered free calls with `release_kv_cache`

**Files:**
- Modify: `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py` (4 changes + 1 import)

- [ ] **Step 6.1: Add import**

At the top of `python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py`, find the imports block. Add (alphabetically appropriate place near other `mem_cache` imports if any, otherwise near other `sgl_jax.srt` imports):

```python
from sgl_jax.srt.mem_cache.common import release_kv_cache
```

- [ ] **Step 6.2: Delete prefill overlap ad-hoc free (lines 98-101)**

Find:

```python
            if self.is_mixed_chunk and self.enable_overlap and req.finished():
                j = len(batch.out_cache_loc) - len(batch.reqs) + i
                self.token_to_kv_pool_allocator.free(batch.out_cache_loc[j : j + 1])
                continue
```

**Delete this entire 4-line block.** Reasoning: the over-alloc tracking now records that this 1 extra slot was allocated (via `kv_allocated_len += 1` at decode step before this finish was detected; for prefill mixed-chunk the slot is part of `out_cache_loc` which is implicit in `seq_len` allocated). The slot will be freed by `release_kv_cache`'s over-alloc step.

- [ ] **Step 6.3: Replace prefill `cache_finished_req` call (line 125)**

Find:

```python
                    self.tree_cache.cache_finished_req(req)
```

(Inside the `if req.finished():` branch at line 108-125.)

Replace with:

```python
                    release_kv_cache(req, self.tree_cache)
```

- [ ] **Step 6.4: Delete decode overlap ad-hoc free (lines 284-293)**

Find:

```python
            indices_to_free = None
            if self.enable_overlap and req.finished():
                if self.page_size == 1:
                    indices_to_free = batch.out_cache_loc[i : i + 1]
                else:
                    if (len(req.origin_input_ids) + len(req.output_ids) - 1) % self.page_size == 0:
                        indices_to_free = batch.out_cache_loc[i : i + 1]
                if indices_to_free is not None:
                    self.token_to_kv_pool_allocator.free(indices_to_free)
                continue
```

**Delete this entire block.** Reasoning: same as 6.2 — over-alloc tracking + `release_kv_cache` handles the extra slot.

- [ ] **Step 6.5: Replace decode `cache_finished_req` call (line 338)**

Find:

```python
                self.tree_cache.cache_finished_req(req)
```

(Inside the decode `if req.finished():` branch, after the EAGLE block at lines 305-321.)

Replace with:

```python
                release_kv_cache(req, self.tree_cache)
```

- [ ] **Step 6.6: Sanity import**

```bash
cd /Users/ramezes/job/sgl-project/sgl-jax/.claude/worktrees/fix-radix-cache
python -c "from sgl_jax.srt.managers.scheduler_output_processor_mixin import SchedulerOutputProcessorMixin; print('ok')" 2>&1 | tail -3
```

Expected: `ok` (or, if the class name differs, just no ImportError — `python -c "import sgl_jax.srt.managers.scheduler_output_processor_mixin"` is the fallback check).

- [ ] **Step 6.7: Commit**

```bash
git add python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py
git commit -m "fix(scheduler): route finished-req KV release through single release_kv_cache entry point

Removes the two ad-hoc out_cache_loc[i:i+1] frees in overlap+finished branches
that were the source of the structural double-free. Over-allocated slots are now
tracked via Req.kv_allocated_len > kv_committed_len and freed inside release_kv_cache."
```

---

## Task 7: End-to-end verification on brian-deepseek-test pod

**Files:** none modified — verification only.

The verification environment is the GKE TPU pod `brian-deepseek-test` (v6e 2x2). The 4-config matrix from the spec must all pass.

- [ ] **Step 7.1: Sync code to the pod**

Check if there is a sync script (likely `gke-tpu` skill helper or a project script):

```bash
cd /Users/ramezes/job/sgl-project/sgl-jax/.claude/worktrees/fix-radix-cache
ls gke.toml 2>/dev/null && echo "gke.toml present" || echo "no gke.toml"
```

If `gke.toml` is present, use the `gke-tpu` skill to sync; otherwise sync via whatever script the project uses (e.g. `scripts/sync.sh`, `rsync ...`). Defer the exact sync command to the operator.

- [ ] **Step 7.2: Run config 1 — page_size=1, radix on (default)**

On the pod (or via `gke-tpu` exec):

```bash
cd /path/to/sgl-jax/on/pod
python -m pytest test/srt/test_engine_determine_generation.py::TestEngineDetermineGeneration::test_1 -v --tb=short 2>&1 | tee /tmp/leak-test-page1-radix-on.log
```

Pass criteria: test exits 0, no `token_to_kv_pool_allocator memory leak detected!` in the log.

If the test does not accept page_size / radix as command-line args, run a minimal repro script (write a small `repro.py` that calls `sgl_jax.Engine(model_path=..., page_size=1)` then `engine.async_generate(prompt, sampling_params={"max_new_tokens": 200})` and checks `engine.tokenizer_manager.scheduler` for memory check assertions). The check that fires is `scheduler.py:1060-1089`.

- [ ] **Step 7.3: Run config 2 — page_size=1, radix off**

```bash
python -m pytest test/srt/test_engine_determine_generation.py::TestEngineDetermineGeneration::test_1 -v --tb=short 2>&1 | tee /tmp/leak-test-page1-radix-off.log
```

(With `--disable-radix-cache` arg or env var; check the test's launcher to pass it through. If the test fixture doesn't expose it, write the minimal repro script and pass `disable_radix_cache=True` to `Engine(...)`.)

Pass criteria: same as 7.2.

- [ ] **Step 7.4: Run config 3 — page_size=4, radix on**

```bash
# (with page_size=4 set)
python -m pytest test/srt/test_engine_determine_generation.py::TestEngineDetermineGeneration::test_1 -v --tb=short 2>&1 | tee /tmp/leak-test-page4-radix-on.log
```

Pass criteria: same. **This is the highest-value config — it had +4 leak before, hardest to fool.**

- [ ] **Step 7.5: Run config 4 — page_size=4, radix off**

```bash
python -m pytest test/srt/test_engine_determine_generation.py::TestEngineDetermineGeneration::test_1 -v --tb=short 2>&1 | tee /tmp/leak-test-page4-radix-off.log
```

Pass criteria: same.

- [ ] **Step 7.6: Spot-check no functional regression**

Run one happy-path engine test with default settings:

```bash
python -m pytest test/srt/test_srt_engine.py -v --tb=short -k "smoke or basic" 2>&1 | tail -20
```

Pass criteria: any matching tests still pass. Record any unrelated failures separately (do not block merge on them unless caused by this change — diff against main if uncertain).

- [ ] **Step 7.7: Final commit (only if any cleanup done) + push**

If verification surfaced bugs, fix them in NEW commits (not amend) and re-run the failing config(s). When all 4 configs green:

```bash
git log --oneline merge/dp..HEAD   # confirm 6 clean commits
git push prim worktree-fix-radix-cache   # respect the "only push to prim" rule
```

(Do not open PR yet — hand back to user for manual sanity / PR description draft.)

---

## Verification summary

| Step | Config | Pre-fix | Post-fix expected |
|---|---|---|---|
| 7.2 | page=1, radix on  | LEAK +1 | NO_LEAK |
| 7.3 | page=1, radix off | LEAK +1 | NO_LEAK |
| 7.4 | page=4, radix on  | LEAK +4 | NO_LEAK |
| 7.5 | page=4, radix off | NO_LEAK (masked) | NO_LEAK |
| 7.6 | smoke tests       | pass | pass |

If any of 7.2-7.5 still leaks, the most likely culprits in priority order:
1. `cache_protected_len` set wrong (check `init_next_round_input` change in Task 2.1)
2. Forgotten increment site (e.g., chunked prefill resumes prefill instead of decoding — check if there's a `prepare_for_continued_extend`)
3. `release_kv_cache` over-alloc range zero-filtering hides a real slot (re-add a print of `start_p, end_p, indices` before the free in `common.py`)

---

## Out of scope (do NOT touch in this PR)

- EAGLE finished free path (`scheduler_output_processor_mixin.py:305-321`)
- Abort path bugs (handled in sibling worktree)
- Retract path (`schedule_batch.py:1038-1071`) — already correct, just gets the new fields reset by Task 2.2
- `cache_unfinished_req` — not on the leak path; same indirect formula but doesn't free `req_to_token_pool`
- SWA radix cache — same family of issues but separate test surface; defer
