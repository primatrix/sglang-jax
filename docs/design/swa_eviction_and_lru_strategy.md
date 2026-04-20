# SWA Eviction and LRU Strategy

## 1. Dual-Pool Architecture

```
+-----------------------------------------------------------+
|                      KV Cache Memory                      |
|                                                           |
|  +-------------------------+  +-------------------------+ |
|  |       Full Pool         |  |        SWA Pool         | |
|  |   (Full Attention lys)  |  |   (Sliding Window lys)  | |
|  |                         |  |                         | |
|  |   Grows with seqlen     |  |   Bounded by window     | |
|  |   Never evicted per-req |  |   Evicted outside window| |
|  +-------------------------+  +-------------------------+ |
|                                                           |
|  Linked by full_to_swa_index_mapping:                     |
|    full_idx --> swa_idx  (or 0 if SWA freed)              |
+-----------------------------------------------------------+
```

### Key Terms

| Term | Meaning |
|------|---------|
| `swa_evicted_seqlen` | SWA slots `[0, swa_evicted_seqlen)` have been freed |
| `protected prefix` | Prefix `[0, protected_prefix_len)` owned by the radix tree, derived from the request's locked `last_node` path; per-request eviction cannot touch it |
| `last_matched_prefix_len` | Page-aligned cached prefix length kept on the request for writeback / retract bookkeeping |
| `tombstone` | Tree node: full KV retained, SWA KV freed |

## 2. Two Cache Modes at a Glance

```
ChunkCache (--disable-radix-cache)       SWARadixCache (radix cache enabled)
+-------------------------------+        +-------------------------------+
| No tree. Request owns all     |        | Radix tree caches KV for     |
| slots. Freed on completion.   |        | prefix reuse across reqs.    |
| No prefix sharing.            |        | Tree outlives requests.      |
+-------------------------------+        +-------------------------------+
```

## 3. Per-Request SWA Eviction (`_evict_swa`)

Frees SWA slots outside the sliding window from a request's `req_to_token` buffer.

```
_evict_swa(req, pre_len):

  1. Derive: protected_prefix_len = prefix_len(req.last_node)
  2. Clamp:  swa_evicted_seqlen = max(swa_evicted_seqlen, protected_prefix_len)
  3. Target: new_evicted = max(swa_evicted_seqlen, pre_len - sliding_window - page_size)
  4. Align:  new_evicted = page_floor(new_evicted)
  5. Free:   free_swa( slots[swa_evicted_seqlen : new_evicted] )

Example (sliding_window=128, page_size=256, seqlen=2049):

  0            1792        2048
  |............|############|
    freed SWA    retained
                 (window)
  ^                         ^
  swa_evicted=1792     seqlen=2049
```

### When does it run?

| Phase  | ChunkCache | SWARadixCache                             |
|--------|------------|-------------------------------------------|
| Extend | Yes        | **No** (skipped)                          |
| Decode | Yes        | Yes (but starts from the tree-derived protected prefix) |

## 4. Extend Phase Behavior

### 4.1 ChunkCache -- Proactive Eviction

With overlap scheduling, `_evict_swa` is gated by `extend_batch_idx`:

- `extend_batch_idx < 2`: skip (previous extend batch may still be running)
- `extend_batch_idx >= 2`: evict with `pre_len -= chunked_prefill_size`

This means chunk N+1 evicts chunk N-1's SWA cache (one chunk delay for safety).

```
8K tokens, chunk_size=2048, sliding_window=128, page_size=256, overlap=True

       0       2048      4096      6144      8192
       |        |         |         |         |
Chk 1: |########|                              extend_batch_idx=0, skip
       alloc [0,2048)

Chk 2: |################|                     extend_batch_idx=1, skip
       alloc [0,4096)

Chk 3: |...........|################|         extend_batch_idx=2
       freed       retained                   pre_len=4096-2048=2048
       [0,1792)                                evict [0,1792)

Chk 4: |...................|################| extend_batch_idx=3
       freed               retained           pre_len=6144-2048=4096
       [0,3840)                                evict [1792,3840)

  . = freed SWA    # = retained SWA
```

Without overlap (`enable_overlap=False`), there is no `extend_batch_idx`
gate and no `pre_len` adjustment -- chunk N directly evicts chunk N-1:

```
Chk 1: |########|                              pre_len=0, nothing
Chk 2: |...|################|                  pre_len=2048, evict [0,1792)
Chk 3: |.............|################|        pre_len=4096, evict [1792,3840)
```

### 4.2 SWARadixCache -- Deferred to Tree

```
8K tokens, chunk_size=2048, sliding_window=128, page_size=256

       0       2048      4096      6144      8192
       |        |         |         |         |
Chk 1: |########|
       _evict_swa: SKIPPED
       cache_unfinished_req --> tree owns [0,2048), all non-tombstone
       protected prefix = 2048 (derived from last_node)

Chk 2: |################|
       _evict_swa: SKIPPED
       cache_unfinished_req --> tree owns [0,4096), all non-tombstone
       protected prefix = 4096 (derived from last_node)

       ... chunks 3, 4 ...  protected prefix = 8192

First decode step:
  protected_prefix_len = prefix_len(last_node) = 8192
  swa_evicted = max(0, 8192) = 8192
  new_evicted = max(8192, 8192-128-256) = 8192
  --> NO EVICTION (all tree-protected)

  # = SWA in tree (non-tombstone, only freed by tree Phase 2)
```

**Trade-off**: Prefill SWA slots stay in tree until pool pressure triggers
tree-level eviction. The ownership boundary now lives inside the cache layer
instead of a request field, but prefill is still less SWA-efficient than ChunkCache.

## 5. Tree-Level Eviction (SWARadixCache)

Triggered by allocation pressure (`evict_from_tree_cache` when pool space < needed).

### Phase 1 -- Full Eviction (delete leaf nodes)

Frees **both** full + SWA KV. Removes node from tree entirely.

```
Before:                             After evicting leaf B:

root --> [A 0..2048] --> [B] leaf   root --> [A 0..2048] --> [C] leaf
                     --> [C] leaf
                                    B: full+SWA freed, node deleted

If parent becomes childless tombstone --> cascade delete:

root --> [A tombstone] --> [B] leaf       root (empty)
         only child                       A, B both deleted
```

### Phase 2 -- SWA-Only Eviction (tombstone internal nodes)

Frees **only SWA** KV. Full KV kept for prefix matching.

```
Before:                                After Phase 2 on node A:

root --> [A 0..2048] --> [B] leaf      root --> [A 0..2048] --> [B] leaf
          full+SWA   --> [C] leaf               TOMBSTONE   --> [C] leaf
                                                 full only
```

### Gradual Tombstone Progression (LRU order, head --> tail)

```
          [0..2048]    [2048..4096]   [4096..6144]   [6144..8192]

Initial:  full+SWA --> full+SWA  --> full+SWA  --> full+SWA

Round 1:  TOMBSTONE--> full+SWA  --> full+SWA  --> full+SWA
          full only

Round 2:  TOMBSTONE--> TOMBSTONE --> full+SWA  --> full+SWA
          full only    full only

Round 3:  TOMBSTONE--> TOMBSTONE --> TOMBSTONE --> full+SWA
          full only    full only     full only     ^ only tail
                                                     retains SWA
```

## 6. Tombstone Insert Logic

`_insert_helper` has two sets of tombstone branch logic:

- **Inside the while loop**: healing of **existing** tombstone nodes
- **After the while loop**: tombstone split when creating **new** nodes

### 6.1 Existing Tombstone Healing (while loop)

When insert walks through an existing tombstone node beyond
`update_kv_after_len`, it checks `swa_evicted_seqlen` against
`[node_start, node_end)`:

```
  BRANCH 1: swa_evicted <= node_start  --> revive entire node
  BRANCH 2: node_start < swa_evicted < node_end --> split, revive back half
  BRANCH 3: swa_evicted >= node_end    --> keep tombstone, free incoming value
```

Branch 1 fires during `cache_unfinished_req` (`swa_evicted_seqlen = 0`)
when `_match_prefix_helper` truncated the prefix due to tombstone safety.

Branches 2/3 fire during `cache_finished_req` when a prior request's
decode nodes have been tombstoned and the current request generated
identical decode tokens (e.g. greedy decoding with the same prefix).

### 6.2 New Node Tombstone Split (after while loop)

When insert has remaining unmatched tokens (`len(key) > 0`), it creates
new nodes.  Invariant: **leaf nodes must never be tombstone**.

```
  Case 1: swa_evicted <= total_prefix_length
          --> create non-tombstone leaf (normal path for cache_unfinished_req)

  Case 2: total_prefix_length < swa_evicted < total_prefix_length + len(key)
          --> split into tombstone prefix + non-tombstone leaf
              (normal path for cache_finished_req: decode suffix split)

  Case 3: swa_evicted >= total_prefix_length + len(key)
          --> all remaining SWA evicted, cannot create non-tombstone leaf
              free incoming value and return (defensive guard, kept safe
              by the extra `- page_size` in `_evict_swa` frontier)
```

### 6.3 Trigger Conditions

| Call site | `swa_evicted_seqlen` | Existing tombstone | New nodes |
|-----------|---------------------|--------------------|-----------|
| `cache_unfinished_req` | Always 0 | Branch 1 only | Case 1 only |
| `cache_finished_req` | >= protected prefix length | Prefill nodes: skip zone. Prior req's decode nodes: Branch 1/2/3 | Case 2 (normal), Case 3 (defensive) |

## 7. LRU Policy

```
+-----------------------------------------------------+
|                   full_lru_list                     |
|  Tracks: ALL non-root nodes                         |
|  Used by: Phase 1 (find LRU leaf for full eviction) |
|                                                     |
|  LRU <--- old nodes --- recent nodes ---> MRU       |
+-----------------------------------------------------+

+-----------------------------------------------------+
|                   swa_lru_list                      |
|  Tracks: non-root, NON-TOMBSTONE nodes only         |
|  Used by: Phase 2 (find LRU node for SWA eviction)  |
|                                                     |
|  Tombstone nodes REMOVED from this list.            |
|  Revived nodes RE-INSERTED at MRU.                  |
+-----------------------------------------------------+
```

On prefix match, the **deepest matched node** and all ancestors are
reset to MRU in both lists, keeping frequently accessed prefixes fresh.

## 8. Summary

```
+-------------------+-----------------+------------------------+
|                   | ChunkCache      | SWARadixCache          |
+-------------------+-----------------+------------------------+
| Extend SWA evict  | Proactive       | Skipped                |
|                   | (per chunk)     | (tree handles)         |
+-------------------+-----------------+------------------------+
| Decode SWA evict  | From pos 0      | From                   |
|                   |                 | protected tree prefix  |
+-------------------+-----------------+------------------------+
| Prefill SWA waste | Low (bounded)   | Higher (until          |
|                   |                 | Phase 2 evicts)        |
+-------------------+-----------------+------------------------+
| Prefix reuse      | None            | Yes (via tree)         |
+-------------------+-----------------+------------------------+
| SWA reclamation   | Per-request     | Tree Phase 2           |
|                   | _evict_swa      | (reactive, LRU)        |
+-------------------+-----------------+------------------------+
```
