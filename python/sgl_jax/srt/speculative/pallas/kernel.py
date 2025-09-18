def assign_req_to_token_pool(
    bs,
    req_to_token_pool,
    req_pool_indices,
    prefix_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
):
    pt = 0
    for i in range(bs):
        req_to_token_pool.write(
            (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
            out_cache_loc[pt : pt + extend_lens[i]],
        )
        pt += extend_lens[i]


def verify_tree_greedy():
    pass


def top_k_renorm_prob():
    pass


def top_p_renorm_prob():
    pass


def tree_speculative_sampling_target_only():
    pass


def align_evict_mask_to_page_size():
    pass


def get_target_cache_loc():
    pass


def filter_finished_cache_loc_kernel():
    pass
