"""DSpark verify-budget planning and ragged target compaction."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.speculative.dspark_tuned_config import DSparkSPSProfile


@dataclass(frozen=True)
class DSparkBudgetDecision:
    """One per-DP-rank static verify-bucket decision."""

    token_bucket: int
    extra_budget: int
    expected_tokens: float
    estimated_step_time_ms: float
    expected_tokens_per_ms: float


def select_dspark_verify_budget(
    profile: DSparkSPSProfile,
    survival: np.ndarray,
    *,
    forced_token_bucket: int | None = None,
) -> DSparkBudgetDecision | None:
    """Choose the SPS point with the best expected accepted tokens per ms.

    ``survival`` contains one row per active request and one column per draft
    position. Every request always receives its anchor row; a profile point's
    remaining rows are assigned to the largest prefix-survival probabilities.
    Points that cannot contain all anchors or exceed verify-all are ignored.
    """
    survival = np.asarray(survival, dtype=np.float64)
    if survival.ndim != 2:
        raise ValueError(f"survival must be rank 2, got shape={survival.shape}.")
    num_requests, gamma = survival.shape
    if num_requests == 0:
        return None
    if not np.all(np.isfinite(survival)):
        raise ValueError("survival must contain only finite values.")

    flat = np.sort(np.clip(survival, 0.0, 1.0).reshape(-1))[::-1]
    max_tokens = num_requests * (gamma + 1)
    request_buckets = sorted(
        {
            int(point.request_bucket_per_dp)
            for point in profile.points
            if point.request_bucket_per_dp is not None
        }
    )
    selected_request_bucket = next(
        (bucket for bucket in request_buckets if bucket >= num_requests),
        None,
    )
    if request_buckets and selected_request_bucket is None:
        return None

    best: DSparkBudgetDecision | None = None
    for point in profile.points:
        if request_buckets and point.request_bucket_per_dp != selected_request_bucket:
            continue
        bucket = int(point.verify_tokens_per_dp)
        if forced_token_bucket is not None and bucket != forced_token_bucket:
            continue
        if bucket < num_requests:
            continue
        # A request-bucket executable may legitimately contain padding when
        # the live row count is below its covering R bucket.  The benchmark
        # force path needs the same behavior to measure boundary points such
        # as T(32, 256) while DP routing briefly yields 31 live rows.
        if bucket > max_tokens and selected_request_bucket is None and forced_token_bucket is None:
            continue
        extra_budget = min(bucket - num_requests, flat.size)
        expected_tokens = float(num_requests + flat[:extra_budget].sum())
        step_time_ms = float(point.median_step_time_ms)
        if step_time_ms <= 0:
            continue
        decision = DSparkBudgetDecision(
            token_bucket=bucket,
            extra_budget=extra_budget,
            expected_tokens=expected_tokens,
            estimated_step_time_ms=step_time_ms,
            expected_tokens_per_ms=expected_tokens / step_time_ms,
        )
        if best is None or (
            decision.expected_tokens_per_ms,
            -decision.token_bucket,
        ) > (best.expected_tokens_per_ms, -best.token_bucket):
            best = decision
    return best


def allocate_dspark_verify_lens(
    confidence: jax.Array,
    active_mask: jax.Array,
    extra_budget_per_dp: jax.Array,
    *,
    dp_size: int,
) -> jax.Array:
    """Allocate each DP rank's extra rows with deterministic prefix ordering."""
    confidence = jnp.clip(confidence.astype(jnp.float32), 0.0, 1.0)
    active_mask = active_mask.astype(jnp.bool_)
    if confidence.ndim != 2:
        raise ValueError(f"confidence must be rank 2, got shape={confidence.shape}.")
    if confidence.shape[0] % dp_size != 0:
        raise ValueError(
            f"confidence rows must be divisible by dp_size: {confidence.shape[0]} vs {dp_size}."
        )

    per_dp_bs = confidence.shape[0] // dp_size
    gamma = confidence.shape[1]

    def _allocate_rank(rank_confidence, rank_active, rank_budget):
        survival = jnp.cumprod(rank_confidence, axis=-1)
        survival = jnp.where(rank_active[:, None], survival, -jnp.inf)
        # Position-major flattening plus stable argsort gives the required tie
        # break: survival descending, position ascending, request index ascending.
        candidates = survival.T.reshape((gamma * per_dp_bs,))
        order = jnp.argsort(-candidates, stable=True)
        rank = jnp.argsort(order, stable=True)
        budget = rank_budget.astype(jnp.int32).reshape(())
        selected = (rank < budget).reshape((gamma, per_dp_bs)).T
        selected = selected & rank_active[:, None]
        return rank_active.astype(jnp.int32) + jnp.sum(selected.astype(jnp.int32), axis=-1)

    sharding = jax.typeof(confidence).sharding
    if isinstance(sharding, jax.sharding.NamedSharding) and not sharding.mesh.empty:
        from jax.sharding import PartitionSpec as P

        if int(sharding.mesh.shape["data"]) != dp_size:
            raise ValueError(
                "DSpark dp_size must match the mesh data axis: "
                f"{dp_size} vs {sharding.mesh.shape['data']}."
            )
        return jax.shard_map(
            _allocate_rank,
            mesh=sharding.mesh,
            in_specs=(P("data", None), P("data"), P("data")),
            out_specs=P("data"),
        )(confidence, active_mask, extra_budget_per_dp)

    return jax.vmap(_allocate_rank)(
        confidence.reshape((dp_size, per_dp_bs, gamma)),
        active_mask.reshape((dp_size, per_dp_bs)),
        extra_budget_per_dp.reshape((dp_size, 1)),
    ).reshape((-1,))


def compact_dspark_verify_inputs(
    input_ids: jax.Array,
    positions: jax.Array,
    cache_loc: jax.Array,
    verify_lens: jax.Array,
    *,
    dp_size: int,
    verify_width: int,
    per_dp_token_bucket: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compact fixed logical verify rows into a static per-DP token bucket.

    Returns compact ids/positions/cache locations, a compact-to-logical index,
    and a mask distinguishing real query rows from static bucket padding.
    """
    total_bs = verify_lens.shape[0]
    if total_bs % dp_size != 0:
        raise ValueError(f"verify_lens must be divisible by dp_size: {total_bs} vs {dp_size}.")
    logical_tokens = total_bs * verify_width
    if input_ids.shape[0] != logical_tokens:
        raise ValueError(
            f"input_ids must contain bs*verify_width rows: {input_ids.shape[0]} vs {logical_tokens}."
        )
    per_dp_bs = total_bs // dp_size
    per_dp_logical = per_dp_bs * verify_width

    def _compact_values(values, rank_valid, *, padding_value):
        # nonzero has a static output size, so this remains one executable per
        # SPS token bucket while the actual verify_lens stay dynamic.
        source = jnp.nonzero(
            rank_valid,
            size=per_dp_token_bucket,
            fill_value=0,
        )[0]
        count = jnp.sum(rank_valid.astype(jnp.int32))
        output_valid = jnp.arange(per_dp_token_bucket, dtype=jnp.int32) < count
        gathered = values[source]
        return (
            jnp.where(output_valid, gathered, jnp.asarray(padding_value, values.dtype)),
            source,
            output_valid,
        )

    def _compact_rank(rank_ids, rank_positions, rank_cache_loc, rank_verify_lens):
        offsets = jnp.arange(verify_width, dtype=jnp.int32)[None, :]
        rank_valid = (offsets < rank_verify_lens[:, None]).reshape((per_dp_logical,))
        compact_ids, source, compact_valid = _compact_values(rank_ids, rank_valid, padding_value=0)
        compact_positions, _, _ = _compact_values(rank_positions, rank_valid, padding_value=0)
        compact_cache_loc, _, _ = _compact_values(rank_cache_loc, rank_valid, padding_value=-1)
        return compact_ids, compact_positions, compact_cache_loc, source, compact_valid

    ids_rows = input_ids.reshape((dp_size, per_dp_logical))
    pos_rows = positions.reshape((dp_size, per_dp_logical))
    cache_rows = cache_loc.reshape((dp_size, per_dp_logical))
    lens_rows = verify_lens.reshape((dp_size, per_dp_bs))
    sharding = jax.typeof(input_ids).sharding
    if isinstance(sharding, jax.sharding.NamedSharding) and not sharding.mesh.empty:
        from jax.sharding import PartitionSpec as P

        if int(sharding.mesh.shape["data"]) != dp_size:
            raise ValueError(
                "DSpark dp_size must match the mesh data axis: "
                f"{dp_size} vs {sharding.mesh.shape['data']}."
            )

        def _compact_shard(rank_ids, rank_positions, rank_cache_loc, rank_verify_lens):
            compact_ids, compact_positions, compact_cache_loc, source, compact_valid = (
                _compact_rank(rank_ids, rank_positions, rank_cache_loc, rank_verify_lens)
            )
            rank_base = jax.lax.axis_index("data") * per_dp_logical
            mapping = source + rank_base
            mapping = jnp.where(compact_valid, mapping, logical_tokens)
            return compact_ids, compact_positions, compact_cache_loc, mapping, compact_valid

        return jax.shard_map(
            _compact_shard,
            mesh=sharding.mesh,
            in_specs=(P("data"), P("data"), P("data"), P("data")),
            out_specs=(P("data"), P("data"), P("data"), P("data"), P("data")),
        )(input_ids, positions, cache_loc, verify_lens)

    compact_ids, compact_positions, compact_cache_loc, source, compact_valid = jax.vmap(
        _compact_rank
    )(ids_rows, pos_rows, cache_rows, lens_rows)
    rank_base = jnp.arange(dp_size, dtype=jnp.int32)[:, None] * per_dp_logical
    compact_to_logical = jnp.where(
        compact_valid,
        source + rank_base,
        logical_tokens,
    )
    return (
        compact_ids.reshape((-1,)),
        compact_positions.reshape((-1,)),
        compact_cache_loc.reshape((-1,)),
        compact_to_logical.reshape((-1,)),
        compact_valid.reshape((-1,)),
    )


def scatter_dspark_compact_rows(
    compact_rows: jax.Array,
    compact_to_logical: jax.Array,
    logical_size: int,
) -> jax.Array:
    """Scatter compact token/hidden rows back to the fixed logical window."""
    sharding = jax.typeof(compact_rows).sharding
    if isinstance(sharding, jax.sharding.NamedSharding) and not sharding.mesh.empty:
        from jax.sharding import PartitionSpec as P

        data_size = int(sharding.mesh.shape["data"])
        if logical_size % data_size != 0:
            raise ValueError(
                f"logical_size must be divisible by the mesh data axis: {logical_size} vs {data_size}."
            )
        local_logical_size = logical_size // data_size
        row_spec = P("data", *([None] * (compact_rows.ndim - 1)))

        def _scatter_rank(local_rows, global_indices):
            rank_base = jax.lax.axis_index("data") * local_logical_size
            local_indices = jnp.where(
                global_indices == logical_size,
                local_logical_size,
                global_indices - rank_base,
            )
            output_shape = (local_logical_size + 1, *local_rows.shape[1:])
            output = jnp.zeros(output_shape, dtype=local_rows.dtype)
            return output.at[local_indices].set(local_rows)[:local_logical_size]

        return jax.shard_map(
            _scatter_rank,
            mesh=sharding.mesh,
            in_specs=(row_spec, P("data")),
            out_specs=row_spec,
        )(compact_rows, compact_to_logical)

    output_shape = (logical_size + 1, *compact_rows.shape[1:])
    output = jnp.zeros(output_shape, dtype=compact_rows.dtype)
    output = output.at[compact_to_logical].set(compact_rows)
    return output[:logical_size]


__all__ = [
    "DSparkBudgetDecision",
    "allocate_dspark_verify_lens",
    "compact_dspark_verify_inputs",
    "scatter_dspark_compact_rows",
    "select_dspark_verify_budget",
]
