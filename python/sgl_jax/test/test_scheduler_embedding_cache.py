from types import SimpleNamespace
from unittest.mock import Mock

from sgl_jax.srt.managers.scheduler import Scheduler, _clear_multimodal_embedding_caches


def test_pathways_inflight_blocks_cache_flush():
    scheduler = SimpleNamespace(
        waiting_queue=[],
        pending_dp_reqs=[],
        running_batch=None,
        cur_batch=None,
        last_batch=None,
        chunked_reqs=[],
        enable_overlap=False,
        disagg_prefill_queue=None,
        disagg_prealloc_queue=None,
        disagg_transfer_queue=None,
        _pd_inflight=1,
    )
    allowed, message = Scheduler._can_flush_cache(scheduler)
    assert not allowed
    assert "pd_inflight=1" in message

    scheduler._pd_inflight = 0
    assert Scheduler._can_flush_cache(scheduler) == (True, "")


def test_embedding_cache_flush_deduplicates_runners():
    first_cache = Mock()
    second_cache = Mock()
    first_runner = SimpleNamespace(multimodal_embedding_cache=first_cache)
    second_runner = SimpleNamespace(multimodal_embedding_cache=second_cache)
    first_worker = SimpleNamespace(get_model_runner=lambda: first_runner)
    first_alias = SimpleNamespace(get_model_runner=lambda: first_runner)
    second_worker = SimpleNamespace(get_model_runner=lambda: second_runner)

    _clear_multimodal_embedding_caches((first_worker, first_alias, second_worker, None))

    first_cache.clear.assert_called_once_with()
    second_cache.clear.assert_called_once_with()
