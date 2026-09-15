"""CPU-only tests of session lease lifetime (no model/JAX initialization)."""

from types import SimpleNamespace

import pytest

from sgl_jax.srt.mem_cache.session_store import SessionStore


@pytest.fixture
def store():
    released = []
    clock = SimpleNamespace(now=0.0)
    store = SessionStore(released.append, timeout=10, capacity=2, clock=lambda: clock.now)
    return store, released, clock


def test_restore_lifetime_and_busy_rejection(store):
    cache, freed, clock = store
    first, second = object(), object()
    session = cache.acquire("a", first)
    with pytest.raises(ValueError, match="in-flight"):
        cache.acquire("a", second)
    assert cache.finish("a", first, retain=True, tokens=[1, 2])
    assert session.owner is first and not freed
    clock.now = 1
    assert cache.acquire("a", second) is session
    cache.finish("a", second)
    assert freed == [first, second]


def test_close_inflight_is_deferred(store):
    cache, freed, _ = store
    req = object()
    cache.acquire("a", req)
    cache.close("a")
    assert not freed
    assert cache.finish("a", req, retain=True, tokens=[1])
    assert freed == [req] and not cache.sessions
    cache.close("a")  # idempotent


def test_idle_ttl_and_auto_recreation(store):
    cache, freed, clock = store
    req = object()
    cache.acquire("a", req)
    clock.now = 20
    cache.reap()  # A long running request is not idle.
    assert not freed and not cache.sessions["a"].closing
    cache.finish("a", req, retain=True, tokens=[1])
    clock.now = 31
    cache.reap()
    assert freed == [req] and not cache.sessions
    assert cache.acquire("a", object()).owner is None


def test_capacity_evicts_only_idle_lru(store):
    cache, freed, clock = store
    a, b = object(), object()
    cache.acquire("a", a)
    cache.finish("a", a, retain=True)
    clock.now = 1
    cache.acquire("b", b)
    cache.acquire("c", object())
    assert freed == [a] and set(cache.sessions) == {"b", "c"}
    with pytest.raises(ValueError, match="busy"):
        cache.open("d")


def test_wrong_request_cannot_release_lease(store):
    cache, freed, _ = store
    req = object()
    cache.acquire("a", req)
    assert not cache.finish("a", object())
    assert cache.sessions["a"].active is req and not freed


def test_reset_checks_busy_and_releases_idle(store):
    cache, freed, _ = store
    req = object()
    cache.acquire("a", req)
    with pytest.raises(RuntimeError, match="in-flight"):
        cache.reset()
    cache.finish("a", req, retain=True)
    cache.reset()
    assert freed == [req] and not cache.sessions


@pytest.mark.parametrize("session_id", [None, "", " ", [], 1])
def test_invalid_session_id(store, session_id):
    with pytest.raises(ValueError, match="nonempty"):
        store[0].acquire(session_id, object())


@pytest.mark.parametrize("timeout,capacity", [(0, 1), (1, 0), (-1, 2)])
def test_invalid_limits(timeout, capacity):
    with pytest.raises(ValueError, match="positive"):
        SessionStore(lambda _: None, timeout=timeout, capacity=capacity)
