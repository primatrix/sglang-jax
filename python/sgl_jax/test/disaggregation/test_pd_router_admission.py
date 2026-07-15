import argparse
import asyncio
from types import SimpleNamespace


def make_lb(**kwargs):
    from sgl_jax.srt.disaggregation.mini_lb import MiniLoadBalancer

    args = SimpleNamespace(
        host="127.0.0.1",
        port=30000,
        request_timeout_secs=10,
        prefill_urls=[("http://prefill", 31000)],
        decode_urls=["http://decode"],
        test_external_dp_routing=False,
        prefill_bootstrap_host=None,
        max_concurrent_requests=None,
        pd_prefill_max_inflight_requests=kwargs.get("prefill_limit", 0),
        pd_decode_prealloc_soft_limit=kwargs.get("prealloc_limit", 0),
        pd_decode_oldest_prealloc_wait_ms_soft_limit=kwargs.get("wait_limit", 0.0),
        pd_router_admission_poll_ms=kwargs.get("poll_ms", 50),
        pd_router_prefill_decode_overlap=kwargs.get("overlap", True),
        policy="random",
        pd_disaggregation=True,
    )
    return MiniLoadBalancer(args)


def test_router_args_parse_pd_decode_admission_limits():
    from sgl_jax.srt.disaggregation.router_args import RouterArgs

    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    args = parser.parse_args([
        "--pd-decode-prealloc-soft-limit",
        "8",
        "--pd-decode-oldest-prealloc-wait-ms-soft-limit",
        "5000",
        "--pd-router-admission-poll-ms",
        "25",
    ])

    router_args = RouterArgs.from_cli_args(args)

    assert router_args.pd_decode_prealloc_soft_limit == 8
    assert router_args.pd_decode_oldest_prealloc_wait_ms_soft_limit == 5000.0
    assert router_args.pd_router_admission_poll_ms == 25


def test_router_args_parse_pd_prefill_inflight_limit():
    from sgl_jax.srt.disaggregation.router_args import RouterArgs

    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    args = parser.parse_args([
        "--pd-prefill-max-inflight-requests",
        "4",
    ])

    router_args = RouterArgs.from_cli_args(args)

    assert router_args.pd_prefill_max_inflight_requests == 4


def test_router_args_can_disable_prefill_decode_overlap():
    from sgl_jax.srt.disaggregation.router_args import RouterArgs

    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    args = parser.parse_args([
        "--no-pd-router-prefill-decode-overlap",
    ])

    router_args = RouterArgs.from_cli_args(args)

    assert router_args.pd_router_prefill_decode_overlap is False


def test_prefill_admission_limits_concurrent_prefill_posts():
    lb = make_lb(prefill_limit=2)
    active = 0
    max_active = 0

    class FakeResponse:
        status = 200

        async def read(self):
            return b"{}"

    class FakeSession:
        async def post(self, url, json):
            nonlocal active, max_active
            assert url == "http://prefill/generate"
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            active -= 1
            return FakeResponse()

    async def run_posts():
        session = FakeSession()
        await asyncio.gather(
            *[
                lb._post_prefill_with_admission(
                    session,
                    "http://prefill",
                    "generate",
                    {"rid": i},
                )
                for i in range(5)
            ]
        )

    asyncio.run(run_posts())

    assert max_active == 2


def test_generate_waits_for_prefill_slot_before_decode_post(monkeypatch):
    import aiohttp

    lb = make_lb(prefill_limit=1)
    prefill_posts = []
    decode_posts = []
    first_prefill_started = asyncio.Event()
    allow_first_prefill_finish = asyncio.Event()

    async def fake_align_dp_requests(request):
        return request, request

    async def fake_wait_for_decode_admission(session, decode_server):
        return None

    lb._align_dp_requests = fake_align_dp_requests
    lb._wait_for_decode_admission = fake_wait_for_decode_admission

    class FakeResponse:
        status = 200

        async def read(self):
            return b"{}"

        async def json(self):
            return {}

    class FakeClientSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            rid = json["rid"]
            if url == "http://prefill/generate":
                prefill_posts.append(rid)
                if rid == "first":
                    first_prefill_started.set()
                    await allow_first_prefill_finish.wait()
                return FakeResponse()
            if url == "http://decode/generate":
                decode_posts.append(rid)
                return FakeResponse()
            raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(aiohttp, "ClientSession", FakeClientSession)
    monkeypatch.setattr(aiohttp, "ClientTimeout", lambda total: object())

    async def run_requests():
        first = asyncio.create_task(
            lb.generate(
                {"rid": "first"},
                "http://prefill",
                "http://decode",
                "generate",
            )
        )
        await first_prefill_started.wait()
        await asyncio.sleep(0)
        assert prefill_posts == ["first"]
        assert decode_posts == ["first"]

        second = asyncio.create_task(
            lb.generate(
                {"rid": "second"},
                "http://prefill",
                "http://decode",
                "generate",
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert prefill_posts == ["first"]
        assert decode_posts == ["first"]

        allow_first_prefill_finish.set()
        await asyncio.gather(first, second)

    asyncio.run(run_requests())

    assert prefill_posts == ["first", "second"]
    assert decode_posts == ["first", "second"]


def test_generate_can_disable_prefill_decode_overlap(monkeypatch):
    import aiohttp

    lb = make_lb(prefill_limit=1, overlap=False)
    prefill_posts = []
    decode_posts = []
    prefill_started = asyncio.Event()
    allow_prefill_finish = asyncio.Event()

    async def fake_align_dp_requests(request):
        return request, request

    async def fake_wait_for_decode_admission(session, decode_server):
        return None

    lb._align_dp_requests = fake_align_dp_requests
    lb._wait_for_decode_admission = fake_wait_for_decode_admission

    class FakeResponse:
        status = 200

        async def read(self):
            return b"{}"

        async def json(self):
            return {}

    class FakeClientSession:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            rid = json["rid"]
            if url == "http://prefill/generate":
                prefill_posts.append(rid)
                prefill_started.set()
                await allow_prefill_finish.wait()
                return FakeResponse()
            if url == "http://decode/generate":
                decode_posts.append(rid)
                return FakeResponse()
            raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(aiohttp, "ClientSession", FakeClientSession)
    monkeypatch.setattr(aiohttp, "ClientTimeout", lambda total: object())

    async def run_request():
        task = asyncio.create_task(
            lb.generate(
                {"rid": "req0"},
                "http://prefill",
                "http://decode",
                "generate",
            )
        )
        await prefill_started.wait()
        await asyncio.sleep(0)
        assert prefill_posts == ["req0"]
        assert decode_posts == []

        allow_prefill_finish.set()
        await task

    asyncio.run(run_request())

    assert decode_posts == ["req0"]


def test_decode_admission_blocked_by_prealloc_queue():
    lb = make_lb(prealloc_limit=8)
    info = {
        "internal_states": [
            {"pd_decode_admission": {"prealloc_queue_size": 8}},
        ]
    }

    assert lb._decode_admission_blocked(info)


def test_decode_admission_blocked_by_oldest_wait():
    lb = make_lb(wait_limit=5000.0)
    info = {
        "internal_states": [
            {
                "pd_decode_admission": {
                    "prealloc_queue_size": 1,
                    "oldest_prealloc_wait_ms": 6000.0,
                }
            },
        ]
    }

    assert lb._decode_admission_blocked(info)


def test_decode_admission_allows_empty_backlog():
    lb = make_lb(prealloc_limit=8, wait_limit=5000.0)
    info = {
        "internal_states": [
            {
                "pd_decode_admission": {
                    "prealloc_queue_size": 0,
                    "oldest_prealloc_wait_ms": None,
                }
            },
        ]
    }

    assert not lb._decode_admission_blocked(info)


def test_wait_for_decode_admission_polls_until_unblocked(monkeypatch):
    import sgl_jax.srt.disaggregation.mini_lb as mini_lb

    lb = make_lb(prealloc_limit=8, poll_ms=25)
    infos = [
        {"internal_states": [{"pd_decode_admission": {"prealloc_queue_size": 9}}]},
        {"internal_states": [{"pd_decode_admission": {"prealloc_queue_size": 0}}]},
    ]
    calls = []
    sleeps = []

    async def fake_fetch_backend_json(session, decode_server, endpoint_candidates):
        calls.append((session, decode_server, endpoint_candidates))
        return infos.pop(0)

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(mini_lb, "fetch_backend_json", fake_fetch_backend_json)
    monkeypatch.setattr(mini_lb.asyncio, "sleep", fake_sleep)

    asyncio.run(lb._wait_for_decode_admission(object(), "http://decode"))

    assert [call[1] for call in calls] == ["http://decode", "http://decode"]
    assert [call[2] for call in calls] == [
        ("get_server_info", "server_info"),
        ("get_server_info", "server_info"),
    ]
    assert sleeps == [0.025]

    state = lb.get_observability_state()
    assert state["decode_admission_wait_count"] == 1
    assert state["decode_admission_poll_count"] == 2
    assert state["decode_admission_blocked_count"] == 1
    assert state["decode_admission_wait_ms_total"] >= 0.0


def test_prefill_admission_observability_tracks_slots_and_waits():
    lb = make_lb(prefill_limit=1)

    async def run_waiter():
        sem = lb._prefill_admission_sems["http://prefill"]
        await sem.acquire()
        waiter = asyncio.create_task(lb._acquire_prefill_admission("http://prefill"))
        await asyncio.sleep(0)

        waiting_state = lb.get_observability_state()
        assert waiting_state["prefill_admission_inflight_by_url"]["http://prefill"] == 1
        assert waiting_state["prefill_admission_waiting_by_url"]["http://prefill"] == 1

        sem.release()
        acquired = await waiter
        acquired.release()

    asyncio.run(run_waiter())

    state = lb.get_observability_state()
    assert state["pd_prefill_max_inflight_requests"] == 1
    assert state["prefill_admission_wait_count"] == 1
    assert state["prefill_admission_inflight_by_url"]["http://prefill"] == 0
    assert state["prefill_admission_available_by_url"]["http://prefill"] == 1
    assert state["prefill_admission_wait_ms_total"] >= 0.0
