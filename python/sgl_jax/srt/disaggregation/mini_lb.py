from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from http import HTTPStatus
from itertools import chain

logger = logging.getLogger(__name__)

AIOHTTP_STREAM_READ_CHUNK_SIZE = 1024 * 64


def _get_dp_size(server_info: dict) -> int:
    dp_size = server_info.get("dp_size")
    if dp_size is not None:
        return int(dp_size)
    return max(1, len(server_info.get("internal_states") or []))


class MiniLoadBalancer:
    def __init__(self, router_args):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.test_external_dp_routing = router_args.test_external_dp_routing
        self.prefill_bootstrap_host = router_args.prefill_bootstrap_host
        self.prefill_dp_size = None
        self.decode_dp_size = None
        self.max_concurrent_requests = getattr(router_args, "max_concurrent_requests", None)
        self.pd_prefill_max_inflight_requests = max(
            0,
            int(getattr(router_args, "pd_prefill_max_inflight_requests", 0) or 0),
        )
        self._prefill_admission_sems = (
            {
                prefill_url: asyncio.Semaphore(self.pd_prefill_max_inflight_requests)
                for prefill_url in self.prefill_urls
            }
            if self.pd_prefill_max_inflight_requests > 0
            else {}
        )
        self.pd_decode_prealloc_soft_limit = getattr(
            router_args,
            "pd_decode_prealloc_soft_limit",
            0,
        )
        self.pd_decode_oldest_prealloc_wait_ms_soft_limit = getattr(
            router_args,
            "pd_decode_oldest_prealloc_wait_ms_soft_limit",
            0.0,
        )
        self.pd_router_admission_poll_ms = getattr(
            router_args,
            "pd_router_admission_poll_ms",
            50,
        )
        self.pd_router_prefill_decode_overlap = bool(
            getattr(router_args, "pd_router_prefill_decode_overlap", True)
        )
        self.prefill_admission_wait_count = 0
        self.prefill_admission_wait_ms_total = 0.0
        self.prefill_admission_wait_ms_max = 0.0
        self.decode_admission_wait_count = 0
        self.decode_admission_wait_ms_total = 0.0
        self.decode_admission_wait_ms_max = 0.0
        self.decode_admission_poll_count = 0
        self.decode_admission_blocked_count = 0

    def _validate_router_args(self, router_args) -> None:
        if getattr(router_args, "policy", "random") != "random":
            logger.warning("[MiniLB] Overriding policy to random")
            router_args.policy = "random"

        if not getattr(router_args, "pd_disaggregation", False):
            raise ValueError("MiniLB only supports PD disaggregation mode")
        if len(router_args.prefill_urls) == 0 or len(router_args.decode_urls) == 0:
            raise ValueError("MiniLB requires at least one prefill and one decode server")

    def start(self) -> None:
        import uvicorn

        global lb, _admission_sem
        lb = self
        if self.max_concurrent_requests:
            _admission_sem = asyncio.Semaphore(self.max_concurrent_requests)
        uvicorn.run(app, host=self.host, port=self.port)

    async def _ensure_dp_sizes(self) -> None:
        if self.prefill_dp_size is not None:
            return

        import aiohttp

        async with aiohttp.ClientSession() as session:
            prefill_info = await fetch_backend_json(
                session,
                self.prefill_urls[0],
                ("get_server_info", "server_info"),
            )
            decode_info = await fetch_backend_json(
                session,
                self.decode_urls[0],
                ("get_server_info", "server_info"),
            )
        self.prefill_dp_size = _get_dp_size(prefill_info)
        self.decode_dp_size = _get_dp_size(decode_info)
        logger.info(
            "[MiniLB] DP sizes: prefill=%s, decode=%s",
            self.prefill_dp_size,
            self.decode_dp_size,
        )

    def _fork_dp_requests(self, request: dict):
        p_rank = random.randint(0, self.prefill_dp_size - 1)
        d_rank = random.randint(0, self.decode_dp_size - 1)

        prefill_req = request.copy()
        decode_req = request.copy()
        prefill_req["dp_rank"] = p_rank
        decode_req["dp_rank"] = d_rank
        decode_req["disagg_prefill_dp_rank"] = p_rank

        return prefill_req, decode_req, d_rank

    @staticmethod
    def _room_rank(bootstrap_room, dp_size: int):
        if isinstance(bootstrap_room, list):
            return [int(room) % dp_size for room in bootstrap_room]
        return int(bootstrap_room) % dp_size

    async def _align_dp_requests(self, request: dict):
        await self._ensure_dp_sizes()
        dp_size = min(int(self.prefill_dp_size or 1), int(self.decode_dp_size or 1))
        if dp_size <= 1:
            return request, request

        forced_rank = os.getenv("SGLANG_JAX_PD_FORCE_DP_RANK")
        dp_rank = (
            int(forced_rank) % dp_size
            if forced_rank is not None
            else self._room_rank(request["bootstrap_room"], dp_size)
        )
        prefill_req = request.copy()
        decode_req = request.copy()
        prefill_req["dp_rank"] = dp_rank
        decode_req["dp_rank"] = dp_rank
        decode_req["disagg_prefill_dp_rank"] = dp_rank
        return prefill_req, decode_req

    def select_pair(self) -> tuple[int, str, int | None, str]:
        pidx = random.randint(0, len(self.prefill_urls) - 1)
        didx = random.randint(0, len(self.decode_urls) - 1)
        return (
            pidx,
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            self.decode_urls[didx],
        )

    def _decode_admission_blocked(self, decode_info: dict) -> bool:
        states = decode_info.get("internal_states") or []
        for state in states:
            admission = state.get("pd_decode_admission") or {}
            prealloc_q = admission.get("prealloc_queue_size", 0)
            oldest_ms = admission.get("oldest_prealloc_wait_ms")
            if (
                self.pd_decode_prealloc_soft_limit > 0
                and prealloc_q >= self.pd_decode_prealloc_soft_limit
            ):
                return True
            if (
                self.pd_decode_oldest_prealloc_wait_ms_soft_limit > 0
                and oldest_ms is not None
                and oldest_ms >= self.pd_decode_oldest_prealloc_wait_ms_soft_limit
            ):
                return True
        return False

    def get_observability_state(self) -> dict:
        return {
            "max_concurrent_requests": self.max_concurrent_requests,
            "pd_prefill_max_inflight_requests": self.pd_prefill_max_inflight_requests,
            "pd_router_prefill_decode_overlap": self.pd_router_prefill_decode_overlap,
            "prefill_admission_inflight_by_url": {
                url: self._semaphore_inflight(sem)
                for url, sem in self._prefill_admission_sems.items()
            },
            "prefill_admission_available_by_url": {
                url: int(getattr(sem, "_value", 0))
                for url, sem in self._prefill_admission_sems.items()
            },
            "prefill_admission_waiting_by_url": {
                url: self._semaphore_waiter_count(sem)
                for url, sem in self._prefill_admission_sems.items()
            },
            "prefill_admission_wait_count": self.prefill_admission_wait_count,
            "prefill_admission_wait_ms_total": round(
                self.prefill_admission_wait_ms_total, 3
            ),
            "prefill_admission_wait_ms_max": round(
                self.prefill_admission_wait_ms_max, 3
            ),
            "decode_admission_wait_count": self.decode_admission_wait_count,
            "decode_admission_wait_ms_total": round(
                self.decode_admission_wait_ms_total, 3
            ),
            "decode_admission_wait_ms_max": round(
                self.decode_admission_wait_ms_max, 3
            ),
            "decode_admission_poll_count": self.decode_admission_poll_count,
            "decode_admission_blocked_count": self.decode_admission_blocked_count,
            "updated_at": time.time(),
        }

    def _semaphore_inflight(self, sem: asyncio.Semaphore) -> int:
        return max(0, self.pd_prefill_max_inflight_requests - int(getattr(sem, "_value", 0)))

    @staticmethod
    def _semaphore_waiter_count(sem: asyncio.Semaphore) -> int:
        waiters = getattr(sem, "_waiters", None) or ()
        return sum(1 for waiter in waiters if not waiter.done())

    def _record_prefill_admission_wait(self, wait_s: float) -> None:
        wait_ms = max(0.0, wait_s * 1000.0)
        self.prefill_admission_wait_count += 1
        self.prefill_admission_wait_ms_total += wait_ms
        self.prefill_admission_wait_ms_max = max(
            self.prefill_admission_wait_ms_max,
            wait_ms,
        )

    def _record_decode_admission_wait(
        self,
        wait_s: float,
        poll_count: int,
        blocked_count: int,
    ) -> None:
        wait_ms = max(0.0, wait_s * 1000.0)
        self.decode_admission_wait_count += 1
        self.decode_admission_wait_ms_total += wait_ms
        self.decode_admission_wait_ms_max = max(
            self.decode_admission_wait_ms_max,
            wait_ms,
        )
        self.decode_admission_poll_count += poll_count
        self.decode_admission_blocked_count += blocked_count

    async def _wait_for_decode_admission(self, session, decode_server: str) -> None:
        if (
            self.pd_decode_prealloc_soft_limit <= 0
            and self.pd_decode_oldest_prealloc_wait_ms_soft_limit <= 0
        ):
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        poll_s = max(1, int(self.pd_router_admission_poll_ms)) / 1000.0
        start = loop.time()
        poll_count = 0
        blocked_count = 0
        while True:
            info = await fetch_backend_json(
                session,
                decode_server,
                ("get_server_info", "server_info"),
            )
            poll_count += 1
            if not self._decode_admission_blocked(info):
                self._record_decode_admission_wait(
                    loop.time() - start,
                    poll_count,
                    blocked_count,
                )
                return
            blocked_count += 1
            if loop.time() >= deadline:
                self._record_decode_admission_wait(
                    loop.time() - start,
                    poll_count,
                    blocked_count,
                )
                raise HTTPException(
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    detail="PD decode admission did not clear before request timeout",
            )
            await asyncio.sleep(poll_s)

    async def _post_prefill_with_admission(
        self,
        session,
        prefill_server: str,
        endpoint: str,
        request: dict,
    ):
        sem = await self._acquire_prefill_admission(prefill_server)
        return await self._post_prefill_and_release_admission(
            session,
            prefill_server,
            endpoint,
            request,
            sem,
        )

    async def _acquire_prefill_admission(self, prefill_server: str):
        sem = self._prefill_admission_sems.get(prefill_server)
        if sem is not None:
            start = asyncio.get_running_loop().time()
            await sem.acquire()
            self._record_prefill_admission_wait(
                asyncio.get_running_loop().time() - start
            )
        return sem

    async def _post_prefill_and_release_admission(
        self,
        session,
        prefill_server: str,
        endpoint: str,
        request: dict,
        sem,
    ):
        try:
            response = await session.post(f"{prefill_server}/{endpoint}", json=request)
            return response, await response.read()
        finally:
            if sem is not None:
                sem.release()

    async def generate(
        self,
        modified_request: dict,
        prefill_server: str,
        decode_server: str,
        endpoint: str,
    ):
        import aiohttp
        from fastapi.responses import ORJSONResponse

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        expected_decode_dp_rank = None
        if self.test_external_dp_routing:
            await self._ensure_dp_sizes()
            (
                prefill_req,
                decode_req,
                expected_decode_dp_rank,
            ) = self._fork_dp_requests(modified_request)
        else:
            prefill_req, decode_req = await self._align_dp_requests(modified_request)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            await self._wait_for_decode_admission(session, decode_server)
            prefill_sem = await self._acquire_prefill_admission(prefill_server)
            if self.pd_router_prefill_decode_overlap:
                tasks = [
                    self._post_prefill_and_release_admission(
                        session,
                        prefill_server,
                        endpoint,
                        prefill_req,
                        prefill_sem,
                    ),
                    session.post(f"{decode_server}/{endpoint}", json=decode_req),
                ]
                prefill_result, decode_response = await asyncio.gather(*tasks)
                prefill_response, prefill_body = prefill_result
            else:
                prefill_response, prefill_body = await (
                    self._post_prefill_and_release_admission(
                        session,
                        prefill_server,
                        endpoint,
                        prefill_req,
                        prefill_sem,
                    )
                )
                decode_response = await session.post(
                    f"{decode_server}/{endpoint}",
                    json=decode_req,
                )

            if "return_logprob" in modified_request:
                prefill_json = json.loads(prefill_body or b"{}")
                ret_json = await decode_response.json()
                if "meta_info" in ret_json and "input_token_logprobs" in ret_json["meta_info"]:
                    ret_json["meta_info"]["input_token_logprobs"] = (
                        prefill_json["meta_info"]["input_token_logprobs"]
                        + ret_json["meta_info"]["input_token_logprobs"]
                    )
            else:
                ret_json = await decode_response.json()

            if expected_decode_dp_rank is not None and decode_response.status < 400:
                if not isinstance(ret_json, dict):
                    return ORJSONResponse(
                        content={
                            "error": (
                                "Decode response must be a JSON object when "
                                "--test-external-dp-routing is enabled"
                            )
                        },
                        status_code=500,
                    )
                meta_info = ret_json.setdefault("meta_info", {})
                if not isinstance(meta_info, dict):
                    meta_info = {}
                    ret_json["meta_info"] = meta_info
                actual = meta_info.get("dp_rank")
                if actual is None:
                    logger.warning(
                        "[MiniLB] decode response missing meta_info.dp_rank; "
                        "assuming externally routed dp_rank=%s",
                        expected_decode_dp_rank,
                    )
                    meta_info["dp_rank"] = expected_decode_dp_rank
                elif actual != expected_decode_dp_rank:
                    return ORJSONResponse(
                        content={
                            "error": (
                                f"DP rank mismatch: expected {expected_decode_dp_rank}, "
                                f"got {actual}"
                            )
                        },
                        status_code=500,
                    )

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    async def generate_stream(
        self,
        modified_request: dict,
        prefill_server: str,
        decode_server: str,
        endpoint: str = "generate",
    ):
        import aiohttp
        import orjson
        from fastapi import HTTPException
        from fastapi.responses import StreamingResponse

        if self.test_external_dp_routing:
            # Streaming cannot enforce or verify per-side DP routing the way
            # generate() does, so fail loudly instead of silently skipping it.
            raise HTTPException(
                status_code=400,
                detail="--test-external-dp-routing is not supported with streaming",
            )

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                await self._wait_for_decode_admission(session, decode_server)
                prefill_sem = await self._acquire_prefill_admission(prefill_server)
                if self.pd_router_prefill_decode_overlap:
                    tasks = [
                        self._post_prefill_and_release_admission(
                            session,
                            prefill_server,
                            endpoint,
                            modified_request,
                            prefill_sem,
                        ),
                        session.post(f"{decode_server}/{endpoint}", json=modified_request),
                    ]
                    prefill_result, decode_response = await asyncio.gather(*tasks)
                    _, prefill_body = prefill_result
                else:
                    _, prefill_body = await self._post_prefill_and_release_admission(
                        session,
                        prefill_server,
                        endpoint,
                        modified_request,
                        prefill_sem,
                    )
                    decode_response = await session.post(
                        f"{decode_server}/{endpoint}",
                        json=modified_request,
                    )

                if modified_request.get("return_logprob", False):
                    first_prefill_chunk = next(
                        line[5:].strip()
                        for line in prefill_body.decode("utf-8").splitlines()
                        if line.startswith("data:") and "[DONE]" not in line
                    )
                    first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                    async for chunk in decode_response.content:
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                first_prefill_chunk_json["meta_info"]["input_token_logprobs"]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )
                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import ORJSONResponse, Response, StreamingResponse

    from sgl_jax.srt.disaggregation.mini_lb_helpers import (
        get_parallel_sample_num,
        inject_bootstrap_fields,
    )

    app = FastAPI()
    lb: MiniLoadBalancer | None = None
    _admission_sem: asyncio.Semaphore | None = None

    async def fetch_backend_json(
        session,
        server_url: str,
        endpoint_candidates: tuple[str, ...],
    ) -> dict:
        last_status = None
        last_error_text = ""
        for endpoint in endpoint_candidates:
            async with session.get(f"{server_url}/{endpoint}") as response:
                if response.status == 200:
                    return await response.json()
                last_status = response.status
                last_error_text = await response.text()
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=(
                f"Failed to get {endpoint_candidates[0]} from {server_url}. "
                f"Last status: {last_status}, Response: {last_error_text}"
            ),
        )

    @app.get("/health")
    async def health_check():
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tasks = []
            for server in chain(lb.prefill_urls, lb.decode_urls):
                tasks.append(session.get(f"{server}/health_generate"))
            for response in asyncio.as_completed(tasks):
                await response
        return Response(status_code=200)

    @app.post("/flush_cache")
    async def flush_cache():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tasks = []
            for server in chain(lb.prefill_urls, lb.decode_urls):
                tasks.append(session.post(f"{server}/flush_cache"))
            for response in asyncio.as_completed(tasks):
                await response
        return Response(status_code=200)

    @app.get("/server_info")
    @app.get("/get_server_info")
    async def get_server_info():
        import aiohttp

        prefill_infos = []
        decode_infos = []
        all_internal_states = []

        async with aiohttp.ClientSession() as session:
            for server in lb.prefill_urls:
                prefill_infos.append(
                    await fetch_backend_json(
                        session,
                        server,
                        ("get_server_info", "server_info"),
                    )
                )
            for server in lb.decode_urls:
                info_json = await fetch_backend_json(
                    session,
                    server,
                    ("get_server_info", "server_info"),
                )
                decode_infos.append(info_json)
                if "internal_states" in info_json:
                    all_internal_states.extend(info_json["internal_states"])

        return {
            "internal_states": (
                all_internal_states
                if all_internal_states
                else [{"last_gen_throughput": 0.0, "avg_spec_accept_length": None}]
            ),
            "prefill": prefill_infos,
            "decode": decode_infos,
            "router": lb.get_observability_state(),
        }

    async def _get_model_info_impl():
        import aiohttp

        if not lb or not lb.prefill_urls:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="There is no server registered",
            )

        target_server_url = lb.prefill_urls[0]
        async with aiohttp.ClientSession() as session:
            return ORJSONResponse(
                content=await fetch_backend_json(
                    session,
                    target_server_url,
                    ("get_model_info", "model_info"),
                )
            )

    @app.get("/model_info")
    async def model_info():
        return await _get_model_info_impl()

    @app.get("/get_model_info")
    async def get_model_info():
        return await _get_model_info_impl()

    async def _do_forward(request_data: dict, endpoint_name: str):
        if get_parallel_sample_num(request_data) > 1:
            raise HTTPException(
                status_code=400,
                detail="PD mini_lb does not support parallel sampling (n > 1)",
            )
        prefill_index, prefill_server, bootstrap_port, decode_server = lb.select_pair()
        modified_request = inject_bootstrap_fields(
            request_data,
            prefill_server=prefill_server,
            bootstrap_port=bootstrap_port,
            bootstrap_host_override=lb.prefill_bootstrap_host,
            prefill_index=prefill_index,
            prefill_count=len(lb.prefill_urls),
        )
        if request_data.get("stream", False):
            return await lb.generate_stream(
                modified_request,
                prefill_server,
                decode_server,
                endpoint=endpoint_name,
            )
        return await lb.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )

    async def _forward_to_backend(request_data: dict, endpoint_name: str):
        if _admission_sem is None:
            return await _do_forward(request_data, endpoint_name)

        # Pending admission: hold the permit while the request runs. Excess
        # requests await the semaphore (held pending at the proxy) and are
        # never rejected or aborted.
        await _admission_sem.acquire()
        released = False
        try:
            resp = await _do_forward(request_data, endpoint_name)
            if isinstance(resp, StreamingResponse):
                # Streaming returns immediately; the real work happens while the
                # body is drained. Transfer permit ownership to the iterator so
                # it is released only after the stream fully completes.
                original_iter = resp.body_iterator

                async def _release_after_stream():
                    try:
                        async for chunk in original_iter:
                            yield chunk
                    finally:
                        _admission_sem.release()

                resp.body_iterator = _release_after_stream()
                released = True
            return resp
        finally:
            if not released:
                _admission_sem.release()

    @app.post("/generate")
    async def handle_generate_request(request_data: dict):
        return await _forward_to_backend(request_data, "generate")

    @app.post("/v1/chat/completions")
    async def handle_chat_completion_request(request_data: dict):
        return await _forward_to_backend(request_data, "v1/chat/completions")

    @app.post("/v1/completions")
    async def handle_completion_request(request_data: dict):
        return await _forward_to_backend(request_data, "v1/completions")

    @app.get("/v1/models")
    async def get_models():
        import aiohttp

        prefill_server = lb.prefill_urls[0]
        async with aiohttp.ClientSession() as session:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())

except ModuleNotFoundError as exc:  # pragma: no cover
    logger.warning("mini_lb web deps unavailable: %s", exc)
    app = None
    lb = None
