from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import zmq

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData

_STOP = object()


@dataclass(slots=True)
class _PublishJob:
    data: EmbeddingData
    done: Future[None]


class EncoderMetadataPublisher:
    """Publish ready metadata without depending on the HTTP event loop."""

    def __init__(self, timeout_s: float | None) -> None:
        self._timeout_s = timeout_s
        self._condition = threading.Condition()
        self._addresses: dict[str, str] = {}
        self._jobs: queue.SimpleQueue[_PublishJob | object] = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="encoder-metadata-publisher",
            daemon=True,
        )
        self._thread.start()

    def register(self, req_id: str, address: str) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("encoder metadata publisher is closed")
            self._addresses[req_id] = address
            self._condition.notify_all()

    def publish(self, data: EmbeddingData) -> None:
        done: Future[None] = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("encoder metadata publisher is closed")
            self._jobs.put(_PublishJob(data, done))
        timeout = self._timeout_s
        done.result(timeout=None if timeout is None or timeout <= 0 else timeout)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._jobs.put(_STOP)
        self._thread.join()

    def _wait_for_address(self, req_id: str) -> str:
        timeout = self._timeout_s
        deadline = None if timeout is None or timeout <= 0 else time.monotonic() + timeout
        with self._condition:
            while req_id not in self._addresses:
                if self._closed:
                    raise RuntimeError("encoder metadata publisher closed before registration")
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for receiver registration: {req_id}")
                self._condition.wait(remaining)
            return self._addresses.pop(req_id)

    def _run(self) -> None:
        context = zmq.Context.instance()
        sockets: dict[str, zmq.Socket] = {}
        try:
            while True:
                item = self._jobs.get()
                if item is _STOP:
                    return
                assert isinstance(item, _PublishJob)
                try:
                    address = self._wait_for_address(item.data.req_id)
                    socket = sockets.get(address)
                    if socket is None:
                        socket = context.socket(zmq.PUSH)
                        socket.setsockopt(zmq.LINGER, 1000)
                        socket.connect(f"tcp://{address}")
                        sockets[address] = socket
                    socket.send_pyobj(item.data)
                except BaseException as exc:
                    item.done.set_exception(exc)
                else:
                    item.done.set_result(None)
        finally:
            for socket in sockets.values():
                socket.close()
