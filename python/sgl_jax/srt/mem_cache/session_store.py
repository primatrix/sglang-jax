"""Scheduler-owned session leases. No device operations or request concatenation.

The cache supplies a release callback: the store never guesses how a model's
KV, sliding window, or recurrent continuation state should be freed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Any


@dataclass
class Session:
    active: Any = None
    owner: Any = None
    tokens: tuple[int, ...] = ()
    touched: float = 0.0
    closing: bool = False


class SessionStore:
    def __init__(self, release: Callable, timeout=300.0, capacity=128, clock=monotonic):
        if not isfinite(timeout) or timeout <= 0 or capacity <= 0:
            raise ValueError("Session timeout and capacity must be positive")
        self.release = release
        self.timeout = timeout
        self.capacity = capacity
        self.clock = clock
        self.sessions: dict[str, Session] = {}

    def open(self, session_id: str):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_params.id must be a nonempty string")
        self.reap()
        if session_id in self.sessions:
            raise ValueError("Session already exists")
        if len(self.sessions) >= self.capacity and not self.evict_one():
            raise ValueError("All session slots are busy; retry after a request finishes")
        self.sessions[session_id] = Session(touched=self.clock())

    def acquire(self, session_id: str, req):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_params.id must be a nonempty string")
        self.reap()
        if session_id not in self.sessions:
            self.open(session_id)
        session = self.sessions[session_id]
        if session.active is not None or session.closing:
            raise ValueError("Session already has an in-flight request or is closing")
        session.active = req
        session.touched = self.clock()
        return session

    def discard_cache(self, session):
        if session.owner is not None:
            self.release(session.owner)
        session.owner = None
        session.tokens = ()

    def finish(self, session_id, req, *, retain=False, tokens=()):
        session = self.sessions.get(session_id)
        if session is None or session.active is not req:
            return False
        self.discard_cache(session)
        session.active = None
        session.touched = self.clock()
        if retain and not session.closing:
            session.owner = req
            session.tokens = tuple(tokens)
        else:
            self.release(req)
        if session.closing:
            del self.sessions[session_id]
        return True

    def close(self, session_id):
        session = self.sessions.get(session_id)
        if session is None:
            return
        if session.active is not None:
            session.closing = True
        else:
            self.discard_cache(session)
            del self.sessions[session_id]

    def reap(self):
        now = self.clock()
        for session_id, session in list(self.sessions.items()):
            if session.active is None and now - session.touched >= self.timeout:
                self.close(session_id)

    def evict_one(self):
        idle = [(s.touched, key) for key, s in self.sessions.items() if s.active is None]
        if not idle:
            return False
        self.close(min(idle)[1])
        return True

    def reset(self):
        if any(s.active is not None for s in self.sessions.values()):
            raise RuntimeError("Cannot reset sessions with in-flight requests")
        for key in list(self.sessions):
            self.close(key)
