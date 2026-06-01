"""Shared topology spec + helpers for PD e2e scripts.

The matrix tests all need to talk to the same set of P / D engines
and bootstrap server, so we centralize:

  * CLI parsing for ``--p-url``, ``--d-url``, ``--bootstrap-url``,
    ``--shared-secret`` (repeated flags for multiple peers).
  * A small ``Topology`` dataclass.
  * Helpers to fire a generate request to a P or D endpoint with the
    same shape (rid, bootstrap_room, sampling params).
  * A common JSON-report writer + ``RESULT: PASS|FAIL`` printer so
    the shell driver can parse all of them uniformly.

Tests should not import any sgl_jax server code — they live entirely
on the operator side and only speak HTTP.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx


@dataclasses.dataclass
class Topology:
    p_urls: List[str]
    d_urls: List[str]
    bootstrap_url: str
    shared_secret: Optional[str] = None
    # Populated lazily by ``refresh_picker``. Maps the bootstrap-picker
    # key list (sorted host:transfer_port) to the matching p_url
    # (which uses the HTTP port). Without this, multi-P fan-out tests
    # send the prefill request to the wrong P and the D-side pull
    # hangs until pull_timeout.
    _key_to_p_url: Dict[str, str] = dataclasses.field(default_factory=dict)
    # Full sorted picker key list from the bootstrap server. Used so
    # ``pick_p_for_room`` matches the server picker exactly even
    # when the operator only passed --p-url for a subset.
    _all_picker_keys: List[str] = dataclasses.field(default_factory=list)

    @property
    def first_p(self) -> str:
        return self.p_urls[0]

    @property
    def first_d(self) -> str:
        return self.d_urls[0]

    def refresh_picker(self) -> None:
        """Read bootstrap /list_prefills and cache the
        bootstrap_key → p_url mapping by matching hosts. Call once
        per test before using ``pick_p_for_room``.

        Registry entries for P peers the operator did not pass via
        --p-url are kept in ``_all_picker_keys`` (so the picker
        modulus matches the server) but absent from
        ``_key_to_p_url`` — ``pick_p_for_room`` will raise for those
        rooms so the caller can pick a room that *does* map into the
        operator-provided subset."""

        rows = list_prefills(self)
        self._all_picker_keys = sorted(r["bootstrap_key"] for r in rows)
        for key in self._all_picker_keys:
            host = key.split(":", 1)[0]
            for u in self.p_urls:
                u_host = u.replace("http://", "").replace("https://", "")
                u_host = u_host.split(":", 1)[0]
                if u_host == host:
                    self._key_to_p_url[key] = u
                    break
        if not self._key_to_p_url:
            raise RuntimeError(
                f"no overlap between bootstrap registry "
                f"({self._all_picker_keys}) and --p-url "
                f"({self.p_urls})"
            )

    def picker_keys(self) -> List[str]:
        if not self._all_picker_keys:
            self.refresh_picker()
        return list(self._all_picker_keys)

    def pick_p_for_room(self, room: int) -> str:
        """Mirror BootstrapServer's ``room % len(sorted_keys)`` picker
        so the test sends prefill to the same P the decode side
        will pull from."""

        keys = self.picker_keys()
        if not keys:
            raise RuntimeError("no P registered in bootstrap")
        chosen = keys[room % len(keys)]
        if chosen not in self._key_to_p_url:
            raise RuntimeError(
                f"room {room} picks P {chosen!r} but no --p-url was "
                f"provided for that host. Pass --p-url for it, or use "
                f"a different room."
            )
        return self._key_to_p_url[chosen]


def add_topology_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--p-url", action="append", required=True,
        help="Repeatable. URL of one prefill engine, e.g. http://10.0.0.1:30100",
    )
    parser.add_argument(
        "--d-url", action="append", required=True,
        help="Repeatable. URL of one decode engine.",
    )
    parser.add_argument(
        "--bootstrap-url", required=True,
        help="Bootstrap server URL, e.g. http://10.0.0.1:8998",
    )
    parser.add_argument(
        "--shared-secret", default=os.environ.get("SGL_JAX_PD_SHARED_SECRET"),
        help="Bearer token for bootstrap calls (env "
        "SGL_JAX_PD_SHARED_SECRET wins).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Write JSON report here. Defaults to /tmp/pd_e2e_<script>.json.",
    )


def parse_topology(args: argparse.Namespace) -> Topology:
    return Topology(
        p_urls=args.p_url,
        d_urls=args.d_url,
        bootstrap_url=args.bootstrap_url.rstrip("/"),
        shared_secret=args.shared_secret,
    )


def _bearer(secret: Optional[str]) -> Dict[str, str]:
    return {"Authorization": f"Bearer {secret}"} if secret else {}


def list_prefills(topo: Topology) -> List[Dict[str, Any]]:
    r = httpx.get(
        f"{topo.bootstrap_url}/list_prefills",
        timeout=5.0, headers=_bearer(topo.shared_secret),
    )
    r.raise_for_status()
    return r.json()["prefills"]


def parse_bootstrap_addr(bootstrap_url: str) -> tuple[str, int]:
    """Pull host+port out of e.g. http://10.0.0.1:8998/."""

    u = bootstrap_url.replace("http://", "").replace("https://", "")
    u = u.rstrip("/")
    host, _, port = u.partition(":")
    return host, int(port or "80")


def fire_pd_request(
    role_url: str,
    *,
    rid: str,
    disagg_transfer_id: Optional[str] = None,
    prompt: str,
    bootstrap_host: str,
    bootstrap_port: int,
    bootstrap_room: int,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Send one /generate request with PD routing fields. Returns
    the parsed JSON or raises."""

    payload = {
        "rid": rid,
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": bootstrap_room,
        "disagg_transfer_id": disagg_transfer_id,
    }
    r = httpx.post(
        f"{role_url}/generate", json=payload, timeout=timeout
    )
    r.raise_for_status()
    return r.json()


def fire_pd_pair(
    topo: Topology, *,
    rid: str, prompt: str, bootstrap_room: int,
    disagg_transfer_id: Optional[str] = None,
    max_new_tokens: int = 8,
    p_url: Optional[str] = None,
    d_url: Optional[str] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Send the same logical request to one P + one D concurrently
    (mini_lb-style fan-out) and return both responses.

    If ``p_url`` is None the P is selected by ``room % len(keys)`` so
    the prefill lands on the same engine the decode will pull from.
    Passing ``p_url`` explicitly is useful when a test needs to
    *misroute* on purpose."""

    import threading

    bh, bp = parse_bootstrap_addr(topo.bootstrap_url)
    p = p_url or topo.pick_p_for_room(bootstrap_room)
    d = d_url or topo.first_d
    transfer_id = disagg_transfer_id or f"{rid}:{uuid.uuid4().hex}"
    out: Dict[str, Any] = {}
    err: Dict[str, BaseException] = {}

    def _go(label: str, url: str):
        try:
            out[label] = fire_pd_request(
                url, rid=rid, prompt=prompt,
                disagg_transfer_id=transfer_id,
                bootstrap_host=bh, bootstrap_port=bp,
                bootstrap_room=bootstrap_room,
                max_new_tokens=max_new_tokens,
                timeout=timeout,
            )
        except BaseException as e:
            err[label] = e

    tp = threading.Thread(target=_go, args=("P", p))
    td = threading.Thread(target=_go, args=("D", d))
    tp.start(); td.start()
    tp.join(timeout=timeout + 5)
    td.join(timeout=timeout + 5)
    if err:
        raise RuntimeError(f"fan-out failed: {err}")
    return out


def write_report(args: argparse.Namespace, script_name: str, summary: Dict[str, Any]) -> str:
    path = args.out or f"/tmp/pd_e2e_{script_name}.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def print_result(passed: bool, msg: str) -> int:
    """Print the line the matrix driver parses. Returns the exit code."""

    print(f"RESULT: {'PASS' if passed else 'FAIL'} {msg}", flush=True)
    return 0 if passed else 1
