"""Stage 4 H-D: PD protocol version skew.

The bootstrap stack tags every ``PrefillInfo`` with a
``protocol_version``; the decode-side ``BootstrapClient`` refuses to
hand back a peer whose version is below
``MIN_COMPATIBLE_VERSION``. This guards rolling upgrades from
returning a peer the local decode can't speak with.
"""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from sgl_jax.srt.disaggregation.bootstrap import (
    BootstrapClient,
    MIN_COMPATIBLE_VERSION,
    PROTOCOL_VERSION,
    PrefillInfo,
    _Registry,
)


def test_prefill_info_defaults_to_current_version():
    info = PrefillInfo(
        bootstrap_key="k", host="h", transfer_port=1, side_channel_port=2,
    )
    assert info.protocol_version == PROTOCOL_VERSION


def test_min_le_current():
    """If MIN > PROTOCOL_VERSION the codebase is internally
    inconsistent — no peer would ever pass the check."""

    assert MIN_COMPATIBLE_VERSION <= PROTOCOL_VERSION


def test_client_rejects_below_min_version(monkeypatch):
    client = BootstrapClient("http://nowhere", shared_secret=None)

    fake = mock.MagicMock()
    fake.raise_for_status.return_value = None
    fake.json.return_value = {
        "bootstrap_key": "k",
        "host": "10.0.0.1",
        "transfer_port": 30001,
        "side_channel_port": 9600,
        "protocol_version": MIN_COMPATIBLE_VERSION - 1,
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: fake)

    with pytest.raises(RuntimeError, match="protocol_version"):
        client.get_prefill_info(42)


def test_client_accepts_current_version(monkeypatch):
    client = BootstrapClient("http://nowhere", shared_secret=None)
    fake = mock.MagicMock()
    fake.raise_for_status.return_value = None
    fake.json.return_value = {
        "bootstrap_key": "k",
        "host": "10.0.0.1",
        "transfer_port": 30001,
        "side_channel_port": 9600,
        "protocol_version": PROTOCOL_VERSION,
    }
    monkeypatch.setattr(httpx, "get", lambda *a, **k: fake)
    info = client.get_prefill_info(42)
    assert info["host"] == "10.0.0.1"


def test_registry_stores_protocol_version():
    reg = _Registry()
    reg.register(
        PrefillInfo(
            bootstrap_key="k", host="h",
            transfer_port=1, side_channel_port=2,
            protocol_version=PROTOCOL_VERSION,
        )
    )
    rows = reg.list()
    assert rows[0].protocol_version == PROTOCOL_VERSION
