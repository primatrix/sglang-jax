"""Shared-secret auth helpers for PD (Stage 4 H-C).

The three PD channels — bootstrap HTTP, transfer pull side channel,
and ZMQ ack channel — all share a single secret. Each channel
applies the secret differently (Bearer header for HTTP, HMAC tag
beside the payload for ZMQ / transfer), but the resolution rules and
constant-time compare live here so every channel agrees on the same
edge cases.

Resolution order:
  1. ``SGL_JAX_PD_SHARED_SECRET`` environment variable.
  2. ``ServerArgs.disaggregation_shared_secret``.
  3. ``None`` → auth is disabled. Channels MUST treat ``None`` as
     "anyone can connect" so existing deployments don't break when
     they upgrade to a release that ships H-C without enabling it.

Authn failures should bump
``pd_transfer_failures_total{reason="auth"}`` so dashboards can
distinguish credential rot from network / peer crashes.
"""

from __future__ import annotations

import hmac
import os
from hashlib import sha256
from typing import Optional


_ENV_VAR = "SGL_JAX_PD_SHARED_SECRET"


def resolve_secret(server_args_value: Optional[str]) -> Optional[str]:
    """Return the effective shared secret (or ``None`` if auth is off).

    Env wins over CLI/config so an operator can rotate the secret
    without restarting via config push.
    """

    env = os.environ.get(_ENV_VAR)
    if env:
        return env
    return server_args_value


def compute_tag(secret: str, payload: bytes) -> bytes:
    """HMAC-SHA256 over ``payload`` with ``secret``. Currently used by
    the ZMQ ack channel (D → P ``send_done``). The transfer-pull
    pre-handshake variant described in RFC §2 is DEFERRED — see
    follow-up RFC; until then the receiver-side ack is the only
    HMAC-protected channel."""

    return hmac.new(secret.encode("utf-8"), payload, sha256).digest()


def verify_tag(
    secret: Optional[str], payload: bytes, candidate: Optional[bytes]
) -> bool:
    """Constant-time compare of ``candidate`` against the expected tag.

    Returns ``True`` if auth is disabled (``secret is None``) — the
    caller has opted out by not configuring a secret.
    """

    if secret is None:
        return True
    if candidate is None:
        return False
    expected = compute_tag(secret, payload)
    return hmac.compare_digest(expected, candidate)


def bearer_header(secret: Optional[str]) -> dict:
    """Return ``{"Authorization": "Bearer <secret>"}`` or ``{}``."""

    if secret is None:
        return {}
    return {"Authorization": f"Bearer {secret}"}


def verify_bearer(
    secret: Optional[str], header_value: Optional[str]
) -> bool:
    """Validate an HTTP ``Authorization`` header.

    With auth disabled (``secret is None``), every request passes
    regardless of the header. With auth enabled, the header must be
    ``Bearer <secret>`` (constant-time compared).
    """

    if secret is None:
        return True
    if not header_value or not header_value.startswith("Bearer "):
        return False
    candidate = header_value[len("Bearer "):]
    return hmac.compare_digest(secret, candidate)
