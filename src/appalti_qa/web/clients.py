"""Client identification for rate limiting and the usage log.

The client address is the TCP peer, unless the peer is a trusted reverse proxy: only then is
`X-Forwarded-For` read, right to left, skipping trusted hops. Addresses are never stored: they are
reduced to a salted SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Iterable
from ipaddress import IPv4Network, IPv6Network, ip_address, ip_network

from starlette.requests import Request


class TrustedProxies:
    """A set of proxy addresses or networks ("10.0.0.0/8"); "*" trusts every peer."""

    def __init__(self, entries: Iterable[str] = ()) -> None:
        self._any = False
        self._networks: list[IPv4Network | IPv6Network] = []
        for entry in (item.strip() for item in entries):
            if entry == "*":
                self._any = True
            elif entry:
                self._networks.append(ip_network(entry, strict=False))

    @classmethod
    def parse(cls, value: str) -> TrustedProxies:
        """From a comma-separated list, as in the FORWARDED_ALLOW_IPS convention."""
        return cls(value.split(","))

    def __contains__(self, host: object) -> bool:
        if self._any:
            return True
        try:
            address = ip_address(str(host))
        except ValueError:
            return False
        return any(address in network for network in self._networks)


def client_address(request: Request, trusted: TrustedProxies) -> str:
    """The address of the client that sent the request, as far as trusted proxies tell."""
    peer = request.client.host if request.client else ""
    if peer not in trusted:
        return peer
    hops = [hop.strip() for hop in request.headers.get("x-forwarded-for", "").split(",")]
    hops = [hop for hop in hops if hop]
    for hop in reversed(hops):
        if hop not in trusted:
            return hop
    return hops[0] if hops else peer


class ClientHasher:
    """Salted digests of client addresses (random salt per process when none is configured)."""

    def __init__(self, salt: str | None = None) -> None:
        self._salt = (salt or secrets.token_hex(16)).encode()

    def __call__(self, address: str) -> str:
        return hashlib.sha256(self._salt + b"\x00" + address.encode()).hexdigest()
