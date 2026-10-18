"""SSRF protection: the agent may only fetch from public internet addresses.

The model chooses which URLs to fetch, and a web page (or a crafted prompt) can
steer it towards internal targets: `http://localhost:8000/api/...`, a database
on the private network, or the cloud metadata service at 169.254.169.254, which
hands out credentials. Checking the URL's hostname is not enough:

- a public-looking name can resolve to a private IP (or be re-pointed between
  the check and the request, "DNS rebinding");
- a public page can redirect to a private address.

So the check lives at the lowest layer instead: a custom network backend that
resolves the hostname, rejects the connection unless every address is public,
and then connects to exactly the address it checked. Every connection the HTTP
client opens, including each redirect hop, goes through it.
"""

import ipaddress
import socket
from collections.abc import Awaitable, Callable, Iterable

import anyio
import httpcore
import httpx

ALLOWED_PORTS = frozenset({80, 443})

Resolver = Callable[[str, int], Awaitable[list[str]]]
"""(host, port) -> the IP addresses the host resolves to."""


class BlockedAddressError(Exception):
    """The request would reach a non-public address or port."""


async def system_resolver(host: str, port: int) -> list[str]:
    """Resolve with the OS resolver (A and AAAA records)."""
    try:
        infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise BlockedAddressError(f"{host} could not be resolved") from exc
    return [str(info[4][0]) for info in infos]


def is_public_address(address: str) -> bool:
    """True only for globally routable unicast addresses.

    Rejects loopback, private (10/8, 172.16/12, 192.168/16, fc00::/7), link-local
    (169.254/16, including cloud metadata), carrier-grade NAT, multicast, and
    reserved ranges, and looks inside IPv6 forms that embed an IPv4 address.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop an IPv6 zone id
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if embedded is not None:
            ip = embedded
    return ip.is_global and not ip.is_multicast


class SafeNetworkBackend(httpcore.AsyncNetworkBackend):
    """Opens TCP connections only to public addresses on ports 80 and 443."""

    def __init__(self, resolver: Resolver = system_resolver) -> None:
        self._inner = httpcore.AnyIOBackend()
        self._resolver = resolver

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 (httpcore's interface)
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        """Resolve, check every address, then connect to the checked address."""
        if port not in ALLOWED_PORTS:
            raise BlockedAddressError(f"port {port} is not allowed (only 80 and 443)")
        addresses = await self._resolver(host, port)
        if not addresses:
            raise BlockedAddressError(f"{host} did not resolve to any address")
        # All of them, not just the first: the OS may pick any of them.
        private = [a for a in addresses if not is_public_address(a)]
        if private:
            raise BlockedAddressError(f"{host} resolves to a non-public address ({private[0]})")
        # Connecting to the IP (not the name) closes the DNS-rebinding window. TLS
        # still verifies the certificate against the original hostname.
        return await self._inner.connect_tcp(
            addresses[0],
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 (httpcore's interface)
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        """Unix sockets are local by definition."""
        raise BlockedAddressError("unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        """Delegate to the standard backend."""
        await self._inner.sleep(seconds)


class SafeTransport(httpx.AsyncHTTPTransport):
    """An httpx transport whose every connection goes through SafeNetworkBackend."""

    def __init__(self, resolver: Resolver = system_resolver) -> None:
        super().__init__(trust_env=False)
        # httpx has no public option for the network backend, so replace the
        # connection pool it built with one that uses ours (same TLS defaults).
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=httpx.create_ssl_context(),
            network_backend=SafeNetworkBackend(resolver),
        )
