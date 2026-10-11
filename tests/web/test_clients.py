import httpx
import pytest
from starlette.requests import Request

from appalti_qa.web.clients import ClientHasher, TrustedProxies, client_address, client_bucket


def request(peer: str, forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "client": (peer, 1234), "headers": headers})


def test_trusted_proxies_parse_networks() -> None:
    trusted = TrustedProxies.parse("127.0.0.1, 10.0.0.0/8,,::1")
    assert "127.0.0.1" in trusted
    assert "10.20.30.40" in trusted
    assert "::1" in trusted
    assert "192.168.1.1" not in trusted
    assert "testclient" not in trusted
    assert "anything" in TrustedProxies(["*"])


def test_invalid_proxy_entry_fails_fast() -> None:
    with pytest.raises(ValueError):
        TrustedProxies.parse("not-an-ip")


def test_forwarded_header_ignored_without_trusted_proxy() -> None:
    req = request("203.0.113.9", "198.51.100.1")
    assert client_address(req, TrustedProxies()) == "203.0.113.9"


def test_forwarded_header_read_from_trusted_proxy() -> None:
    trusted = TrustedProxies.parse("10.0.0.0/8")
    # spoofed first hop, real client added by our proxy chain
    req = request("10.0.0.2", "1.2.3.4, 198.51.100.1, 10.0.0.5")
    assert client_address(req, trusted) == "198.51.100.1"
    assert client_address(request("10.0.0.2"), trusted) == "10.0.0.2"
    assert client_address(request("10.0.0.2", "10.0.0.9"), trusted) == "10.0.0.9"


def test_trust_any_uses_only_the_hop_added_by_the_proxy() -> None:
    trusted = TrustedProxies.parse("*")
    assert trusted.trusts_any and not TrustedProxies.parse("10.0.0.0/8").trusts_any
    # the client wrote "1.2.3.4"; the proxy appended the address it saw
    req = request("172.18.0.2", "1.2.3.4, 198.51.100.1")
    assert client_address(req, trusted) == "198.51.100.1"
    assert client_address(request("172.18.0.2", " 198.51.100.1 "), trusted) == "198.51.100.1"
    assert client_address(request("172.18.0.2"), trusted) == "172.18.0.2"


@pytest.mark.parametrize(
    ("address", "bucket"),
    [
        ("198.51.100.1", "198.51.100.1"),
        ("2001:db8:1:2:3:4:5:6", "2001:db8:1:2::/64"),
        ("2001:DB8:1:2::ffff", "2001:db8:1:2::/64"),
        ("::ffff:198.51.100.1", "198.51.100.1"),
        ("fe80::1%eth0", "fe80::/64"),
        ("testclient", "testclient"),
        ("", ""),
    ],
)
def test_client_bucket(address: str, bucket: str) -> None:
    assert client_bucket(address) == bucket


def test_ipv6_clients_are_hashed_per_64() -> None:
    hasher = ClientHasher("salt")
    assert hasher("2001:db8:1:2::1") == hasher("2001:db8:1:2:ffff:ffff:ffff:ffff")
    assert hasher("2001:db8:1:2::1") != hasher("2001:db8:1:3::1")
    assert hasher("::ffff:198.51.100.1") == hasher("198.51.100.1")


def test_rate_limit_spans_an_ipv6_64(make_client) -> None:
    client = make_client(rate_limit_per_ip_per_hour=1, trusted_proxies=("*",), peer="10.0.0.2")

    def ask(forwarded: str) -> httpx.Response:
        headers = {"X-Forwarded-For": forwarded}
        return client.post("/api/v1/answer", json={"question": "Domanda?"}, headers=headers)

    assert ask("2001:db8:1:2::1").status_code == 200
    assert ask("2001:db8:1:2::abcd").status_code == 429  # same /64
    assert ask("198.51.100.9, 2001:db8:1:2::1").status_code == 429  # spoofed leftmost hop
    assert ask("2001:db8:1:3::1").status_code == 200


def test_client_hash_is_salted() -> None:
    first, second = ClientHasher("salt-a"), ClientHasher("salt-b")
    assert first("198.51.100.1") == first("198.51.100.1")
    assert first("198.51.100.1") != second("198.51.100.1")
    assert len(first("198.51.100.1")) == 64
    assert ClientHasher()("x") != ClientHasher()("x")  # random salt per instance


def test_rate_limit_follows_forwarded_client(make_client) -> None:
    client = make_client(
        rate_limit_per_ip_per_hour=1, trusted_proxies=("10.0.0.0/8",), peer="10.0.0.2"
    )

    def ask(address: str) -> httpx.Response:
        headers = {"X-Forwarded-For": address}
        return client.post("/api/v1/answer", json={"question": "Domanda?"}, headers=headers)

    assert ask("198.51.100.1").status_code == 200
    assert ask("198.51.100.2").status_code == 200
    assert ask("198.51.100.1").status_code == 429


def test_rate_limit_ignores_forwarded_header_by_default(make_client) -> None:
    client = make_client(rate_limit_per_ip_per_hour=1)
    for address, status in (("198.51.100.1", 200), ("198.51.100.2", 429)):
        response = client.post(
            "/api/v1/answer", json={"question": "Domanda?"}, headers={"X-Forwarded-For": address}
        )
        assert response.status_code == status
