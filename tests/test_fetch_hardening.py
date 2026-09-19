"""Reference fetch hardening (Task 4, §3.4): host/IP policy per hop, manual
redirects, streaming byte and time caps.

Unlike tests/test_error_contract.py (which exercises the existing HTTP-status
mapping via a minimal `_FakeResponse` with only `status_code` +
`raise_for_status`), this suite drives the redirect/stream loop itself:
`requests.get` is patched to return richer fakes with `headers`,
`iter_content()` and `close()`, and `image_utils._resolve_host` is patched
per test (an autouse fixture in conftest.py pins it to a public IP by
default; test 13 below undoes that to exercise the real resolver).
"""

from __future__ import annotations

import io
import ipaddress
import shutil
import socket
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

from face_recognition_service.config import settings
from face_recognition_service.schemas.api_schemas import ErrorCode
from face_recognition_service.utils import image_utils as image_utils_module
from face_recognition_service.utils.image_utils import (
    ImageProcessingError,
    fetch_image_from_url,
)

AUTH_HEADERS = {"Authorization": f"Bearer {settings.api_token}"}

# Saved at import time (before the autouse `_stub_resolve_host` fixture in
# conftest.py ever runs) so TestRealResolver can restore the genuine
# `_resolve_host` explicitly instead of `monkeypatch.undo()` -- `undo()`
# rolls back *every* patch this test's `monkeypatch` fixture instance has
# made so far, including ones unrelated to `_resolve_host`, which is a much
# blunter (and more fragile, if this test ever grows another patch before
# this point) revert than just putting the real function back.
_REAL_RESOLVE_HOST = image_utils_module._resolve_host


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeResponse:
    """requests.Response stand-in with everything the hop loop touches."""

    def __init__(
        self,
        status_code: int,
        *,
        headers: dict | None = None,
        content: bytes = b"",
        chunk_size: int = 65536,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._content = content
        self._chunk_size = chunk_size
        self.closed = False

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            error = requests.exceptions.HTTPError(f"{self.status_code} error for url: ...")
            error.response = self
            raise error

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    def close(self) -> None:
        self.closed = True


class _InfiniteResponse:
    """A response whose iter_content never ends -- for the byte-cap test."""

    def __init__(self) -> None:
        self.status_code = 200
        self.headers: dict = {}
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        while True:
            yield b"a" * chunk_size

    def close(self) -> None:
        self.closed = True


class _SlowResponse:
    """A response whose iter_content yields chunks but never finishes -- for
    the total-deadline test (paired with a patched time.monotonic)."""

    def __init__(self, n_chunks: int) -> None:
        self.status_code = 200
        self.headers: dict = {}
        self.closed = False
        self._n_chunks = n_chunks

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for _ in range(self._n_chunks):
            yield b"a" * 100

    def close(self) -> None:
        self.closed = True


URL = "https://reference.example.com/photos/photo123.jpg"
SIGNED_URL = "https://reference.example.com/photos/photo123.jpg?sig=super-secret-token&exp=999"
SECRET = "super-secret-token"


class TestRequestShape:
    def test_get_called_with_manual_redirects_stream_and_tuple_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def _get(url, **kwargs):
            calls.append((url, kwargs))
            return _FakeResponse(200, content=_png_bytes())

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        fetch_image_from_url(URL)

        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        assert kwargs["timeout"] == (
            settings.fetch_connect_timeout,
            settings.fetch_read_timeout,
        )

    def test_accept_encoding_has_no_br(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def _get(url, **kwargs):
            calls.append(kwargs)
            return _FakeResponse(200, content=_png_bytes())

        monkeypatch.setattr(image_utils_module.requests, "get", _get)
        fetch_image_from_url(URL)

        accept_encoding = calls[0]["headers"]["Accept-Encoding"]
        assert "br" not in accept_encoding
        assert "gzip" in accept_encoding
        assert "deflate" in accept_encoding


class TestRedirectHandling:
    def test_relative_redirect_then_200_joins_url_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def _get(url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                return _FakeResponse(302, headers={"Location": "/photos/photo123-final.jpg"})
            return _FakeResponse(200, content=_png_bytes())

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        image = fetch_image_from_url(URL)

        assert image is not None
        assert calls[0] == URL
        assert calls[1] == "https://reference.example.com/photos/photo123-final.jpg"

    def test_redirect_to_disallowed_host_is_not_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "reference_url_allowed_hosts", ["reference.example.com"])

        def _get(url, **kwargs):
            if url == URL:
                return _FakeResponse(302, headers={"Location": "https://evil.example.com/x.jpg"})
            raise AssertionError("should not follow to evil.example.com")

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED

    def test_redirect_to_private_host_blocked_when_allow_private_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "reference_url_allow_private", False)

        def _resolve_host(host, **kwargs):
            if host == "reference.example.com":
                return [ipaddress.ip_address("93.184.216.34")]
            return [ipaddress.ip_address("10.0.0.5")]

        monkeypatch.setattr(image_utils_module, "_resolve_host", _resolve_host)

        def _get(url, **kwargs):
            if url == URL:
                return _FakeResponse(302, headers={"Location": "https://internal.example.com/x.jpg"})
            raise AssertionError("should not be reached")

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED

    def test_too_many_redirects_is_service_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "fetch_max_redirects", 3)
        calls = {"n": 0}

        def _get(url, **kwargs):
            calls["n"] += 1
            return _FakeResponse(302, headers={"Location": f"/hop{calls['n']}.jpg"})

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE
        # at most fetch_max_redirects + 1 requests
        assert calls["n"] <= 4

    def test_redirect_missing_location_is_service_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda url, **kwargs: _FakeResponse(302)
        )

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE


class TestHostIpPolicy:
    def test_metadata_ip_always_blocked_even_with_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            image_utils_module,
            "_resolve_host",
            lambda host, **kwargs: [ipaddress.ip_address("169.254.169.254")],
        )
        monkeypatch.setattr(
            image_utils_module.requests,
            "get",
            lambda url, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
        )

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED

    def test_loopback_passes_with_defaults_blocked_with_allow_private_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            image_utils_module, "_resolve_host", lambda host, **kwargs: [ipaddress.ip_address("127.0.0.1")]
        )
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda url, **kwargs: _FakeResponse(200, content=_png_bytes())
        )

        # Passes with default allow_private=True
        assert fetch_image_from_url(URL) is not None

        monkeypatch.setattr(settings, "reference_url_allow_private", False)
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED

    def test_allowlist_is_case_insensitive_exact_host_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "reference_url_allowed_hosts", ["reference.example.com"])
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda url, **kwargs: _FakeResponse(200, content=_png_bytes())
        )

        assert fetch_image_from_url("https://REFERENCE.example.com/x.jpg") is not None

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url("https://evil.example.com/x.jpg")
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED

    def test_allowlist_ignores_a_trailing_dot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # M2: "reference.example.com." (trailing-dot FQDN form) must match an
        # allowlist entry of "reference.example.com" -- and vice versa -- rather
        # than being rejected as a different host.
        monkeypatch.setattr(settings, "reference_url_allowed_hosts", ["reference.example.com"])
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda url, **kwargs: _FakeResponse(200, content=_png_bytes())
        )

        assert fetch_image_from_url("https://reference.example.com./x.jpg") is not None

    @pytest.mark.parametrize("ip", ["100.100.100.200", "fd00:ec2::254"])
    def test_explicit_metadata_literals_always_blocked(
        self, monkeypatch: pytest.MonkeyPatch, ip: str
    ) -> None:
        # I2: these two don't fall into any of the address *classes*
        # (link-local/multicast/unspecified/reserved) Python's `ipaddress`
        # recognizes, so they need an explicit always-blocked literal list.
        monkeypatch.setattr(image_utils_module, "_resolve_host", lambda host, **kwargs: [ipaddress.ip_address(ip)])
        monkeypatch.setattr(
            image_utils_module.requests,
            "get",
            lambda url, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
        )

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED

    @pytest.mark.parametrize("ip", ["100.64.0.1", "fc00::1"])
    def test_cgnat_and_ula_pass_by_default_blocked_with_allow_private_false(
        self, monkeypatch: pytest.MonkeyPatch, ip: str
    ) -> None:
        # I2: CGNAT (100.64.0.0/10) and IPv6 ULA (fc00::/7) aren't
        # `is_private`/`is_loopback` in Python's `ipaddress`, so the old gate
        # missed them; `not ip.is_global` catches both once
        # REFERENCE_URL_ALLOW_PRIVATE=false. Neither is always-blocked (must
        # not lock out a legitimately CGNAT'd or ULA'd reference host by default).
        monkeypatch.setattr(image_utils_module, "_resolve_host", lambda host, **kwargs: [ipaddress.ip_address(ip)])
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda url, **kwargs: _FakeResponse(200, content=_png_bytes())
        )

        assert fetch_image_from_url(URL) is not None

        monkeypatch.setattr(settings, "reference_url_allow_private", False)
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED


class TestParserDifferential:
    """C1: `_check_reference_url` must reject a URL whose host, as `requests`/
    urllib3 will actually connect to it, differs from what `urlsplit` reads --
    a backslash or userinfo trick can make the two parsers disagree."""

    def test_backslash_userinfo_trick_is_not_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "reference_url_allowed_hosts", ["reference.example.com"])
        monkeypatch.setattr(
            image_utils_module.requests,
            "get",
            lambda url, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
        )

        # urlsplit reads the host as reference.example.com (the allowed one);
        # requests/urllib3 actually connect to 127.0.0.1:9999.
        evil = "http://127.0.0.1:9999\\@reference.example.com/photo.jpg"
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(evil)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED

    def test_backslash_userinfo_trick_via_redirect_is_not_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "reference_url_allowed_hosts", ["reference.example.com"])

        def _get(url, **kwargs):
            if url == URL:
                return _FakeResponse(
                    302,
                    headers={"Location": "http://127.0.0.1:9999\\@reference.example.com/x.jpg"},
                )
            raise AssertionError(f"must not be called for {url}")

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED

    def test_userinfo_form_targeting_metadata_ip_is_not_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Default settings (no allowlist): urlsplit and requests/urllib3
        # agree on the host here (both read "a"), but the point of this form
        # is testing the userinfo ("169.254.169.254" as the *username*, not
        # the host) doesn't smuggle the metadata IP past anything -- the
        # actual connect target must be checked, which it is ("a" resolves
        # to whatever _resolve_host says).
        monkeypatch.setattr(
            image_utils_module.requests,
            "get",
            lambda url, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
        )
        monkeypatch.setattr(
            image_utils_module, "_resolve_host", lambda host, **kwargs: [ipaddress.ip_address("169.254.169.254")]
        )

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url("http://169.254.169.254%2f@a/")
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED

    def test_userinfo_form_allowed_com_at_evil_com_connects_to_evil(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # http://allowed.com@evil.com/ -- both urlsplit and requests/urllib3
        # agree the host is evil.com (no backslash trick here), so the
        # allowlist correctly rejects it as evil.com, not allowed.com.
        monkeypatch.setattr(settings, "reference_url_allowed_hosts", ["allowed.com"])
        monkeypatch.setattr(
            image_utils_module.requests,
            "get",
            lambda url, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
        )

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url("http://allowed.com@evil.com/")
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED


class TestBracketedIPv6Literal:
    """Regression: `urllib3.util.parse_url(...).host` keeps the brackets on
    a literal IPv6 host (e.g. "[::1]"), but `urlsplit(url).hostname` strips
    them -- without normalising that away, the parser-differential check
    (C1's fix) would reject *every* bracketed IPv6 literal as a host
    mismatch, including an ordinary public one."""

    def test_public_looking_ipv6_literal_passes_under_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            image_utils_module, "_resolve_host", lambda host, **kwargs: [ipaddress.ip_address("2001:db8::1")]
        )
        monkeypatch.setattr(
            image_utils_module.requests, "get", lambda url, **kwargs: _FakeResponse(200, content=_png_bytes())
        )

        assert fetch_image_from_url("http://[2001:db8::1]/x.jpg") is not None

    def test_bracketed_ipv4_mapped_metadata_still_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The host here *is* the IP literal being tested, so use the real
        # parsing behaviour (bracket-stripped literal -> ipaddress.ip_address)
        # rather than the autouse stub, which ignores its `host` argument
        # entirely and would make this pass under an unrelated public IP.
        monkeypatch.setattr(
            image_utils_module, "_resolve_host", lambda host, **kwargs: [ipaddress.ip_address(host)]
        )
        monkeypatch.setattr(
            image_utils_module.requests,
            "get",
            lambda url, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
        )

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url("http://[::ffff:a9fe:a9fe]/")
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED


class TestIPv4MappedNormalisation:
    """An IPv4-mapped IPv6 address (`::ffff:a.b.c.d`) is a distinct
    `IPv6Address` object from the equivalent `IPv4Address` -- without
    unwrapping it first, it wouldn't match `_ALWAYS_BLOCKED_LITERALS`
    membership (a set of plain `IPv4Address`/`IPv6Address` objects), letting
    the mapped form of a blocked address slip through."""

    @pytest.mark.parametrize(
        "mapped",
        ["::ffff:100.100.100.200", "::ffff:169.254.169.254"],
    )
    def test_ipv4_mapped_metadata_addresses_always_blocked(
        self, monkeypatch: pytest.MonkeyPatch, mapped: str
    ) -> None:
        monkeypatch.setattr(image_utils_module, "_resolve_host", lambda host, **kwargs: [ipaddress.ip_address(mapped)])
        monkeypatch.setattr(
            image_utils_module.requests,
            "get",
            lambda url, **kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
        )

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.REFERENCE_URL_NOT_ALLOWED


class TestRedirectHopErrorCodes:
    """M1: a malformed redirect target (bad scheme / no host) is our own
    fetch-path fault, not evidence the original reference photo is bad --
    SERVICE_UNAVAILABLE (untagged), not INVALID_IMAGE (tagged "reference",
    which would lock the enrolled person out). The *original* URL keeps
    INVALID_IMAGE since that's the client's own input."""

    def test_first_hop_bad_scheme_is_invalid_image(self) -> None:
        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url("ftp://reference.example.com/x.jpg")
        assert exc_info.value.error_code == ErrorCode.INVALID_IMAGE

    def test_redirect_hop_to_non_http_scheme_is_service_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _get(url, **kwargs):
            if url == URL:
                return _FakeResponse(302, headers={"Location": "ftp://reference.example.com/x.jpg"})
            raise AssertionError(f"must not be called for {url}")

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE

    def test_redirect_hop_to_hostless_url_is_service_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _get(url, **kwargs):
            if url == URL:
                return _FakeResponse(302, headers={"Location": "http:///no-host.jpg"})
            raise AssertionError(f"must not be called for {url}")

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE

    def test_redirect_hop_urljoin_parse_failure_is_service_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `urljoin` calls `urlsplit` internally and can itself raise a raw
        # `ValueError` on a malformed bracketed-IPv6 netloc (e.g.
        # "http://[::1]@b.com/") *before* `_check_reference_url` ever runs
        # on the joined URL -- that raw ValueError must still map to
        # SERVICE_UNAVAILABLE (a redirect target is always hop > 0), not
        # fall through to the generic INVALID_IMAGE handler.
        def _get(url, **kwargs):
            if url == URL:
                return _FakeResponse(302, headers={"Location": "http://[::1]@b.com/"})
            raise AssertionError(f"must not be called for {url}")

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE


class TestByteAndTimeCaps:
    def test_infinite_stream_stops_at_image_too_large(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _InfiniteResponse()
        monkeypatch.setattr(image_utils_module.requests, "get", lambda url, **kwargs: resp)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.IMAGE_TOO_LARGE
        assert resp.closed

    def test_content_length_too_large_short_circuits_before_iter_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = {"iter_content": False}

        class _Resp:
            status_code = 200
            headers = {"Content-Length": "999999999"}
            closed = False

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                called["iter_content"] = True
                return iter([])

            def close(self):
                self.closed = True

        monkeypatch.setattr(image_utils_module.requests, "get", lambda url, **kwargs: _Resp())

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.IMAGE_TOO_LARGE
        assert called["iter_content"] is False

    def test_total_deadline_exceeded_is_service_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _SlowResponse(n_chunks=5)
        monkeypatch.setattr(image_utils_module.requests, "get", lambda url, **kwargs: resp)

        state = {"t": 0.0}

        def _monotonic():
            state["t"] += 25.0
            return state["t"]

        monkeypatch.setattr(image_utils_module.time, "monotonic", _monotonic)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE


class _DripServer:
    """A real localhost HTTP/1.1 server that drips its response slowly --
    each individual write sits comfortably under a short read timeout, so
    only a wall-clock deadline (not requests' own per-op timeout) can bound
    it. Used to prove the watchdog actually force-closes a stalled
    connection instead of just checking the deadline between full chunks
    (which a drip that never completes a chunk would never trip).
    """

    def __init__(self, mode: str, *, n_chunks: int = 16, interval: float = 0.5) -> None:
        self.mode = mode
        self.n_chunks = n_chunks
        self.interval = interval
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> int:
        mode, n_chunks, interval = self.mode, self.n_chunks, self.interval

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                try:
                    if mode == "headers":
                        self.wfile.write(b"HTTP/1.1 200 OK\r\n")
                        self.wfile.flush()
                        for _ in range(n_chunks):
                            time.sleep(interval)
                            self.wfile.write(b"X-Drip: x\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"Content-Length: 1\r\n\r\nx")
                    else:
                        self.send_response(200)
                        self.send_header("Content-Length", "100000")
                        self.end_headers()
                        for _ in range(n_chunks):
                            time.sleep(interval)
                            self.wfile.write(b"a")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # the watchdog closed us first -- expected
                finally:
                    # This server only ever expects one request per
                    # connection. Without this, a client that fetches the
                    # response but is never force-closed by the watchdog
                    # (e.g. the fix under test not being in place yet) would
                    # leave an HTTP/1.1 keep-alive connection open on both
                    # ends; this handler thread would then block forever in
                    # the next handle_one_request() waiting for a second
                    # request that's never coming, and __exit__'s
                    # server.shutdown() (which waits for the in-flight
                    # request to finish) would hang the whole test run.
                    self.close_connection = True

            def log_message(self, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.server.server_address[1]

    def __exit__(self, *exc_info: object) -> None:
        assert self.server is not None
        self.server.shutdown()
        self.server.server_close()


class TestDeadlineWatchdog:
    """I1: the total-deadline wall clock must bound a slow drip that never
    trips requests' own per-op read timeout, in both the header phase
    (blocking inside `requests.get()` itself) and the body phase (blocking
    inside `iter_content()`), using a real localhost server -- not a fake
    `requests.get` -- so the watchdog's socket-capture hook is genuinely
    exercised.
    """

    def test_header_drip_times_out_within_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "fetch_total_timeout", 1.0)
        monkeypatch.setattr(settings, "fetch_read_timeout", 0.5)
        monkeypatch.setattr(settings, "fetch_connect_timeout", 0.5)

        with _DripServer("headers", n_chunks=16, interval=0.5) as port:
            start = time.monotonic()
            with pytest.raises(ImageProcessingError) as exc_info:
                fetch_image_from_url(f"http://127.0.0.1:{port}/p.jpg")
            elapsed = time.monotonic() - start

        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE
        assert elapsed < settings.fetch_total_timeout + 1.0

    def test_body_drip_times_out_within_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "fetch_total_timeout", 1.0)
        monkeypatch.setattr(settings, "fetch_read_timeout", 0.5)
        monkeypatch.setattr(settings, "fetch_connect_timeout", 0.5)

        with _DripServer("body", n_chunks=16, interval=0.5) as port:
            start = time.monotonic()
            with pytest.raises(ImageProcessingError) as exc_info:
                fetch_image_from_url(f"http://127.0.0.1:{port}/p.jpg")
            elapsed = time.monotonic() - start

        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE
        assert elapsed < settings.fetch_total_timeout + 1.0


def _generate_self_signed_cert(cert_path, key_path) -> None:
    """Write a self-signed cert (CN/SAN = 127.0.0.1) + key to the given
    paths, via the `cryptography` package if installed, else an `openssl`
    subprocess. Skips the calling test if neither is available."""
    try:
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        openssl = shutil.which("openssl")
        if not openssl:
            pytest.skip("neither the cryptography package nor an openssl binary is available")
        subprocess.run(
            [
                openssl, "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_path), "-out", str(cert_path),
                "-days", "1", "-nodes", "-subj", "/CN=127.0.0.1",
                "-addext", "subjectAltName=IP:127.0.0.1",
            ],
            check=True,
            capture_output=True,
        )
        return

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


@pytest.fixture
def _self_signed_cert(tmp_path):
    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"
    _generate_self_signed_cert(cert_path, key_path)
    return cert_path, key_path


class _TLSDripServer(_DripServer):
    """Same slow-drip behaviour as `_DripServer`, over TLS -- proves the
    watchdog's dup+shutdown fix actually survives the SSL socket wrap (a
    plain `close()` on the pre-wrap socket is a no-op against the live TLS
    connection, which is exactly the gap this round's fix closes)."""

    def __init__(self, mode: str, cert_path, key_path, *, n_chunks: int = 16, interval: float = 0.5) -> None:
        super().__init__(mode, n_chunks=n_chunks, interval=interval)
        self.cert_path = cert_path
        self.key_path = key_path

    def __enter__(self) -> int:
        port = super().__enter__()
        # _DripServer.__enter__ already created + started self.server on a
        # plain socket; wrapping it in place and relying on the same
        # serve_forever thread means every accepted connection changes to a
        # true TLS one for the handler above.
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(self.cert_path), str(self.key_path))
        assert self.server is not None
        self.server.socket = ctx.wrap_socket(self.server.socket, server_side=True)
        return port


class TestDeadlineWatchdogHTTPS:
    """I1 follow-up: the watchdog must also bound an HTTPS drip, not just a
    plain-HTTP one -- SSLSocket wraps (and can detach from) the raw socket
    `_new_conn` returns, which is why the fix captures a `dup()` of it."""

    def test_header_drip_https_times_out_within_budget(
        self, monkeypatch: pytest.MonkeyPatch, _self_signed_cert
    ) -> None:
        cert_path, key_path = _self_signed_cert
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert_path))
        monkeypatch.setattr(settings, "fetch_total_timeout", 1.0)
        monkeypatch.setattr(settings, "fetch_read_timeout", 0.5)
        monkeypatch.setattr(settings, "fetch_connect_timeout", 0.5)

        with _TLSDripServer("headers", cert_path, key_path, n_chunks=16, interval=0.5) as port:
            start = time.monotonic()
            with pytest.raises(ImageProcessingError) as exc_info:
                fetch_image_from_url(f"https://127.0.0.1:{port}/p.jpg")
            elapsed = time.monotonic() - start

        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE
        assert elapsed < settings.fetch_total_timeout + 1.0

    def test_body_drip_https_times_out_within_budget(
        self, monkeypatch: pytest.MonkeyPatch, _self_signed_cert
    ) -> None:
        cert_path, key_path = _self_signed_cert
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert_path))
        monkeypatch.setattr(settings, "fetch_total_timeout", 1.0)
        monkeypatch.setattr(settings, "fetch_read_timeout", 0.5)
        monkeypatch.setattr(settings, "fetch_connect_timeout", 0.5)

        with _TLSDripServer("body", cert_path, key_path, n_chunks=16, interval=0.5) as port:
            start = time.monotonic()
            with pytest.raises(ImageProcessingError) as exc_info:
                fetch_image_from_url(f"https://127.0.0.1:{port}/p.jpg")
            elapsed = time.monotonic() - start

        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE
        assert elapsed < settings.fetch_total_timeout + 1.0


class TestDeadlineWatchdogLateSocket:
    """I1 follow-up: a socket opened *after* the watchdog has already fired
    must never escape it.

    `remaining` (the per-hop timeout budget) is computed before
    `_check_reference_url`, which itself calls `_resolve_host` -- a real DNS
    round trip with no timeout of its own. If that resolution is slow enough
    that the watchdog's `threading.Timer` fires *during* it, the connection
    `requests.get()` opens immediately afterwards is a socket the watchdog
    never had a chance to capture: the `_force_close_sockets` callback only
    ever runs once, against whatever was in `captured_sockets` at that
    moment, so a socket that shows up later is never shut down and a slow
    drip server can then stall the fetch far past `fetch_total_timeout`.
    """

    def test_slow_resolve_then_drip_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(settings, "fetch_total_timeout", 0.3)
        monkeypatch.setattr(settings, "fetch_read_timeout", 0.3)
        monkeypatch.setattr(settings, "fetch_connect_timeout", 0.3)

        def _slow_resolve(host, **kwargs):
            time.sleep(0.5)  # long enough for the watchdog to fire first
            return [ipaddress.ip_address("127.0.0.1")]

        monkeypatch.setattr(image_utils_module, "_resolve_host", _slow_resolve)

        with _DripServer("headers", n_chunks=20, interval=0.15) as port:
            start = time.monotonic()
            with pytest.raises(ImageProcessingError) as exc_info:
                fetch_image_from_url(f"http://127.0.0.1:{port}/p.jpg")
            elapsed = time.monotonic() - start

        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE
        # Generous bound: real budget is ~0.3s total-timeout + ~0.5s slow
        # resolve, plus watchdog scheduling slop -- well under the ~3s a
        # socket that escaped the watchdog would take (20 * 0.15s drip).
        assert elapsed < 2.0


class _InstantServer:
    """A real localhost HTTP server that answers immediately with a valid
    PNG -- used only to prove the socket-capture hook actually fires on a
    real connection (see TestSocketCaptureRegressionGuard), not to test any
    timeout behaviour."""

    def __enter__(self) -> int:
        body = _png_bytes()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.server.server_address[1]

    def __exit__(self, *exc_info: object) -> None:
        self.server.shutdown()
        self.server.server_close()


class TestSocketCaptureRegressionGuard:
    """If a future urllib3 upgrade stops routing through
    `HTTPConnection._new_conn` (or renames/restructures it), the watchdog's
    socket capture silently stops working -- the deadline checks in-band
    would still catch most cases, but a slow drip would once again stall
    indefinitely. Assert directly that a real connection populates the
    capture list, so that regression fails loudly instead of only showing
    up as an intermittent timing flake in TestDeadlineWatchdog.
    """

    def test_capture_list_is_populated_for_a_real_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_holder: list = []

        class _SpyTimer:
            def __init__(self, interval, function, args=(), kwargs=None) -> None:
                captured_holder.append(args[0])
                self._function = function
                self._args = args
                self._kwargs = kwargs or {}
                self.daemon = False

            def start(self) -> None:
                pass  # never fire -- this test isn't exercising the timeout

            def cancel(self) -> None:
                pass

        monkeypatch.setattr(image_utils_module.threading, "Timer", _SpyTimer)

        with _InstantServer() as port:
            image = fetch_image_from_url(f"http://127.0.0.1:{port}/p.jpg")

        assert image is not None
        assert captured_holder, "threading.Timer was never armed"
        assert len(captured_holder[0]) > 0, (
            "no socket was captured for a real connection -- "
            "urllib3.connection.HTTPConnection._new_conn may no longer be the capture hook"
        )


class TestWatchdogMessageConsistency:
    """A watchdog-forced socket close typically surfaces to `requests` as a
    `ConnectionError`, which was already mapped to SERVICE_UNAVAILABLE --
    but with a message ("Reference URL unreachable") that didn't say this
    was a timeout. Every other watchdog-triggered branch uses the same
    "timed out" message; this one should too, for consistent debugging.
    """

    def test_connection_error_after_watchdog_fired_uses_timed_out_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A fake threading.Timer whose start() fires the watchdog callback
        # immediately (synchronously), so `requests.get` raising
        # ConnectionError right after is guaranteed to see watchdog_fired
        # already set -- no real clock/sleep needed.
        class _ImmediateTimer:
            def __init__(self, interval, function, args=(), kwargs=None) -> None:
                self._function = function
                self._args = args
                self._kwargs = kwargs or {}
                self.daemon = False

            def start(self) -> None:
                self._function(*self._args, **self._kwargs)

            def cancel(self) -> None:
                pass

        monkeypatch.setattr(image_utils_module.threading, "Timer", _ImmediateTimer)

        def _raise(*args: object, **kwargs: object) -> None:
            raise requests.exceptions.ConnectionError("connection refused")

        monkeypatch.setattr(image_utils_module.requests, "get", _raise)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE
        assert exc_info.value.message == "Reference URL fetch timed out"


class TestRealResolver:
    def test_gaierror_maps_to_service_unavailable_with_redacted_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Restore the real _resolve_host (over the autouse stub) to exercise it.
        monkeypatch.setattr(image_utils_module, "_resolve_host", _REAL_RESOLVE_HOST)

        def _raise_gaierror(*args, **kwargs):
            raise socket.gaierror("Name or service not known")

        monkeypatch.setattr(image_utils_module.socket, "getaddrinfo", _raise_gaierror)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(SIGNED_URL)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE
        assert SECRET not in exc_info.value.message
        assert "?" not in exc_info.value.message


class TestResolveHostIdnaError:
    """M2: `socket.getaddrinfo` raises a raw `UnicodeError` (not
    `socket.gaierror`) for a hostname IDNA-encoding can't handle -- e.g. a
    label over 63 octets, or an empty label from a doubled dot ("a..b.com").
    Before this fix, that raw `UnicodeError` fell through `_resolve_host`
    uncaught, out through `_check_reference_url`, and was caught only by
    `fetch_image_from_url`'s generic `except Exception` -- which always maps
    to INVALID_IMAGE, even on a redirect hop, where that's a reference-
    tagged error that locks the enrolled person out over a fault in *our* fetch
    path, not their photo. `socket.gaierror` (a real DNS failure) must keep
    mapping to SERVICE_UNAVAILABLE regardless of hop.
    """

    def test_first_hop_idna_error_is_invalid_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(image_utils_module, "_resolve_host", _REAL_RESOLVE_HOST)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url("http://" + "a" * 64 + ".com/p.jpg")
        assert exc_info.value.error_code == ErrorCode.INVALID_IMAGE
        assert exc_info.value.image is None

    def test_redirect_hop_idna_error_is_service_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Hop 0 uses an IP literal (`_resolve_host` short-circuits via
        # `ipaddress.ip_address`, no real DNS round trip) so this test
        # doesn't depend on outbound DNS actually working in CI/sandboxes --
        # only the *redirect target* (hop 1) needs the real resolver's IDNA
        # failure.
        first_hop_url = "http://127.0.0.1:1/p.jpg"
        monkeypatch.setattr(image_utils_module, "_resolve_host", _REAL_RESOLVE_HOST)

        def _get(url, **kwargs):
            if url == first_hop_url:
                return _FakeResponse(
                    302, headers={"Location": "http://" + "a" * 64 + ".com/p.jpg"}
                )
            raise AssertionError(f"must not be called for {url}")

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(first_hop_url)
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE

    def test_gaierror_still_service_unavailable_regardless_of_hop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Belt-and-suspenders alongside TestRealResolver's existing coverage:
        # a genuine DNS failure (socket.gaierror) must not be swept into the
        # new UnicodeError/ValueError handling and start reading as
        # INVALID_IMAGE on hop 0.
        monkeypatch.setattr(image_utils_module, "_resolve_host", _REAL_RESOLVE_HOST)

        def _raise_gaierror(*args, **kwargs):
            raise socket.gaierror("Name or service not known")

        monkeypatch.setattr(image_utils_module.socket, "getaddrinfo", _raise_gaierror)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url("http://this-host-does-not-resolve.example/p.jpg")
        assert exc_info.value.error_code == ErrorCode.SERVICE_UNAVAILABLE


class TestNoSecretLeak:
    def test_redirect_error_message_has_no_query_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "reference_url_allowed_hosts", ["reference.example.com"])

        def _get(url, **kwargs):
            if "reference.example.com" in url:
                return _FakeResponse(302, headers={"Location": "https://evil.example.com/x.jpg?tok=leak"})
            raise AssertionError("should not follow")

        monkeypatch.setattr(image_utils_module.requests, "get", _get)

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(SIGNED_URL)
        assert SECRET not in exc_info.value.message
        assert "leak" not in exc_info.value.message
        assert "?" not in exc_info.value.message

    def test_not_allowed_error_message_has_no_query_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "reference_url_allowed_hosts", ["only-this-host.example.com"])

        with pytest.raises(ImageProcessingError) as exc_info:
            fetch_image_from_url(SIGNED_URL)
        assert SECRET not in exc_info.value.message
        assert "?" not in exc_info.value.message


class TestApiLevel:
    def test_compare_photos_with_disallowed_host_is_400_not_allowed_with_null_image(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "reference_url_allowed_hosts", ["reference.example.com"])

        response = client.post(
            "/api/v1/compare-photos",
            data={"image1": "https://evil.example.com/ref.jpg", "distance_metric": "cosine"},
            files={"image2": ("selfie.jpg", b"selfie-bytes", "image/jpeg")},
            headers=AUTH_HEADERS,
        )

        assert response.status_code == 400
        body = response.json()
        assert body["error_code"] == ErrorCode.REFERENCE_URL_NOT_ALLOWED
        assert body["image"] is None
