"""Image processing utilities for face recognition."""

import base64
import io
import ipaddress
import logging
import math
import socket
import threading
import time
from typing import Optional, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit

import cv2
import numpy as np
import requests
import urllib3.connection
from PIL import Image, ImageOps
from urllib3.util import parse_url as _urllib3_parse_url

from ..config import settings
from ..errors import FaceServiceError
from ..schemas.api_schemas import ErrorCode

logger = logging.getLogger(__name__)

# Status codes fetch_image_from_url follows manually (allow_redirects=False
# on every requests.get call, so this loop -- not urllib3/requests -- decides
# whether, and where, to follow).
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}

# --- Deadline watchdog plumbing --------------------------------------------
#
# requests' `timeout=(connect, read)` only bounds a *single* socket
# operation, not the whole call: a server that trickles one byte every
# 400ms with a 1s read timeout never trips that per-op timeout, yet
# `requests.get()` (header phase) or `iter_content()` (body phase) can block
# for many times `fetch_total_timeout` waiting for it -- pinning whatever
# threadpool worker is running `fetch_image_from_url`. There is no supported
# way to abort a blocking `requests.get()` call from another thread, so
# `fetch_image_from_url` arms a `threading.Timer` for the *whole* fetch and,
# if it fires, force-closes every socket opened during this thread's fetch --
# closing a socket a thread is blocked reading always unblocks it (with a
# `ConnectionError`/`OSError`), on every platform this runs on.
#
# The socket is captured via a process-wide patch of
# `urllib3.connection.HTTPConnection._new_conn` (the one hook shared by both
# HTTP and HTTPS -- `HTTPSConnection.connect` calls the same method before
# wrapping the socket in TLS). The patch itself is a global, permanent
# no-op unless the calling thread has opted in via `_connection_capture`
# (`threading.local()`, so concurrent fetches on other threads are
# unaffected); it only ever appends to a list `fetch_image_from_url` set up
# on its own thread, so it's safe under concurrency without a lock.
#
# The watchdog captures a *dup* of the socket (`sock.dup()`), not the socket
# itself: for HTTPS, `HTTPSConnection.connect()` wraps the plain socket
# `_new_conn()` returns in an `ssl.SSLSocket` immediately after, and that
# wrap can detach the original Python-level socket object from the live
# connection -- `sock.close()` on the object we captured would then be a
# no-op against the still-open TLS connection actually being read from,
# which is why an earlier version of this fix bounded plain-HTTP drips but
# not HTTPS ones. `dup()` duplicates the underlying OS file descriptor:
# `shutdown(SHUT_RDWR)` operates at the kernel socket-endpoint level, so
# calling it on the dup immediately affects the shared connection no matter
# which Python object (plain or SSL-wrapped) currently references it --
# this also interrupts a stalled TLS handshake, not just a stalled read.
# The dups are only ever shut down, never read/written by this code; they're
# `close()`d in `fetch_image_from_url`'s `finally` once the fetch is done,
# to release the duplicated file descriptors.
_connection_capture = threading.local()
_original_new_conn = urllib3.connection.HTTPConnection._new_conn


def _capturing_new_conn(self: urllib3.connection.HTTPConnection) -> socket.socket:
    sock = _original_new_conn(self)
    sockets = getattr(_connection_capture, "sockets", None)
    if sockets is not None:
        fired = getattr(_connection_capture, "fired", None)
        if fired is not None and fired.is_set():
            # The watchdog's `threading.Timer` only ever fires once, and
            # `_force_close_sockets` only shuts down whatever was already in
            # `sockets` at that moment. A connection opened *after* it fired
            # -- e.g. because this thread was still blocked in
            # `_resolve_host`'s DNS round trip when the deadline hit -- would
            # otherwise never be captured by that one-shot callback and could
            # then stall this fetch indefinitely (a slow-drip server never
            # trips `requests`' own per-op timeout). Catch that case here:
            # if the deadline has already fired by the time a *new* socket
            # shows up, shut it down immediately instead of ever letting it
            # be used for a blocking connect/read.
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            sockets.append(sock.dup())
        except OSError:
            pass
    return sock


urllib3.connection.HTTPConnection._new_conn = _capturing_new_conn


def _force_close_sockets(sockets: list, fired: threading.Event) -> None:
    """Watchdog callback: mark the deadline as fired and forcibly interrupt
    whatever this fetch's thread is currently blocked reading, by shutting
    down every captured dup (see the module-level note above for why a
    `shutdown()` on the dup, not a `close()`, is what's needed here). The
    dups themselves are closed later, in `fetch_image_from_url`'s `finally`.
    """
    fired.set()
    for dup in list(sockets):
        try:
            dup.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _redact_url(url: str) -> str:
    """Strip the query string and fragment from a URL before it's logged or
    echoed back in an error.

    The reference photo URL (image1 on /compare-photos) is often a signed or
    presigned link from the calling application with its auth embedded in
    the query string; the raw URL must never reach a client-facing error or
    a log line.
    """
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class ImageProcessingError(FaceServiceError):
    """Base exception for image processing errors."""

    def __init__(self, message: str, error_code: str, image: Optional[str] = None):
        # Which photo (reference/selfie) this error is about; set by the
        # endpoint that knows the role, not by this generic utility.
        super().__init__(message, error_code, image=image)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes (e.g. read straight off a multipart upload, with
    no base64 round trip) to a numpy array.

    Args:
        image_bytes: Raw encoded image bytes

    Returns:
        numpy array in BGR format (OpenCV format)

    Raises:
        ImageProcessingError: INVALID_IMAGE on empty input or a decode
            failure, IMAGE_TOO_LARGE if the bytes exceed settings.max_image_size
    """
    if not image_bytes:
        raise ImageProcessingError(
            "Image data is empty",
            ErrorCode.INVALID_IMAGE
        )

    # Check size limit
    if len(image_bytes) > settings.max_image_size:
        raise ImageProcessingError(
            f"Image size ({len(image_bytes)} bytes) exceeds maximum allowed "
            f"({settings.max_image_size} bytes)",
            ErrorCode.IMAGE_TOO_LARGE
        )

    try:
        # Convert bytes to numpy array
        return load_image_from_bytes(image_bytes)
    except Exception as e:
        if isinstance(e, ImageProcessingError):
            raise
        raise ImageProcessingError(
            f"Failed to decode image: {str(e)}",
            ErrorCode.INVALID_IMAGE
        ) from e


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64-encoded image string to a numpy array.

    Args:
        base64_string: Base64-encoded image string (with or without data URI prefix)

    Returns:
        numpy array in BGR format (OpenCV format)

    Raises:
        ImageProcessingError: If decoding fails or image is invalid
    """
    # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
    if "," in base64_string and base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]

    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
    except base64.binascii.Error as e:
        raise ImageProcessingError(
            f"Invalid base64 encoding: {str(e)}",
            ErrorCode.INVALID_IMAGE
        ) from e

    return decode_image_bytes(image_bytes)


def _resolve_host(
    host: str, *, is_redirect_hop: bool = False
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve `host` to every address it maps to.

    An IP-literal host is parsed directly (no DNS round trip). Otherwise
    `socket.getaddrinfo` is used, which -- unlike `socket.gethostbyname` --
    returns every A/AAAA record, not just the first one; `_check_reference_url`
    must reject the host if *any* of them is disallowed, since which one a
    later connect() picks is not something this code controls.

    Raises:
        ImageProcessingError: SERVICE_UNAVAILABLE on a resolution failure
            (`socket.gaierror`) -- a real DNS failure, always our fetch
            path's own fault regardless of hop, with a redacted, host-free
            message. `_invalid_reference_url_error(is_redirect_hop)` (INVALID_IMAGE
            on the original URL, SERVICE_UNAVAILABLE on a redirect target)
            for a host `getaddrinfo` can't even attempt to resolve: a
            `UnicodeError` (IDNA encoding failure -- an over-long label, or
            an empty one from a doubled dot, e.g. "a..b.com") or a `ValueError`
            (some platforms raise this instead for the same malformed-label
            cases). Unlike `gaierror`, these mean the host string itself was
            malformed, the same class of fault `_check_reference_url` already
            maps this way for a bad scheme or missing host.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass

    try:
        addr_infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ImageProcessingError(
            "Reference URL host could not be resolved",
            ErrorCode.SERVICE_UNAVAILABLE,
        ) from None
    except (UnicodeError, ValueError):
        raise _invalid_reference_url_error(is_redirect_hop) from None

    return [ipaddress.ip_address(info[4][0]) for info in addr_infos]


# Always blocked regardless of `reference_url_allow_private`, in addition to
# the address *classes* in `_is_always_blocked`: known cloud-metadata
# endpoints that don't fall into any of those classes on their own.
# 100.100.100.200 (Alibaba Cloud/AWS-workaround metadata) is a plain global
# unicast address to Python's `ipaddress` (not private/link-local/reserved);
# fd00:ec2::254 (AWS IMDSv2 over IPv6) is ULA space, which is only blocked
# today when `reference_url_allow_private=False` -- but metadata must never
# be reachable regardless of that setting.
_ALWAYS_BLOCKED_LITERALS = {
    ipaddress.ip_address("100.100.100.200"),
    ipaddress.ip_address("fd00:ec2::254"),
}


def _is_always_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Link-local, multicast, unspecified and reserved addresses (the
    cloud-metadata class, e.g. 169.254.169.254), plus a short explicit list
    of known metadata endpoints that don't fall into any of those classes
    (`_ALWAYS_BLOCKED_LITERALS`), are blocked no matter what
    `reference_url_allow_private` says -- no legitimate reference photo ever
    lives there."""
    return (
        ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
        or ip in _ALWAYS_BLOCKED_LITERALS
    )


def _invalid_reference_url_error(is_redirect_hop: bool) -> ImageProcessingError:
    """A bad scheme or missing host: INVALID_IMAGE on the URL the caller
    gave us directly (their input, their mistake to fix), but SERVICE_UNAVAILABLE
    on a redirect target (hop > 0) -- that's a fault in *our* fetch path (the
    far end sent us a malformed/schemeless redirect), not evidence the
    original reference photo is bad, and INVALID_IMAGE would get tagged
    `image: "reference"` and lock the enrolled person out over something
    they didn't do."""
    if is_redirect_hop:
        return ImageProcessingError(
            "Reference URL redirected to an invalid target",
            ErrorCode.SERVICE_UNAVAILABLE,
        )
    return ImageProcessingError(
        "Invalid URL format. URL must start with http:// or https://",
        ErrorCode.INVALID_IMAGE,
    )


def _check_reference_url(url: str, *, is_redirect_hop: bool = False) -> None:
    """Host/IP policy gate, run before every hop's `requests.get` (the
    initial URL and every redirect target alike).

    Every check below is derived from what `requests`/urllib3 will *actually*
    connect to (`urllib3.util.parse_url` on the prepared request URL), not
    from `urlsplit(url)` alone: a raw backslash in the netloc, or a
    userinfo-style `user@host` trick combined with one, can make `urlsplit`
    read a different host than the one urllib3 opens a socket to (e.g.
    `http://127.0.0.1:PORT\\@example.com/` -- urlsplit sees
    `example.com`, urllib3 connects to `127.0.0.1:PORT`). Any
    disagreement between the two parses, or a literal backslash in the
    netloc, fails closed rather than trusting either parser.

    Raises:
        ImageProcessingError: INVALID_IMAGE for a bad scheme or missing host
            on the original URL (SERVICE_UNAVAILABLE for the same on a
            redirect target, or for a `urlsplit(url)` parse failure such as
            `http://[::1]@b.com/` on a redirect hop -- see
            `_invalid_reference_url_error`); REFERENCE_URL_NOT_ALLOWED for a
            urlsplit/urllib3 host mismatch, a backslash in the netloc, an
            allowlist miss, or a blocked/private resolved address.
    """
    # Both `urlsplit(url)` and preparing/parsing the request can raise (a
    # malformed bracketed-IPv6 netloc fails urlsplit itself, not just
    # `.hostname`) -- both belong to the same "couldn't figure out where
    # this URL actually points" failure mode, mapped by
    # `_invalid_reference_url_error` the same way regardless of which parser
    # tripped.
    try:
        split = urlsplit(url)
        prepared_url = requests.Request("GET", url).prepare().url
        actual = _urllib3_parse_url(prepared_url)
    except Exception:
        raise _invalid_reference_url_error(is_redirect_hop) from None

    # A backslash in the netloc is parsed inconsistently by urlsplit vs.
    # urllib3/browsers; block outright rather than trust either reading of
    # a URL containing one.
    if "\\" in split.netloc:
        raise ImageProcessingError(
            f"Reference URL host is not allowed: {_redact_url(url)}",
            ErrorCode.REFERENCE_URL_NOT_ALLOWED,
        )

    actual_scheme = (actual.scheme or "").lower()
    if actual_scheme not in ("http", "https"):
        raise _invalid_reference_url_error(is_redirect_hop)

    actual_host = actual.host
    if not actual_host:
        raise _invalid_reference_url_error(is_redirect_hop)
    # urllib3's `.host` keeps the brackets on a literal IPv6 host (e.g.
    # "[::1]"); `urlsplit(...).hostname` strips them. Strip them here too --
    # otherwise every bracketed IPv6 literal would "mismatch" against
    # urlsplit's reading of the very same host and get rejected outright.
    actual_host = actual_host.strip("[]").lower().rstrip(".")

    split_host = (split.hostname or "").lower().rstrip(".")
    if actual_host != split_host:
        raise ImageProcessingError(
            f"Reference URL host is not allowed: {_redact_url(url)}",
            ErrorCode.REFERENCE_URL_NOT_ALLOWED,
        )

    host = actual_host

    allowed_hosts = settings.reference_url_allowed_hosts
    if allowed_hosts and host not in {h.lower().rstrip(".") for h in allowed_hosts}:
        raise ImageProcessingError(
            f"Reference URL host is not allowed: {_redact_url(url)}",
            ErrorCode.REFERENCE_URL_NOT_ALLOWED,
        )

    for ip in _resolve_host(host, is_redirect_hop=is_redirect_hop):
        # An IPv4-mapped IPv6 address (`::ffff:a.b.c.d`) is a different
        # `IPv6Address` object from the equivalent `IPv4Address`, so
        # membership in `_ALWAYS_BLOCKED_LITERALS` (and some of
        # `is_global`/`is_private`'s special-casing) would silently miss it
        # without unwrapping first -- e.g. `::ffff:100.100.100.200` must hit
        # the same explicit block as `100.100.100.200`. `ipv4_mapped` only
        # exists on `IPv6Address`, hence `getattr` (a plain `IPv4Address`
        # falls through unchanged).
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if _is_always_blocked(ip):
            raise ImageProcessingError(
                f"Reference URL host is not allowed: {_redact_url(url)}",
                ErrorCode.REFERENCE_URL_NOT_ALLOWED,
            )
        # `not ip.is_global` (rather than `is_private or is_loopback`) also
        # catches CGNAT (100.64.0.0/10), IPv6 ULA (fc00::/7) and 6to4/Teredo
        # transition addresses -- none of which are `is_private` in Python's
        # `ipaddress`, but none of which are reachable from the public
        # internet either.
        if not settings.reference_url_allow_private and not ip.is_global:
            raise ImageProcessingError(
                f"Reference URL host is not allowed: {_redact_url(url)}",
                ErrorCode.REFERENCE_URL_NOT_ALLOWED,
            )


def fetch_image_from_url(url: str) -> np.ndarray:
    """
    Fetch an image from a URL and convert to numpy array.

    Manual redirect loop (at most `settings.fetch_max_redirects + 1`
    requests total): every hop -- the original URL and each redirect target
    -- goes through `_check_reference_url` before it is requested, and every
    `requests.get` call passes `allow_redirects=False`, so a redirect can
    never sidestep the host/IP policy the way `allow_redirects=True` would.

    Bytes are read via `iter_content` with a running total, capped at
    `settings.max_image_size`.

    Two independent mechanisms enforce the `settings.fetch_total_timeout`
    wall-clock budget, because a single one doesn't cover every blocking
    point:
    - Between each streamed chunk and before each hop, elapsed time is
      checked directly (`time.monotonic()` against a deadline set once
      before the first hop) -- cheap, but only runs when control returns to
      this function, so it can't interrupt a call that's still blocked.
    - A `threading.Timer` armed for the whole fetch force-closes every
      socket this thread has opened (via a process-wide capture hook on
      `urllib3.connection.HTTPConnection._new_conn`, gated per-thread by
      `_connection_capture`) if it fires -- this is what actually bounds a
      slow-drip server (e.g. one byte every 400ms, comfortably under the
      read timeout): `requests.get()`'s header read and `iter_content()`'s
      body read both block in a socket `recv()` that no Python-level check
      can interrupt, but closing the socket from another thread does. A
      forced close always surfaces as `SERVICE_UNAVAILABLE` ("timed out"),
      regardless of the specific exception `requests`/urllib3 raises for it.

    Actual guarantee: every hop finishes within `fetch_total_timeout` of
    `fetch_image_from_url` being called, plus the watchdog's own scheduling
    slop (sub-second in practice) -- not the "one read timeout" overshoot
    this function used to document, which only accounted for the in-band
    check and ignored that a drip never trips it.

    Residual risks (also documented in the README): the host/IP check-then-
    connect is a TOCTOU window -- a host could resolve to an allowed address
    at `_check_reference_url` time and a blocked one by the time `requests`
    actually connects (DNS-rebinding); `_resolve_host`'s own wall-clock cost
    is bounded only by the OS resolver, not by `fetch_total_timeout`; and,
    since `iter_content` transparently decodes `Content-Encoding` (gzip/
    deflate), the running total this function checks is already a cap on
    *decoded* bytes, not wire bytes -- but a single ~64 KiB compressed read
    can still decompress to several MB in memory before that chunk is even
    handed back to this loop for the running-total check, so a highly
    compressible single chunk is a transient memory spike this cap doesn't
    prevent.

    Args:
        url: Image URL to fetch

    Returns:
        numpy array in BGR format (OpenCV format)

    Raises:
        ImageProcessingError: If fetching fails or image is invalid
    """
    if not url.startswith(('http://', 'https://')):
        raise ImageProcessingError(
            "Invalid URL format. URL must start with http:// or https://",
            ErrorCode.INVALID_IMAGE
        )

    # Prepare headers to mimic a browser request
    # Many CDN services block requests without proper User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        # No `br` (brotli): requests/urllib3 don't decode it without the
        # optional `brotli` package installed, which would hand
        # load_image_from_bytes still-compressed bytes.
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    total_timeout = settings.fetch_total_timeout
    deadline = time.monotonic() + total_timeout
    current = url
    response = None

    watchdog_fired = threading.Event()
    captured_sockets: list = []
    _connection_capture.sockets = captured_sockets
    # Consulted by `_capturing_new_conn` so a socket opened *after* the
    # watchdog has already fired (see its docstring) gets shut down the
    # moment it's created, rather than escaping the one-shot
    # `_force_close_sockets` callback entirely.
    _connection_capture.fired = watchdog_fired
    timer = threading.Timer(
        max(total_timeout, 0), _force_close_sockets, args=(captured_sockets, watchdog_fired)
    )
    timer.daemon = True
    timer.start()

    try:
        for hop_index in range(settings.fetch_max_redirects + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ImageProcessingError(
                    "Reference URL fetch timed out",
                    ErrorCode.SERVICE_UNAVAILABLE,
                )

            _check_reference_url(current, is_redirect_hop=(hop_index > 0))

            # `_check_reference_url` includes `_resolve_host`'s DNS round
            # trip, which has no timeout of its own and can eat a large
            # chunk (or all) of the remaining budget while this thread is
            # blocked and uninterruptible. `remaining` above was computed
            # *before* that call, so it must be recomputed here -- using the
            # stale value would hand `requests.get` a positive-looking
            # timeout even though the deadline has already passed.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ImageProcessingError(
                    "Reference URL fetch timed out",
                    ErrorCode.SERVICE_UNAVAILABLE,
                )

            response = requests.get(
                current,
                headers=headers,
                timeout=(
                    min(settings.fetch_connect_timeout, remaining),
                    min(settings.fetch_read_timeout, remaining),
                ),
                stream=True,
                allow_redirects=False,
            )
            if response.status_code not in _REDIRECT_STATUS_CODES:
                break

            location = response.headers.get("Location")
            getattr(response, "close", lambda: None)()
            if not location:
                raise ImageProcessingError(
                    "Reference URL redirected without a Location header",
                    ErrorCode.SERVICE_UNAVAILABLE,
                )
            try:
                # `urljoin` calls `urlsplit` internally and can raise on the
                # same malformed-bracketed-netloc inputs `_check_reference_url`
                # guards against (e.g. a `Location: http://[::1]@b.com/`) --
                # but this happens before `_check_reference_url` ever runs on
                # the joined URL (that's the *next* iteration), so it needs
                # its own mapping to the redirect-hop error code; always
                # SERVICE_UNAVAILABLE here since a joined URL is by
                # definition a redirect target, never the original request.
                current = urljoin(current, location)
            except Exception:
                raise _invalid_reference_url_error(is_redirect_hop=True) from None
        else:
            raise ImageProcessingError(
                "Too many redirects following the reference URL",
                ErrorCode.SERVICE_UNAVAILABLE,
            )

        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None
            if declared_size is not None and declared_size > settings.max_image_size:
                raise ImageProcessingError(
                    f"Image size ({declared_size} bytes) exceeds maximum allowed "
                    f"({settings.max_image_size} bytes)",
                    ErrorCode.IMAGE_TOO_LARGE
                )

        chunks: list[bytes] = []
        total_bytes = 0
        for chunk in response.iter_content(65536):
            total_bytes += len(chunk)
            if total_bytes > settings.max_image_size:
                raise ImageProcessingError(
                    f"Image size exceeds maximum allowed ({settings.max_image_size} bytes)",
                    ErrorCode.IMAGE_TOO_LARGE
                )
            if time.monotonic() > deadline:
                raise ImageProcessingError(
                    "Reference URL fetch timed out",
                    ErrorCode.SERVICE_UNAVAILABLE
                )
            chunks.append(chunk)

        # A forced close can race a hop that was just about to finish
        # cleanly: the socket closes right as the last bytes are flushed,
        # and requests/urllib3 sometimes reads that as a normal (if
        # truncated) end of body rather than raising -- so `chunks` can hold
        # partial data with no exception at all. Treat that the same as any
        # other forced close: SERVICE_UNAVAILABLE, never a photo-content
        # code (a truncated-by-us body must not look like a bad photo).
        if watchdog_fired.is_set():
            raise ImageProcessingError(
                "Reference URL fetch timed out", ErrorCode.SERVICE_UNAVAILABLE
            )

        return load_image_from_bytes(b"".join(chunks))

    # `from None` below: requests' exception messages contain the full URL,
    # query string included, so chaining them would put a signed URL's auth
    # back into any logged traceback.
    except requests.exceptions.HTTPError as e:
        if watchdog_fired.is_set():
            raise ImageProcessingError(
                "Reference URL fetch timed out", ErrorCode.SERVICE_UNAVAILABLE
            ) from None
        status_code = e.response.status_code if e.response is not None else None
        if status_code is not None and 400 <= status_code < 500:
            # The reference link itself is bad -- 404 gone, 403 forbidden,
            # 410 removed, etc. The enrolled person needs a fresh reference
            # photo/link, so this is a REFERENCE_UNAVAILABLE the caller can tag
            # "reference".
            raise ImageProcessingError(
                f"Reference URL returned {status_code}: {_redact_url(current)}",
                ErrorCode.REFERENCE_UNAVAILABLE
            ) from None
        # A 5xx (or an HTTPError with no response at all) is the far end's
        # own outage, not evidence this particular link is broken -- treat
        # it like the fetch being unreachable (see SERVICE_UNAVAILABLE below),
        # not like a bad photo.
        raise ImageProcessingError(
            f"Reference URL fetch failed ({_redact_url(current)}): HTTP {status_code}",
            ErrorCode.SERVICE_UNAVAILABLE
        ) from None
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        # Timeout, DNS failure, refused connection, TLS/SSL errors
        # (requests.exceptions.SSLError subclasses ConnectionError), and a
        # watchdog-forced socket close (surfaces as a ConnectionError too)
        # all mean the face service's own outbound path is down, not that
        # this specific reference photo is bad -- so this must never read as
        # "your reference photo is broken" (image stays null; see
        # SERVICE_UNAVAILABLE's status mapping in main.py, which routes it
        # like a real outage). Already SERVICE_UNAVAILABLE either way; the
        # watchdog case just gets the same "timed out" message as every
        # other watchdog-triggered branch, for consistent debugging.
        if watchdog_fired.is_set():
            raise ImageProcessingError(
                "Reference URL fetch timed out", ErrorCode.SERVICE_UNAVAILABLE
            ) from None
        raise ImageProcessingError(
            f"Reference URL unreachable: {_redact_url(current)}",
            ErrorCode.SERVICE_UNAVAILABLE
        ) from None
    except requests.exceptions.RequestException as e:
        if watchdog_fired.is_set():
            raise ImageProcessingError(
                "Reference URL fetch timed out", ErrorCode.SERVICE_UNAVAILABLE
            ) from None
        # Any other requests-level failure (too many redirects, malformed
        # URL post-parse, etc.) gets the same "our fetch path failed"
        # treatment as the two branches above, not a photo-content code.
        # Uses the exception type rather than str(e): requests embeds the
        # full requested URL (query string and all) in most of its
        # exception messages, which would leak a signed URL's auth right
        # back into this "dev message".
        raise ImageProcessingError(
            f"Failed to fetch image from URL ({_redact_url(current)}): {type(e).__name__}",
            ErrorCode.SERVICE_UNAVAILABLE
        ) from None
    except ImageProcessingError:
        if watchdog_fired.is_set():
            raise ImageProcessingError(
                "Reference URL fetch timed out", ErrorCode.SERVICE_UNAVAILABLE
            ) from None
        raise
    except Exception as e:
        if watchdog_fired.is_set():
            raise ImageProcessingError(
                "Reference URL fetch timed out", ErrorCode.SERVICE_UNAVAILABLE
            ) from None
        raise ImageProcessingError(
            f"Failed to load image from URL: {str(e)}",
            ErrorCode.INVALID_IMAGE
        ) from e
    finally:
        timer.cancel()
        _connection_capture.sockets = None
        _connection_capture.fired = None
        for dup in captured_sockets:
            try:
                dup.close()
            except OSError:
                pass
        if response is not None:
            getattr(response, "close", lambda: None)()


def _to_rgb(image: Image.Image) -> Image.Image:
    """RGB view of `image`; transparent areas become white instead of exposing hidden pixels."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image if image.mode == "RGB" else image.convert("RGB")


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes into a BGR array, oriented the way the photo is meant to be seen.

    - rejects formats outside settings.allowed_image_formats (UNSUPPORTED_FORMAT)
    - rejects images whose header declares more than settings.max_image_pixels
      before any pixel is decoded (IMAGE_TOO_LARGE)
    - applies the EXIF orientation tag: phone cameras store rotated pixels, and
      the face detector is not rotation-invariant
    - composites transparency onto white
    - downscales so the longer side is at most settings.max_image_side (0 = off);
      JPEGs are decoded at reduced scale directly

    Raises:
        ImageProcessingError: UNSUPPORTED_FORMAT, IMAGE_TOO_LARGE or INVALID_IMAGE
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))

        if pil_image.format and pil_image.format.lower() not in settings.allowed_image_formats:
            raise ImageProcessingError(
                f"Unsupported image format: {pil_image.format}. "
                f"Allowed formats: {settings.allowed_image_formats}",
                ErrorCode.UNSUPPORTED_FORMAT
            )

        width, height = pil_image.size
        if width * height > settings.max_image_pixels:
            raise ImageProcessingError(
                f"Image dimensions {width}x{height} exceed the maximum of "
                f"{settings.max_image_pixels} pixels",
                ErrorCode.IMAGE_TOO_LARGE
            )

        max_side = settings.max_image_side
        if max_side and max(width, height) > max_side:
            pil_image.draft("RGB", (max_side, max_side))  # JPEG/MPO: decode at a reduced scale

        try:
            pil_image = ImageOps.exif_transpose(pil_image)
        except Exception:
            # Malformed EXIF shouldn't fail the whole request -- fall back to
            # the un-rotated image rather than rejecting an otherwise-valid
            # photo. Never log image bytes or URLs here.
            logger.warning("Failed to apply EXIF orientation; using image as decoded")
        pil_image = _to_rgb(pil_image)

        if max_side and max(pil_image.size) > max_side:
            pil_image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    except ImageProcessingError:
        raise
    except Image.DecompressionBombError as e:
        raise ImageProcessingError(
            f"Image is too large to decode safely: {str(e)}",
            ErrorCode.IMAGE_TOO_LARGE
        ) from e
    except Exception as e:
        raise ImageProcessingError(
            f"Failed to load image from bytes: {str(e)}",
            ErrorCode.INVALID_IMAGE
        ) from e


def validate_image(image: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image array is suitable for face detection.

    Args:
        image: Image as numpy array

    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "Image is None"

    if not isinstance(image, np.ndarray):
        return False, "Image must be a numpy array"

    if image.size == 0:
        return False, "Image is empty"

    if len(image.shape) not in [2, 3]:
        return False, f"Invalid image shape: {image.shape}. Expected 2D or 3D array"

    # Check if image has valid dimensions
    if len(image.shape) == 3:
        height, width, channels = image.shape
        if channels not in [1, 3, 4]:
            return False, f"Invalid number of channels: {channels}. Expected 1, 3, or 4"
    else:
        height, width = image.shape

    # Check minimum dimensions
    min_size = 32
    if height < min_size or width < min_size:
        return False, f"Image too small: {width}x{height}. Minimum size: {min_size}x{min_size}"

    # Check maximum dimensions (prevent memory issues)
    max_size = 8192
    if height > max_size or width > max_size:
        return False, f"Image too large: {width}x{height}. Maximum size: {max_size}x{max_size}"

    return True, None


def _enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE and auto-gamma correction to improve face recognition accuracy
    under varying lighting conditions.

    Args:
        image: BGR image as numpy array

    Returns:
        Enhanced BGR image
    """
    # --- Auto gamma correction ---
    # Measures mean brightness and corrects toward a neutral 128
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness > 0:
        gamma = math.log(128) / math.log(mean_brightness + 1)
        gamma = max(0.5, min(gamma, 2.5))  # clamp to safe range
        lut = np.array(
            [min(255, int(((i / 255.0) ** (1.0 / gamma)) * 255)) for i in range(256)],
            dtype=np.uint8,
        )
        image = cv2.LUT(image, lut)

    # --- CLAHE on the L channel of LAB ---
    # Improves local contrast without blowing out highlights
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for face detection: validate and normalise channels only.

    Enhancement (CLAHE + auto-gamma) is no longer applied here -- it is
    decided per `settings.effective_enhance_mode` inside
    `FaceRecognitionModel.analyze`, which needs to choose between the
    original and the enhanced copy per detection attempt. `_enhance_image`
    stays in this module and is imported from there.

    Args:
        image: Image as numpy array in BGR format

    Returns:
        Preprocessed image in BGR format

    Raises:
        ImageProcessingError: If preprocessing fails
    """
    try:
        # Validate image
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            raise ImageProcessingError(error_msg, ErrorCode.INVALID_IMAGE)

        # Ensure image is in BGR format (3 channels)
        if len(image.shape) == 2:
            # Grayscale to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # BGRA to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    except ImageProcessingError:
        raise
    except Exception as e:
        raise ImageProcessingError(
            f"Failed to preprocess image: {str(e)}",
            ErrorCode.PROCESSING_ERROR
        ) from e


def encode_image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """
    Encode a numpy array image to base64 string.

    Args:
        image: Image as numpy array (BGR format)
        format: Image format for encoding (JPEG, PNG, etc.)

    Returns:
        Base64-encoded image string with data URI prefix

    Raises:
        ImageProcessingError: If encoding fails
    """
    try:
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)

        # Encode to base64
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Add data URI prefix
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{base64_string}"

    except Exception as e:
        raise ImageProcessingError(
            f"Failed to encode image to base64: {str(e)}",
            ErrorCode.PROCESSING_ERROR
        ) from e
