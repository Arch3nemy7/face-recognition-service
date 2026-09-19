"""CORS defaults/credentials and `__main__` reload safety.

The service is deployed behind an nginx-proxy in front of a server-to-server
calling application that authenticates with a bearer token, not cookies, so:
  - CORS should be off by default (`CORS_ENABLED` defaults to false both in
    `Settings` and in docker-compose.yml).
  - Even when a deployer opts in, `Access-Control-Allow-Credentials: true`
    must never be sent: bearer tokens don't need credentialed CORS, and
    combining credentials with a wildcard `allow_origins=["*"]` (the
    documented default origins list) is unsafe.

And `uvicorn.run(..., reload=True)` must never run as the container's PID 1
(the reloader spawns a subprocess and doesn't forward signals the same way,
which breaks graceful shutdown under `docker stop`), so `__main__` always
passes `reload=False` regardless of `DEBUG`.
"""

import inspect
from pathlib import Path

from starlette.applications import Starlette
from starlette.testclient import TestClient as StarletteTestClient

import face_recognition_service.main as main_module
from face_recognition_service.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cors_enabled_defaults_to_false() -> None:
    assert Settings.model_fields["cors_enabled"].default is False


def test_cors_kwargs_never_allow_credentials() -> None:
    """`_cors_kwargs()` is the single source of CORSMiddleware config, so
    pinning `allow_credentials=False` here covers every place it's used."""
    kwargs = main_module._cors_kwargs()
    assert kwargs["allow_credentials"] is False


def test_cors_preflight_absent_by_default(client) -> None:
    """The `client` fixture builds `main_module.app` with the pinned test
    env, which does not set CORS_ENABLED, so the code default (False)
    applies and no CORSMiddleware is installed at all."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


def test_cors_enabled_never_sends_allow_credentials_header() -> None:
    """When CORS is enabled, CORSMiddleware is configured with the same
    `_cors_kwargs()` (allow_credentials=False) that main.py itself uses, so
    a preflight response must never carry
    `Access-Control-Allow-Credentials: true`. Built as a standalone
    Starlette app (rather than reimporting main with CORS_ENABLED=true)
    because `main_module.app`'s middleware stack is fixed at import time."""
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    async def homepage(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/", homepage)])
    app.add_middleware(CORSMiddleware, **main_module._cors_kwargs())

    with StarletteTestClient(app) as test_client:
        response = test_client.options(
            "/",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
    headers = {k.lower(): v for k, v in response.headers.items()}
    assert headers.get("access-control-allow-credentials") != "true"
    assert "access-control-allow-credentials" not in headers


def test_main_block_passes_reload_false() -> None:
    """Text test: `python -m face_recognition_service.main` runs as the
    container's PID 1, so it must never start uvicorn's reloader --
    `reload=settings.debug` would do that whenever DEBUG=true reaches the
    container. Reading the source (rather than only checking behaviour)
    catches a future edit that reintroduces `reload=settings.debug`."""
    source = inspect.getsource(main_module)
    assert "reload=False" in source
    assert "reload=settings.debug" not in source
