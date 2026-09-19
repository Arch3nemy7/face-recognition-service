"""Guards `.env.example` and `docker-compose.yml` against silently drifting
from `Settings`: every environment variable either names a real `Settings`
field or is explicitly listed in `DEPLOY_ONLY_KEYS` as consumed by
docker-compose/nginx/certbot itself rather than the service.

Without this test, a renamed or removed `Settings` field could leave a
stale, ignored entry in `.env.example` (silently ignored: `Settings` has
`extra="ignore"`) or in `docker-compose.yml`'s `environment:` block, and a
deployer would have no way to know the variable they set does nothing.
"""

import re
from pathlib import Path

from face_recognition_service.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"

# Keys that are real environment variables read by docker-compose.yml,
# nginx or certbot, not by `Settings`. Each is documented here so a
# reviewer can see why it's exempt from the "must be a Settings field" rule.
DEPLOY_ONLY_KEYS: set[str] = {
    # docker-compose container/restart configuration (compose file itself,
    # not read by the Python process).
    "CONTAINER_PREFIX",
    "RESTART_POLICY",
    # This stack's own internal/external Docker network names.
    "INTERNAL_NETWORK_NAME",
    "EXTERNAL_NETWORK_NAME",
    # nginx vhost / Let's Encrypt domain, substituted into nginx/nginx.conf
    # by hand (nginx does not read this .env file), not by the service.
    "DOMAIN_NAME",
    "LETSENCRYPT_EMAIL",
    # DNS-01 challenge credential used by certbot, not by the service.
    "CLOUDFLARE_API_TOKEN",
    # docker-compose `deploy.resources` limits/reservations -- consumed by
    # the Docker/Compose engine, not read inside the container.
    "FACE_RECOGNITION_CPU_LIMIT",
    "FACE_RECOGNITION_MEMORY_LIMIT",
    "FACE_RECOGNITION_CPU_RESERVATION",
    "FACE_RECOGNITION_MEMORY_RESERVATION",
}

ENV_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
COMPOSE_ENV_LINE_RE = re.compile(r"^\s*-\s+([A-Za-z_][A-Za-z0-9_]*)=")


def _keys_in_env_example() -> set[str]:
    keys: set[str] = set()
    for line in ENV_EXAMPLE_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = ENV_KEY_RE.match(stripped)
        if match:
            keys.add(match.group(1))
    return keys


def _keys_in_compose_environment() -> set[str]:
    """Every `- KEY=...` line under docker-compose's `environment:` block.

    Parsed with a regex instead of a YAML library so this test has no new
    dependency; docker-compose.yml's `environment:` entries are always
    `- KEY=value` list items, which this pattern matches regardless of which
    top-level block they're nested under. This compose file has more than
    one service, but only `face-recognition` has an `environment:` block
    today, so this is equivalent to scoping to it.
    """
    keys: set[str] = set()
    in_environment_block = False
    for raw_line in COMPOSE_PATH.read_text().splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            continue
        if stripped == "environment:":
            in_environment_block = True
            continue
        if in_environment_block:
            match = COMPOSE_ENV_LINE_RE.match(raw_line)
            if match:
                keys.add(match.group(1))
                continue
            # A line that is part of the list but not a `- KEY=` entry (or
            # any non-indented/non-list line) ends the block.
            if not raw_line.startswith(" ") or not stripped.startswith("-"):
                in_environment_block = False
    return keys


def test_env_example_has_no_stale_keys() -> None:
    settings_fields = {name.upper() for name in Settings.model_fields}
    for key in sorted(_keys_in_env_example()):
        assert key in settings_fields or key in DEPLOY_ONLY_KEYS, (
            f".env.example sets {key}, which is neither a Settings field nor in DEPLOY_ONLY_KEYS"
        )


def test_compose_environment_has_no_stale_keys() -> None:
    settings_fields = {name.upper() for name in Settings.model_fields}
    for key in sorted(_keys_in_compose_environment()):
        assert key in settings_fields or key in DEPLOY_ONLY_KEYS, (
            f"docker-compose.yml sets {key}, which is neither a Settings field nor in DEPLOY_ONLY_KEYS"
        )


def test_compose_environment_parsing_found_expected_keys() -> None:
    """Sanity check that the regex parser actually found compose's env block
    (rather than silently matching nothing and vacuously passing above)."""
    keys = _keys_in_compose_environment()
    assert "MODEL_NAME" in keys
    assert "API_TOKEN" in keys
    assert "EUCLIDEAN_MATCH_THRESHOLD" not in keys


def test_env_example_parsing_found_expected_keys() -> None:
    keys = _keys_in_env_example()
    assert "MODEL_NAME" in keys
    assert "API_TOKEN" in keys


def test_deploy_only_keys_are_actually_used() -> None:
    """Every DEPLOY_ONLY_KEYS entry should appear somewhere (.env.example or
    compose), or it's dead documentation that should be removed."""
    present = _keys_in_env_example() | _keys_in_compose_environment()
    for key in DEPLOY_ONLY_KEYS:
        assert key in present, f"DEPLOY_ONLY_KEYS lists {key}, but it doesn't appear in .env.example or docker-compose.yml"


class TestListEnvVarParsing:
    """docker-compose.yml passes REFERENCE_URL_ALLOWED_HOSTS through as
    `${REFERENCE_URL_ALLOWED_HOSTS:-[]}` -- a plain shell-style default, not
    Python -- so this only works if pydantic-settings actually parses a
    JSON-array *string* (empty or populated) out of the environment for a
    `list[str]` field. Same mechanism `CORS_ORIGINS`/`CORS_METHODS` already
    rely on; this pins it down explicitly for the new pass-through vars.
    """

    def test_empty_json_array_parses_to_empty_list(self, monkeypatch) -> None:
        monkeypatch.setenv("API_TOKEN", "x")
        monkeypatch.setenv("REFERENCE_URL_ALLOWED_HOSTS", "[]")
        settings = Settings(_env_file=None)
        assert settings.reference_url_allowed_hosts == []

    def test_populated_json_array_parses_to_list_of_hosts(self, monkeypatch) -> None:
        monkeypatch.setenv("API_TOKEN", "x")
        monkeypatch.setenv(
            "REFERENCE_URL_ALLOWED_HOSTS", '["photos.example.com", "cdn.example.com"]'
        )
        settings = Settings(_env_file=None)
        assert settings.reference_url_allowed_hosts == ["photos.example.com", "cdn.example.com"]
