"""Tests must never see a developer's local .env (real token, permissive thresholds)."""

from face_recognition_service.config import Settings, settings
from tests.conftest import PINNED_ENV

# Keys that are intentionally NOT expected to equal the `Settings` code
# default. Keep this empty unless a pin is deliberately different from the
# default, and say why:
#   - MODEL_NAME: tests use buffalo_l, the smaller pack; production default
#     is antelopev2. buffalo_l ships as a flat directory the loader can read
#     without any extra setup, so it's the only pack that's locally
#     loadable without fetching and placing antelopev2's nested layout; the
#     test suite pins it for that reason, not because it's the recommended
#     model.
#   - ENHANCE_MODE: the `enhance_mode` field's own default is `None`
#     (precedence falls through to the deprecated `enhance_image`); this pin
#     sets it explicitly to "detect_fallback" purely to match the *effective*
#     default (`Settings.effective_enhance_mode` with both fields unset),
#     so the suite exercises the mode production will actually run under
#     once ENHANCE_MODE replaces ENHANCE_IMAGE in deploy.
INTENTIONAL_TEST_OVERRIDES: set[str] = {"MODEL_NAME", "ENHANCE_MODE"}

# API_TOKEN has no code default (it's a required field with no fallback), so
# there is nothing in `Settings.model_fields` to diff it against.
NO_DEFAULT_KEYS = {"API_TOKEN"}


def test_api_token_is_the_throwaway_test_token() -> None:
    assert settings.api_token == "test-token-for-pytest-only"


def test_ml_thresholds_are_pinned_by_conftest() -> None:
    """Documents the values conftest forces; see test_conftest_pins_match_code_defaults
    for the guard that these values track the `Settings` code defaults."""
    assert settings.detection_threshold == 0.5
    assert settings.min_face_quality == 0.7
    assert settings.cosine_match_threshold == 0.5
    assert settings.effective_enhance_mode == "detect_fallback"
    assert settings.model_name == "buffalo_l"


def test_conftest_pins_match_code_defaults() -> None:
    """Every ML/behaviour value conftest pins must equal the `Settings` field
    default, unless explicitly listed in INTENTIONAL_TEST_OVERRIDES.

    Without this guard, a future change to a `Settings` code default (e.g.
    MODEL_NAME becoming "antelopev2") could silently be masked by conftest's
    env pin, so tests would keep exercising the old default forever.
    """
    checked = 0
    for env_key, pinned_value in PINNED_ENV.items():
        if env_key in NO_DEFAULT_KEYS or env_key in INTENTIONAL_TEST_OVERRIDES:
            continue

        field_name = env_key.lower()
        assert field_name in Settings.model_fields, f"no Settings field for pinned env var {env_key}"
        default = Settings.model_fields[field_name].default

        if isinstance(default, bool):
            assert pinned_value.lower() == str(default).lower(), (
                f"{env_key}: conftest pins {pinned_value!r}, code default is {default!r}"
            )
        elif isinstance(default, int | float):
            assert type(default)(pinned_value) == default, (
                f"{env_key}: conftest pins {pinned_value!r}, code default is {default!r}"
            )
        else:
            assert pinned_value == default, (
                f"{env_key}: conftest pins {pinned_value!r}, code default is {default!r}"
            )
        checked += 1

    # Sanity check that the loop actually compared something, so a typo in
    # NO_DEFAULT_KEYS/INTENTIONAL_TEST_OVERRIDES can't silently empty it out.
    assert checked >= len(PINNED_ENV) - len(NO_DEFAULT_KEYS) - len(INTENTIONAL_TEST_OVERRIDES)
