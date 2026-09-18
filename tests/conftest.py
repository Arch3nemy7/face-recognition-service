"""Shared pytest setup.

`Settings.api_token` has no default, so importing `face_recognition_service`
anywhere (directly or via `main`/`config`) instantiates `Settings()` and
raises a pydantic `ValidationError` unless `API_TOKEN` is already set --
normally via a local `.env` this repo does not ship. Set a throwaway value
here, before any test module imports the package, so the suite is
self-sufficient without adding a secret to the repo.
"""

import os

os.environ.setdefault("API_TOKEN", "test-token-for-pytest-only")
