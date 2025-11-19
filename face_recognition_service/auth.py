"""Authentication and authorization utilities for the face recognition service."""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

# HTTP Bearer token security scheme
security = HTTPBearer(
    scheme_name="Bearer Token",
    description="API Bearer Token for endpoint authorization",
    auto_error=False,
)


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]
) -> None:
    """
    Verify the API bearer token from the Authorization header.

    This dependency validates that the request contains a valid bearer token
    matching the configured API_TOKEN from environment variables.

    Args:
        credentials: HTTP Authorization credentials (Bearer token)

    Raises:
        HTTPException: 401 if token is missing or invalid

    Example:
        Authorization: Bearer your-secure-token-here
    """
    # Check if credentials are provided
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract the token from credentials
    provided_token = credentials.credentials

    # Use constant-time comparison to prevent timing attacks
    expected_token = settings.api_token

    if not secrets.compare_digest(provided_token, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Token is valid - authentication successful
    return None
