"""
Auth dependency — verifies Clerk JWTs on every protected request.

How it works:
  1. Extract the Bearer token from the Authorization header
  2. Fetch Clerk's JWKS (JSON Web Key Set) — public keys used to verify tokens
  3. Decode and verify the JWT — checks signature, expiry, and issuer
  4. Return the user_id (Clerk's `sub` claim) for use in route handlers

The JWKS endpoint is fetched asynchronously and cached for 1 hour.
Clerk rotates keys infrequently so this is safe.

Environment variables required:
  CLERK_JWKS_URL  — e.g. https://clerk.your-app.clerk.accounts.dev/.well-known/jwks.json
                    Found in Clerk dashboard → API Keys → Advanced
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import time

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# FastAPI security scheme — extracts Bearer token from Authorization header
bearer_scheme = HTTPBearer(auto_error=True)

# Simple in-memory JWKS cache — avoids fetching on every request
_jwks_cache: dict | None = None
_jwks_fetched_at: float = 0
_JWKS_CACHE_TTL = 3600  # 1 hour


async def _get_jwks() -> dict:
    """
    Fetch Clerk's public keys asynchronously, with 1-hour in-memory cache.
    Uses httpx.AsyncClient so the event loop is never blocked.
    """
    global _jwks_cache, _jwks_fetched_at

    now = time.time()
    if _jwks_cache and (now - _jwks_fetched_at) < _JWKS_CACHE_TTL:
        return _jwks_cache

    if not settings.clerk_jwks_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CLERK_JWKS_URL is not configured",
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(settings.clerk_jwks_url, timeout=10)
            response.raise_for_status()
            _jwks_cache = response.json()
            _jwks_fetched_at = now
            logger.info("Fetched Clerk JWKS")
            return _jwks_cache
    except Exception as exc:
        logger.error(f"Failed to fetch Clerk JWKS: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """
    FastAPI dependency — verifies the Clerk JWT and returns the user_id.

    Usage in routers:
        @router.get("/my-endpoint")
        async def my_endpoint(user_id: str = Depends(get_current_user)):
            ...

    Raises 401 if the token is missing, expired, or invalid.
    """
    token = credentials.credentials

    try:
        jwks = await _get_jwks()

        # Decode header to find the key ID (kid)
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        # Find the matching public key
        rsa_key = None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                rsa_key = key
                break

        if not rsa_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: signing key not found",
            )

        # Verify and decode the token
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            options={"verify_aud": False},  # Clerk tokens don't include audience
        )

        user_id: str | None = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
            )

        return user_id

    except JWTError as exc:
        logger.warning(f"JWT verification failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )