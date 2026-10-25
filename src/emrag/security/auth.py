"""Authentication.

Two schemes behind one resolver:

* **API keys** shaped ``emk_<tenant>_<32 chars>``. Only an Argon2 hash is stored,
  in ``configs/tenants.yaml`` for development or the ``TENANTS_JSON`` environment
  variable in production.
* **JWTs**, HS256 with a shared secret or RS256 against a tenant's JWKS URL.
  Claims: ``sub``, ``tenant_id``, ``groups``, ``clearance``, ``exp``.

In both cases the tenant is taken from the credential. A request body or header
can never choose its own tenant, except for an admin key using ``X-Tenant-ID``.
"""

from __future__ import annotations

import functools
import secrets
import time
from typing import Any

from emrag.core.errors import Forbidden, Unauthenticated
from emrag.core.ids import TENANT_ID_RE
from emrag.core.types import Classification, Principal, Role
from emrag.observability.logging import get_logger
from emrag.settings import Settings

logger = get_logger(__name__)

KEY_PREFIX = "emk"
_JWKS_TTL_S = 900


class TenantConfig:
    """One tenant's authentication and policy configuration."""

    __slots__ = ("clearance", "jwt", "keys", "name", "pii_mode", "tenant_id")

    def __init__(self, raw: dict[str, Any]) -> None:
        tenant_id = str(raw.get("tenant_id", "")).strip()
        if not TENANT_ID_RE.match(tenant_id):
            raise ValueError(f"invalid tenant_id in tenant config: {tenant_id!r}")
        self.tenant_id = tenant_id
        self.name = str(raw.get("name", tenant_id))
        self.pii_mode = str(raw.get("pii_mode", "allow"))
        self.clearance: Classification = raw.get("clearance", "internal")
        self.keys: list[dict[str, Any]] = list(raw.get("keys", []))
        self.jwt: dict[str, Any] = dict(raw.get("jwt", {}) or {})


class Authenticator:
    """Resolves bearer credentials into a :class:`Principal`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tenants: dict[str, TenantConfig] = {}
        for raw in settings.tenant_definitions:
            try:
                tenant = TenantConfig(raw)
            except ValueError as exc:
                logger.error("auth.tenant_config_invalid", error=str(exc))
                continue
            self._tenants[tenant.tenant_id] = tenant
        self._jwks_cache: dict[str, tuple[float, Any]] = {}
        self._verifier = _password_hasher()

    @property
    def tenants(self) -> dict[str, TenantConfig]:
        """Configured tenants by id."""
        return self._tenants

    def tenant(self, tenant_id: str) -> TenantConfig | None:
        """Return one tenant's configuration."""
        return self._tenants.get(tenant_id)

    def pii_mode_for(self, tenant_id: str) -> str:
        """Return the tenant's PII policy, defaulting to ``allow``."""
        tenant = self._tenants.get(tenant_id)
        return tenant.pii_mode if tenant else "allow"

    async def authenticate(
        self, authorization: str | None, *, tenant_override: str | None = None
    ) -> Principal:
        """Resolve a credential, raising :class:`Unauthenticated` when invalid."""
        token = _bearer(authorization)
        if token is None:
            if self._settings.allow_dev_anon and self._settings.app_env != "prod":
                return Principal(
                    subject="dev-anon",
                    tenant_id="demo",
                    groups=["all"],
                    clearance="confidential",
                    role="admin",
                    auth_method="dev_anon",
                )
            raise Unauthenticated("missing bearer credential")

        principal = (
            self._authenticate_api_key(token)
            if token.startswith(f"{KEY_PREFIX}_")
            else await self._authenticate_jwt(token)
        )

        if tenant_override and tenant_override != principal.tenant_id:
            if not principal.is_admin():
                raise Forbidden("X-Tenant-ID requires an admin credential")
            if tenant_override not in self._tenants:
                raise Forbidden(f"unknown tenant {tenant_override}")
            principal = principal.model_copy(update={"tenant_id": tenant_override})
        return principal

    # ---- API keys ------------------------------------------------------------
    def _authenticate_api_key(self, token: str) -> Principal:
        parts = token.split("_", 2)
        if len(parts) != 3 or parts[0] != KEY_PREFIX:
            raise Unauthenticated("malformed API key")
        _, tenant_id, _secret = parts
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise Unauthenticated("unknown API key")

        for entry in tenant.keys:
            stored = str(entry.get("argon2_hash", ""))
            if not stored or "REPLACE_ME" in stored:
                continue
            if self._verify(stored, token):
                role: Role = entry.get("role", "reader")
                return Principal(
                    subject=str(entry.get("key_id", "api-key")),
                    tenant_id=tenant_id,
                    groups=list(entry.get("groups", ["all"])) or ["all"],
                    clearance=entry.get("clearance", tenant.clearance),
                    role=role,
                    auth_method="api_key",
                )
        raise Unauthenticated("unknown API key")

    def _verify(self, stored_hash: str, token: str) -> bool:
        """Constant-time credential check against an Argon2 hash."""
        if self._verifier is None:
            # Argon2 unavailable: fall back to a constant-time compare of a
            # pre-hashed value so development still works without the wheel.
            return secrets.compare_digest(stored_hash, token)
        try:
            self._verifier.verify(stored_hash, token)
        except Exception:  # noqa: BLE001 - any verification error is a mismatch
            return False
        return True

    # ---- JWT -----------------------------------------------------------------
    async def _authenticate_jwt(self, token: str) -> Principal:
        import jwt

        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
        except Exception as exc:
            raise Unauthenticated(f"malformed JWT: {exc}") from exc

        tenant_id = str(unverified.get("tenant_id", "")).strip()
        if not tenant_id:
            raise Unauthenticated("JWT is missing the tenant_id claim")
        tenant = self._tenants.get(tenant_id)
        if tenant is None and self._settings.app_env == "prod":
            raise Unauthenticated(f"unknown tenant in JWT: {tenant_id}")

        jwt_config = tenant.jwt if tenant else {}
        jwks_url = jwt_config.get("jwks_url") or self._settings.jwt_jwks_url
        secret = self._settings.jwt_hs256_secret

        try:
            if jwks_url:
                key = await self._jwks_key(jwks_url, token)
                claims = jwt.decode(
                    token,
                    key=key,
                    algorithms=["RS256", "ES256"],
                    audience=self._settings.jwt_audience,
                )
            elif secret:
                claims = jwt.decode(
                    token,
                    key=secret,
                    algorithms=["HS256"],
                    audience=self._settings.jwt_audience,
                )
            else:
                raise Unauthenticated("no JWT verification key is configured")
        except Unauthenticated:
            raise
        except Exception as exc:
            raise Unauthenticated(f"JWT verification failed: {exc}") from exc

        groups = [str(g) for g in (claims.get("groups") or ["all"])] or ["all"]
        return Principal(
            subject=str(claims.get("sub", "jwt-user")),
            tenant_id=tenant_id,
            groups=groups,
            clearance=claims.get("clearance", tenant.clearance if tenant else "internal"),
            role=claims.get("role", "reader"),
            auth_method="jwt",
        )

    async def _jwks_key(self, jwks_url: str, token: str) -> Any:
        """Fetch and cache the signing key for a token from a JWKS endpoint."""
        import httpx
        import jwt

        cached = self._jwks_cache.get(jwks_url)
        now = time.monotonic()
        if cached is None or now - cached[0] > _JWKS_TTL_S:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(jwks_url)
                response.raise_for_status()
                jwks = response.json()
            self._jwks_cache[jwks_url] = (now, jwks)
        else:
            jwks = cached[1]

        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        for entry in jwks.get("keys", []):
            if kid is None or entry.get("kid") == kid:
                return jwt.PyJWK(entry).key
        raise Unauthenticated("no matching JWKS key for this token")


def generate_api_key(tenant_id: str) -> tuple[str, str]:
    """Return a fresh ``(api_key, argon2_hash)`` pair for a tenant."""
    if not TENANT_ID_RE.match(tenant_id):
        raise ValueError(f"invalid tenant_id: {tenant_id!r}")
    token = f"{KEY_PREFIX}_{tenant_id}_{secrets.token_urlsafe(24)}"
    hasher = _password_hasher()
    if hasher is None:
        raise RuntimeError("argon2-cffi is required to generate API keys")
    return token, hasher.hash(token)


@functools.lru_cache(maxsize=1)
def _password_hasher() -> Any:
    """Return an Argon2 hasher, or ``None`` when the wheel is unavailable."""
    try:
        from argon2 import PasswordHasher

        return PasswordHasher()
    except ImportError:  # pragma: no cover - development without the wheel
        logger.warning("auth.argon2_unavailable")
        return None


def _bearer(authorization: str | None) -> str | None:
    """Extract a bearer token from an Authorization header."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None
