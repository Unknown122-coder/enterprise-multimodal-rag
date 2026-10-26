"""Application configuration.

Single source of truth for runtime config. Values come from environment variables
(12-factor), YAML files hold only *tuning* defaults that are not secrets. Validation
runs at import of :func:`get_settings`, so a misconfigured deployment fails fast.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
if not CONFIG_DIR.exists():  # installed wheel: fall back to CWD/configs
    CONFIG_DIR = Path.cwd() / "configs"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or inconsistent."""


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping")
    return data


class Settings(BaseSettings):
    """Environment-driven settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # app
    app_env: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"
    log_query_text: bool = False
    retrieval_debug: bool = False
    cors_origins: str = ""

    # vector store
    vector_backend: Literal["qdrant", "pinecone"] = "qdrant"
    collection_name: str = "chunks_v1"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    pinecone_api_key: str | None = None
    pinecone_index_dense: str = "emrag-dense-v1"
    pinecone_index_sparse: str = "emrag-sparse-v1"

    # llm
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    # artifact store
    artifact_backend: Literal["local", "hf"] = "local"
    artifact_local_root: Path = Path("./var/store")
    hf_token: str | None = None
    hf_dataset_repo: str | None = None
    hf_webhook_secret: str | None = None

    # models
    embed_backend: Literal["fastembed", "gemini", "deterministic"] = "fastembed"
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_enabled: bool = True
    fastembed_cache_path: Path | None = None
    use_docling: bool = False

    # retrieval / generation
    retrieval_gate_tau: float = 0.05
    candidates_k: int = 40
    top_k: int = 8
    evidence_token_budget: int = 2500
    default_deadline_ms: int = 4500
    quote_match_threshold: int = 90

    # auth
    tenants_json: str | None = None
    tenants_file: Path | None = None
    jwt_hs256_secret: str | None = None
    jwt_jwks_url: str | None = None
    jwt_audience: str = "emrag"
    allow_dev_anon: bool = True

    # limits
    max_upload_mb: int = 50
    max_pages_full: int = 500
    max_pages_lite: int = 20
    rate_limit_query: str = "60/minute"
    rate_limit_upload: str = "10/minute"

    # optional
    vt_api_key: str | None = None
    otel_exporter_otlp_endpoint: str | None = None

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        up = v.upper()
        if up not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return up

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.vector_backend == "pinecone" and not self.pinecone_api_key:
            raise ConfigError("VECTOR_BACKEND=pinecone requires PINECONE_API_KEY")
        if self.artifact_backend == "hf" and not (self.hf_token and self.hf_dataset_repo):
            raise ConfigError("ARTIFACT_BACKEND=hf requires HF_TOKEN and HF_DATASET_REPO")
        if self.embed_backend == "gemini" and not self.gemini_api_key:
            raise ConfigError("EMBED_BACKEND=gemini requires GEMINI_API_KEY")
        if self.app_env == "prod":
            if self.allow_dev_anon:
                raise ConfigError("ALLOW_DEV_ANON must be false when APP_ENV=prod")
            if not self.tenant_definitions:
                raise ConfigError("APP_ENV=prod requires TENANTS_JSON (or TENANTS_FILE)")
            if not (self.groq_api_key or self.gemini_api_key):
                raise ConfigError("at least one of GROQ_API_KEY / GEMINI_API_KEY is required")
        if not 0 <= self.retrieval_gate_tau <= 1:
            raise ConfigError("RETRIEVAL_GATE_TAU must be within [0, 1]")
        if self.top_k > self.candidates_k:
            raise ConfigError("TOP_K must not exceed CANDIDATES_K")
        return self

    # ---- derived -------------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @functools.cached_property
    def tuning(self) -> dict[str, Any]:
        """Pipeline/retrieval tuning values from ``configs/default.yaml``."""
        return _load_yaml("default.yaml")

    @functools.cached_property
    def provider_catalogue(self) -> dict[str, Any]:
        """Provider + model catalogue from ``configs/providers.yaml``."""
        return _load_yaml("providers.yaml")

    @functools.cached_property
    def tenant_definitions(self) -> list[dict[str, Any]]:
        """Tenant records from ``TENANTS_JSON`` or ``TENANTS_FILE`` (YAML/JSON)."""
        if self.tenants_json:
            parsed = json.loads(self.tenants_json)
            if not isinstance(parsed, list):
                raise ConfigError("TENANTS_JSON must be a JSON array of tenant objects")
            return parsed
        candidates = [self.tenants_file] if self.tenants_file else [CONFIG_DIR / "tenants.yaml"]
        for path in candidates:
            if path and path.exists():
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    raise ConfigError(f"{path} must contain a list of tenant objects")
                return data
        return []

    def tuning_section(self, name: str) -> dict[str, Any]:
        """Return one section of the tuning config, raising if absent."""
        try:
            section = self.tuning[name]
        except KeyError as exc:
            raise ConfigError(f"configs/default.yaml is missing section '{name}'") from exc
        if not isinstance(section, dict):
            raise ConfigError(f"tuning section '{name}' must be a mapping")
        return section


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (validated on first call)."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings; used by tests that patch the environment."""
    get_settings.cache_clear()
