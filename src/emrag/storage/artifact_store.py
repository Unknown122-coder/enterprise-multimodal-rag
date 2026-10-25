"""Artifact store.

Artifacts are the source of truth: the vector store can always be rebuilt from
them. Two backends:

* ``local`` - a directory, used for development, tests and docker-compose;
* ``hf`` - a private Hugging Face Hub dataset repository, which is the free-tier
  durable store for uploads, derived artifacts, the manifest and job records.

Layout (identical in both backends)::

    inbox/{tenant}/{doc_id}.pdf              uploaded bytes awaiting ingestion
    artifacts/{tenant}/{doc_id}/...          chunks.parquet, parents.parquet, pages/, figures/
    manifest/{tenant}/{doc_id}.json          DocumentRecord
    jobs/{job_id}.json                       Job
    state/usage.json                         quota ledger snapshot
"""

from __future__ import annotations

import abc
import asyncio
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from emrag.core.errors import NotFound, StoreUnavailable
from emrag.observability.logging import get_logger

logger = get_logger(__name__)


def _validate_key(key: str) -> str:
    """Reject absolute paths and traversal, which could escape the store root."""
    pure = PurePosixPath(key)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise ValueError(f"unsafe artifact key: {key!r}")
    if not pure.parts:
        raise ValueError("artifact key must not be empty")
    return str(pure)


class ArtifactStore(abc.ABC):
    """Key-value store for bytes, keyed by POSIX-style paths."""

    backend: str

    @abc.abstractmethod
    async def put_bytes(self, key: str, data: bytes) -> str:
        """Write ``data`` at ``key`` and return a URI for it."""

    @abc.abstractmethod
    async def get_bytes(self, key: str) -> bytes:
        """Read the bytes at ``key``, raising :class:`NotFound` if absent."""

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True when ``key`` is present."""

    @abc.abstractmethod
    async def list_prefix(self, prefix: str) -> list[str]:
        """List keys under ``prefix``."""

    @abc.abstractmethod
    async def delete(self, key: str) -> None:
        """Delete one key; absent keys are not an error."""

    @abc.abstractmethod
    async def delete_prefix(self, prefix: str) -> int:
        """Delete every key under ``prefix`` and return the count."""

    @abc.abstractmethod
    def uri(self, key: str) -> str:
        """Return a stable URI for ``key`` without touching the network."""

    async def put_json(self, key: str, payload: Any) -> str:
        """Write a JSON document."""
        return await self.put_bytes(
            key, json.dumps(payload, ensure_ascii=False, default=str, indent=2).encode("utf-8")
        )

    async def get_json(self, key: str) -> Any:
        """Read a JSON document."""
        raw = await self.get_bytes(key)
        return json.loads(raw.decode("utf-8"))

    async def get_json_or_none(self, key: str) -> Any | None:
        """Read a JSON document, returning ``None`` when it does not exist."""
        try:
            return await self.get_json(key)
        except NotFound:
            return None


class LocalArtifactStore(ArtifactStore):
    """Filesystem-backed artifact store."""

    backend = "local"

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self._root / _validate_key(key)).resolve()
        if not str(path).startswith(str(self._root)):
            raise ValueError(f"artifact key escapes the store root: {key!r}")
        return path

    async def put_bytes(self, key: str, data: bytes) -> str:
        """Write bytes atomically (temp file plus rename)."""

        def _write() -> None:
            path = self._path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)

        await asyncio.to_thread(_write)
        return self.uri(key)

    async def get_bytes(self, key: str) -> bytes:
        """Read bytes, raising :class:`NotFound` when missing."""

        def _read() -> bytes:
            path = self._path(key)
            if not path.exists():
                raise NotFound(f"artifact not found: {key}")
            return path.read_bytes()

        return await asyncio.to_thread(_read)

    async def exists(self, key: str) -> bool:
        """Return True when the file exists."""
        return await asyncio.to_thread(lambda: self._path(key).exists())

    async def list_prefix(self, prefix: str) -> list[str]:
        """List files under ``prefix`` relative to the store root."""

        def _list() -> list[str]:
            base = (self._root / PurePosixPath(prefix)).resolve()
            if not str(base).startswith(str(self._root)) or not base.exists():
                return []
            if base.is_file():
                return [prefix]
            return sorted(
                str(p.relative_to(self._root)).replace("\\", "/")
                for p in base.rglob("*")
                if p.is_file() and not p.name.endswith(".tmp")
            )

        return await asyncio.to_thread(_list)

    async def delete(self, key: str) -> None:
        """Delete a file if present."""

        def _delete() -> None:
            path = self._path(key)
            if path.exists():
                path.unlink()

        await asyncio.to_thread(_delete)

    async def delete_prefix(self, prefix: str) -> int:
        """Remove a subtree and return the number of files deleted."""
        keys = await self.list_prefix(prefix)

        def _remove() -> None:
            base = (self._root / PurePosixPath(prefix)).resolve()
            if base.is_dir():
                shutil.rmtree(base, ignore_errors=True)
            elif base.is_file():
                base.unlink()

        await asyncio.to_thread(_remove)
        return len(keys)

    def uri(self, key: str) -> str:
        """Return a ``file://`` URI."""
        return self._path(key).as_uri()

    @property
    def root(self) -> Path:
        """The store root directory."""
        return self._root


class HFArtifactStore(ArtifactStore):
    """Hugging Face Hub dataset repository backed store.

    Every write is a commit, which costs 1-2 seconds. Callers batch writes where
    possible (the ingestion pipeline uploads a whole document folder at once).
    """

    backend = "hf"

    def __init__(self, repo_id: str, token: str, *, cache_dir: Path | None = None) -> None:
        from huggingface_hub import HfApi

        self._repo_id = repo_id
        self._token = token
        self._api = HfApi(token=token)
        self._cache = Path(cache_dir) if cache_dir else Path(".cache/hf-artifacts")
        self._api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    async def put_bytes(self, key: str, data: bytes) -> str:
        """Upload a single file as one commit."""
        key = _validate_key(key)

        def _upload() -> None:
            self._api.upload_file(
                path_or_fileobj=data,
                path_in_repo=key,
                repo_id=self._repo_id,
                repo_type="dataset",
                commit_message=f"put {key}",
            )

        try:
            await asyncio.to_thread(_upload)
        except Exception as exc:
            raise StoreUnavailable(f"hf upload failed for {key}: {exc}") from exc
        return self.uri(key)

    async def put_folder(self, local_dir: Path, repo_prefix: str) -> str:
        """Upload a whole directory as one commit (the ingestion checkpoint)."""
        prefix = _validate_key(repo_prefix)

        def _upload() -> None:
            self._api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=prefix,
                repo_id=self._repo_id,
                repo_type="dataset",
                commit_message=f"ingest checkpoint {prefix}",
            )

        try:
            await asyncio.to_thread(_upload)
        except Exception as exc:
            raise StoreUnavailable(f"hf folder upload failed for {prefix}: {exc}") from exc
        return self.uri(prefix)

    async def get_bytes(self, key: str) -> bytes:
        """Download a file through the Hub cache."""
        key = _validate_key(key)

        def _download() -> bytes:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            try:
                path = hf_hub_download(
                    repo_id=self._repo_id,
                    filename=key,
                    repo_type="dataset",
                    token=self._token,
                    cache_dir=str(self._cache),
                )
            except EntryNotFoundError as exc:
                raise NotFound(f"artifact not found: {key}") from exc
            return Path(path).read_bytes()

        try:
            return await asyncio.to_thread(_download)
        except NotFound:
            raise
        except Exception as exc:
            raise StoreUnavailable(f"hf download failed for {key}: {exc}") from exc

    async def exists(self, key: str) -> bool:
        """Check presence via the repo file listing."""
        key = _validate_key(key)
        keys = await self.list_prefix(str(PurePosixPath(key).parent))
        return key in keys

    async def list_prefix(self, prefix: str) -> list[str]:
        """List repository files under ``prefix``."""

        def _list() -> list[str]:
            files = self._api.list_repo_files(repo_id=self._repo_id, repo_type="dataset")
            norm = str(PurePosixPath(prefix))
            return sorted(f for f in files if f == norm or f.startswith(norm.rstrip("/") + "/"))

        try:
            return await asyncio.to_thread(_list)
        except Exception as exc:
            raise StoreUnavailable(f"hf list failed for {prefix}: {exc}") from exc

    async def delete(self, key: str) -> None:
        """Delete one repository file."""
        key = _validate_key(key)

        def _delete() -> None:
            from huggingface_hub.errors import EntryNotFoundError

            try:
                self._api.delete_file(
                    path_in_repo=key,
                    repo_id=self._repo_id,
                    repo_type="dataset",
                    commit_message=f"delete {key}",
                )
            except EntryNotFoundError:
                return

        await asyncio.to_thread(_delete)

    async def delete_prefix(self, prefix: str) -> int:
        """Delete a repository subtree as one commit."""
        keys = await self.list_prefix(prefix)
        if not keys:
            return 0

        def _delete() -> None:
            self._api.delete_folder(
                path_in_repo=str(PurePosixPath(prefix)),
                repo_id=self._repo_id,
                repo_type="dataset",
                commit_message=f"delete {prefix}",
            )

        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:
            raise StoreUnavailable(f"hf delete failed for {prefix}: {exc}") from exc
        return len(keys)

    def uri(self, key: str) -> str:
        """Return an ``hf://datasets/...`` URI."""
        return f"hf://datasets/{self._repo_id}/{_validate_key(key)}"


def build_artifact_store(settings: Any = None) -> ArtifactStore:
    """Return the artifact store selected by ``ARTIFACT_BACKEND``."""
    from emrag.settings import get_settings

    cfg = settings or get_settings()
    if cfg.artifact_backend == "hf":
        if not (cfg.hf_dataset_repo and cfg.hf_token):
            raise StoreUnavailable("ARTIFACT_BACKEND=hf requires HF_DATASET_REPO and HF_TOKEN")
        return HFArtifactStore(cfg.hf_dataset_repo, cfg.hf_token)
    return LocalArtifactStore(cfg.artifact_local_root)
