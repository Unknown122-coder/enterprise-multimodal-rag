"""Artifact writing and reading.

Artifacts are the durable record of an ingestion run and the source of truth for
rebuilding a vector store. One folder per document::

    artifacts/{tenant}/{doc}/document.json        DocumentRecord
    artifacts/{tenant}/{doc}/chunks.parquet       chunk fields + dense/sparse vectors
    artifacts/{tenant}/{doc}/parents.parquet      parent sections
    artifacts/{tenant}/{doc}/elements.json        raw parser elements + table JSON
    artifacts/{tenant}/{doc}/quality_report.json  quality gate output
    artifacts/{tenant}/{doc}/docstore.sqlite      query-time shard
    artifacts/{tenant}/{doc}/pages/{n}.webp       page renders
    artifacts/{tenant}/{doc}/figures/{key}        figure crops
    artifacts/{tenant}/{doc}/source.pdf          original bytes

Parquet keeps vectors compact (30k chunks at 768 dimensions is roughly 90 MB) and
is readable from both Colab and the API host.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

from emrag.core.types import Chunk, DenseSparse, DocumentRecord, PageInfo, Parent, QualityReport
from emrag.observability.logging import get_logger
from emrag.storage.artifact_store import ArtifactStore
from emrag.storage.docstore import DOCSTORE_FILENAME, DocStore
from emrag.storage.registry import artifacts_prefix

logger = get_logger(__name__)

CHUNKS_FILE = "chunks.parquet"
PARENTS_FILE = "parents.parquet"
DOCUMENT_FILE = "document.json"
ELEMENTS_FILE = "elements.json"
QUALITY_FILE = "quality_report.json"
SOURCE_FILE = "source.pdf"


class ArtifactWriter:
    """Writes one document's artifact folder, then publishes it in one go."""

    def __init__(self, store: ArtifactStore, tenant_id: str, doc_id: str) -> None:
        self._store = store
        self._prefix = artifacts_prefix(tenant_id, doc_id)
        self._tenant_id = tenant_id
        self._doc_id = doc_id

    @property
    def prefix(self) -> str:
        """Artifact prefix for this document."""
        return self._prefix

    async def write_source(self, data: bytes, *, filename: str = SOURCE_FILE) -> str:
        """Store the original document bytes."""
        return await self._store.put_bytes(f"{self._prefix}/{filename}", data)

    async def write_document(self, record: DocumentRecord) -> str:
        """Store the document record."""
        return await self._store.put_json(
            f"{self._prefix}/{DOCUMENT_FILE}", record.model_dump(mode="json")
        )

    async def write_quality(self, report: QualityReport) -> str:
        """Store the quality report."""
        return await self._store.put_json(
            f"{self._prefix}/{QUALITY_FILE}", report.model_dump(mode="json")
        )

    async def write_elements(self, elements: list[dict[str, Any]]) -> str:
        """Store the raw parser elements, including structured table JSON."""
        return await self._store.put_json(f"{self._prefix}/{ELEMENTS_FILE}", elements)

    async def write_chunks(self, chunks: list[Chunk], vectors: list[DenseSparse]) -> str:
        """Store chunks with their vectors as Parquet."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        data = _chunks_to_parquet(chunks, vectors)
        return await self._store.put_bytes(f"{self._prefix}/{CHUNKS_FILE}", data)

    async def write_parents(self, parents: list[Parent]) -> str:
        """Store parent sections as Parquet."""
        return await self._store.put_bytes(
            f"{self._prefix}/{PARENTS_FILE}", _parents_to_parquet(parents)
        )

    async def write_docstore(
        self, parents: list[Parent], pages: list[PageInfo], meta: dict[str, str]
    ) -> str:
        """Build the SQLite shard locally and upload it."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / DOCSTORE_FILENAME
            store = DocStore(path)
            store.write(parents, pages, meta)
            data = path.read_bytes()
        return await self._store.put_bytes(f"{self._prefix}/{DOCSTORE_FILENAME}", data)

    async def write_page_renders(self, renders: dict[int, bytes]) -> int:
        """Store page previews. Returns the number written."""
        for page, data in renders.items():
            await self._store.put_bytes(f"{self._prefix}/pages/{page}.webp", data)
        return len(renders)

    async def write_figures(self, images: dict[str, bytes]) -> int:
        """Store figure crops. Returns the number written."""
        for key, data in images.items():
            await self._store.put_bytes(f"{self._prefix}/figures/{key}", data)
        return len(images)


class ArtifactReader:
    """Reads a document's artifacts, for rebuilds, migration and page serving."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    async def read_document(self, tenant_id: str, doc_id: str) -> DocumentRecord | None:
        """Read the stored document record."""
        payload = await self._store.get_json_or_none(
            f"{artifacts_prefix(tenant_id, doc_id)}/{DOCUMENT_FILE}"
        )
        return DocumentRecord.model_validate(payload) if payload else None

    async def read_chunks(
        self, tenant_id: str, doc_id: str
    ) -> tuple[list[Chunk], list[DenseSparse]]:
        """Read chunks and vectors back from Parquet."""
        data = await self._store.get_bytes(f"{artifacts_prefix(tenant_id, doc_id)}/{CHUNKS_FILE}")
        return _parquet_to_chunks(data)

    async def read_parents(self, tenant_id: str, doc_id: str) -> list[Parent]:
        """Read parent sections back from Parquet."""
        data = await self._store.get_bytes(f"{artifacts_prefix(tenant_id, doc_id)}/{PARENTS_FILE}")
        return _parquet_to_parents(data)

    async def read_page_render(self, tenant_id: str, doc_id: str, page: int) -> bytes:
        """Read one page preview."""
        return await self._store.get_bytes(
            f"{artifacts_prefix(tenant_id, doc_id)}/pages/{page}.webp"
        )

    async def read_figure(self, tenant_id: str, doc_id: str, image_ref: str) -> bytes:
        """Read one figure crop by its stored reference."""
        reference = image_ref.lstrip("/")
        if not reference.startswith("figures/"):
            reference = f"figures/{reference}"
        return await self._store.get_bytes(f"{artifacts_prefix(tenant_id, doc_id)}/{reference}")

    async def read_source(self, tenant_id: str, doc_id: str) -> bytes:
        """Read the original document bytes."""
        return await self._store.get_bytes(f"{artifacts_prefix(tenant_id, doc_id)}/{SOURCE_FILE}")

    async def list_documents(self, tenant_id: str | None = None) -> list[tuple[str, str]]:
        """List ``(tenant_id, doc_id)`` pairs that have chunk artifacts."""
        prefix = f"artifacts/{tenant_id}" if tenant_id else "artifacts"
        keys = await self._store.list_prefix(prefix)
        out: set[tuple[str, str]] = set()
        for key in keys:
            parts = key.split("/")
            if len(parts) >= 4 and parts[0] == "artifacts" and parts[3] == CHUNKS_FILE:
                out.add((parts[1], parts[2]))
        return sorted(out)


# --------------------------------------------------------------------------- #
# Parquet conversion
# --------------------------------------------------------------------------- #
_VECTOR_COLUMNS = ("dense", "sparse_indices", "sparse_values")


def _chunks_to_parquet(chunks: list[Chunk], vectors: list[DenseSparse]) -> bytes:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        row = chunk.model_dump(mode="json")
        row["dense"] = list(vector.dense)
        row["sparse_indices"] = list(vector.sparse_indices)
        row["sparse_values"] = list(vector.sparse_values)
        rows.append(row)
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buffer, index=False, compression="zstd")
    return buffer.getvalue()


def _parquet_to_chunks(data: bytes) -> tuple[list[Chunk], list[DenseSparse]]:
    import pandas as pd

    frame = pd.read_parquet(io.BytesIO(data))
    chunks: list[Chunk] = []
    vectors: list[DenseSparse] = []
    for record in frame.to_dict(orient="records"):
        # Parquet list columns come back as numpy arrays, which have no truth
        # value, so every access is an explicit None check.
        vector = DenseSparse(
            dense=[float(x) for x in _as_list(record.get("dense"))],
            sparse_indices=[int(i) for i in _as_list(record.get("sparse_indices"))],
            sparse_values=[float(v) for v in _as_list(record.get("sparse_values"))],
        )
        payload = {k: v for k, v in record.items() if k not in _VECTOR_COLUMNS}
        payload = _coerce_lists(payload)
        chunks.append(Chunk.model_validate(payload))
        vectors.append(vector)
    return chunks, vectors


def _parents_to_parquet(parents: list[Parent]) -> bytes:
    import pandas as pd

    buffer = io.BytesIO()
    pd.DataFrame([p.model_dump(mode="json") for p in parents]).to_parquet(
        buffer, index=False, compression="zstd"
    )
    return buffer.getvalue()


def _parquet_to_parents(data: bytes) -> list[Parent]:
    import pandas as pd

    frame = pd.read_parquet(io.BytesIO(data))
    return [Parent.model_validate(_coerce_lists(r)) for r in frame.to_dict(orient="records")]


def _as_list(value: Any) -> list[Any]:
    """Coerce a Parquet list cell (numpy array, list or None) into a list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _coerce_lists(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a Parquet record back into plain Python values.

    Two Parquet round-trip artefacts are undone here: list columns arrive as
    numpy arrays, and missing values in object columns arrive as ``NaN`` rather
    than ``None``, which pydantic would reject for an optional string field.
    """
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        elif (isinstance(value, float) and value != value) or value is None:  # NaN
            out[key] = None
        elif isinstance(value, str) and key == "section_path" and value.startswith("["):
            out[key] = json.loads(value)
        else:
            out[key] = value
    return out
