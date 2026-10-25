"""Per-document SQLite shard.

Holds the data the query path needs but which does not belong in the vector
payload: parent sections for context expansion, and page geometry for citation
rendering. One file per document keeps writes local to ingestion and lets the API
download only the documents it actually serves.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from emrag.core.types import PageInfo, Parent
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parents (
    parent_id    TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    tenant_id    TEXT NOT NULL,
    section_path TEXT NOT NULL,
    page_start   INTEGER NOT NULL,
    page_end     INTEGER NOT NULL,
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL,
    child_ids    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parents_doc ON parents(doc_id);
CREATE TABLE IF NOT EXISTS pages (
    page           INTEGER PRIMARY KEY,
    width_pt       REAL NOT NULL,
    height_pt      REAL NOT NULL,
    char_count     INTEGER NOT NULL,
    image_coverage REAL NOT NULL,
    scanned        INTEGER NOT NULL,
    blank          INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

DOCSTORE_FILENAME = "docstore.sqlite"


class DocStore:
    """Synchronous SQLite accessor for one document's shard."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success and always close it.

        Closing matters on Windows, where an open handle blocks the enclosing
        temporary directory from being removed.
        """
        conn = sqlite3.connect(self._path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialise(self) -> None:
        """Create the schema if needed."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def write(self, parents: list[Parent], pages: list[PageInfo], meta: dict[str, str]) -> None:
        """Replace the shard contents in a single transaction."""
        self.initialise()
        with self._connect() as conn:
            conn.execute("DELETE FROM parents")
            conn.execute("DELETE FROM pages")
            conn.executemany(
                "INSERT INTO parents VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        p.parent_id,
                        p.doc_id,
                        p.tenant_id,
                        json.dumps(p.section_path),
                        p.page_start,
                        p.page_end,
                        p.text,
                        p.token_count,
                        json.dumps(p.child_ids),
                    )
                    for p in parents
                ],
            )
            conn.executemany(
                "INSERT INTO pages VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        pg.page,
                        pg.width_pt,
                        pg.height_pt,
                        pg.char_count,
                        pg.image_coverage,
                        int(pg.scanned),
                        int(pg.blank),
                    )
                    for pg in pages
                ],
            )
            conn.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", list(meta.items()))

    def get_parent(self, parent_id: str) -> Parent | None:
        """Return one parent section, or ``None``."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM parents WHERE parent_id = ?", (parent_id,)).fetchone()
        return _row_to_parent(row) if row else None

    def get_parents(self, parent_ids: list[str]) -> list[Parent]:
        """Return several parent sections in the requested order."""
        if not parent_ids:
            return []
        placeholders = ",".join("?" for _ in parent_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM parents WHERE parent_id IN ({placeholders})", parent_ids
            ).fetchall()
        by_id = {row["parent_id"]: _row_to_parent(row) for row in rows}
        return [by_id[pid] for pid in parent_ids if pid in by_id]

    def get_page(self, page: int) -> PageInfo | None:
        """Return one page's geometry, or ``None``."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pages WHERE page = ?", (page,)).fetchone()
        if not row:
            return None
        return PageInfo(
            page=row["page"],
            width_pt=row["width_pt"],
            height_pt=row["height_pt"],
            char_count=row["char_count"],
            image_coverage=row["image_coverage"],
            scanned=bool(row["scanned"]),
            blank=bool(row["blank"]),
        )

    def meta(self) -> dict[str, str]:
        """Return the shard's metadata key-value pairs."""
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    @property
    def path(self) -> Path:
        """Filesystem path of the shard."""
        return self._path


def _row_to_parent(row: sqlite3.Row) -> Parent:
    return Parent(
        parent_id=row["parent_id"],
        doc_id=row["doc_id"],
        tenant_id=row["tenant_id"],
        section_path=json.loads(row["section_path"]),
        page_start=row["page_start"],
        page_end=row["page_end"],
        text=row["text"],
        token_count=row["token_count"],
        child_ids=json.loads(row["child_ids"]),
    )


class DocStoreCache:
    """Lazily downloads document shards from the artifact store and caches them.

    The cache lives on the host's ephemeral disk; losing it only costs a re-download.
    """

    def __init__(self, store: object, cache_dir: Path) -> None:
        self._store = store
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, tenant_id: str, doc_id: str) -> DocStore | None:
        """Return the shard for a document, downloading it on first use."""
        local = self._dir / tenant_id / doc_id / DOCSTORE_FILENAME
        if local.exists():
            return DocStore(local)

        lock = self._locks.setdefault(f"{tenant_id}/{doc_id}", asyncio.Lock())
        async with lock:
            if local.exists():
                return DocStore(local)
            key = f"artifacts/{tenant_id}/{doc_id}/{DOCSTORE_FILENAME}"
            try:
                data = await self._store.get_bytes(key)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - absence is normal for lite ingests
                logger.debug("docstore.unavailable", doc_id=doc_id, error=str(exc)[:160])
                return None
            local.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(local.write_bytes, data)
            return DocStore(local)

    def evict(self, tenant_id: str, doc_id: str) -> None:
        """Drop a cached shard, after deletion or re-ingestion."""
        local = self._dir / tenant_id / doc_id / DOCSTORE_FILENAME
        if local.exists():
            local.unlink(missing_ok=True)
