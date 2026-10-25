"""Upload validation.

An untrusted file is checked before anything parses it:

1. size and declared type;
2. magic bytes, which must agree with both the extension and the declared MIME;
3. PDF structure via pikepdf, rejecting active content (JavaScript, launch
   actions, embedded files) that has no place in a document corpus;
4. page count against the profile limit;
5. optionally, a VirusTotal hash lookup, which is a *lookup* only and never
   uploads the file.

Nothing here executes or renders the file. The only parser that touches bytes
before validation passes is pikepdf's structural reader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import BinaryIO

import httpx

from emrag.core.errors import PayloadTooLarge, UnsupportedMedia, ValidationFailed
from emrag.observability.logging import get_logger

logger = get_logger(__name__)

ALLOWED_MIME = {
    "application/pdf": {".pdf"},
    "image/png": {".png"},
    "image/jpeg": {".jpg", ".jpeg"},
    "image/tiff": {".tif", ".tiff"},
}

_MAGIC = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)

_DANGEROUS_PDF_KEYS = ("/JavaScript", "/JS", "/Launch", "/EmbeddedFiles", "/RichMedia")
_MAX_IMAGE_PIXELS = 20_000_000
_VT_URL = "https://www.virustotal.com/api/v3/files/{sha256}"


@dataclass
class UploadCheck:
    """Result of validating an upload."""

    mime: str
    extension: str
    sha256: str
    size_bytes: int
    page_count: int
    encrypted: bool = False
    warnings: list[str] | None = None


def sniff_mime(data: bytes) -> str | None:
    """Detect a MIME type from magic bytes."""
    for signature, mime in _MAGIC:
        if data.startswith(signature):
            return mime
    try:
        import filetype

        guess = filetype.guess(data[:4096])
    except ImportError:
        return None
    return guess.mime if guess else None


def validate_upload(
    data: bytes,
    filename: str,
    *,
    declared_mime: str | None = None,
    max_bytes: int,
    max_pages: int,
) -> UploadCheck:
    """Validate an uploaded file, raising on any rejection."""
    if not data:
        raise ValidationFailed("uploaded file is empty")
    if len(data) > max_bytes:
        raise PayloadTooLarge(
            f"file is {len(data) / 1e6:.1f} MB, limit is {max_bytes / 1e6:.0f} MB"
        )

    extension = PurePosixPath(filename).suffix.lower()
    detected = sniff_mime(data)
    if detected is None or detected not in ALLOWED_MIME:
        raise UnsupportedMedia(
            f"unsupported file type (detected {detected or 'unknown'}); "
            f"accepted: {', '.join(sorted(ALLOWED_MIME))}"
        )
    if extension and extension not in ALLOWED_MIME[detected]:
        raise UnsupportedMedia(f"extension {extension} does not match the detected type {detected}")
    if declared_mime and declared_mime.split(";")[0].strip() not in (detected, ""):
        raise UnsupportedMedia(
            f"declared content type {declared_mime} does not match detected {detected}"
        )

    warnings: list[str] = []
    page_count = 1
    encrypted = False

    if detected == "application/pdf":
        page_count, encrypted, pdf_warnings = _inspect_pdf(data)
        warnings.extend(pdf_warnings)
        if page_count > max_pages:
            raise PayloadTooLarge(
                f"document has {page_count} pages, limit for this profile is {max_pages}"
            )
    else:
        _inspect_image(data)

    return UploadCheck(
        mime=detected,
        extension=extension or next(iter(ALLOWED_MIME[detected])),
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        page_count=page_count,
        encrypted=encrypted,
        warnings=warnings or None,
    )


def _inspect_pdf(data: bytes) -> tuple[int, bool, list[str]]:
    """Structurally inspect a PDF and reject active content."""
    try:
        import pikepdf
    except ImportError:
        logger.warning("upload.pikepdf_unavailable")
        return _page_count_pymupdf(data), False, ["PDF not structurally validated"]

    try:
        with pikepdf.open(io_bytes(data)) as pdf:
            page_count = len(pdf.pages)
            root = pdf.Root
            found: list[str] = []
            names = root.get("/Names")
            for key in _DANGEROUS_PDF_KEYS:
                if names is not None and key in names:
                    found.append(key)
            open_action = root.get("/OpenAction")
            if open_action is not None and "/JS" in str(open_action):
                found.append("/OpenAction+JS")
            if "/AcroForm" in root and "/XFA" in str(root.get("/AcroForm")):
                found.append("/XFA")
            if found:
                raise UnsupportedMedia(
                    f"PDF contains active content and was rejected: {', '.join(sorted(set(found)))}"
                )
            return page_count, False, []
    except UnsupportedMedia:
        raise
    except Exception as exc:
        message = str(exc)
        if "password" in message.lower() or "encrypt" in message.lower():
            raise UnsupportedMedia("encrypted PDFs are not supported") from exc
        raise ValidationFailed(f"PDF could not be parsed: {message[:200]}") from exc


def _page_count_pymupdf(data: bytes) -> int:
    from emrag.pdfcompat import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        return int(doc.page_count)


def _inspect_image(data: bytes) -> None:
    """Reject decompression bombs."""
    try:
        import io as _io

        from PIL import Image

        with Image.open(_io.BytesIO(data)) as img:
            if img.width * img.height > _MAX_IMAGE_PIXELS:
                raise PayloadTooLarge(
                    f"image is {img.width}x{img.height}, limit is {_MAX_IMAGE_PIXELS} pixels"
                )
    except PayloadTooLarge:
        raise
    except ImportError:
        return
    except Exception as exc:
        raise ValidationFailed(f"image could not be parsed: {str(exc)[:200]}") from exc


def io_bytes(data: bytes) -> BinaryIO:
    """Wrap bytes in a seekable stream for libraries that need one."""
    import io as _io

    return _io.BytesIO(data)


async def virustotal_lookup(sha256: str, api_key: str, *, timeout_s: float = 5.0) -> str | None:
    """Look up a file hash on VirusTotal.

    Returns ``"malicious"``, ``"clean"``, ``"unknown"`` or ``None`` when the
    lookup could not be performed. Never uploads the file.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(
                _VT_URL.format(sha256=sha256), headers={"x-apikey": api_key}
            )
    except httpx.HTTPError as exc:
        logger.warning("upload.vt_lookup_failed", error=str(exc)[:160])
        return None
    if response.status_code == 404:
        return "unknown"
    if response.status_code != 200:
        logger.warning("upload.vt_lookup_status", status=response.status_code)
        return None
    stats = response.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
    malicious = int(stats.get("malicious", 0)) + int(stats.get("suspicious", 0))
    return "malicious" if malicious > 0 else "clean"
