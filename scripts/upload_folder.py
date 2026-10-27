"""Upload every PDF in a folder and report how each one was handled.

    python scripts/upload_folder.py "D:\\docs" --key emk_acme_... --tag finance --tag 2025

Each file is validated and ingested by the API. Short native PDFs are ingested
inline and the script waits for them; longer or scanned ones are queued for the
Colab pipeline and reported as such. Re-running on the same folder is safe: a
document already ingested is reported as a duplicate and skipped by the server.

The key can also come from the EMRAG_API_KEY environment variable. Without a key
the request is anonymous, which works only while ALLOW_DEV_ANON=true.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

TERMINAL = {"ready", "ready_degraded", "failed"}


def upload(client: httpx.Client, url: str, path: Path, metadata: dict[str, object]) -> dict:
    """POST one file; return the parsed response or an error dict."""
    with path.open("rb") as handle:
        response = client.post(
            f"{url}/v1/documents",
            files={"file": (path.name, handle, "application/pdf")},
            data={"metadata": json.dumps(metadata)},
        )
    try:
        body = response.json()
    except ValueError:
        body = {"detail": response.text[:200]}
    body["_http"] = response.status_code
    return body


def wait_ready(client: httpx.Client, url: str, document_id: str, timeout_s: float) -> dict:
    """Poll a document until ingestion settles or the timeout passes."""
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"{url}/v1/documents/{document_id}")
        if response.status_code == 200:
            last = response.json()
            if last.get("status") in TERMINAL:
                return last
        time.sleep(1.0)
    last.setdefault("status", "timeout")
    return last


def main() -> int:
    """Upload a folder of PDFs and print a per-file summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="folder containing PDFs")
    parser.add_argument("--url", default=os.environ.get("EMRAG_API_URL", "http://localhost:8000"))
    parser.add_argument("--key", default=os.environ.get("EMRAG_API_KEY"))
    parser.add_argument("--tag", action="append", default=[], help="repeatable")
    parser.add_argument(
        "--classification", choices=["public", "internal", "confidential"], default="internal"
    )
    parser.add_argument("--group", action="append", default=[], help="ACL group, repeatable")
    parser.add_argument(
        "--wait", type=float, default=120.0, help="seconds to wait per inline ingest"
    )
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if not args.folder.is_dir():
        sys.stderr.write(f"not a folder: {args.folder}\n")
        return 2
    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    files = sorted(p for p in args.folder.glob(pattern) if p.is_file())
    if not files:
        sys.stderr.write(f"no PDFs in {args.folder}\n")
        return 2

    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    url = args.url.rstrip("/")
    metadata = {
        "tags": args.tag,
        "classification": args.classification,
        "acl_groups": args.group or ["all"],
    }

    counts = {"ready": 0, "degraded": 0, "queued": 0, "duplicate": 0, "rejected": 0, "failed": 0}
    sys.stdout.write(f"uploading {len(files)} file(s) to {url}\n\n")
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0), headers=headers) as client:
        for path in files:
            try:
                body = upload(client, url, path, metadata)
            except httpx.HTTPError as exc:
                sys.stdout.write(f"  FAILED    {path.name}: {exc}\n")
                counts["failed"] += 1
                continue

            status = body.get("_http")
            if status not in (200, 202):
                detail = body.get("detail") or body.get("title") or "?"
                sys.stdout.write(f"  REJECTED  {path.name}: {body.get('code', status)} {detail}\n")
                counts["rejected"] += 1
                continue

            doc_id = body.get("document_id", "?")
            if body.get("detail", "").startswith("document already ingested"):
                sys.stdout.write(f"  duplicate {path.name}  ({doc_id})\n")
                counts["duplicate"] += 1
                continue

            if body.get("pipeline") == "full":
                sys.stdout.write(
                    f"  QUEUED    {path.name}  ({doc_id}) -> needs the Colab pipeline: "
                    f"{body.get('detail', '')}\n"
                )
                counts["queued"] += 1
                continue

            detail = wait_ready(client, url, doc_id, args.wait)
            state = detail.get("status")
            if state == "ready":
                counts["ready"] += 1
                sys.stdout.write(
                    f"  ready     {path.name}  ({doc_id}) {detail.get('chunk_count')} chunks, "
                    f"{detail.get('page_count')} pages\n"
                )
            elif state == "ready_degraded":
                counts["degraded"] += 1
                reasons = "; ".join(
                    detail.get("quality_report", {}).get("degraded_reasons", [])[:2]
                )
                sys.stdout.write(
                    f"  ready*    {path.name}  ({doc_id}) {detail.get('chunk_count')} chunks, "
                    f"{detail.get('page_count')} pages  [{reasons}]\n"
                )
            else:
                counts["failed"] += 1
                sys.stdout.write(
                    f"  FAILED    {path.name}  ({doc_id}) status={state}: {detail.get('error')}\n"
                )

    sys.stdout.write(
        f"\nready={counts['ready']} ready*={counts['degraded']} queued={counts['queued']} "
        f"duplicate={counts['duplicate']} rejected={counts['rejected']} failed={counts['failed']}\n"
    )
    if counts["degraded"]:
        sys.stdout.write(
            "ready* means ingested without the layout model (normal locally); "
            "tables have no structure.\n"
        )
    if counts["queued"]:
        sys.stdout.write(
            "queued files are waiting in the inbox for notebooks/01_ingest_colab.ipynb.\n"
        )
    return 1 if counts["failed"] or counts["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
