"""Streamlit front end.

A thin client over the HTTP API: it never imports the pipeline, so what it can
see is exactly what any other API consumer can see. Three panes:

* **upload**, with ingestion status polling;
* **chat**, which streams the answer and then shows the verified citations;
* **citation panel**, where each reference expands to the quote, the page image
  with the cited region highlighted, and the figure crop for figure citations.

Run with ``streamlit run ui/app.py``; point it at the API with ``EMRAG_API_URL``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("EMRAG_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("EMRAG_API_KEY", "")
TIMEOUT = httpx.Timeout(90.0, connect=10.0)

STATUS_STYLE = {
    "answered": ("Answered", "#1b7f4f"),
    "partial": ("Partially answered", "#8a6400"),
    "not_found": ("Not found in the documents", "#7a2f2f"),
    "refused": ("Refused", "#7a2f2f"),
}


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def api_get(path: str, **params: Any) -> Any:
    """GET a JSON endpoint."""
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(f"{API_URL}{path}", headers=_headers(), params=params or None)
    if response.status_code >= 400:
        raise RuntimeError(_problem(response))
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    """POST a JSON body."""
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(f"{API_URL}{path}", headers=_headers(), json=payload)
    if response.status_code >= 400:
        raise RuntimeError(_problem(response))
    return response.json()


def api_upload(name: str, data: bytes, metadata: dict[str, Any]) -> Any:
    """Upload one document."""
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            f"{API_URL}/v1/documents",
            headers=_headers(),
            files={"file": (name, data, "application/pdf")},
            data={"metadata": json.dumps(metadata)},
        )
    if response.status_code >= 400:
        raise RuntimeError(_problem(response))
    return response.json()


def api_image(path: str) -> bytes | None:
    """Fetch a page render or figure crop."""
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(f"{API_URL}{path}", headers=_headers())
    return response.content if response.status_code == 200 else None


def _problem(response: httpx.Response) -> str:
    try:
        body = response.json()
        return f"{body.get('code', response.status_code)}: {body.get('detail', response.text)}"
    except Exception:  # noqa: BLE001
        return f"HTTP {response.status_code}: {response.text[:200]}"


# --------------------------------------------------------------------------- #
# panes
# --------------------------------------------------------------------------- #
def sidebar() -> dict[str, Any]:
    """Render the sidebar and return the current query options."""
    st.sidebar.title("Enterprise RAG")
    try:
        meta = api_get("/v1/meta")
        st.sidebar.caption(f"v{meta['version']} · {meta['vector_backend']} · {meta['embed_model']}")
        ready = api_get("/readyz")
        if ready["ready"]:
            st.sidebar.success("Service ready")
        else:
            failed = [k for k, v in ready["checks"].items() if not v]
            st.sidebar.warning(f"Warming up: {', '.join(failed)}")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"API unreachable at {API_URL}\n\n{exc}")

    st.sidebar.divider()
    st.sidebar.subheader("Answering")
    tier = st.sidebar.radio(
        "Model tier",
        ["fast", "quality"],
        help="Quality uses the largest free model, which has a much smaller daily budget.",
    )
    top_k = st.sidebar.slider("Passages in context", 1, 12, 8)
    rewrite = st.sidebar.selectbox("Query rewriting", ["auto", "off", "on"], index=0)
    parent = st.sidebar.checkbox("Expand to section context", value=True)
    repair = st.sidebar.checkbox("Repair unverified claims", value=True)
    deadline = st.sidebar.slider("Deadline (ms)", 1000, 10000, 4500, step=500)

    st.sidebar.divider()
    st.sidebar.subheader("Filter")
    documents = _document_options()
    chosen = st.sidebar.multiselect(
        "Documents", options=list(documents), format_func=lambda d: documents[d]
    )
    modalities = st.sidebar.multiselect(
        "Evidence type", ["text", "table", "figure", "ocr_text"], default=[]
    )

    return {
        "options": {
            "model_tier": tier,
            "top_k": top_k,
            "rewrite": rewrite,
            "parent_expansion": parent,
            "repair": repair,
            "deadline_ms": deadline,
            "include_page_images": True,
        },
        "filters": {
            k: v
            for k, v in {"doc_ids": chosen or None, "modalities": modalities or None}.items()
            if v
        },
    }


def _document_options() -> dict[str, str]:
    try:
        listing = api_get("/v1/documents", limit=100)
    except Exception:  # noqa: BLE001
        return {}
    return {
        item["document_id"]: f"{item['filename']} ({item['chunk_count']} chunks)"
        for item in listing.get("items", [])
        if item["status"] in ("ready", "ready_degraded")
    }


def upload_pane() -> None:
    """Upload documents and watch ingestion settle."""
    st.subheader("Documents")
    uploaded = st.file_uploader(
        "PDF or page image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"]
    )
    col_left, col_right = st.columns(2)
    tags = col_left.text_input("Tags (comma separated)", value="")
    classification = col_right.selectbox(
        "Classification", ["internal", "public", "confidential"], index=0
    )

    if uploaded is not None and st.button("Ingest", type="primary"):
        metadata = {
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "classification": classification,
        }
        try:
            accepted = api_upload(uploaded.name, uploaded.getvalue(), metadata)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return

        st.info(
            f"Accepted as `{accepted['document_id']}` on the "
            f"**{accepted['pipeline']}** path. {accepted.get('detail') or ''}"
        )
        if accepted["pipeline"] == "full":
            st.caption("Queued for the offline pipeline. Run the Colab notebook to process it.")
            return

        progress = st.progress(0.0, text="Ingesting")
        for attempt in range(60):
            try:
                detail = api_get(f"/v1/documents/{accepted['document_id']}")
            except Exception:  # noqa: BLE001
                time.sleep(1.0)
                continue
            if detail["status"] in ("ready", "ready_degraded", "failed"):
                progress.progress(1.0, text=detail["status"])
                break
            progress.progress(min(0.95, (attempt + 1) / 60), text=detail["status"])
            time.sleep(1.0)
        else:
            st.warning("Still processing. Reload to check again.")
            return

        if detail["status"] == "failed":
            st.error(detail.get("error") or "ingestion failed")
        else:
            st.success(
                f"{detail['chunk_count']} chunks across {detail['page_count']} pages "
                f"({detail['table_count']} tables, {detail['figure_count']} figures)."
            )
            reasons = detail.get("quality_report", {}).get("degraded_reasons") or []
            if reasons:
                with st.expander("Quality warnings"):
                    for reason in reasons:
                        st.write(f"- {reason}")

    st.divider()
    try:
        listing = api_get("/v1/documents", limit=50)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return
    if not listing["items"]:
        st.caption("No documents yet.")
        return
    for item in listing["items"]:
        columns = st.columns([5, 2, 2, 1])
        columns[0].write(f"**{item['filename']}**")
        columns[1].write(item["status"])
        columns[2].write(f"{item['chunk_count']} chunks")
        if columns[3].button("Delete", key=f"del-{item['document_id']}"):
            try:
                api_delete(item["document_id"])
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))


def api_delete(document_id: str) -> None:
    """Delete one document."""
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.delete(f"{API_URL}/v1/documents/{document_id}", headers=_headers())
    if response.status_code >= 400:
        raise RuntimeError(_problem(response))


def chat_pane(config: dict[str, Any]) -> None:
    """Ask questions and show verified citations."""
    st.subheader("Ask")
    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("result"):
                render_result(turn["result"])

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    payload: dict[str, Any] = {"question": question, "options": config["options"]}
    if config["filters"]:
        payload["filters"] = config["filters"]
    summary = _history_summary()
    if summary:
        payload["conversation"] = {"summary": summary}

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and verifying"):
            try:
                result = api_post("/v1/query", payload)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                return
        answer = (result.get("answer") or {}).get("markdown") or ""
        if answer:
            st.markdown(answer)
        render_result(result)

    st.session_state.history.append({"role": "assistant", "content": answer, "result": result})


def _history_summary() -> str:
    """Compact the last few turns, so only a summary is sent upstream."""
    turns = st.session_state.get("history", [])[-4:]
    if not turns:
        return ""
    joined = " ".join(f"{t['role']}: {t['content'][:120]}" for t in turns if t.get("content"))
    return joined[:600]


def render_result(result: dict[str, Any]) -> None:
    """Render status, citations and retrieval telemetry for one answer."""
    status = result.get("status", "not_found")
    label, colour = STATUS_STYLE.get(status, (status, "#555"))
    st.markdown(
        f"<span style='color:{colour};font-weight:600'>{label}</span>", unsafe_allow_html=True
    )

    if status in ("not_found", "refused"):
        reason = result.get("reason")
        if reason:
            st.caption(f"Reason: {reason.replace('_', ' ')}")
        nearest = result.get("nearest_sources") or []
        if nearest:
            st.caption("Closest passages, for you to check yourself:")
            for citation in nearest:
                st.write(
                    f"- {citation['filename']}, page {citation['page']} ({citation['modality']})"
                )

    citations = result.get("citations") or []
    if citations:
        st.markdown("**Sources**")
        for citation in citations:
            citation_card(citation)

    for warning in result.get("warnings") or []:
        st.caption(f"Note: {warning.replace('_', ' ')}")

    retrieval = result.get("retrieval") or {}
    generation = result.get("generation") or {}
    st.caption(
        f"{retrieval.get('used', 0)} of {retrieval.get('candidates', 0)} passages used · "
        f"{retrieval.get('strategy', '')} · "
        f"{generation.get('model', 'no model call')} · "
        f"{result.get('latency_ms', 0)} ms"
    )
    if result.get("debug"):
        with st.expander("Retrieval debug"):
            st.json(result["debug"])


def citation_card(citation: dict[str, Any]) -> None:
    """One expandable citation with its quote and source image."""
    title = (
        f"[{citation['ref']}] {citation['filename']} · page {citation['page']} "
        f"· {citation['modality']}"
    )
    with st.expander(title):
        if citation.get("section_path"):
            st.caption(" > ".join(citation["section_path"]))
        if citation.get("quote"):
            st.markdown(f"> {citation['quote']}")
        st.caption(
            f"verified: {citation.get('verified')} · "
            f"quote match {citation.get('match_score', 0):.2f} · "
            f"retrieval score {citation.get('retrieval_score', 0):.3f} · "
            f"chunk `{citation['chunk_id']}`"
        )
        if citation.get("image_url"):
            image = api_image(citation["image_url"])
            if image:
                st.image(image, caption="Cited figure", use_container_width=True)
        if citation.get("page_image_url"):
            page = api_image(citation["page_image_url"])
            if page:
                st.image(
                    page,
                    caption=f"Page {citation['page']}, cited region highlighted",
                    use_container_width=True,
                )


def usage_pane() -> None:
    """Show the quota ledger, which is the binding constraint on free tiers."""
    st.subheader("Quota")
    try:
        usage = api_get("/v1/usage")
    except Exception as exc:  # noqa: BLE001
        st.info(f"Usage requires an admin credential. {exc}")
        return
    rows = []
    for key, values in usage.get("models", {}).items():
        rows.append(
            {
                "model": key,
                "requests/min": f"{values['rpm_used']}/{values['rpm_limit'] or '-'}",
                "tokens/min": f"{values['tpm_used']}/{values['tpm_limit'] or '-'}",
                "requests/day": f"{values['rpd_used']}/{values['rpd_limit'] or '-'}",
                "tokens/day": f"{values['tpd_used']}/{values['tpd_limit'] or '-'}",
                "cooling": f"{values['cooling_s']}s" if values["cooling_s"] else "",
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No models configured.")


def main() -> None:
    """Entry point."""
    st.set_page_config(page_title="Enterprise Multimodal RAG", page_icon="📄", layout="wide")
    config = sidebar()
    chat_tab, documents_tab, quota_tab = st.tabs(["Ask", "Documents", "Quota"])
    with chat_tab:
        chat_pane(config)
    with documents_tab:
        upload_pane()
    with quota_tab:
        usage_pane()


main()
