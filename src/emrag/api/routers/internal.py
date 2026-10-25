"""Internal endpoints.

The Hugging Face dataset repository fires a webhook when artifacts change, which
lets the API refresh its registry cache without polling. The signature is
verified with the shared secret; an unsigned or mismatched request is rejected.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Annotated

from fastapi import APIRouter, Header, Request, Response

from emrag.api.deps import StateDep
from emrag.core.errors import Forbidden
from emrag.observability.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/internal", tags=["internal"], include_in_schema=False)


@router.post("/hf-webhook", status_code=204, summary="Artifact store change notification")
async def hf_webhook(
    request: Request,
    state: StateDep,
    x_webhook_secret: Annotated[str | None, Header()] = None,
    x_webhook_signature: Annotated[str | None, Header()] = None,
) -> Response:
    """Refresh the document registry after an artifact store change."""
    secret = state.settings.hf_webhook_secret
    if not secret:
        raise Forbidden("webhook is not configured")

    body = await request.body()
    if x_webhook_signature:
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        supplied = x_webhook_signature.removeprefix("sha256=").strip()
        if not hmac.compare_digest(expected, supplied):
            raise Forbidden("invalid webhook signature")
    elif not (x_webhook_secret and hmac.compare_digest(secret, x_webhook_secret)):
        raise Forbidden("invalid webhook secret")

    count = await state.registry.refresh()
    logger.info("internal.registry_refreshed", documents=count)
    return Response(status_code=204)
