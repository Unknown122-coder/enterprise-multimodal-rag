"""Answer status policy.

One place decides what the caller is told, so the rules cannot drift between the
streaming and non-streaming paths. Clients branch on ``status``; the API never
returns prose without citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emrag.citation.verifier import VerificationReport
from emrag.core.types import QueryStatus


@dataclass
class StatusDecision:
    """The status, reason code and warnings for one answer."""

    status: QueryStatus
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)


def decide(
    report: VerificationReport | None,
    *,
    gate_passed: bool,
    model_said_sufficient: bool,
    had_candidates: bool,
) -> StatusDecision:
    """Resolve the final status of a query."""
    if not had_candidates:
        return StatusDecision("not_found", reason="no_matching_content")
    if not gate_passed:
        return StatusDecision("not_found", reason="retrieval_below_threshold")
    if not model_said_sufficient:
        return StatusDecision("not_found", reason="insufficient_evidence")

    if report is None or not report.supported_claims:
        return StatusDecision(
            "not_found",
            reason="all_claims_unverified",
            warnings=["all_claims_unverified"],
        )

    warnings: list[str] = []
    if report.dropped_claims:
        warnings.append(f"dropped_unverified_claims:{len(report.dropped_claims)}")
    if report.rejected_citations:
        warnings.append(f"rejected_citations:{report.rejected_citations}")

    status: QueryStatus = "partial" if report.dropped_claims else "answered"
    return StatusDecision(status, warnings=warnings)


def refusal(reason: str) -> StatusDecision:
    """Build a refusal decision for a question the system will not process."""
    return StatusDecision("refused", reason=reason, warnings=[reason])


def should_repair(report: VerificationReport, *, enabled: bool, budget_ms: int) -> bool:
    """Decide whether a repair round is worth attempting.

    Only when something was actually dropped, something survived to anchor the
    answer, and there is time left in the request deadline.
    """
    if not enabled or budget_ms < 1200:
        return False
    return bool(report.dropped_claims)
