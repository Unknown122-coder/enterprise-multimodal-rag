"""Prompt-injection defence for retrieved content.

Defence in depth, in order of importance:

1. the prompt delimits evidence and states that it is data, never instructions;
2. the model's output is constrained to a JSON grammar with no tool access;
3. every citation is verified against the source text, so an injected
   "answer" that is not actually quoted cannot reach the user;
4. this module, which flags suspicious passages at ingestion time and strips
   invisible characters used to smuggle instructions.

Flagging is deliberately non-destructive: a flagged chunk stays searchable,
because a document that genuinely discusses prompt injection is legitimate
content. Flags drive UI warnings and exclude a chunk from parent expansion.
"""

from __future__ import annotations

import re
import unicodedata

INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_instructions", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+", re.I)),
    ("disregard_instructions", re.compile(r"disregard\s+(the\s+)?(above|previous|system)", re.I)),
    ("role_injection", re.compile(r"^\s*(system|assistant)\s*:", re.I | re.M)),
    ("prompt_override", re.compile(r"(new|updated)\s+(system\s+)?(prompt|instructions)", re.I)),
    ("exfiltration", re.compile(r"(reveal|print|repeat)\s+(your\s+)?(system\s+)?prompt", re.I)),
    ("tool_injection", re.compile(r"<\s*(tool_call|function_call|\|im_start\|)", re.I)),
    ("markdown_exfil", re.compile(r"!\[[^\]]*\]\(\s*https?://[^)]*\{", re.I)),
    ("jailbreak", re.compile(r"\b(do anything now|developer mode|DAN mode)\b", re.I)),
)

# Zero-width, bidirectional-override and soft-hyphen code points: the usual
# carriers for text a human reader cannot see but a model still reads.
_INVISIBLE_CODEPOINTS = (
    0x00AD,
    *range(0x200B, 0x2010),
    *range(0x202A, 0x202F),
    *range(0x2060, 0x2065),
    0xFEFF,
)
_INVISIBLE_RE = re.compile("[" + "".join(chr(c) for c in _INVISIBLE_CODEPOINTS) + "]")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_LONG_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{120,}={0,2}\b")


def scan_text(text: str) -> list[str]:
    """Return risk flags for a passage. Empty means nothing suspicious."""
    flags: list[str] = []
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            flags.append(f"injection:{name}")
    if _INVISIBLE_RE.search(text):
        flags.append("injection:invisible_characters")
    if _HTML_COMMENT_RE.search(text):
        flags.append("injection:html_comment")
    if _LONG_BASE64_RE.search(text):
        flags.append("injection:opaque_blob")
    return flags


def sanitise(text: str) -> str:
    """Remove the channels used to hide instructions, keeping visible content.

    Strips invisible and bidirectional control characters, HTML comments, and any
    stray evidence-block delimiters that could let document content fake the end
    of its own block.
    """
    out = unicodedata.normalize("NFKC", text)
    out = _INVISIBLE_RE.sub("", out)
    out = _HTML_COMMENT_RE.sub(" ", out)
    out = out.replace("</evidence>", "[/evidence]").replace("<evidence", "[evidence")
    return out


def is_suspicious_question(question: str) -> str | None:
    """Return a refusal reason when the *question* itself attacks the system."""
    lowered = question.lower()
    if re.search(r"(reveal|show|print|repeat)\s+(me\s+)?(your\s+)?(system\s+)?prompt", lowered):
        return "prompt_disclosure_request"
    if re.search(r"ignore\s+(all\s+)?(previous|prior|your)\s+instructions", lowered):
        return "instruction_override_attempt"
    # "another tenant's documents", and also a tenant named outright: "the demo
    # tenant's files". Possessives that mean the caller's own tenant are left
    # alone, since "my tenant's documents" is a legitimate question.
    names_tenant = re.search(
        r"\b(other|another|different)\s+tenant('s)?\b"
        r"|\b(the|from|of|for)\s+(?!my\b|our\b|this\b|same\b|current\b|own\b)\w+\s+tenant('s)?\b",
        lowered,
    )
    if names_tenant and re.search(r"\b(document|data|file)s?\b", lowered):
        return "cross_tenant_request"
    return None
