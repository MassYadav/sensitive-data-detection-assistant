"""
utils/detectors.py
------------------
Regex-based sensitive data detectors, risk classification engine,
and data masking / redaction utilities.

Each detector function returns a list of matched strings found in the text.
``scan_text`` orchestrates all detectors and returns a structured ``ScanResult``.
``classify_risk`` assigns a risk level based on the counts.
``redact_text`` replaces every detected entity with ``[REDACTED - TYPE]``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Regex Patterns ──────────────────────────────────────────────────────────

# Aadhaar: 12 digits, optionally grouped as XXXX XXXX XXXX or XXXX-XXXX-XXXX
_AADHAAR_RE = re.compile(
    r"""
    (?<!\d)                     # not preceded by a digit
    [2-9]\d{3}                  # first group starts with 2-9
    [\ \-]?                     # optional separator
    \d{4}                       # second group
    [\ \-]?                     # optional separator
    \d{4}                       # third group
    (?!\d)                      # not followed by a digit
    """,
    re.VERBOSE,
)

# PAN: ABCDE1234F  (5 letters, 4 digits, 1 letter)
_PAN_RE = re.compile(
    r"""
    (?<![A-Z0-9])               # not preceded by alnum
    [A-Z]{5}                    # 5 uppercase letters
    \d{4}                       # 4 digits
    [A-Z]                       # 1 uppercase letter
    (?![A-Z0-9])                # not followed by alnum
    """,
    re.VERBOSE,
)

# Email addresses
_EMAIL_RE = re.compile(
    r"""
    [a-zA-Z0-9._%+\-]+         # local part
    @
    [a-zA-Z0-9.\-]+            # domain
    \.
    [a-zA-Z]{2,}               # TLD
    """,
    re.VERBOSE,
)

# Phone numbers — Indian (+91 / 0) and common international formats
_PHONE_RE = re.compile(
    r"""
    (?<!\d)
    (?:
        (?:\+?\d{1,3}[\s\-]?)?  # optional country code
        (?:\(?\d{2,5}\)?[\s\-]?)  # optional area code
        \d{3,4}[\s\-]?          # first group
        \d{3,4}                 # second group
    )
    (?!\d)
    """,
    re.VERBOSE,
)

# Credit card numbers — 13-19 digits, optionally grouped by 4 with space/dash
_CREDIT_CARD_RE = re.compile(
    r"""
    (?<!\d)
    (?:
        \d{4}[\ \-]?\d{4}[\ \-]?\d{4}[\ \-]?\d{1,7}  # 4-4-4-(1..7)
    )
    (?!\d)
    """,
    re.VERBOSE,
)

# API keys / secrets — long hex, base64-ish, AWS-style keys, JWTs
_API_KEY_PATTERNS: list[re.Pattern] = [
    # AWS Access Key ID  (starts with AKIA)
    re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
    # Generic long hex string (32+ hex chars — could be secret/token)
    re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])"),
    # JWT (three dot-separated base64url segments)
    re.compile(r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    # Generic high-entropy alphanumeric (40+ chars, mixed case + digits)
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9/+]{40,}={0,2}(?![A-Za-z0-9/+=])"),
]

# Password patterns — lines/fields that look like password assignments
_PASSWORD_RE = re.compile(
    r"""
    (?i)                        # case insensitive
    (?:password|passwd|pwd|secret)  # keyword
    \s*[:=]\s*                  # assignment operator
    \S+                         # the value
    """,
    re.VERBOSE,
)


# ── Detector Functions ──────────────────────────────────────────────────────

def find_aadhaar(text: str) -> List[str]:
    """Return all Aadhaar-number–like matches."""
    return _AADHAAR_RE.findall(text)


def find_pan(text: str) -> List[str]:
    """Return all PAN-number–like matches."""
    return _PAN_RE.findall(text)


def find_emails(text: str) -> List[str]:
    """Return all email addresses found in the text."""
    return _EMAIL_RE.findall(text)


def find_phones(text: str) -> List[str]:
    """Return all phone-number–like matches.

    Filters out results shorter than 7 digits to reduce false positives.
    """
    raw = _PHONE_RE.findall(text)
    return [p for p in raw if sum(c.isdigit() for c in p) >= 7]


def find_credit_cards(text: str) -> List[str]:
    """Return all credit-card–number–like matches.

    Applies a basic Luhn-checksum filter to reduce false positives.
    """
    candidates = _CREDIT_CARD_RE.findall(text)
    return [cc for cc in candidates if _passes_luhn(cc)]


def find_api_keys(text: str) -> List[str]:
    """Return all API-key/token–like matches."""
    results: list[str] = []
    for pattern in _API_KEY_PATTERNS:
        results.extend(pattern.findall(text))
    return results


def find_passwords(text: str) -> List[str]:
    """Return all lines that appear to contain hardcoded passwords."""
    return _PASSWORD_RE.findall(text)


# ── Luhn Helper ─────────────────────────────────────────────────────────────

def _passes_luhn(number_str: str) -> bool:
    """Check if a numeric string passes the Luhn algorithm."""
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# ── Scan Result ─────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    """Container for a full document scan."""

    aadhaar: List[str] = field(default_factory=list)
    pan: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    credit_cards: List[str] = field(default_factory=list)
    api_keys: List[str] = field(default_factory=list)
    passwords: List[str] = field(default_factory=list)

    @property
    def total_findings(self) -> int:
        return (
            len(self.aadhaar)
            + len(self.pan)
            + len(self.emails)
            + len(self.phones)
            + len(self.credit_cards)
            + len(self.api_keys)
            + len(self.passwords)
        )

    def counts(self) -> Dict[str, int]:
        """Return a human-friendly mapping of category → count."""
        return {
            "Aadhaar Numbers": len(self.aadhaar),
            "PAN Numbers": len(self.pan),
            "Email Addresses": len(self.emails),
            "Phone Numbers": len(self.phones),
            "Credit Card Numbers": len(self.credit_cards),
            "API Keys / Tokens": len(self.api_keys),
            "Passwords": len(self.passwords),
        }


def scan_text(text: str) -> ScanResult:
    """Run every detector on *text* and return a ``ScanResult``."""
    return ScanResult(
        aadhaar=find_aadhaar(text),
        pan=find_pan(text),
        emails=find_emails(text),
        phones=find_phones(text),
        credit_cards=find_credit_cards(text),
        api_keys=find_api_keys(text),
        passwords=find_passwords(text),
    )


# ── Risk Classification ────────────────────────────────────────────────────

def classify_risk(result: ScanResult) -> str:
    """Classify the document's risk level.

    Rules
    -----
    * **High Risk** — passwords, API keys, credit cards found, OR
      more than 5 Aadhaar/PAN combined.
    * **Medium Risk** — 1-4 Aadhaar/PAN found, or multiple emails/phones.
    * **Low Risk** — nothing sensitive detected.
    """
    has_critical = (
        len(result.passwords) > 0
        or len(result.api_keys) > 0
        or len(result.credit_cards) > 0
    )
    pii_count = len(result.aadhaar) + len(result.pan)

    if has_critical or pii_count > 5:
        return "🔴 High Risk"

    if pii_count >= 1 or len(result.emails) > 2 or len(result.phones) > 2:
        return "🟡 Medium Risk"

    return "🟢 Low Risk"


# ── Data Masking / Redaction ────────────────────────────────────────────────

# Order matters: longer / more specific patterns should be replaced first
# to prevent partial replacements.
_REDACTION_MAP: List[Tuple[str, re.Pattern]] = [
    ("API_KEY", _API_KEY_PATTERNS[2]),   # JWT first (longest)
    ("API_KEY", _API_KEY_PATTERNS[3]),   # high-entropy
    ("API_KEY", _API_KEY_PATTERNS[0]),   # AWS
    ("API_KEY", _API_KEY_PATTERNS[1]),   # hex
    ("CREDIT_CARD", _CREDIT_CARD_RE),
    ("PASSWORD", _PASSWORD_RE),
    ("AADHAAR", _AADHAAR_RE),
    ("PAN", _PAN_RE),
    ("EMAIL", _EMAIL_RE),
    ("PHONE", _PHONE_RE),
]


def redact_text(text: str, scan_result: ScanResult | None = None) -> Tuple[str, int]:
    """Replace all detected sensitive entities with ``[REDACTED - TYPE]``.

    Uses the same regex patterns as the detectors to ensure consistency.

    Args:
        text: The original document text.
        scan_result: Optional pre-computed scan result (unused, kept for API).

    Returns:
        A tuple of (redacted_text, total_redactions_applied).
    """
    redacted = text
    total = 0

    for label, pattern in _REDACTION_MAP:
        tag = f"[REDACTED - {label}]"
        new_text, count = pattern.subn(tag, redacted)
        total += count
        redacted = new_text

    return redacted, total
