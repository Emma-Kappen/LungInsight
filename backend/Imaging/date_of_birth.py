"""Age from a stored date of birth. The date of birth is what is saved; age is always derived at use time."""
from __future__ import annotations

import re
from datetime import date, datetime, timezone

_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def parse_date_of_birth(text: object) -> date:
    """Parse a strict ISO ``YYYY-MM-DD`` date of birth; reject anything else, including future dates."""
    match = _ISO_DATE.fullmatch(text) if isinstance(text, str) else None
    if not match:
        raise ValueError("date_of_birth must be an ISO date (YYYY-MM-DD).")
    try:
        born = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError as exc:
        raise ValueError("date_of_birth is not a valid calendar date.") from exc
    if born > datetime.now(timezone.utc).date():
        raise ValueError("date_of_birth cannot be in the future.")
    return born


def age_from_date_of_birth(text: object, *, as_of: date | None = None) -> int:
    """Whole years between the date of birth and ``as_of`` (today, UTC, by default)."""
    born = parse_date_of_birth(text)
    today = as_of or datetime.now(timezone.utc).date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
