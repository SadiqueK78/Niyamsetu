# backend/core/summary_check.py
# ─────────────────────────────────────────────────────────
# Numeric faithfulness check for generated summaries.
#
# A summary of a government order is only as good as its numbers: an
# amount, a date or an order number the model invented is worse than one
# it left out. This checks exactly that and nothing else — every number in
# the summary must appear, as the same token, somewhere in the source.
#
# It does NOT check wording, names or whether the summary is complete.
# A score of 1.0 means "no invented numbers", not "a correct summary".
#
# Pure: no I/O, no settings, no LLM.
# ─────────────────────────────────────────────────────────

import re

# Devanagari digits ०-९ (U+0966-U+096F) to ASCII 0-9. OCR of Marathi
# documents produces both, often in the same line, and the model may
# answer in either; a number must match whichever script it was written in.
_DEVANAGARI_DIGITS = str.maketrans({chr(0x0966 + d): str(d) for d in range(10)})

# A run of digits optionally joined by "." or "," — 3,01,71,099 and
# 17.12.2016 are one token each, not four.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def normalise_digits(text: str) -> str:
    """Devanagari digits to ASCII; everything else unchanged."""
    return (text or "").translate(_DEVANAGARI_DIGITS)


def number_tokens(text: str) -> list:
    """
    Numeric tokens in the text, in order of appearance.

    Digits are normalised first, then commas inside a token are removed,
    so the Indian grouping 3,01,71,099 and the plain 30171099 are the same
    token. Dots are kept: 17.12.2016 is a date and 1712.2016 is not.
    """
    return [token.replace(",", "") for token in _NUMBER.findall(normalise_digits(text))]


def check_faithfulness(summary_text: str, source_text: str) -> dict:
    """
    Which numbers in the summary are not in the source.

    Tokens of length 1 are ignored: list numbering, "1 year" and the like
    match almost any source and would only dilute the score. Each distinct
    summary token is checked once, so repeating a number does not count
    it twice.

    Returns:
        {"score": supported / checked (1.0 when nothing was checked),
         "checked": number of distinct tokens checked,
         "unsupported": the tokens absent from the source, in order}
    """
    source = set(number_tokens(source_text))
    checked = []
    for token in number_tokens(summary_text):
        if len(token) > 1 and token not in checked:
            checked.append(token)

    unsupported = [token for token in checked if token not in source]
    total = len(checked)
    return {
        "score":       1.0 if total == 0 else (total - len(unsupported)) / total,
        "checked":     total,
        "unsupported": unsupported,
    }
