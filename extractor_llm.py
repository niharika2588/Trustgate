"""
OPTIONAL LLM fallback for TrustGate extraction.

Only used when the environment variable ANTHROPIC_API_KEY is set. It fills fields that the
deterministic rules could not read; it NEVER overwrites a value that a rule already extracted and it
never invents values - whatever it cannot see stays missing. Without the env var nothing is sent
anywhere and this module does nothing.
"""
import json
import os

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

PROMPT = """You are extracting structured data from a single {doc_type} document (plain text from a
file upload). Reply with JSON only, no prose.

Rules:
- Copy values exactly as printed. Strip currency symbols and thousands separators from numbers.
- Dates must be YYYY-MM-DD. Currency must be an ISO 4217 code (USD, EUR, ...).
- For a field you cannot see, put null. NEVER guess, never calculate, never copy a value from another
  field. Do not "fix" anything: mismatches between numbers must be reported as printed.
- product_id and hs_code are strings.

Fields to fill (null when not visible):
{fields}

Document text:
---
{text}
---
"""


def available():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def fill_missing(doc_type, fields, text, max_chars=24000):
    """Returns {field: value} for fields that are still missing, or {} when the LLM pass is off/unavailable."""
    wanted = [f for f, v in fields.items() if v in (None, "")]
    if not wanted or not available():
        return {}
    try:
        import anthropic
    except Exception:
        return {}
    body = PROMPT.format(doc_type=doc_type,
                         fields="\n".join("- " + f for f in wanted),
                         text=text[:max_chars])
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(model=MODEL, max_tokens=1500,
                                     messages=[{"role": "user", "content": body}])
        raw = "".join(getattr(b, "text", "") for b in msg.content)
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
    except Exception:
        return {}
    out = {}
    for f in wanted:
        if f in obj and obj[f] not in (None, "", []):
            out[f] = obj[f]
    return out