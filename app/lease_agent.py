"""Lease checker agent.

1. Extraction (vision/PDF in, schema-constrained JSON out): key terms, each with an exact quote.
2. Agent loop (text-only tool calling): the model flags clauses, sets reminders and drafts a landlord message.

Verifiable by design: a flag is only accepted if its quote is found in the lease text. For PDFs and .txt files that
text is read locally (independent of the model); for photos it is the model's own transcription (weaker check).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import unicodedata

from . import config
from .meta import get_client, prepare_image

CATEGORIES = ["fee", "deposit", "termination", "entry", "maintenance", "liability", "renewal", "inspection", "other"]
SEVERITIES = ["low", "medium", "high"]
LEASE_CHAR_LIMIT = 120_000
MAX_FLAGS = 10

_PUNCT = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
)


# zero-width joiners/spaces, BOM and soft hyphen: invisible, and often dropped or added when text is copied
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"))
MIN_QUOTE_WORDS, MIN_QUOTE_CHARS = 5, 30


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").translate(_INVISIBLE).translate(_PUNCT)
    return re.sub(r"\s+", " ", s.lower()).strip()


def quote_long_enough(quote: str) -> bool:
    """A flag must quote a real clause, not a fragment such as "$1,150" or a single word that could come from anywhere."""
    q = norm(quote)
    return len(q.split()) >= MIN_QUOTE_WORDS or len(q) >= MIN_QUOTE_CHARS


def quote_in_text(quote: str, source_norm: str) -> bool:
    """True if every segment of the quote (split on ...) appears in the normalized source."""
    parts = [p.strip(" .") for p in re.split(r"\.\.\.|…", norm(quote))]
    parts = [p for p in parts if p]
    return bool(parts) and all(len(p) >= 8 and p in source_norm for p in parts)


_FILLER = {"a", "an", "the", "of", "on", "in", "to", "by", "for", "and", "or", "at", "before", "after", "due", "within", "your", "my"}


def _title_words(title: str) -> set[str]:
    words = re.findall(r"\w[\w$,.\-]*", norm(title))
    return {w.rstrip("s") if len(w) > 3 else w for w in words if w not in _FILLER}


def similar_titles(a: str, b: str) -> bool:
    """Same title, or most of the meaningful words shared ("Pay security deposit before keys" ~ "Pay security deposit before key handover")."""
    if norm(a) == norm(b):
        return True
    wa, wb = _title_words(a), _title_words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


# ---------------------------------------------------------------- extraction
EXTRACT_PROMPT = """Read this rental lease. Return JSON with:
- document_language: the language the lease is written in.
{transcribe}- terms: key facts. Each has `name`, `value` (short value written in {language}) and `quote` (words copied EXACTLY from the lease, never paraphrased).
  Prefer these names when the lease covers them: monthly rent, security deposit, lease start, lease end, late fee,
  move-in inspection deadline, deposit return terms, early termination, utilities and fees, landlord entry,
  guests and subletting, maintenance responsibility, renewal.
  If the lease does not say something, leave that term out. Do not guess."""


def _terms_schema(include_text: bool) -> dict:
    props = {
        "document_language": {"type": "string"},
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "value", "quote"],
                "properties": {"name": {"type": "string"}, "value": {"type": "string"}, "quote": {"type": "string"}},
            },
        },
    }
    required = ["document_language", "terms"]
    if include_text:
        props["full_text"] = {"type": "string"}
        required.insert(1, "full_text")
    return {"type": "object", "additionalProperties": False, "required": required, "properties": props}


def _garbled(text: str) -> bool:
    """pypdf often turns the conjunct letters of Indic scripts into NUL or replacement characters."""
    bad = sum(text.count(c) for c in ("\x00", "\ufffd"))
    return bad >= 3 or bad > len(text) * 0.001


def _local_text(mime: str, data: bytes) -> str | None:
    if mime == "text/plain":
        return data.decode("utf-8", "replace")
    if mime == "application/pdf":
        try:
            from pypdf import PdfReader

            text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        except Exception:
            return None
        if len(text.strip()) < 200 or _garbled(text):
            return None  # scanned or badly encoded: let the model read the PDF itself
        return text
    return None


# ---------------------------------------------------------------- agent tools
SYSTEM = """You are Move-In Guardian, an agent that helps an international student understand a rental lease BEFORE they sign.
Rules:
- You are not a lawyer. Never say a clause is illegal and never give legal advice. Frame every point as something to understand or ask the landlord about.
- Work only from the lease text provided. Do not invent clauses.
- For every concerning, unusual or costly clause call flag_clause with a quote copied EXACTLY from the lease. If the tool says the quote was not found, copy it again exactly or drop the flag.
- Write `explanation`, `question_to_ask`, each reminder's `title` and `note`, and the finish `summary` in {language}. Keep amounts, dates and quotes exactly as the lease gives them.
- Call set_reminder for every deadline or action date the student must handle (for example a move-in inspection window or a notice period). Use YYYY-MM-DD for due_date; if the lease gives only a relative time, put it in the note and leave due_date null.
- Call draft_landlord_message once: a polite email in English asking about the flagged items.
- Flag at most {max_flags} items, most important first. Set one reminder per deadline. When you are done, call finish.
- Work in as few turns as possible: call several tools in the same turn (for example all your flag_clause calls together, then all set_reminder calls).
Today's date: {today}."""


def _tool(name, description, props, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


TOOLS = [
    _tool(
        "flag_clause",
        "Flag one clause the student should understand or ask about. The quote must be copied exactly from the lease.",
        {
            "quote": {"type": "string", "description": "Exact words from the lease"},
            "category": {"type": "string", "enum": CATEGORIES},
            "severity": {"type": "string", "enum": SEVERITIES},
            "explanation": {"type": "string", "description": "Plain-language explanation in the student's language"},
            "question_to_ask": {"type": "string", "description": "A question to ask the landlord, in the student's language"},
        },
        ["quote", "category", "severity", "explanation", "question_to_ask"],
    ),
    _tool(
        "set_reminder",
        "Save a reminder for a date or deadline the student must act on.",
        {
            "title": {"type": "string"},
            "due_date": {"type": ["string", "null"], "description": "YYYY-MM-DD or null"},
            "note": {"type": "string"},
        },
        ["title", "note"],
    ),
    _tool(
        "draft_landlord_message",
        "Save a polite English email to the landlord asking about the flagged items.",
        {"subject": {"type": "string"}, "body": {"type": "string"}},
        ["subject", "body"],
    ),
    _tool(
        "finish",
        "End the run with a short summary.",
        {
            "summary": {"type": "string"},
            "inspection_deadline": {"type": ["string", "null"], "description": "Move-in inspection / damage-report deadline as the lease states it"},
            "deposit_summary": {"type": ["string", "null"]},
        },
        ["summary"],
    ),
]


class LeaseSession:
    """Holds the agent's state and executes its tool calls. All verification lives here, in code."""

    def __init__(self, source_text: str):
        self.source = norm(source_text)
        self.flags: list[dict] = []
        self.reminders: list[dict] = []
        self.draft: dict | None = None
        self.finished: dict | None = None

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    def t_flag_clause(self, quote, category="other", severity="medium", explanation="", question_to_ask=""):
        if not quote_long_enough(quote):
            return {"accepted": False, "reason": f"quote too short; copy the whole sentence or clause (at least {MIN_QUOTE_WORDS} words)"}
        if not quote_in_text(quote, self.source):
            return {"accepted": False, "reason": "quote not found in the lease text; copy the exact words or drop this flag"}
        key = norm(quote)
        if any(norm(f["quote"]) == key for f in self.flags):
            return {"accepted": False, "reason": "already flagged"}
        if len(self.flags) >= MAX_FLAGS:
            return {"accepted": False, "reason": "flag limit reached"}
        self.flags.append(
            {
                "quote": quote.strip(),
                "category": category if category in CATEGORIES else "other",
                "severity": severity if severity in SEVERITIES else "medium",
                "explanation": explanation,
                "question_to_ask": question_to_ask,
                "verified": True,
            }
        )
        return {"accepted": True, "flags_so_far": len(self.flags)}

    def t_set_reminder(self, title, note="", due_date=None):
        if str(due_date).strip().lower() in ("", "null", "none", "n/a"):
            due_date = None
        if due_date:
            try:
                due_date = dt.date.fromisoformat(str(due_date)).isoformat()
            except ValueError:
                note = f"{note} (date given as: {due_date})".strip()
                due_date = None
        for r in self.reminders:
            if r["due_date"] == due_date and similar_titles(r["title"], title):
                return {"saved": False, "reason": f"a reminder for this already exists: {r['title']!r}"}
        self.reminders.append({"title": title, "due_date": due_date, "note": note})
        return {"saved": True, "reminders_so_far": [r["title"] for r in self.reminders]}

    def t_draft_landlord_message(self, subject, body):
        self.draft = {"subject": subject, "body": body}
        return {"saved": True}

    def t_finish(self, summary, inspection_deadline=None, deposit_summary=None):
        self.finished = {"summary": summary, "inspection_deadline": inspection_deadline, "deposit_summary": deposit_summary}
        return {"ok": True}


def _brief(msg: dict) -> dict:
    out = {"role": msg["role"], "content": (msg.get("content") or "")[:1500]}
    if msg.get("tool_calls"):
        out["tool_calls"] = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"][:1500]} for c in msg["tool_calls"]]
    return out


def _agent_loop(client, session: LeaseSession, messages: list[dict]) -> tuple[int, list[dict]]:
    transcript = [_brief(messages[0]), {"role": "user", "content": "[student language, extracted terms and the lease text omitted from this saved chat]"}]
    nudged = False
    turns = 0
    for turns in range(1, config.LEASE_MAX_TURNS + 1):
        msg = client.chat(messages, TOOLS)
        messages.append(msg)
        transcript.append(_brief(msg))
        calls = msg.get("tool_calls") or []
        if not calls:
            if session.finished is not None or nudged:
                break
            nudged = True
            nudge = {"role": "user", "content": "Use the tools now: flag_clause, set_reminder, draft_landlord_message, then finish."}
            messages.append(nudge)
            transcript.append(_brief(nudge))
            continue
        _run_calls(session, calls, messages, transcript)
        if session.finished is not None:
            break
    if session.finished is None:
        turns += _wrap_up(client, session, messages, transcript)
    return turns, transcript


def _run_calls(session: LeaseSession, calls: list[dict], messages: list[dict], transcript: list[dict]) -> None:
    for c in calls:
        name = c["function"]["name"]
        try:
            result = session.call(name, json.loads(c["function"]["arguments"] or "{}"))
        except json.JSONDecodeError:
            result = {"error": "arguments were not valid JSON"}
        content = json.dumps(result, ensure_ascii=False)
        messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
        transcript.append({"role": "tool", "name": name, "content": content[:1500]})


def _wrap_up(client, session: LeaseSession, messages: list[dict], transcript: list[dict]) -> int:
    """Out of turns: one last call so the landlord message and the summary are never lost."""
    missing = ([] if session.draft else ["draft_landlord_message"]) + ["finish"]
    nudge = {"role": "user", "content": f"You are out of turns. Do not flag anything else. Call {' and '.join(missing)} now."}
    messages.append(nudge)
    transcript.append(_brief(nudge))
    msg = client.chat(messages, [t for t in TOOLS if t["function"]["name"] in missing])
    messages.append(msg)
    transcript.append(_brief(msg))
    _run_calls(session, msg.get("tool_calls") or [], messages, transcript)
    return 1


def run(filename: str, mime: str, data: bytes, language: str = "English") -> dict:
    client = get_client()
    language = (language or "English").strip()[:60] or "English"

    # 1) extraction
    local = _local_text(mime, data)
    if local is not None:
        text = local[:LEASE_CHAR_LIMIT]
        extracted = client.vision_json(
            EXTRACT_PROMPT.format(transcribe="", language=language) + "\n\nLEASE TEXT:\n" + text, _terms_schema(False), "lease_terms"
        )
        source_text, verification_source = local, "local_text"
    else:
        if mime == "application/pdf":
            kwargs = {"files": [(filename, data, mime)]}
        else:
            jpeg, _, _ = prepare_image(data)
            kwargs = {"images": [jpeg]}
        extracted = client.vision_json(
            EXTRACT_PROMPT.format(
                transcribe="- full_text: a faithful, complete transcription of the lease in reading order. Do not summarize.\n", language=language
            ),
            _terms_schema(True),
            "lease_terms",
            **kwargs,
        )
        source_text, verification_source = extracted.get("full_text", ""), "model_transcription"

    session = LeaseSession(source_text)
    terms = []
    for t in extracted.get("terms", []):
        terms.append({**t, "verified": quote_in_text(t.get("quote", ""), session.source)})

    # 2) agent loop
    system = {"role": "system", "content": SYSTEM.format(language=language, max_flags=MAX_FLAGS, today=dt.date.today().isoformat())}
    user = {
        "role": "user",
        "content": (
            f"STUDENT LANGUAGE: {language}\n\n"
            f"EXTRACTED TERMS (check against the text before relying on them):\n{json.dumps(terms, ensure_ascii=False)}\n\n"
            f"LEASE TEXT:\n{source_text[:LEASE_CHAR_LIMIT]}"
        ),
    }
    messages = [system, user]
    turns, transcript = _agent_loop(client, session, messages)

    fin = session.finished or {}
    summary = fin.get("summary") or f"{len(session.flags)} item(s) flagged for questions."
    return {
        "summary": summary,
        "transcript": transcript,
        "reminders": session.reminders,
        "data": {
            "filename": filename,
            "language": language,
            "document_language": extracted.get("document_language", ""),
            "verification_source": verification_source,
            "terms": terms,
            "flags": session.flags,
            "draft": session.draft,
            "inspection_deadline": fin.get("inspection_deadline"),
            "deposit_summary": fin.get("deposit_summary"),
            "turns": turns,
            "finished": session.finished is not None,
        },
    }
