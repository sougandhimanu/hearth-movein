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

from . import config
from .meta import get_client, prepare_image

CATEGORIES = ["fee", "deposit", "termination", "entry", "maintenance", "liability", "renewal", "inspection", "other"]
SEVERITIES = ["low", "medium", "high"]
LEASE_CHAR_LIMIT = 120_000

_PUNCT = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").translate(_PUNCT).lower()).strip()


def quote_in_text(quote: str, source_norm: str) -> bool:
    """True if every segment of the quote (split on ...) appears in the normalized source."""
    parts = [p.strip(" .") for p in re.split(r"\.\.\.|…", norm(quote))]
    parts = [p for p in parts if p]
    return bool(parts) and all(len(p) >= 8 and p in source_norm for p in parts)


# ---------------------------------------------------------------- extraction
EXTRACT_PROMPT = """Read this rental lease. Return JSON with:
- document_language: the language the lease is written in.
{transcribe}- terms: key facts. Each has `name`, `value` (short plain-English value) and `quote` (words copied EXACTLY from the lease, never paraphrased).
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


def _local_text(mime: str, data: bytes) -> str | None:
    if mime == "text/plain":
        return data.decode("utf-8", "replace")
    if mime == "application/pdf":
        try:
            from pypdf import PdfReader

            text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        except Exception:
            return None
        return text if len(text.strip()) >= 200 else None
    return None


# ---------------------------------------------------------------- agent tools
SYSTEM = """You are Move-In Guardian, an agent that helps an international student understand a rental lease BEFORE they sign.
Rules:
- You are not a lawyer. Never say a clause is illegal and never give legal advice. Frame every point as something to understand or ask the landlord about.
- Work only from the lease text provided. Do not invent clauses.
- For every concerning, unusual or costly clause call flag_clause with a quote copied EXACTLY from the lease. If the tool says the quote was not found, copy it again exactly or drop the flag.
- Write `explanation` and `question_to_ask` in {language}.
- Call set_reminder for every deadline or action date the student must handle (for example a move-in inspection window or a notice period). Use YYYY-MM-DD for due_date; if the lease gives only a relative time, put it in the note and leave due_date null.
- Call draft_landlord_message once: a polite email in English asking about the flagged items.
- Flag at most 10 items, most important first. When you are done, call finish.
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
        if not quote_in_text(quote, self.source):
            return {"accepted": False, "reason": "quote not found in the lease text; copy the exact words or drop this flag"}
        key = norm(quote)
        if any(norm(f["quote"]) == key for f in self.flags):
            return {"accepted": False, "reason": "already flagged"}
        if len(self.flags) >= 12:
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
        if due_date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(due_date)):
            note = f"{note} (date given as: {due_date})".strip()
            due_date = None
        self.reminders.append({"title": title, "due_date": due_date, "note": note})
        return {"saved": True}

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
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
                result = session.call(name, args)
            except json.JSONDecodeError:
                result = {"error": "arguments were not valid JSON"}
            content = json.dumps(result, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
            transcript.append({"role": "tool", "name": name, "content": content[:1500]})
        if session.finished is not None:
            break
    return turns, transcript


def run(filename: str, mime: str, data: bytes, language: str = "English") -> dict:
    client = get_client()
    language = (language or "English").strip()[:60] or "English"

    # 1) extraction
    local = _local_text(mime, data)
    if local is not None:
        text = local[:LEASE_CHAR_LIMIT]
        extracted = client.vision_json(
            EXTRACT_PROMPT.format(transcribe="") + "\n\nLEASE TEXT:\n" + text, _terms_schema(False), "lease_terms"
        )
        source_text, verification_source = local, "local_text"
    else:
        if mime == "application/pdf":
            kwargs = {"files": [(filename, data, mime)]}
        else:
            jpeg, _, _ = prepare_image(data)
            kwargs = {"images": [jpeg]}
        extracted = client.vision_json(
            EXTRACT_PROMPT.format(transcribe="- full_text: a faithful, complete transcription of the lease in reading order. Do not summarize.\n"),
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
    system = {"role": "system", "content": SYSTEM.format(language=language, today=dt.date.today().isoformat())}
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
