"""Deposit dispute agent: checks a landlord's deduction list against the renter's evidence and lease.

1. Extraction (vision/PDF in, schema-constrained JSON out): each charge with its amount and an exact quote.
2. Agent loop (text-only tool calling): the model decides each charge (dispute, accept or ask for proof) and drafts a reply.

Verifiable by design, enforced here in code:
  - a charge can only be disputed with evidence: a finding confirmed "already there at move-in",
    or an exact lease quote that is found in the lease text;
  - the totals are computed in code from the decisions, never taken from the model.
"""
from __future__ import annotations

import json

from . import config
from .lease_agent import _brief, _local_text, _run_calls, norm, quote_in_text, quote_long_enough
from .meta import get_client, prepare_image

DECISIONS = ["dispute", "accept", "ask_for_proof"]

EXTRACT_PROMPT = """This is a landlord's list of deductions from a tenant's security deposit. Return JSON with:
- currency: the currency symbol or code used (for example "$", "€", "INR").
- charges: every deduction. Each has `description` (what the charge is for, in the list's words), `amount` (a number) and
  `quote` (the words of that line copied EXACTLY from the list).
- total_claimed: the total deducted, as a number, if the list states one; otherwise null.
Do not invent charges."""

CHARGES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["currency", "charges", "total_claimed"],
    "properties": {
        "currency": {"type": "string"},
        "total_claimed": {"type": ["number", "null"]},
        "charges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "amount", "quote"],
                "properties": {"description": {"type": "string"}, "amount": {"type": "number"}, "quote": {"type": "string"}},
            },
        },
    },
}

SYSTEM = """You are Move-In Guardian's deposit dispute agent. The landlord has deducted charges from a student's security deposit.
Decide each charge using ONLY the evidence and lease below. You are not a lawyer: be factual and polite, never threatening.

For EACH charge call assess_charge once with one decision:
- dispute: the damage was already there at move-in (pass the evidence_ids of findings whose status is "already there at move-in"),
  or the lease already covers or excludes this charge (pass an exact quote from the lease as lease_quote).
- ask_for_proof: there is no evidence either way (the damage is not in the reports, or its status is unclear). Ask for photos and receipts.
- accept: the evidence shows the damage is new or worse since move-in, or the lease clearly allows the charge.
The tool rejects a dispute without valid evidence; if that happens, choose ask_for_proof or accept instead.
Write each `reason` in {language}, in one or two sentences a student can understand.

Then call draft_reply once: a polite, firm email in English to the landlord that goes through the charges one by one,
cites the move-in report date and the lease where relevant, states the total amount disputed, and asks for receipts and photos where needed.
Write it as the tenant, in the first person ("I"), and end it with "Kind regards," and "[Your name]". Never sign it as Move-In Guardian.
Then call finish with a short summary in {language}. In reasons, the email and the summary, refer to charges and evidence by what they are
(for example "the $120 carpet cleaning"), never by ids such as c1 or e2. Call several tools in the same turn when you can.

EVIDENCE ({evidence_title}):
{evidence}

LEASE ({lease_title}): key clauses
{lease}

CHARGES ({currency}):
{charges}"""


def _tool(name, description, props, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    _tool(
        "assess_charge",
        "Decide one charge. A dispute needs evidence_ids of findings already there at move-in, or an exact lease_quote.",
        {
            "charge_id": {"type": "string"},
            "decision": {"type": "string", "enum": DECISIONS},
            "reason": {"type": "string", "description": "Why, in the student's language"},
            "evidence_ids": {"type": ["array", "null"], "items": {"type": "string"}},
            "lease_quote": {"type": ["string", "null"], "description": "Exact words from the lease, if the lease supports the decision"},
            "disputed_amount": {"type": ["number", "null"], "description": "Only if disputing part of the charge"},
        },
        ["charge_id", "decision", "reason"],
    ),
    _tool("draft_reply", "Save the English email to the landlord.", {"subject": {"type": "string"}, "body": {"type": "string"}}, ["subject", "body"]),
    _tool("finish", "End the run with a short summary.", {"summary": {"type": "string"}}, ["summary"]),
]


class DisputeSession:
    """Holds the charges and evidence and executes the tool calls. All checks live here, in code."""

    def __init__(self, charges: list[dict], evidence: list[dict], lease_text: str):
        self.charges = {c["id"]: c for c in charges}
        self.evidence = {e["id"]: e for e in evidence}
        self.lease = norm(lease_text)
        self.assessments: dict[str, dict] = {}
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

    def t_assess_charge(self, charge_id, decision, reason, evidence_ids=None, lease_quote=None, disputed_amount=None):
        charge = self.charges.get(str(charge_id))
        if not charge:
            return {"accepted": False, "reason": f"unknown charge_id {charge_id}; use the ids from the CHARGES list"}
        if decision not in DECISIONS:
            return {"accepted": False, "reason": f"decision must be one of {DECISIONS}"}
        ids = [str(e) for e in (evidence_ids or [])]
        unknown = [e for e in ids if e not in self.evidence]
        if unknown:
            return {"accepted": False, "reason": f"unknown evidence ids {unknown}; use the ids from the EVIDENCE list"}
        pre_existing = [e for e in ids if self.evidence[e]["status"] == "pre_existing"]
        quote_ok = bool(lease_quote) and quote_long_enough(lease_quote) and quote_in_text(lease_quote, self.lease)
        if decision == "dispute" and not (pre_existing or quote_ok):
            why = []
            if ids:
                why.append("none of those findings is marked already there at move-in")
            if lease_quote:
                why.append("the lease quote was not found word for word in the lease")
            return {"accepted": False, "reason": "a dispute needs evidence: " + ("; ".join(why) or "pass evidence_ids or an exact lease_quote")}
        amount = charge["amount"]
        if decision == "dispute":
            try:
                disputed = amount if disputed_amount is None else max(0.0, min(float(disputed_amount), amount))
            except (TypeError, ValueError):
                disputed = amount
        else:
            disputed = 0.0
        self.assessments[charge["id"]] = {
            "decision": decision,
            "reason": str(reason)[:600],
            "evidence_ids": pre_existing if decision == "dispute" else ids,
            "lease_quote": lease_quote.strip() if quote_ok else None,
            "disputed_amount": disputed,
        }
        left = [c for c in self.charges if c not in self.assessments]
        return {"accepted": True, "decision": decision, "disputed_amount": disputed, "charges_left": left}

    def t_draft_reply(self, subject, body):
        self.draft = {"subject": subject, "body": body}
        return {"saved": True}

    def t_finish(self, summary):
        self.finished = {"summary": summary}
        return {"ok": True}


def _wrap_up(client, session: DisputeSession, messages: list[dict], transcript: list[dict]) -> int:
    """Out of turns: one last call so the reply and summary are never lost."""
    missing = ([] if session.draft else ["draft_reply"]) + ["finish"]
    nudge = {"role": "user", "content": f"You are out of turns. Call {' and '.join(missing)} now."}
    messages.append(nudge)
    transcript.append(_brief(nudge))
    msg = client.chat(messages, [t for t in TOOLS if t["function"]["name"] in missing])
    messages.append(msg)
    transcript.append(_brief(msg))
    _run_calls(session, msg.get("tool_calls") or [], messages, transcript)
    return 1


def run(filename: str, mime: str, data: bytes, evidence: list[dict], evidence_title: str, lease: dict | None, lease_text: str, language: str = "English") -> dict:
    """evidence: [{id, room, label, description, status, status_reason}] with status in pre_existing/worse/new/unclear.
    lease: {title, clauses: [str]} for the prompt; lease_text: the text lease quotes are checked against."""
    client = get_client()
    language = (language or "English").strip()[:60] or "English"

    # 1) read the deduction list
    local = _local_text(mime, data)
    if local is not None:
        res = client.vision_json(EXTRACT_PROMPT + "\n\nDEDUCTION LIST:\n" + local[:40_000], CHARGES_SCHEMA, "deduction_items")
        source = norm(local)
    else:
        kwargs = {"files": [(filename, data, mime)]} if mime == "application/pdf" else {"images": [prepare_image(data)[0]]}
        res = client.vision_json(EXTRACT_PROMPT, CHARGES_SCHEMA, "deduction_items", **kwargs)
        source = None
    charges = []
    for n, c in enumerate(res.get("charges", [])[:30], 1):
        try:
            amount = round(abs(float(c.get("amount") or 0)), 2)
        except (TypeError, ValueError):
            amount = 0.0
        quote = str(c.get("quote", ""))[:400]
        charges.append({
            "id": f"c{n}",
            "description": str(c.get("description", ""))[:300],
            "amount": amount,
            "quote": quote,
            "quote_verified": quote_in_text(quote, source) if source is not None else None,
        })
    currency = str(res.get("currency") or "")[:8]

    # 2) agent loop
    session = DisputeSession(charges, evidence, lease_text)
    clauses = "\n".join(f"- {c}" for c in (lease or {}).get("clauses", [])) or "(no lease linked)"
    system = SYSTEM.format(
        language=language,
        evidence_title=evidence_title,
        evidence=json.dumps(evidence, ensure_ascii=False),
        lease_title=(lease or {}).get("title", "none"),
        lease=clauses,
        currency=currency,
        charges=json.dumps([{k: c[k] for k in ("id", "description", "amount")} for c in charges], ensure_ascii=False),
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "Please check every charge."}]
    transcript = [{"role": "system", "content": system[:1500]}, _brief(messages[1])]
    turns, nudged = 0, False
    for turns in range(1, config.LEASE_MAX_TURNS + 1):
        msg = client.chat(messages, TOOLS)
        messages.append(msg)
        transcript.append(_brief(msg))
        calls = msg.get("tool_calls") or []
        if not calls:
            if session.finished is not None or nudged:
                break
            nudged = True
            nudge = {"role": "user", "content": "Use the tools: assess_charge for every charge, then draft_reply, then finish."}
            messages.append(nudge)
            transcript.append(_brief(nudge))
            continue
        _run_calls(session, calls, messages, transcript)
        if session.finished is not None:
            break
    if session.finished is None:
        turns += _wrap_up(client, session, messages, transcript)

    # totals come from the decisions, computed here
    for c in charges:
        a = session.assessments.get(c["id"]) or {"decision": "ask_for_proof", "reason": "Not assessed by the agent.", "evidence_ids": [], "lease_quote": None, "disputed_amount": 0.0}
        c.update(a)
        c["evidence"] = [session.evidence[e] for e in a["evidence_ids"] if e in session.evidence]
    total = round(sum(c["amount"] for c in charges), 2)
    totals = {
        "charged": total,
        "disputed": round(sum(c["disputed_amount"] for c in charges), 2),
        "on_hold": round(sum(c["amount"] for c in charges if c["decision"] == "ask_for_proof"), 2),
        "accepted": round(sum(c["amount"] - c["disputed_amount"] for c in charges if c["decision"] != "ask_for_proof"), 2),
    }
    claimed = res.get("total_claimed")
    fin = session.finished or {}
    summary = fin.get("summary") or f"{sum(c['decision'] == 'dispute' for c in charges)} of {len(charges)} charge(s) disputed."
    return {
        "summary": summary,
        "transcript": transcript,
        "data": {
            "mode": "dispute",
            "language": language,
            "currency": currency,
            "charges": charges,
            "totals": totals,
            "total_claimed": claimed,
            "total_matches": claimed is None or abs(float(claimed) - total) < 0.01,
            "verification_source": "local_text" if source is not None else "model_transcription",
            "draft": session.draft,
            "turns": turns,
            "finished": session.finished is not None,
        },
    }
