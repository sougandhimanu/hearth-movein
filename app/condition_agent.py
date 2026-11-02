"""Condition report agent: move-in reports, and move-out checks against a move-in report.

The orchestrating model is text-only and uses tools; the tools themselves call Muse Spark (vision) and SAM 3.1:
  scan_photo     -> Muse Spark lists visible defects as short noun phrases
  segment        -> SAM 3.1 outlines a noun phrase (boxes; empty result is normal)
  check_outline  -> Muse Spark looks at the photo with numbered boxes drawn on it and says which ones show the defect
  record_finding -> saved; the confirmed boxes are merged into one outline (enforced here, in code)

Move-out checks add:
  compare_with_move_in   -> Muse Spark sees the move-in and move-out photos side by side and says whether a defect is new
  draft_landlord_message -> an English email listing what was already there at move-in
A finding's status always comes from compare_with_move_in (enforced here), never from the agent's own claim.
"""
from __future__ import annotations

import datetime as dt
import json

from . import config
from .meta import get_client
from .overlay import draw_boxes

SEVERITIES = ["low", "medium", "high"]
STATUSES = ["pre_existing", "worse", "new", "unclear"]

ISSUES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "description", "severity"],
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": SEVERITIES},
                },
            },
        }
    },
}
CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["matching_boxes", "covers_all", "reason"],
    "properties": {
        "matching_boxes": {"type": "array", "items": {"type": "integer"}},
        "covers_all": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}

COMPARE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "reason"],
    "properties": {"status": {"type": "string", "enum": STATUSES}, "reason": {"type": "string"}},
}
COMPARE_PROMPT = (
    "Image 1 is the MOVE-IN photo of {room}, taken on {move_in_date}. Image 2 is the MOVE-OUT photo of the same room, taken now. "
    "The photos may be taken from slightly different angles or in different light.\n"
    "Damage recorded in the move-in report for this room: {before}\n"
    "Defect seen in the move-out photo: {description}\n"
    "`status`: 'pre_existing' if this defect is clearly visible in the move-in photo or listed in the move-in report; "
    "'worse' if it was there at move-in but is now clearly larger or worse; 'new' if that area is clearly visible in the move-in photo "
    "and the defect is not there; 'unclear' if the move-in photo does not show that area well enough to tell. Do not guess.\n"
    "`reason`: one sentence in {language} saying what you compared."
)

SCAN_PROMPT = (
    "You are documenting the condition of a rental on {day}. Room: {room}.\n"
    "List only visible defects or damage (stains, cracks, holes, scuffs, peeling paint, mold, broken fixtures, burns...). "
    "For each: `label` = a SHORT English noun phrase of 1-3 words naming the defect, usable as an image-segmentation prompt "
    "(for example 'water stain', 'crack', 'scuff mark'); `description` = one factual sentence in {language} including where it is in the photo; "
    "`severity` = low, medium or high. If there is no visible damage return an empty list. Do not invent defects."
)
CHECK_PROMPT = (
    "{n} numbered colored rectangles (1 to {n}) are drawn on this photo. The defect is: {expected}\n"
    "`matching_boxes`: the numbers of the rectangles that really contain part of this defect (empty if none). "
    "`covers_all`: true only if those rectangles together cover the whole visible defect, false if a clear part of it is outside them. "
    "`reason`: one brief sentence."
)

SYSTEM = """You are Move-In Guardian's condition-report agent. A student is documenting the condition of a rental on move-in day to protect their deposit.
Photos: {photos}
{lease_context}
For EACH photo:
1. Call scan_photo(photo_id) to list visible defects.
2. For each defect call segment(photo_id, concept) with its short label. If count is 0, retry ONCE with a simpler single noun (for example "crack" instead of "hairline crack").
3. Call check_outline(photo_id, concept, instance_indices, expected_description) ONCE per defect with ALL instance indices segment returned. One defect is often split into several instances.
   If covers_all is false, you may call segment once with a broader phrase (for example "damaged floor") and check that.
4. Call record_finding for every real defect with the concept and the instance_indices that check_outline listed in matching_instances. If none matched, pass null for both.
Language: segment concepts stay in English (the outline model only understands English). Write record_finding `label` and `description` and the finish `summary` in {language}.
In descriptions and the summary, refer to photos by their room name, never by photo id (p1, p2...). Call several tools in the same turn when you can.
Never record a defect you did not see. When all photos are done, call finish with a short, factual summary."""


MOVE_OUT_SYSTEM = """You are Move-In Guardian's move-out agent. A student is moving out and needs proof of which damage was already there at move-in, so they are not charged for it.
Move-in report: {move_in}
Move-out photos (each with the move-in room it matches): {photos}
{lease_context}
For EACH photo:
1. Call scan_photo(photo_id) to list visible defects.
2. For each defect call compare_with_move_in(photo_id, description). The status it returns is final: never contradict it.
3. Outline each defect: segment(photo_id, concept) with its short English label, then check_outline(photo_id, concept, instance_indices, expected_description) once with ALL instance indices segment returned.
4. Call record_finding for every defect with the concept and the instance_indices check_outline listed in matching_instances (null if none),
   and the comparison_id step 2 returned for THAT defect. Its status comes from that comparison; without it the defect is marked unclear.
Then call draft_landlord_message once: a polite, factual English email that lists each defect that was already there at move-in (citing the move-in report date),
asks that none of it is deducted from the deposit, and asks for an itemised list with costs for any deduction. Do not admit fault for new or worse items and do not mention them as admissions.
Language: segment concepts stay in English. Write record_finding `label` and `description` and the finish `summary` in {language}.
In descriptions and the summary, refer to photos by their room name, never by photo id. Call several tools in the same turn when you can.
Never record a defect you did not see. Finish with a short summary counting new, worse, already-there and unclear items."""


def _concept(concept) -> str:
    return " ".join(str(concept or "").split())[:60]


def _indices(value) -> list[int]:
    if isinstance(value, int):
        value = [value]
    return list(dict.fromkeys(i for i in value or [] if isinstance(i, int)))


def _merge(boxes: list[list[int]]) -> list[int]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _tool(name, description, props, required):
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": props, "required": required}},
    }


TOOLS = [
    _tool("scan_photo", "List the visible defects in one photo.", {"photo_id": {"type": "string"}}, ["photo_id"]),
    _tool(
        "segment",
        "Outline a defect with SAM 3.1. concept must be a short noun phrase (max 4 words).",
        {"photo_id": {"type": "string"}, "concept": {"type": "string"}},
        ["photo_id", "concept"],
    ),
    _tool(
        "check_outline",
        "Show SAM instances to Muse Spark: which ones really show the defect, and do they cover all of it?",
        {
            "photo_id": {"type": "string"},
            "concept": {"type": "string"},
            "instance_indices": {"type": "array", "items": {"type": "integer"}},
            "expected_description": {"type": "string"},
        },
        ["photo_id", "concept", "instance_indices", "expected_description"],
    ),
    _tool(
        "record_finding",
        "Save one defect to the report. Its confirmed instances are merged into one outline.",
        {
            "photo_id": {"type": "string"},
            "label": {"type": "string"},
            "description": {"type": "string"},
            "severity": {"type": "string", "enum": SEVERITIES},
            "instance_indices": {"type": ["array", "null"], "items": {"type": "integer"}},
            "concept": {"type": ["string", "null"]},
            "comparison_id": {"type": ["string", "null"], "description": "Move-out checks only: the comparison_id compare_with_move_in returned for this defect"},
        },
        ["photo_id", "label", "description", "severity"],
    ),
    _tool("finish", "End the run with a short summary.", {"summary": {"type": "string"}}, ["summary"]),
]

MOVE_OUT_TOOLS = TOOLS + [
    _tool(
        "compare_with_move_in",
        "Compare a defect in a move-out photo with the matching move-in photo and report: was it already there?",
        {"photo_id": {"type": "string"}, "description": {"type": "string"}},
        ["photo_id", "description"],
    ),
    _tool(
        "draft_landlord_message",
        "Save a polite English email to the landlord about the damage that was already there at move-in.",
        {"subject": {"type": "string"}, "body": {"type": "string"}},
        ["subject", "body"],
    ),
]


class ConditionSession:
    def __init__(self, client, photos: dict[str, dict], language: str = "English", move_in: dict | None = None):
        self.client = client
        self.language = language
        self.photos = photos  # photo_id -> {room, jpeg, w, h}; move-out photos may also have `before` {jpeg, findings}
        self.move_in = move_in  # move-out checks only: {id, title, generated_at}
        self.comparisons: dict[tuple, dict] = {}
        self.draft: dict | None = None
        self.issues: dict[str, list[dict]] = {}
        self.segments: dict[tuple, list[dict]] = {}
        self.checks: dict[tuple, dict] = {}
        self.findings: list[dict] = []
        self.finished: dict | None = None

    def call(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}

    def _photo(self, photo_id):
        return self.photos.get(photo_id)

    def t_scan_photo(self, photo_id):
        p = self._photo(photo_id)
        if not p:
            return {"error": f"unknown photo_id {photo_id}"}
        if photo_id not in self.issues:
            day = "move-out day" if self.move_in else "move-in day"
            prompt = SCAN_PROMPT.format(day=day, room=p["room"], language=self.language)
            res = self.client.vision_json(prompt, ISSUES_SCHEMA, "photo_issues", images=[p["jpeg"]])
            self.issues[photo_id] = [
                {
                    "label": str(i.get("label", ""))[:60],
                    "description": str(i.get("description", ""))[:300],
                    "severity": i.get("severity") if i.get("severity") in SEVERITIES else "medium",
                }
                for i in res.get("issues", [])[:6]
            ]
        return {"photo_id": photo_id, "room": p["room"], "issues": self.issues[photo_id]}

    def t_segment(self, photo_id, concept):
        p = self._photo(photo_id)
        if not p:
            return {"error": f"unknown photo_id {photo_id}"}
        concept = _concept(concept)
        if not concept or len(concept.split()) > 4:
            return {"error": "concept must be a short noun phrase (1-4 words)"}
        key = (photo_id, concept)
        if key not in self.segments:
            self.segments[key] = self.client.segment(p["jpeg"], concept)
        inst = self.segments[key][:8]
        return {
            "concept": concept,
            "count": len(inst),
            "instances": [{"index": i["index"], "box": i["box"], "confidence": i["confidence"]} for i in inst],
        }

    def _instance(self, photo_id, concept, idx):
        for i in self.segments.get((photo_id, concept), []):
            if i["index"] == idx:
                return i
        return None

    def t_check_outline(self, photo_id, concept, instance_indices, expected_description):
        p = self._photo(photo_id)
        concept = _concept(concept)
        insts = [self._instance(photo_id, concept, i) for i in _indices(instance_indices)[:8]] if p else []
        if not insts or None in insts:
            return {"error": "unknown instance; call segment first and use the indices it returned"}
        marked = draw_boxes(p["jpeg"], [i["box"] for i in insts])
        prompt = CHECK_PROMPT.format(n=len(insts), expected=expected_description)
        res = self.client.vision_json(prompt, CHECK_SCHEMA, "outline_check", images=[marked])
        numbers = set(_indices(res.get("matching_boxes")))
        matching = [i["index"] for n, i in enumerate(insts, 1) if n in numbers]
        for i in insts:
            self.checks[(photo_id, concept, i["index"])] = {"matches": i["index"] in matching}
        return {"matching_instances": matching, "covers_all": bool(res.get("covers_all")), "reason": str(res.get("reason", ""))[:300]}

    def t_compare_with_move_in(self, photo_id, description):
        p = self._photo(photo_id)
        if not p or not self.move_in:
            return {"error": f"unknown photo_id {photo_id}" if not p else "this is not a move-out check"}
        before = p.get("before")
        if not before:
            out = {"status": "unclear", "reason": "No move-in photo was matched to this room, so it cannot be compared."}
        else:
            listed = "; ".join(f"{f['label']}: {f['description']}" for f in before["findings"]) or "none recorded"
            prompt = COMPARE_PROMPT.format(
                room=p["room"], move_in_date=self.move_in["generated_at"][:10], before=listed, description=description, language=self.language
            )
            res = self.client.vision_json(prompt, COMPARE_SCHEMA, "move_out_compare", images=[before["jpeg"], p["jpeg"]])
            status = res.get("status") if res.get("status") in STATUSES else "unclear"
            out = {"status": status, "reason": str(res.get("reason", ""))[:300]}
        comparison_id = f"c{len(self.comparisons) + 1}"
        self.comparisons[comparison_id] = {**out, "photo_id": photo_id, "used": False}
        return {"comparison_id": comparison_id, **out}

    def _take_comparison(self, photo_id, comparison_id) -> dict | None:
        """A finding gets a verdict only from its own comparison: same photo, used once. Anything else is 'unclear'."""
        c = self.comparisons.get(str(comparison_id or ""))
        if not c or c["photo_id"] != photo_id or c["used"]:
            return None
        c["used"] = True
        return c

    def t_draft_landlord_message(self, subject, body):
        if not self.move_in:
            return {"error": "this is not a move-out check"}
        self.draft = {"subject": subject, "body": body}
        return {"saved": True}

    def t_record_finding(self, photo_id, label, description, severity="medium", instance_indices=None, concept=None, comparison_id=None):
        p = self._photo(photo_id)
        if not p:
            return {"error": f"unknown photo_id {photo_id}"}
        box, verified, note = None, False, ""
        concept, wanted = _concept(concept), _indices(instance_indices)
        if wanted and concept:
            ok = [self._instance(photo_id, concept, i) for i in wanted if self.checks.get((photo_id, concept, i), {}).get("matches")]
            ok = [i for i in ok if i]
            if ok:
                box, verified = _merge([i["box"] for i in ok]), True
                if len(ok) < len(wanted):
                    note = f"{len(wanted) - len(ok)} unverified instance(s) left out of the outline"
            else:
                note = "outline not verified, so it was not attached"
        if any(f["photo_id"] == photo_id and f["label"] == label and f["box"] == box for f in self.findings):
            return {"recorded": False, "reason": "duplicate"}
        finding = {
            "photo_id": photo_id,
            "room": p["room"],
            "label": label[:60],
            "description": description[:400],
            "severity": severity if severity in SEVERITIES else "medium",
            "box": box,
            "outline_verified": verified,
            "note": note,
        }
        if self.move_in:
            c = self._take_comparison(photo_id, comparison_id)
            finding["status"] = c["status"] if c else "unclear"
            finding["status_reason"] = c["reason"] if c else "Not compared with the move-in photo, so it is marked unclear."
        self.findings.append(finding)
        out = {"recorded": True, "outline_verified": verified, "note": note}
        if self.move_in:
            out["status"] = finding["status"]
        return out

    def t_finish(self, summary):
        self.finished = {"summary": summary}
        return {"ok": True}


def _brief(msg: dict) -> dict:
    out = {"role": msg["role"], "content": (msg.get("content") or "")[:1500]}
    if msg.get("tool_calls"):
        out["tool_calls"] = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"][:1500]} for c in msg["tool_calls"]]
    return out


def run(photos: list[dict], lease: dict | None = None, language: str = "English", move_in: dict | None = None) -> dict:
    """photos: [{photo_id, room, jpeg, w, h, before?}]. lease: optional lease item (for deadline context).
    move_in: {id, title, generated_at} of the move-in report, for a move-out check; matched photos carry `before`."""
    client = get_client()
    language = (language or "English").strip()[:60] or "English"
    by_id = {p["photo_id"]: p for p in photos}
    session = ConditionSession(client, by_id, language, move_in)

    lease_context = ""
    lease_info = None
    if lease:
        d = lease.get("data", {})
        lease_info = {"lease_id": lease["id"], "title": lease["title"], "inspection_deadline": d.get("inspection_deadline"), "deposit_summary": d.get("deposit_summary")}
        lease_context = (
            f"Linked lease: {lease['title']}. Move-in inspection deadline per the lease: {d.get('inspection_deadline') or 'not stated'}. "
            f"Deposit: {d.get('deposit_summary') or 'not stated'}. Mention the deadline in your summary."
        )
    if move_in:
        photo_list = json.dumps([{"id": p["photo_id"], "room": p["room"], "has_move_in_photo": bool(p.get("before"))} for p in photos])
        about = json.dumps({"title": move_in["title"], "date": move_in["generated_at"][:10]})
        system = MOVE_OUT_SYSTEM.format(move_in=about, photos=photo_list, lease_context=lease_context, language=language)
        tools, ask = MOVE_OUT_TOOLS, "Please compare every move-out photo with move-in."
    else:
        photo_list = json.dumps([{"id": p["photo_id"], "room": p["room"]} for p in photos])
        system = SYSTEM.format(photos=photo_list, lease_context=lease_context, language=language)
        tools, ask = TOOLS, "Please document every photo."
    messages = [{"role": "system", "content": system}, {"role": "user", "content": ask}]
    transcript = [_brief(m) for m in messages]

    turns, nudged = 0, False
    for turns in range(1, config.CONDITION_MAX_TURNS + 1):
        msg = client.chat(messages, tools)
        messages.append(msg)
        transcript.append(_brief(msg))
        calls = msg.get("tool_calls") or []
        if not calls:
            if session.finished is not None or nudged:
                break
            nudged = True
            nudge = {"role": "user", "content": "Use the tools to process every photo, then call finish."}
            messages.append(nudge)
            transcript.append(_brief(nudge))
            continue
        for c in calls:
            name = c["function"]["name"]
            try:
                result = session.call(name, json.loads(c["function"]["arguments"] or "{}"))
            except json.JSONDecodeError:
                result = {"error": "arguments were not valid JSON"}
            except Exception as e:  # a failed model call becomes a tool error the agent can react to
                result = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            content = json.dumps(result, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
            transcript.append({"role": "tool", "name": name, "content": content[:1500]})
        if session.finished is not None:
            break

    if session.finished is None:  # out of turns: one last call so the summary (and a move-out email) is not lost
        missing = (["draft_landlord_message"] if move_in and not session.draft else []) + ["finish"]
        nudge = {"role": "user", "content": f"You are out of turns. Call {' and '.join(missing)} now about what you recorded."}
        messages.append(nudge)
        transcript.append(_brief(nudge))
        msg = client.chat(messages, [t for t in tools if t["function"]["name"] in missing])
        transcript.append(_brief(msg))
        for c in msg.get("tool_calls") or []:
            if c["function"]["name"] in missing:
                try:
                    session.call(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
                except json.JSONDecodeError:
                    pass

    # annotated overlays: only verified outlines are drawn, numbered so the report list can point at them
    overlays = {}
    for pid, p in by_id.items():
        outlined = [f for f in session.findings if f["photo_id"] == pid and f["box"]]
        for n, f in enumerate(outlined, 1):
            f["marker"] = n
        if outlined:
            overlays[pid] = draw_boxes(p["jpeg"], [f["box"] for f in outlined], [f["label"] for f in outlined])

    order = {"high": 0, "medium": 1, "low": 2}
    first = {"new": 0, "worse": 1, "unclear": 2, "pre_existing": 3}  # move-out: what you may be charged for first
    findings = sorted(session.findings, key=lambda f: (first.get(f.get("status"), 0), order[f["severity"]]))
    summary = (session.finished or {}).get("summary") or f"{len(findings)} issue(s) documented across {len(photos)} photo(s)."
    return {
        "summary": summary,
        "transcript": transcript,
        "overlays": overlays,
        "data": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "language": language,
            "rooms": [p["room"] for p in photos],
            "findings": findings,
            "lease": lease_info,
            "turns": turns,
            "finished": session.finished is not None,
            "segment_calls": len(session.segments),
            "outlines_verified": sum(1 for f in findings if f["outline_verified"]),
            **(
                {
                    "mode": "move_out",
                    "move_in": move_in,
                    "status_counts": {s: sum(1 for f in findings if f.get("status") == s) for s in STATUSES},
                    "draft": session.draft,
                }
                if move_in
                else {}
            ),
        },
    }
