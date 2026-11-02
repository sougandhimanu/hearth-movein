"""Move-in condition report agent.

The orchestrating model is text-only and uses tools; the tools themselves call Muse Spark (vision) and SAM 3.1:
  scan_photo     -> Muse Spark lists visible defects as short noun phrases
  segment        -> SAM 3.1 outlines a noun phrase (boxes; empty result is normal)
  check_outline  -> Muse Spark looks at the photo with the box drawn on it and says whether it shows the defect
  record_finding -> saved; an outline is only attached if check_outline said it matches (enforced here, in code)
"""
from __future__ import annotations

import datetime as dt
import json

from . import config
from .meta import get_client
from .overlay import draw_boxes

SEVERITIES = ["low", "medium", "high"]

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
    "required": ["matches", "reason"],
    "properties": {"matches": {"type": "boolean"}, "reason": {"type": "string"}},
}

SCAN_PROMPT = (
    "You are documenting the condition of a rental on move-in day. Room: {room}.\n"
    "List only visible defects or damage (stains, cracks, holes, scuffs, peeling paint, mold, broken fixtures, burns...). "
    "For each: `label` = a SHORT noun phrase of 1-3 words naming the defect, usable as an image-segmentation prompt "
    "(for example 'water stain', 'crack', 'scuff mark'); `description` = one factual sentence including where it is in the photo; "
    "`severity` = low, medium or high. If there is no visible damage return an empty list. Do not invent defects."
)
CHECK_PROMPT = (
    "A colored rectangle is drawn on this photo. Does the area inside the rectangle show this: {expected}? "
    "Say matches=true only if the highlighted area really contains it, and give a brief reason."
)

SYSTEM = """You are Move-In Guardian's condition-report agent. A student is documenting the condition of a rental on move-in day to protect their deposit.
Photos: {photos}
{lease_context}
For EACH photo:
1. Call scan_photo(photo_id) to list visible defects.
2. For each defect call segment(photo_id, concept) with its short label. If count is 0, retry ONCE with a simpler single noun (for example "crack" instead of "hairline crack").
3. For each outline you want to use call check_outline(photo_id, concept, instance_index, expected_description). Only use outlines where matches is true.
4. Call record_finding for every real defect. Pass instance_index and concept only if its outline check passed; otherwise pass null for both.
Never record a defect you did not see. When all photos are done, call finish with a short, factual summary."""


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
        "Verify that one SAM instance really shows the defect.",
        {
            "photo_id": {"type": "string"},
            "concept": {"type": "string"},
            "instance_index": {"type": "integer"},
            "expected_description": {"type": "string"},
        },
        ["photo_id", "concept", "instance_index", "expected_description"],
    ),
    _tool(
        "record_finding",
        "Save one defect to the report.",
        {
            "photo_id": {"type": "string"},
            "label": {"type": "string"},
            "description": {"type": "string"},
            "severity": {"type": "string", "enum": SEVERITIES},
            "instance_index": {"type": ["integer", "null"]},
            "concept": {"type": ["string", "null"]},
        },
        ["photo_id", "label", "description", "severity"],
    ),
    _tool("finish", "End the run with a short summary.", {"summary": {"type": "string"}}, ["summary"]),
]


class ConditionSession:
    def __init__(self, client, photos: dict[str, dict]):
        self.client = client
        self.photos = photos  # photo_id -> {room, jpeg, w, h}
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
            res = self.client.vision_json(SCAN_PROMPT.format(room=p["room"]), ISSUES_SCHEMA, "photo_issues", images=[p["jpeg"]])
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
        concept = " ".join(str(concept).split())[:60]
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

    def t_check_outline(self, photo_id, concept, instance_index, expected_description):
        p = self._photo(photo_id)
        inst = self._instance(photo_id, concept, instance_index) if p else None
        if not inst:
            return {"error": "no such instance; call segment first and use an index it returned"}
        marked = draw_boxes(p["jpeg"], [inst["box"]])
        res = self.client.vision_json(CHECK_PROMPT.format(expected=expected_description), CHECK_SCHEMA, "outline_check", images=[marked])
        out = {"matches": bool(res.get("matches")), "reason": str(res.get("reason", ""))[:300]}
        self.checks[(photo_id, concept, instance_index)] = out
        return out

    def t_record_finding(self, photo_id, label, description, severity="medium", instance_index=None, concept=None):
        p = self._photo(photo_id)
        if not p:
            return {"error": f"unknown photo_id {photo_id}"}
        box, verified, note = None, False, ""
        if instance_index is not None and concept:
            check = self.checks.get((photo_id, concept, instance_index))
            inst = self._instance(photo_id, concept, instance_index)
            if check and check["matches"] and inst:
                box, verified = inst["box"], True
            else:
                note = "outline not verified, so it was not attached"
        if any(f["photo_id"] == photo_id and f["label"] == label and f["box"] == box for f in self.findings):
            return {"recorded": False, "reason": "duplicate"}
        self.findings.append(
            {
                "photo_id": photo_id,
                "room": p["room"],
                "label": label[:60],
                "description": description[:400],
                "severity": severity if severity in SEVERITIES else "medium",
                "box": box,
                "outline_verified": verified,
                "note": note,
            }
        )
        return {"recorded": True, "outline_verified": verified, "note": note}

    def t_finish(self, summary):
        self.finished = {"summary": summary}
        return {"ok": True}


def _brief(msg: dict) -> dict:
    out = {"role": msg["role"], "content": (msg.get("content") or "")[:1500]}
    if msg.get("tool_calls"):
        out["tool_calls"] = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"][:1500]} for c in msg["tool_calls"]]
    return out


def run(photos: list[dict], lease: dict | None = None) -> dict:
    """photos: [{photo_id, room, jpeg, w, h}]. lease: optional lease item (for deadline context)."""
    client = get_client()
    by_id = {p["photo_id"]: p for p in photos}
    session = ConditionSession(client, by_id)

    lease_context = ""
    lease_info = None
    if lease:
        d = lease.get("data", {})
        lease_info = {"lease_id": lease["id"], "title": lease["title"], "inspection_deadline": d.get("inspection_deadline"), "deposit_summary": d.get("deposit_summary")}
        lease_context = (
            f"Linked lease: {lease['title']}. Move-in inspection deadline per the lease: {d.get('inspection_deadline') or 'not stated'}. "
            f"Deposit: {d.get('deposit_summary') or 'not stated'}. Mention the deadline in your summary."
        )
    photo_list = json.dumps([{"id": p["photo_id"], "room": p["room"]} for p in photos])
    messages = [
        {"role": "system", "content": SYSTEM.format(photos=photo_list, lease_context=lease_context)},
        {"role": "user", "content": "Please document every photo."},
    ]
    transcript = [_brief(m) for m in messages]

    turns, nudged = 0, False
    for turns in range(1, config.CONDITION_MAX_TURNS + 1):
        msg = client.chat(messages, TOOLS)
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

    # annotated overlays: only verified outlines are drawn
    overlays = {}
    for pid, p in by_id.items():
        boxes = [f["box"] for f in session.findings if f["photo_id"] == pid and f["box"]]
        labels = [f["label"] for f in session.findings if f["photo_id"] == pid and f["box"]]
        if boxes:
            overlays[pid] = draw_boxes(p["jpeg"], boxes, labels)

    order = {"high": 0, "medium": 1, "low": 2}
    findings = sorted(session.findings, key=lambda f: order[f["severity"]])
    summary = (session.finished or {}).get("summary") or f"{len(findings)} issue(s) documented across {len(photos)} photo(s)."
    return {
        "summary": summary,
        "transcript": transcript,
        "overlays": overlays,
        "data": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "rooms": [p["room"] for p in photos],
            "findings": findings,
            "lease": lease_info,
            "turns": turns,
            "finished": session.finished is not None,
            "segment_calls": len(session.segments),
            "outlines_verified": sum(1 for f in findings if f["outline_verified"]),
        },
    }
