"""Offline stand-in for the Meta API, so the app and the tests run without a key.

It is clearly labelled in the UI. It drives the REAL agent loops and tools, including one deliberately fabricated quote
(which the quote verification must reject) and one SAM concept that returns nothing (which the agent must retry).
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

from PIL import Image

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "sample_lease.txt"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 25]


def _pick(sents, *kws):
    return next((s for s in sents if all(k in s.lower() for k in kws)), None)


def _history(messages):
    """Yield (tool_name, args, result) for every executed tool call."""
    calls = {}
    for m in messages:
        if m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                calls[c["id"]] = (c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
        elif m["role"] == "tool" and m["tool_call_id"] in calls:
            name, args = calls[m["tool_call_id"]]
            yield name, args, json.loads(m["content"])


def _msg(calls: list[tuple[str, dict]], n_assistant: int) -> dict:
    return {
        "role": "assistant",
        "content": "(mock model) calling tools",
        "tool_calls": [
            {"id": f"mock_{n_assistant}_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(calls)
        ],
    }


class MockMeta:
    # -- structured JSON calls ---------------------------------------------------------------
    def vision_json(self, prompt, schema, name, images=(), files=()):
        if name == "lease_terms":
            if "LEASE TEXT:\n" in prompt:
                text = prompt.split("LEASE TEXT:\n", 1)[1]
            else:  # photo / scanned PDF path: pretend to transcribe the sample lease
                text = SAMPLE.read_text() if SAMPLE.exists() else ""
            sents = _sentences(text)
            wanted = [
                ("monthly rent", ("monthly rent",)),
                ("security deposit", ("security deposit of",)),
                ("late fee", ("late fee",)),
                ("move-in inspection deadline", ("within 48 hours",)),
                ("early termination", ("early termination",)),
                ("landlord entry", ("enter the unit",)),
            ]
            terms = []
            for term_name, kws in wanted:
                s = _pick(sents, *kws)
                if s:
                    terms.append({"name": term_name, "value": s[:90], "quote": s})
            out = {"document_language": "English", "terms": terms}
            if "full_text" in schema["properties"]:
                out["full_text"] = text
            return out
        if name == "photo_issues":
            room = (re.search(r"Room: (.*?)\.", prompt) or [None, "room"])[1]
            return {
                "issues": [
                    {"label": "water stain", "description": f"Brown water stain on the ceiling near the corner ({room}).", "severity": "medium"},
                    {"label": "hairline crack", "description": f"Thin crack running down the wall ({room}).", "severity": "low"},
                ]
            }
        if name == "outline_check":
            return {"matches": True, "reason": "(mock) the outlined region covers the described defect"}
        raise ValueError(f"mock has no handler for {name}")

    # -- SAM ----------------------------------------------------------------------------------
    def segment(self, jpeg, concept):
        if "hairline" in concept:  # forces the agent to retry with a simpler noun
            return []
        img = Image.open(io.BytesIO(jpeg))
        w, h = img.size
        seed = sum(ord(c) for c in concept)
        x1 = int(w * (0.12 + 0.08 * (seed % 5)))
        y1 = int(h * (0.10 + 0.07 * (seed % 4)))
        return [{"index": 0, "box": [x1, y1, min(w - 1, x1 + w // 4), min(h - 1, y1 + h // 5)], "w": w, "h": h, "confidence": 0.9, "has_mask": False}]

    # -- agent brain --------------------------------------------------------------------------
    def chat(self, messages, tools=None):
        names = {t["function"]["name"] for t in tools or []}
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        if "flag_clause" in names:
            return self._lease_policy(messages, n_assistant)
        return self._condition_policy(messages, n_assistant)

    def _lease_policy(self, messages, n):
        text = messages[1]["content"].split("LEASE TEXT:\n", 1)[-1]
        sents = _sentences(text)

        def flag(sentence, category, severity):
            return (
                "flag_clause",
                {
                    "quote": sentence,
                    "category": category,
                    "severity": severity,
                    "explanation": "(mock) Plain-language explanation of why this clause matters.",
                    "question_to_ask": "(mock) Could you explain how this clause works in practice?",
                },
            )

        if n == 0:
            calls = [
                flag("The landlord may keep the entire deposit for any reason at all.", "deposit", "high")  # fabricated: must be rejected
            ]
            for kws, cat, sev in [(("non-refundable",), "deposit", "high"), (("within 48 hours",), "inspection", "high")]:
                s = _pick(sents, *kws)
                if s:
                    calls.append(flag(s, cat, sev))
            insp = _pick(sents, "within 48 hours") or "Check the lease for an inspection deadline."
            calls.append(("set_reminder", {"title": "Submit written move-in damage list", "due_date": None, "note": insp[:200]}))
            calls.append(("draft_landlord_message", {"subject": "Questions before signing", "body": "Hello,\n\nBefore I sign, could you clarify the deposit, the move-in inspection window and landlord entry? Thank you."}))
            return _msg(calls, n)
        if n == 1:
            calls = []
            for kws, cat, sev in [(("enter the unit",), "entry", "medium"), (("early termination",), "termination", "medium")]:
                s = _pick(sents, *kws)
                if s:
                    calls.append(flag(s, cat, sev))
            return _msg(calls or [("finish", {"summary": "(mock) done"})], n)
        insp = _pick(sents, "within 48 hours")
        dep = _pick(sents, "security deposit of")
        return _msg([("finish", {"summary": "(mock) Several clauses are worth asking about before signing.", "inspection_deadline": insp, "deposit_summary": dep})], n)

    def _condition_policy(self, messages, n):
        photos = json.loads(re.search(r"Photos: (\[.*?\])", messages[0]["content"], re.S).group(1))
        hist = list(_history(messages))
        scanned = {a["photo_id"]: r for nm, a, r in hist if nm == "scan_photo"}
        seg = {(a["photo_id"], a["concept"]): r for nm, a, r in hist if nm == "segment"}
        checked = {(a["photo_id"], a["concept"], a["instance_index"]): r for nm, a, r in hist if nm == "check_outline"}
        recorded = {(a["photo_id"], a["label"]) for nm, a, r in hist if nm == "record_finding"}

        todo = [("scan_photo", {"photo_id": p["id"]}) for p in photos if p["id"] not in scanned]
        if todo:
            return _msg(todo, n)

        issues = [(pid, i) for pid, r in scanned.items() for i in r.get("issues", [])]

        def concept_used(pid, label):
            simple = label.split()[-1]
            if seg.get((pid, label), {}).get("count", 0) > 0:
                return label
            if seg.get((pid, simple), {}).get("count", 0) > 0:
                return simple
            return None

        todo = []
        for pid, i in issues:
            label, simple = i["label"], i["label"].split()[-1]
            if (pid, label) not in seg:
                todo.append(("segment", {"photo_id": pid, "concept": label}))
            elif seg[(pid, label)]["count"] == 0 and simple != label and (pid, simple) not in seg:
                todo.append(("segment", {"photo_id": pid, "concept": simple}))
        if todo:
            return _msg(todo, n)

        todo = []
        for pid, i in issues:
            c = concept_used(pid, i["label"])
            if c and (pid, c, 0) not in checked:
                todo.append(("check_outline", {"photo_id": pid, "concept": c, "instance_index": 0, "expected_description": i["description"]}))
        if todo:
            return _msg(todo, n)

        todo = []
        for pid, i in issues:
            if (pid, i["label"]) in recorded:
                continue
            c = concept_used(pid, i["label"])
            ok = bool(c and checked.get((pid, c, 0), {}).get("matches"))
            todo.append(
                (
                    "record_finding",
                    {
                        "photo_id": pid,
                        "label": i["label"],
                        "description": i["description"],
                        "severity": i["severity"],
                        "instance_index": 0 if ok else None,
                        "concept": c if ok else None,
                    },
                )
            )
        if todo:
            return _msg(todo, n)
        return _msg([("finish", {"summary": f"(mock) Documented {len(recorded)} issue(s) across {len(photos)} photo(s)."})], n)
