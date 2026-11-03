"""Offline stand-in for the Meta API, so the app and the tests run without a key.

It is clearly labelled in the UI. It drives the REAL agent loops and tools, including one deliberately fabricated quote
(which the quote verification must reject) and one SAM concept that returns nothing (which the agent must retry) and one defect that SAM splits into two instances
(whose boxes must be merged into one outline).

For the sample photos in samples/, it replays the real Muse Spark + SAM 3.1 results recorded in
samples/recorded_results.json, so the outlines land on the real damage. Photos are recognised by a small
average hash (robust to re-encoding); any other photo gets the placeholder results below.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

from PIL import Image

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
SAMPLE = SAMPLES / "sample_lease.txt"


def _ahash(img: Image.Image) -> int:
    """64-bit average hash: the same photo still matches after resizing or JPEG re-encoding."""
    px = list(img.convert("L").resize((8, 8)).tobytes())
    avg = sum(px) / len(px)
    return sum(1 << i for i, v in enumerate(px) if v > avg)


def _load_recorded() -> list[dict]:
    try:
        photos = json.loads((SAMPLES / "recorded_results.json").read_text(encoding="utf-8"))["photos"]
        for ph in photos:
            ph["hash"] = _ahash(Image.open(SAMPLES / ph["file"]))
        return photos
    except (OSError, ValueError, KeyError):
        return []


RECORDED = _load_recorded()


def _recorded(jpeg: bytes) -> dict | None:
    """The recorded real results for this photo, if it is one of the sample photos."""
    if not RECORDED:
        return None
    h = _ahash(Image.open(io.BytesIO(jpeg)))
    best = min(RECORDED, key=lambda ph: bin(ph["hash"] ^ h).count("1"))
    return best if bin(best["hash"] ^ h).count("1") <= 6 else None


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


def _msg(calls: list[tuple[str, dict]], n_assistant: int, note: str = "calling tools") -> dict:
    return {
        "role": "assistant",
        "content": f"(mock) {note}",
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
            rec = _recorded(images[0]) if images else None
            if rec:
                return {"issues": [{k: f[k] for k in ("label", "description", "severity")} for f in rec["findings"]]}
            room = (re.search(r"Room: (.*?)\.", prompt) or [None, "room"])[1]
            return {
                "issues": [
                    {"label": "water stain", "description": f"Brown water stain on the ceiling near the corner ({room}).", "severity": "medium"},
                    {"label": "hairline crack", "description": f"Thin crack running down the wall ({room}).", "severity": "low"},
                ]
            }
        if name == "move_out_compare":  # identical photos, or the stain: already there at move-in; anything else is new
            same_photo = len(images) == 2 and images[0] == images[1]
            if same_photo:
                return {"status": "pre_existing", "reason": "(mock) The move-in and move-out photos show the same damage, so it was already there."}
            if "stain" in prompt.split("Defect seen in the move-out photo:", 1)[1].lower():
                return {"status": "pre_existing", "reason": "(mock) The same stain is visible in the move-in photo."}
            return {"status": "new", "reason": "(mock) This area is clear in the move-in photo."}
        if name == "outline_check":
            n = int(re.match(r"(\d+) numbered", prompt).group(1))
            return {"matching_boxes": list(range(1, n + 1)), "covers_all": True, "reason": "(mock) the outlined regions cover the described defect"}
        raise ValueError(f"mock has no handler for {name}")

    # -- SAM ----------------------------------------------------------------------------------
    def segment(self, jpeg, concept):
        rec = _recorded(jpeg)
        if rec:  # the box the real run confirmed for this defect; nothing if the real run found no outline
            for f in rec["findings"]:
                if concept in (f["label"], f["label"].split()[-1]):
                    w, h = rec["size"]
                    return [{"index": 0, "box": f["box"], "w": w, "h": h, "confidence": 0.9, "has_mask": False}] if f["box"] else []
            return []
        if "hairline" in concept:  # forces the agent to retry with a simpler noun
            return []
        img = Image.open(io.BytesIO(jpeg))
        w, h = img.size
        seed = sum(ord(c) for c in concept)
        x1 = int(w * (0.12 + 0.08 * (seed % 5)))
        y1 = int(h * (0.10 + 0.07 * (seed % 4)))
        out = [{"index": 0, "box": [x1, y1, min(w - 1, x1 + w // 4), min(h - 1, y1 + h // 5)], "w": w, "h": h, "confidence": 0.9, "has_mask": False}]
        if "stain" in concept:  # one defect split into two pieces
            out.append({"index": 1, "box": [x1 + w // 5, y1 + h // 6, min(w - 1, x1 + w // 2), min(h - 1, y1 + h // 3)], "w": w, "h": h, "confidence": 0.7, "has_mask": False})
        return out

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
        photos = json.loads(re.search(r"[Pp]hotos[^:\n]*: (\[.*?\])", messages[0]["content"], re.S).group(1))
        hist = list(_history(messages))
        scanned = {a["photo_id"]: r for nm, a, r in hist if nm == "scan_photo"}
        seg = {(a["photo_id"], a["concept"]): r for nm, a, r in hist if nm == "segment"}
        checked = {(a["photo_id"], a["concept"]): r for nm, a, r in hist if nm == "check_outline"}
        recorded = {(a["photo_id"], a["label"]) for nm, a, r in hist if nm == "record_finding"}

        todo = [("scan_photo", {"photo_id": p["id"]}) for p in photos if p["id"] not in scanned]
        if todo:
            return _msg(todo, n, "Scanning each photo for visible damage.")

        issues = [(pid, i) for pid, r in scanned.items() for i in r.get("issues", [])]
        move_out = "move-out agent" in messages[0]["content"]
        compared = {(a["photo_id"], a["description"]): r.get("comparison_id") for nm, a, r in hist if nm == "compare_with_move_in"}
        todo = [("compare_with_move_in", {"photo_id": pid, "description": i["description"]}) for pid, i in issues if move_out and (pid, i["description"]) not in compared]
        if todo:
            return _msg(todo, n, "Comparing each defect with the move-in photo.")

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
            retry = any((pid, i["label"]) in seg for pid, i in issues)
            return _msg(todo, n, "Nothing was outlined for some defects, so I'll retry with a simpler word." if retry else "Outlining each defect with SAM 3.1.")

        todo = []
        for pid, i in issues:
            c = concept_used(pid, i["label"])
            if c and (pid, c) not in checked:
                every = [x["index"] for x in seg[(pid, c)]["instances"]]
                todo.append(("check_outline", {"photo_id": pid, "concept": c, "instance_indices": every, "expected_description": i["description"]}))
        if todo:
            return _msg(todo, n, "Checking that each outline really shows the damage.")

        todo = []
        for pid, i in issues:
            if (pid, i["label"]) in recorded:
                continue
            c = concept_used(pid, i["label"])
            matching = checked.get((pid, c), {}).get("matching_instances") if c else None
            todo.append(
                (
                    "record_finding",
                    {
                        "photo_id": pid,
                        "label": i["label"],
                        "description": i["description"],
                        "severity": i["severity"],
                        "instance_indices": matching or None,
                        "concept": c if matching else None,
                        **({"comparison_id": compared.get((pid, i["description"]))} if move_out else {}),
                    },
                )
            )
        if todo:
            return _msg(todo, n, "Saving the findings to the report.")
        if move_out and not any(nm == "draft_landlord_message" for nm, a, r in hist):
            body = "Hello,\n\nThe damage listed in my move-in report was already there when I moved in. Please do not deduct it from my deposit. Thank you."
            return _msg([("draft_landlord_message", {"subject": "Move-out: damage already there at move-in", "body": body})], n, "Drafting the email to the landlord.")
        labels = ", ".join(sorted({label for _, label in recorded}))
        summary = f"(mock) Documented {len(recorded)} issue(s) across {len(photos)} photo(s)" + (f": {labels}." if labels else ".")
        return _msg([("finish", {"summary": summary})], n, "Done; writing the summary.")
