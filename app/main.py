"""Hearth Move-In Guardian: lease checker + move-in condition report, with a Vault and saved agent chats."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import condition_agent, config, db, dispute_agent, lease_agent
from .meta import prepare_image

app = FastAPI(title="Hearth Move-In Guardian")

LEASE_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
PHOTO_TYPES = {".png", ".jpg", ".jpeg", ".webp"}


def _read(upload: UploadFile) -> bytes:
    data = upload.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large (25 MB max).")
    if not data:
        raise HTTPException(400, "The file is empty.")
    return data


def _ext(name: str) -> str:
    return Path(name or "").suffix.lower()


def _full(item_id: str) -> dict:
    item = db.get_item(item_id)
    if not item:
        raise HTTPException(404, "Not found")
    item["files"] = [{"id": f["id"], "role": f["role"], "filename": f["filename"], "mime": f["mime"]} for f in db.list_files(item_id)]
    item["chats"] = db.list_chats(item_id)
    item["reminders"] = db.list_reminders(item_id)
    return item


@app.get("/")
def index():
    # no-cache: the browser re-checks the page on every load, so an updated app is never hidden behind an old copy
    return FileResponse(config.ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status():
    return {"mock": config.MOCK_MODE, "spark_model": config.SPARK_MODEL, "sam_model": config.SAM_MODEL}


@app.post("/api/lease")
def create_lease(file: UploadFile = File(...), language: str = Form("English"), title: str = Form("")):
    ext = _ext(file.filename)
    if ext not in LEASE_TYPES:
        raise HTTPException(400, "Upload a PDF, a text file, or a photo (png, jpg, webp) of the lease.")
    data = _read(file)
    mime = LEASE_TYPES[ext]
    item_id = db.create_item("lease", title.strip() or file.filename or "Lease")
    try:
        db.add_file(item_id, "original", file.filename or "lease", mime, data)
        result = lease_agent.run(file.filename or "lease", mime, data, language)
        db.update_item(item_id, summary=result["summary"], data={**result["data"], "mock": config.MOCK_MODE})
        for r in result["reminders"]:
            db.add_reminder(item_id, r["title"], r.get("due_date"), r.get("note", ""))
        db.add_chat(item_id, "lease", result["transcript"])
    except HTTPException:
        db.delete_item(item_id)
        raise
    except Exception as e:  # do not keep half-finished items
        db.delete_item(item_id)
        raise HTTPException(502, f"The lease agent failed: {type(e).__name__}: {str(e)[:300]}")
    return _full(item_id)


@app.post("/api/condition")
def create_condition(
    files: list[UploadFile] = File(...),
    rooms: str = Form(""),
    lease_id: str = Form(""),
    title: str = Form(""),
    language: str = Form("English"),
    move_in_id: str = Form(""),
    matches: str = Form(""),
):
    """A move-in report, or (with move_in_id) a move-out check. `matches` lists, per photo, the move-in photo_id it shows."""
    if not 1 <= len(files) <= config.MAX_PHOTOS:
        raise HTTPException(400, f"Upload between 1 and {config.MAX_PHOTOS} photos.")
    try:
        room_list = json.loads(rooms) if rooms.strip().startswith("[") else [r.strip() for r in rooms.split(",")]
    except json.JSONDecodeError:
        room_list = []
    lease = None
    if lease_id:
        lease = db.get_item(lease_id)
        if not lease or lease["kind"] != "lease":
            raise HTTPException(400, "The linked lease was not found.")
    move_in, before_by_id = None, {}
    if move_in_id:
        move_in = db.get_item(move_in_id)
        if not move_in or move_in["kind"] != "condition" or move_in["data"].get("mode") == "move_out":
            raise HTTPException(400, "The move-in report was not found.")
        if not lease and move_in.get("parent_id"):
            lease = db.get_item(move_in["parent_id"])  # the lease the move-in report was linked to, for deposit context
        before_by_id = {bp["photo_id"]: bp for bp in move_in["data"].get("photos", [])}
    try:
        match_list = json.loads(matches) if matches.strip() else []
    except json.JSONDecodeError:
        match_list = []

    photos = []
    for n, f in enumerate(files):
        if _ext(f.filename) not in PHOTO_TYPES:
            raise HTTPException(400, f"{f.filename}: upload png, jpg or webp photos.")
        raw = _read(f)
        try:
            jpeg, w, h = prepare_image(raw)
        except Exception:
            raise HTTPException(400, f"{f.filename}: this does not look like a valid image.")
        room = (room_list[n] if n < len(room_list) and room_list[n] else f"Room {n + 1}")[:60]
        photo = {"photo_id": f"p{n + 1}", "room": room, "jpeg": jpeg, "w": w, "h": h, "filename": f.filename or f"photo{n + 1}.jpg"}
        bp = before_by_id.get(match_list[n]) if n < len(match_list) else None
        stored = db.get_file(bp["file_id"]) if bp else None
        if stored and os.path.exists(stored["abs"]):
            photo["room"] = bp["room"]
            photo["before"] = {
                "jpeg": Path(stored["abs"]).read_bytes(),
                "findings": [f for f in move_in["data"].get("findings", []) if f["photo_id"] == bp["photo_id"]],
            }
        photos.append(photo)

    default_title = "Move-out check" if move_in else "Move-in condition report"
    parent = move_in["id"] if move_in else (lease["id"] if lease else None)
    item_id = db.create_item("condition", title.strip() or default_title, parent_id=parent)
    try:
        file_ids = {p["photo_id"]: db.add_file(item_id, "photo", p["filename"], "image/jpeg", p["jpeg"]) for p in photos}
        # keep our own copy of each matched move-in photo, so the check survives if the move-in report is deleted
        before_ids = {p["photo_id"]: db.add_file(item_id, "before", f"move-in-{p['photo_id']}.jpg", "image/jpeg", p["before"]["jpeg"]) for p in photos if p.get("before")}
        about = {"id": move_in["id"], "title": move_in["title"], "generated_at": move_in["data"].get("generated_at", "")} if move_in else None
        result = condition_agent.run(photos, lease, language, about)
        overlay_ids = {pid: db.add_file(item_id, "overlay", f"outlined-{pid}.jpg", "image/jpeg", jpeg) for pid, jpeg in result["overlays"].items()}
        data = {**result["data"], "mock": config.MOCK_MODE}
        data["photos"] = [
            {
                "photo_id": p["photo_id"],
                "room": p["room"],
                "file_id": file_ids[p["photo_id"]],
                "overlay_file_id": overlay_ids.get(p["photo_id"]),
                "before_file_id": before_ids.get(p["photo_id"]),
            }
            for p in photos
        ]
        db.update_item(item_id, summary=result["summary"], data=data)
        db.add_chat(item_id, "condition", result["transcript"])
    except Exception as e:
        db.delete_item(item_id)
        raise HTTPException(502, f"The condition agent failed: {type(e).__name__}: {str(e)[:300]}")
    return _full(item_id)


STATUS_TEXT = {"pre_existing": "already there at move-in", "worse": "worse than at move-in", "new": "new since move-in", "unclear": "unclear"}


def _evidence(report: dict) -> tuple[list[dict], str, dict | None]:
    """Findings of a move-out check (with verdicts) or a move-in report (all documented at move-in), and the linked lease."""
    d = report["data"]
    if d.get("mode") == "move_out":
        move_in = db.get_item((d.get("move_in") or {}).get("id", "")) if d.get("move_in") else None
        lease = db.get_item(move_in["parent_id"]) if move_in and move_in.get("parent_id") else None
        title = f"move-out check of {d.get('generated_at', '')[:10]}, compared with the move-in report of {(d.get('move_in') or {}).get('generated_at', '')[:10]}"
        rows = [(f, f["status"], f.get("status_reason", "")) for f in d.get("findings", [])]
    else:
        lease = db.get_item(report["parent_id"]) if report.get("parent_id") else None
        date = d.get("generated_at", "")[:10]
        title = f"move-in report of {date}"
        rows = [(f, "pre_existing", f"Listed in the move-in report dated {date}.") for f in d.get("findings", [])]
    evidence = [
        {"id": f"e{n}", "room": f["room"], "label": f["label"], "description": f["description"], "status": status, "status_text": STATUS_TEXT.get(status, status), "status_reason": reason}
        for n, (f, status, reason) in enumerate(rows, 1)
    ]
    return evidence, title, lease if lease and lease["kind"] == "lease" else None


def _lease_text(lease: dict) -> tuple[dict, str]:
    """Key clauses for the prompt, and the text lease quotes are checked against (the original file when it can be read locally)."""
    d = lease["data"]
    clauses = list(dict.fromkeys([t["quote"] for t in d.get("terms", []) if t.get("verified")] + [f["quote"] for f in d.get("flags", [])]))
    text = None
    original = next((f for f in db.list_files(lease["id"]) if f["role"] == "original"), None)
    if original and os.path.exists(original["abs"]):
        text = lease_agent._local_text(original["mime"], Path(original["abs"]).read_bytes())
    return {"title": lease["title"], "clauses": clauses[:30]}, text if text is not None else "\n".join(clauses)


@app.post("/api/dispute")
def create_dispute(file: UploadFile = File(...), evidence_id: str = Form(...), lease_id: str = Form(""), language: str = Form("English"), title: str = Form("")):
    """Check a landlord's deduction list against a move-out check (or move-in report) and the lease."""
    ext = _ext(file.filename)
    if ext not in LEASE_TYPES:
        raise HTTPException(400, "Upload the deduction list as a PDF, a text file, or a photo (png, jpg, webp).")
    data = _read(file)
    report = db.get_item(evidence_id)
    if not report or report["kind"] != "condition":
        raise HTTPException(400, "Choose a move-out check or move-in report as evidence.")
    evidence, evidence_title, lease = _evidence(report)
    if lease_id:
        lease = db.get_item(lease_id)
        if not lease or lease["kind"] != "lease":
            raise HTTPException(400, "The linked lease was not found.")
    lease_info, lease_text = _lease_text(lease) if lease else (None, "")
    item_id = db.create_item("dispute", title.strip() or "Deduction dispute", parent_id=report["id"])
    try:
        db.add_file(item_id, "original", file.filename or "deductions", LEASE_TYPES[ext], data)
        result = dispute_agent.run(file.filename or "deductions", LEASE_TYPES[ext], data, evidence, evidence_title, lease_info, lease_text, language)
        about = {"evidence": {"id": report["id"], "title": evidence_title}, "lease": {"id": lease["id"], "title": lease["title"]} if lease else None}
        db.update_item(item_id, summary=result["summary"], data={**result["data"], **about, "mock": config.MOCK_MODE})
        db.add_chat(item_id, "dispute", result["transcript"])
    except Exception as e:
        db.delete_item(item_id)
        raise HTTPException(502, f"The dispute agent failed: {type(e).__name__}: {str(e)[:300]}")
    return _full(item_id)


@app.get("/api/items")
def list_items():
    out = []
    for it in db.list_items():
        d = it["data"]
        out.append(
            {
                "id": it["id"],
                "kind": it["kind"],
                "title": it["title"],
                "summary": it["summary"],
                "created_at": it["created_at"],
                "parent_id": it["parent_id"],
                "mode": d.get("mode", ""),
                "count": len(d.get("flags", d.get("findings", d.get("charges", [])))),
                "mock": d.get("mock", False),
            }
        )
    return out


@app.get("/api/items/{item_id}")
def get_item(item_id: str):
    return _full(item_id)


@app.delete("/api/items/{item_id}")
def delete_item(item_id: str):
    if not db.delete_item(item_id):
        raise HTTPException(404, "Not found")
    return {"deleted": item_id}


@app.get("/api/reminders")
def reminders():
    items = {i["id"]: i for i in db.list_items()}
    return [
        {**r, "item_title": items[r["item_id"]]["title"], "item_created_at": items[r["item_id"]]["created_at"]}
        for r in db.list_reminders()
        if r["item_id"] in items
    ]


@app.get("/api/files/{file_id}")
def get_file(file_id: str):
    f = db.get_file(file_id)
    if not f or not os.path.exists(f["abs"]):
        raise HTTPException(404, "Not found")
    return FileResponse(f["abs"], media_type=f["mime"], headers={"X-Content-Type-Options": "nosniff"})
