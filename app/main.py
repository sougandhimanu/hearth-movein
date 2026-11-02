"""Hearth Move-In Guardian: lease checker + move-in condition report, with a Vault and saved agent chats."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import condition_agent, config, db, lease_agent
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
    return FileResponse(config.ROOT / "static" / "index.html")


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
):
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
        photos.append({"photo_id": f"p{n + 1}", "room": room, "jpeg": jpeg, "w": w, "h": h, "filename": f.filename or f"photo{n + 1}.jpg"})

    item_id = db.create_item("condition", title.strip() or "Move-in condition report", parent_id=lease["id"] if lease else None)
    try:
        file_ids = {p["photo_id"]: db.add_file(item_id, "photo", p["filename"], "image/jpeg", p["jpeg"]) for p in photos}
        result = condition_agent.run(photos, lease)
        overlay_ids = {pid: db.add_file(item_id, "overlay", f"outlined-{pid}.jpg", "image/jpeg", jpeg) for pid, jpeg in result["overlays"].items()}
        data = {**result["data"], "mock": config.MOCK_MODE}
        data["photos"] = [
            {"photo_id": p["photo_id"], "room": p["room"], "file_id": file_ids[p["photo_id"]], "overlay_file_id": overlay_ids.get(p["photo_id"])}
            for p in photos
        ]
        db.update_item(item_id, summary=result["summary"], data=data)
        db.add_chat(item_id, "condition", result["transcript"])
    except Exception as e:
        db.delete_item(item_id)
        raise HTTPException(502, f"The condition agent failed: {type(e).__name__}: {str(e)[:300]}")
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
                "count": len(d.get("flags", d.get("findings", []))),
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
    titles = {i["id"]: i["title"] for i in db.list_items()}
    return [{**r, "item_title": titles.get(r["item_id"], "")} for r in db.list_reminders()]


@app.get("/api/files/{file_id}")
def get_file(file_id: str):
    f = db.get_file(file_id)
    if not f or not os.path.exists(f["abs"]):
        raise HTTPException(404, "Not found")
    return FileResponse(f["abs"], media_type=f["mime"], headers={"X-Content-Type-Options": "nosniff"})
