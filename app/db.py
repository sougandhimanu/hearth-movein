"""SQLite storage for the Vault: items, files, saved agent chats and reminders."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import config

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS items(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, parent_id TEXT,
  created_at REAL NOT NULL, summary TEXT DEFAULT '', data TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS files(
  id TEXT PRIMARY KEY, item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  role TEXT NOT NULL, filename TEXT NOT NULL, mime TEXT NOT NULL, path TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS chats(
  id TEXT PRIMARY KEY, item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  agent TEXT NOT NULL, created_at REAL NOT NULL, transcript TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reminders(
  id TEXT PRIMARY KEY, item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  title TEXT NOT NULL, due_date TEXT, note TEXT DEFAULT '', created_at REAL NOT NULL);
"""


@contextmanager
def tx():
    with _lock:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.DATA_DIR / "vault.db")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---- items -----------------------------------------------------------------
def _item(row) -> dict:
    d = dict(row)
    d["data"] = json.loads(d.get("data") or "{}")
    return d


def create_item(kind: str, title: str, parent_id: str | None = None) -> str:
    item_id = new_id()
    with tx() as c:
        c.execute(
            "INSERT INTO items(id,kind,title,parent_id,created_at) VALUES(?,?,?,?,?)",
            (item_id, kind, title[:200], parent_id, time.time()),
        )
    return item_id


def update_item(item_id: str, summary: str | None = None, data: dict | None = None, title: str | None = None) -> None:
    sets, vals = [], []
    if summary is not None:
        sets.append("summary=?")
        vals.append(summary)
    if data is not None:
        sets.append("data=?")
        vals.append(json.dumps(data, ensure_ascii=False))
    if title is not None:
        sets.append("title=?")
        vals.append(title[:200])
    if not sets:
        return
    with tx() as c:
        c.execute(f"UPDATE items SET {', '.join(sets)} WHERE id=?", (*vals, item_id))


def get_item(item_id: str) -> dict | None:
    with tx() as c:
        row = c.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
    return _item(row) if row else None


def list_items() -> list[dict]:
    with tx() as c:
        rows = c.execute("SELECT * FROM items ORDER BY created_at DESC").fetchall()
    return [_item(r) for r in rows]


def delete_item(item_id: str) -> bool:
    """Delete an item, its files on disk, chats and reminders. Reports linked to a deleted lease are unlinked."""
    paths = [Path(f["abs"]) for f in list_files(item_id)]
    with tx() as c:
        exists = c.execute("SELECT 1 FROM items WHERE id=?", (item_id,)).fetchone()
        if not exists:
            return False
        c.execute("UPDATE items SET parent_id=NULL WHERE parent_id=?", (item_id,))
        c.execute("DELETE FROM items WHERE id=?", (item_id,))
    for p in paths:
        p.unlink(missing_ok=True)
    return True


# ---- files -----------------------------------------------------------------
def add_file(item_id: str, role: str, filename: str, mime: str, raw: bytes) -> str:
    file_id = new_id()
    ext = Path(filename).suffix.lower()[:8]
    if not re.fullmatch(r"\.[a-z0-9]+", ext):
        ext = ""
    rel = Path("files") / f"{file_id}{ext}"
    path = config.DATA_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    with tx() as c:
        c.execute(
            "INSERT INTO files(id,item_id,role,filename,mime,path,created_at) VALUES(?,?,?,?,?,?,?)",
            (file_id, item_id, role, filename[:200], mime, str(rel), time.time()),
        )
    return file_id


def _file(row) -> dict | None:
    if not row:
        return None
    d = dict(row)
    root = config.DATA_DIR.resolve()
    absolute = (config.DATA_DIR / d["path"]).resolve()
    if root not in absolute.parents:  # never serve anything outside the data folder
        return None
    d["abs"] = str(absolute)
    return d


def get_file(file_id: str) -> dict | None:
    with tx() as c:
        row = c.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
    return _file(row)


def list_files(item_id: str) -> list[dict]:
    with tx() as c:
        rows = c.execute("SELECT * FROM files WHERE item_id=? ORDER BY created_at", (item_id,)).fetchall()
    return [f for f in (_file(r) for r in rows) if f]


# ---- saved agent chats ---------------------------------------------------------
def add_chat(item_id: str, agent: str, transcript: list[dict]) -> str:
    chat_id = new_id()
    with tx() as c:
        c.execute(
            "INSERT INTO chats(id,item_id,agent,created_at,transcript) VALUES(?,?,?,?,?)",
            (chat_id, item_id, agent, time.time(), json.dumps(transcript, ensure_ascii=False)),
        )
    return chat_id


def list_chats(item_id: str) -> list[dict]:
    with tx() as c:
        rows = c.execute("SELECT * FROM chats WHERE item_id=? ORDER BY created_at", (item_id,)).fetchall()
    return [{**dict(r), "transcript": json.loads(r["transcript"])} for r in rows]


# ---- reminders -------------------------------------------------------------------
def add_reminder(item_id: str, title: str, due_date: str | None, note: str = "") -> str:
    rid = new_id()
    with tx() as c:
        c.execute(
            "INSERT INTO reminders(id,item_id,title,due_date,note,created_at) VALUES(?,?,?,?,?,?)",
            (rid, item_id, title[:200], due_date, note[:500], time.time()),
        )
    return rid


def list_reminders(item_id: str | None = None) -> list[dict]:
    with tx() as c:
        if item_id:
            rows = c.execute("SELECT * FROM reminders WHERE item_id=? ORDER BY due_date IS NULL, due_date", (item_id,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM reminders ORDER BY due_date IS NULL, due_date").fetchall()
    return [dict(r) for r in rows]
