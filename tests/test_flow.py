"""Mock-mode tests. Env vars must be set before the app is imported."""
import io
import json
import os
import tempfile

os.environ["MOCK_MODE"] = "1"
os.environ["DATA_DIR"] = tempfile.mkdtemp()

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.lease_agent import norm, quote_in_text
from app.main import app
from app.meta import parse_sam_output

client = TestClient(app)
LEASE = open(os.path.join(os.path.dirname(__file__), "..", "samples", "sample_lease.txt")).read()


def jpeg(color=(200, 200, 190)):
    b = io.BytesIO()
    Image.new("RGB", (640, 480), color).save(b, "JPEG")
    return b.getvalue()


def test_quote_verification():
    src = norm(LEASE)
    assert quote_in_text("$500 of the security deposit is non-refundable", src)
    assert quote_in_text("Tenant must submit a written list ... within 48 hours of move-in", src)
    assert not quote_in_text("The landlord may keep the entire deposit for any reason at all.", src)
    assert not quote_in_text("", src)


def test_parse_sam_positional_and_empty():
    out = parse_sam_output("<|box;10;20;110;220;640;480|> <|mask;abc|>")
    assert out[0]["box"] == [10, 20, 110, 220] and out[0]["has_mask"] and out[0]["w"] == 640
    assert parse_sam_output("") == []
    assert parse_sam_output("<|box;x1=1;y1=2;x2=3;y2=4;w=9;h=9|>")[0]["box"] == [1, 2, 3, 4]


def post_lease():
    r = client.post("/api/lease", files={"file": ("lease.txt", LEASE.encode(), "text/plain")}, data={"language": "English"})
    assert r.status_code == 200, r.text
    return r.json()


def test_status_and_index():
    assert client.get("/api/status").json()["mock"] is True
    assert client.get("/").status_code == 200


def test_lease_flow_rejects_fabricated_quote():
    item = post_lease()
    flags = item["data"]["flags"]
    assert flags, "expected verified flags"
    assert all(f["verified"] for f in flags)
    assert not any("any reason at all" in f["quote"] for f in flags)
    assert item["data"]["draft"] and item["reminders"] and item["chats"]


def test_condition_flow_with_retry_overlay_and_link():
    lease = post_lease()
    files = [("files", ("a.jpg", jpeg(), "image/jpeg")), ("files", ("b.jpg", jpeg((180, 190, 200)), "image/jpeg"))]
    r = client.post("/api/condition", files=files, data={"rooms": '["Living room","Bedroom"]', "lease_id": lease["id"]})
    assert r.status_code == 200, r.text
    d = r.json()["data"]
    assert len(d["findings"]) == 4
    assert d["outlines_verified"] >= 2
    assert d["lease"]["inspection_deadline"]
    ov = [p["overlay_file_id"] for p in d["photos"] if p["overlay_file_id"]]
    assert ov and client.get("/api/files/" + ov[0]).headers["content-type"] == "image/jpeg"


def test_delete_cascades():
    lease = post_lease()
    r = client.post("/api/condition", files=[("files", ("a.jpg", jpeg(), "image/jpeg"))], data={"rooms": "Kitchen", "lease_id": lease["id"]})
    cid = r.json()["id"]
    assert client.delete("/api/items/" + lease["id"]).status_code == 200
    assert client.get("/api/items/" + lease["id"]).status_code == 404
    assert client.delete("/api/items/" + cid).status_code in (200, 404)


@pytest.mark.parametrize("name,data", [("x.exe", b"abc"), ("empty.txt", b"")])
def test_lease_upload_validation(name, data):
    assert client.post("/api/lease", files={"file": (name, data, "application/octet-stream")}).status_code == 400


def test_photo_validation():
    assert client.post("/api/condition", files=[("files", ("bad.jpg", b"notanimage", "image/jpeg"))]).status_code == 400
    many = [("files", (f"{i}.jpg", jpeg(), "image/jpeg")) for i in range(9)]
    assert client.post("/api/condition", files=many).status_code == 400


def test_split_defect_is_merged_into_one_outline():
    r = client.post("/api/condition", files=[("files", ("a.jpg", jpeg(), "image/jpeg"))], data={"rooms": "Hall"})
    stain = next(f for f in r.json()["data"]["findings"] if f["label"] == "water stain")
    # the mock splits the stain into two SAM instances; both must be inside the saved outline
    first, second = parse_boxes_for("water stain")
    assert stain["outline_verified"]
    assert stain["box"] == [min(first[0], second[0]), min(first[1], second[1]), max(first[2], second[2]), max(first[3], second[3])]


def parse_boxes_for(concept):
    from app.meta import prepare_image
    from app.mock import MockMeta

    raw, _, _ = prepare_image(jpeg())
    return [i["box"] for i in MockMeta().segment(raw, concept)]


def test_outline_check_ignores_extra_spaces_and_unchecked_instances():
    from app.condition_agent import ConditionSession
    from app.meta import prepare_image
    from app.mock import MockMeta

    raw, w, h = prepare_image(jpeg())
    s = ConditionSession(MockMeta(), {"p1": {"room": "Hall", "jpeg": raw, "w": w, "h": h}})
    assert s.call("segment", {"photo_id": "p1", "concept": "water  stain"})["count"] == 2
    assert s.call("check_outline", {"photo_id": "p1", "concept": " water stain", "instance_indices": [0], "expected_description": "stain"})["matching_instances"] == [0]
    out = s.call("record_finding", {"photo_id": "p1", "label": "stain", "description": "d", "instance_indices": [0, 1], "concept": "water stain "})
    assert out["outline_verified"] and "1 unverified" in out["note"]
    assert s.findings[0]["box"] == s.segments[("p1", "water stain")][0]["box"]


def test_reminders_are_deduplicated_and_dates_validated():
    from app.lease_agent import LeaseSession

    s = LeaseSession(LEASE)
    assert s.call("set_reminder", {"title": "Pay rent", "due_date": "2026-11-01", "note": ""})["saved"]
    assert not s.call("set_reminder", {"title": "pay  rent", "due_date": "2026-11-01", "note": ""})["saved"]
    s.call("set_reminder", {"title": "Bad date", "due_date": "2026-13-45", "note": "x"})
    assert s.reminders[-1]["due_date"] is None and "2026-13-45" in s.reminders[-1]["note"]
    s.call("set_reminder", {"title": "No date", "due_date": "null", "note": "within 48 hours"})
    assert s.reminders[-1] == {"title": "No date", "due_date": None, "note": "within 48 hours"}


def test_reminders_carry_their_lease():
    lease = post_lease()
    mine = [r for r in client.get("/api/reminders").json() if r["item_id"] == lease["id"]]
    assert mine and all(r["item_title"] == lease["title"] and r["item_created_at"] == lease["created_at"] for r in mine)


def test_fix_text_repairs_mojibake_only():
    from app.meta import fix_text

    assert fix_text("Documenting Room 1 â€” scanning") == "Documenting Room 1 — scanning"
    assert fix_text("plain text") == "plain text"
    assert fix_text("SÃO PAULO") == "SÃO PAULO"


def test_similar_reminders_are_merged_but_different_ones_kept():
    from app.lease_agent import LeaseSession

    s = LeaseSession(LEASE)
    assert s.call("set_reminder", {"title": "Pay $2,300 security deposit before keys", "due_date": "2026-11-01", "note": ""})["saved"]
    assert not s.call("set_reminder", {"title": "Pay security deposit before key handover", "due_date": "2026-11-01", "note": ""})["saved"]
    assert s.call("set_reminder", {"title": "Monthly rent due on the 1st", "due_date": "2026-11-01", "note": ""})["saved"]
    assert s.call("set_reminder", {"title": "Pay security deposit before key handover", "due_date": "2026-12-01", "note": ""})["saved"]
    assert len(s.reminders) == 3


def test_parse_sam_confidence_in_key_value_form():
    assert parse_sam_output("<|box;x1=1;y1=2;x2=3;y2=4;confidence=0.82|>")[0]["confidence"] == 0.82
    assert parse_sam_output("<|box;X1=1;Y1=2;X2=3;Y2=4;score=0.5|>")[0]["confidence"] == 0.5
    assert parse_sam_output("<|box;10;20;110;220;640;480;0.9|>")[0]["confidence"] == 0.9
    assert parse_sam_output("<|box;x1=1;y1=2;x2=3;y2=4;c=oops|>")[0]["confidence"] is None


def test_condition_report_language():
    r = client.post("/api/condition", files=[("files", ("a.jpg", jpeg(), "image/jpeg"))], data={"rooms": "Hall", "language": "Telugu"})
    item = r.json()
    assert item["data"]["language"] == "Telugu"
    system = item["chats"][0]["transcript"][0]["content"]
    assert "summary` in Telugu" in system and "segment concepts stay in English" in system


def test_non_latin_labels_are_left_off_the_image():
    from app.overlay import _drawable

    assert _drawable("scratch marks") and _drawable("mancha de humedad")
    assert not _drawable("గీతల గుర్తులు") and not _drawable("दाग")
    r = client.post("/api/condition", files=[("files", ("a.jpg", jpeg(), "image/jpeg"))], data={"rooms": "Hall"})
    markers = [f["marker"] for f in r.json()["data"]["findings"] if f["box"]]
    assert sorted(markers) == list(range(1, len(markers) + 1))


def test_norm_ignores_invisible_characters_and_unicode_forms():
    import unicodedata

    with_zwnj = "ఫ్లాట్‌లోకి ప్రవేశించవచ్చు"
    assert norm(with_zwnj) == norm("ఫ్లాట్లోకి ప్రవేశించవచ్చు")
    assert norm(unicodedata.normalize("NFD", "café fee")) == norm("café fee")


def test_fragment_quotes_are_rejected():
    from app.lease_agent import LeaseSession

    s = LeaseSession(LEASE)
    out = s.call("flag_clause", {"quote": "$1,150", "category": "fee", "severity": "high", "explanation": "", "question_to_ask": ""})
    assert not out["accepted"] and "too short" in out["reason"]
    out = s.call("flag_clause", {"quote": "Monthly rent is $1,150, due on the 1st of each month.", "category": "fee", "severity": "low", "explanation": "", "question_to_ask": ""})
    assert out["accepted"]


def test_garbled_pdf_text_goes_to_the_model():
    from app.lease_agent import _garbled

    assert _garbled("కలి\x00త నమూనా లీజు ఒప\x00ందం ఆ\x00\x00" + "x" * 300)
    assert not _garbled(LEASE)


def test_out_of_turns_still_drafts_message_and_summary(monkeypatch):
    from app import config, lease_agent

    class Stalling:
        """Flags the same clause forever, until it is told it is out of turns."""

        def chat(self, messages, tools):
            names = [t["function"]["name"] for t in tools]
            if "flag_clause" in names:
                args = {"title": "Pay rent", "due_date": None, "note": ""}
                return {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{len(messages)}", "type": "function", "function": {"name": "set_reminder", "arguments": json.dumps(args)}}]}
            calls = [{"id": f"w{i}", "type": "function", "function": {"name": n, "arguments": json.dumps({"subject": "Q", "body": "B"} if n == "draft_landlord_message" else {"summary": "done"})}} for i, n in enumerate(names)]
            return {"role": "assistant", "content": "", "tool_calls": calls}

    monkeypatch.setattr(config, "LEASE_MAX_TURNS", 3)
    session = lease_agent.LeaseSession(LEASE)
    turns, transcript = lease_agent._agent_loop(Stalling(), session, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert turns == 4 and session.draft == {"subject": "Q", "body": "B"} and session.finished["summary"] == "done"


def test_extraction_values_follow_the_language():
    from app.lease_agent import EXTRACT_PROMPT

    assert "written in Telugu" in EXTRACT_PROMPT.format(transcribe="", language="Telugu")


def test_move_out_check_marks_damage_already_there_at_move_in():
    files = [("files", ("a.jpg", jpeg(), "image/jpeg")), ("files", ("b.jpg", jpeg((180, 190, 200)), "image/jpeg"))]
    move_in = client.post("/api/condition", files=files, data={"rooms": '["Living room","Bedroom"]'}).json()
    out_files = [("files", ("out-a.jpg", jpeg((190, 190, 180)), "image/jpeg")), ("files", ("out-b.jpg", jpeg((170, 180, 190)), "image/jpeg"))]
    r = client.post("/api/condition", files=out_files, data={"move_in_id": move_in["id"], "matches": '["p1",""]'})
    assert r.status_code == 200, r.text
    item = r.json()
    d = item["data"]
    assert d["mode"] == "move_out" and d["move_in"]["id"] == move_in["id"] and item["parent_id"] == move_in["id"]
    room1 = [f for f in d["findings"] if f["photo_id"] == "p1"]
    assert {f["label"]: f["status"] for f in room1} == {"water stain": "pre_existing", "hairline crack": "new"}
    assert all(f["status"] == "unclear" for f in d["findings"] if f["photo_id"] == "p2")  # no move-in photo matched
    assert d["photos"][0]["room"] == "Living room" and d["photos"][0]["before_file_id"] and not d["photos"][1]["before_file_id"]
    assert d["status_counts"]["pre_existing"] == 1 and d["draft"]["body"]
    assert any(i["id"] == item["id"] and i["mode"] == "move_out" for i in client.get("/api/items").json())
    # the check keeps its own copy of the move-in photo, so it survives deleting the move-in report
    assert client.delete("/api/items/" + move_in["id"]).status_code == 200
    assert client.get("/api/files/" + d["photos"][0]["before_file_id"]).status_code == 200


def test_move_out_status_comes_from_the_comparison_not_the_agent():
    from app.condition_agent import ConditionSession
    from app.meta import prepare_image
    from app.mock import MockMeta

    raw, w, h = prepare_image(jpeg())
    s = ConditionSession(MockMeta(), {"p1": {"room": "Hall", "jpeg": raw, "w": w, "h": h}}, move_in={"id": "x", "title": "t", "generated_at": "2026-11-01"})
    out = s.call("record_finding", {"photo_id": "p1", "label": "stain", "description": "a stain", "severity": "low", "status": "pre_existing"})
    assert "error" in out  # the agent cannot set the status itself
    out = s.call("record_finding", {"photo_id": "p1", "label": "stain", "description": "a stain", "severity": "low"})
    assert out["status"] == "unclear"  # never compared, so never "already there"
    assert s.call("compare_with_move_in", {"photo_id": "p1", "description": "a stain"})["status"] == "unclear"  # no move-in photo


def test_move_out_rejects_a_bad_move_in_report():
    lease = post_lease()
    r = client.post("/api/condition", files=[("files", ("a.jpg", jpeg(), "image/jpeg"))], data={"move_in_id": lease["id"]})
    assert r.status_code == 400


def test_move_out_verdict_is_never_borrowed_from_another_defect():
    from app.condition_agent import ConditionSession
    from app.meta import prepare_image
    from app.mock import MockMeta

    raw, w, h = prepare_image(jpeg())
    photos = {"p1": {"room": "Hall", "jpeg": raw, "w": w, "h": h, "before": {"jpeg": raw, "findings": []}},
              "p2": {"room": "Bed", "jpeg": raw, "w": w, "h": h, "before": {"jpeg": raw, "findings": []}}}
    s = ConditionSession(MockMeta(), photos, move_in={"id": "x", "title": "t", "generated_at": "2026-11-01"})
    cid = s.call("compare_with_move_in", {"photo_id": "p1", "description": "Brown water stain"})["comparison_id"]
    crack = {"label": "crack", "description": "Thin crack", "severity": "low"}
    assert s.call("record_finding", {"photo_id": "p1", **crack})["status"] == "unclear"  # not compared
    assert s.call("record_finding", {"photo_id": "p2", **crack, "comparison_id": cid})["status"] == "unclear"  # other photo
    stain = {"photo_id": "p1", "label": "stain", "description": "Brown water stain", "severity": "low", "comparison_id": cid}
    assert s.call("record_finding", stain)["status"] == "pre_existing"
    assert s.call("record_finding", {**stain, "label": "stain 2"})["status"] == "unclear"  # each comparison is used once


@pytest.mark.parametrize("raw", [
    "MODEL_API_KEY=abc123\n".encode("utf-16"),  # Windows PowerShell 5: echo ... > .env
    b"MODEL_API_KEY=abc123 \r\n",  # Windows cmd: echo keeps the space before >
    b"MODEL_API_KEY=abc123\n",  # macOS / Linux
    "﻿MODEL_API_KEY=abc123\n".encode("utf-8"),  # UTF-8 with BOM
])
def test_env_file_is_read_however_the_terminal_wrote_it(tmp_path, raw):
    import io as _io

    from dotenv import dotenv_values

    from app.config import _read_env_file

    (tmp_path / ".env").write_bytes(raw)
    assert dotenv_values(stream=_io.StringIO(_read_env_file(tmp_path / ".env")))["MODEL_API_KEY"] == "abc123"


def test_readme_walkthrough_same_photos_are_already_there():
    """README step 4: the move-out check with the same sample photos marks everything as already there (mock mode)."""
    sample = lambda n: open(os.path.join(os.path.dirname(__file__), "..", "samples", n), "rb").read()
    photos = lambda: [("files", ("room_living.jpg", sample("room_living.jpg"), "image/jpeg")), ("files", ("room_bedroom.jpg", sample("room_bedroom.jpg"), "image/jpeg"))]
    move_in = client.post("/api/condition", files=photos()).json()
    out = client.post("/api/condition", files=photos(), data={"move_in_id": move_in["id"], "matches": '["p1","p2"]'}).json()
    assert out["data"]["findings"] and all(f["status"] == "pre_existing" for f in out["data"]["findings"])
