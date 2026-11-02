"""Mock-mode tests. Env vars must be set before the app is imported."""
import io
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
